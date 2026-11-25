# SPDX-License-Identifier: MIT
"""Unified recipe entrypoint. All device operations are explicit and run-scoped."""

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from recipe_common import (
    BOOK,
    IMAGE,
    PYTHON,
    ROOT,
    config,
    data_dir,
    launch,
    lock,
    plan,
    read_json,
    write_json,
)


def execute(command, **kwargs):
    print("+ " + shlex.join(map(str, command)), flush=True)
    try:
        return subprocess.run(list(map(str, command)), check=True, **kwargs)
    except subprocess.CalledProcessError as error:
        for text in (error.stdout, error.stderr):
            if text:
                print(text[-12000:], file=sys.stderr)
        raise


def remote(node, command, **kwargs):
    if node["ssh"] == "local":
        return execute(command, **kwargs)
    return execute(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=20",
            "--",
            node["ssh"],
            shlex.join(map(str, command)),
        ],
        **kwargs,
    )


def node_action(action, payload):
    result = remote(
        payload["node"],
        ["python3", "-", action, json.dumps(payload)],
        input=(ROOT / "scripts/node.py").read_text(),
        text=True,
        capture_output=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.startswith("RLK_RECIPE_RECEIPT=")]
    if len(lines) != 1:
        raise RuntimeError(
            f"missing unique {action} receipt: {result.stdout[-2000:]} {result.stderr[-1000:]}"
        )
    return json.loads(lines[0].split("=", 1)[1])


def build(c):
    pins = lock()
    if subprocess.check_output(["git", "-C", ROOT, "status", "--porcelain"], text=True).strip():
        raise ValueError("build requires a clean, committed recipe checkout; keep local config in site.json")
    recipe_commit = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    for name, expected in pins["patches"].items():
        if hashlib.sha256((BOOK / "patches" / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"patch checksum differs: {name}")
    runtime = Path(c["work_dir"]) / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    tag = IMAGE + "-" + c.get("gpu_arch", "gfx942") + "-" + recipe_commit[:12]
    args = dict(
        BASE_IMAGE=pins["base_image"]["reference"],
        RLK_SHA=pins["kernel_base"],
        VERL_SHA=pins["verl"],
        RLK_TREE=pins["kernel_tree"],
        VERL_TREE=pins["verl_tree"],
        GPU_ARCH=c.get("gpu_arch", "gfx942"),
        RECIPE_COMMIT=recipe_commit,
    )
    command = ["docker", "build", "--pull", "--file", BOOK / "Dockerfile", "--tag", tag]
    for name, value in args.items():
        command += ["--build-arg", f"{name}={value}"]
    command.append(ROOT)
    execute(command)
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", "--format={{.Id}}", tag], text=True
    ).strip()
    archive = runtime / f"{image_id[7:19]}.tar"
    execute(["docker", "save", "--output", archive, image_id])
    # The archive provides identical bytes to every host; no unpublished tag assumed.
    record = dict(
        image=tag,
        image_id=image_id,
        archive=str(archive),
        sources=pins,
        gpu_arch=c.get("gpu_arch", "gfx942"),
        recipe_commit=subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip(),
        recipe_dirty=bool(
            subprocess.check_output(["git", "-C", ROOT, "status", "--porcelain"], text=True).strip()
        ),
    )
    write_json(runtime / "runtime.lock.json", record)
    return record


def runtime(c):
    record = read_json(Path(c["work_dir"]) / "runtime/runtime.lock.json")
    if record["sources"] != lock():
        raise ValueError("source lock changed; rebuild before running")
    if record["gpu_arch"] != c.get("gpu_arch", "gfx942"):
        raise ValueError("GPU architecture changed; rebuild the image")
    head = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    if (
        record["recipe_commit"] != head
        or subprocess.check_output(["git", "-C", ROOT, "status", "--porcelain"], text=True).strip()
    ):
        raise ValueError("recipe source changed; use the recorded checkout or rebuild")
    return record


def prepare(c, download):
    record = runtime(c)
    root = Path(c["work_dir"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / "prepare-config.json"
    write_json(path, c)
    command = ["docker", "run", "--rm", "--network=bridge", "--entrypoint", PYTHON]
    # Preparation is CPU-only. Existing external assets may be mounted read-only.
    command += ["--mount", f"type=bind,source={root},target={root}"]
    for key in ("model_dir", "baseline_data_dir"):
        p = Path(c[key])
        if not p.is_relative_to(root):
            if not p.is_dir():
                raise ValueError(f"external {key} must already exist: {p}")
            command += ["--mount", f"type=bind,source={p},target={p},readonly"]
    command += [record["image_id"], "/opt/recipes/scripts/prepare_runtime.py", "--config", path]
    if download:
        command.append("--download")
    execute(command)


def operate(c, output, verify_only=False):
    output = Path(output).resolve()
    if not output.is_relative_to(Path(c["work_dir"])):
        raise ValueError("output must be under work_dir, shared across all participating nodes")
    if verify_only:
        record = read_json(output / "runtime.lock.json")
        command = [
            "docker",
            "run",
            "--rm",
            "--network=none",
            "--entrypoint",
            PYTHON,
            "--mount",
            f"type=bind,source={c['work_dir']},target={c['work_dir']}",
            record["image_id"],
            "/opt/recipes/scripts/runtime.py",
            "verify",
            "--config",
            output / "site.json",
            "--output",
            output,
        ]
        execute(command)
        return
    record = runtime(c)
    if not (data_dir(c) / "manifest.json").exists():
        raise ValueError("prepare this dataset before running")
    # Long run requires BOTH correctness and same-load math smoke on this topology/image.
    if c["profile"] != "smoke":
        gate = c.get("gates", {}).get("correctness_run")
        require_gate(c, record, gate, "smoke")
        if c["profile"] == "math-200":
            require_gate(c, record, c.get("gates", {}).get("math_smoke_run"), "math-smoke")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "site.json", c)
    write_json(output / "runtime.lock.json", record)
    write_json(output / "plan.json", plan(c))
    write_json(output / "input-manifest.json", read_json(data_dir(c) / "manifest.json"))
    nonce, run_id = uuid4().hex, uuid4().hex[:12]
    (output / "storage-challenge").write_text(nonce)
    payloads = [
        dict(
            config=c,
            node=n,
            output=str(output),
            nonce=nonce,
            run_id=run_id,
            image=record["image_id"],
            image_id=record["image_id"],
            container=f"rlk-recipe-{run_id}-{n['name']}",
        )
        for n in c["nodes"]
    ]
    started, failure, probes = [], None, []
    try:
        for payload in payloads:
            remote(payload["node"], ["docker", "load", "--input", record["archive"]])
            check = node_action("check", payload)
            write_json(output / f"preflight-{payload['node']['name']}.json", check)
        for n in c["nodes"]:
            if read_json(output / f"storage-{n['name']}.json")["nonce"] != nonce:
                raise RuntimeError("shared storage peer marker differs")
        for payload in payloads:
            # Track before start: a later probe failure still cleans up this run.
            started.append(payload)
            write_json(output / f"container-{payload['node']['name']}.json", node_action("start", payload))
            probe = remote(
                payload["node"],
                [
                    "docker",
                    "exec",
                    payload["container"],
                    PYTHON,
                    "/opt/recipes/scripts/runtime.py",
                    "probe",
                    "--config",
                    output / "site.json",
                    "--output",
                    output,
                ],
                text=True,
                capture_output=True,
            )
            (output / f"probe-{payload['node']['name']}.log").write_text(probe.stdout + probe.stderr)
            markers = [line for line in probe.stdout.splitlines() if line.startswith("RUNTIME_PROBE=")]
            if len(markers) != 1:
                raise RuntimeError("missing runtime GPU/source probe receipt")
            probes.append(json.loads(markers[0].split("=", 1)[1]))
        if len({json.dumps(p["identity"]["dependencies"], sort_keys=True) for p in probes}) != 1:
            raise RuntimeError("runtime dependency versions differ across nodes")
        if any(p["inputs"] != read_json(output / "input-manifest.json") for p in probes):
            raise RuntimeError("prepared input manifest differs across nodes")
        if len({d["arch"] for p in probes for d in p["devices"]}) != 1:
            raise RuntimeError("mixed GPU architectures are outside this recipe")
        for payload in payloads:
            write_json(output / f"ray-{payload['node']['name']}.json", node_action("ray", payload))
        head = payloads[0]
        with (output / "entrypoint.log").open("w") as log:
            remote(
                head["node"],
                [
                    "docker",
                    "exec",
                    head["container"],
                    PYTHON,
                    "/opt/recipes/scripts/runtime.py",
                    "run",
                    "--config",
                    output / "site.json",
                    "--output",
                    output,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
    except BaseException as error:
        failure = repr(error)
        raise
    finally:
        cleanup = []
        for payload in reversed(started):
            try:
                cleanup.append(node_action("stop", payload))
            except Exception as error:
                cleanup.append(dict(error=repr(error), container=payload["container"]))
        write_json(output / "orchestration.json", dict(failure=failure, cleanup=cleanup))
        from scripts.report import report

        report(output)


def require_gate(c, record, path, profile):
    if not path:
        raise ValueError(
            f"set gates.{('correctness_run' if profile == 'smoke' else 'math_smoke_run')} to a verified run directory"
        )
    old = read_json(Path(path) / "site.json")
    result = read_json(Path(path) / "verification.json")
    previous = read_json(Path(path) / "runtime.lock.json")
    if (
        result.get("status") != "PASS"
        or old["profile"] != profile
        or previous["image_id"] != record["image_id"]
    ):
        raise ValueError("gate must be an independently verified matching-image run")
    for key in ("topology", "nodes", "model_dir", "network", "gpu_arch", "numa_interleave"):
        if old.get(key) != c.get(key):
            raise ValueError(f"gate differs in {key}")
    if profile == "math-smoke" and old["dataset"] != c["dataset"]:
        raise ValueError("math smoke must use the same dataset and load")
    if profile == "math-smoke" and read_json(Path(path) / "input-manifest.json") != read_json(
        data_dir(c) / "manifest.json"
    ):
        raise ValueError("prepared math inputs changed after the smoke gate")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "build", "prepare", "run", "verify", "report"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    c = config(args.config)
    if args.action == "plan":
        command, env = launch(
            c,
            args.output or str(Path(c["work_dir"]) / "runs/NEW-RUN"),
            ["<audit A>", "<audit B>", "<audit C>"],
        )
        print(json.dumps(dict(plan=plan(c), command_template=command, environment=env), indent=2))
    elif args.action == "build":
        build(c)
    elif args.action == "prepare":
        prepare(c, args.download)
    elif args.action in ("run", "verify"):
        if not args.output:
            parser.error("--output is required")
        operate(c, args.output, verify_only=args.action == "verify")
    else:
        from scripts.report import report

        if not args.output:
            parser.error("--output is required")
        report(Path(args.output))


if __name__ == "__main__":
    main()
