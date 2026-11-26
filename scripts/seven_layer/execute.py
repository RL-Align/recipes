# SPDX-License-Identifier: MIT
"""Resource-gated revised N1-N5 fixed scoring and three-update phase execution.

N5 (EP4, experimental) is opt-in, runs after N1-N4 and never blocks them. Every actual phase
runs committed image code; one phase exits before another uses physical GPUs.
"""

import argparse
import contextlib
import datetime
import fcntl
import json
import os
import socket
import subprocess
import time
from pathlib import Path
from uuid import uuid4

from matrix import profiles
from matrix_report import summarize
from resource_policy import ResourceConflict, evaluate
from watch import snapshot

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "scripts"))
from node import allocation  # noqa: E402


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(value, indent=2) + "\n")
    pending.replace(path)


def resource_policy(root):
    path = root / "resource-policy-lock.json"
    return json.loads(path.read_text()) if path.exists() else None


def observe(root, gpus, *, own_container=None):
    errors = []
    for attempt in range(5):
        try:
            sample = snapshot(gpus)
            receipt = evaluate(sample, resource_policy(root), own_container=own_container)
            if errors:
                sample["bounded_observation_retries"] = errors
                write(
                    root / ("observation-retry-" + uuid4().hex[:8] + ".json"),
                    dict(errors=errors, final_sample=sample),
                )
            return sample, receipt
        except ResourceConflict:
            raise
        except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
            errors.append(f"{type(error).__name__}: {error}")
            if attempt < 4:
                time.sleep(0.5)
    write(root / ("observation-failure-" + uuid4().hex[:8] + ".json"), dict(errors=errors))
    raise RuntimeError("resource observation did not stabilize after five bounded attempts: " + errors[-1])


UNSTABLE_OBSERVATION_LIMIT_SECONDS = 120


def instability_exceeded(first_unstable, now):
    return now - first_unstable > UNSTABLE_OBSERVATION_LIMIT_SECONDS


def stop_owned_container(cid, owner, timeout=15):
    """Stop only a live container with our exact label; --rm may have won the race."""
    inspected = subprocess.run(
        ["docker", "inspect", '--format={{index .Config.Labels "rl-align.seven-layer-N"}}', cid],
        text=True,
        capture_output=True,
    )
    if inspected.returncode:
        if "no such object" in inspected.stderr.lower():
            return
        raise RuntimeError("cannot verify own container identity: " + inspected.stderr)
    if inspected.stdout.strip() != owner:
        raise RuntimeError("refusing to stop a container whose ownership label differs")
    stopped = subprocess.run(
        ["docker", "stop", "--timeout", str(timeout), cid], text=True, capture_output=True
    )
    if stopped.returncode and "no such container" not in stopped.stderr.lower():
        raise RuntimeError("own container stop failed: " + stopped.stderr)


def wait_idle(root, gpus):
    while True:
        try:
            sample, receipt = observe(root, gpus)
            with (root / "gpu-samples.jsonl").open("a") as f:
                f.write(json.dumps(sample) + "\n")
            write(root / "latest-preflight.json", receipt)
            return
        except (RuntimeError, subprocess.CalledProcessError) as e:
            write(
                root / "resource-wait.json",
                dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(), reason=str(e)),
            )
            time.sleep(30)


def container_base(work, run, active, image, out, *, gpu=True):
    ray_logs = out / "ray-runtime"
    ray_logs.mkdir(parents=True, exist_ok=True)
    command = [
        "docker",
        "create",
        "--rm",
        "--init",
        "--label",
        "rl-align.seven-layer-N=" + run.name,
        "--network=host",
        "--ipc=private",
        "--shm-size=16g",
        "--mount",
        f"type=bind,source={work},target=/experiment",
        "--mount",
        f"type=bind,source={work}/models,target=/experiment/models,readonly",
        "--mount",
        f"type=bind,source={work}/data,target=/experiment/data,readonly",
        "--mount",
        f"type=bind,source={out},target=/outputs",
        "--mount",
        f"type=bind,source={ray_logs},target=/tmp/rv4",
        "-e",
        "OMP_NUM_THREADS=4",
        "-e",
        "HF_HOME=/experiment/cache-N/huggingface",
        "-e",
        "TRITON_CACHE_DIR=/experiment/cache-N/triton",
        "-e",
        "TORCH_EXTENSIONS_DIR=/experiment/cache-N/torch_extensions",
    ]
    trace_pattern = os.environ.get("RLK_TRACE_AUDIT_PATTERN")
    if trace_pattern:
        command += ["-e", "RLK_TRACE_AUDIT_PATTERN=" + trace_pattern]
    # Diagnostic-only rollout row dump (off unless a diagnosis driver sets both).
    for name in ("RLK_ROUTED_ROWS_DUMP_DIR", "RLK_ROUTED_ROWS_DUMP_M"):
        if os.environ.get(name):
            command += ["-e", name + "=" + os.environ[name]]
    if gpu:
        command += [
            "--device=/dev/kfd",
            "--mount",
            "type=bind,source=/sys,target=/sys,readonly",
            "-e",
            "LD_PRELOAD=/opt/dsv4/lib/rlk-isolated-kfd.so",
            "-e",
            "RLK_ALLOWED_KFD_GPU_IDS=" + ",".join(str(r["kfd_gpu_id"]) for r in active),
            "-e",
            "HIP_VISIBLE_DEVICES=" + ",".join(map(str, range(len(active)))),
            "-e",
            "CUDA_VISIBLE_DEVICES=" + ",".join(map(str, range(len(active)))),
            "-e",
            "VLLM_ENABLE_V1_MULTIPROCESSING=1",
            "-e",
            "VLLM_ROCM_USE_AITER=1",
        ]
        for row in active:
            command += ["--device=" + row["render_node"]]
        for gid in sorted({os.stat("/dev/kfd").st_gid, *[os.stat(r["render_node"]).st_gid for r in active]}):
            command += ["--group-add", str(gid)]
    else:
        command += ["-e", "HIP_VISIBLE_DEVICES=", "-e", "CUDA_VISIBLE_DEVICES="]
    return command + ["--entrypoint", "/opt/dsv4-venv/bin/python", image]


def mapped(work, path):
    return "/experiment/" + str(Path(path).resolve().relative_to(work))


def actor_nproc(profile):
    return profile["actor"]["gpus"]


def compare_actor_scores(first, second):
    """Raw-byte comparison of two actor-score dirs' sample-*.f32; topology metadata is not compared."""
    a = {p.name: p for p in Path(first).glob("sample-*.f32")}
    b = {p.name: p for p in Path(second).glob("sample-*.f32")}
    common = sorted(set(a) & set(b))
    differing = [n for n in common if a[n].read_bytes() != b[n].read_bytes()]
    missing = sorted(set(a) ^ set(b))
    return dict(
        expected=len(set(a) | set(b)),
        compared=len(common),
        mismatch=len(differing) + len(missing),
        differing=differing,
        missing=missing,
    )


# Per-dataset phase wall-clock limit; DAPO-Math-17k phases legitimately run longer than copy-task ones.
DEFAULT_PHASE_TIMEOUT_SECONDS = 7200
PHASE_TIMEOUT_SECONDS = {"copy": 7200, "dapo": 8 * 3600, None: DEFAULT_PHASE_TIMEOUT_SECONDS}


def launch(
    work,
    run,
    rows,
    image,
    profile,
    action,
    out,
    *,
    version=0,
    policy=None,
    batch=None,
    resume=None,
    checkpoint=None,
    timeout_seconds=DEFAULT_PHASE_TIMEOUT_SECONDS,
):
    out.mkdir(parents=True, exist_ok=False)
    role = "actor" if action.startswith("actor") else "rollout"
    active = rows[: profile[role]["gpus"]]
    gpus = [r["physical_gpu"] for r in active]
    if len(active) != profile[role]["gpus"]:
        raise ValueError("selected physical GPU pool cannot satisfy unchanged target topology")
    phase_state = dict(
        group=profile["id"],
        action=action,
        version=version,
        image_id=image,
        physical_gpus=gpus,
        output=str(out),
        status="WAITING_FOR_SELECTED_DEVICES",
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    write(run / "current-phase.json", phase_state)
    retries = 0
    while True:
        port = free_loopback_port() if role == "actor" else None
        record = _launch_once(
            work, run, rows, image, profile, action, out, role, active, gpus, phase_state,
            port, retries, version, policy, batch, resume, checkpoint, timeout_seconds,
        )  # fmt: skip
        code = record["returncode"]
        if not code:
            return record
        log_text = (out / "driver.log").read_text(errors="replace")
        if role == "actor" and is_rendezvous_port_race(log_text) and retries < RDZV_MAX_RETRIES - 1:
            # Port was taken between probe and container start; keep evidence, retry on a new port.
            retries += 1
            kept = out.with_name(out.name + f".rdzv-attempt-{retries}")
            out.rename(kept)
            write(
                kept / "rendezvous-retry.json",
                dict(
                    time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    port=port,
                    returncode=code,
                    reason="EADDRINUSE at c10d rendezvous",
                ),
            )
            out.mkdir(parents=True, exist_ok=False)
            continue
        if any(
            token in log_text
            for token in ["DistBackendError", "ncclSystemError", "ncclUnhandled", "NCCL error", "RCCL error"]
        ):
            raise ResourceConflict(
                "distributed communication error; stop this experiment and retain evidence"
            )
        raise RuntimeError(f"{profile['id']} {action} exited {code}; see {out}/driver.log")


def free_loopback_port():
    with socket.socket() as rendezvous:
        rendezvous.bind(("127.0.0.1", 0))
        return rendezvous.getsockname()[1]


RDZV_MAX_RETRIES = 3


def is_rendezvous_port_race(log_text):
    # c10d TCPStore bind failure: port stolen by another host-network process.
    return "EADDRINUSE" in log_text and ("DistNetworkError" in log_text or "failed to listen" in log_text)


def _launch_once(
    work, run, rows, image, profile, action, out, role, active, gpus, phase_state,
    port, retries, version, policy, batch, resume, checkpoint, timeout_seconds,
):  # fmt: skip
    wait_idle(run, gpus)
    command = container_base(work, run, active, image, out)
    if role == "actor":
        # Hostname "10" is interpreted as numeric IPv4 0.0.0.10 by c10d's
        # dynamic bootstrap store. An explicit single-node loopback rendezvous
        # avoids advertising that hostname; no framework source patch is needed.
        command += [
            "-m",
            "torch.distributed.run",
            "--rdzv_backend=static",
            "--nnodes=1",
            "--node_rank=0",
            "--master_addr=127.0.0.1",
            "--master_port=" + str(port),
            "--nproc_per_node=" + str(actor_nproc(profile)),
        ]
    command += [
        "-m",
        "verl.experimental.rl_kernel.phase_runtime",
        action,
        "--profile",
        mapped(work, work / f"profiles/{profile['id']}.json"),
        "--model",
        "/experiment/models/DeepSeek-V4-Flash-7L",
        "--suite",
        "/experiment/evidence/fixed-input-suite.json",
        "--snapshot-sha256",
        json.loads((work / "models/DeepSeek-V4-Flash-7L/checkpoint.lock.json").read_text())[
            "safetensors_sha256"
        ],
        "--version",
        str(version),
        "--output",
        "/outputs",
        "--provenance",
        mapped(work, run / "provenance.json"),
    ]
    for flag, value in [
        ("--policy", policy),
        ("--batch", batch),
        ("--resume", resume),
        ("--next-checkpoint", checkpoint),
    ]:
        if value is not None:
            command += [flag, mapped(work, value)]
    write(out / "command.json", command)
    cid = subprocess.check_output(command, text=True).strip()
    write(out / "container.json", dict(container_id=cid, action=action, version=version, profile=profile))
    # Creating a container is not a reservation; guard again immediately at actual start.
    wait_idle(run, gpus)
    phase_state.update(
        status="RUNNING", container_id=cid, timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    write(run / "current-phase.json", phase_state)
    began = time.monotonic()
    next_sample = 0.0
    first_unstable = None
    with (out / "driver.log").open("w") as log:
        proc = subprocess.Popen(["docker", "start", "-a", cid], stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            time.sleep(1)
            elapsed = time.monotonic() - began
            if elapsed > timeout_seconds:
                stop_owned_container(cid, run.name, timeout=30)
                raise TimeoutError(
                    f"own phase exceeded {timeout_seconds}s timeout; retained logs and checkpoints"
                )
            if elapsed >= next_sample:
                next_sample = elapsed + 5
                try:
                    sample, _ = observe(run, gpus, own_container=cid)
                    first_unstable = None
                    with (run / "gpu-samples.jsonl").open("a") as f:
                        f.write(json.dumps(sample) + "\n")
                except Exception as error:
                    now = time.monotonic()
                    unstable = isinstance(
                        error, (RuntimeError, OSError, subprocess.CalledProcessError)
                    ) and not isinstance(error, ResourceConflict)
                    if unstable:
                        if first_unstable is None:
                            first_unstable = now
                        stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
                        with (out / "observation-instability.jsonl").open("a") as f:
                            f.write(json.dumps(dict(time=stamp, elapsed=elapsed, error=str(error))) + "\n")
                        if not instability_exceeded(first_unstable, now):
                            continue
                        error = RuntimeError(
                            f"observation unstable continuously for {now - first_unstable:.0f}s "
                            f"(limit {UNSTABLE_OBSERVATION_LIMIT_SECONDS}s): {error}"
                        )
                    write(out / "external-conflict.json", dict(error=str(error), container_id=cid))
                    stop_owned_container(cid, run.name)
                    proc.wait()
                    if unstable:
                        raise error
                    raise
        code = proc.returncode
    record = dict(
        action=action,
        version=version,
        container_id=cid,
        returncode=code,
        seconds=time.monotonic() - began,
        active_gpus=len(active),
        selected_physical_gpus=gpus,
        selected_pool_gpus=len(rows),
        machine_gpus=8,
        status="EXECUTED" if code == 0 else "FAIL",
        rendezvous_retries=retries,
        timeout_seconds=timeout_seconds,
    )
    write(out / "execution.json", record)
    write(run / "current-phase.json", dict(phase_state, **record))
    return record


def verify(work, run, image, profile, actor, rollout, policy, out, *, suite=None):
    command = container_base(work, run, [], image, out, gpu=False)
    command += [
        "-m",
        "verl.experimental.rl_kernel.phase_acceptance",
        "--profile",
        mapped(work, work / f"profiles/{profile['id']}.json"),
        "--allocation",
        mapped(work, run / "allocation.json"),
        "--actor",
        mapped(work, actor),
        "--rollout",
        mapped(work, rollout),
        "--policy",
        mapped(work, policy),
        "--model-lock",
        "/experiment/models/DeepSeek-V4-Flash-7L/checkpoint.lock.json",
        "--suite",
        mapped(work, suite or work / "evidence/fixed-input-suite.json"),
        "--snapshot-sha256",
        json.loads((work / "models/DeepSeek-V4-Flash-7L/checkpoint.lock.json").read_text())[
            "safetensors_sha256"
        ],
        "--output",
        "/outputs/verification.json",
    ]
    cid = subprocess.check_output(command, text=True).strip()
    with (out / "verifier.log").open("w") as log:
        result = subprocess.run(["docker", "start", "-a", cid], stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError("independent verifier failed: " + str(out / "verification.json"))
    return json.loads((out / "verification.json").read_text())


def sha256_file(path):
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def completed_steps(group):
    """Receipts of the leading run of fully complete, verified steps (1..3) of a prior trajectory.

    Complete = steps.json entry PASS + checkpoint-v{k}/manifest.json + update-{k}/policy-v{k}.pt.
    The first incomplete step ends the prefix; later steps are never counted.
    """
    group = Path(group)
    path = group / "steps.json"
    steps = json.loads(path.read_text()) if path.is_file() else []
    done = []
    for k in (1, 2, 3):
        entry = next((s for s in steps if s.get("update") == k), None)
        if not (
            entry
            and entry.get("post_update_fresh_and_fixed_verification") == "PASS"
            and (group / f"checkpoint-v{k}" / "manifest.json").is_file()
            and (group / f"update-{k}" / f"policy-v{k}.pt").is_file()
        ):
            break
        done.append(entry)
    return done


def step_dirs(k):
    """Directories the final verifier (and aggregator) read for completed step k."""
    return [
        f"v{k - 1}-generate",
        f"update-{k}",
        f"checkpoint-v{k}",
        f"v{k}-post-generate",
        f"v{k}-actor-audit",
        f"v{k}-fixed-actor",
        f"v{k}-fixed-rollout",
    ]


def link_prior_steps(work, group, prior_group, count):
    """Relative symlinks (resolvable inside the /experiment mount) to a prior run's step dirs; no copies."""
    work_real = Path(os.path.realpath(work))
    group_real = os.path.realpath(group)
    links = []
    for k in range(1, count + 1):
        for name in step_dirs(k):
            src = Path(prior_group) / name
            if not src.is_dir():
                raise RuntimeError(f"resume source step directory missing: {src}")
            real = Path(os.path.realpath(src))
            real.relative_to(work_real)  # must live under the mounted work dir
            (group / name).symlink_to(os.path.relpath(real, group_real))
            links.append(dict(name=name, target=str(real)))
    return links


def resume_plan(prior_group):
    """(count, policy, checkpoint, receipts) to continue a prior dapo trajectory; count 0 = from scratch."""
    prior_group = Path(prior_group)
    done = completed_steps(prior_group) if prior_group.is_dir() else []
    count = len(done)
    if not count:
        return 0, None, None, []
    real = Path(os.path.realpath(prior_group))
    receipts = []
    for r in done:
        receipts.append(dict(r, resumed_from=dict(group=str(real), upstream=r.get("resumed_from"))))
    policy = Path(os.path.realpath(real / f"update-{count}" / f"policy-v{count}.pt"))
    checkpoint = Path(os.path.realpath(real / f"checkpoint-v{count}"))
    return count, policy, checkpoint, receipts


def training(
    work, run, rows, image, profile, initial_policy, dataset,
    *, start_version=0, initial_checkpoint=None, prior_receipts=None,
):  # fmt: skip
    """Three-update trajectory; with start_version > 0 the first start_version steps come from a prior run.

    prior_receipts: completed steps' receipts, each carrying resumed_from.group (the prior group dir).
    initial_policy / initial_checkpoint: the prior run's policy-v{start} and checkpoint-v{start}, used read-only.
    """
    group = run / profile["id"] / dataset
    group.mkdir(parents=True)
    timeout = PHASE_TIMEOUT_SECONDS[dataset]
    policy, checkpoint = initial_policy, initial_checkpoint
    receipts = []
    links = []
    if start_version:
        if not 0 < start_version <= 3 or len(prior_receipts or []) != start_version or checkpoint is None:
            raise ValueError("resume needs start_version receipts and an initial checkpoint")
        links = link_prior_steps(work, group, prior_receipts[0]["resumed_from"]["group"], start_version)
        receipts = [dict(r) for r in prior_receipts]
        write(group / "steps.json", receipts)
    for version in range(start_version, 3):
        step_started = time.monotonic()
        request = work / f"data/requests-N/{dataset}-step{version + 1}.json"
        generated = group / f"v{version}-generate"
        generation = launch(
            work,
            run,
            rows,
            image,
            profile,
            "rollout-generate",
            generated,
            version=version,
            policy=policy,
            batch=request,
            timeout_seconds=timeout,
        )
        update = group / f"update-{version + 1}"
        next_checkpoint = group / f"checkpoint-v{version + 1}"
        train = launch(
            work,
            run,
            rows,
            image,
            profile,
            "actor-update",
            update,
            version=version,
            batch=generated / "generated-batch.json",
            resume=checkpoint,
            checkpoint=next_checkpoint,
            timeout_seconds=timeout,
        )
        policy, checkpoint = update / f"policy-v{version + 1}.pt", next_checkpoint
        raw = json.loads((generated / "generated-batch.json").read_text())
        receipt = dict(
            update=version + 1,
            generation=generation,
            actor=train,
            response_tokens=raw["response_tokens"],
            raw_rewards=[c["raw_reward"] for c in raw["samples"]],
            post_update_fresh_and_fixed_verification="PENDING",
        )
        receipts.append(receipt)
        write(group / "steps.json", receipts)
        # Every update, including the third, gets a fresh generation at its new weights.
        post = group / f"v{version + 1}-post-generate"
        post_request = work / f"data/requests-N/{dataset}-step4.json" if version == 2 else request
        post_generation = launch(
            work,
            run,
            rows,
            image,
            profile,
            "rollout-generate",
            post,
            version=version + 1,
            policy=policy,
            batch=post_request,
            timeout_seconds=timeout,
        )
        audit = group / f"v{version + 1}-actor-audit"
        launch(
            work,
            run,
            rows,
            image,
            profile,
            "actor-audit",
            audit,
            version=version + 1,
            resume=checkpoint,
            batch=post / "generated-batch.json",
            timeout_seconds=timeout,
        )
        fresh = verify(
            work, run, image, profile, audit, post, policy, audit, suite=post / "generated-suite.json"
        )
        fixed_actor = group / f"v{version + 1}-fixed-actor"
        launch(
            work,
            run,
            rows,
            image,
            profile,
            "actor-audit",
            fixed_actor,
            version=version + 1,
            resume=checkpoint,
            timeout_seconds=timeout,
        )
        fixed_rollout = group / f"v{version + 1}-fixed-rollout"
        launch(
            work,
            run,
            rows,
            image,
            profile,
            "rollout-score",
            fixed_rollout,
            version=version + 1,
            policy=policy,
            timeout_seconds=timeout,
        )
        fixed = verify(work, run, image, profile, fixed_actor, fixed_rollout, policy, fixed_actor)
        receipt.update(
            post_update_fresh_and_fixed_verification="PASS",
            fresh_verification=fresh,
            fixed_verification=fixed,
            post_update_generation=post_generation,
            end_to_end_step_seconds=time.monotonic() - step_started,
        )
        # Rank receipts represent one logical batch; PP ranks never multiply token counts.
        for rank in range(actor_nproc(profile)):
            evidence = json.loads((update / f"actor-rank-{rank}-v{version + 1}.json").read_text())
            if (
                not evidence["gradients_finite"]
                or not evidence["gradients_nonzero"]
                or not evidence["changed_parameters"]
            ):
                raise RuntimeError("native effective-update evidence failed")
            if not evidence["optimizer_continuity"] or evidence["weight_version"] != version + 1:
                raise RuntimeError("native optimizer/version continuity failed")
        write(group / "steps.json", receipts)
    command = container_base(work, run, [], image, group, gpu=False)
    command += [
        "-m",
        "verl.experimental.rl_kernel.phase_update_acceptance",
        "--root",
        mapped(work, group),
        "--output",
        "/outputs/update-verification.json",
    ]
    cid = subprocess.check_output(command, text=True).strip()
    with (group / "update-verifier.log").open("w") as log:
        verified = subprocess.run(["docker", "start", "-a", cid], stdout=log, stderr=subprocess.STDOUT)
    if verified.returncode:
        raise RuntimeError(
            "independent three-update replay failed; see " + str(group / "update-verification.json")
        )
    result = dict(
        status="PASS",
        scope="three effective updates, complete recovery and independent post-v1/v2/v3 audits",
        completed_updates=3,
        dataset=dataset,
        steps=receipts,
        resumed_updates=start_version,
        resume_links=links,
    )
    write(group / "result.json", result)
    return result


def acquire_gpu_locks(work, gpus, stack):
    """Hold one exclusive flock per physical GPU plus the legacy global lock in shared mode.

    Disjoint pools can run concurrently; overlapping pools and old-style exclusive
    holders of execute-N.lock conflict. On failure everything taken is released.
    """
    with contextlib.ExitStack() as taken:
        legacy = taken.enter_context((work / "execute-N.lock").open("a"))
        try:
            fcntl.flock(legacy, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("legacy execute-N.lock is held exclusively by another driver") from exc
        names, busy = [], []
        for g in sorted(gpus):
            name = f"execute-N-gpu{g}.lock"
            handle = taken.enter_context((work / name).open("a"))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                busy.append(g)
            names.append(name)
        if busy:
            raise RuntimeError(f"physical GPU(s) {busy} already locked by another matrix run")
        stack.push(taken.pop_all())
    return names


def build_provenance(image, runtime):
    return dict(
        image_id=image,
        sources=runtime["sources"],
        trace_audit_pattern=os.environ.get("RLK_TRACE_AUDIT_PATTERN"),
    )


def check_correctness_run(correctness_run, groups, image, runtime_lock):
    """Per-group `correctness_from` records; refuse unless every group is CORRECTNESS_PASS on this image."""
    root = Path(correctness_run).resolve()
    runtime = json.loads(Path(runtime_lock).read_text())
    if runtime["image_id"] != image:
        raise ValueError("runtime image lock differs")
    summary_path = root / "summary.json"
    if not summary_path.is_file() or not (root / "provenance.json").is_file():
        raise ValueError(f"{root}: correctness run needs summary.json and provenance.json")
    if json.loads((root / "provenance.json").read_text()).get("image_id") != image:
        raise ValueError("correctness run image_id differs from this run's image")
    summary = json.loads(summary_path.read_text())
    records = {}
    for name in groups:
        if summary.get(name, {}).get("status") != "CORRECTNESS_PASS":
            raise ValueError(f"{name}: correctness run summary status is not CORRECTNESS_PASS")
        verification = root / name / "copy" / "update-verification.json"
        if not verification.is_file():
            raise ValueError(f"{name}: correctness run has no copy/update-verification.json")
        records[name] = dict(
            run_root=str(root),
            summary_sha256=sha256_file(summary_path),
            update_verification=str(verification),
            update_verification_sha256=sha256_file(verification),
        )
    return records


def check_resume_run(resume_target, provenance):
    """Phase checkpoints embed provenance and must match exactly; resume only within the same image/sources."""
    root = Path(resume_target).resolve()
    path = root / "provenance.json"
    if not path.is_file():
        raise ValueError(f"{root}: resume run has no provenance.json")
    if json.loads(path.read_text()) != provenance:
        raise ValueError("resume run provenance (image/sources/trace pattern) differs; resume refused")
    return root


def run(
    work, image, gpus, groups, runtime_lock, shared_policy, health_receipt, fixed_actor=None,
    *, target_only=False, correctness_run=None, resume_target=None,
):  # fmt: skip
    targets = {p["id"]: p for p in profiles()}
    if not groups or len(set(groups)) != len(groups) or set(groups) - set(targets):
        raise ValueError("select unique groups from N1..N5")
    if not gpus or len(set(gpus)) != len(gpus) or any(g not in range(8) for g in gpus):
        raise ValueError("select unique physical GPU ordinals in 0..7")
    if target_only and not correctness_run:
        raise ValueError("--target-only requires --correctness-run")
    if correctness_run and not target_only:
        raise ValueError("--correctness-run is only valid with --target-only")
    if resume_target and not target_only:
        raise ValueError("--resume-target is only valid with --target-only")
    correctness_from, resume_root = {}, None
    if target_only:
        if not runtime_lock:
            raise ValueError("runtime lock required")
        correctness_from = check_correctness_run(correctness_run, groups, image, runtime_lock)
        if resume_target:
            resume_root = check_resume_run(
                resume_target, build_provenance(image, json.loads(Path(runtime_lock).read_text()))
            )
    # N5 always runs last so it can never delay or block N1..N4.
    primary = [targets[name] for name in sorted(groups, key=lambda n: n == "N5")]
    common_profile = next((p for p in primary if p["id"] != "N5"), targets["N1"])
    if any(p["peak_physical_gpus"] > len(gpus) for p in primary):
        raise ValueError("physical pool too small; do not reduce target TP")
    runs = work / "runs-N"
    runs.mkdir(exist_ok=True)
    with contextlib.ExitStack() as locks:
        gpu_locks = acquire_gpu_locks(work, gpus, locks)
        run_root = runs / (
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        )
        run_root.mkdir()
        if not health_receipt:
            raise ValueError("single-GPU health and eight-GPU communication receipt required")
        health = json.loads(health_receipt.read_text())
        if (
            health["status"] != "PASS"
            or health["image_id"] != image
            or health["communication_world_size"] != 8
        ):
            raise ValueError("health/communication receipt does not qualify this exact runtime image")
        if shared_policy:
            policy = json.loads(shared_policy.read_text())
            import hashlib

            if health["resource_policy_sha256"] != hashlib.sha256(shared_policy.read_bytes()).hexdigest():
                raise ValueError("shared exception changed since health qualification")
            write(run_root / "resource-policy-lock.json", policy)
        write(run_root / "health-receipt.json", health)
        rows = allocation(gpus)
        write(run_root / "allocation.json", rows)
        runtime = json.loads(runtime_lock.read_text())
        if runtime["image_id"] != image:
            raise ValueError("runtime image lock differs")
        write(run_root / "provenance.json", build_provenance(image, runtime))
        summary = {
            p["id"]: dict(status="NOT_RUN", fixed_weight="NOT_RUN", updates=0, target_load="NOT_RUN")
            for p in profiles()
        }
        for name, record in correctness_from.items():
            summary[name]["correctness_from"] = record
        if target_only:
            write(
                run_root / "target-only.json",
                dict(correctness_run=str(Path(correctness_run).resolve()), resume_target=str(resume_root) if resume_root else None),
            )
        for name in set(summary) - set(groups):
            summary[name].update(status="DEFERRED", reason="not selected for this run")
        write(
            run_root / "resource-policy.json",
            dict(
                selected_physical_gpus=gpus,
                gpu_locks=gpu_locks,
                selected_groups=groups,
                machine_gpus=8,
                local_lock_is_reservation=False,
                phase_checks="actual active devices only",
            ),
        )
        write(run_root / "summary.json", summary)
        summarize(run_root)
        common = fixed_actor or run_root / "common-fixed-actor"
        if fixed_actor:
            score = json.loads((common / "scores.json").read_text())
            import hashlib

            if (
                score["input_suite_sha256"]
                != hashlib.sha256((work / "evidence/fixed-input-suite.json").read_bytes()).hexdigest()
                or score["producer"]["weight_version"] != 0
            ):
                raise ValueError("retained actor snapshot/suite/version does not qualify")
            write(
                run_root / "retained-fixed-actor.json",
                dict(
                    path=str(common), scope="actual prior independent actor forward; original snapshot only"
                ),
            )
        else:
            launch(work, run_root, rows, image, common_profile, "actor-score", common)
        initial_policy = common / "policy-v0.pt"
        policies = {}
        for profile in primary:
            name = profile["id"]
            base = run_root / name
            base.mkdir()
            summary[name].update(status="RUNNING", fixed_weight="RUNNING")
            write(run_root / "summary.json", summary)
            try:
                policy0, reference = initial_policy, common
                if name == "N5":
                    # EP4/DP4 actor topology differs from the common EP1 actor: score it on its own,
                    # require bitwise equality with the EP1 actor, then pin rollout to the EP4 actor.
                    reference = base / "fixed-actor"
                    launch(work, run_root, rows, image, profile, "actor-score", reference)
                    cross = compare_actor_scores(reference, common)
                    write(base / "ep4-vs-ep1-actor.json", cross)
                    if not cross["compared"] or cross["mismatch"]:
                        raise RuntimeError(
                            f"N5 EP4 actor scores differ from EP1 actor: {cross['mismatch']} of {cross['expected']}"
                        )
                    manifests = [json.loads((d / "policy-v0-manifest.json").read_text()) for d in (reference, common)]
                    if not manifests[0] or manifests[0] != manifests[1]:
                        raise RuntimeError("N5 EP4 actor initial policy export differs from the EP1 export")
                    policy0 = reference / "policy-v0.pt"
                policies[name] = policy0
                rollout = base / "fixed-rollout"
                launch(work, run_root, rows, image, profile, "rollout-score", rollout, policy=policy0)
                verification = verify(work, run_root, image, profile, reference, rollout, policy0, rollout)
                summary[name].update(fixed_weight="PASS", fixed_verification=verification)
                write(run_root / "summary.json", summary)
                summarize(run_root)
                if target_only:
                    # Copy-task correctness is taken from the referenced run (see correctness_from).
                    summary[name].update(status="TARGET_PENDING")
                else:
                    correct = training(work, run_root, rows, image, profile, policy0, "copy")
                    summary[name].update(status="CORRECTNESS_PASS", updates=correct["completed_updates"])
            except ResourceConflict:
                raise
            except Exception as e:
                summary[name].update(status="FAIL", error=str(e))
                if summary[name]["fixed_weight"] == "RUNNING":
                    summary[name]["fixed_weight"] = "FAIL"
                partial = run_root / name / "copy/steps.json"
                if partial.exists():
                    saved = json.loads(partial.read_text())
                    summary[name].update(
                        recorded_optimizer_steps=len(saved),
                        independently_audited_steps=sum(
                            s["post_update_fresh_and_fixed_verification"] == "PASS" for s in saved
                        ),
                    )
            write(run_root / "summary.json", summary)
            summarize(run_root)
        # Real-data load is its own fresh three-update trajectory and cannot replace correctness.
        for profile in primary:
            name = profile["id"]
            if summary[name]["status"] != ("TARGET_PENDING" if target_only else "CORRECTNESS_PASS"):
                continue
            try:
                resume = {}
                if resume_root:
                    count, r_policy, r_checkpoint, r_receipts = resume_plan(resume_root / name / "dapo")
                    summary[name]["resume"] = dict(from_run=str(resume_root), completed_updates=count)
                    if count:
                        resume = dict(
                            start_version=count, initial_checkpoint=r_checkpoint, prior_receipts=r_receipts
                        )
                        write(run_root / "summary.json", summary)
                target = training(
                    work, run_root, rows, image, profile,
                    r_policy if resume else policies[name], "dapo", **resume,
                )  # fmt: skip
                summary[name].update(target_load="PASS", target_updates=target["completed_updates"])
                if resume:
                    summary[name]["resume"]["links"] = target["resume_links"]
                if target_only:
                    summary[name]["status"] = "TARGET_ONLY_PASS"
            except ResourceConflict:
                raise
            except Exception as e:
                summary[name].update(target_load="FAIL", target_error=str(e))
                if target_only:
                    summary[name]["status"] = "FAIL"
            write(run_root / "summary.json", summary)
            summarize(run_root)
        if "N5" not in groups:
            summary["N5"].update(
                status="DEFERRED",
                reason="not selected for this run; N5 EP4 experimental, GPU acceptance pending; no EP1 substitution",
                primary_correctness_gate=all(
                    summary[p["id"]]["status"] in ("CORRECTNESS_PASS", "TARGET_ONLY_PASS") for p in profiles()[:4]
                ),
            )
        write(run_root / "summary.json", summary)
        summarize(run_root)
        print(run_root, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--gpus", nargs="+", type=int, required=True)
    ap.add_argument("--groups", nargs="+", choices=["N1", "N2", "N3", "N4", "N5"], required=True)
    ap.add_argument("--runtime-lock", type=Path, required=True)
    ap.add_argument("--resource-policy", type=Path)
    ap.add_argument("--health-receipt", type=Path, required=True)
    ap.add_argument("--fixed-actor", type=Path)
    ap.add_argument("--target-only", action="store_true", help="skip copy-task training; run fixed-weight + DAPO only")
    ap.add_argument("--correctness-run", type=Path, help="run root whose summary shows CORRECTNESS_PASS (needs --target-only)")
    ap.add_argument("--resume-target", type=Path, help="earlier target-only run root to continue DAPO from (needs --target-only)")
    args = ap.parse_args()
    if args.target_only and not args.correctness_run:
        ap.error("--target-only requires --correctness-run")
    if (args.correctness_run or args.resume_target) and not args.target_only:
        ap.error("--correctness-run/--resume-target require --target-only")
    run(
        args.work.resolve(),
        args.image,
        args.gpus,
        args.groups,
        args.runtime_lock.resolve(),
        args.resource_policy,
        args.health_receipt,
        args.fixed_actor,
        target_only=args.target_only,
        correctness_run=args.correctness_run,
        resume_target=args.resume_target,
    )


if __name__ == "__main__":
    main()
