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
            port, retries, version, policy, batch, resume, checkpoint,
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
    port, retries, version, policy, batch, resume, checkpoint,
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
            if elapsed > 7200:
                stop_owned_container(cid, run.name, timeout=30)
                raise TimeoutError("own phase exceeded two-hour timeout; retained logs and checkpoints")
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


def training(work, run, rows, image, profile, initial_policy, dataset):
    group = run / profile["id"] / dataset
    group.mkdir(parents=True)
    policy, checkpoint = initial_policy, None
    receipts = []
    for version in range(3):
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


def run(work, image, gpus, groups, runtime_lock, shared_policy, health_receipt, fixed_actor=None):
    targets = {p["id"]: p for p in profiles()}
    if not groups or len(set(groups)) != len(groups) or set(groups) - set(targets):
        raise ValueError("select unique groups from N1..N5")
    if not gpus or len(set(gpus)) != len(gpus) or any(g not in range(8) for g in gpus):
        raise ValueError("select unique physical GPU ordinals in 0..7")
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
        write(
            run_root / "provenance.json",
            dict(
                image_id=image,
                sources=runtime["sources"],
                trace_audit_pattern=os.environ.get("RLK_TRACE_AUDIT_PATTERN"),
            ),
        )
        summary = {
            p["id"]: dict(status="NOT_RUN", fixed_weight="NOT_RUN", updates=0, target_load="NOT_RUN")
            for p in profiles()
        }
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
            if summary[name]["status"] != "CORRECTNESS_PASS":
                continue
            try:
                target = training(work, run_root, rows, image, profile, policies[name], "dapo")
                summary[name].update(target_load="PASS", target_updates=target["completed_updates"])
            except ResourceConflict:
                raise
            except Exception as e:
                summary[name].update(target_load="FAIL", target_error=str(e))
            write(run_root / "summary.json", summary)
            summarize(run_root)
        if "N5" not in groups:
            summary["N5"].update(
                status="DEFERRED",
                reason="not selected for this run; N5 EP4 experimental, GPU acceptance pending; no EP1 substitution",
                primary_correctness_gate=all(
                    summary[p["id"]]["status"] == "CORRECTNESS_PASS" for p in profiles()[:4]
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
    args = ap.parse_args()
    run(
        args.work.resolve(),
        args.image,
        args.gpus,
        args.groups,
        args.runtime_lock.resolve(),
        args.resource_policy,
        args.health_receipt,
        args.fixed_actor,
    )


if __name__ == "__main__":
    main()
