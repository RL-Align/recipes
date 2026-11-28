# SPDX-License-Identifier: MIT
"""Sequential shared GPU0 health then eight-GPU RCCL check in the exact image."""

import argparse
import hashlib
import json
import socket
import subprocess
import time
from pathlib import Path
from uuid import uuid4

from execute import allocation, container_base, mapped, observe, stop_owned_container, wait_idle, write


def phase(work, root, image, policy, rows, mode, *, trace_run=None, active=None, label=None):
    active = active if active is not None else (rows if mode == "communication" else rows[:1])
    gpus = [r["physical_gpu"] for r in active]
    out = root / (label or mode)
    out.mkdir()
    wait_idle(root, gpus)
    command = container_base(work, root, active, image, out)
    if mode == "communication":
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        command += [
            "-m",
            "torch.distributed.run",
            "--rdzv_backend=static",
            "--nnodes=1",
            "--node_rank=0",
            "--master_addr=127.0.0.1",
            f"--master_port={port}",
            f"--nproc_per_node={len(active)}",
        ]
    if mode in ("q-up-replay", "indexer-replay"):
        if trace_run is None:
            raise ValueError("Q-up replay requires saved real paired trace operands")
        command += [
            "-m",
            "examples.operators.dsv4_4l.replay_q_up"
            if mode == "q-up-replay"
            else "examples.operators.dsv4_4l.replay_indexer",
            "--trace-run",
            mapped(work, trace_run),
            "--model",
            "/experiment/models/DeepSeek-V4-Flash-7L",
            "--output",
            "/outputs",
        ]
    else:
        command += ["/opt/recipes/scripts/seven_layer/health.py", mode, "--output", "/outputs"]
        if mode == "communication" and len(active) != 8:
            command += ["--world-size", str(len(active))]
    write(out / "command.json", command)
    cid = subprocess.check_output(command, text=True).strip()
    write(out / "container.json", dict(id=cid, mode=mode))
    wait_idle(root, gpus)
    started, next_sample = time.monotonic(), 0
    with (out / "driver.log").open("w") as log:
        proc = subprocess.Popen(["docker", "start", "-a", cid], stdout=log, stderr=subprocess.STDOUT)
        try:
            while proc.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed > 180:
                    raise TimeoutError("health/communication phase exceeded 180 seconds")
                if elapsed >= next_sample:
                    next_sample = elapsed + 5
                    sample, _ = observe(root, gpus, own_container=cid)
                    with (root / "gpu-samples.jsonl").open("a") as f:
                        f.write(json.dumps(sample) + "\n")
                time.sleep(1)
        except BaseException as error:
            write(out / "conflict-or-error.json", dict(error=str(error)))
            stop_owned_container(cid, root.name)
            proc.wait()
            raise
    if proc.returncode:
        raise RuntimeError("actual health/communication failed; retain logs and do not launch model phases")
    records = [json.loads((out / f"rank-{r}.json").read_text()) for r in range(len(active))]
    if any(
        row["status"] != "PASS"
        or row["rank"] != rank
        or row["identity"]["pci_bus_id"] != active[rank]["pci_bus_id"]
        for rank, row in enumerate(records)
    ):
        raise RuntimeError("actual health results/ranks/PCI identity differ")
    wait_idle(root, gpus)
    return dict(status="PASS", seconds=time.monotonic() - started, ranks=records)


def validate_pool(gpus):
    pool = sorted(gpus)
    if len(pool) < 2 or len(set(pool)) != len(pool) or any(g not in range(8) for g in pool):
        raise ValueError("qualification pool: select at least two unique physical GPU ordinals in 0..7")
    return pool


def qualify(work, runtime_lock, resource_policy, *, q_up_trace=None, indexer_trace=None, gpus=None):
    """Health + RCCL receipt of a physical GPU pool (default: all eight, exactly as before).

    A smaller pool runs the same single-card check on EVERY selected card (the full pool checks card 0 only, unchanged) and an
    all_reduce/broadcast/reduce_scatter/all_to_all check across exactly the pool's ranks. Every phase is gated by wait_idle on the
    pool's GPUs (idle, no foreign process). The receipt records qualified_physical_gpus; execute.py accepts it only for runs whose
    selected GPUs are a subset of that pool.
    """
    pool = list(range(8)) if gpus is None else validate_pool(gpus)
    full = pool == list(range(8))
    info = json.loads(runtime_lock.read_text())
    policy = json.loads(resource_policy.read_text())
    root = work / "evidence" / ("health-" + uuid4().hex[:12])
    root.mkdir()
    write(root / "resource-policy-lock.json", policy)
    image = info["image_id"]
    rows = allocation(pool)
    write(root / "allocation.json", rows)
    result = dict(
        status="PENDING",
        image_id=image,
        resource_policy_sha256=hashlib.sha256(resource_policy.read_bytes()).hexdigest(),
        qualified_physical_gpus=pool,
        scope="single-card HIP health and actual eight-rank RCCL only; no model/EP4 acceptance"
        if full
        else f"pool-scoped: single-card HIP health on each of GPUs {pool} and actual {len(pool)}-rank RCCL across exactly that pool; "
        "no model acceptance; cannot qualify groups needing more GPUs",
    )
    try:
        if full:
            result["single"] = phase(work, root, image, policy, rows, "single")
        else:
            cards = {
                str(row["physical_gpu"]): phase(
                    work, root, image, policy, rows, "single", active=[row], label=f"single-gpu{row['physical_gpu']}"
                )
                for row in rows
            }
            result["single_cards"] = cards
            result["single"] = cards[str(rows[0]["physical_gpu"])]
        result["communication"] = phase(work, root, image, policy, rows, "communication")
        if q_up_trace:
            result["q_up_diagnostic_replay"] = phase(
                work, root, image, policy, rows, "q-up-replay", trace_run=q_up_trace
            )
        if indexer_trace:
            result["indexer_diagnostic_replay"] = phase(
                work, root, image, policy, rows, "indexer-replay", trace_run=indexer_trace
            )
        result.update(status="PASS", communication_world_size=len(pool))
    except BaseException as error:
        result.update(status="FAIL", error=str(error))
        raise
    finally:
        write(root / "qualification.json", result)
        print(root / "qualification.json", flush=True)
    return root / "qualification.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--runtime-lock", type=Path, required=True)
    ap.add_argument("--resource-policy", type=Path, required=True)
    ap.add_argument("--gpus", nargs="+", type=int, help="qualify only this physical pool (default: all eight); the receipt then covers only these GPUs")
    args = ap.parse_args()
    qualify(args.work, args.runtime_lock, args.resource_policy, gpus=args.gpus)


if __name__ == "__main__":
    main()
