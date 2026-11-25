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

from execute import allocation, container_base, observe, wait_idle, write


def phase(work, root, image, policy, rows, mode):
    active = rows[:1] if mode == "single" else rows
    gpus = [r["physical_gpu"] for r in active]
    out = root / mode
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
            "--nproc_per_node=8",
        ]
    command += ["/opt/recipes/scripts/seven_layer/health.py", mode, "--output", "/outputs"]
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
            label = subprocess.check_output(
                ["docker", "inspect", '--format={{index .Config.Labels "rl-align.seven-layer-N"}}', cid],
                text=True,
            ).strip()
            if label == root.name:
                subprocess.run(["docker", "stop", "--timeout", "15", cid], check=True)
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


def qualify(work, runtime_lock, resource_policy):
    info = json.loads(runtime_lock.read_text())
    policy = json.loads(resource_policy.read_text())
    root = work / "evidence" / ("health-" + uuid4().hex[:12])
    root.mkdir()
    write(root / "resource-policy-lock.json", policy)
    image = info["image_id"]
    rows = allocation(list(range(8)))
    write(root / "allocation.json", rows)
    result = dict(
        status="PENDING",
        image_id=image,
        resource_policy_sha256=hashlib.sha256(resource_policy.read_bytes()).hexdigest(),
        scope="single-card HIP health and actual eight-rank RCCL only; no model/EP4 acceptance",
    )
    try:
        result["single"] = phase(work, root, image, policy, rows, "single")
        result["communication"] = phase(work, root, image, policy, rows, "communication")
        result.update(status="PASS", communication_world_size=8)
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
    args = ap.parse_args()
    qualify(args.work, args.runtime_lock, args.resource_policy)


if __name__ == "__main__":
    main()
