# SPDX-License-Identifier: MIT
"""Sent to each allocated host over SSH; never stops an existing Ray service."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path


def run(command, **kwargs):
    return subprocess.check_output(command, text=True, **kwargs).strip()


def allocation(gpus):
    bus = json.loads(run(["rocm-smi", "-d", *map(str, gpus), "--showbus", "--json"]))
    topology = {}
    for node in Path("/sys/class/kfd/kfd/topology/nodes").iterdir():
        props = dict(line.split(" ", 1) for line in (node / "properties").read_text().splitlines())
        minor = int(props.get("drm_render_minor", 0))
        if minor:
            topology[f"renderD{minor}"] = int((node / "gpu_id").read_text())
    rows = []
    for ordinal in gpus:
        pci = bus[f"card{ordinal}"]["PCI Bus"].lower()
        matches = [
            p for p in Path("/sys/class/drm").glob("renderD*") if (p / "device").resolve().name.lower() == pci
        ]
        if len(matches) != 1 or matches[0].name not in topology:
            raise RuntimeError(f"physical GPU {ordinal}: missing unique PCI/DRM/KFD mapping")
        rows.append(
            dict(
                physical_gpu=ordinal,
                pci_bus_id=pci,
                render_node="/dev/dri/" + matches[0].name,
                kfd_gpu_id=topology[matches[0].name],
            )
        )
    if len({r["pci_bus_id"] for r in rows}) != len(gpus):
        raise RuntimeError("GPU allocation contains duplicate physical PCI devices")
    return rows


def idle(gpus):
    processes = run(["rocm-smi", "--showpidgpus"])
    owner, occupied = None, []
    for line in processes.splitlines():
        match = re.match(r"PID (\d+) is using (\d+) DRM", line)
        if match:
            owner = (int(match[1]), int(match[2]))
        elif owner and owner[1] and re.fullmatch(r"[\d\s]+", line):
            if set(map(int, line.split())) & set(gpus):
                occupied.append(owner[0])
    memory = json.loads(run(["rocm-smi", "-d", *map(str, gpus), "--showmeminfo", "vram", "--json"]))
    resident = {g: int(memory[f"card{g}"]["VRAM Total Used Memory (B)"]) for g in gpus}
    if occupied or any(n >= 1024**3 for n in resident.values()):
        raise RuntimeError(f"selected GPUs are occupied: PIDs={occupied}, VRAM={resident}")
    return resident


def check(payload):
    node = payload["node"]
    c = payload["config"]
    output = Path(payload["output"])
    if (output / "storage-challenge").read_text() != payload["nonce"]:
        raise RuntimeError("work_dir is not the same shared filesystem on every node")
    (output / f"storage-{node['name']}.json").write_text(
        json.dumps(dict(nonce=payload["nonce"], node=node["name"]))
    )
    image_id = run(["docker", "image", "inspect", "--format={{.Id}}", payload["image"]])
    if image_id != payload["image_id"]:
        raise RuntimeError("node image ID differs from the locked build")
    return dict(
        node=node,
        image_id=image_id,
        memory=idle(node["gpus"]),
        allocation=allocation(node["gpus"]),
        filesystem=run(["df", "-h", c["work_dir"]]),
    )


def start(payload):
    c, node = payload["config"], payload["node"]
    # Recheck immediately before allocating. This is a guard, not a scheduler lock.
    idle(node["gpus"])
    rows = allocation(node["gpus"])
    command = [
        "docker",
        "run",
        "-d",
        "--init",
        "--name",
        payload["container"],
        "--label",
        "rl-align.recipe.run=" + payload["run_id"],
        "--network=host",
        "--ipc=private",
        "--shm-size=16g",
        "--device=/dev/kfd",
        # Rootless Docker's synthetic sysfs can omit KFD/DRM/PCI topology.
        # This is metadata only; DRM devices and the KFD allowlist stay selected.
        "--mount",
        "type=bind,source=/sys,target=/sys,readonly",
    ]
    if c.get("numa_interleave", False):
        command += ["--cap-add=SYS_NICE"]
    for row in rows:
        command += ["--device=" + row["render_node"]]
    for gid in sorted({os.stat("/dev/kfd").st_gid, *[os.stat(r["render_node"]).st_gid for r in rows]}):
        command += ["--group-add", str(gid)]
    if c["network"].get("rdma"):
        if not Path("/dev/infiniband").exists():
            raise RuntimeError("RDMA requested but /dev/infiniband is absent")
        command += ["--device=/dev/infiniband"]
    output = Path(payload["output"])
    allocation_file = output / f"allocation-{node['name']}.json"
    for local, row in enumerate(rows):
        row["container_gpu"] = local
        if len(c["nodes"]) == 1:
            row["role"] = "actor" if local < len(rows) // 2 else "rollout"
        else:
            row["role"] = "actor" if node in c["nodes"][:2] else "rollout"
    allocation_file.write_text(json.dumps(rows, indent=2))
    cache = Path(c["work_dir"]) / "cache" / node["name"]
    cache.mkdir(parents=True, exist_ok=True)
    mounts = [
        # Native lossless failure captures use /outputs/trace-tensors.
        # Bind it to this attempt so diagnostics survive container cleanup.
        (str(output), "/outputs", "rw"),
        (c["work_dir"], c["work_dir"], "rw"),
        (c["model_dir"], c["model_dir"], "ro"),
        (c["baseline_data_dir"], c["baseline_data_dir"], "ro"),
    ]
    for source, dest, mode in mounts:
        command += [
            "--mount",
            f"type=bind,source={source},target={dest}" + (",readonly" if mode == "ro" else ""),
        ]
    env = dict(
        LD_PRELOAD="/opt/dsv4/lib/rlk-isolated-kfd.so",
        RLK_ALLOWED_KFD_GPU_IDS=",".join(str(r["kfd_gpu_id"]) for r in rows),
        HIP_VISIBLE_DEVICES=",".join(map(str, range(len(rows)))),
        VLLM_HOST_IP=node["ip"],
        RLK_DEVICE_ALLOCATION=str(allocation_file),
        HF_HOME=str(cache / "huggingface"),
        TRITON_CACHE_DIR=str(cache / "triton"),
        TORCH_EXTENSIONS_DIR=str(cache / "torch_extensions"),
        OMP_NUM_THREADS="4",
        NCCL_DEBUG="INFO",
        RAY_USAGE_STATS_ENABLED="0",
    )
    if c["network"].get("ifname"):
        env.update(NCCL_SOCKET_IFNAME=c["network"]["ifname"], GLOO_SOCKET_IFNAME=c["network"]["ifname"])
    for key, val in env.items():
        command += ["-e", f"{key}={val}"]
    command += ["--entrypoint", "/bin/sh", payload["image"], "-c", "exec sleep infinity"]
    container_id = run(command)
    return dict(container_id=container_id, command=command, allocation=rows)


def owned(payload):
    label = run(
        ["docker", "inspect", '--format={{index .Config.Labels "rl-align.recipe.run"}}', payload["container"]]
    )
    if label != payload["run_id"]:
        raise RuntimeError("refusing to operate on a container belonging to another run")


def ray_start(payload):
    owned(payload)
    c, node = payload["config"], payload["node"]
    port = c["network"]["ray_port"]
    command = [
        "docker",
        "exec",
        payload["container"],
        "/opt/dsv4-venv/bin/ray",
        "start",
        "--node-ip-address=" + node["ip"],
        f"--num-gpus={len(node['gpus'])}",
        "--num-cpus=48",
        "--disable-usage-stats",
        f"--node-manager-port={port + 1}",
        f"--object-manager-port={port + 2}",
        f"--dashboard-agent-listen-port={port + 5}",
        f"--dashboard-agent-grpc-port={port + 6}",
        f"--runtime-env-agent-port={port + 7}",
        f"--min-worker-port={port + 100}",
        f"--max-worker-port={port + 299}",
    ]
    if c.get("numa_interleave", False):
        command[3:3] = ["numactl", "--interleave=all"]
    if node["name"] == c["nodes"][0]["name"]:
        command += [
            "--head",
            f"--port={port}",
            f"--ray-client-server-port={port + 3}",
            "--include-dashboard=false",
            f"--dashboard-port={port + 4}",
        ]
    else:
        command += [f"--address={c['nodes'][0]['ip']}:{port}"]
    return dict(command=command, stdout=run(command, stderr=subprocess.STDOUT))


def stop(payload):
    owned(payload)
    output = Path(payload["output"])
    logs = subprocess.run(["docker", "logs", payload["container"]], capture_output=True, text=True)
    (output / f"container-{payload['node']['name']}.log").write_text(logs.stdout + logs.stderr)
    return dict(stdout=run(["docker", "rm", "-f", payload["container"]]))


if __name__ == "__main__":
    try:
        action, encoded = sys.argv[1:]
        result = {"check": check, "start": start, "ray": ray_start, "stop": stop}[action](json.loads(encoded))
        print("RLK_RECIPE_RECEIPT=" + json.dumps(result))
    except Exception as error:
        print(f"RECIPE_NODE_FAILED {type(error).__name__}: {error}", file=sys.stderr)
        raise
