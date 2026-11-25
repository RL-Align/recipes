# SPDX-License-Identifier: MIT
"""Run inside the head container; retain process exits and independent acceptance."""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "/opt/recipes")
from recipe_common import KERNEL, config, data_dir, launch, lock, plan, verify_inputs, write_json


def preflight(c):
    import ray
    import torch

    if torch.version.hip is None or torch.cuda.device_count() != 8:
        raise RuntimeError("runtime must see exactly the eight allocated ROCm GPUs")
    devices = []
    for index in range(8):
        props = torch.cuda.get_device_properties(index)
        arch = props.gcnArchName.split(":")[0]
        if arch != c.get("gpu_arch", "gfx942"):
            raise RuntimeError(f"GPU {arch} differs from the configured/built architecture")
        value = torch.ones(1, device=f"cuda:{index}")
        if value.item() != 1:
            raise RuntimeError("GPU health probe failed")
        devices.append(dict(device=index, arch=arch, name=props.name))
    from verl.experimental.rl_kernel.runtime import process_identity

    identity = process_identity("recipe-preflight")
    pins = lock()
    for project, key in (("RL-Kernel", "kernel_tree"), ("verl", "verl_tree")):
        root = f"/opt/dsv4/{project}"
        tree = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD^{tree}"], text=True).strip()
        if tree != pins[key] or identity["projects"][project]["dirty"]:
            raise RuntimeError(f"{project}: effective source tree differs or is dirty")
    for project, key in (("vllm", "vllm"), ("Megatron-LM", "megatron")):
        if identity["projects"][project]["sha"] != pins[key]:
            raise RuntimeError(f"{project}: dependency source differs")
    inputs = verify_inputs(c)
    binaries = json.loads(Path("/opt/recipe-build/binaries.json").read_text())
    return dict(
        devices=devices, identity=identity, ray_version=ray.__version__, inputs=inputs, binaries=binaries
    )


def verify(c, output):
    verifier = Path(KERNEL) / "examples/operators/dsv4_4l/verify_grpo.py"
    spec = importlib.util.spec_from_file_location("recipe_native_verifier", verifier)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The locked source patch generalizes only layout/counts to this workload.
    # Raw tensors, optimizer/DDP/PP and complete stage readback remain required.
    return module.verify(output / "policy-sync", data_dir(c) / "checkpoint-inventory.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("probe", "run", "verify"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    c, output = config(args.config), args.output
    if args.action == "probe":
        print("RUNTIME_PROBE=" + json.dumps(preflight(c)), flush=True)
        return
    if args.action == "verify":
        write_json(output / "verification.json", dict(status="VERIFYING"))
        result = verify(c, output)
        write_json(output / "verification.json", result)
        return
    import ray

    address = f"{c['nodes'][0]['ip']}:{c['network']['ray_port']}"
    ray.init(address=address, log_to_driver=False)
    live = {n["NodeManagerAddress"]: n for n in ray.nodes() if n["Alive"]}
    if set(live) != {n["ip"] for n in c["nodes"]} or any(
        n["Resources"].get("GPU") != 8 for n in live.values()
    ):
        raise RuntimeError("Ray cluster node/GPU allocation differs from the declared dedicated cluster")
    ray.shutdown()
    prompts = json.loads((data_dir(c) / "extended-audit-prompts.json").read_text())
    if len(prompts) != 3 or not all(isinstance(p, str) and p for p in prompts):
        raise ValueError("three nonempty audit prompts required")
    command, env = launch(c, output, prompts)
    if c.get("numa_interleave", False):
        command = ["numactl", "--interleave=all", *command]
    write_json(output / "launch.json", dict(command=command, environment=env, plan=plan(c)))
    start = time.monotonic()
    with (output / "driver.log").open("w") as stream:
        proc = subprocess.run(command, env={**os.environ, **env}, stdout=stream, stderr=subprocess.STDOUT)
    write_json(output / "exit-code.json", dict(exit_code=proc.returncode))
    write_json(
        output / "job-result.json", dict(exit_code=proc.returncode, wall_seconds=time.monotonic() - start)
    )
    if proc.returncode:
        raise RuntimeError(f"training failed ({proc.returncode}); see {output / 'driver.log'}")
    result = verify(c, output)
    write_json(output / "verification.json", result)
    print(f"INDEPENDENT_ACCEPTANCE_PASS {output}", flush=True)


if __name__ == "__main__":
    main()
