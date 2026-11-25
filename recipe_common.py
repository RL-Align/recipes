# SPDX-License-Identifier: MIT
"""Configuration and command contracts; no GPU/framework imports on the host."""

import hashlib
import ipaddress
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BOOK = ROOT / "models/deepseek-v4/rocm"
PYTHON = "/opt/dsv4-venv/bin/python"
KERNEL = "/opt/dsv4/RL-Kernel"
VERL = "/opt/dsv4/verl"
IMAGE = "rl-align/dsv4-recipe:local"


def read_json(path):
    return json.loads(Path(path).read_text())


def lock():
    return read_json(BOOK / "sources.lock.json")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_inputs(c):
    data = data_dir(c)
    manifest = read_json(data / "manifest.json")
    for name, expected in manifest["files"].items():
        if Path(name).name != name or sha256(data / name) != expected:
            raise ValueError(f"prepared data hash differs: {name}")
    model = Path(c["model_dir"])
    weights = list(model.glob("*.safetensors"))
    if len(weights) != 1 or sha256(weights[0]) != lock()["checkpoint"]["sha256"]:
        raise ValueError("four-layer checkpoint differs from the locked original")
    metadata = read_json(
        Path(KERNEL)
        / "examples/operators/dsv4_4l/results/native_grpo_pp2_dp1_20261117/checkpoint-assets.json"
    )
    for name, expected in metadata.items():
        if sha256(model / name) != expected["sha256"]:
            raise ValueError(f"model asset hash differs: {name}")
    return manifest


def config(path):
    data = read_json(path)
    validate(data)
    return data


def validate(c):
    if c.get("schema_version") != 1 or c.get("recipe") != "deepseek-v4/rocm":
        raise ValueError("expected recipe deepseek-v4/rocm, schema_version=1")
    if c.get("topology") not in ("single-node-8gpu", "four-node-32gpu"):
        raise ValueError("only single-node-8gpu and four-node-32gpu are implemented")
    if c.get("gpu_arch", "gfx942") not in ("gfx942", "gfx950"):
        raise ValueError("gpu_arch must be gfx942 (MI300X) or gfx950 (MI350X)")
    if type(c.get("numa_interleave", False)) is not bool:
        raise ValueError("numa_interleave must be boolean")
    count = 1 if c["topology"] == "single-node-8gpu" else 4
    nodes = c.get("nodes", [])
    if len(nodes) != count:
        raise ValueError(f"this topology requires exactly {count} physical nodes")
    ips = []
    for n in nodes:
        ip = str(ipaddress.IPv4Address(n["ip"]))
        ips.append(ip)
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,23}", n["name"]):
            raise ValueError("use a short lowercase node name")
        if n["ssh"] != "local" and not re.fullmatch(r"[a-zA-Z0-9_.@-]+", n["ssh"]):
            raise ValueError(
                "SSH target must be a hostname/user@hostname; configure ports/keys in ~/.ssh/config"
            )
        if n["ssh"].startswith("-"):
            raise ValueError("SSH target cannot be an option")
        if sorted(n["gpus"]) != list(range(8)) or any(type(i) is not int for i in n["gpus"]):
            raise ValueError("each selected node requires the eight idle physical GPUs 0..7")
    if len(set(ips)) != count or len({n["name"] for n in nodes}) != count:
        raise ValueError("nodes must have unique names and reachable physical IPs")
    for key in ("work_dir", "model_dir", "baseline_data_dir"):
        p = c[key]
        if not isinstance(p, str) or not p.startswith("/") or p == "/" or any(x in p for x in ",:\n"):
            raise ValueError(f"{key} must be an absolute path without commas, colons or newlines")
    profile = c.get("profile", "smoke")
    if profile not in ("smoke", "math-smoke", "math-200"):
        raise ValueError("profile must be smoke, math-smoke or math-200")
    if c.get("dataset") not in ("copy", "dapo", "gsm8k"):
        raise ValueError("dataset must be copy, dapo or gsm8k")
    if (profile == "smoke") != (c["dataset"] == "copy"):
        raise ValueError("smoke uses copy; both math profiles require dapo/gsm8k")
    net = c["network"]
    if type(net["ray_port"]) is not int or not 1024 <= net["ray_port"] <= 64000:
        raise ValueError("ray_port must be an unused port in 1024..64000")
    if not re.fullmatch(r"[a-zA-Z0-9_,.=^-]*", net.get("ifname", "")):
        raise ValueError("invalid network interface selector")
    if type(net.get("rdma", False)) is not bool:
        raise ValueError("network.rdma must be boolean")


def plan(c):
    validate(c)
    multi = len(c["nodes"]) == 4
    math = c["profile"] != "smoke"
    return dict(
        nodes=len(c["nodes"]),
        total_gpus=32 if multi else 8,
        actor_nodes=[n["ip"] for n in c["nodes"][:2]] if multi else [],
        rollout_nodes=[n["ip"] for n in c["nodes"][2:]] if multi else [],
        actor_gpus=16 if multi else 4,
        actor_pp=2,
        actor_dp=8 if multi else 2,
        rollout_gpus=16 if multi else 4,
        rollout_replicas=8 if multi else 2,
        rollout_pp=2,
        tp=1,
        ep=1,
        cp=1,
        original_layer_partition=[[0, 1, 2], [3]],
        steps=200 if c["profile"] == "math-200" else 3,
        prompts=8 if multi or math else 2,
        samples=16 if math else 2,
        max_prompt_length=1024 if math else 128,
        max_response_length=7168 if math else 16,
        status="EXPERIMENTAL_GPU_VALIDATION_PENDING",
    )


def data_dir(c):
    p = plan(c)
    return Path(c["work_dir"]) / "data" / f"{c['dataset']}-p{p['prompts']}"


def launch(c, output, prompts):
    """Native verl GRPO, fixed numerical settings, explicit physical placement."""
    p = plan(c)
    multi = p["nodes"] > 1
    data = data_dir(c)
    output = Path(output)
    values = {
        "actor_rollout_ref.model.path": c["model_dir"],
        "data.train_files": str(data / "train.parquet"),
        "data.val_files": str(data / "val.parquet"),
        "data.train_batch_size": p["prompts"],
        "data.max_prompt_length": p["max_prompt_length"],
        "data.max_response_length": p["max_response_length"],
        "data.dataloader_num_workers": 0,
        "data.shuffle": False,
        "actor_rollout_ref.actor.ppo_mini_batch_size": p["prompts"],
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.rollout.n": p["samples"],
        "actor_rollout_ref.rollout.seed": 42,
        "actor_rollout_ref.rollout.max_model_len": 8192 if c["profile"] != "smoke" else 256,
        "actor_rollout_ref.rollout.max_num_batched_tokens": 8192 if c["profile"] != "smoke" else 512,
        "actor_rollout_ref.rollout.max_num_seqs": 128 if c["profile"] != "smoke" else 8,
        "actor_rollout_ref.rollout.enable_chunked_prefill": False,
        "actor_rollout_ref.rollout.agent.num_workers": 1,
        "trainer.nnodes": 2 if multi else 1,
        "trainer.n_gpus_per_node": 8 if multi else 4,
        "actor_rollout_ref.rollout.nnodes": 2 if multi else 1,
        "actor_rollout_ref.rollout.n_gpus_per_node": 8 if multi else 4,
        "trainer.total_training_steps": p["steps"],
        "trainer.total_epochs": 1,
        "trainer.save_freq": -1,
        "trainer.default_local_dir": str(output / "checkpoints"),
        "reward.num_workers": 1,
        "reward.custom_reward_function.path": f"{KERNEL}/examples/operators/dsv4_4l/copy_task.py"
        if c["dataset"] == "copy"
        else None,
        "reward.custom_reward_function.name": "compute_score",
        "rlk.multinode": multi,
        "rlk.actor_node_ips": p["actor_nodes"],
        "actor_rollout_ref.rollout.rlk_node_ips": p["rollout_nodes"],
        "rlk.sync_dir": str(output / "policy-sync"),
        "rlk.audit_response_length": 16,
        "rlk.audit_prompts": prompts,
        "+rlk.stress_final_repeats": 0,
        "rlk.preflight_files": [
            str(data / name)
            for name in (
                "checkpoint-inventory.json",
                "replay-fixed-sequences.json",
                "extended-audit-prompts.json",
                "manifest.json",
            )
        ],
        "+ray_kwargs.ray_init.address": f"{c['nodes'][0]['ip']}:{c['network']['ray_port']}",
        "ray_kwargs.ray_init.num_cpus": None,
        "hydra.run.dir": str(output / "hydra"),
    }
    env = dict(
        OMP_NUM_THREADS="4",
        TOKENIZERS_PARALLELISM="false",
        VLLM_ROCM_USE_AITER="1",
        VLLM_ENABLE_V1_MULTIPROCESSING="1",
        VLLM_PP_LAYER_PARTITION="3,1",
        DSV4_TRACE_FORMAT="cas-v1",
        HYDRA_FULL_ERROR="1",
        RLK_REPLAY_FIXED_CASES=str(data / "replay-fixed-sequences.json"),
    )
    # Do not propagate the head's HIP/CUDA visibility or VLLM_HOST_IP to peers.
    for k, v in env.items():
        values["++ray_kwargs.ray_init.runtime_env.env_vars." + k] = v
    command = [PYTHON, "-m", "verl.experimental.rl_kernel.main_ppo", "--config-name", "dsv4_grpo"]
    return command + [f"{k}={json.dumps(v)}" for k, v in values.items()], env
