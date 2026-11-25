# SPDX-License-Identifier: MIT
"""Single-node seven-layer matrix preparation and genuine GPU startup diagnostics.

Diagnostics never satisfy the fixed-weight gate and cannot start GRPO updates.
"""

import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from node import allocation, idle  # noqa: E402


def profile(name, pp, tp, ep, etp, rollout_tp):
    world = pp * tp
    if world % (pp * etp * ep):
        raise ValueError("expert topology does not form complete groups")
    edp = world // (pp * etp * ep)
    if edp != 1:
        raise ValueError("expert data parallelism must be exactly one")
    return dict(
        id=name,
        status="experimental",
        original_layers=list(range(7)),
        physical_reservation=list(range(8)),
        actor=dict(
            pp=pp, tp=tp, ep=ep, etp=etp, dp=1, expert_dp=edp, cp=1, gpus=world, sequence_parallel=tp > 1
        ),
        rollout=dict(pp=1, tp=rollout_tp, ep=1, dp=1, cp=1, gpus=rollout_tp, instances=1),
        phase_reuse=True,
        peak_physical_gpus=max(world, rollout_tp),
        actor_pp_layers=[list(range(7))] if pp == 1 else [[0, 1, 2, 3], [4, 5, 6]],
        update_limit=3,
        target_load=dict(prompts=8, responses_per_prompt=16, prompt_cap=1024, response_cap=7168),
        acceptance="Full fixed-weight bit parity, then 3 effective GRPO updates with post-v3 validation; startup is insufficient",
    )


def profiles():
    return [
        profile("M1", 1, 1, 1, 1, 2),
        profile("M2", 1, 2, 1, 2, 4),
        profile("M3", 1, 4, 1, 4, 8),
        profile("M4", 1, 8, 1, 8, 1),
        profile("M5", 2, 4, 4, 1, 8),
    ]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare(work):
    (work / "profiles").mkdir(parents=True, exist_ok=True)
    for p in profiles():
        write(work / "profiles" / (p["id"] + ".json"), p)
    # Planned mappings are explicitly not evidence of live ranks.
    planned = []
    for p in profiles():
        for rank in range(p["actor"]["gpus"]):
            pp, tp = divmod(rank, p["actor"]["tp"])
            ep = tp if p["actor"]["ep"] > 1 else 0
            experts = p["actor"]["ep"]
            planned.append(
                dict(
                    group=p["id"],
                    role="actor",
                    kind="PLANNED_NOT_OBSERVED",
                    physical_gpu=rank,
                    world_rank=rank,
                    pp_rank=pp,
                    tp_rank=tp,
                    ep_rank=ep,
                    etp_rank=tp if experts == 1 else 0,
                    layers=p["actor_pp_layers"][pp],
                    experts=list(range(ep * (256 // experts), (ep + 1) * (256 // experts))),
                )
            )
        for rank in range(p["rollout"]["gpus"]):
            planned.append(
                dict(
                    group=p["id"],
                    role="rollout",
                    kind="PLANNED_NOT_OBSERVED",
                    physical_gpu=rank,
                    world_rank=rank,
                    pp_rank=0,
                    tp_rank=rank,
                    ep_rank=0,
                    layers=list(range(7)),
                )
            )
    write(work / "evidence" / "planned-rank-map.json", planned)
    samples = []
    lengths = [3, 4, 5, 127, 128, 129, 511, 512, 513, 2047, 2048, 2049, 2177]
    for length in lengths:
        prompt = [10 + i % 1000 for i in range(length)]
        samples.append(
            dict(
                id=f"boundary-{length}",
                batch_id=f"single-{length}",
                prompt_ids=prompt,
                response_ids=[42, 43, 44],
                position_ids=list(range(length + 3)),
                mask="full-causal-unpadded-all-response-tokens",
            )
        )
    # Separate observations make AB, BA, and repetitions auditable without duplicate-rank counting.
    for batch, order in [("AB", [127, 2177]), ("BA", [2177, 127]), ("AB-repeat", [127, 2177])]:
        for index, length in enumerate(order):
            samples.append(
                dict(
                    id=f"{batch}-{index}",
                    batch_id=batch,
                    prompt_ids=[10 + i % 1000 for i in range(length)],
                    response_ids=[42, 43, 44],
                    position_ids=list(range(length + 3)),
                    mask="full-causal-unpadded-all-response-tokens",
                )
            )
    write(
        work / "evidence" / "fixed-input-suite.json",
        dict(scope="deterministic fixed-token replay inputs; not generated GRPO data", samples=samples),
    )
    cfg = json.loads((work / "models/DeepSeek-V4-Flash-7L/config.json").read_text())
    checks = []
    for p in profiles():
        tp, etp, ep = (p["actor"][k] for k in ("tp", "etp", "ep"))
        dimensions = dict(
            hidden=cfg["hidden_size"],
            attention_heads=cfg["num_attention_heads"],
            index_heads=cfg["index_n_heads"],
            output_groups=cfg["o_groups"],
            vocab=cfg["vocab_size"],
        )
        if (
            any(v % tp for v in dimensions.values())
            or cfg["n_routed_experts"] % ep
            or cfg["moe_intermediate_size"] % etp
        ):
            raise ValueError("nondivisible model geometry for " + p["id"])
        checks.append(
            dict(
                group=p["id"],
                ordinary_dp=1,
                expert_dp=1,
                dimensions=dimensions,
                divisibility="CPU_CHECKED",
                quantized_cut_alignment="NOT_GPU_VALIDATED",
                sequence_parallel=p["actor"]["sequence_parallel"],
                status="BLOCKED",
                blockers=[
                    "GPU0 root-owned process cannot be signalled by this account",
                    "Real strict TP/EP operator and weight-resharding support is incomplete",
                    "GRPO phase reuse with continuous optimizer state is not implemented",
                ],
            )
        )
    write(work / "evidence" / "topology-preflight.json", checks)


def diagnose(work, image):
    # This command is reached by the durable watcher after two idle checks. It repeats preflight.
    idle(list(range(8)))
    if (
        subprocess.check_output(["docker", "image", "inspect", "--format={{.Id}}", image], text=True).strip()
        != image
    ):
        raise ValueError("use an immutable image ID, not a mutable tag")
    rows = allocation(list(range(8)))
    run = (
        work
        / "gpu-diagnostics"
        / (datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    )
    run.mkdir(parents=True)
    write(run / "allocation.json", rows)
    tasks = profiles()
    diagnostic_tasks = [
        profile("D-TP4-control", 1, 4, 1, 4, 4),
        profile("D-TP4-PP2", 2, 4, 1, 4, 4),
        profile("D-TP4-EP4", 1, 4, 4, 1, 4),
    ]
    results = []
    m5_failed = False
    for p in tasks + diagnostic_tasks:
        diagnostic = p["id"].startswith("D-")
        if diagnostic and not m5_failed:
            continue
        write(work / "profiles" / (p["id"] + ".json"), p)
        for role in ["actor"] if diagnostic else ["actor", "rollout"]:
            # Never launch into an external task or reuse devices left by a failed child.
            idle(list(range(8)))
            out = run / p["id"] / role
            out.mkdir(parents=True)
            active = rows[: p[role]["gpus"]]
            name = "dsv4-7l-" + uuid4().hex[:12]
            command = [
                "docker",
                "run",
                "--rm",
                "--init",
                "--name",
                name,
                "--label",
                "rl-align.seven-layer=" + run.name,
                "--network=host",
                "--ipc=private",
                "--shm-size=16g",
                "--device=/dev/kfd",
                "--mount",
                "type=bind,source=/sys,target=/sys,readonly",
                "--mount",
                f"type=bind,source={work},target=/experiment,readonly",
                "--mount",
                f"type=bind,source={out},target=/outputs",
                "-e",
                "LD_PRELOAD=/opt/dsv4/lib/rlk-isolated-kfd.so",
                "-e",
                "RLK_ALLOWED_KFD_GPU_IDS=" + ",".join(str(r["kfd_gpu_id"]) for r in active),
                "-e",
                "HIP_VISIBLE_DEVICES=" + ",".join(map(str, range(len(active)))),
                "-e",
                "CUDA_VISIBLE_DEVICES=" + ",".join(map(str, range(len(active)))),
                "-e",
                "VLLM_ROCM_USE_AITER=1",
                "-e",
                "OMP_NUM_THREADS=4",
                "-e",
                "HF_HOME=/tmp/huggingface",
                "-e",
                "TRITON_CACHE_DIR=/tmp/triton",
                "-e",
                "TORCH_EXTENSIONS_DIR=/tmp/torch_extensions",
                "-e",
                "VLLM_ENABLE_V1_MULTIPROCESSING=1",
            ]
            for r in active:
                command += ["--device=" + r["render_node"]]
            import os

            for gid in sorted(
                {os.stat("/dev/kfd").st_gid, *[os.stat(r["render_node"]).st_gid for r in active]}
            ):
                command += ["--group-add", str(gid)]
            command += ["--entrypoint", "/opt/dsv4-venv/bin/python", image]
            if role == "actor":
                command += [
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc_per_node=" + str(p[role]["gpus"]),
                ]
            command += [
                "-m",
                "verl.experimental.rl_kernel.probe_topology",
                "--profile",
                f"/experiment/profiles/{p['id']}.json",
                "--role",
                role,
                "--model",
                "/experiment/models/DeepSeek-V4-Flash-7L",
                "--output",
                "/outputs",
            ]
            write(out / "command.json", command)
            began = time.monotonic()
            with (out / "driver.log").open("w") as log:
                try:
                    code = subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT, timeout=1800
                    ).returncode
                except subprocess.TimeoutExpired:
                    # Stop only the exact container created by this invocation, after checking its run label.
                    label = subprocess.check_output(
                        [
                            "docker",
                            "inspect",
                            '--format={{index .Config.Labels "rl-align.seven-layer"}}',
                            name,
                        ],
                        text=True,
                    ).strip()
                    if label == run.name:
                        subprocess.run(["docker", "stop", "--time", "30", name], check=True)
                    code = 124
            receipt = dict(
                group=p["id"],
                diagnostic=diagnostic,
                role=role,
                returncode=code,
                seconds=time.monotonic() - began,
                requested=p[role],
                scope="GPU startup only; not fixed-weight parity or GRPO",
                status="STARTUP_ONLY" if code == 0 else "FAIL",
                optimizer_updates=0,
            )
            write(out / "result.json", receipt)
            results.append(receipt)
            if p["id"] == "M5" and code:
                m5_failed = True
            write(run / "results.json", results)
    write(
        run / "acceptance.json",
        dict(
            status="BLOCKED",
            fixed_weight_validation="NOT_RUN",
            grpo_updates=0,
            reason="Startup diagnostics do not implement complete strict TP/EP computation or phased GRPO; no acceptance is inferred",
            results=results,
        ),
    )
    return run


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["prepare", "diagnose"])
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--image")
    args = ap.parse_args()
    if args.action == "prepare":
        prepare(args.work.resolve())
    else:
        if not args.image:
            ap.error("--image must be the immutable derived image ID")
        print(diagnose(args.work.resolve(), args.image))


if __name__ == "__main__":
    main()
