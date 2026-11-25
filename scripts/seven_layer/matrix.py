# SPDX-License-Identifier: MIT
"""Prepare the revised actor-TP1 matrix. Actual execution lives in execute.py."""

import argparse
import json
from pathlib import Path


def profile(name, pp, tp, ep, etp, rollout_tp, dense_dp=1):
    if tp != 1 or etp != 1:
        raise ValueError("DSV4 actor TP and ETP must remain one")
    world = pp * tp * dense_dp
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
            pp=pp,
            tp=tp,
            ep=ep,
            etp=etp,
            dp=dense_dp,
            expert_dp=edp,
            cp=1,
            gpus=world,
            sequence_parallel=False,
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
        profile("N1", 2, 1, 1, 1, 1),
        profile("N2", 2, 1, 1, 1, 2),
        profile("N3", 2, 1, 1, 1, 4),
        profile("N4", 2, 1, 1, 1, 8),
        dict(
            profile("N5", 2, 1, 4, 1, 8, dense_dp=4),
            execution_gate="N1-N4 complete fixed-weight and three-update independent verification first",
            purpose="Expert partition/synchronization extension; not a DP performance test",
        ),
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
            pp, within_stage = divmod(rank, p["actor"]["tp"] * p["actor"]["dp"])
            dp, tp = divmod(within_stage, p["actor"]["tp"])
            ep = dp if p["actor"]["ep"] > 1 else 0
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
    write(work / "evidence" / "planned-rank-map-N.json", planned)
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
                ordinary_dp=p["actor"]["dp"],
                expert_dp=1,
                dimensions=dimensions,
                divisibility="CPU_CHECKED",
                quantized_cut_alignment="NOT_GPU_VALIDATED",
                sequence_parallel=p["actor"]["sequence_parallel"],
                status="BLOCKED",
                blockers=[
                    "GPU0 root-owned process cannot be signalled by this account",
                    "Rollout TP computation/resharding and N5 EP4 support still need qualification",
                    "Native phased GRPO implementation and recovery require actual GPU qualification",
                ],
            )
        )
    write(work / "evidence" / "topology-preflight-N.json", checks)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["prepare"])
    ap.add_argument("--work", type=Path, required=True)
    args = ap.parse_args()
    prepare(args.work.resolve())


if __name__ == "__main__":
    main()
