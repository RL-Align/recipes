# SPDX-License-Identifier: MIT
"""Actual HIP health/RCCL communication checks; not DSV4 model acceptance."""

import argparse
import datetime
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from rl_engine.integrations.common.device_identity import hip_device_identity


def check(actual, expected, name):
    if not bool(torch.isfinite(actual).all()) or not torch.equal(actual, expected):
        raise RuntimeError(f"{name} produced wrong/nonfinite communication or compute output")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["single", "communication"])
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    args.output.mkdir(parents=True, exist_ok=True)
    record = dict(rank=rank, mode=args.mode, identity=hip_device_identity(), checks=[])
    try:
        if args.mode == "single":
            if torch.cuda.device_count() != 1:
                raise RuntimeError("single-card isolation exposed another GPU")
            a = torch.ones((1024, 1024), device="cuda", dtype=torch.bfloat16)
            for _ in range(8):
                check(a @ a, torch.full_like(a, 1024), "single-card BF16 matmul")
            record["checks"].append("actual BF16 matmul finite/exact x8")
            record["world_size"] = 1
        else:
            dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=60))
            world = dist.get_world_size()
            if world != 8 or torch.cuda.device_count() != 8:
                raise RuntimeError("actual eight-rank/eight-device communication required")
            for iteration in range(3):
                original = torch.arange(1024, device="cuda", dtype=torch.float32)
                value = original + rank
                dist.all_reduce(value)
                check(value, original * world + sum(range(world)), "RCCL all_reduce")
                value = torch.full((16,), rank, dtype=torch.int64, device="cuda")
                dist.broadcast(value, src=3)
                check(value, torch.full_like(value, 3), "RCCL broadcast")
                value = torch.full((world * 16,), rank + 1, dtype=torch.float32, device="cuda")
                reduced = torch.empty(16, dtype=torch.float32, device="cuda")
                dist.reduce_scatter_tensor(reduced, value)
                check(reduced, torch.full_like(reduced, 36), "RCCL reduce_scatter")
                sent = torch.arange(world, dtype=torch.int64, device="cuda") + rank * world
                received = torch.empty_like(sent)
                dist.all_to_all_single(received, sent)
                check(received, torch.arange(world, device="cuda") * world + rank, "RCCL all_to_all")
                dist.barrier()
                record["checks"].append(
                    dict(
                        iteration=iteration,
                        all_reduce="PASS",
                        broadcast="PASS",
                        reduce_scatter="PASS",
                        all_to_all="PASS",
                    )
                )
            record["world_size"] = world
            dist.destroy_process_group()
        torch.cuda.synchronize()
        record["status"] = "PASS"
    except BaseException as error:
        record.update(status="FAIL", error=str(error))
        raise
    finally:
        (args.output / f"rank-{rank}.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
