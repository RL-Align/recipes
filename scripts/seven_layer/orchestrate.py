# SPDX-License-Identifier: MIT
"""One persistent entry: exact-image health, eight-rank communication, N1..N4."""

import argparse
import fcntl
import json
from pathlib import Path

from execute import run
from qualify import qualify


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--runtime-lock", type=Path, required=True)
    ap.add_argument("--resource-policy", type=Path, required=True)
    ap.add_argument("--q-up-trace-run", type=Path)
    ap.add_argument("--indexer-trace-run", type=Path)
    ap.add_argument("--gpus", nargs="+", type=int, default=list(range(8)))
    ap.add_argument("--groups", nargs="+", choices=["N1", "N2", "N3", "N4", "N5"], default=["N1", "N2", "N3", "N4"])
    ap.add_argument("--target-only", action="store_true")
    ap.add_argument("--correctness-run", type=Path)
    ap.add_argument("--resume-target", type=Path)
    args = ap.parse_args()
    if args.target_only and not args.correctness_run:
        ap.error("--target-only requires --correctness-run")
    if (args.correctness_run or args.resume_target) and not args.target_only:
        ap.error("--correctness-run/--resume-target require --target-only")
    work = args.work.resolve()
    with (work / "orchestrate-N.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        health = qualify(
            work,
            args.runtime_lock.resolve(),
            args.resource_policy.resolve(),
            q_up_trace=args.q_up_trace_run.resolve() if args.q_up_trace_run else None,
            indexer_trace=args.indexer_trace_run.resolve() if args.indexer_trace_run else None,
            gpus=args.gpus if sorted(args.gpus) != list(range(8)) else None,
        )
        info = json.loads(args.runtime_lock.read_text())
        run(
            work,
            info["image_id"],
            args.gpus,
            args.groups,
            args.runtime_lock.resolve(),
            args.resource_policy.resolve(),
            health,
            target_only=args.target_only,
            correctness_run=args.correctness_run,
            resume_target=args.resume_target,
        )


if __name__ == "__main__":
    main()
