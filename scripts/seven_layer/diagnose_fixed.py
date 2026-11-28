# SPDX-License-Identifier: MIT
"""Diagnostic replay of a failed post-update fixed-weight audit with real GPU boundary traces.

Restores the source run's checkpoint-v<N> and policy-v<N>.pt, re-runs the fixed actor/rollout
scoring with trace capture on matching batches, and compares traces layer boundary by layer
boundary. Scope is diagnosis only; the output is never acceptance evidence.
"""

import argparse
import datetime
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

from execute import allocation, container_base, launch, mapped, verify, write
from matrix import profiles

COMPARER = "/opt/recipes/scripts/seven_layer/compare_traces.py"


def mismatches(verification):
    """id -> (mismatch_count, first_difference bits) for rows that differ."""
    return {
        r["id"]: dict(
            mismatch_count=r["mismatch_count"],
            token_index=(r.get("first_difference") or {}).get("token_index"),
            left_bits=(r.get("first_difference") or {}).get("left_bits"),
            right_bits=(r.get("first_difference") or {}).get("right_bits"),
        )
        for r in verification.get("rows", [])
        if r.get("mismatch_count")
    }


def trace_dirs(actor, rollout, batch):
    def key(p):
        m = re.fullmatch(r"actor-rank-(\d+)-chunk-(\d+)-.*", p.name)
        return int(m[1]), int(m[2])

    actors = sorted((actor / "traces").glob(f"actor-rank-*-chunk-*-{batch}"), key=key)
    rollouts = sorted((rollout / "policy-sync" / "traces").glob(f"rollout-rank-*-{batch}"))
    return actors, rollouts


def compare_batch(work, root, image, group_dir, actors, rollout, batch):
    out = group_dir / "trace-compare"
    out.mkdir(exist_ok=True)
    command = container_base(work, root, [], image, out, gpu=False)
    command += [
        COMPARER,
        "--actor",
        *[mapped(work, d) for d in actors],
        "--rollout",
        mapped(work, rollout),
        "--output",
        f"/outputs/first-difference-{batch}.json",
    ]
    cid = subprocess.check_output(command, text=True).strip()
    with (out / f"compare-{batch}.log").open("w") as log:
        result = subprocess.run(["docker", "start", "-a", cid], stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        return dict(batch=batch, error=f"comparer exited {result.returncode}; see {out}/compare-{batch}.log")
    return dict(batch=batch, **json.loads((out / f"first-difference-{batch}.json").read_text()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--runtime-lock", type=Path, required=True)
    ap.add_argument("--source-run", type=Path, required=True)
    ap.add_argument("--group", required=True)
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument("--dataset", default="copy")
    ap.add_argument("--gpus", nargs="+", type=int, required=True)
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--routed-dump-m", type=int, help="diagnostic: dump rollout routed rows for M-token batches")
    ap.add_argument("--rollout-group", help="diagnostic: rollout profile to score with (default: --group)")
    args = ap.parse_args()
    work, source = args.work.resolve(), args.source_run.resolve()
    runtime = json.loads(args.runtime_lock.resolve().read_text())
    image = runtime["image_id"]
    profile = {p["id"]: p for p in profiles()}[args.group]
    # Optional: score the rollout with another group's profile (e.g. N2 rollout TP2) against
    # the same actor checkpoint; N1-N4 share the actor topology, so only the rollout differs.
    rollout_profile = {p["id"]: p for p in profiles()}[args.rollout_group or args.group]
    if rollout_profile["actor"] != profile["actor"]:
        raise ValueError("rollout group must share the source group's actor topology")
    if len(set(args.gpus)) != len(args.gpus) or any(g not in range(8) for g in args.gpus):
        raise ValueError("select unique physical GPU ordinals in 0..7")
    re.compile(args.pattern)
    src = source / args.group / args.dataset
    checkpoint = src / f"checkpoint-v{args.version}"
    policy = src / f"update-{args.version}" / f"policy-v{args.version}.pt"
    source_verification = src / f"v{args.version}-fixed-actor" / "verification.json"
    for path in (source / "provenance.json", checkpoint, policy, source_verification):
        if not path.exists():
            raise FileNotFoundError(path)
    runs = work / "runs-N"
    runs.mkdir(exist_ok=True)
    # The matrix orchestrator holds execute-N.lock for its whole run; diagnostics use
    # their own lock and disjoint physical GPUs (launch() still waits for them to be idle).
    with (work / "diagnose-N.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        root = runs / f"{stamp}-{uuid4().hex[:8]}-diag-{args.group}-v{args.version}"
        root.mkdir(exist_ok=False)
        # Checkpoint restore requires the identical provenance; copy bytes unchanged.
        shutil.copyfile(source / "provenance.json", root / "provenance.json")
        write(
            root / "diagnostic-receipt.json",
            dict(
                scope="diagnostic trace replay only; never acceptance evidence",
                image_id=image,
                runtime_lock_sources=runtime["sources"],
                source_run=str(source),
                source_provenance_sha256=hashlib.sha256((source / "provenance.json").read_bytes()).hexdigest(),
                group=args.group,
                version=args.version,
                dataset=args.dataset,
                pattern=args.pattern,
                physical_gpus=args.gpus,
                rollout_group=rollout_profile["id"],
                note="provenance.json is copied only as the checkpoint restore key; it does not describe this image",
            ),
        )
        rows = allocation(args.gpus)
        write(root / "allocation.json", rows)
        os.environ["RLK_TRACE_AUDIT_PATTERN"] = args.pattern
        base = root / args.group
        base.mkdir()
        actor = base / "fixed-actor"
        rollout = base / "fixed-rollout"
        launch(
            work, root, rows, image, profile, "actor-audit", actor, version=args.version, resume=checkpoint
        )
        if args.routed_dump_m:
            os.environ["RLK_ROUTED_ROWS_DUMP_DIR"] = "/outputs/routed-rows"
            os.environ["RLK_ROUTED_ROWS_DUMP_M"] = str(args.routed_dump_m)
        launch(
            work, root, rows, image, rollout_profile, "rollout-score", rollout, version=args.version, policy=policy
        )
        verification_error = None
        try:
            verify(work, root, image, rollout_profile, actor, rollout, policy, actor)
        except RuntimeError as e:
            verification_error = str(e)  # FAIL is the expected reproduction
        diag = json.loads((actor / "verification.json").read_text())
        source_diag = json.loads(source_verification.read_text())
        a, b = mismatches(source_diag), mismatches(diag)
        write(
            base / "reproduction.json",
            dict(
                reproduced=bool(b) and a == b,
                source_mismatches=a,
                diagnostic_mismatches=b,
                source_status=source_diag.get("status"),
                diagnostic_status=diag.get("status"),
                diagnostic_error=diag.get("error"),
                verify_exception=verification_error,
            ),
        )
        # Compare traces per matching batch in a CPU container (no torch on the host).
        results = []
        batches = sorted(
            {m[1] for p in (actor / "traces").glob("actor-rank-*-chunk-*-*") if (m := re.fullmatch(r"actor-rank-\d+-chunk-\d+-(.*)", p.name))}
        )
        for batch in batches:
            actors, rollouts = trace_dirs(actor, rollout, batch)
            if not actors or len(rollouts) != 1:
                results.append(
                    dict(batch=batch, error=f"need actor dirs and exactly one rollout dir, got {len(actors)}/{len(rollouts)}")
                )
                continue
            results.append(compare_batch(work, root, image, base, actors, rollouts[0], batch))
        write(
            base / "first-difference.json",
            dict(
                scope="diagnostic trace comparison only; never acceptance evidence",
                pattern=args.pattern,
                batches={r["batch"]: {k: v for k, v in r.items() if k != "records"} for r in results},
                note="per-leaf records are in trace-compare/first-difference-<batch>.json",
            ),
        )
        print(root, flush=True)


if __name__ == "__main__":
    main()
