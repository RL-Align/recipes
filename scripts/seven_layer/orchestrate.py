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
    ap.add_argument("--gpus", nargs="+", type=int, default=list(range(8)))
    ap.add_argument("--groups", nargs="+", choices=["N1", "N2", "N3", "N4"], default=["N1", "N2", "N3", "N4"])
    args = ap.parse_args()
    work = args.work.resolve()
    with (work / "orchestrate-N.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        health = qualify(work, args.runtime_lock.resolve(), args.resource_policy.resolve())
        info = json.loads(args.runtime_lock.read_text())
        run(
            work,
            info["image_id"],
            args.gpus,
            args.groups,
            args.runtime_lock.resolve(),
            args.resource_policy.resolve(),
            health,
        )


if __name__ == "__main__":
    main()
