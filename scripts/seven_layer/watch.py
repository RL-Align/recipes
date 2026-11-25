# SPDX-License-Identifier: MIT
"""Recoverable eight-GPU gate. A local flock is not a scheduler reservation."""

import argparse
import datetime
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def run(args):
    p = subprocess.run(args, text=True, capture_output=True, timeout=30)
    if p.returncode:
        raise RuntimeError(f"{args!r}: exit={p.returncode}: {p.stderr}")
    return p.stdout


def snapshot():
    raw = run(
        [
            "rocm-smi",
            "--showproductname",
            "--showbus",
            "--showmeminfo",
            "vram",
            "--showuse",
            "--showpids",
            "--json",
        ]
    )
    cards = json.loads(raw)
    mapping = run(["rocm-smi", "--showpidgpus"])
    pids = [int(x) for x in re.findall(r"PID (\d+) is using", mapping)]
    # Both APIs must agree; missing visibility is never idle.
    summary_pids = sorted(int(k[3:]) for k in cards.get("system", {}) if re.fullmatch(r"PID\d+", k))
    if sorted(pids) != summary_pids:
        raise RuntimeError("ROCm process APIs disagree; retry on next poll")
    processes = []
    for pid in pids:
        info = run(["ps", "-p", str(pid), "-o", "pid,ppid,user,lstart,args"])
        try:
            cgroup = Path(f"/proc/{pid}/cgroup").read_text()
        except OSError as e:
            cgroup = str(e)
        processes.append(dict(pid=pid, ps=info, cgroup=cgroup))
    if sorted(k for k in cards if re.fullmatch(r"card\d+", k)) != [f"card{i}" for i in range(8)]:
        raise RuntimeError("exactly physical GPUs 0-7 must be visible")
    idle = not pids and all(
        int(cards[f"card{i}"]["VRAM Total Used Memory (B)"]) < 1024**3
        and int(cards[f"card{i}"]["GPU use (%)"]) == 0
        for i in range(8)
    )
    return dict(time=now(), idle=idle, cards=cards, pid_gpu_mapping=mapping, processes=processes)


def atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--ready",
        type=Path,
        required=True,
        help="JSON with argv and immutable prerequisite files and SHA256s",
    )
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (args.output / "monitor.pid").write_text(str(os.getpid()) + "\n")
        if (args.output / "launched.json").exists():
            raise SystemExit("already launched; inspect existing run rather than duplicate")
        while True:
            state = dict(time=now(), pid=os.getpid(), status="WAITING", local_lock_is_reservation=False)
            try:
                sample = snapshot()
                with (args.output / "gpu-samples.jsonl").open("a") as f:
                    f.write(json.dumps(sample) + "\n")
                state["gpu_idle"] = sample["idle"]
                usage = shutil.disk_usage(args.output)
                state["disk"] = dict(free=usage.free, used=usage.used, total=usage.total)
                state["ready_file_exists"] = args.ready.is_file()
                if sample["idle"] and args.ready.is_file():
                    ready_bytes = args.ready.read_bytes()
                    ready = json.loads(ready_bytes)
                    if not ready.get("argv") or not ready.get("prerequisites"):
                        raise ValueError("ready file needs an actual command and immutable prerequisites")
                    for item in ready["prerequisites"]:
                        with Path(item["path"]).open("rb") as f:
                            if hashlib.file_digest(f, "sha256").hexdigest() != item["sha256"]:
                                raise ValueError("prerequisite digest mismatch: " + item["path"])
                    # Two checks separated by seconds, then a last check at launch.
                    time.sleep(5)
                    for check in (snapshot(), snapshot()):
                        with (args.output / "gpu-samples.jsonl").open("a") as f:
                            f.write(json.dumps(check) + "\n")
                        if not check["idle"]:
                            raise RuntimeError("conflict during final resource checks")
                    if args.ready.read_bytes() != ready_bytes:
                        raise RuntimeError("ready file changed during preflight")
                    state.update(
                        status="LAUNCHED",
                        argv=ready["argv"],
                        ready_sha256=hashlib.sha256(ready_bytes).hexdigest(),
                    )
                    atomic_json(args.output / "launched.json", state)
                    atomic_json(args.output / "status.json", state)
                    with (args.output / "execution.log").open("a") as log:
                        p = subprocess.Popen(
                            ready["argv"], stdout=log, stderr=subprocess.STDOUT, cwd=ready.get("cwd")
                        )
                        state["child_pid"] = p.pid
                        atomic_json(args.output / "status.json", state)
                        state.update(returncode=p.wait(), status="FINISHED", finished=now())
                    atomic_json(args.output / "status.json", state)
                    return
            except Exception as e:
                state["reason"] = f"{type(e).__name__}: {e}"
            atomic_json(args.output / "status.json", state)
            print(json.dumps(state), flush=True)
            if args.once:
                return
            time.sleep(30)


if __name__ == "__main__":
    main()
