# SPDX-License-Identifier: MIT
"""Recoverable selected-physical-GPU gate. A local flock is not a scheduler reservation."""

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


def selected_occupants(mapping, selected):
    occupants = {}
    owner = None
    for line in mapping.splitlines():
        match = re.match(r"PID (\d+) is using (\d+) DRM", line)
        if match:
            owner = (int(match[1]), int(match[2]))
            if owner[1] == 0:
                occupants[owner[0]] = []
        elif owner and re.fullmatch(r"[\d\s]+", line) and line.strip():
            devices = list(map(int, line.split()))
            if len(devices) != owner[1] or len(set(devices)) != len(devices):
                raise RuntimeError("incomplete or duplicated GPU process mapping")
            occupants[owner[0]] = devices
            owner = None
    expected = {int(x) for x in re.findall(r"PID (\d+) is using", mapping)}
    if set(occupants) != expected:
        raise RuntimeError("GPU process device mapping missing; fail closed")
    return sorted(pid for pid, devices in occupants.items() if set(devices) & set(selected))


def _alive(pid):
    return os.path.exists(f"/proc/{pid}")


def snapshot(selected=None):
    selected = list(range(8)) if selected is None else list(selected)
    if not selected or len(set(selected)) != len(selected) or any(g not in range(8) for g in selected):
        raise ValueError("selected physical GPUs must be nonempty unique ordinals in 0..7")
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
    # showpids includes KFD contexts with GPU(s)=0. showpidgpus may omit
    # those contexts; compare only actual device-associated processes.
    pids = selected_occupants(mapping, list(range(8)))
    summary_pids = []
    transient_exited = set()
    for key, value in cards.get("system", {}).items():
        if not re.fullmatch(r"PID\d+", key):
            continue
        fields = [part.strip() for part in value.split(",")]
        if len(fields) != 5:
            raise RuntimeError("ROCm process summary lacks GPU activity fields")
        if not fields[1].isdigit():
            # A just-started process reports unknown fields; if it has already exited it cannot occupy a GPU.
            if not _alive(int(key[3:])):
                transient_exited.add(int(key[3:]))
                continue
            raise RuntimeError(f"ROCm process GPU count temporarily unavailable: {key}={value!r}")
        if int(fields[1]) > 0:
            summary_pids.append(int(key[3:]))
    # Drop one-sided PIDs that have already exited; a live one-sided PID still fails closed below.
    for pid in set(pids) ^ set(summary_pids):
        if not _alive(pid):
            transient_exited.add(pid)
    pids = [p for p in pids if p not in transient_exited]
    summary_pids = [p for p in summary_pids if p not in transient_exited]
    if sorted(pids) != sorted(summary_pids):
        raise RuntimeError(
            f"ROCm active process APIs disagree: device_map={pids}, process_summary={summary_pids}"
        )
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
    occupied = selected_occupants(mapping, selected)
    idle = not occupied and all(
        int(cards[f"card{i}"]["VRAM Total Used Memory (B)"]) < 1024**3
        and int(cards[f"card{i}"]["GPU use (%)"]) == 0
        for i in selected
    )
    return dict(
        time=now(),
        idle=idle,
        selected_physical_gpus=selected,
        selected_occupant_pids=occupied,
        cards=cards,
        pid_gpu_mapping=mapping,
        processes=processes,
        transient_exited_pids=sorted(transient_exited),
    )


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
    ap.add_argument("--gpus", nargs="+", type=int, default=list(range(8)))
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
                sample = snapshot(args.gpus)
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
                    for check in (snapshot(args.gpus), snapshot(args.gpus)):
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
