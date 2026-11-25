# SPDX-License-Identifier: MIT
"""Narrow recorded shared-context exception; never disable selected-device checks."""

import hashlib
from pathlib import Path


class ResourceConflict(RuntimeError):
    """Stop this experiment when external activity conflicts with its devices."""


def identity(pid):
    proc = Path(f"/proc/{pid}")
    command = (proc / "cmdline").read_bytes()
    cgroup = (proc / "cgroup").read_text()
    fields = (proc / "stat").read_text().rsplit(") ", 1)[1].split()
    status = (proc / "status").read_text()
    uid = int(next(line.split()[1] for line in status.splitlines() if line.startswith("Uid:")))
    return dict(
        pid=pid,
        uid=uid,
        start_ticks=int(fields[19]),
        cmdline_sha256=hashlib.sha256(command).hexdigest(),
        cgroup_sha256=hashlib.sha256(cgroup.encode()).hexdigest(),
        cgroup=cgroup,
        command=command.replace(b"\0", b" ").decode(),
    )


def mapping_rows(mapping):
    import re

    result, owner = {}, None
    for line in mapping.splitlines():
        match = re.match(r"PID (\d+) is using (\d+) DRM", line)
        if match:
            owner = (int(match[1]), int(match[2]))
            if owner[1] == 0:
                result[owner[0]] = []
        elif owner and line.strip() and re.fullmatch(r"[\d\s]+", line):
            values = list(map(int, line.split()))
            if len(values) != owner[1]:
                raise ResourceConflict("incomplete physical GPU mapping")
            result[owner[0]] = values
            owner = None
    if set(result) != {int(p) for p in re.findall(r"PID (\d+) is using", mapping)}:
        raise ResourceConflict("missing physical GPU process mapping")
    return result


def evaluate(sample, policy=None, *, own_container=None, identity_loader=identity):
    selected = sample["selected_physical_gpus"]
    rows = mapping_rows(sample["pid_gpu_mapping"])
    exceptions = {r["identity"]["pid"]: r for r in (policy or {}).get("shared_contexts", [])}
    accepted = []
    for pid, devices in rows.items():
        if not set(devices) & set(selected):
            continue
        actual = identity_loader(pid)
        if own_container and own_container in actual["cgroup"]:
            continue
        allowed = exceptions.get(pid)
        if not allowed:
            raise ResourceConflict(f"unapproved external PID {pid} on selected physical GPUs {devices}")
        expected = allowed["identity"]
        if any(
            actual[k] != expected[k] for k in ["pid", "uid", "start_ticks", "cmdline_sha256", "cgroup_sha256"]
        ):
            raise ResourceConflict("shared context identity changed")
        if devices != allowed["physical_gpus"]:
            raise ResourceConflict("shared context moved to another GPU or expanded its task")
        fields = [v.strip() for v in sample["cards"]["system"][f"PID{pid}"].split(",")]
        if len(fields) != 5 or int(fields[1]) != len(devices):
            raise ResourceConflict("missing per-process VRAM/SDMA/CU activity evidence")
        memory, sdma, cu = map(int, fields[2:])
        if memory > allowed["max_process_vram_bytes"] or sdma != allowed["baseline_sdma_used"] or cu != 0:
            raise ResourceConflict("shared context has new memory/SDMA/compute activity")
        accepted.append(
            dict(pid=pid, physical_gpus=devices, process_vram=memory, sdma_used=sdma, cu_occupancy=cu)
        )
    budget = (policy or {}).get("budget", {})
    for gpu in selected:
        card = sample["cards"][f"card{gpu}"]
        used = int(card["VRAM Total Used Memory (B)"])
        total = int(card["VRAM Total Memory (B)"])
        if own_container:
            if total - used < budget.get("minimum_free_bytes_during_run", 4 * 1024**3):
                raise ResourceConflict(f"physical GPU{gpu} is below the recorded memory safety reserve")
        else:
            cap = (policy or {}).get("idle_vram_limit_bytes", 1024**3)
            if used >= cap or int(card["GPU use (%)"]) != 0:
                raise ResourceConflict(f"physical GPU{gpu} has active compute or exceeds idle memory cap")
            if total - used < budget.get("planned_peak_bytes_per_gpu", 0) + budget.get("reserve_bytes", 0):
                raise ResourceConflict(f"physical GPU{gpu} cannot meet the declared peak memory budget")
    return dict(
        status="AVAILABLE_WITH_RECORDED_EXCEPTION" if accepted else "AVAILABLE",
        shared_contexts=accepted,
        selected_physical_gpus=selected,
        own_container=own_container,
        machine_is_exclusive=False,
    )
