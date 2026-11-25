# SPDX-License-Identifier: MIT
"""Resource gate controls only; these tests never grant GPU acceptance."""

import importlib.util
import json
from pathlib import Path

import pytest


def load(name):
    path = Path(__file__).resolve().parents[1] / "scripts/seven_layer" / (name + ".py")
    spec = importlib.util.spec_from_file_location("seven_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cards(occupied=False, resident=300000000):
    result = {f"card{i}": {"VRAM Total Used Memory (B)": str(resident), "GPU use (%)": "0"} for i in range(8)}
    result["system"] = {"PID123": "python"} if occupied else {}
    return result


def test_zero_utilization_with_a_gpu_process_is_never_idle(monkeypatch):
    watch = load("watch")

    def run(argv):
        if "--json" in argv:
            return json.dumps(cards(occupied=True))
        if "--showpidgpus" in argv:
            return "PID 123 is using 1 DRM device(s):\n0 \n"
        return "123 1 root startup python workload.py"

    monkeypatch.setattr(watch, "run", run)
    assert watch.snapshot()["idle"] is False


def test_resident_memory_and_disagreeing_process_apis_fail_closed(monkeypatch):
    watch = load("watch")
    monkeypatch.setattr(
        watch, "run", lambda argv: json.dumps(cards(resident=2 * 1024**3)) if "--json" in argv else ""
    )
    assert watch.snapshot()["idle"] is False
    monkeypatch.setattr(
        watch, "run", lambda argv: json.dumps(cards(occupied=True)) if "--json" in argv else ""
    )
    with pytest.raises(RuntimeError, match="disagree"):
        watch.snapshot()


def test_no_duplicate_dp_replica_or_added_actor_rollout_card_counts():
    matrix = load("matrix")
    targets = matrix.profiles()
    assert [p["actor"]["gpus"] for p in targets] == [2, 2, 2, 2, 8]
    assert [p["rollout"]["gpus"] for p in targets] == [1, 2, 4, 8, 8]
    assert targets[-1]["peak_physical_gpus"] == 8
    assert all(
        p["actor"]["dp"] == p["actor"]["expert_dp"] == 1 and p["rollout"]["instances"] == 1
        for p in targets[:4]
    )
    assert targets[-1]["actor_pp_layers"] == [[0, 1, 2, 3], [4, 5, 6]]

    assert targets[-1]["actor"]["dp"] == 4
    assert targets[-1]["actor"]["expert_dp"] == 1
    assert "execution_gate" in targets[-1]
    assert all(p["actor"]["tp"] == p["actor"]["etp"] == 1 for p in targets)
    with pytest.raises(ValueError, match="must remain one"):
        matrix.profile("unsupported", 2, 2, 1, 1, 2)


def test_selected_pool_ignores_only_unselected_tasks_and_rejects_selected_occupancy(monkeypatch):
    watch = load("watch")
    state = cards(occupied=True)
    state["card0"]["VRAM Total Used Memory (B)"] = str(90 * 1024**3)
    state["card0"]["GPU use (%)"] = "90"
    mapping = "PID 123 is using 1 DRM device(s):\n0\n"

    def run(argv):
        if "--json" in argv:
            return json.dumps(state)
        if "--showpidgpus" in argv:
            return mapping
        return "123 1 root startup python workload.py"

    monkeypatch.setattr(watch, "run", run)
    assert watch.snapshot([4, 5, 6, 7])["idle"] is True
    assert watch.snapshot()["idle"] is False
    mapping = "PID 123 is using 2 DRM device(s):\n0 5\n"
    assert watch.snapshot([4, 5, 6, 7])["idle"] is False
    mapping = "PID 123 is using 1 DRM device(s):\n0\n"
    state["card6"]["VRAM Total Used Memory (B)"] = str(2 * 1024**3)
    assert watch.snapshot([4, 5, 6, 7])["idle"] is False
    state["card6"]["VRAM Total Used Memory (B)"] = "300000000"
    state["card7"]["GPU use (%)"] = "1"
    assert watch.snapshot([4, 5, 6, 7])["idle"] is False
    with pytest.raises(RuntimeError, match="mapping"):
        watch.selected_occupants("PID 123 is using 2 DRM device(s):\n0\n", [4, 5, 6, 7])
    with pytest.raises(ValueError, match="unique"):
        watch.snapshot([4, 4])
