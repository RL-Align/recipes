# SPDX-License-Identifier: MIT
import copy
import types

import pytest

from recipe_common import BOOK, config, launch, plan, validate
from scripts import node
from scripts.placement import check_single_node_placement


def four():
    return config(BOOK / "configs/single-node-4gpu.json")


def evidence():
    c = four()
    rows = [
        dict(
            physical_gpu=i + 4,
            container_gpu=i,
            role="actor" if i < 2 else "rollout",
            pci_bus_id=str(i),
            render_node=f"/dev/dri/renderD{161 + 8 * i}",
            kfd_gpu_id=100 + i,
        )
        for i in range(4)
    ]
    actors = [
        dict(rank=i, source_identity=dict(node_id="node"), device_identity=dict(pci_bus_id=str(i)))
        for i in range(2)
    ]
    stages = [
        dict(replica_rank=0, pipeline_stage=i, node_id="node", device_identity=dict(pci_bus_id=str(i + 2)))
        for i in range(2)
    ]
    observed = dict(actor_world_size=2, actor_dp_size=1, rollout_replicas=1)
    return c, rows, actors, stages, observed


@pytest.mark.parametrize("gpus", [[4, 5, 6], [4, 5, 6, 6], [5, 4, 6, 7], [4, 5, 6, 8], [True, 5, 6, 7]])
def test_four_requires_explicit_unique_physical_devices(gpus):
    c = four()
    c["nodes"][0]["gpus"] = gpus
    with pytest.raises(ValueError):
        validate(c)


def test_four_math_keeps_target_load_and_native_two_gpu_roles():
    c = four()
    c.update(profile="math-smoke", dataset="dapo")
    p = plan(c)
    assert (p["actor_gpus"], p["actor_dp"], p["rollout_replicas"], p["total_gpus"]) == (2, 1, 1, 4)
    assert (p["steps"], p["prompts"], p["samples"], p["max_prompt_length"], p["max_response_length"]) == (
        3,
        8,
        16,
        1024,
        7168,
    )
    command, _ = launch(c, "/output", ["a", "b", "c"])
    assert "trainer.n_gpus_per_node=2" in command
    assert "actor_rollout_ref.rollout.n_gpus_per_node=2" in command
    assert "reward.custom_reward_function.path=null" in command


def test_four_isolates_selected_kfd_drm_and_ray(monkeypatch, tmp_path):
    c, rows, *_ = evidence()
    c["work_dir"] = str(tmp_path)
    calls, idle_calls = [], []
    monkeypatch.setattr(node, "idle", lambda g: idle_calls.append(list(g)))
    monkeypatch.setattr(node, "allocation", lambda g: copy.deepcopy(rows))
    monkeypatch.setattr(node, "os", types.SimpleNamespace(stat=lambda p: types.SimpleNamespace(st_gid=992)))
    monkeypatch.setattr(node, "run", lambda cmd, **kw: calls.append(cmd) or "container")
    payload = dict(
        config=c,
        node=c["nodes"][0],
        output=str(tmp_path),
        container="test",
        run_id="test",
        image="test-image",
    )
    receipt = node.start(payload)
    cmd = receipt["command"]
    assert idle_calls == [[4, 5, 6, 7]]
    assert {x for x in cmd if x.startswith("--device=")} == {
        "--device=/dev/kfd",
        *["--device=" + r["render_node"] for r in rows],
    }
    assert "HIP_VISIBLE_DEVICES=0,1,2,3" in cmd
    assert "RLK_ALLOWED_KFD_GPU_IDS=100,101,102,103" in cmd
    assert [r["role"] for r in receipt["allocation"]] == ["actor", "actor", "rollout", "rollout"]
    monkeypatch.setattr(node, "owned", lambda p: None)
    node.ray_start(payload)
    assert "--num-gpus=4" in calls[-1] and "--num-gpus=8" not in calls[-1]


def test_four_actual_placement_matches_declared_physical_roles():
    assert check_single_node_placement(*evidence())["status"] == "PASS"


@pytest.mark.parametrize(
    "fault",
    ["swapped_roles", "missing_actor", "duplicate_stage", "overlap", "eight_gpu_native", "wrong_physical"],
)
def test_four_verifier_rejects_wrong_runtime_topology(fault):
    c, rows, actors, stages, observed = evidence()
    if fault == "swapped_roles":
        actors[0]["device_identity"]["pci_bus_id"], stages[0]["device_identity"]["pci_bus_id"] = "2", "0"
    elif fault == "missing_actor":
        actors.pop()
    elif fault == "duplicate_stage":
        stages[1]["pipeline_stage"] = 0
    elif fault == "overlap":
        stages[0]["device_identity"]["pci_bus_id"] = "0"
    elif fault == "eight_gpu_native":
        observed.update(actor_world_size=4, actor_dp_size=2, rollout_replicas=2)
    else:
        rows[0]["physical_gpu"] = 0
    with pytest.raises(ValueError):
        check_single_node_placement(c, rows, actors, stages, observed)
