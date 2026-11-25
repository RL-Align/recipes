# SPDX-License-Identifier: MIT
import asyncio
import copy
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recipe_common import BOOK, config, launch, plan, validate, write_json
from run import require_gate
from scripts.report import report


def fixture_config(topology="four-node-32gpu"):
    return config(BOOK / "configs" / f"{topology}.json")


@pytest.mark.parametrize(
    "topology,actor,replicas,total", [("single-node-8gpu", 4, 2, 8), ("four-node-32gpu", 16, 8, 32)]
)
def test_effective_workload(topology, actor, replicas, total):
    c = fixture_config(topology)
    p = plan(c)
    assert (p["actor_gpus"], p["rollout_replicas"], p["total_gpus"]) == (actor, replicas, total)
    command, env = launch(c, "/shared/dsv4-recipe/runs/test", ["a", "b", "c"])
    overrides = dict(x.split("=", 1) for x in command[command.index("dsv4_grpo") + 1 :])
    assert json.loads(overrides["trainer.nnodes"]) * json.loads(overrides["trainer.n_gpus_per_node"]) == actor
    assert (
        json.loads(overrides["actor_rollout_ref.rollout.nnodes"])
        * json.loads(overrides["actor_rollout_ref.rollout.n_gpus_per_node"])
        == replicas * 2
    )
    assert not any("HIP_VISIBLE_DEVICES" in k or "VLLM_HOST_IP" in k for k in overrides)
    assert env["VLLM_PP_LAYER_PARTITION"] == "3,1"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c["nodes"][1].update(ip=c["nodes"][0]["ip"]),
        lambda c: c["nodes"][0].update(gpus=[0, 1, 2, 3]),
        lambda c: c.update(work_dir="/"),
        lambda c: c["nodes"][1].update(ssh="-oProxyCommand=bad"),
        lambda c: c.update(profile="math-200", dataset="copy"),
    ],
)
def test_reject_ambiguous_or_unsupported_allocations(mutation):
    c = fixture_config()
    mutation(c)
    with pytest.raises(ValueError):
        validate(c)


def test_math_load_and_reward_are_not_copy_smoke():
    c = fixture_config()
    c.update(profile="math-200", dataset="dapo")
    p = plan(c)
    assert (p["steps"], p["prompts"], p["samples"], p["max_response_length"]) == (200, 8, 16, 7168)
    command, _ = launch(c, "/shared/dsv4-recipe/runs/test", ["a", "b", "c"])
    assert "reward.custom_reward_function.path=null" in command
    assert "trainer.total_training_steps=200" in command


@pytest.mark.parametrize("topology", ["single-node-8gpu", "four-node-32gpu"])
@pytest.mark.parametrize("profile", ["smoke", "math-smoke", "math-200"])
def test_native_hydra_and_trainer_contract(contracts, topology, profile):
    from hydra import compose, initialize_config_dir

    root = Path(os.environ["RLK_RECIPE_TEST_SOURCE"]) / "verl/verl/experimental/rl_kernel"
    c = fixture_config(topology)
    c.update(profile=profile, dataset="copy" if profile == "smoke" else "dapo")
    command, _ = launch(c, "/shared/dsv4-recipe/runs/test", ["a", "b", "c"])
    overrides = command[command.index("dsv4_grpo") + 1 :] + [
        f"hydra.searchpath=[file://{root.parents[1] / 'trainer/config'}]"
    ]
    with initialize_config_dir(config_dir=str(root / "config"), version_base=None):
        result = compose(config_name="dsv4_grpo", overrides=overrides)
    contracts["trainer_contract"].validate_trainer_config(result)
    assert result.trainer.total_training_steps == (200 if profile == "math-200" else 3)


def test_gate_requires_independent_pass_same_image_and_nodes(tmp_path):
    c = fixture_config()
    record = dict(image_id="sha256:a")
    write_json(tmp_path / "site.json", c)
    write_json(tmp_path / "runtime.lock.json", record)
    write_json(tmp_path / "verification.json", dict(status="NOT_PASSED"))
    with pytest.raises(ValueError):
        require_gate(c, record, tmp_path, "smoke")
    write_json(tmp_path / "verification.json", dict(status="PASS"))
    require_gate(c, record, tmp_path, "smoke")
    with pytest.raises(ValueError):
        require_gate(c, dict(image_id="sha256:b"), tmp_path, "smoke")
    c["nodes"][0]["ip"] = "10.0.0.99"
    with pytest.raises(ValueError):
        require_gate(c, record, tmp_path, "smoke")


def test_process_success_does_not_become_acceptance(tmp_path):
    write_json(tmp_path / "plan.json", plan(fixture_config()))
    write_json(tmp_path / "exit-code.json", dict(exit_code=0))
    result = report(tmp_path)
    assert result["status"] == "NOT_PASSED" and result["independently_verified_updates"] == 0


@pytest.fixture(scope="module")
def contracts():
    source = os.environ.get("RLK_RECIPE_TEST_SOURCE")
    if not source:
        pytest.skip(
            "set RLK_RECIPE_TEST_SOURCE after scripts/sources.py to exercise patched framework contracts"
        )
    root = Path(source) / "verl/verl/experimental/rl_kernel"
    name = "_recipe_contract_tests"
    package = types.ModuleType(name)
    package.__path__ = [str(root)]
    sys.modules[name] = package
    modules = {}
    for file in ("cluster_contract", "trainer_contract", "trainer_sync"):
        spec = importlib.util.spec_from_file_location(name + "." + file, root / (file + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules[file] = module
    return modules


def resolved(c):
    command, _ = launch(c, "/shared/dsv4-recipe/runs/test", ["a", "b", "c"])
    result = {}
    for item in command[command.index("dsv4_grpo") + 1 :]:
        key, val = item.split("=", 1)
        current = result
        path = key.lstrip("+").split(".")
        for part in path[:-1]:
            current = current.setdefault(part, {})
        current[path[-1]] = json.loads(val)
    return result


def test_32_gpu_real_topology_and_replica_host_mapping(contracts):
    module = contracts["cluster_contract"]
    p = module.topology(resolved(fixture_config()))
    assert (p["actor_world_size"], p["actor_dp_size"], p["rollout_replicas"]) == (16, 8, 8)
    assert [module.rollout_replica_nodes(p["rollout_node_ips"], 8, i) for i in range(8)] == [
        ["10.0.0.12"]
    ] * 4 + [["10.0.0.13"]] * 4
    c = resolved(fixture_config())
    c["actor_rollout_ref"]["rollout"]["rlk_node_ips"] = ["10.0.0.10", "10.0.0.12"]
    with pytest.raises(ValueError):
        module.topology(c)


def test_32_gpu_placement_rejects_missing_overlap_and_wrong_replica_host(contracts):
    module = contracts["cluster_contract"]
    p = module.topology(resolved(fixture_config()))
    nodes = {str(i): f"10.0.0.{10 + i}" for i in range(4)}
    actor = [dict(rank=i, node_id=str(i // 8), pci_bus_id=f"{i % 8}") for i in range(16)]
    rollout = [
        dict(replica_rank=i // 2, rank=i % 2, node_id=str(2 + i // 8), pci_bus_id=f"{i % 8}")
        for i in range(16)
    ]
    module.verify_placement(p, actor, rollout, nodes)
    with pytest.raises(RuntimeError):
        module.verify_placement(p, actor[:-1], rollout, nodes)
    bad = copy.deepcopy(rollout)
    bad[0]["node_id"] = "3"
    with pytest.raises(RuntimeError):
        module.verify_placement(p, actor, bad, nodes)
    bad = copy.deepcopy(actor)
    bad[1]["pci_bus_id"] = bad[0]["pci_bus_id"]
    with pytest.raises(RuntimeError):
        module.verify_placement(p, bad, rollout, nodes)


def test_all_16_actor_exports_and_raw_bit_equality_required(contracts):
    module = contracts["trainer_contract"]
    exports = [
        dict(rank=i, dp_rank=i // 2, pipeline_stage=i % 2, weight_version=3, manifest={"w": "bytes"})
        for i in range(16)
    ]
    module.validate_actor_exports(exports, 16, 3)
    with pytest.raises(RuntimeError):
        module.validate_actor_exports(exports[:-1], 16, 3)
    import torch

    a = torch.tensor([[-1.0]], dtype=torch.float32)
    b = a.clone()
    b.view(torch.int32)[0, 0] ^= 1
    with pytest.raises(RuntimeError):
        module.strict_response_parity(a, b, torch.ones_like(a))


def test_sync_requires_all_eight_rollout_replicas(contracts, tmp_path):
    sync = contracts["trainer_sync"].RLKPolicySync
    sync(object(), [object() for _ in range(8)], tmp_path, actor_world_size=16, rollout_replicas=8)
    with pytest.raises(ValueError):
        sync(object(), [object() for _ in range(7)], tmp_path, actor_world_size=16, rollout_replicas=8)


@pytest.mark.parametrize("fail_last", [False, True])
def test_eight_replica_sync_barrier_and_failure_keep_admission_closed(contracts, tmp_path, fail_last):
    events = []
    tensors = {name: dict(sha256=name, dtype="torch.float32", shape=[1]) for name in ("a", "b")}
    rows = [
        dict(rank=i, dp_rank=i // 2, pipeline_stage=i % 2, weight_version=0, manifest=tensors)
        for i in range(16)
    ]

    def export(path):
        Path(path).write_bytes(b"unit-test-policy")
        events.append(("export", 0))
        return rows

    class Server:
        def __init__(self, rank):
            self.rank = rank
            self.abort_all_requests = types.SimpleNamespace(remote=self.abort)
            self.rlk_load_policy = types.SimpleNamespace(remote=self.load)
            self.resume_generation = types.SimpleNamespace(remote=self.resume)

        async def abort(self, **kwargs):
            events.append(("abort", self.rank))

        async def load(self, path, version):
            events.append(("load", self.rank))
            if fail_last and self.rank == 7:
                raise RuntimeError("injected replica 7 failure")
            return [
                dict(
                    pipeline_stage=p,
                    pid=100 + 2 * self.rank + p,
                    weight_version=version,
                    parameter_load_completed=True,
                    verified_original_tensors={name: tensors[name]},
                )
                for p, name in enumerate(("a", "b"))
            ]

        async def resume(self):
            events.append(("resume", self.rank))

    sync = contracts["trainer_sync"].RLKPolicySync(
        types.SimpleNamespace(rlk_export_policy=export),
        [Server(i) for i in range(8)],
        tmp_path,
        actor_world_size=16,
        rollout_replicas=8,
    )
    if fail_last:
        with pytest.raises(RuntimeError, match="replica 7"):
            asyncio.run(sync.update(0))
        assert sync.version is None and not any(e[0] == "resume" for e in events)
        assert not (tmp_path / "sync-v0.json").exists()
    else:
        asyncio.run(sync.update(0))
        assert len(sync.replica_pids) == 8
        assert events.index(("export", 0)) > max(events.index(("abort", i)) for i in range(8))
        assert min(events.index(("resume", i)) for i in range(8)) > max(
            events.index(("load", i)) for i in range(8)
        )
        assert len(json.loads((tmp_path / "sync-v0-raw.json").read_text())["actor_exports"]) == 16
