# SPDX-License-Identifier: MIT
"""Independently check the physical allocation from actual native worker evidence."""

from pathlib import Path

from recipe_common import plan, read_json, write_json


def check_single_node_placement(c, allocation, actors, stages, observed_plan):
    p = plan(c)
    if (
        observed_plan["actor_world_size"] != p["actor_gpus"]
        or observed_plan["actor_dp_size"] != p["actor_dp"]
        or observed_plan["rollout_replicas"] != p["rollout_replicas"]
    ):
        raise ValueError("native runtime topology differs from recipe topology")
    expected = [r for r in p["physical_allocation"] if r["node"] == c["nodes"][0]["name"]]
    if len(allocation) != len(expected):
        raise ValueError("allocation cardinality differs from recipe")
    for actual, declared in zip(allocation, expected, strict=True):
        if any(actual[k] != declared[k] for k in ("physical_gpu", "container_gpu", "role")):
            raise ValueError("physical allocation differs from site.json")
    if {a["rank"] for a in actors} != set(range(p["actor_gpus"])) or len(actors) != p["actor_gpus"]:
        raise ValueError("missing or duplicate actual actor rank")
    if len(stages) != p["rollout_gpus"]:
        raise ValueError("missing actual rollout stage")
    if {(s["replica_rank"], s["pipeline_stage"]) for s in stages} != {
        (r, s) for r in range(p["rollout_replicas"]) for s in (0, 1)
    }:
        raise ValueError("missing or duplicate actual rollout coordinate")
    actor_nodes = {a["source_identity"]["node_id"] for a in actors}
    rollout_nodes = {s["node_id"] for s in stages}
    if len(actor_nodes | rollout_nodes) != 1:
        raise ValueError("single-node workers did not execute on one node")
    used = set()
    observed = {}
    for role, records in (("actor", actors), ("rollout", stages)):
        buses = [r["device_identity"]["pci_bus_id"].lower() for r in records]
        wanted = {r["pci_bus_id"].lower() for r in allocation if r["role"] == role}
        if len(set(buses)) != len(buses) or set(buses) != wanted or used & set(buses):
            raise ValueError(f"{role}: actual worker PCI devices differ from declared allocation")
        used.update(buses)
        observed[role] = buses
    return dict(status="PASS", scope=p["scope"], physical_allocation=allocation, observed_pci=observed)


def verify_runtime_layout(c, output):
    from omegaconf import OmegaConf
    from verl.experimental.rl_kernel.cluster_contract import topology
    from verl.experimental.rl_kernel.trainer_contract import replica_stages

    output = Path(output)
    declared = OmegaConf.load(output / "hydra/.hydra/config.yaml")
    observed = topology(declared)
    p = plan(c)
    # Match the executed workload, rather than trusting only sidecar files.
    if (
        int(declared.trainer.total_training_steps) != p["steps"]
        or int(declared.data.train_batch_size) != p["prompts"]
        or int(declared.actor_rollout_ref.rollout.n) != p["samples"]
        or int(declared.data.max_prompt_length) != p["max_prompt_length"]
        or int(declared.data.max_response_length) != p["max_response_length"]
    ):
        raise ValueError("executed workload differs from the recipe plan")
    if len(c["nodes"]) != 1:
        return dict(scope=c["topology"], native_verifier_checks_placement=True)
    allocation = read_json(output / f"allocation-{c['nodes'][0]['name']}.json")
    actors = [read_json(output / f"policy-sync/actor-rank-{i}-identity.json") for i in range(p["actor_gpus"])]
    initial = read_json(output / "policy-sync/sync-v0-raw.json")
    stages = [
        dict(stage, replica_rank=r)
        for r in range(p["rollout_replicas"])
        for stage in replica_stages(initial, r)
    ]
    result = check_single_node_placement(c, allocation, actors, stages, observed)
    write_json(output / "recipe-placement.json", result)
    return result
