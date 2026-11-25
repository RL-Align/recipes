# SPDX-License-Identifier: MIT
"""Security/resource exception boundaries, not numerical GPU acceptance."""

import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "resource_policy", Path(__file__).resolve().parents[1] / "scripts/seven_layer/resource_policy.py"
)
policy_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy_module)


def fixture():
    identity = dict(
        pid=123,
        uid=0,
        start_ticks=9,
        cmdline_sha256="command",
        cgroup_sha256="container",
        cgroup="/docker/external",
    )
    allowed = dict(
        identity=identity, physical_gpus=[0], max_process_vram_bytes=256 * 1024**2, baseline_sdma_used=836
    )
    policy = dict(
        shared_contexts=[allowed],
        idle_vram_limit_bytes=1024**3,
        budget=dict(planned_peak_bytes_per_gpu=128 * 1024**3, reserve_bytes=8 * 1024**3),
    )
    sample = dict(
        selected_physical_gpus=[0],
        pid_gpu_mapping="PID 123 is using 1 DRM device(s):\n0\n",
        cards={
            "card0": {
                "VRAM Total Memory (B)": str(192 * 1024**3),
                "VRAM Total Used Memory (B)": str(500 * 1024**2),
                "GPU use (%)": "0",
            },
            "system": {"PID123": "python, 1, 201367552, 836, 0"},
        },
    )
    return identity, policy, sample


def test_only_precisely_identified_inactive_context_is_accepted():
    identity, policy, sample = fixture()
    result = policy_module.evaluate(sample, policy, identity_loader=lambda _: identity)
    assert result["status"] == "AVAILABLE_WITH_RECORDED_EXCEPTION"
    changed = dict(identity, start_ticks=10)
    with pytest.raises(policy_module.ResourceConflict, match="identity"):
        policy_module.evaluate(sample, policy, identity_loader=lambda _: changed)
    with pytest.raises(policy_module.ResourceConflict, match="unapproved"):
        policy_module.evaluate(sample, None, identity_loader=lambda _: identity)
    sample["pid_gpu_mapping"] = "PID 123 is using 2 DRM device(s):\n0 1\n"
    with pytest.raises(policy_module.ResourceConflict, match="expanded"):
        policy_module.evaluate(sample, policy, identity_loader=lambda _: identity)


@pytest.mark.parametrize(
    "statistics",
    ["python, 1, 300000000, 836, 0", "python, 1, 201367552, 837, 0", "python, 1, 201367552, 836, 1"],
)
def test_memory_growth_sdma_or_compute_activity_stop_even_while_own_kernel_runs(statistics):
    identity, policy, sample = fixture()
    sample["cards"]["card0"]["GPU use (%)"] = "90"
    sample["cards"]["system"]["PID123"] = statistics
    with pytest.raises(policy_module.ResourceConflict, match="activity"):
        policy_module.evaluate(
            sample, policy, own_container="own-container", identity_loader=lambda _: identity
        )


def test_memory_budget_and_unapproved_new_process_are_not_bypassed():
    identity, policy, sample = fixture()
    sample["cards"]["card0"]["VRAM Total Memory (B)"] = str(128 * 1024**3)
    with pytest.raises(policy_module.ResourceConflict, match="budget"):
        policy_module.evaluate(sample, policy, identity_loader=lambda _: identity)
    _, _, original = fixture()
    sample = copy.deepcopy(original)
    sample["pid_gpu_mapping"] += "PID 456 is using 1 DRM device(s):\n0\n"
    with pytest.raises(policy_module.ResourceConflict, match="unapproved"):
        policy_module.evaluate(
            sample, policy, own_container="our-container", identity_loader=lambda pid: dict(identity, pid=pid)
        )
