# SPDX-License-Identifier: MIT
"""lcc_stats.py: per-update rows from recorded evidence and cross-group identity (synthetic run roots)."""

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/seven_layer/lcc_stats.py"
spec = importlib.util.spec_from_file_location("lcc_stats", SCRIPT)
stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def run_root(tmp_path, name, group, digest="aa" * 32, rewards=(0.0, 0.5, 0.25, 0.25)):
    base = tmp_path / name / group / "lcc"
    samples = [
        dict(prompt_ids=[1] * 4, response_ids=[2] * n) for n in (8, 3, 8, 5)
    ]
    _write(base / "v0-generate/generated-batch.json", dict(samples=samples))
    advantages = [-1.5, 1.5, 0.0, 0.0]
    _write(base / "update-1/reward-advantage-rank-0.json", dict(
        raw_rewards=list(rewards), advantages=advantages, advantage_mean_abs=0.75, advantage_std=1.0607))
    _write(base / "steps.json", [dict(
        update=1, response_tokens=24, generation=dict(seconds=10.04), actor=dict(seconds=20.0),
        fresh_verification=dict(policy_state_digest=digest, status="PASS", compared_tokens=24, mismatch_count=0),
        fixed_verification=dict(status="PASS"))])
    return str(tmp_path / name)


def test_rows_come_from_the_recorded_batch_receipt_and_audits(tmp_path):
    (row,) = stats.update_rows(run_root(tmp_path, "r1", "N1"), "N1", cap=8)
    assert row["truncated"] == "2/4"
    assert row["response_len_range"] == [3, 8] and row["response_len_mean"] == 6.0
    assert row["advantage_nonzero"] == "2/4" and row["advantage_range"] == [-1.5, 1.5]
    assert row["reward_mean"] == 0.25 and row["fresh_mismatch"] == 0 and row["fixed"] == "PASS"
    assert row["policy_digest"] == "aa" * 8


def test_identity_requires_identical_digest_rewards_and_tokens(tmp_path):
    n1 = stats.update_rows(run_root(tmp_path, "r1", "N1"), "N1")
    n2 = stats.update_rows(run_root(tmp_path, "r2", "N2"), "N2")
    assert stats.identity({"N1": n1}) is None
    assert stats.identity({"N1": n1, "N2": n2}) is True
    n3 = stats.update_rows(run_root(tmp_path, "r3", "N3", digest="bb" * 32), "N3")
    assert stats.identity({"N1": n1, "N3": n3}) is False
    n4 = stats.update_rows(run_root(tmp_path, "r4", "N4", rewards=(0.0, 0.5, 0.25, 0.3)), "N4")
    assert stats.identity({"N1": n1, "N4": n4}) is False
