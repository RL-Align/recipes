# SPDX-License-Identifier: MIT
"""Long-context copy (lcc) target: matrix rows, advantage, cross-config, coverage, W&B groups (synthetic run roots)."""

import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "scripts/seven_layer"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rep = _load(HERE / "test_seven_layer_report.py", "seven_report_tests_for_lcc")
fx, rd, wr = rep.fx, rep.rd, rep.wr
SUITE_ROWS = [("boundary-2048", 2048), ("AB-0", 127)]


def lcc_world(tmp_path, groups=("N1", "N2"), diverge_n2=True, legacy=False):
    """work/runs-N/run-l with fixed + copy + lcc evidence; N2 differs from N1 in the update-2 model/optimizer digests only.

    The legacy state_digest always differs per group (it covers provenance and process identity); legacy=True
    receipts carry only that digest.
    """
    work = tmp_path / "work"
    run = fx.make_run(work / "runs-N", "run-l", list(groups))
    suite = work / "evidence/fixed-input-suite.json"
    fx.dump(suite, dict(samples=[dict(id=i, prompt_ids=[0] * n, response_ids=[0] * 3) for i, n in SUITE_ROWS]))
    for g in groups:
        fx.build_group(run, g, "copy")
        fx.build_group(run, g, "lcc")
        verification = fx.verif(100)
        verification.update(
            input_suite_sha256=rd.agg.sha256_of(suite),
            rows=[dict(id=i, status="PASS", expected_tokens=3, mismatch_count=0) for i, _ in SUITE_ROWS],
        )
        fx.dump(run / g / "fixed-rollout/verification.json", verification)
        base = run / g / "lcc"
        for n in (1, 2, 3):
            path = base / f"v{n - 1}-generate/generated-batch.json"
            batch = json.loads(path.read_text())
            for i, s in enumerate(batch["samples"]):
                s["prompt_ids"] = [0] * 100
                s["reward_evidence"] = dict(score=s["raw_reward"], truncated=(i == 1))
            passages = [dict(passage=0, source_rows=[1024, 1025, 1026], partial_last_row=True)]
            batch["request"] = dict(max_tokens=4, source=dict(parquet_sha256="a" * 64, passages=passages))
            fx.dump(path, batch)
            for rank in (0, 1):
                path = base / f"update-{n}/updated-rank-{rank}.json"
                receipt = json.loads(path.read_text())
                digest = f"x{n}-{rank}" if (diverge_n2 and g == "N2" and n == 2) else f"w{n}-{rank}"
                checkpoint = dict(state_digest=f"s{n}-{rank}-{g}")
                if not legacy:
                    checkpoint.update(model_state_digest=digest, optimizer_state_digest="o" + digest)
                fx.dump(path, dict(receipt, rank=rank, checkpoint=checkpoint))
                fx.dump(
                    base / f"update-{n}/reward-advantage-rank-{rank}.json",
                    dict(raw_rewards=[1.0, 0.0], group_ids=["0", "0"], nonzero_advantages=9),
                )
    return run


def lcc_md(tmp_path, **kw):
    run = lcc_world(tmp_path, **kw)
    out = rd.agg.aggregate([run], tmp_path / "out", fx.lock_file(tmp_path))
    return run, out, (tmp_path / "out/matrix.md").read_text(encoding="utf-8")


def test_advantage_summary_is_recomputed_or_read_from_the_receipt(tmp_path):
    agg = rd.agg
    fx.dump(tmp_path / "u/reward-advantage-rank-0.json", dict(raw_rewards=[1.0, 0.0, 0.5, 0.5], group_ids=["a", "a", "b", "b"]))
    s = agg.advantage_summary(tmp_path / "u")
    assert s["samples"] == 4 and s["nonzero"] == 2 and "recomputed" in s["source"]
    assert s["mean_abs"] == pytest.approx(0.35355, abs=1e-3)
    fx.dump(tmp_path / "z/reward-advantage-rank-0.json", dict(raw_rewards=[-1.0] * 4, group_ids=["a"] * 4))
    assert agg.advantage_summary(tmp_path / "z")["nonzero"] == 0  # all-equal rewards: no signal, never invented
    fx.dump(
        tmp_path / "r/reward-advantage-rank-0.json",
        dict(advantage_samples=8, advantage_nonzero_samples=7, advantage_mean_abs=0.5, advantage_std=0.9, raw_rewards=[]),
    )
    assert agg.advantage_summary(tmp_path / "r") == dict(samples=8, nonzero=7, mean_abs=0.5, std=0.9, source="receipt")
    assert agg.advantage_summary(tmp_path / "missing") is None


def test_lcc_matrix_rows_notes_and_no_dapo_section(tmp_path):
    run, out, md = lcc_md(tmp_path)
    cells = out["cells"]["N1"]
    assert cells["reward_per_step"]["value"] == "0.5 / 0.5 / 0.5 (mean 0.5)"
    assert cells["truncation_rate"]["value"] == pytest.approx(0.5)
    assert cells["advantage_stats"]["value"] == "mean|A| 0.7071, std 1, nonzero 6/6"
    assert cells["effective_steps"]["value"] == "3/3" and cells["target_workload"]["value"] == "PASS"
    assert "| LCC Rollout Token SHA256 |" in md and "| Truncation Rate | 0.5 |" in md and "| Mean Reward per Step |" in md
    assert "Target task: Long-Context Verifiable Sequence GRPO Benchmark (not math reasoning), dataset key lcc" in md
    assert "Reward (deterministic, reference-based, no shaping): 1 - EditDistance(out_ids, target_ids)" in md
    assert "Seven-layer rewards are chance-level" in md and "not learning or task quality" in md
    assert "LCC passage source" in md and "passage 0: rows 1024-1026 (last row partial)" in md
    assert "parquet sha256 " + "a" * 64 in md
    assert "Units: times in seconds" in md and "Active GPUs per phase (profile): N1 actor 2" in md and "Aggregation:" in md
    assert "Not run in the supplied runs (columns BLOCKED): N3, N4, N5." in md
    assert "N4, N5 need all 8 GPUs (rollout TP8 / 8-GPU EP4 actor); they are never run on a smaller GPU pool" in md
    assert "### Correctness (copy task)" in md and "DAPO-Math-17k (numeric" not in md
    matrix = json.loads((tmp_path / "out/matrix.json").read_text())
    assert matrix["target_dataset"] == "lcc" and "dapo_numeric_consistency" not in matrix
    assert (tmp_path / "out/steps-N1-lcc.json").is_file() and not (tmp_path / "out/steps-N1-dapo.json").exists()


def test_cross_config_reports_match_and_first_divergence(tmp_path):
    run, out, md = lcc_md(tmp_path)
    cross = json.loads((tmp_path / "out/matrix.json").read_text())["cross_config"]
    assert cross["v0"]["tokens_all_match"] and cross["v0"]["rewards_all_match"] and cross["v0"]["reward_distribution_all_match"]
    t = cross["trajectory"]
    assert t["status"] == "DIVERGES"
    first = t["first_divergence"]
    assert (first["update"], first["field"], first["group"], first["reference_group"]) == (2, "model_digest", "N2", "N1")
    assert [(d["update"], d["field"]) for d in t["divergences"]] == [(2, "model_digest"), (2, "optimizer_digest")]
    assert [r["tokens_sha256"] for r in t["per_update"]] == ["MATCH"] * 3
    assert [r["model_digest"] for r in t["per_update"]] == ["MATCH", "DIVERGES", "MATCH"]
    assert [r["optimizer_digest"] for r in t["per_update"]] == ["MATCH", "DIVERGES", "MATCH"]
    # Every update and field is compared; the per-group legacy state_digest is never a divergence.
    rows = t["comparisons"]["N2"]["rows"]
    assert len(rows) == 3 * 4 and t["comparisons"]["N2"]["compared_fields"] == 12
    assert {r["status"] for r in rows} == {"MATCH", "DIVERGES"}
    cells = out["cells"]
    assert cells["N1"]["cross_config"]["value"] == "reference"
    assert cells["N2"]["cross_config"]["value"] == (
        "v0 tokens MATCH, rewards MATCH, trajectory DIVERGES at update 2: model_digest, optimizer_digest"
    )
    assert "### Cross-config consistency" in md and "first divergence at update 2, field model_digest" in md
    assert "All 2 divergent comparisons:" in md and "- update 2 optimizer_digest: N1=" in md
    assert "| Update | Response tokens | Rewards | Model digests | Optimizer digests |" in md
    assert "(c) N5 (reported separately):" in md and "N/A: N5 has no generation evidence" in md


def test_cross_config_reports_every_divergent_update_not_only_the_first(tmp_path):
    run = lcc_world(tmp_path)
    path = run / "N2/lcc/v2-generate/generated-batch.json"  # update 3 trains on different tokens too
    batch = json.loads(path.read_text())
    batch["samples"][0]["response_ids"] = [7, 7]
    batch["samples"][0]["raw_reward"] = 0.25
    path.write_text(json.dumps(batch))
    out = rd.agg.aggregate([run], tmp_path / "out", fx.lock_file(tmp_path))
    t = json.loads((tmp_path / "out/matrix.json").read_text())["cross_config"]["trajectory"]
    assert [(d["update"], d["field"]) for d in t["divergences"]] == [
        (2, "model_digest"), (2, "optimizer_digest"), (3, "tokens_sha256"), (3, "rewards"),
    ]  # fmt: skip
    assert out["cells"]["N2"]["cross_config"]["value"].endswith(
        "DIVERGES at update 2: model_digest, optimizer_digest; update 3: tokens_sha256, rewards"
    )


def test_cross_config_legacy_state_digest_is_not_comparable_not_divergent(tmp_path):
    run = lcc_world(tmp_path, diverge_n2=False, legacy=True)
    out = rd.agg.aggregate([run], tmp_path / "out", fx.lock_file(tmp_path))
    t = json.loads((tmp_path / "out/matrix.json").read_text())["cross_config"]["trajectory"]
    legacy = rd.agg.LEGACY_DIGEST
    assert legacy == "not comparable (legacy digest includes run metadata)"
    assert t["status"] == "PARTIAL" and t["divergences"] == [] and t["first_divergence"] is None
    assert [r["model_digest"] for r in t["per_update"]] == [legacy] * 3
    assert {(r["field"], r["status"]) for r in t["comparisons"]["N2"]["rows"]} == {
        ("tokens_sha256", "MATCH"), ("rewards", "MATCH"),
        ("model_digest", "NOT_COMPARABLE"), ("optimizer_digest", "NOT_COMPARABLE"),
    }  # fmt: skip
    assert out["cells"]["N2"]["cross_config"]["value"] == (
        f"v0 tokens MATCH, rewards MATCH, trajectory MATCH on tokens/rewards; weight digests {legacy}"
    )
    md = (tmp_path / "out/matrix.md").read_text(encoding="utf-8")
    assert f"Weight digests: {legacy}" in md and "Trajectory PARTIAL" in md


def test_cross_config_match_and_n5_is_separate(tmp_path):
    run = lcc_world(tmp_path, groups=("N1", "N2", "N5"), diverge_n2=False)
    agg = rd.agg
    out = agg.aggregate([run], tmp_path / "out", fx.lock_file(tmp_path))
    cross = json.loads((tmp_path / "out/matrix.json").read_text())["cross_config"]
    assert cross["trajectory"]["status"] == "MATCH" and cross["trajectory"]["groups"] == ["N1", "N2"]
    assert cross["n5"]["status"] == "MATCH" and "not compared with the EP1 digests" in cross["n5"]["note"]
    assert out["cells"]["N5"]["cross_config"]["value"].startswith("v0 tokens MATCH, rewards MATCH; trajectory vs N1: MATCH")
    path = run / "N2/lcc/v0-generate/generated-batch.json"  # a changed v0 token must be reported, not hidden
    batch = json.loads(path.read_text())
    batch["samples"][0]["response_ids"] = [9, 9]
    path.write_text(json.dumps(batch))
    agg.aggregate([run], tmp_path / "out2", fx.lock_file(tmp_path))
    cross2 = json.loads((tmp_path / "out2/matrix.json").read_text())["cross_config"]
    assert not cross2["v0"]["tokens_all_match"] and cross2["v0"]["rewards_all_match"]
    assert cross2["trajectory"]["first_divergence"]["field"] == "tokens_sha256"


def test_cross_config_is_na_for_a_single_group(tmp_path):
    run, out, md = lcc_md(tmp_path, groups=("N1",))
    assert out["cells"]["N1"]["cross_config"]["value"] == "N/A"
    assert "N/A: needs generation evidence of at least two groups" in md


def test_long_context_coverage_separates_real_tokens_from_the_fixed_suite(tmp_path):
    run, out, md = lcc_md(tmp_path)
    cov = json.loads((tmp_path / "out/matrix.json").read_text())["long_context_coverage"]
    c = cov["training"]["N1"]
    assert (c["updates"], c["samples"], c["max_total"], c["max_prompt"], c["max_response"]) == (3, 6, 104, 100, 4)
    assert c["above"] == {"2048": 0, "4096": 0, "8192": 0}
    rows = {r["id"]: r for r in cov["fixed_suite"]["rows"]}
    assert rows["boundary-2048"]["prompt_tokens"] == 2048 and rows["boundary-2048"]["total_tokens"] == 2051
    assert rows["boundary-2048"]["status"] == {"N1": "PASS", "N2": "PASS"}
    assert "(a) Real generated training tokens of the LCC trajectory" in md
    assert "(b) Operator coverage by the independent fixed-weight long-sequence audit suite" in md
    assert "| boundary-2048 | 2048 | 3 | 2051 | PASS | PASS |" in md
    assert "never part of any throughput" in md


def test_dapo_only_runs_still_aggregate_and_do_not_leak_into_an_lcc_matrix(tmp_path):
    legacy = rep.attempted_run(tmp_path)  # old DAPO-only run with a refused update
    out = rd.agg.aggregate([legacy], tmp_path / "old", fx.lock_file(tmp_path))
    assert out["cells"]["N1"]["effective_steps"]["value"] == "0/3"
    old_md = (tmp_path / "old/matrix.md").read_text(encoding="utf-8")
    assert "DAPO Rollout Token SHA256" in old_md and "Target task: Long-Context" not in old_md
    lcc = lcc_world(tmp_path / "n", groups=("N1",))
    both = rd.agg.aggregate([lcc, legacy], tmp_path / "both", fx.lock_file(tmp_path))  # lcc run wins N1 without --prefer
    md = (tmp_path / "both/matrix.md").read_text(encoding="utf-8")
    assert both["cells"]["N1"]["effective_steps"]["value"] == "3/3"
    assert "Math-task effective training" not in md and "(numeric consistency only)" not in md


def test_wandb_runs_are_split_by_task_with_distinct_groups_and_names(tmp_path):
    groups, _ = rd.load_groups([lcc_world(tmp_path)])
    runs = wr.payloads(groups)
    ids = [r["id"] for r in runs]
    expected = [f"dsv4-7l-{g}{s}" for g in ("N1", "N2") for s in ("", "-fixed", "-copy", "-lcc")]
    assert ids == expected
    lcc = runs[3]
    assert lcc["job_type"] == "lcc" and lcc["wandb_group"] == "dsv4-7l-lcc" and "lcc" in lcc["tags"]
    assert lcc["summary"]["task_name"] == "Long-Context Verifiable Sequence GRPO Benchmark (not math reasoning)"
    assert all(k.startswith("lcc/") for item in lcc["history"] for k in item)  # no mixed trajectories
    assert any("lcc/advantage/mean_abs" in item and "lcc/response/truncation_rate" in item for item in lcc["history"])
    copy = runs[2]
    assert copy["wandb_group"] == "dsv4-7l-copy" and all(k.startswith("copy/") for item in copy["history"] for k in item)
    assert runs[0]["history"] == [] and runs[0]["job_type"] == "group"
    assert not any("dapo" in r["id"] for r in runs)
    fake = rep.FakeWandb()
    wr.upload(fake, runs, rd.matrix_table(groups), [], mode="offline", entity="RL-Kernel", project="p", base_url=None, directory=tmp_path / "wb")
    assert {i["job_type"] for i in fake.inits} == {"group", "fixed", "copy", "lcc", "summary"}
    assert {i["group"] for i in fake.inits} >= {"dsv4-7l-lcc", "dsv4-7l-copy", "dsv4-7l-matrix"}
    table = fake.runs[-1].log_calls[0]["performance_consistency_matrix"]
    assert "LCC Rollout Token SHA256" in [r[0] for r in table["data"]]


def test_lcc_plots_and_package(tmp_path):
    pytest.importorskip("matplotlib")
    import zipfile

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot = _load(SCRIPTS / "plot_report.py", "seven_plot_report_lcc")
    pk = _load(SCRIPTS / "package_report.py", "seven_package_report_lcc")
    run = lcc_world(tmp_path)
    groups, _ = rd.load_groups([run])
    rows = rd.all_rows(groups)
    assert {r["task"] for r in rows} == {"fixed", "copy", "lcc"}
    assert plot.series(rows, "N1", "lcc", "reward")[1] == [0.5, 0.5, 0.5]
    out = tmp_path / "plots"
    out.mkdir()
    plot.matrix_png(plt, groups, out)
    for fn in (plot.summary_png, plot.trajectories_png, plot.consistency_png, plot.logp_png):
        fn(plt, groups, rows, out)
    assert (out / "performance-matrix.png").stat().st_size > 0
    result = pk.build([run], tmp_path / "pkg/r.zip", runtime_lock=fx.lock_file(tmp_path), wandb=rep.FakeWandb())
    z = zipfile.ZipFile(result["zip"])
    sync = z.read("dsv4-7l-report/SYNC.md").decode()
    assert "dsv4-7l-N1-lcc" in sync and "lcc = Long-Context Verifiable Sequence GRPO Benchmark" in sync
    assert "Cross-config consistency" in z.read("dsv4-7l-report/matrix.md").decode()
