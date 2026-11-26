# SPDX-License-Identifier: MIT
"""Report row building and W&B payloads on tiny synthetic run roots (CPU only; wandb/matplotlib optional)."""

import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fx = _load(HERE / "test_seven_layer.py", "seven_fixtures")  # build_group / make_run / dump
SCRIPTS = HERE.parent / "scripts/seven_layer"
rd = _load(SCRIPTS / "report_data.py", "seven_report_data")
wr = _load(SCRIPTS / "wandb_report.py", "seven_wandb_report")


def run_with(tmp_path, selected=("N1",), copy=True, **kw):
    run = fx.full_run(tmp_path, selected=selected, **kw)
    if copy:
        for g in selected:
            fx.build_group(run, g, "copy", **kw)
    return run


def by_task(rows, task):
    return [r for r in rows if r["task"] == task]


def test_rows_per_step_and_metric_sources(tmp_path):
    groups, _ = rd.load_groups([run_with(tmp_path)])
    rows = groups["N1"]["rows"]
    assert [r["task"] for r in rows] == ["fixed"] + ["copy"] * 3 + ["dapo"] * 3
    fixed = rows[0]
    assert fixed["step"] == 0 and fixed["compared_count"] == 100 and fixed["mismatch_count"] == 0
    r2 = by_task(rows, "dapo")[1]
    assert r2["step"] == 2 and r2["effective"]
    assert r2["generation_seconds"] == pytest.approx(4.0) and r2["weight_sync_seconds"] == pytest.approx(5.0)
    assert r2["rollout_phase_seconds"] == pytest.approx(9.0)
    assert r2["rollout_tok_s"] == pytest.approx(6 / 4.0) and r2["rollout_tok_per_active_gpu_s"] == pytest.approx(1.5)
    assert r2["actor_logprob_seconds"] == pytest.approx(3.0) and r2["actor_train_seconds"] == pytest.approx(6.0)
    assert r2["e2e_seconds"] == pytest.approx(200.0) and r2["response_length"] == pytest.approx(3.0)
    assert r2["reward"] == pytest.approx(0.5) and r2["policy_loss"] == pytest.approx(0.6)
    assert r2["mismatch_count"] == 0 and r2["compared_count"] == 6  # one rank, not rank-multiplied
    assert r2["mean_abs_diff"] == pytest.approx(1.4 / 6) and r2["max_abs_diff"] == pytest.approx(0.5)
    assert r2["audit_fixed_mismatch"] == 0


def test_kl_is_missing_with_reason_and_never_filled(tmp_path):
    groups, _ = rd.load_groups([run_with(tmp_path)])
    for r in by_task(groups["N1"]["rows"], "dapo"):
        assert r["kl"] is None and "use_kl_loss" in r["missing"]["kl"]


def test_missing_evidence_is_omitted_and_listed(tmp_path):
    run = run_with(tmp_path, copy=False)
    (run / "N1/dapo/v1-generate/execution.json").unlink()  # step 2 weight sync evidence gone
    groups, _ = rd.load_groups([run])
    r2 = by_task(groups["N1"]["rows"], "dapo")[1]
    assert r2["weight_sync_seconds"] is None and r2["rollout_phase_seconds"] is None
    assert "weight_sync_seconds" in r2["missing"]
    assert any("copy: no steps" in n for n in groups["N1"]["notes"])
    runs = wr.payloads(groups)
    assert not any("dapo/timing/weight_sync" in k for k in runs[0]["history"][2])  # omitted, not zero
    assert any("dapo/step2/weight_sync_seconds" in m for m in runs[0]["summary"]["missing"])


def test_nonfinite_policy_loss_is_not_a_number(tmp_path):
    groups, _ = rd.load_groups([run_with(tmp_path, copy=False, bad_loss=True)])
    r2 = by_task(groups["N1"]["rows"], "dapo")[1]
    assert r2["policy_loss"] is None and "non-finite" in r2["missing"]["policy_loss"]


def test_only_selected_groups_get_runs_and_config(tmp_path):
    groups, _ = rd.load_groups([run_with(tmp_path, selected=("N1", "N2"))])
    assert list(groups) == ["N1", "N2"]
    cfg = groups["N2"]["config"]
    assert (cfg["actor_pp"], cfg["actor_tp"], cfg["actor_ep"], cfg["actor_nproc"], cfg["rollout_tp"]) == (2, 1, 1, 2, 2)
    assert cfg["run_id"] == "run-a" and cfg["correctness_from"] is None


def test_wandb_payload_namespaces_and_stable_ids(tmp_path):
    groups, _ = rd.load_groups([run_with(tmp_path)])
    run = wr.payloads(groups)[0]
    assert run["id"] == "dsv4-7l-N1" and run["config"]["actor_nproc"] == 2
    hist = run["history"]
    assert hist[0]["fixed/step"] == 0 and hist[1]["copy/step"] == 1 and hist[-1]["dapo/step"] == 3
    assert "dapo/loss/policy" in hist[-1] and "dapo/loss/kl" not in hist[-1]
    assert run["summary"]["matrix/fixed_consistency"] == "PASS"


class FakeRun:
    def __init__(self, log):
        self.log_calls, self.summary, self.finished = log, {}, False

    def log(self, item):
        self.log_calls.append(item)

    def finish(self):
        self.finished = True


class FakeWandb:
    def __init__(self):
        self.inits, self.runs, self.metrics = [], [], []

    def init(self, **kw):
        self.inits.append(kw)
        self.runs.append(FakeRun([]))
        return self.runs[-1]

    def define_metric(self, *a, **kw):
        self.metrics.append((a, kw))

    Table = staticmethod(lambda columns, data: dict(columns=columns, data=data))
    Image = staticmethod(lambda path: dict(image=path))


def test_upload_one_run_per_group_plus_summary_offline(tmp_path):
    groups, _ = rd.load_groups([run_with(tmp_path, selected=("N1", "N2"))])
    fake = FakeWandb()
    ids = wr.upload(
        fake, wr.payloads(groups), rd.matrix_table(groups), [tmp_path / "performance-matrix.png"],
        mode="offline", entity="RL-Kernel", project="dsv4-rocm-single-node", base_url=wr.DEFAULT_BASE_URL, directory=tmp_path / "wb",
    )  # fmt: skip
    assert ids == ["dsv4-7l-N1", "dsv4-7l-N2", "dsv4-7l-summary"]
    assert {i["mode"] for i in fake.inits} == {"offline"} and fake.inits[0]["entity"] == "RL-Kernel"
    assert fake.inits[0]["resume"] == "allow" and all(r.finished for r in fake.runs)
    table = fake.runs[-1].log_calls[0]["performance_consistency_matrix"]
    assert table["columns"] == ["Metric", "N1", "N2"] and len(table["data"]) == len(rd.agg.ROWS)
    assert "performance-matrix" in fake.runs[-1].log_calls[1]


def test_plots_and_rounds_csv(tmp_path):
    pytest.importorskip("matplotlib")
    plot = _load(SCRIPTS / "plot_report.py", "seven_plot_report")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups, _ = rd.load_groups([run_with(tmp_path)])
    rows = rd.all_rows(groups)
    out = tmp_path / "plots"
    out.mkdir()
    plot.write_rounds(out / "rounds.csv", rows)
    plot.write_summary(out / "summary.json", groups, rows, "cmd")
    plot.matrix_png(plt, groups, out)
    plot.summary_png(plt, groups, rows, out)
    plot.trajectories_png(plt, groups, rows, out)
    plot.consistency_png(plt, groups, rows, out)
    plot.logp_png(plt, groups, rows, out)
    for name in ("performance-matrix", "performance-summary", "performance-trajectories", "consistency-reward", "mean-logp-diff"):
        assert (out / f"{name}.png").stat().st_size > 0
    assert len((out / "rounds.csv").read_text().splitlines()) == 1 + 7
    assert json.loads((out / "summary.json").read_text())["groups"]["N1"]["config"]["group"] == "N1"


def test_package_zip_contents_and_secret_scan(tmp_path):
    pytest.importorskip("matplotlib")
    import zipfile

    pk = _load(SCRIPTS / "package_report.py", "seven_package_report")
    run = run_with(tmp_path)
    result = pk.build([run], tmp_path / "out" / "report.zip", wandb=FakeWandb())
    names = zipfile.ZipFile(result["zip"]).namelist()
    for part in ("raw/rounds.csv", "raw/rounds.json", "matrix.md", "SYNC.md", "figures/performance-matrix.png", "config/N1-config.json"):
        assert f"dsv4-7l-report/{part}" in names
    assert any("/verifier/run-a/N1/dapo/update-verification.json" in n for n in names)
    assert not any(n.endswith(".f32") for n in names)
    sync = zipfile.ZipFile(result["zip"]).read("dsv4-7l-report/SYNC.md").decode()
    assert "WANDB_BASE_URL=https://api.forge.coreweave.com" in sync and "forge.coreweave.com/wandb/RL-Kernel/dsv4-rocm-single-node" in sync and "RL-Kernel" in sync and "dsv4-rocm-single-node" in sync


def test_secret_scan_flags_credentials(tmp_path):
    pk = _load(SCRIPTS / "package_report.py", "seven_package_report2")
    (tmp_path / "ok.json").write_text('{"compared_tokens": 3, "response_tokens": 5}')
    assert pk.scan_secrets(tmp_path) == []
    (tmp_path / "bad.txt").write_text("export WANDB_API_KEY=abc")
    assert [p for p, _ in pk.scan_secrets(tmp_path)] == ["bad.txt"]
