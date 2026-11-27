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
    result["system"] = {"PID123": "python, 1, 300000000, 0, 0"} if occupied else {}
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
    monkeypatch.setattr(watch, "_alive", lambda pid: True)
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


def test_zero_gpu_kfd_context_is_recorded_without_masking_any_actual_gpu_owner(monkeypatch):
    watch = load("watch")
    monkeypatch.setattr(watch, "_alive", lambda pid: True)
    state = cards()
    state["system"] = {"PID123": "python, 0, 2097152, 0, 0"}
    mapping = ""

    def run(argv):
        if "--json" in argv:
            return json.dumps(state)
        if "--showpidgpus" in argv:
            return mapping
        return "123 root context"

    monkeypatch.setattr(watch, "run", run)
    assert watch.snapshot([0])["idle"] is True
    mapping = "PID 123 is using 1 DRM device(s):\n0\n"
    with pytest.raises(RuntimeError, match="disagree"):
        watch.snapshot([0])
    state["system"]["PID123"] = "python, 1, 2097152, 0, 0"
    assert watch.snapshot([0])["idle"] is False


def test_unknown_gpu_count_is_retryable_but_never_admitted(monkeypatch):
    watch = load("watch")
    monkeypatch.setattr(watch, "_alive", lambda pid: True)
    state = cards()
    state["system"] = {"PID123": "python, unknown, unknown, unknown, unknown"}
    monkeypatch.setattr(watch, "run", lambda argv: json.dumps(state) if "--json" in argv else "")
    with pytest.raises(RuntimeError, match="temporarily unavailable"):
        watch.snapshot([0])


def test_exited_unknown_pid_is_skipped_but_live_one_fails_closed(monkeypatch):
    watch = load("watch")
    alive = False
    monkeypatch.setattr(watch, "_alive", lambda pid: alive)
    state = cards()
    state["system"] = {"PID123": "unknown, unknown, unknown, unknown, unknown"}
    monkeypatch.setattr(watch, "run", lambda argv: json.dumps(state) if "--json" in argv else "")
    sample = watch.snapshot([0])
    assert sample["idle"] is True
    assert sample["transient_exited_pids"] == [123]
    alive = True
    with pytest.raises(RuntimeError, match="temporarily unavailable"):
        watch.snapshot([0])


def test_exited_disagreeing_pid_is_dropped_but_live_one_fails_closed(monkeypatch):
    watch = load("watch")
    alive = False
    monkeypatch.setattr(watch, "_alive", lambda pid: alive)
    state = cards()
    state["system"] = {"PID123": "python, 0, 2097152, 0, 0"}
    mapping = "PID 123 is using 1 DRM device(s):\n4\n"

    def run(argv):
        if "--json" in argv:
            return json.dumps(state)
        if "--showpidgpus" in argv:
            return mapping
        return "123 1 root startup python workload.py"

    monkeypatch.setattr(watch, "run", run)
    sample = watch.snapshot([0])
    assert sample["idle"] is True
    assert sample["transient_exited_pids"] == [123]
    assert sample["processes"] == []
    alive = True
    with pytest.raises(RuntimeError, match="disagree"):
        watch.snapshot([0])


def load_execute():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/seven_layer"))
    return load("execute")


def test_actor_nproc_follows_profile_and_n5_is_selectable_but_needs_eight_gpus(tmp_path):
    execute, matrix = load_execute(), load("matrix")
    assert [execute.actor_nproc(p) for p in matrix.profiles()] == [2, 2, 2, 2, 8]
    with pytest.raises(ValueError, match="select unique groups"):
        execute.run(tmp_path, "image", list(range(8)), ["N6"], None, None, None)
    with pytest.raises(ValueError, match="physical pool too small"):
        execute.run(tmp_path, "image", [0, 1, 2, 3], ["N5"], None, None, None)


def test_per_gpu_locks_allow_disjoint_pools_and_block_overlap_and_legacy(tmp_path):
    import contextlib
    import fcntl

    execute = load_execute()
    with contextlib.ExitStack() as a, contextlib.ExitStack() as b:
        assert execute.acquire_gpu_locks(tmp_path, [1, 0], a) == [
            "execute-N-gpu0.lock",
            "execute-N-gpu1.lock",
        ]
        execute.acquire_gpu_locks(tmp_path, [2, 3], b)
        with contextlib.ExitStack() as c:
            with pytest.raises(RuntimeError, match=r"\[1\]"):
                execute.acquire_gpu_locks(tmp_path, [1, 4], c)
        # GPU 4 was released by the failed attempt.
        with contextlib.ExitStack() as d:
            execute.acquire_gpu_locks(tmp_path, [4, 5], d)
    # Legacy exclusive holder blocks new runs.
    with (tmp_path / "execute-N.lock").open("a") as legacy:
        fcntl.flock(legacy, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with contextlib.ExitStack() as e:
            with pytest.raises(RuntimeError, match="legacy"):
                execute.acquire_gpu_locks(tmp_path, [6], e)
    with contextlib.ExitStack() as f:
        execute.acquire_gpu_locks(tmp_path, [6], f)


def test_phase_timeout_table():
    execute = load_execute()
    assert execute.PHASE_TIMEOUT_SECONDS["copy"] == 7200
    assert execute.PHASE_TIMEOUT_SECONDS["dapo"] == 28800
    assert execute.PHASE_TIMEOUT_SECONDS[None] == 7200


def test_rendezvous_port_race_detection():
    execute = load_execute()
    race = (
        "torch.distributed.DistNetworkError: The server socket has failed to listen on any "
        "local network address. port: 39793, useIpv6: 0, code: -98, name: EADDRINUSE"
    )
    assert execute.is_rendezvous_port_race(race)
    assert not execute.is_rendezvous_port_race("torch.distributed.DistBackendError: RCCL error")
    assert not execute.is_rendezvous_port_race("Traceback (most recent call last):\nValueError: x")
    assert not execute.is_rendezvous_port_race("DistNetworkError: connection reset")


def test_ep4_vs_ep1_actor_scores_compare_raw_bytes(tmp_path):
    execute = load_execute()
    a, b = tmp_path / "ep4", tmp_path / "ep1"
    a.mkdir()
    b.mkdir()
    for d in (a, b):
        (d / "sample-0000.f32").write_bytes(b"\x00\x00\x80\x3f")
        (d / "sample-0001.f32").write_bytes(b"\x00\x00\x00\x40")
        (d / "scores.json").write_text(d.name)  # topology metadata must not matter
    same = execute.compare_actor_scores(a, b)
    assert (same["expected"], same["compared"], same["mismatch"]) == (2, 2, 0)
    (b / "sample-0001.f32").write_bytes(b"\x01\x00\x00\x40")
    assert execute.compare_actor_scores(a, b)["mismatch"] == 1
    (b / "sample-0002.f32").write_bytes(b"\x00" * 4)
    bad = execute.compare_actor_scores(a, b)
    assert (bad["expected"], bad["compared"], bad["mismatch"]) == (3, 2, 2)
    empty = tmp_path / "empty"
    empty.mkdir()
    assert execute.compare_actor_scores(empty, empty)["compared"] == 0


def test_trace_compare_pairs_and_reports_first_byte_difference(tmp_path):
    torch = pytest.importorskip("torch")
    traces = load("compare_traces")
    for name, value in (("a", 1.0), ("r", 1.0000001)):
        d = tmp_path / name
        d.mkdir()
        events = [dict(name="rlk.residual", layer=0, file="0.pt"), dict(name="layer.0.attn.output", layer=0, file="1.pt")]
        (d / "manifest.json").write_text(json.dumps(events + ([dict(name="only-actor", layer=0, file="0.pt")] if name == "a" else [])))
        torch.save(torch.ones(2, 3), d / "0.pt")
        second = torch.ones(2, 3)
        second[1, 2] = value
        torch.save(second, d / "1.pt")
    result = traces.compare([tmp_path / "a"], tmp_path / "r", lambda p: torch.load(p))
    first = result["first_difference"]
    assert (first["name"], first["first_row"], first["first_col"]) == ("layer.0.attn.output", 1, 2)
    assert result["unmatched_actor_names"] == [dict(layer=0, name="only-actor", count=1)]
    assert [r["status"] for r in result["records"]] == ["equal", "differs"]


# ---- aggregate_matrix: tiny synthetic run roots ----


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def verif(compared, mismatch=0, mean_abs=0.0, max_abs=0.0, status=None):
    return dict(
        status=status or ("FAIL" if mismatch else "PASS"),
        expected_tokens=compared,
        compared_tokens=compared,
        mismatch_count=mismatch,
        mean_abs_diff=mean_abs,
        max_abs_diff=max_abs,
    )


def build_group(run, group, dataset="dapo", *, steps=3, fixed_mismatch=0, drop_audit_exec=False, offload_extra=7.0, bad_loss=False):
    """Step n: 2 samples (2+4 tokens), generation 2n s, per-rank timings differ so max-over-ranks matters."""
    base = run / group / dataset
    receipts = []
    for n in range(1, steps + 1):
        samples = [dict(response_ids=[1, 2], raw_reward=1.0), dict(response_ids=[1, 2, 3, 4], raw_reward=0.0)]
        dump(
            base / f"v{n - 1}-generate/generated-batch.json",
            dict(samples=samples, response_tokens=6, generation_seconds=2.0 * n),
        )
        rows = [
            dict(expected_tokens=2, compared_tokens=2, mismatch_count=0, mean_abs_diff=0.1, max_abs_diff=0.2),
            dict(expected_tokens=4, compared_tokens=4, mismatch_count=0, mean_abs_diff=0.3, max_abs_diff=0.5),
        ]
        for rank, scale in ((0, 1.0), (1, 1.5)):  # both ranks hold the same logical batch
            dump(
                base / f"update-{n}/training-parity-rank-{rank}.json",
                dict(expected_tokens=6, compared_tokens=6, mismatch_count=0, rows=rows, actor_logprob_seconds=scale * n),
            )
            dump(
                base / f"update-{n}/updated-rank-{rank}.json",
                dict(actor_train_seconds=(2 + rank) * n, actor_train_and_checkpoint_seconds=(3 + rank) * n),
            )
        dump(base / f"v{n - 1}-generate/execution.json", dict(seconds=2.0 * n + 5.0))  # weight sync = 5
        dump(base / f"update-{n}/execution.json", dict(seconds=5.5 * n + offload_extra))  # offload = offload_extra
        dump(base / f"update-{n}/native-loss-rank0-v{n}.json", [])  # first PP stage computes no loss
        calls = [dict(value=0.2, finite=True), dict(value=0.4, finite=not (bad_loss and n == 2))]
        dump(base / f"update-{n}/native-loss-rank1-v{n}.json", calls)
        dump(base / f"update-{n}/native-loss-rank2-v{n}.json", [dict(value=0.6, finite=True)])
        mismatch = fixed_mismatch if n == 2 else 0
        dump(base / f"v{n}-actor-audit/verification.json", verif(6))
        dump(base / f"v{n}-fixed-actor/verification.json", verif(10, mismatch, 0.5, 0.7 if mismatch else 0.4))
        for phase in ("post-generate", "actor-audit", "fixed-actor", "fixed-rollout"):
            if not (drop_audit_exec and phase == "fixed-rollout" and n == 1):
                dump(base / f"v{n}-{phase}/execution.json", dict(seconds=10.0))
        receipts.append(
            dict(update=n, post_update_fresh_and_fixed_verification="PASS", end_to_end_step_seconds=100.0 * n)
        )
    dump(base / "steps.json", receipts)
    dump(base / "update-verification.json", dict(status="PASS", completed_updates=steps))
    if dataset == "dapo":
        dump(run / group / "fixed-rollout/verification.json", verif(100))


def make_run(root, name, selected, states=None):
    run = root / name
    states = states or {}
    summary = {
        g: states.get(g, dict(status="CORRECTNESS_PASS", fixed_weight="PASS", target_load="PASS"))
        if g in selected
        else dict(status="DEFERRED", reason="not selected for this run")
        for g in ["N1", "N2", "N3", "N4", "N5"]
    }
    dump(run / "summary.json", summary)
    dump(run / "resource-policy.json", dict(selected_physical_gpus=[0, 1], selected_groups=selected))
    return run


def lock_file(tmp_path):
    path = tmp_path / "runtime.lock.json"
    dump(path, dict(image_id="sha256:abc", recipe_commit="r1", sources=dict(datasets=dict(dapo=dict(revision="rev123")))))
    return path


def full_run(tmp_path, name="run-a", selected=("N1",), **kw):
    run = make_run(tmp_path, name, list(selected))
    for g in selected:
        build_group(run, g, **kw)
    return run


def test_aggregate_means_and_cumulative_consistency(tmp_path):
    agg = load("aggregate_matrix")
    run = full_run(tmp_path)
    cells = agg.aggregate([run], tmp_path / "out", lock_file(tmp_path))["cells"]["N1"]
    value = lambda key: cells[key]["value"]
    assert value("actor_topology") == "2/1/1" and value("rollout_gpus") == 1
    assert value("fixed_consistency") == "PASS" and value("effective_steps") == "3/3"
    # one rank's receipt: 3 updates x (6 parity + 6 fresh + 10 fixed), never rank-multiplied
    assert value("compared_tokens") == 66 and value("mismatch_count") == 0
    assert value("mean_abs_diff") == pytest.approx((0.2 + 1.2 + 5.0) / 22)  # weighted by compared tokens
    assert value("max_abs_diff") == pytest.approx(0.5)
    assert value("rollout_seconds") == pytest.approx(4.0)
    assert value("rollout_tok_per_active_gpu_s") == pytest.approx((3 + 1.5 + 1) / 3)
    assert value("rollout_tok_per_8gpu_s") == pytest.approx((3 + 1.5 + 1) / 3 / 8)
    assert value("longest_sample_tok_s") == pytest.approx((2 + 1 + 4 / 6) / 3)
    assert value("actor_logprob_seconds") == pytest.approx(3.0)  # max over ranks, mean over steps
    assert value("actor_train_seconds") == pytest.approx(9 * 2 / 3)
    assert value("total_train_seconds") == pytest.approx(1.5 * 2 + 4 * 2)
    assert value("audit_seconds") == pytest.approx(40.0)
    assert value("e2e_seconds") == pytest.approx(200.0)
    assert value("response_length") == pytest.approx(3.0) and value("reward") == pytest.approx(0.5)
    assert value("target_workload") == "PASS" and value("independent_verifier") == "PASS"
    assert value("policy_loss") == pytest.approx(0.6) and value("rollout_setup_sync_seconds") == pytest.approx(5.0)
    assert value("offload_restore_seconds") == pytest.approx(7.0)
    for key in ("reference_logprob_seconds", "kl_loss"):
        assert value(key) == "N/A" and cells[key]["reason"]


def test_mismatch_sums_and_max_follow_the_failing_audit(tmp_path):
    agg = load("aggregate_matrix")
    run = full_run(tmp_path, fixed_mismatch=3)
    cells = agg.aggregate([run], tmp_path / "out", lock_file(tmp_path))["cells"]["N1"]
    assert cells["mismatch_count"]["value"] == 3 and cells["max_abs_diff"]["value"] == pytest.approx(0.7)
    assert cells["independent_verifier"]["value"] == "FAIL"
    assert cells["fixed_consistency"]["value"] == "PASS"  # v0 fixed verification is a separate gate


def test_missing_evidence_is_na_with_reason_not_zero(tmp_path):
    agg = load("aggregate_matrix")
    run = full_run(tmp_path, drop_audit_exec=True)
    (run / "N1/dapo/update-2/training-parity-rank-0.json").unlink()
    (run / "N1/dapo/update-2/training-parity-rank-1.json").unlink()
    cells = agg.aggregate([run], tmp_path / "out", lock_file(tmp_path))["cells"]["N1"]
    assert cells["audit_seconds"]["value"] == "N/A" and "execution.json" in cells["audit_seconds"]["reason"]
    assert cells["compared_tokens"]["value"] == "N/A" and "training-parity" in cells["compared_tokens"]["reason"]
    assert cells["mismatch_count"]["value"] == "N/A" and cells["actor_logprob_seconds"]["value"] == "N/A"
    assert cells["rollout_seconds"]["value"] == pytest.approx(4.0)  # unaffected rows still measured


def test_blocked_and_unselected_groups(tmp_path):
    agg = load("aggregate_matrix")
    run = make_run(
        tmp_path,
        "run-a",
        ["N1", "N2"],
        dict(N2=dict(status="BLOCKED", fixed_weight="NOT_RUN", reason="GPU0 root-owned process")),
    )
    build_group(run, "N1")
    result = agg.aggregate([run], tmp_path / "out", lock_file(tmp_path))
    n2, n3 = result["cells"]["N2"], result["cells"]["N3"]
    assert n2["mismatch_count"]["value"] == "BLOCKED" and "GPU0" in n2["mismatch_count"]["reason"]
    assert n2["rollout_tp"]["value"] == 2  # topology still comes from the profile
    assert n3["target_workload"]["value"] == "BLOCKED" and "not selected" in n3["target_workload"]["reason"]
    assert result["copy_cells"]["N2"]["three_update"] == "BLOCKED"
    text = (tmp_path / "out/matrix.md").read_text(encoding="utf-8")
    assert "TBD" not in text and "### Correctness (copy task)" in text


def test_group_is_taken_from_the_run_that_selected_it(tmp_path):
    agg = load("aggregate_matrix")
    a = full_run(tmp_path, "run-a", ("N1",))
    b = full_run(tmp_path, "run-b", ("N2", "N3"))
    result = agg.aggregate([a, b], tmp_path / "out", lock_file(tmp_path))
    groups = json.loads((tmp_path / "out/matrix.json").read_text(encoding="utf-8"))["groups"]
    assert groups["N1"]["run"] == str(a.resolve()) and groups["N2"]["run"] == str(b.resolve())
    assert groups["N3"]["run"] == str(b.resolve()) and groups["N4"]["run"] is None
    assert result["cells"]["N3"]["effective_steps"]["value"] == "3/3"
    assert result["cells"]["N4"]["effective_steps"]["value"] == "BLOCKED"


def test_group_in_two_runs_needs_prefer(tmp_path):
    agg = load("aggregate_matrix")
    a = full_run(tmp_path, "run-a", ("N1",))
    b = full_run(tmp_path, "run-b", ("N1",), fixed_mismatch=3)
    with pytest.raises(ValueError, match="--prefer"):
        agg.aggregate([a, b], tmp_path / "out1", lock_file(tmp_path))
    cells = agg.aggregate([a, b], tmp_path / "out2", lock_file(tmp_path), prefer="run-b")["cells"]["N1"]
    assert cells["mismatch_count"]["value"] == 3
    with pytest.raises(ValueError, match="not one of"):
        agg.aggregate([a, b], tmp_path / "out3", lock_file(tmp_path), prefer="run-z")


def test_output_directory_is_never_overwritten(tmp_path):
    agg = load("aggregate_matrix")
    run = full_run(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    with pytest.raises(FileExistsError):
        agg.aggregate([run], out, lock_file(tmp_path))
    assert (out / "keep.txt").read_text() == "x" and not (out / "matrix.md").exists()
    (out / "keep.txt").unlink()
    agg.aggregate([run], out, lock_file(tmp_path))  # an empty existing directory is fine
    with pytest.raises(FileExistsError):
        agg.aggregate([run], out, lock_file(tmp_path))


def test_outputs_copy_task_and_evidence_index(tmp_path):
    agg = load("aggregate_matrix")
    run = full_run(tmp_path)
    build_group(run, "N1", "copy", fixed_mismatch=2)
    dump(run / "N1/dapo/v1-generate/command.json", ["docker", "create"])
    out = tmp_path / "out"
    result = agg.aggregate([run], out, lock_file(tmp_path))
    assert result["cells"]["N1"]["mismatch_count"]["value"] == 0  # copy task never enters the matrix rows
    copy = json.loads((out / "matrix.json").read_text(encoding="utf-8"))["copy_task"]["N1"]
    assert copy["consistency"]["mismatch_count"] == 2 and copy["three_update"] == "PASS"
    md = (out / "matrix.md").read_text(encoding="utf-8")
    assert "| **Mismatch Count** | 0 |" in md and "pinned revision rev123" in md
    assert md.splitlines()[0].startswith("### Seven-layer DeepSeek V4")
    header = (out / "matrix.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == "Metric,N1,N2,N3,N4,N5"
    index = json.loads((out / "evidence-index.json").read_text())["per_group"]["N1"]["files"]
    paths = {f["path"] for f in index}
    assert "N1/fixed-rollout/verification.json" in paths and "N1/dapo/update-verification.json" in paths
    assert all(len(f["sha256"]) == 64 for f in index)
    steps = json.loads((out / "steps-N1-dapo.json").read_text())
    assert [s["update"] for s in steps["steps"]] == [1, 2, 3]
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["image_id"] == "sha256:abc"
    assert prov["runs"][0]["launch_commands"]["N1"][0]["command"] == ["docker", "create"]
    assert any("aggregate_matrix.py" in c for c in prov["reproduction_commands"])


def test_policy_loss_nonfinite_fails_and_negative_offload_is_clamped(tmp_path):
    agg = load("aggregate_matrix")
    run = full_run(tmp_path, bad_loss=True, offload_extra=-2.0)
    result = agg.aggregate([run], tmp_path / "out", lock_file(tmp_path))
    cells = result["cells"]["N1"]
    assert cells["policy_loss"]["value"] == "FAIL" and "non-finite" in cells["policy_loss"]["reason"]
    assert cells["offload_restore_seconds"]["value"] == 0.0
    assert any("clamped" in w for w in result["warnings"])


def test_loss_sync_and_offload_are_na_only_when_evidence_is_missing(tmp_path):
    agg = load("aggregate_matrix")
    run = full_run(tmp_path)
    (run / "N1/dapo/v0-generate/execution.json").unlink()
    (run / "N1/dapo/update-3/execution.json").unlink()
    for rank in range(3):
        (run / f"N1/dapo/update-1/native-loss-rank{rank}-v1.json").unlink()
    cells = agg.aggregate([run], tmp_path / "out", lock_file(tmp_path))["cells"]["N1"]
    for key in ("rollout_setup_sync_seconds", "offload_restore_seconds", "policy_loss"):
        assert cells[key]["value"] == "N/A" and cells[key]["reason"]
    assert cells["rollout_seconds"]["value"] == pytest.approx(4.0)


# ---- target-only / resume ----


def sha_of(path):
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def correctness_run_dir(tmp_path, *, image="sha256:abc", status="CORRECTNESS_PASS", with_verification=True):
    run = make_run(tmp_path, "run-c", ["N1"], states=dict(N1=dict(status=status, fixed_weight="PASS", target_load="FAIL")))
    dump(run / "provenance.json", dict(image_id=image, sources={}, trace_audit_pattern=None))
    if with_verification:
        build_group(run, "N1", "copy")
        dump(run / "N1/fixed-rollout/verification.json", verif(50))
    return run


def run_target_only(execute, tmp_path, correctness, **kw):
    return execute.run(
        tmp_path, "sha256:abc", [0, 1], ["N1"], lock_file(tmp_path), None, None,
        target_only=True, correctness_run=correctness, **kw,
    )  # fmt: skip


def test_target_only_argument_validation(tmp_path):
    execute = load_execute()
    with pytest.raises(ValueError, match="requires --correctness-run"):
        execute.run(tmp_path, "image", [0, 1], ["N1"], None, None, None, target_only=True)
    with pytest.raises(ValueError, match="only valid with --target-only"):
        execute.run(tmp_path, "image", [0, 1], ["N1"], None, None, None, correctness_run=tmp_path)
    with pytest.raises(ValueError, match="only valid with --target-only"):
        execute.run(tmp_path, "image", [0, 1], ["N1"], None, None, None, resume_target=tmp_path)


def test_target_only_refuses_wrong_image_or_unpassed_correctness(tmp_path):
    execute = load_execute()
    with pytest.raises(ValueError, match="image_id differs"):
        run_target_only(execute, tmp_path, correctness_run_dir(tmp_path, image="sha256:other"))
    (tmp_path / "run-c").rename(tmp_path / "old")
    with pytest.raises(ValueError, match="not CORRECTNESS_PASS"):
        run_target_only(execute, tmp_path, correctness_run_dir(tmp_path, status="FAIL"))
    (tmp_path / "run-c").rename(tmp_path / "old2")
    with pytest.raises(ValueError, match="update-verification"):
        run_target_only(execute, tmp_path, correctness_run_dir(tmp_path, with_verification=False))
    with pytest.raises(ValueError, match="runtime image lock differs"):
        execute.check_correctness_run(tmp_path / "old", ["N1"], "sha256:zzz", lock_file(tmp_path))


def test_correctness_from_record_carries_hashes(tmp_path):
    execute = load_execute()
    run = correctness_run_dir(tmp_path)
    rec = execute.check_correctness_run(run, ["N1"], "sha256:abc", lock_file(tmp_path))["N1"]
    assert rec["run_root"] == str(run.resolve())
    assert rec["summary_sha256"] == sha_of(run / "summary.json")
    assert rec["update_verification_sha256"] == sha_of(run / "N1/copy/update-verification.json")


def dapo_prior(root, complete, *, pending=False, group="N1"):
    """Synthetic prior dapo dir with `complete` verified steps and optionally one PENDING step after."""
    base = root / group / "dapo"
    receipts = []
    for k in range(1, complete + 1):
        receipts.append(dict(update=k, post_update_fresh_and_fixed_verification="PASS"))
        for name in (f"v{k - 1}-generate", f"v{k}-post-generate", f"v{k}-actor-audit", f"v{k}-fixed-actor", f"v{k}-fixed-rollout"):
            dump(base / name / "execution.json", {})
        dump(base / f"checkpoint-v{k}/manifest.json", {})
        (base / f"update-{k}").mkdir(parents=True, exist_ok=True)
        (base / f"update-{k}/policy-v{k}.pt").write_bytes(f"p{k}".encode())
    if pending:
        k = complete + 1
        receipts.append(dict(update=k, post_update_fresh_and_fixed_verification="PENDING"))
        dump(base / f"checkpoint-v{k}/manifest.json", {})
        (base / f"update-{k}").mkdir(parents=True, exist_ok=True)
        (base / f"update-{k}/policy-v{k}.pt").write_bytes(b"p")
    if receipts:
        dump(base / "steps.json", receipts)
    else:
        base.mkdir(parents=True, exist_ok=True)
    return base


@pytest.mark.parametrize("complete", [0, 1, 2, 3])
def test_resume_point_counts_complete_steps_and_ignores_incomplete(tmp_path, complete):
    execute = load_execute()
    base = dapo_prior(tmp_path / "prior", complete, pending=complete < 3)
    assert [s["update"] for s in execute.completed_steps(base)] == list(range(1, complete + 1))
    count, policy, checkpoint, receipts = execute.resume_plan(base)
    assert count == complete and len(receipts) == complete
    if complete:
        assert policy.name == f"policy-v{complete}.pt" and checkpoint.name == f"checkpoint-v{complete}"
        assert receipts[0]["resumed_from"]["group"] == str(base.resolve())
    else:
        assert policy is None and checkpoint is None


def test_resume_point_requires_manifest_and_policy_and_missing_dir_is_scratch(tmp_path):
    execute = load_execute()
    base = dapo_prior(tmp_path / "prior", 2)
    (base / "update-2/policy-v2.pt").unlink()
    assert execute.resume_plan(base)[0] == 1
    (base / "checkpoint-v1/manifest.json").unlink()
    assert execute.resume_plan(base)[0] == 0
    assert execute.resume_plan(tmp_path / "nope")[0] == 0


def test_resume_refuses_different_provenance(tmp_path):
    execute = load_execute()
    prov = dict(image_id="sha256:abc", sources={"a": 1}, trace_audit_pattern=None)
    dump(tmp_path / "prior/provenance.json", prov)
    assert execute.check_resume_run(tmp_path / "prior", prov) == (tmp_path / "prior").resolve()
    with pytest.raises(ValueError, match="resume refused"):
        execute.check_resume_run(tmp_path / "prior", dict(prov, image_id="sha256:other"))
    with pytest.raises(ValueError, match="no provenance.json"):
        execute.check_resume_run(tmp_path / "missing", prov)


def test_link_prior_steps_makes_relative_symlinks_without_copying(tmp_path):
    import os

    execute = load_execute()
    work = tmp_path / "work"
    base = dapo_prior(work / "runs-N/old", 2)
    new_group = work / "runs-N/new/N1/dapo"
    new_group.mkdir(parents=True)
    links = execute.link_prior_steps(work, new_group, base, 2)
    assert {link["name"] for link in links} >= {"v0-generate", "update-1", "checkpoint-v2", "v2-fixed-rollout"}
    link = new_group / "update-2"
    assert link.is_symlink() and not os.path.isabs(os.readlink(link))
    assert (new_group / "update-2/policy-v2.pt").read_bytes() == b"p2"
    assert not (new_group / "v3-generate").exists()
    outside = dapo_prior(tmp_path / "elsewhere", 1)
    other = work / "runs-N/new2/N1/dapo"
    other.mkdir(parents=True)
    with pytest.raises(ValueError):
        execute.link_prior_steps(work, other, outside, 1)


def test_aggregate_merges_correctness_from_and_target_only_run(tmp_path):
    import shutil

    agg = load("aggregate_matrix")
    crun = correctness_run_dir(tmp_path)
    cf = dict(
        run_root=str(crun),
        summary_sha256=sha_of(crun / "summary.json"),
        update_verification_sha256=sha_of(crun / "N1/copy/update-verification.json"),
    )
    state = dict(status="TARGET_ONLY_PASS", fixed_weight="PASS", target_load="PASS", correctness_from=cf)
    trun = make_run(tmp_path, "run-t", ["N1"], states=dict(N1=state))
    build_group(trun, "N1")  # dapo only; no copy dir in the target-only run
    # Step 1 lives in a prior run and is only symlinked here.
    prior = tmp_path / "run-p"
    build_group(prior, "N1")
    for name in ("v0-generate", "update-1", "v1-actor-audit", "v1-fixed-actor", "v1-post-generate", "v1-fixed-rollout"):
        shutil.rmtree(trun / "N1/dapo" / name)
        (trun / "N1/dapo" / name).symlink_to(prior / "N1/dapo" / name)
    ref = agg.aggregate([full_run(tmp_path, "run-ref")], tmp_path / "out-ref", lock_file(tmp_path))["cells"]["N1"]
    result = agg.aggregate([trun], tmp_path / "out", lock_file(tmp_path))
    cells = result["cells"]["N1"]
    for key in ("compared_tokens", "mismatch_count", "rollout_seconds", "e2e_seconds", "target_workload", "effective_steps"):
        assert cells[key]["value"] == ref[key]["value"]
    copy = result["copy_cells"]["N1"]
    assert copy["three_update"] == "PASS" and copy["verified_updates"] == 3
    assert copy["fixed_weight"] == "PASS" and copy["source_run"] == str(crun)
    out = tmp_path / "out"
    matrix = json.loads((out / "matrix.json").read_text(encoding="utf-8"))
    assert matrix["groups"]["N1"]["parts"]["copy_correctness"] == str(crun)
    assert matrix["groups"]["N1"]["parts"]["dapo_target"] == str(trun.resolve())
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["group_parts"]["N1"]["copy_correctness"] == str(crun)
    assert any(r.get("role", "").startswith("copy-task correctness") for r in prov["runs"])
    steps_copy = json.loads((out / "steps-N1-copy.json").read_text())
    assert steps_copy["run"] == str(crun)
    index = json.loads((out / "evidence-index.json").read_text())["per_group"]["N1"]
    assert any(f["path"].startswith("N1/dapo/update-1/") for f in index["files"])  # symlinked step indexed
    assert index["correctness_files"]


def test_aggregate_refuses_changed_correctness_run(tmp_path):
    agg = load("aggregate_matrix")
    crun = correctness_run_dir(tmp_path)
    cf = dict(run_root=str(crun), summary_sha256="0" * 64, update_verification_sha256="0" * 64)
    trun = make_run(tmp_path, "run-t", ["N1"], states=dict(N1=dict(status="TARGET_ONLY_PASS", correctness_from=cf)))
    build_group(trun, "N1")
    with pytest.raises(ValueError, match="changed"):
        agg.aggregate([trun], tmp_path / "out", lock_file(tmp_path))


# ---- long-context copy (lcc / lccsmoke) request generation and target selection ----


def lcc_fake_rows(count=200, width=20, first=1024):
    return [(first + i, chr(97 + i % 26) * width) for i in range(count)]


def test_build_passages_takes_the_longest_fitting_prefix_and_is_deterministic():
    prep = load("prepare_requests")
    fits = lambda text: len(text) <= 50  # noqa: E731
    passages = prep.build_passages(lcc_fake_rows(), fits, count=5)
    assert passages == prep.build_passages(lcc_fake_rows(), fits, count=5)
    used = [r for p in passages for r in p["source_rows"]]
    assert used == sorted(set(used)) and used[0] == 1024  # file order, rows never reused
    for p in passages:
        assert fits(p["text"]) and not p["text"].endswith(" ")
        assert len(p["text"]) == 50  # the next character would not fit: longest prefix
        assert p["partial_last_row"] is True and "\n\n" in p["text"]
    with pytest.raises(ValueError, match="exhausted"):
        prep.build_passages(lcc_fake_rows(count=3), fits, count=5)


def lcc_requests(prep, dataset, passages=None):
    passages = passages or prep.build_passages(lcc_fake_rows(), lambda t: len(t) <= 50)
    return prep.lcc_requests(
        passages, "f" * 64, 1, lambda text: list(range(len(text))), lambda text: [ord(c) for c in text], dataset
    )


def test_lcc_requests_shape_source_and_token_targets():
    prep = load("prepare_requests")
    files = lcc_requests(prep, "lcc")
    assert sorted(files) == [f"lcc-step{k}.json" for k in (1, 2, 3, 4)]
    first = files["lcc-step1.json"]
    assert (first["dataset"], first["samples_per_prompt"], first["max_tokens"], first["prompt_cap"]) == ("lcc", 16, 7168, 1024)
    assert len(first["prompts"]) == 8 and first["purpose"] == "optimizer update 1/3"
    assert files["lcc-step4.json"]["purpose"].startswith("post-v3 audit only")
    p = first["prompts"][0]
    assert p["data_source"] == "rlk/long-context-copy" and p["extra_info"]["eos_token_id"] == 1
    assert p["extra_info"]["target_ids"] == [ord(c) for c in p["ground_truth"]]  # encode(passage), no special tokens
    assert len(p["prompt_ids"]) <= 1024
    source = first["source"]
    assert source["parquet_sha256"] == "f" * 64 and "Reward = 1 - token_edit_distance" in source["construction_rule"]
    assert [s["passage"] for s in source["passages"]] == list(range(8))
    assert source["passages"][0]["source_rows"] == p["extra_info"]["source_rows"]
    # steps 2/3 take the next 8 passages, step 4 audits the first 8 again (like copy/dapo)
    assert [q["group_id"] for q in files["lcc-step2.json"]["prompts"]] == [str(i) for i in range(8, 16)]
    assert [q["group_id"] for q in files["lcc-step4.json"]["prompts"]] == [str(i) for i in range(8)]


def test_lccsmoke_is_small_labelled_and_uses_the_same_passages_and_reward():
    prep = load("prepare_requests")
    passages = prep.build_passages(lcc_fake_rows(), lambda t: len(t) <= 50)
    formal, smoke = lcc_requests(prep, "lcc", passages), lcc_requests(prep, "lccsmoke", passages)
    s1 = smoke["lccsmoke-step1.json"]
    assert (s1["samples_per_prompt"], s1["max_tokens"], len(s1["prompts"])) == (4, 2048, 2)
    assert "DIAGNOSTIC" in s1["label"] and "never part of the formal matrix" in s1["label"]
    assert s1["prompts"][0] == formal["lcc-step1.json"]["prompts"][0]
    assert s1["source"]["dataset_key"] == "lccsmoke" and sorted(smoke) == [f"lccsmoke-step{k}.json" for k in (1, 2, 3, 4)]


def test_add_to_lock_extends_without_touching_existing_files(tmp_path):
    prep = load("prepare_requests")
    out = tmp_path / "requests-N"
    out.mkdir()
    (out / "dapo-step1.json").write_text("{}\n")
    original = dict(files={"dapo-step1.json": prep.sha(out / "dapo-step1.json")}, scope="orig")
    (out / "requests.lock.json").write_text(json.dumps(original))
    files = {**lcc_requests(prep, "lcc"), **lcc_requests(prep, "lccsmoke")}
    lock = prep.add_to_lock(out, files)
    assert lock["files"]["dapo-step1.json"] == original["files"]["dapo-step1.json"] and lock["scope"] == "orig"
    assert set(lock["files"]) == {"dapo-step1.json", *files}
    assert all(prep.sha(out / name) == digest for name, digest in lock["files"].items())
    with pytest.raises(ValueError, match="never overwritten"):
        prep.add_to_lock(out, files)
    (out / "dapo-step1.json").write_text("changed\n")
    with pytest.raises(ValueError, match="differs from requests.lock.json"):
        prep.read_lock(out)


def test_target_dataset_option_and_timeouts(monkeypatch, tmp_path):
    execute = load_execute()
    assert execute.TARGET_DATASETS == ("dapo", "lcc", "lccsmoke")
    assert execute.PHASE_TIMEOUT_SECONDS["lcc"] == 28800 and execute.PHASE_TIMEOUT_SECONDS["lccsmoke"] == 7200
    assert execute.PHASE_TIMEOUT_SECONDS["dapo"] == 28800  # unchanged
    with pytest.raises(ValueError, match="target dataset"):
        execute.run(tmp_path, "image", [0, 1], ["N1"], None, None, None, target_dataset="bogus")
    seen = {}
    monkeypatch.setattr(execute, "run", lambda *a, **kw: seen.update(kw))
    base = ["x", "--work", str(tmp_path), "--image", "i", "--gpus", "4", "5", "--groups", "N1", "--runtime-lock", "l",
            "--health-receipt", "h", "--target-only", "--correctness-run", "c"]  # fmt: skip
    monkeypatch.setattr("sys.argv", base)
    execute.main()
    assert seen["target_dataset"] == "dapo"  # default: old behaviour unchanged
    monkeypatch.setattr("sys.argv", base + ["--target-dataset", "lcc", "--resume-target", "r"])
    execute.main()
    assert seen["target_dataset"] == "lcc" and seen["resume_target"] == Path("r")
    monkeypatch.setattr("sys.argv", base + ["--target-dataset", "nope"])
    with pytest.raises(SystemExit):
        execute.main()


@pytest.mark.parametrize("dataset, timeout", [("lcc", 28800), ("lccsmoke", 7200), ("dapo", 28800)])
def test_training_locates_requests_and_output_by_dataset(monkeypatch, tmp_path, dataset, timeout):
    execute = load_execute()
    calls = []

    class Stop(Exception):
        pass

    def fake_launch(work, run, rows, image, profile, action, out, **kw):
        calls.append((action, kw))
        raise Stop

    monkeypatch.setattr(execute, "launch", fake_launch)
    profile = {p["id"]: p for p in execute.profiles()}["N1"]
    run = tmp_path / "runs-N/r"
    with pytest.raises(Stop):
        execute.training(tmp_path, run, [], "img", profile, tmp_path / "policy.pt", dataset)
    assert (run / "N1" / dataset).is_dir()
    action, kw = calls[0]
    assert action == "rollout-generate" and kw["batch"] == tmp_path / f"data/requests-N/{dataset}-step1.json"
    assert kw["timeout_seconds"] == timeout


def test_aggregate_ignores_lccsmoke_runs(tmp_path):
    agg = load("aggregate_matrix")
    smoke = make_run(tmp_path, "run-s", ["N1"])
    dump(smoke / "target-only.json", dict(target_dataset="lccsmoke", correctness_run="c", resume_target=None))
    assert agg.is_smoke_run(smoke) and not agg.is_smoke_run(full_run(tmp_path, "run-f"))
    assert agg.pick_runs([smoke]) == {}
    formal = full_run(tmp_path, "run-g")
    assert list(agg.pick_runs([smoke, formal])) == ["N1"]  # smoke never competes for a group (no --prefer needed)


# ---- pool-scoped health receipts (4 free GPUs) ----


def pool_profiles(*names):
    return [p for p in load("matrix").profiles() if p["id"] in names]


def receipt(**kw):
    base = dict(status="PASS", image_id="img", communication_world_size=8)
    base.update(kw)
    return base


def test_full_receipt_keeps_working_as_before_with_and_without_the_new_field():
    execute = load_execute()
    legacy = receipt()  # original receipt: no qualified_physical_gpus
    execute.check_health_receipt(legacy, "img", list(range(8)), pool_profiles("N4", "N5"))
    execute.check_health_receipt(legacy, "img", [4, 5], pool_profiles("N1"))
    execute.check_health_receipt(receipt(qualified_physical_gpus=list(range(8))), "img", [0, 1], pool_profiles("N1"))
    for bad in (receipt(status="FAIL"), receipt(image_id="other"), receipt(communication_world_size=4)):
        with pytest.raises(ValueError, match="does not qualify"):
            execute.check_health_receipt(bad, "img", [0, 1], pool_profiles("N1"))


def test_pool_receipt_qualifies_only_its_gpus_and_small_groups():
    execute = load_execute()
    pool = receipt(qualified_physical_gpus=[4, 5, 6, 7], communication_world_size=4)
    execute.check_health_receipt(pool, "img", [4, 5], pool_profiles("N1"))
    execute.check_health_receipt(pool, "img", [6, 7], pool_profiles("N2"))
    execute.check_health_receipt(pool, "img", [4, 5, 6, 7], pool_profiles("N1", "N2", "N3"))
    with pytest.raises(ValueError, match="not inside the receipt's qualified pool"):
        execute.check_health_receipt(pool, "img", [3, 4], pool_profiles("N1"))
    with pytest.raises(ValueError, match="not inside the receipt's qualified pool"):
        execute.check_health_receipt(pool, "img", list(range(8)), pool_profiles("N1"))
    for group in ("N4", "N5"):  # TP8 rollout / 8-GPU EP4 actor are never admitted on a 4-GPU receipt
        with pytest.raises(ValueError, match="need more GPUs"):
            execute.check_health_receipt(pool, "img", [4, 5, 6, 7], pool_profiles(group))
    with pytest.raises(ValueError, match="does not qualify"):  # inconsistent receipt: world size must equal the pool
        execute.check_health_receipt(receipt(qualified_physical_gpus=[4, 5, 6, 7]), "img", [4, 5], pool_profiles("N1"))


def test_run_refuses_four_gpu_pool_for_n4_and_n5_before_any_phase(tmp_path):
    execute = load_execute()
    for group in ("N4", "N5"):
        with pytest.raises(ValueError, match="physical pool too small"):
            execute.run(tmp_path, "img", [4, 5, 6, 7], [group], tmp_path / "lock.json", None, tmp_path / "h.json")


def test_qualify_pool_runs_per_card_and_pool_wide_communication(monkeypatch, tmp_path):
    load_execute()
    qualify = load("qualify")
    calls, allocated = [], []
    monkeypatch.setattr(qualify, "allocation", lambda gpus: allocated.append(list(gpus)) or [dict(physical_gpu=g, pci_bus_id=f"p{g}") for g in gpus])

    def fake_phase(work, root, image, policy, rows, mode, **kw):
        calls.append((mode, [r["physical_gpu"] for r in (kw.get("active") or rows)], kw.get("label")))
        return dict(status="PASS")

    monkeypatch.setattr(qualify, "phase", fake_phase)
    lock, policy = tmp_path / "lock.json", tmp_path / "policy.json"
    dump(lock, dict(image_id="img"))
    dump(policy, {})
    (tmp_path / "evidence").mkdir()
    path = qualify.qualify(tmp_path, lock, policy, gpus=[7, 5, 4, 6])
    result = json.loads(path.read_text())
    assert allocated == [[4, 5, 6, 7]] and result["qualified_physical_gpus"] == [4, 5, 6, 7]
    assert result["status"] == "PASS" and result["communication_world_size"] == 4 and "pool-scoped" in result["scope"]
    assert calls == [("single", [g], f"single-gpu{g}") for g in (4, 5, 6, 7)] + [("communication", [4, 5, 6, 7], None)]
    # the default pool is the full node and behaves exactly as before: one card-0 single check + 8-rank communication
    calls.clear()
    full = json.loads(qualify.qualify(tmp_path, lock, policy).read_text())
    assert full["communication_world_size"] == 8 and full["qualified_physical_gpus"] == list(range(8))
    assert calls == [("single", list(range(8)), None), ("communication", list(range(8)), None)]
    for bad in ([4], [4, 4, 5], [4, 9]):
        with pytest.raises(ValueError, match="qualification pool"):
            qualify.qualify(tmp_path, lock, policy, gpus=bad)
