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
    assert value("policy_loss") == pytest.approx(0.6) and value("weight_sync_seconds") == pytest.approx(5.0)
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
    for key in ("weight_sync_seconds", "offload_restore_seconds", "policy_loss"):
        assert cells[key]["value"] == "N/A" and cells[key]["reason"]
    assert cells["rollout_seconds"]["value"] == pytest.approx(4.0)
