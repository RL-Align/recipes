# SPDX-License-Identifier: MIT
"""Per-step report rows for the seven-layer N1-N5 matrix, shared by wandb_report.py and plot_report.py.

Stdlib only. All parsing is delegated to aggregate_matrix (same run selection, `correctness_from`
semantics and metric definitions). Nothing is synthesized: a value the evidence lacks is None in the row
and its reason is recorded in the row's `missing` mapping.
"""

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aggregate_matrix as agg  # noqa: E402
from matrix import profiles  # noqa: E402

TASKS = ["fixed", "copy", "dapo", "lcc"]  # lcc: Long-Context Verifiable Sequence GRPO Benchmark (formal target)
# Row field -> metric key of aggregate_matrix.step_record (same definitions as the matrix rows).
FROM_METRICS = {
    "reward": "reward",
    "policy_loss": "policy_loss",
    "rollout_seconds": "rollout_seconds",  # generation_seconds of generated-batch.json
    "rollout_setup_sync_seconds": "rollout_setup_sync_seconds",
    "actor_train_seconds": "actor_train_seconds",
    "actor_logprob_seconds": "actor_logprob_seconds",
    "offload_restore_seconds": "offload_restore_seconds",
    "audit_seconds": "audit_seconds",
    "total_train_seconds": "total_train_seconds",
    "e2e_seconds": "e2e_seconds",
    "response_length": "response_length",
    "truncation_rate": "truncation_rate",
    "rollout_tok_per_active_gpu_s": "rollout_tok_per_active_gpu_s",
    "rollout_tok_per_8gpu_s": "rollout_tok_per_8gpu_s",
    "longest_sample_tok_s": "longest_sample_tok_s",
    "actor_train_tok_s": "actor_train_tok_s",
}
FIELDS = [
    "group", "task", "step", "effective", "response_tokens",
    "mismatch_count", "compared_count", "mean_abs_diff", "max_abs_diff",
    "audit_fresh_mismatch", "audit_fixed_mismatch",
    "kl", "generation_seconds", "rollout_phase_seconds", "rollout_tok_s",
    "advantage_mean_abs", "advantage_std", "advantage_nonzero", "advantage_samples",
    *FROM_METRICS,
]  # fmt: skip
NUMERIC = [f for f in FIELDS if f not in ("group", "task", "effective")]
KL_REASON = "KL loss not enabled (use_kl_loss False); no KL evidence recorded"


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def component(record, index):
    c = record["components"][index]
    return None if c.get("missing") else c


def step_row(group, task, record):
    """One row per GRPO update (step = update index) of a copy/dapo trajectory."""
    m, why = record["metrics"], dict(record["why"])
    row = dict.fromkeys(FIELDS)
    row.update(group=group, task=task, step=record["update"], effective=bool(record["effective"]))
    row["response_tokens"] = record["response_tokens"]
    for field, key in FROM_METRICS.items():
        row[field] = m[key] if number(m[key]) else None
        if m[key] == agg.FAIL:
            why[field] = why.get(key) or "non-finite value"
    parity, fresh, fixed = (component(record, i) for i in range(3))
    if parity:  # training batch: actor logprobs vs rollout logprobs
        row.update(
            mismatch_count=parity["mismatch_count"],
            compared_count=parity["compared_tokens"],
            mean_abs_diff=parity["mean_abs_diff"],
            max_abs_diff=parity["max_abs_diff"],
        )
    else:
        for f in ("mismatch_count", "compared_count", "mean_abs_diff", "max_abs_diff"):
            why[f] = record["components"][0].get("missing", "training-parity receipt missing")
    for field, c, name in (("audit_fresh_mismatch", fresh, "fresh"), ("audit_fixed_mismatch", fixed, "fixed")):
        row[field] = c["mismatch_count"] if c else None
        if c is None or row[field] is None:
            why[field] = f"post-update {name} audit verification.json missing"
    gen, sync = row["rollout_seconds"], row["rollout_setup_sync_seconds"]
    row["generation_seconds"] = gen
    if gen is not None and record["response_tokens"] is not None:
        row["rollout_tok_s"] = record["response_tokens"] / gen
    if gen is not None and sync is not None:
        row["rollout_phase_seconds"] = gen + sync  # = v{n-1}-generate/execution.json seconds
    row["kl"] = None
    why["kl"] = KL_REASON
    adv = record.get("advantage")
    if adv:
        row.update(
            advantage_mean_abs=adv["mean_abs"], advantage_std=adv["std"],
            advantage_nonzero=adv["nonzero"], advantage_samples=adv["samples"],
        )  # fmt: skip
    for f in ("advantage_mean_abs", "advantage_std", "advantage_nonzero", "advantage_samples"):
        why[f] = "update-{n}/reward-advantage-rank-0.json missing".replace("{n}", str(record["update"]))
    why.setdefault("rollout_tok_s", why.get("rollout_seconds"))
    why.setdefault("generation_seconds", why.get("rollout_seconds"))
    why.setdefault("rollout_phase_seconds", why.get("rollout_setup_sync_seconds") or why.get("rollout_seconds"))
    row["missing"] = {f: why.get(f) or "evidence missing" for f in FIELDS if row[f] is None}
    if record.get("attempted_error"):  # refused update: step 1 exists but is not an effective GRPO step
        row["attempted_error"] = record["attempted_error"]
        row["missing"] = {
            f: why_f if why_f.startswith("update ") else f"update {record['update']} refused ({record['attempted_error']}); {why_f}"
            for f, why_f in row["missing"].items()
        }
    return row


def fixed_row(group, verification, source):
    """Single step-0 row of the fixed-weight check (fixed-rollout/verification.json)."""
    row = dict.fromkeys(FIELDS)
    row.update(group=group, task="fixed", step=0, effective=verification.get("status") == agg.PASS)
    for field, key in (
        ("mismatch_count", "mismatch_count"),
        ("compared_count", "compared_tokens"),
        ("mean_abs_diff", "mean_abs_diff"),
        ("max_abs_diff", "max_abs_diff"),
    ):
        row[field] = verification.get(key) if number(verification.get(key)) else None
    row["missing"] = {f: f"{key} absent in {source}" for f, key in (
        ("mismatch_count", "mismatch_count"), ("compared_count", "compared_tokens"),
        ("mean_abs_diff", "mean_abs_diff"), ("max_abs_diff", "max_abs_diff")) if row[f] is None}  # fmt: skip
    return row


def tooling_commit():
    """Commit of the report-tooling checkout: git HEAD of this directory, else env RLK_REPORT_COMMIT, else None."""
    try:
        out = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        if out.stdout.strip():
            return dict(commit=out.stdout.strip(), source="git rev-parse HEAD of the report tooling checkout")
    except (OSError, subprocess.CalledProcessError):
        pass
    env = os.environ.get("RLK_REPORT_COMMIT", "").strip()
    if env:
        return dict(commit=env, source="environment variable RLK_REPORT_COMMIT (no git checkout available)")
    return dict(commit=None, source="unavailable: no git checkout of the report tooling and RLK_REPORT_COMMIT is not set")


def runtime_lock_commit(runtime_lock):
    lock = agg.read(runtime_lock, {}) if runtime_lock else {}
    return lock.get("recipe_commit")


def group_config(name, profile, root, state, source, lock_commit=None):
    prov = agg.read(root / "provenance.json", {}) if root else {}
    sources = prov.get("sources") or {}
    actor, rollout = profile["actor"], profile["rollout"]
    ref = state.get("correctness_from") or {}
    tooling = tooling_commit()
    return dict(
        group=name,
        actor_pp=actor["pp"], actor_tp=actor["tp"], actor_ep=actor["ep"], actor_etp=actor["etp"], actor_dp=actor["dp"],
        actor_nproc=actor["gpus"],
        rollout_tp=rollout["tp"], rollout_gpus=rollout["gpus"],
        image_id=prov.get("image_id"),
        rl_kernel_commit=(sources.get("kernel") or {}).get("commit"),
        verl_commit=(sources.get("verl") or {}).get("commit"),
        recipes_commit=prov.get("recipe_commit") or lock_commit,
        recipes_commit_source="run provenance.json recipe_commit" if prov.get("recipe_commit") else (
            "runtime lock recipe_commit (the commit the image was built from)" if lock_commit else "unavailable: neither run provenance nor runtime lock has recipe_commit"
        ),
        report_tooling_commit=tooling["commit"],
        report_tooling_commit_source=tooling["source"],
        run_id=root.name if root else None,
        run_root=str(root) if root else None,
        correctness_run_id=source[0].name if source else None,
        correctness_from=ref.get("run_root"),
        status=state.get("status"),
    )  # fmt: skip


def load_groups(run_roots, prefer=None, runtime_lock=None):
    """Selected groups (N1..N5 order, only those a supplied run selected) with rows, cells and config.

    The matrix target is lcc when any chosen group run holds formal lcc evidence (else dapo, as before); with an lcc target
    DAPO evidence is not shown (out of scope for an lcc matrix); a dapo-only run set behaves as before.
    """
    chosen = agg.pick_runs(run_roots, prefer)
    target = agg.matrix_target(chosen)
    targets = {p["id"]: p for p in profiles()}
    groups, warnings = {}, []
    lock_commit = runtime_lock_commit(runtime_lock)
    for g in agg.GROUPS:
        if g not in chosen:
            continue
        root, state = chosen[g]
        source = agg.correctness_source(g, state)
        cells, copy_cells, dapo, copy = agg.group_cells(g, targets[g], state, root / g, warnings, source, target)
        copy_root = source[0] if source else root
        results = [("copy", copy), (target, dapo)]
        rows, notes = [], []
        for task, result in results:
            if not result["records"]:
                notes.append(f"{task}: no steps.json/updates recorded")
            rows += [step_row(g, task, r) for r in result["records"]]
        fixed_dir = root / g / "fixed-rollout" / "verification.json"
        if not fixed_dir.is_file():
            fixed_dir = copy_root / g / "fixed-rollout" / "verification.json"
        fixed = agg.read(fixed_dir)
        if fixed:
            rows.insert(0, fixed_row(g, fixed, str(fixed_dir)))
        else:
            notes.append("fixed: fixed-rollout/verification.json absent")
        config = group_config(g, targets[g], root, state, source, lock_commit)
        config.update(target_dataset=target)
        groups[g] = dict(
            name=g,
            target=target,
            config=config,
            rows=rows,
            cells=cells,
            copy_cells=copy_cells,
            notes=notes,
            fixed_source=str(fixed_dir) if fixed else None,
        )
    agg.compute_cross_config(chosen, target, {g: v["cells"] for g, v in groups.items()})
    return groups, warnings


def group_target(groups):
    """Target dataset of the loaded groups ('lcc' when present, else 'dapo')."""
    return next((v.get("target", "dapo") for v in groups.values()), "dapo")


def target_tasks(groups):
    """Task names of the target trajectories shown in the main curves: copy plus the matrix target."""
    return ("copy", group_target(groups))


def all_rows(groups):
    return [row for g in groups.values() for row in g["rows"]]


def matrix_table(groups):
    """(columns, rows) of the Performance & Consistency matrix: label + one value per selected group."""
    columns = list(groups)
    rows = [
        [label.strip("*")] + [agg.show(groups[g]["cells"][key]) for g in columns] for label, key, _ in agg.rows_for(group_target(groups))
    ]
    return columns, rows


def matrix_footnotes(groups):
    """Footnote lines under the matrix: attempted-step markers plus the cross-group rollout token identity."""
    cells = {g: v["cells"] for g, v in groups.items()}
    label = agg.TARGETS[group_target(groups)]["short"]
    return [agg.rollout_identity(cells, list(groups), label)["text"]] + agg.footnotes(cells, list(groups))


# Row field -> W&B/panel name `<task>/<section>/<name>` (sections: consistency, reward, loss, timing, throughput, response).
SECTIONS = {
    "mismatch_count": "consistency/mismatch_count",
    "compared_count": "consistency/compared_tokens",
    "mean_abs_diff": "consistency/logprob_abs_diff_mean",
    "max_abs_diff": "consistency/logprob_abs_diff_max",
    "audit_fresh_mismatch": "consistency/audit_fresh_mismatch_count",
    "audit_fixed_mismatch": "consistency/audit_fixed_mismatch_count",
    "reward": "reward/mean",
    "kl": "loss/kl",
    "policy_loss": "loss/policy",
    "rollout_seconds": "timing/rollout",
    "rollout_phase_seconds": "timing/rollout_phase",
    "rollout_setup_sync_seconds": "timing/rollout_setup_and_weight_sync",
    "actor_logprob_seconds": "timing/actor_logprob",
    "actor_train_seconds": "timing/actor_train",
    "offload_restore_seconds": "timing/offload_restore",
    "audit_seconds": "timing/audit",
    "total_train_seconds": "timing/total_train",
    "e2e_seconds": "timing/step_e2e",
    "rollout_tok_per_active_gpu_s": "throughput/rollout_tok_per_gpu_s",
    "rollout_tok_per_8gpu_s": "throughput/rollout_tok_per_8gpu_s",
    "rollout_tok_s": "throughput/rollout_tok_s",
    "longest_sample_tok_s": "throughput/longest_sample_tok_s",
    "actor_train_tok_s": "throughput/actor_train_tok_s",
    "response_length": "response/length_mean",
    "truncation_rate": "response/truncation_rate",
    "response_tokens": "response/tokens",
    "advantage_mean_abs": "advantage/mean_abs",
    "advantage_std": "advantage/std",
    "advantage_nonzero": "advantage/nonzero_count",
    "advantage_samples": "advantage/samples",
}  # generation_seconds duplicates rollout_seconds (rounds.csv only)
