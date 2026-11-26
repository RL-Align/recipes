# SPDX-License-Identifier: MIT
"""Host-side aggregation of the seven-layer ROCm N1-N5 matrix from retained run roots.

Stdlib only (no torch): reads the JSON receipts written by execute.py / phase_runtime.py.
Nothing is invented: a value that the evidence does not contain is N/A with a recorded reason.
"""

import argparse
import csv
import fnmatch
import hashlib
import json
import os
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matrix import profiles  # noqa: E402

NA, FAIL, BLOCKED, PASS = "N/A", "FAIL", "BLOCKED", "PASS"
GROUPS = ["N1", "N2", "N3", "N4", "N5"]
BLOCKED_STATES = {"BLOCKED", "NOT_RUN", "DEFERRED"}
UPDATES = 3
ACTIVE_NOTE = "8 prompts x 16 responses"

# (label, key, definition)
ROWS = [
    ("Actor PP/TP/EP", "actor_topology", "Actor pipeline/tensor/expert parallel sizes from the group profile."),
    ("Actor GPUs", "actor_gpus", "Physical GPUs used by the actor phase (profile)."),
    ("Rollout TP", "rollout_tp", "vLLM rollout tensor-parallel size (profile)."),
    ("Rollout GPUs", "rollout_gpus", "Physical GPUs used by the rollout phase (profile)."),
    (
        "Fixed-weight Consistency",
        "fixed_consistency",
        "PASS only if <group>/fixed-rollout/verification.json (independent fixed-weight verifier) is PASS.",
    ),
    (
        "Effective GRPO Steps",
        "effective_steps",
        "'<n>/3' independently verified effective updates of the DAPO trajectory (update-verification.json).",
    ),
    (
        "Compared Tokens",
        "compared_tokens",
        "Cumulative over the DAPO trajectory: per update, training-batch parity + post-update fresh audit + "
        "post-update fixed audit. Logical tokens: one rank's parity receipt, never rank-multiplied.",
    ),
    ("**Mismatch Count**", "mismatch_count", "Cumulative FP32 bitwise mismatching tokens, same scope as Compared Tokens."),
    ("Mean Abs Diff", "mean_abs_diff", "Mean |actor - rollout| logprob, weighted by compared tokens, same scope."),
    ("Max Abs Diff", "max_abs_diff", "Maximum |actor - rollout| logprob over the same scope."),
    (
        "Rollout Time (s/step)",
        "rollout_seconds",
        "generation_seconds of v{n-1}-generate/generated-batch.json (all server.generate calls); mean over effective updates.",
    ),
    (
        "Rollout Throughput (tok/active-GPU/s)",
        "rollout_tok_per_active_gpu_s",
        "response_tokens / Rollout Time / rollout GPUs, per step then mean.",
    ),
    (
        "Rollout Throughput (tok/8-GPU/s)",
        "rollout_tok_per_8gpu_s",
        "response_tokens / Rollout Time / 8, per step then mean.",
    ),
    (
        "Longest-sample Throughput (tok/s)",
        "longest_sample_tok_s",
        "Longest response length / Rollout Time (per-sample wall time is not recorded; the batch ends with the longest sample).",
    ),
    (
        "Actor Logprob Time (s/step)",
        "actor_logprob_seconds",
        "actor_logprob_seconds of training-parity-rank-*.json (max over ranks): actor recompute before PPO.",
    ),
    ("Reference Logprob Time (s/step)", "reference_logprob_seconds", "No reference model (use_kl_loss False)."),
    (
        "Actor Train Time (s/step)",
        "actor_train_seconds",
        "actor_train_seconds of updated-rank-*.json (max over ranks): native train_batch only.",
    ),
    ("Actor Train Throughput (tok/s)", "actor_train_tok_s", "response_tokens / Actor Train Time, per step then mean."),
    (
        "Total Train Time (s/step)",
        "total_train_seconds",
        "Actor Logprob Time + actor_train_and_checkpoint_seconds (train, checkpoint save, policy export; in-process).",
    ),
    (
        "Weight Sync Time (s/step)",
        "weight_sync_seconds",
        "Rollout phase non-generation time: engine start + policy load/sync + readback. Per step: "
        "v{n-1}-generate/execution.json seconds - generated-batch.json generation_seconds.",
    ),
    (
        "Offload/Restore Time (s/step)",
        "offload_restore_seconds",
        "Actor phase container start, checkpoint restore and teardown outside measured compute. Per step: "
        "update-{n}/execution.json seconds - max_r actor_logprob_seconds - max_r actor_train_and_checkpoint_seconds, "
        "clamped at >= 0 (a negative value is clamped and recorded as a warning).",
    ),
    (
        "Audit Time (s/step)",
        "audit_seconds",
        "Sum of container wall seconds (execution.json) of v{n}-post-generate, -actor-audit, -fixed-actor, "
        "-fixed-rollout; independent verifier runtime is not recorded.",
    ),
    (
        "End-to-End Step Time (s/step)",
        "e2e_seconds",
        "end_to_end_step_seconds of steps.json: whole step incl. stage switching, restore, sync and all audits.",
    ),
    ("Mean Response Length", "response_length", "response_tokens / samples of the training batch, mean over steps."),
    ("Mean Reward", "reward", "Mean raw_reward of the training batch samples, mean over steps."),
    ("Mean KL Loss", "kl_loss", "KL loss disabled (use_kl_loss False)."),
    (
        "Mean Policy Loss",
        "policy_loss",
        "Per update: sum of `value` over all update-{n}/native-loss-rank*-v{n}.json with a non-empty list "
        "(only last-PP-stage ranks compute the loss; ppo_loss is scaled -sum/global_batch_tokens*dp_size and "
        "DDP averages over DP) divided by the number of such files; all `finite` must be true, else FAIL. "
        "Arithmetic mean over effective DAPO steps.",
    ),
    (
        "Target Workload",
        "target_workload",
        "PASS only if the DAPO-Math-17k trajectory completed 3 independently verified updates.",
    ),
    (
        "Independent Verifier",
        "independent_verifier",
        "PASS only if fixed, every post-update (fresh+fixed) and update-verification are PASS.",
    ),
]
STATIC_NA = {
    "reference_logprob_seconds": "no reference model; use_kl_loss is False",
    "kl_loss": "KL loss not enabled (use_kl_loss False)",
}
PERF_KEYS = [
    "weight_sync_seconds",
    "offload_restore_seconds",
    "policy_loss",
    "rollout_seconds",
    "rollout_tok_per_active_gpu_s",
    "rollout_tok_per_8gpu_s",
    "longest_sample_tok_s",
    "actor_logprob_seconds",
    "actor_train_seconds",
    "actor_train_tok_s",
    "total_train_seconds",
    "audit_seconds",
    "e2e_seconds",
    "response_length",
    "reward",
]


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank_files(directory, prefix):
    """Receipts of `<prefix>-rank-<n>.json`, ordered by rank."""
    found = []
    for p in Path(directory).glob(prefix + "-rank-*.json"):
        tail = p.stem.rsplit("-", 1)[1]
        if tail.isdigit():
            found.append((int(tail), p))
    return [read(p) for _, p in sorted(found)]


def cell(value, reason=None):
    return dict(value=NA if value is None else value, reason=reason)


def fmt(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def weighted_mean(items):
    """items: (mean, weight); None if empty or any mean is missing."""
    if not items or any(m is None for m, _ in items):
        return None
    total = sum(w for _, w in items)
    return sum(m * w for m, w in items) / total if total else None


def parity_component(directory, warnings):
    ranks = rank_files(directory, "training-parity")
    if not ranks:
        return None
    first = ranks[0]
    # Every rank gathers all scores, so ranks must agree; the lowest rank is the logical receipt.
    if len({(r["expected_tokens"], r["compared_tokens"], r["mismatch_count"]) for r in ranks}) > 1:
        warnings.append(f"{directory}: training-parity rank receipts disagree; lowest rank used")
    rows = first.get("rows")
    return dict(
        kind="training_parity",
        status=PASS if not first["mismatch_count"] else FAIL,
        compared_tokens=first["compared_tokens"],
        mismatch_count=first["mismatch_count"],
        mean_abs_diff=weighted_mean([(r["mean_abs_diff"], r.get("compared_tokens", r["expected_tokens"])) for r in rows])
        if rows
        else None,
        max_abs_diff=max(r["max_abs_diff"] for r in rows) if rows else None,
    )


def verification_component(kind, value):
    if not value:
        return None
    return dict(
        kind=kind,
        status=value.get("status"),
        compared_tokens=value.get("compared_tokens"),
        mismatch_count=value.get("mismatch_count"),
        mean_abs_diff=value.get("mean_abs_diff"),
        max_abs_diff=value.get("max_abs_diff"),
    )


def cumulative(components):
    """Cumulative consistency cells; any missing component or field makes the cell N/A with a reason."""
    if not components:
        return {k: (None, "no update recorded") for k in ("compared_tokens", "mismatch_count", "mean_abs_diff", "max_abs_diff")}
    absent = [c for c in components if c.get("missing")]
    if absent:
        reason = "missing evidence: " + ", ".join(c["missing"] for c in absent)
        return {k: (None, reason) for k in ("compared_tokens", "mismatch_count", "mean_abs_diff", "max_abs_diff")}
    out = {}
    for key, fn in (
        ("compared_tokens", sum),
        ("mismatch_count", sum),
        ("max_abs_diff", max),
    ):
        vals = [c[key] for c in components]
        out[key] = (None, f"field {key} missing in a component") if any(v is None for v in vals) else (fn(vals), None)
    avg = weighted_mean([(c["mean_abs_diff"], c["compared_tokens"]) for c in components])
    out["mean_abs_diff"] = (avg, None if avg is not None else "mean_abs_diff missing or zero compared tokens")
    return out


def step_record(base, step, profile, warnings):
    n = step["update"]
    metrics = {k: None for k in PERF_KEYS}
    why = {}
    gen = read(base / f"v{n - 1}-generate" / "generated-batch.json")
    parity = parity_component(base / f"update-{n}", warnings)
    updated = rank_files(base / f"update-{n}", "updated")
    tokens = None
    if gen and gen.get("generation_seconds"):
        samples = gen["samples"]
        tokens, seconds = gen["response_tokens"], gen["generation_seconds"]
        metrics.update(
            rollout_seconds=seconds,
            rollout_tok_per_active_gpu_s=tokens / seconds / profile["rollout"]["gpus"],
            rollout_tok_per_8gpu_s=tokens / seconds / 8,
            longest_sample_tok_s=max(len(c["response_ids"]) for c in samples) / seconds,
            response_length=tokens / len(samples),
            reward=mean(c["raw_reward"] for c in samples),
        )
    else:
        for key in ("rollout_seconds", "rollout_tok_per_active_gpu_s", "rollout_tok_per_8gpu_s",
                    "longest_sample_tok_s", "response_length", "reward"):  # fmt: skip
            why[key] = f"v{n - 1}-generate/generated-batch.json missing or without generation_seconds"
    if parity is not None:
        ranks = rank_files(base / f"update-{n}", "training-parity")
        metrics["actor_logprob_seconds"] = max(r["actor_logprob_seconds"] for r in ranks)
    else:
        why["actor_logprob_seconds"] = f"update-{n}/training-parity-rank-*.json missing"
    if updated:
        metrics["actor_train_seconds"] = max(u["actor_train_seconds"] for u in updated)
        if tokens is not None and metrics["actor_train_seconds"] > 0:
            metrics["actor_train_tok_s"] = tokens / metrics["actor_train_seconds"]
        if metrics["actor_logprob_seconds"] is not None:
            metrics["total_train_seconds"] = metrics["actor_logprob_seconds"] + max(
                u["actor_train_and_checkpoint_seconds"] for u in updated
            )
    else:
        for key in ("actor_train_seconds", "actor_train_tok_s", "total_train_seconds"):
            why[key] = f"update-{n}/updated-rank-*.json missing"
    for key in ("actor_train_tok_s", "total_train_seconds"):
        if metrics[key] is None and key not in why:
            why[key] = "needs actor_logprob_seconds, response tokens and a positive train time"
    audit = [read(base / f"v{n}-{d}" / "execution.json") for d in ("post-generate", "actor-audit", "fixed-actor", "fixed-rollout")]
    if all(a and "seconds" in a for a in audit):
        metrics["audit_seconds"] = sum(a["seconds"] for a in audit)
    else:
        why["audit_seconds"] = f"execution.json missing in a v{n} audit phase directory"
    if "end_to_end_step_seconds" in step:
        metrics["e2e_seconds"] = step["end_to_end_step_seconds"]
    else:
        why["e2e_seconds"] = "end_to_end_step_seconds not recorded in steps.json"
    # Weight sync: rollout phase wall time minus the generation itself.
    rollout_exec = read(base / f"v{n - 1}-generate" / "execution.json")
    if rollout_exec and "seconds" in rollout_exec and gen and "generation_seconds" in gen:
        metrics["weight_sync_seconds"] = rollout_exec["seconds"] - gen["generation_seconds"]
    else:
        why["weight_sync_seconds"] = f"v{n - 1}-generate/execution.json or generation_seconds missing"
    # Offload/restore: actor phase wall time outside the measured compute.
    actor_exec = read(base / f"update-{n}" / "execution.json")
    if actor_exec and "seconds" in actor_exec and updated and metrics["actor_logprob_seconds"] is not None:
        value = actor_exec["seconds"] - metrics["actor_logprob_seconds"] - max(
            u["actor_train_and_checkpoint_seconds"] for u in updated
        )
        if value < 0:
            warnings.append(f"{base}: update-{n} offload/restore time {value:.6g}s is negative; clamped to 0")
            value = 0.0
        metrics["offload_restore_seconds"] = value
    else:
        why["offload_restore_seconds"] = f"update-{n}/execution.json, updated-rank or training-parity receipt missing"
    # Policy loss: mean over last-PP-stage ranks (non-empty lists) of the summed microbatch losses.
    losses = []
    for path in sorted((base / f"update-{n}").glob(f"native-loss-rank*-v{n}.json")):
        calls = read(path)
        if calls:
            losses.append(calls)
    if not losses:
        why["policy_loss"] = f"update-{n}/native-loss-rank*-v{n}.json missing or empty"
    elif not all(c.get("finite") is True for calls in losses for c in calls):
        metrics["policy_loss"] = FAIL
        why["policy_loss"] = f"non-finite native loss in update-{n}"
    else:
        metrics["policy_loss"] = sum(c["value"] for calls in losses for c in calls) / len(losses)
    fresh = read(base / f"v{n}-actor-audit" / "verification.json") or step.get("fresh_verification")
    fixed = read(base / f"v{n}-fixed-actor" / "verification.json") or step.get("fixed_verification")
    components = [
        parity or dict(missing=f"update-{n}/training-parity-rank-*.json"),
        verification_component("post_update_fresh", fresh) or dict(missing=f"v{n}-actor-audit/verification.json"),
        verification_component("post_update_fixed", fixed) or dict(missing=f"v{n}-fixed-actor/verification.json"),
    ]
    return dict(
        update=n,
        effective=step.get("post_update_fresh_and_fixed_verification") == PASS,
        response_tokens=tokens,
        metrics=metrics,
        why=why,
        components=components,
        fresh_status=(fresh or {}).get("status"),
        fixed_status=(fixed or {}).get("status"),
    )


def dataset_result(base, profile, warnings):
    base = Path(base)
    steps = read(base / "steps.json", [])
    records = [step_record(base, s, profile, warnings) for s in steps]
    used = [r for r in records if r["effective"]]
    perf = {}
    for key in PERF_KEYS:
        vals = [r["metrics"][key] for r in used]
        if not used:
            perf[key] = (None, "no effective (independently audited) update recorded")
        elif any(v == FAIL for v in vals):
            perf[key] = (FAIL, next(r["why"].get(key) for r in used if r["metrics"][key] == FAIL))
        elif any(v is None for v in vals):
            perf[key] = (None, next(r["why"].get(key) for r in used if r["metrics"][key] is None) or "value missing")
        else:
            perf[key] = (mean(vals), None)
    return dict(
        records=records,
        used=used,
        perf=perf,
        consistency=cumulative([c for r in used for c in r["components"]]),
        update_verification=read(base / "update-verification.json"),
        steps_json=steps,
    )


def group_cells(name, profile, state, group_dir, warnings):
    """All matrix cells of one group plus the copy-task summary."""
    status = state.get("status", "NOT_RUN")
    blocked = status in BLOCKED_STATES
    reason = state.get("reason") or state.get("error") or f"group status {status}"
    dapo = dataset_result(group_dir / "dapo", profile, warnings)
    copy = dataset_result(group_dir / "copy", profile, warnings)
    fixed = read(group_dir / "fixed-rollout" / "verification.json")
    cells = {
        "actor_topology": cell(f"{profile['actor']['pp']}/{profile['actor']['tp']}/{profile['actor']['ep']}"),
        "actor_gpus": cell(profile["actor"]["gpus"]),
        "rollout_tp": cell(profile["rollout"]["tp"]),
        "rollout_gpus": cell(profile["rollout"]["gpus"]),
    }
    for key, why in STATIC_NA.items():
        cells[key] = cell(None, why)

    def gated(key, compute):
        """BLOCKED groups never show numbers; otherwise compute() -> (value, reason)."""
        cells[key] = cell(BLOCKED, reason) if blocked else cell(*compute())

    gated("fixed_consistency", lambda: fixed_result(fixed, state))
    uv = dapo["update_verification"]
    gated("effective_steps", lambda: effective_steps(dapo, uv))
    for key in ("compared_tokens", "mismatch_count", "mean_abs_diff", "max_abs_diff"):
        gated(key, lambda key=key: dapo["consistency"][key])
    for key in PERF_KEYS:
        gated(key, lambda key=key: dapo["perf"][key])
    gated("target_workload", lambda: target_workload(dapo, uv, state))
    gated("independent_verifier", lambda: verifier_status(dapo, uv, fixed))
    copy_uv = copy["update_verification"]
    copy_cells = dict(
        fixed_weight=cells["fixed_consistency"]["value"],
        three_update=BLOCKED if blocked else update_state(copy, copy_uv),
        verified_updates=BLOCKED if blocked else (copy_uv or {}).get("completed_updates", NA),
        consistency={k: (BLOCKED if blocked else (v if v is not None else NA)) for k, (v, _) in copy["consistency"].items()},
        consistency_reasons={k: why for k, (_, why) in copy["consistency"].items() if why},
        scope="copy task (functional correctness): same cumulative scope as the DAPO consistency rows",
    )
    return cells, copy_cells, dapo, copy


def fixed_result(fixed, state):
    if fixed:
        return (PASS, None) if fixed.get("status") == PASS else (FAIL, fixed.get("error") or "fixed verification not PASS")
    if state.get("fixed_weight") == FAIL:
        return FAIL, "summary.json fixed_weight FAIL; no verification.json"
    return None, "fixed-rollout/verification.json absent"


def effective_steps(result, uv):
    if uv and uv.get("status") == PASS:
        return f"{uv['completed_updates']}/{UPDATES}", None
    ok = [r for r in result["used"] if r["fresh_status"] == PASS and r["fixed_status"] == PASS]
    if result["records"]:
        return f"{len(ok)}/{UPDATES}", "update-verification.json not PASS; counting per-step fresh+fixed verifier PASS only"
    return None, "no DAPO update recorded"


def update_state(result, uv):
    if uv:
        return PASS if uv.get("status") == PASS and uv.get("completed_updates") == UPDATES else FAIL
    return FAIL if result["records"] else NA


def target_workload(result, uv, state):
    if uv and uv.get("status") == PASS and uv.get("completed_updates") == UPDATES and len(result["used"]) == UPDATES:
        return PASS, None
    if state.get("target_load") == FAIL or uv or result["records"]:
        return FAIL, state.get("target_error") or (uv or {}).get("error") or "DAPO trajectory did not complete 3 verified updates"
    return None, "DAPO target workload not run"


def verifier_status(result, uv, fixed):
    statuses = [(fixed or {}).get("status")]
    for r in result["used"]:
        statuses += [r["fresh_status"], r["fixed_status"]]
    statuses.append((uv or {}).get("status"))
    if all(s == PASS for s in statuses) and len(result["used"]) == UPDATES:
        return PASS, None
    if not result["records"] and not fixed and not uv:
        return None, "no verification evidence"
    return FAIL, "some verification is missing or not PASS: " + ", ".join(str(s) for s in statuses)


def pick_runs(run_roots, prefer=None):
    """group -> (run_root, summary_entry); error if two runs selected a group unless --prefer picks one."""
    chosen, candidates = {}, {g: [] for g in GROUPS}
    roots = [Path(r).resolve() for r in run_roots]
    for root in roots:
        summary = read(root / "summary.json")
        if summary is None:
            raise ValueError(f"{root}: summary.json missing")
        policy = read(root / "resource-policy.json")
        names = set(policy["selected_groups"]) if policy and "selected_groups" in policy else {
            g for g, s in summary.items() if s.get("status") != "DEFERRED"
        }
        for g in GROUPS:
            if g in names and g in summary:
                candidates[g].append((root, summary[g]))
    preferred = None
    if prefer:
        preferred = [r for r in roots if r == Path(prefer).resolve() or r.name == prefer]
        if not preferred:
            raise ValueError(f"--prefer {prefer} is not one of the --run roots")
    for g, found in candidates.items():
        if len(found) > 1:
            pick = [f for f in found if preferred and f[0] == preferred[0]]
            if not pick:
                raise ValueError(f"group {g} selected in several runs ({', '.join(str(f[0]) for f in found)}); use --prefer RUN")
            found = pick
        if found:
            chosen[g] = found[0]
    return chosen


def evidence_files(root):
    wanted_names = {
        "verification.json", "scores.json", "update-verification.json", "steps.json", "result.json",
        "ep4-vs-ep1-actor.json", "execution.json",
    }  # fmt: skip
    for dirpath, _, files in os.walk(root):
        for name in sorted(files):
            if (
                name in wanted_names
                or fnmatch.fnmatch(name, "training-parity-rank-*.json")
                or fnmatch.fnmatch(name, "train-*-actor.f32")
                or fnmatch.fnmatch(name, "train-*-rollout.f32")
            ):
                yield Path(dirpath) / name


def index_files(root, scope):
    rows = []
    for p in sorted(evidence_files(scope)):
        rows.append(dict(path=p.relative_to(root).as_posix(), bytes=p.stat().st_size, sha256=sha256_of(p)))
    return rows


def commands(run_root, group):
    found = []
    for scope in (run_root / group, run_root / "common-fixed-actor"):
        for dirpath, _, files in os.walk(scope):
            if "command.json" in files:
                d = Path(dirpath)
                found.append(
                    dict(
                        path=(d / "command.json").relative_to(run_root).as_posix(),
                        command=read(d / "command.json"),
                        execution=read(d / "execution.json"),
                    )
                )
    return sorted(found, key=lambda x: x["path"])


def dataset_revision(lock):
    for holder in (lock.get("sources", {}), lock):
        rev = holder.get("datasets", {}).get("dapo", {}).get("revision")
        if rev:
            return rev
    raise ValueError("runtime lock has no datasets.dapo.revision")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def notes(revision):
    return [
        "Hardware: 1 node, 8x AMD Instinct MI300X.",
        "Model: DeepSeek V4, original layers 0-6.",
        "Framework: verl + RL-Kernel + vLLM.",
        f"Dataset: DAPO-Math-17k, pinned revision {revision}.",
        f"Workload: {ACTIVE_NOTE}.",
        "Prompt limit: 1024 tokens.",
        "Response limit: 7168 tokens.",
        "GRPO horizon: 3 effective updates per configuration.",
        "Consistency: strict FP32 bitwise comparison of all valid response tokens with independent post-update verification.",
        "Resource conditions: phase-wise GPU reuse and verified GPU0 sharing exception where applicable.",
        "Three-step short tests are not steady-state performance; no native baseline/speedup is claimed.",
        "Performance and training rows are arithmetic means over the 3 effective DAPO-Math-17k updates (the copy task is excluded).",
        "Compared Tokens / Mismatch Count / Mean Abs Diff / Max Abs Diff are cumulative over the DAPO trajectory: "
        "per update the training-batch parity, post-update fresh audit and post-update fixed audit "
        "(Mean Abs Diff weighted by compared tokens, Max = maximum); copy-task consistency is reported separately.",
        "Cells: measured value, N/A (evidence absent or not applicable; reasons in matrix.json), FAIL or BLOCKED.",
    ]


def render_md(columns, cells, copy_cells, revision):
    lines = [
        "### Seven-layer DeepSeek V4 — ROCm Performance & Consistency Matrix",
        "",
        "| Metric | " + " | ".join(columns) + " |",
        "| --- |" + " --- |" * len(columns),
    ]
    for label, key, _ in ROWS:
        lines.append(f"| {label} | " + " | ".join(fmt(cells[g][key]["value"]) for g in columns) + " |")
    lines += ["", "Notes:", ""] + [f"- {n}" for n in notes(revision)]
    lines += [
        "",
        "### Correctness (copy task)",
        "",
        "| Group | Fixed-weight | 3-update copy | Verified updates | Compared Tokens | Mismatch Count | Max Abs Diff |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for g in columns:
        c = copy_cells[g]
        k = c["consistency"]
        lines.append(
            f"| {g} | {fmt(c['fixed_weight'])} | {fmt(c['three_update'])} | {fmt(c['verified_updates'])} | "
            f"{fmt(k['compared_tokens'])} | {fmt(k['mismatch_count'])} | {fmt(k['max_abs_diff'])} |"
        )
    return "\n".join(lines) + "\n"


def aggregate(run_roots, output, runtime_lock, prefer=None):
    output, runtime_lock = Path(output), Path(runtime_lock)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"{output} exists and is not empty; refusing to overwrite")
    lock = json.loads(runtime_lock.read_text())
    revision = dataset_revision(lock)
    chosen = pick_runs(run_roots, prefer)
    output.mkdir(parents=True, exist_ok=True)
    targets = {p["id"]: p for p in profiles()}
    warnings, cells, copy_cells, groups = [], {}, {}, {}
    index, provenance_runs = {}, {}
    for g in GROUPS:
        if g in chosen:
            root, state = chosen[g]
            gdir = root / g
        else:
            root, gdir = None, Path("/nonexistent-run-root") / g
            state = dict(status="BLOCKED", reason="group not selected by any supplied run")
        cells[g], copy_cells[g], dapo, copy = group_cells(g, targets[g], state, gdir, warnings)
        groups[g] = dict(
            run=str(root) if root else None,
            summary=state,
            dapo_steps=len(dapo["records"]),
            copy_steps=len(copy["records"]),
            copy_task=copy_cells[g],
        )
        if root is None:
            continue
        for dataset, result in (("dapo", dapo), ("copy", copy)):
            if result["records"]:
                write_json(
                    output / f"steps-{g}-{dataset}.json",
                    dict(
                        group=g,
                        dataset=dataset,
                        run=str(root),
                        scope="per-step raw metrics (derived) plus the original steps.json receipts",
                        steps=[
                            dict(
                                update=r["update"],
                                effective=r["effective"],
                                response_tokens=r["response_tokens"],
                                metrics={k: (v if v is not None else NA) for k, v in r["metrics"].items()},
                                unavailable=r["why"],
                                consistency_components=r["components"],
                            )
                            for r in result["records"]
                        ],
                        receipts=result["steps_json"],
                    ),
                )
        index[g] = dict(run=str(root), files=index_files(root, gdir))
        provenance_runs.setdefault(root, []).append(g)
    run_index = {}
    for root in provenance_runs:
        shared = root / "common-fixed-actor"
        run_index[str(root)] = index_files(root, shared) if shared.exists() else []
    write_json(output / "evidence-index.json", dict(per_group=index, per_run_common=run_index))
    columns = GROUPS
    (output / "matrix.md").write_text(render_md(columns, cells, copy_cells, revision), encoding="utf-8")
    with open(output / "matrix.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric"] + columns)
        for label, key, _ in ROWS:
            writer.writerow([label.strip("*")] + [fmt(cells[g][key]["value"]) for g in columns])
    write_json(
        output / "matrix.json",
        dict(
            title="Seven-layer DeepSeek V4 — ROCm Performance & Consistency Matrix",
            columns=columns,
            dataset_revision=revision,
            notes=notes(revision),
            rows=[
                dict(metric=label.strip("*"), key=key, definition=definition, cells={g: cells[g][key] for g in columns})
                for label, key, definition in ROWS
            ],
            groups=groups,
            copy_task={g: copy_cells[g] for g in columns},
            warnings=warnings,
        ),
    )
    write_json(output / "provenance.json", provenance(lock, runtime_lock, provenance_runs, chosen, output, run_roots))
    return dict(cells=cells, copy_cells=copy_cells, warnings=warnings)


def provenance(lock, runtime_lock, runs, chosen, output, run_roots):
    runs_out, repro = [], []
    for root, names in runs.items():
        policy = read(root / "resource-policy.json", {})
        work = root.parent.parent
        runs_out.append(
            dict(
                run_root=str(root),
                groups=names,
                summary=read(root / "summary.json"),
                resource_policy=policy,
                health_receipt=read(root / "health-receipt.json"),
                provenance=read(root / "provenance.json"),
                launch_commands={g: commands(root, g) for g in names},
            )
        )
        gpus = policy.get("selected_physical_gpus", ["<gpus>"])
        repro.append(
            "python3 scripts/seven_layer/execute.py "
            f"--work {work} --image {lock.get('image_id')} --gpus {' '.join(map(str, gpus))} "
            f"--groups {' '.join(policy.get('selected_groups', names))} --runtime-lock {runtime_lock} "
            f"--health-receipt {root / 'health-receipt.json'}"
            + (f" --resource-policy {work}/<shared-policy.json>" if (root / "resource-policy-lock.json").exists() else "")
        )
    repro.append(
        "python3 scripts/seven_layer/aggregate_matrix.py --run "
        + " ".join(str(Path(r).resolve()) for r in run_roots)
        + f" --output {output} --runtime-lock {runtime_lock}"
    )
    return dict(
        runtime_lock=lock,
        runtime_lock_path=str(runtime_lock),
        runtime_lock_sha256=sha256_of(runtime_lock),
        image_id=lock.get("image_id"),
        source_commits=dict(recipe_commit=lock.get("recipe_commit"), sources=lock.get("sources")),
        group_run=dict((g, str(r)) for g, (r, _) in chosen.items()),
        runs=runs_out,
        reproduction_commands=repro,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", nargs="+", action="extend", required=True, type=Path, help="run roots (runs-N/<id>)")
    ap.add_argument("--output", type=Path, required=True, help="new, empty results directory")
    ap.add_argument("--runtime-lock", type=Path, required=True)
    ap.add_argument("--prefer", help="run root (or its name) that wins when a group appears in several runs")
    args = ap.parse_args()
    try:
        aggregate(args.run, args.output, args.runtime_lock, args.prefer)
    except (ValueError, FileExistsError) as error:
        sys.exit(f"error: {error}")


if __name__ == "__main__":
    main()
