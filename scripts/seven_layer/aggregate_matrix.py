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
import struct
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
        "'<n>/3' independently verified effective updates of the DAPO trajectory (update-verification.json). "
        "A refused attempted update is never counted.",
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
        "Rollout Setup + Weight Sync Time (s/step)",
        "rollout_setup_sync_seconds",
        "Rollout phase wall time minus generation: engine start + policy load/weight sync + readback (not weight "
        "sync alone). Per step: v{n-1}-generate/execution.json seconds - generated-batch.json generation_seconds.",
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
    (
        "Mean Reward per Step",
        "reward_per_step",
        "Mean raw_reward of each effective update's training batch (update 1 / 2 / 3), then their arithmetic mean (= Mean Reward).",
    ),
    (
        "Truncation Rate",
        "truncation_rate",
        "Fraction of training-batch samples that hit the response cap without EOS: reward_evidence.truncated when recorded, else "
        "len(response_ids) == the request's max_tokens; mean over effective updates.",
    ),
    (
        "Advantage Statistics",
        "advantage_stats",
        "From reward-advantage-rank-0.json of each update (verl compute_grpo_outcome_advantage, group-std normalized, "
        "eps 1e-6; one advantage per sample): mean |A| and sample std of A (arithmetic means over effective updates) and "
        "nonzero advantages / samples (summed over effective updates). Recomputed from raw_rewards/group_ids with the "
        "same formula when the receipt lacks the statistics.",
    ),
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
        "DAPO Rollout Token SHA256",
        "dapo_rollout_sha256",
        "First 16 hex digits (full digest in matrix.json) of sha256 over the concatenated response_ids "
        "(uint32 little-endian, sample order) of the samples of dapo/v0-generate/generated-batch.json.",
    ),
    (
        "Cross-config Token/Reward Consistency",
        "cross_config",
        "Per group against the reference group (lowest N1-N4 present): v0 response-token sha256 and reward list equality, and "
        "for N1-N4 every update of the trajectory (response-token sha256, rewards, post-update model and optimizer digests of "
        "updated-rank-*.json checkpoint.model_state_digest / optimizer_state_digest; legacy state_digest-only receipts are "
        "reported as not comparable). N5 (Dense DP4 / EP4) is compared on v0 only; its trajectory is reported separately. "
        "Details in the Cross-config consistency section and matrix.json.",
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
TARGETS = {
    "dapo": dict(short="DAPO", name="DAPO-Math-17k"),
    "lcc": dict(short="LCC", name="Long-Context Copy (lcc)"),
}
LCC_TASK_NAME = "Long-Context Verifiable Sequence GRPO Benchmark (not math reasoning)"
SMOKE_DATASET = "lccsmoke"  # diagnostic feasibility set: never part of the formal matrix


def retarget(text, target):
    """Rename the target dataset in a label/definition ('DAPO-Math-17k' / 'DAPO') for a non-dapo target."""
    if target == "dapo":
        return text
    t = TARGETS[target]
    return text.replace("DAPO-Math-17k", t["name"]).replace("DAPO", t["short"])


def rows_for(target="dapo"):
    """Matrix rows (label, key, definition) with the target dataset named; identical to ROWS for dapo."""
    return [(retarget(label, target), key, retarget(definition, target)) for label, key, definition in ROWS]


STATIC_NA = {
    "reference_logprob_seconds": "no reference model; use_kl_loss is False",
    "kl_loss": "KL loss not enabled (use_kl_loss False)",
}
PERF_KEYS = [
    "rollout_setup_sync_seconds",
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
    "truncation_rate",
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


def grpo_advantages(rewards, groups, eps=1e-6):
    """verl compute_grpo_outcome_advantage (one scalar per sample): (r - group mean) / (unbiased group std + eps)."""
    members = {}
    for i, g in enumerate(groups):
        members.setdefault(g, []).append(i)
    out = [0.0] * len(rewards)
    for idx in members.values():
        values = [rewards[i] for i in idx]
        if len(values) == 1:
            center, spread = 0.0, 1.0
        else:
            center = sum(values) / len(values)
            spread = (sum((v - center) ** 2 for v in values) / (len(values) - 1)) ** 0.5
        for i in idx:
            out[i] = (rewards[i] - center) / (spread + eps)
    return out


def advantage_summary(directory):
    """dict(samples, nonzero, mean_abs, std, source) of one update's reward-advantage receipt; None if unavailable."""
    receipts = [r for r in rank_files(directory, "reward-advantage") if r]
    if not receipts:
        return None
    r = receipts[0]  # advantages are global and identical on every rank
    if "advantage_mean_abs" in r:
        return dict(
            samples=r["advantage_samples"], nonzero=r["advantage_nonzero_samples"],
            mean_abs=r["advantage_mean_abs"], std=r["advantage_std"], source="receipt",
        )  # fmt: skip
    raw, ids = r.get("raw_rewards"), r.get("group_ids")
    if not raw or not ids or len(raw) != len(ids):
        return None
    adv = grpo_advantages(raw, ids)
    n = len(adv)
    center = sum(adv) / n
    return dict(
        samples=n,
        nonzero=sum(1 for a in adv if abs(a) > 1e-9),
        mean_abs=sum(abs(a) for a in adv) / n,
        std=(sum((a - center) ** 2 for a in adv) / (n - 1)) ** 0.5 if n > 1 else 0.0,
        source="recomputed from raw_rewards/group_ids (same formula)",
    )


def advantage_total(records):
    """Aggregate over the given step records: mean of mean|A| and std, summed counts; None if any lacks the receipt."""
    got = [r.get("advantage") for r in records]
    if not got or any(a is None for a in got):
        return None
    return dict(
        updates=len(got),
        mean_abs=mean(a["mean_abs"] for a in got),
        std=mean(a["std"] for a in got),
        nonzero=sum(a["nonzero"] for a in got),
        samples=sum(a["samples"] for a in got),
        source=sorted({a["source"] for a in got}),
    )


def advantage_cell(result):
    total = result.get("advantage")
    if total is None:
        return None, "no reward-advantage receipt (or no raw_rewards/group_ids) for the effective updates"
    return f"mean|A| {total['mean_abs']:.4g}, std {total['std']:.4g}, nonzero {total['nonzero']}/{total['samples']}", None


def cell(value, reason=None):
    return dict(value=NA if value is None else value, reason=reason)


def show(c):
    """Display text of a matrix cell: value plus the dagger marker of an attempted (non-effective) step."""
    return fmt(c["value"]) + c.get("marker", "")


def rollout_digest(path):
    """(sha256 hex, samples, tokens) over concatenated response_ids (uint32 LE, sample order); None if absent."""
    gen = read(path)
    if not gen or not gen.get("samples"):
        return None
    digest, tokens = hashlib.sha256(), 0
    for sample in gen["samples"]:
        ids = sample["response_ids"]
        digest.update(struct.pack(f"<{len(ids)}I", *ids))
        tokens += len(ids)
    return digest.hexdigest(), len(gen["samples"]), tokens


def rollout_identity(cells, columns, label="DAPO"):
    """Cross-group target-dataset rollout token identity from the per-group digests (computed, never assumed)."""
    columns = [g for g in columns if cells[g]["dapo_rollout_sha256"]["value"] != BLOCKED]  # groups that were run
    found = {g: cells[g]["dapo_rollout_sha256"] for g in columns if cells[g]["dapo_rollout_sha256"].get("sha256")}
    if not found:
        return dict(status=NA, text=f"{label} rollout token identity: N/A (no generated-batch evidence)")
    digests = {c["sha256"] for c in found.values()}
    counts = {(c["samples"], c["tokens"]) for c in found.values()}
    head = f"{label} rollout token identity (sha256 over the concatenated response_ids of the samples, uint32 LE, sample order)"
    if len(digests) == 1 and len(counts) == 1 and len(found) == len(columns):
        samples, tokens = next(iter(counts))
        sha = next(iter(digests))
        return dict(
            status="MATCH", sha256=sha, samples=samples, tokens=tokens, groups=list(found),
            text=f"{head}: MATCH across {', '.join(found)}: {samples} samples, {tokens} tokens, sha256 {sha}",
        )  # fmt: skip
    status = "MISMATCH" if len(digests) > 1 or len(counts) > 1 else "PARTIAL"
    detail = "; ".join(f"{g}={c['sha256'][:16]} ({c['tokens']} tokens)" for g, c in found.items())
    missing = [g for g in columns if g not in found]
    return dict(
        status=status, per_group={g: c["sha256"] for g, c in found.items()}, missing=missing,
        text=f"{head}: {status}: {detail}" + (f"; no evidence: {', '.join(missing)}" if missing else ""),
    )  # fmt: skip


def footnotes(cells, columns):
    """Unique footnotes of cells that carry an attempted-step marker."""
    out = []
    for g in columns:
        for c in cells[g].values():
            if c.get("footnote") and c["footnote"] not in out:
                out.append(c["footnote"])
    return out + (cause_notes(cells, columns) if out else [])


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
            truncation_rate=truncation_rate(gen, profile),
            reward=mean(c["raw_reward"] for c in samples),
        )
    else:
        for key in ("rollout_seconds", "rollout_tok_per_active_gpu_s", "rollout_tok_per_8gpu_s",
                    "longest_sample_tok_s", "response_length", "reward", "truncation_rate"):  # fmt: skip
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
        metrics["rollout_setup_sync_seconds"] = rollout_exec["seconds"] - gen["generation_seconds"]
    else:
        why["rollout_setup_sync_seconds"] = f"v{n - 1}-generate/execution.json or generation_seconds missing"
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
    if step.get("attempted_error"):  # update refused: every absent value is absent because of the refusal
        why = {k: f"update {n} refused ({step['attempted_error']}); {v}" for k, v in why.items()}
    return dict(
        update=n,
        effective=step.get("post_update_fresh_and_fixed_verification") == PASS,
        attempted_error=step.get("attempted_error"),
        response_tokens=tokens,
        metrics=metrics,
        why=why,
        components=components,
        fresh_status=(fresh or {}).get("status"),
        fixed_status=(fixed or {}).get("status"),
        advantage=advantage_summary(base / f"update-{n}"),
    )


def dataset_result(base, profile, warnings, label="DAPO"):
    base = Path(base)
    steps = read(base / "steps.json", [])
    records = [step_record(base, s, profile, warnings) for s in steps]
    attempted = attempted_record(base, len(steps) + 1, profile, warnings, label)
    if attempted:
        records.append(attempted)
    used = [r for r in records if r["effective"]]  # an attempted record is never effective
    marked = set()
    perf = {}
    for key in PERF_KEYS:
        vals = [r["metrics"][key] for r in used]
        if not used and attempted:  # real values of the refused step, shown with a marker by the caller
            value = attempted["metrics"][key]
            perf[key] = (value, None) if value is not None else (None, attempted["why"].get(key) or "value missing")
            if value is not None:
                marked.add(key)
        elif not used:
            perf[key] = (None, "no effective (independently audited) update recorded")
        elif any(v == FAIL for v in vals):
            perf[key] = (FAIL, next(r["why"].get(key) for r in used if r["metrics"][key] == FAIL))
        elif any(v is None for v in vals):
            perf[key] = (None, next(r["why"].get(key) for r in used if r["metrics"][key] is None) or "value missing")
        else:
            perf[key] = (mean(vals), None)
    if used or not attempted:
        consistency = cumulative([c for r in used for c in r["components"]])
    else:  # training-batch parity of the attempted step only (one rank)
        consistency = cumulative([attempted["components"][0]])
        marked |= {k for k, (v, _) in consistency.items() if v is not None}
    advantage = advantage_total(used) if used else (advantage_total([attempted]) if attempted else None)
    if not used and advantage is not None:
        marked.add("advantage_stats")
    return dict(
        records=records,
        used=used,
        attempted=attempted,
        advantage=advantage,
        marked=marked,
        perf=perf,
        consistency=consistency,
        update_verification=read(base / "update-verification.json"),
        steps_json=steps,
    )


def attempted_record(base, k, profile, warnings, label="DAPO"):
    """Non-effective record of `update-{k}` that steps.json does not cover (refused or failed); else None."""
    directory = base / f"update-{k}"
    if not directory.is_dir():
        return None
    errors = sorted((int(p.stem.rsplit("-", 1)[1]), p) for p in directory.glob("error-*.json") if p.stem.rsplit("-", 1)[1].isdigit())
    error = (read(errors[0][1]) or {}).get("error") if errors else None
    if not error:
        return None  # no failure receipt: nothing is recorded as attempted
    record = step_record(base, dict(update=k, attempted_error=error), profile, warnings)
    record["effective"] = False
    adv = [r for r in rank_files(directory, "reward-advantage") if r]
    raw = adv[0].get("raw_rewards", []) if adv else []
    rewards = sorted(set(raw))
    reward_text = f"all {len(raw)} rewards {rewards[0]:g}" if len(rewards) == 1 else f"{len(raw)} rewards"
    nonzero = adv[0].get("nonzero_advantages") if adv else None
    guard = "refused by the zero-advantage guard" if nonzero == 0 else "failed"
    record["attempted_note"] = (
        f"\u2020 {label} update {k} measured through actor logprob + FP32 bitwise parity (training batch, one rank); "
        f"update {guard} ({reward_text}"
        + (f", nonzero advantages {nonzero}" if nonzero is not None else "")
        + f"; error: {error}); no actor train, policy loss, audit or end-to-end data. Rollout Setup + Weight Sync = "
        f"v{k - 1} rollout phase wall time minus generation (engine start + initial weight load). "
        "Not an effective GRPO update."
    )
    record["attempted_cause"] = attempted_cause(base, k, profile)
    return record


def attempted_cause(base, k, profile):
    """Evidence-derived facts about the rollout batch of the refused update (counts only what the files record)."""
    gen = read(base / f"v{k - 1}-generate" / "generated-batch.json")
    samples = (gen or {}).get("samples") or []
    if not samples:
        return None
    cap = ((gen.get("request") or {}).get("max_tokens")) or profile["target_load"]["response_cap"]
    at_cap = sum(1 for s in samples if len(s["response_ids"]) == cap)
    preds = [(s.get("reward_evidence") or {}).get("pred") for s in samples]
    invalid = sum(1 for p in preds if p == "[INVALID]")
    sources = sorted({s["reward_source"] for s in samples if s.get("reward_source")})
    has_pred = any(p is not None for p in preds)
    return dict(samples=len(samples), cap=cap, at_cap=at_cap, invalid=invalid, has_pred=has_pred, reward_sources=sources)


def cause_text(cause):
    text = f"{cause['at_cap']}/{cause['samples']} responses at the {cause['cap']}-token response cap"
    if cause.get("has_pred", True):  # math rewards only: the token-level lcc reward has no `pred`
        text += f"; {cause['invalid']}/{cause['samples']} samples with reward_evidence pred \"[INVALID]\""
    if cause["reward_sources"]:
        text += f"; reward source {', '.join(cause['reward_sources'])} (native reward function, unchanged)"
    return text


def cause_notes(cells, columns):
    """One line with the evidence-derived cause of the dagger; per group when the groups differ."""
    found = {}
    for g in columns:
        for c in cells[g].values():
            if c.get("cause"):
                found[g] = cause_text(c["cause"])
                break
    if not found:
        return []
    if len(set(found.values())) == 1:
        return [f"† Cause (from the recorded rollout batch, {', '.join(found)}): {next(iter(found.values()))}."]
    return ["† Cause (from the recorded rollout batch), per group: " + "; ".join(f"{g}: {t}" for g, t in found.items()) + "."]


def correctness_source(name, state):
    """(root, state) of the run supplying copy-task correctness when the group has `correctness_from`; else None."""
    ref = state.get("correctness_from")
    if not ref:
        return None
    root = Path(ref["run_root"])
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        raise ValueError(f"{name}: correctness run {root} has no summary.json")
    if sha256_of(summary_path) != ref["summary_sha256"]:
        raise ValueError(f"{name}: correctness run summary.json changed since the target-only run")
    other = json.loads(summary_path.read_text()).get(name, {})
    if other.get("status") != "CORRECTNESS_PASS":
        raise ValueError(f"{name}: correctness run status is {other.get('status')}, not CORRECTNESS_PASS")
    verification = root / name / "copy" / "update-verification.json"
    if not verification.is_file() or sha256_of(verification) != ref["update_verification_sha256"]:
        raise ValueError(f"{name}: correctness run update-verification.json missing or changed")
    return root, other


def group_cells(name, profile, state, group_dir, warnings, correctness=None, target="dapo"):
    """All matrix cells of one group plus the copy-task summary.

    correctness: (run_root, summary_entry) when copy-task evidence comes from another run.
    """
    status = state.get("status", "NOT_RUN")
    blocked = status in BLOCKED_STATES
    reason = state.get("reason") or state.get("error") or f"group status {status}"
    label = TARGETS[target]["short"]
    dapo = dataset_result(group_dir / target, profile, warnings, label)
    fixed = read(group_dir / "fixed-rollout" / "verification.json")
    if correctness:
        copy_dir = correctness[0] / name
        copy = dataset_result(copy_dir / "copy", profile, warnings)
        copy_fixed = fixed_result(read(copy_dir / "fixed-rollout" / "verification.json"), correctness[1])
    else:
        copy = dataset_result(group_dir / "copy", profile, warnings)
        copy_fixed = None
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
    gated("effective_steps", lambda: effective_steps(dapo, uv, label))
    for key in ("compared_tokens", "mismatch_count", "mean_abs_diff", "max_abs_diff"):
        gated(key, lambda key=key: dapo["consistency"][key])
    for key in PERF_KEYS:
        gated(key, lambda key=key: dapo["perf"][key])
    digest = rollout_digest(group_dir / target / "v0-generate" / "generated-batch.json")
    gated("dapo_rollout_sha256", lambda: (digest[0][:16], None) if digest else (None, f"{target}/v0-generate/generated-batch.json absent"))
    gated("advantage_stats", lambda: advantage_cell(dapo))
    gated("reward_per_step", lambda: reward_per_step_cell(dapo))
    cells["cross_config"] = cell(None, "computed across groups (aggregate): needs at least two groups with generation evidence")
    if digest and not blocked:
        cells["dapo_rollout_sha256"].update(sha256=digest[0], samples=digest[1], tokens=digest[2])
    if dapo["attempted"] and not blocked:
        for key in dapo["marked"]:  # effective_steps keeps its plain "n/3": a refused update is never counted
            cells[key].update(marker="†", footnote=dapo["attempted"]["attempted_note"], cause=dapo["attempted"].get("attempted_cause"))
    gated("target_workload", lambda: target_workload(dapo, uv, state, label))
    gated("independent_verifier", lambda: verifier_status(dapo, uv, fixed, label))
    copy_uv = copy["update_verification"]
    copy_cells = dict(
        fixed_weight=cells["fixed_consistency"]["value"] if copy_fixed is None else (BLOCKED if blocked else cell(*copy_fixed)["value"]),
        source_run=str(correctness[0]) if correctness else None,
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


def effective_steps(result, uv, label="DAPO"):
    if uv and uv.get("status") == PASS:
        return f"{uv['completed_updates']}/{UPDATES}", None
    ok = [r for r in result["used"] if r["fresh_status"] == PASS and r["fixed_status"] == PASS]
    if result["attempted"]:
        a = result["attempted"]
        return f"{len(ok)}/{UPDATES}", f"update {a['update']} attempted but refused: {a['attempted_error']}"
    if result["records"]:
        return f"{len(ok)}/{UPDATES}", "update-verification.json not PASS; counting per-step fresh+fixed verifier PASS only"
    return None, f"no {label} update recorded"


def update_state(result, uv):
    if uv:
        return PASS if uv.get("status") == PASS and uv.get("completed_updates") == UPDATES else FAIL
    return FAIL if result["records"] else NA


def target_workload(result, uv, state, label="DAPO"):
    if uv and uv.get("status") == PASS and uv.get("completed_updates") == UPDATES and len(result["used"]) == UPDATES:
        return PASS, None
    if state.get("target_load") == FAIL or uv or result["records"]:
        return FAIL, state.get("target_error") or (uv or {}).get("error") or f"{label} trajectory did not complete 3 verified updates"
    return None, f"{label} target workload not run"


def verifier_status(result, uv, fixed, label="DAPO"):
    statuses = [(fixed or {}).get("status")]
    for r in result["used"]:
        statuses += [r["fresh_status"], r["fixed_status"]]
    statuses.append((uv or {}).get("status"))
    if all(s == PASS for s in statuses) and len(result["used"]) == UPDATES:
        return PASS, None
    if not result["records"] and not fixed and not uv:
        return None, "no verification evidence"
    if not result["used"] and not uv and all(r.get("attempted_error") for r in result["records"]):
        # Nothing to verify: the only target update was refused before backward, so no verifier ran.
        # BLOCKED here means "prerequisite missing" (no effective update), the group itself did run.
        return BLOCKED, (
            f"no effective {label} update to verify (update refused); fixed-weight verifier "
            f"{(fixed or {}).get('status', 'missing')}; copy-task verifier: see the Correctness table"
        )
    return FAIL, "some verification is missing or not PASS: " + ", ".join(str(s) for s in statuses)


def is_smoke_run(root):
    """True for a run whose target dataset is the diagnostic lccsmoke set (and that has no formal lcc/dapo evidence)."""
    info = read(Path(root) / "target-only.json", {})
    return info.get("target_dataset") == SMOKE_DATASET


def matrix_target(chosen):
    """'lcc' when any chosen group run holds formal lcc evidence (the run's target dataset), else 'dapo'."""
    for g, (root, _) in chosen.items():
        if (root / g / "lcc").is_dir():
            return "lcc"
    return "dapo"


def pick_runs(run_roots, prefer=None):
    """group -> (run_root, summary_entry); error if two runs selected a group unless --prefer picks one."""
    chosen, candidates = {}, {g: [] for g in GROUPS}
    roots = [Path(r).resolve() for r in run_roots]
    for root in roots:
        summary = read(root / "summary.json")
        if summary is None:
            raise ValueError(f"{root}: summary.json missing")
        if is_smoke_run(root):  # diagnostic feasibility run: never part of the formal matrix
            continue
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
            if not pick and not preferred:  # the one run with formal lcc evidence supplies the target trajectory
                pick = [f for f in found if (f[0] / g / "lcc").is_dir()]
                pick = pick if len(pick) == 1 else []
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
    for dirpath, _, files in os.walk(root, followlinks=True):
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


def commands(run_root, group, only=None):
    found = []
    scopes = [run_root / group / s for s in only] if only else [run_root / group, run_root / "common-fixed-actor"]
    for scope in scopes:
        for dirpath, _, files in os.walk(scope, followlinks=True):
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


def row_ranges(rows):
    """[128, 129, 130, 132] -> '128-130, 132'."""
    parts = []
    for r in sorted(set(rows)):
        if parts and r == parts[-1][1] + 1:
            parts[-1][1] = r
        else:
            parts.append([r, r])
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in parts)


def dapo_prompt_rows(base):
    """{update: [source_row, ...]} from the recorded requests of dapo/v{n-1}-generate (stops at the first absent one)."""
    out, n = {}, 1
    while True:
        gen = read(Path(base) / f"v{n - 1}-generate" / "generated-batch.json")
        prompts = ((gen or {}).get("request") or {}).get("prompts")
        if not prompts:
            return out
        out[n] = [p["source_row"] for p in prompts if "source_row" in p]
        n += 1


def truncation_rate(gen, profile):
    """Fraction of samples that hit the response cap without EOS (reward_evidence.truncated, else length == max_tokens)."""
    samples = gen["samples"]
    cap = ((gen.get("request") or {}).get("max_tokens")) or profile["target_load"]["response_cap"]
    flags = []
    for s in samples:
        evidence = s.get("reward_evidence")
        if isinstance(evidence, dict) and "truncated" in evidence:
            flags.append(bool(evidence["truncated"]))
        else:
            flags.append(len(s["response_ids"]) == cap)
    return sum(flags) / len(flags)


def reward_per_step_cell(result):
    used = result["used"]
    if not used:
        return None, "no effective update recorded"
    values = [r["metrics"]["reward"] for r in used]
    if any(not isinstance(v, (int, float)) for v in values):
        return None, "reward missing for an effective update (generated-batch.json absent)"
    return " / ".join(f"{v:.6g}" for v in values) + f" (mean {mean(values):.6g})", None


DIGEST_FIELDS = {"model_digest": "model_state_digest", "optimizer_digest": "optimizer_state_digest"}
LEGACY_DIGEST = "not comparable (legacy digest includes run metadata)"


def _rank_digests(receipts, key):
    """{rank: checkpoint[key]} of the update receipts, or None unless every receipt records it."""
    out = {}
    for i, receipt in enumerate(receipts):
        value = ((receipt or {}).get("checkpoint") or {}).get(key)
        if not value:
            return None
        out[str((receipt or {}).get("rank", i))] = value
    return out or None


def trajectory_fingerprint(base):
    """Per update k: training-batch response-token sha256 + rewards (v{k-1}-generate) and post-update digests.

    Model/optimizer digests cover only the training state. The legacy checkpoint.state_digest also covers run
    provenance and process identity (pid, object ids, data pointers), so it is kept for display but never compared.
    """
    out, k = [], 1
    while True:
        path = Path(base) / f"v{k - 1}-generate" / "generated-batch.json"
        digest = rollout_digest(path)
        if not digest:
            return out
        samples = read(path)["samples"]
        receipts = rank_files(Path(base) / f"update-{k}", "updated")
        row = dict(
            update=k, tokens_sha256=digest[0], tokens=digest[2], rewards=[s["raw_reward"] for s in samples],
            state_digest=_rank_digests(receipts, "state_digest"),
        )  # fmt: skip
        for field, key in DIGEST_FIELDS.items():
            row[field] = _rank_digests(receipts, key)
        row["legacy_digest"] = row["state_digest"] is not None and row["model_digest"] is None
        out.append(row)
        k += 1


def _short(field, value):
    if field == "tokens_sha256":
        return value[:16]
    if field == "rewards":
        return f"{len(value)} rewards, mean {mean(value):.6g}, first {value[0]:.6g}"
    return {rank: digest[:16] for rank, digest in value.items()}


def compare_pair(a, b, fields):
    """(every per-update field comparison, number of compared fields) between two groups' fingerprints.

    Each entry has status MATCH, DIVERGES or NOT_COMPARABLE (a digest field where a side only has the legacy
    state_digest); fields absent on both sides for an update are omitted.
    """
    rows, compared = [], 0
    for ua, ub in zip(a, b):
        for field in fields:
            va, vb = ua.get(field), ub.get(field)
            if va is None or vb is None:
                if field in DIGEST_FIELDS and (ua.get("legacy_digest") or ub.get("legacy_digest")):
                    rows.append(dict(update=ua["update"], field=field, status="NOT_COMPARABLE", reason=LEGACY_DIGEST))
                continue
            compared += 1
            entry = dict(update=ua["update"], field=field, status="MATCH" if va == vb else "DIVERGES")
            if va != vb:
                entry.update(reference=_short(field, va), other=_short(field, vb))
            rows.append(entry)
    return rows, compared


def divergences(rows):
    return [r for r in rows if r["status"] == "DIVERGES"]


def divergence_text(rows):
    """'update 2: model_digest, optimizer_digest; update 3: tokens_sha256' over every divergent entry."""
    per_update = {}
    for r in divergences(rows):
        per_update.setdefault(r["update"], []).append(r["field"])
    return "; ".join(f"update {k}: {', '.join(v)}" for k, v in per_update.items())


def trajectory_status(rows, compared):
    if not compared:
        return NA
    if divergences(rows):
        return "DIVERGES"
    return "PARTIAL" if any(r["status"] == "NOT_COMPARABLE" for r in rows) else "MATCH"


TRAJECTORY_FIELDS = ("tokens_sha256", "rewards", "model_digest", "optimizer_digest")


def cross_config(fps):
    """Cross-configuration consistency from per-group fingerprints {group: trajectory_fingerprint}; all derived from evidence."""
    present = [g for g in GROUPS if fps.get(g)]
    if len(present) < 2:
        return dict(status=NA, groups=present, reason="needs generation evidence of at least two groups")
    ref = next((g for g in present if g != "N5"), present[0])
    v0 = {g: fps[g][0] for g in present}
    per_group_v0 = {
        g: dict(
            tokens_sha256=x["tokens_sha256"], tokens=x["tokens"], samples=len(x["rewards"]),
            reward_mean=mean(x["rewards"]), reward_min=min(x["rewards"]), reward_max=max(x["rewards"]),
        )  # fmt: skip
        for g, x in v0.items()
    }
    v0_info = dict(
        per_group=per_group_v0,
        tokens_all_match=len({x["tokens_sha256"] for x in v0.values()}) == 1,
        rewards_all_match=len({tuple(x["rewards"]) for x in v0.values()}) == 1,
        reward_distribution_all_match=len({tuple(sorted(x["rewards"])) for x in v0.values()}) == 1,
    )
    core = [g for g in present if g != "N5"]
    per_update = []
    for k in range(1, max(len(fps[g]) for g in core) + 1):
        row = dict(update=k)
        rows = [fps[g][k - 1] for g in core if len(fps[g]) >= k]
        for field in TRAJECTORY_FIELDS:
            values = {g: json.dumps(fps[g][k - 1].get(field), sort_keys=True) for g in core if len(fps[g]) >= k and fps[g][k - 1].get(field) is not None}
            if field in DIGEST_FIELDS and any(r.get("legacy_digest") for r in rows):
                row[field] = LEGACY_DIGEST
            else:
                row[field] = NA if len(values) < 2 else ("MATCH" if len(set(values.values())) == 1 else "DIVERGES")
            row[field + "_groups"] = sorted(values)
        per_update.append(row)
    compared_total, comparisons, all_rows = 0, {}, []
    for g in core:
        if g == ref:
            continue
        rows, compared = compare_pair(fps[ref], fps[g], TRAJECTORY_FIELDS)
        compared_total += compared
        comparisons[g] = dict(status=trajectory_status(rows, compared), compared_fields=compared, rows=rows)
        all_rows += [dict(r, group=g, reference_group=ref) for r in rows]
    order = lambda r: (r["update"], TRAJECTORY_FIELDS.index(r["field"]))  # noqa: E731
    found = sorted(divergences(all_rows), key=order)
    trajectory = dict(
        groups=core, reference=ref, per_update=per_update, compared_fields=compared_total,
        comparisons=comparisons, divergences=found, first_divergence=found[0] if found else None,
        status=NA if len(core) < 2 else trajectory_status(all_rows, compared_total),
    )  # fmt: skip
    n5 = None
    if "N5" in present:
        rows, compared = compare_pair(fps[ref], fps["N5"], ("tokens_sha256", "rewards"))
        found = divergences(rows)
        n5 = dict(
            reference=ref, compared_fields=compared, rows=rows, divergences=found,
            first_divergence=found[0] if found else None, status=trajectory_status(rows, compared),
            weight_digests={
                str(f["update"]): f.get("model_digest") or f.get("state_digest")
                for f in fps["N5"] if f.get("model_digest") or f.get("state_digest")
            },  # fmt: skip
            note="N5 (Dense DP4 / EP4) weight digests are per-rank EP4 states and are not compared with the EP1 digests; "
            "its train-vs-rollout bitwise parity at its own weight versions is in the matrix consistency rows.",
        )  # fmt: skip
    cells = {}
    for g in present:
        v_tok = "MATCH" if g == ref or per_group_v0[g]["tokens_sha256"] == per_group_v0[ref]["tokens_sha256"] else "MISMATCH"
        v_rew = "MATCH" if g == ref or tuple(v0[g]["rewards"]) == tuple(v0[ref]["rewards"]) else "MISMATCH"
        if g == ref:
            cells[g] = "reference"
        elif g == "N5":
            tail = f"; trajectory vs {ref}: {n5['status']} (tokens/rewards only; separate)" if n5 else ""
            cells[g] = f"v0 tokens {v_tok}, rewards {v_rew}{tail}"
        else:
            c = comparisons[g]
            traj = {
                NA: "N/A",
                "MATCH": "MATCH",
                "DIVERGES": f"DIVERGES at {divergence_text(c['rows'])}",
                "PARTIAL": f"MATCH on tokens/rewards; weight digests {LEGACY_DIGEST}",
            }[c["status"]]
            if c["status"] == "DIVERGES" and any(r["status"] == "NOT_COMPARABLE" for r in c["rows"]):
                traj += f" (weight digests {LEGACY_DIGEST})"
            cells[g] = f"v0 tokens {v_tok}, rewards {v_rew}, trajectory {traj}"
    return dict(status="OK", reference=ref, groups=present, v0=v0_info, trajectory=trajectory, n5=n5, cells=cells)


def compute_cross_config(chosen, target, cells):
    """Fingerprint every non-blocked chosen group, run cross_config and set each group's `cross_config` cell."""
    fps = {
        g: trajectory_fingerprint(root / g / target)
        for g, (root, state) in chosen.items()
        if state.get("status", "NOT_RUN") not in BLOCKED_STATES
    }
    result = cross_config(fps)
    for g in cells:
        if result["status"] == "OK" and g in result["cells"]:
            cells[g]["cross_config"] = cell(result["cells"][g])
        elif g in fps and result["status"] != "OK":
            cells[g]["cross_config"] = cell(None, result["reason"])
    return result


def cross_config_md(columns, cross):
    lines = ["", "### Cross-config consistency", ""]
    if cross["status"] != "OK":
        return lines + [f"N/A: {cross['reason']}.", ""]
    v0 = cross["v0"]
    lines += [
        "(a) v0 generation (same v0 weights, same sampling configuration): response-token sha256 over the concatenated response ids "
        "(uint32 LE, sample order) and reward statistics per group.",
        "",
        "| Group | v0 response-token sha256 | Tokens | Samples | Reward mean | min | max |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for g in columns:
        x = v0["per_group"].get(g)
        lines.append(
            f"| {g} | {x['tokens_sha256']} | {x['tokens']} | {x['samples']} | {fmt(x['reward_mean'])} | {fmt(x['reward_min'])} | {fmt(x['reward_max'])} |"
            if x else f"| {g} |" + " N/A |" * 6
        )
    lines += [
        "",
        f"v0 response tokens identical across {', '.join(v0['per_group'])}: {'MATCH' if v0['tokens_all_match'] else 'MISMATCH'}; "
        f"per-sample rewards identical: {'MATCH' if v0['rewards_all_match'] else 'MISMATCH'}; "
        f"reward distributions (sorted) identical: {'MATCH' if v0['reward_distribution_all_match'] else 'MISMATCH'}.",
        "",
    ]
    t = cross["trajectory"]
    lines += [
        f"(b) {', '.join(t['groups'])} (same Actor configuration; rollout expected bitwise identical to TP1): per update, "
        "training-batch response tokens (v{k-1} generation), rewards and post-update model / optimizer digests "
        "(updated-rank-*.json checkpoint.model_state_digest / optimizer_state_digest per rank; training state only). "
        "The legacy checkpoint.state_digest also covers run provenance and process identity and is never compared.",
        "",
        "| Update | Response tokens | Rewards | Model digests | Optimizer digests |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in t["per_update"]:
        lines.append(
            f"| {row['update']} | {row['tokens_sha256']} | {row['rewards']} | {row['model_digest']} | {row['optimizer_digest']} |"
        )
    if t["divergences"]:
        d = t["first_divergence"]
        lines += [
            "",
            f"Trajectory {t['status']}: first divergence at update {d['update']}, field {d['field']}. "
            f"All {len(t['divergences'])} divergent comparisons:",
        ]
        lines += [
            f"- update {d['update']} {d['field']}: {d['reference_group']}={d['reference']} vs {d['group']}={d['other']}"
            for d in t["divergences"]
        ]
    else:
        lines += ["", f"Trajectory {t['status']} across {', '.join(t['groups'])} ({t['compared_fields']} field comparisons against {t['reference']})."]
    if any(r["status"] == "NOT_COMPARABLE" for c in t["comparisons"].values() for r in c["rows"]):
        lines += ["", f"Weight digests: {LEGACY_DIGEST}; these receipts predate model/optimizer digests."]
    n5 = cross.get("n5")
    lines += ["", "(c) N5 (reported separately):"]
    if n5:
        lines.append(
            f"- vs {n5['reference']}: tokens/rewards {n5['status']}"
            + (f"; first divergence at update {n5['first_divergence']['update']}, field {n5['first_divergence']['field']}" if n5["first_divergence"] else "")
            + f". {n5['note']}"
        )
    else:
        lines.append("- N/A: N5 has no generation evidence in the supplied runs.")
    return lines + [""]


def lcc_sources(base):
    """{update: request source dict} of the recorded lcc rollout requests (stops at the first absent one)."""
    out, n = {}, 1
    while True:
        gen = read(Path(base) / f"v{n - 1}-generate" / "generated-batch.json")
        source = ((gen or {}).get("request") or {}).get("source")
        if not source:
            return out
        out[n] = source
        n += 1


def passage_rows_text(source):
    """'passage 0: rows 1024-1027 (last partial)' per passage of one recorded request source."""
    parts = []
    for p in source.get("passages", []):
        text = f"passage {p['passage']}: rows {row_ranges(p['source_rows'])}"
        parts.append(text + (" (last row partial)" if p.get("partial_last_row") else ""))
    return ", ".join(parts)


LENGTH_THRESHOLDS = (2048, 4096, 8192)


def percentile(values, q):
    ordered = sorted(values)
    return ordered[max(0, -(-len(ordered) * q // 100) - 1)]  # nearest rank


def length_coverage(base):
    """Prompt+response length distribution of the real generated training batches of one target trajectory."""
    totals, prompts, responses, updates, n = [], [], [], 0, 1
    while True:
        gen = read(Path(base) / f"v{n - 1}-generate" / "generated-batch.json")
        if not gen or not gen.get("samples"):
            break
        if any("prompt_ids" not in s for s in gen["samples"]):
            return None
        for s in gen["samples"]:
            prompts.append(len(s["prompt_ids"]))
            responses.append(len(s["response_ids"]))
            totals.append(prompts[-1] + responses[-1])
        updates += 1
        n += 1
    if not totals:
        return None
    return dict(
        updates=updates, samples=len(totals), min_total=min(totals), mean_total=sum(totals) / len(totals),
        p50_total=percentile(totals, 50), p90_total=percentile(totals, 90), max_total=max(totals),
        max_prompt=max(prompts), max_response=max(responses),
        above={str(t): sum(1 for x in totals if x > t) for t in LENGTH_THRESHOLDS},
    )  # fmt: skip


def fixed_suite_coverage(group_roots):
    """Operator coverage of the independent fixed-weight audit suite: [(group, run_root)] -> rows with real lengths."""
    rows, statuses, sha, suite = [], {}, None, None
    for g, root in group_roots:
        verification = read(Path(root) / g / "fixed-rollout" / "verification.json")
        if not verification or not verification.get("rows"):
            continue
        statuses[g] = {r["id"]: r for r in verification["rows"]}
        if not rows:
            rows = [r["id"] for r in verification["rows"]]
            sha = verification.get("input_suite_sha256")
            path = Path(root).parent.parent / "evidence" / "fixed-input-suite.json"
            if path.is_file() and sha256_of(path) == sha:
                suite = {c["id"]: c for c in read(path)["samples"]}
    if not rows:
        return None
    first = next(iter(statuses.values()))
    out = []
    for rid in rows:
        case = (suite or {}).get(rid)
        prompt = len(case["prompt_ids"]) if case else None
        response = first[rid].get("expected_tokens")
        out.append(
            dict(
                id=rid, prompt_tokens=prompt, response_tokens=response,
                total_tokens=prompt + response if prompt is not None and response is not None else None,
                status={g: s[rid]["status"] if rid in s else NA for g, s in statuses.items()},
                mismatch={g: s[rid].get("mismatch_count") if rid in s else NA for g, s in statuses.items()},
            )  # fmt: skip
        )
    return dict(
        input_suite_sha256=sha,
        lengths_source="evidence/fixed-input-suite.json (sha256 matches the verification)" if suite else "unavailable: suite file absent or hash differs",
        rows=out,
    )


def gpu_note(columns):
    targets = {p["id"]: p for p in profiles()}
    return "Active GPUs per phase (profile): " + "; ".join(
        f"{g} actor {targets[g]['actor']['gpus']} (PP/TP/EP {targets[g]['actor']['pp']}/{targets[g]['actor']['tp']}/{targets[g]['actor']['ep']}), "
        f"rollout {targets[g]['rollout']['gpus']}"
        for g in columns
    ) + "; phases reuse the same physical GPUs sequentially, so per-active-GPU rates divide by the phase's own GPU count and per-8-GPU rates by 8."


def evidence_notes(per_group, target="dapo"):
    """Notes derived from the retained evidence (never hardcoded per group).

    per_group: {group: dict(effective=n, attempted=update or None, prompt_rows={update: [rows]})} for run groups.
    """
    out = []
    if any(v["effective"] < UPDATES for v in per_group.values()):
        text = f"Achieved: {TARGETS[target]['short']} effective updates " + ", ".join(f"{g} {v['effective']}/{UPDATES}" for g, v in per_group.items())
        refused = {g: v["attempted"] for g, v in per_group.items() if v["attempted"]}
        if refused:
            text += (
                "; values marked † come from the refused attempted update ("
                + ", ".join(f"{g} update {k}" for g, k in refused.items())
                + "; training-batch parity only, no post-update audit)"
            )
        out.append(text + ".")
    if target == "lcc":
        sources = {g: v["lcc_sources"] for g, v in per_group.items() if v.get("lcc_sources")}
        if sources:

            def render_lcc(by_update):
                return "; ".join(f"update {n}: {passage_rows_text(s)}" for n, s in sorted(by_update.items()))

            head = "LCC passage source (pinned DAPO-Math-17k parquet rows of the recorded rollout requests): "
            if len({json.dumps(r, sort_keys=True) for r in sources.values()}) == 1:
                out.append(head + render_lcc(next(iter(sources.values()))) + f" (identical in {', '.join(sources)}).")
            else:
                out.append(head + "; ".join(f"{g}: {render_lcc(r)}" for g, r in sources.items()) + ".")
        return out
    selections = {g: v["prompt_rows"] for g, v in per_group.items() if v["prompt_rows"]}
    if selections:

        def render(rows):
            return "; ".join(f"update {n}: source rows {row_ranges(r)}" for n, r in sorted(rows.items()))

        head = (
            "DAPO prompt selection (source_row of the recorded rollout requests; prepare_requests.py skips the first "
            "128 rows of the pinned parquet and any prompt over the prompt cap, keeps file order and cycles the "
            "selected prompts): "
        )
        if len({json.dumps(r, sort_keys=True) for r in selections.values()}) == 1:
            out.append(head + render(next(iter(selections.values()))) + f" (identical in {', '.join(selections)}).")
        else:
            out.append(head + "; ".join(f"{g}: {render(r)}" for g, r in selections.items()) + ".")
    return out


def notes(revision, evidence=(), target="dapo", columns=GROUPS, lcc=None):
    common = [
        "Hardware: 1 node, 8x AMD Instinct MI300X.",
        "Model: DeepSeek V4, original layers 0-6.",
        "Framework: verl + RL-Kernel + vLLM.",
        f"Dataset: DAPO-Math-17k, pinned revision {revision}.",
        f"Workload: {ACTIVE_NOTE}.",
        "Prompt limit: 1024 tokens.",
        "Response limit: 7168 tokens.",
        "GRPO horizon (target): 3 effective updates per configuration; see Achieved below when a group fell short.",
        "Consistency (target design): strict FP32 bitwise comparison of all valid response tokens with independent post-update verification of each effective update.",
        "Resource conditions: phase-wise GPU reuse and verified GPU0 sharing exception where applicable.",
        "Three-step short tests are not steady-state performance; no native baseline/speedup is claimed.",
        "Performance and training rows (target) are arithmetic means over the 3 effective DAPO-Math-17k updates (the copy task is excluded); a group without effective updates shows the values of its refused attempt, marked.",
        "Compared Tokens / Mismatch Count / Mean Abs Diff / Max Abs Diff are cumulative over the DAPO trajectory: "
        "per update the training-batch parity, post-update fresh audit and post-update fixed audit "
        "(Mean Abs Diff weighted by compared tokens, Max = maximum); copy-task consistency is reported separately.",
        "Cells: measured value, N/A (evidence absent or not applicable; reasons in matrix.json), FAIL or BLOCKED "
        "(group not run, or a prerequisite is missing, e.g. Independent Verifier is BLOCKED when there is no effective DAPO update to verify; reason in matrix.json).",
        "Longest-sample Throughput assumes the batch generation time equals the longest sample's time (per-sample wall time is not recorded), "
        "so it is a lower bound of that sample's real rate.",
        "A dagger marks a value of an attempted but refused (non-effective) DAPO update: it is never averaged into effective-step "
        "means and never counted as an effective GRPO step.",
    ]
    if target != "dapo":
        common = [retarget(n, target) for n in common if not n.startswith("Dataset:")]
        sha = (lcc or {}).get("parquet_sha256")
        lcc_notes = [
            f"Target task: {LCC_TASK_NAME}, dataset key lcc: the policy must copy a passage of at most 1024 prompt tokens "
            "(user message \"Copy the following text exactly, without any explanation:\" + passage). It is a verifiable "
            "sequence-reproduction workload for validating the GRPO system, not a math or reasoning task.",
            "Reward (deterministic, reference-based, no shaping): 1 - EditDistance(out_ids, target_ids) / max(len(out_ids), "
            "len(target_ids), 1) over the generated TOKEN ids (one trailing EOS stripped, every other token incl. special "
            "tokens kept); target_ids = tokenizer encoding of the passage (add_special_tokens=False); score in [0, 1]; "
            "implemented in RL-Kernel examples/operators/dsv4_4l/long_context_copy.py (exact bit-parallel Levenshtein).",
            f"Passage source: consecutive user-message texts of the pinned DAPO-Math-17k revision {revision} "
            "(parquet rows from 1024 in file order, joined by a blank line, longest prefix within the 1024-token prompt, "
            "text only, no math task is used)" + (f"; parquet sha256 {sha}" if sha else "") + "; per-passage source rows below.",
            "Seven-layer rewards are chance-level: the model (original layers 0-6 only) has no language capability, so its "
            "responses are not copies. This run validates the GRPO system (nonzero advantages, real gradients, updated "
            "parameters), strict FP32 consistency and performance, not learning or task quality.",
        ]
        common = common[:3] + lcc_notes + common[3:]
    common += [
        "Units: times in seconds, throughput in tokens/second, token counts are logical response/compared tokens "
        "(never multiplied by ranks or DP replicas); logprob differences in nats.",
        gpu_note(columns),
        "Aggregation: per-step values are means over the effective updates (3 per configuration); consistency token and mismatch "
        "counts are cumulative sums over the update's training-batch parity and post-update audits (one rank's receipt); "
        "Advantage Statistics: mean |A| and std are means over updates, nonzero/total are summed.",
    ]
    return common + list(evidence)


def render_md(columns, cells, copy_cells, revision, evidence=(), target="dapo", lcc=None, coverage=None, cross=None):
    lines = [
        "### Seven-layer DeepSeek V4 — ROCm Performance & Consistency Matrix",
        "",
        "| Metric | " + " | ".join(columns) + " |",
        "| --- |" + " --- |" * len(columns),
    ]
    for label, key, _ in rows_for(target):
        lines.append(f"| {label} | " + " | ".join(show(cells[g][key]) for g in columns) + " |")
    lines += ["", rollout_identity(cells, columns, TARGETS[target]["short"])["text"]]
    lines += [""] + footnotes(cells, columns)
    lines += ["", "Notes:", ""] + [f"- {n}" for n in notes(revision, evidence, target, columns, lcc)]
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
    lines += cross_config_md(columns, cross) if cross else []
    lines += coverage_md(columns, coverage, target) if coverage else []
    return "\n".join(lines) + "\n"


def coverage_md(columns, coverage, target):
    """Long-context coverage: real generated training lengths, kept apart from the audit-suite operator coverage."""
    lines = ["", "### Long-context coverage", ""]
    lines += [
        f"(a) Real generated training tokens of the {TARGETS[target]['short']} trajectory (prompt + response tokens per sample, "
        "all recorded training batches):",
        "",
        "| Group | Updates | Samples | Min | Mean | p50 | p90 | Max total | Max prompt | Max response | > 2048 | > 4096 | > 8192 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for g in columns:
        c = (coverage.get("training") or {}).get(g)
        if not c:
            lines.append(f"| {g} |" + " N/A |" * 12)
            continue
        lines.append(
            f"| {g} | {c['updates']} | {c['samples']} | {c['min_total']} | {c['mean_total']:.1f} | {c['p50_total']} | {c['p90_total']} | "
            f"{c['max_total']} | {c['max_prompt']} | {c['max_response']} | "
            + " | ".join(str(c["above"][str(t)]) for t in LENGTH_THRESHOLDS)
            + " |"
        )
    lines += [
        "",
        "(b) Operator coverage by the independent fixed-weight long-sequence audit suite (fixed audit inputs; not generated "
        "training tokens and never part of any throughput):",
        "",
    ]
    suite = coverage.get("fixed_suite")
    if not suite:
        return lines + ["N/A: no fixed-rollout verification rows recorded.", ""]
    names = list(suite["rows"][0]["status"]) if suite["rows"] else []
    lines += [
        "| Suite row | Prompt tokens | Response tokens | Total tokens | " + " | ".join(names) + " |",
        "| --- | --- | --- | --- |" + " --- |" * len(names),
    ]
    for r in suite["rows"]:
        lines.append(
            f"| {r['id']} | {fmt(r['prompt_tokens'] if r['prompt_tokens'] is not None else NA)} | {fmt(r['response_tokens'] if r['response_tokens'] is not None else NA)} | "
            f"{fmt(r['total_tokens'] if r['total_tokens'] is not None else NA)} | " + " | ".join(str(r["status"][g]) for g in names) + " |"
        )
    return lines + ["", f"Suite lengths: {suite['lengths_source']}.", ""]


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
    index, provenance_runs, correctness_runs, per_group = {}, {}, {}, {}
    target = matrix_target(chosen)
    training_cov = {}
    for g in GROUPS:
        if g in chosen:
            root, state = chosen[g]
            gdir = root / g
        else:
            root, gdir = None, Path("/nonexistent-run-root") / g
            state = dict(status="BLOCKED", reason="group not selected by any supplied run")
        source = correctness_source(g, state) if root else None
        cells[g], copy_cells[g], dapo, copy = group_cells(g, targets[g], state, gdir, warnings, source, target)
        copy_root = source[0] if source else root
        groups[g] = dict(
            run=str(root) if root else None,
            summary=state,
            target_dataset=target,
            dapo_steps=len(dapo["records"]),
            copy_steps=len(copy["records"]),
            copy_task=copy_cells[g],
            parts=dict(
                fixed_weight_row=str(root) if root else None,
                dapo_target=str(root) if root else None,
                copy_correctness=str(copy_root) if root else None,
                copy_fixed_weight_column=str(copy_root) if root else None,
            ),
        )
        if root is None:
            continue
        extra = []
        for dataset, result, owner in ((target, dapo, root), ("copy", copy, copy_root), *extra):
            if result["records"]:
                write_json(
                    output / f"steps-{g}-{dataset}.json",
                    dict(
                        group=g,
                        dataset=dataset,
                        run=str(owner),
                        scope="per-step raw metrics (derived) plus the original steps.json receipts",
                        steps=[
                            dict(
                                update=r["update"],
                                effective=r["effective"],
                                attempted_error=r.get("attempted_error"),
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
        if source:
            index[g]["correctness_run"] = str(source[0])
            index[g]["correctness_files"] = index_files(source[0], source[0] / g / "copy")
            correctness_runs.setdefault(source[0], []).append(g)
        provenance_runs.setdefault(root, []).append(g)
        if state.get("status", "NOT_RUN") not in BLOCKED_STATES:
            per_group[g] = dict(
                effective=len(dapo["used"]),
                attempted=dapo["attempted"]["update"] if dapo["attempted"] else None,
                prompt_rows=dapo_prompt_rows(gdir / "dapo") if target == "dapo" else {},
                lcc_sources=lcc_sources(gdir / "lcc") if target == "lcc" else {},
            )
            training_cov[g] = length_coverage(gdir / target)
    run_index = {}
    for root in provenance_runs:
        shared = root / "common-fixed-actor"
        run_index[str(root)] = index_files(root, shared) if shared.exists() else []
    write_json(output / "evidence-index.json", dict(per_group=index, per_run_common=run_index))
    columns = GROUPS
    cross = compute_cross_config(chosen, target, cells)
    evidence = evidence_notes(per_group, target)
    pending = [g for g in GROUPS if g not in per_group]
    if pending and per_group:
        evidence.append(f"Not run in the supplied runs (columns BLOCKED): {', '.join(pending)}.")
        eight = [g for g in pending if targets[g]["peak_physical_gpus"] == 8]
        if eight:
            evidence.append(
                f"{', '.join(eight)} need all 8 GPUs (rollout TP8 / 8-GPU EP4 actor); they are never run on a smaller GPU pool "
                "and TP8/EP4 is never emulated, so they stay pending until all 8 GPUs are free."
            )
    first_source = next((s for v in per_group.values() for s in v.get("lcc_sources", {}).values()), {})
    lcc_info = dict(parquet_sha256=first_source.get("parquet_sha256"), source=first_source)
    coverage = dict(
        training={g: c for g, c in training_cov.items() if c},
        fixed_suite=fixed_suite_coverage([(g, chosen[g][0]) for g in chosen if chosen[g][1].get("status") not in BLOCKED_STATES]),
    )
    (output / "matrix.md").write_text(
        render_md(columns, cells, copy_cells, revision, evidence, target, lcc_info, coverage, cross), encoding="utf-8"
    )
    with open(output / "matrix.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric"] + columns)
        for label, key, _ in rows_for(target):
            writer.writerow([label.strip("*")] + [show(cells[g][key]) for g in columns])
    write_json(
        output / "matrix.json",
        dict(
            title="Seven-layer DeepSeek V4 — ROCm Performance & Consistency Matrix",
            columns=columns,
            dataset_revision=revision,
            notes=notes(revision, evidence, target, columns, lcc_info),
            target_dataset=target,
            target_task=LCC_TASK_NAME if target == "lcc" else "DAPO-Math-17k",
            rows=[
                dict(metric=label.strip("*"), key=key, definition=definition, cells={g: cells[g][key] for g in columns})
                for label, key, definition in rows_for(target)
            ],
            groups=groups,
            long_context_coverage=coverage,
            cross_config=cross,
            dapo_rollout_identity=rollout_identity(cells, columns, TARGETS[target]["short"]),
            footnotes=footnotes(cells, columns),
            copy_task={g: copy_cells[g] for g in columns},
            warnings=warnings,
        ),
    )
    write_json(
        output / "provenance.json",
        provenance(lock, runtime_lock, provenance_runs, chosen, output, run_roots, correctness_runs, target),
    )
    return dict(cells=cells, copy_cells=copy_cells, warnings=warnings)


def provenance(lock, runtime_lock, runs, chosen, output, run_roots, correctness_runs=None, target="dapo"):
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
            + (f" --target-only --correctness-run {read(root / 'target-only.json', {}).get('correctness_run')} --target-dataset {read(root / 'target-only.json', {}).get('target_dataset', 'dapo')}" if (root / "target-only.json").exists() else "")
        )
    for root, names in (correctness_runs or {}).items():
        runs_out.append(
            dict(
                run_root=str(root),
                role="copy-task correctness source (fixed-weight + copy three updates)",
                groups=names,
                summary=read(root / "summary.json"),
                health_receipt=read(root / "health-receipt.json"),
                provenance=read(root / "provenance.json"),
                launch_commands={g: commands(root, g, only=["copy", "fixed-rollout"]) for g in names},
            )
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
        group_parts={
            g: dict(
                fixed_weight_row=str(r),
                dapo_target=str(r),
                target_dataset=target,
                target_run=str(r),
                copy_correctness=(s.get("correctness_from") or {}).get("run_root", str(r)),
                correctness_from=s.get("correctness_from"),
                resume=s.get("resume"),
            )
            for g, (r, s) in chosen.items()
        },
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
