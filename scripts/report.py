# SPDX-License-Identifier: MIT
"""Summarize retained evidence; never infer acceptance from process success."""

import csv
import json
import statistics
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def report(output):
    output = Path(output)
    expected = read(output / "plan.json")
    root = output / "policy-sync"
    verified = output / "verification.json"
    passed = verified.exists() and read(verified).get("status") == "PASS"
    rows = []
    for file in sorted(root.glob("metrics-step*.json"), key=lambda p: int(p.stem.split("step")[-1])):
        record = read(file)
        step = int(record["global_steps"])
        batch_file = root / f"training-batch-step{step}.json"
        audit_file = root / f"post-update-v{step}.json"
        if not batch_file.exists():
            continue
        batch = read(batch_file)
        audit = read(audit_file) if audit_file.exists() else {}
        t, m = record["timing_seconds"], record["metrics"]
        masks = batch["response_mask"]
        tokens = sum(sum(row) for row in masks)
        lengths = [sum(row) for row in masks]
        rewards = batch["rewards_per_response"]

        def rate(seconds, gpus=1, tokens=tokens):
            return tokens / seconds / gpus if seconds and seconds > 0 else None

        rows.append(
            dict(
                step=step,
                status="VERIFIED" if passed else "UNVERIFIED",
                response_tokens=tokens,
                responses=len(lengths),
                rollout_s=t.get("gen"),
                rollout_tokens_per_gpu_s=rate(t.get("gen"), expected["rollout_gpus"]),
                actor_logprob_s=t.get("old_log_prob"),
                reference_logprob_s=t.get("ref"),
                actor_train_s=t.get("update_actor"),
                actor_response_tokens_per_s=rate(t.get("update_actor")),
                weight_sync_s=t.get("update_weights"),
                independent_audit_s=audit.get("independent_audit_seconds"),
                end_to_end_s=t.get("step"),
                end_to_end_response_tokens_per_gpu_s=rate(t.get("step"), expected["total_gpus"]),
                mean_response_length=statistics.mean(lengths) if lengths else None,
                mean_reward=statistics.mean(rewards) if rewards else None,
                kl_loss=m.get("actor/kl_loss"),
                policy_loss=m.get("actor/pg_loss"),
                mismatch_count=audit.get("mismatch_count"),
                compared_audit_tokens=audit.get("compared_tokens"),
            )
        )
    if rows:
        with (output / "steps.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    selected = rows[3:] if len(rows) > 3 else rows
    averages = {
        key: statistics.mean([r[key] for r in selected if isinstance(r.get(key), (int, float))])
        for key in (
            "rollout_s",
            "rollout_tokens_per_gpu_s",
            "actor_logprob_s",
            "actor_train_s",
            "actor_response_tokens_per_s",
            "weight_sync_s",
            "independent_audit_s",
            "end_to_end_s",
            "mean_response_length",
            "mean_reward",
        )
        if any(isinstance(r.get(key), (int, float)) for r in selected)
    }
    summary = dict(
        status="PASS" if passed else "NOT_PASSED",
        requested_updates=expected["steps"],
        independently_verified_updates=expected["steps"] if passed else 0,
        recorded_metric_steps=len(rows),
        excluded_warmup_steps=3 if len(rows) > 3 else 0,
        averages=averages,
        limitations=[
            "Timings include evidence collection; step timing includes independent audits.",
            "Rollout time includes the native reward loop; actor throughput numerator is effective response tokens, not total training tokens.",
            "No native verl baseline, speedup, or longest-sample decoding speed is inferred.",
            "KL is disabled in this recipe and is reported as N/A.",
        ],
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        f"# Run {output.name}",
        "",
        f"Status: **{summary['status']}**. Independently verified updates: "
        f"{summary['independently_verified_updates']}/{expected['steps']}.",
        "",
        "| Metric | Single path arithmetic mean |",
        "| --- | --- |",
    ]
    lines += [f"| {k} | {v:.6g} |" for k, v in averages.items()]
    lines += [
        "| KL loss | N/A (disabled) |",
        "| Reference logprob | N/A (no reference policy) |",
        "| Longest-sample decode throughput | N/A (not instrumented) |",
        "",
        *["- " + x for x in summary["limitations"]],
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))
    return summary
