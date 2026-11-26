# SPDX-License-Identifier: MIT
"""Summarize retained revised-matrix evidence without converting missing data to zero."""

import csv
import json
from pathlib import Path
from statistics import mean

from matrix import profiles

NA = "N/A"


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def summarize(root):
    summary = read(root / "summary.json", {})
    tables = {}
    fixed_rows = []
    for profile in profiles():
        name = profile["id"]
        evidence = read(root / name / "fixed-rollout/verification.json", {})
        state = summary.get(name, {})
        fixed_rows.append(
            dict(
                group=name,
                fixed_weight_status=state.get("fixed_weight", "NOT_RUN"),
                independent_verifier=evidence.get("status", "NOT_RUN"),
                rank_and_weight_readback=evidence.get("rank_and_weight_readback", "NOT_RUN"),
                scope="fixed-weight scoring only; distinct from optimizer-update and generation metrics",
                **{
                    key: evidence.get(key, NA)
                    for key in [
                        "expected_tokens",
                        "compared_tokens",
                        "mismatch_count",
                        "mean_abs_diff",
                        "max_abs_diff",
                    ]
                },
            )
        )
    tables["fixed"] = fixed_rows
    with (root / "metrics-fixed.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(fixed_rows[0]))
        writer.writeheader()
        writer.writerows(fixed_rows)
    for dataset in ["copy", "dapo"]:
        rows = []
        for p in profiles():
            name = p["id"]
            base = root / name / dataset
            steps = read(base / "steps.json", [])
            checked = read(base / "update-verification.json", {})
            row = dict(
                group=name,
                actor_pp=p["actor"]["pp"],
                actor_tp=1,
                actor_ep=p["actor"]["ep"],
                dense_dp=p["actor"]["dp"],
                expert_dp=1,
                rollout_tp=p["rollout"]["tp"],
                status=summary.get(name, {}).get("status", "NOT_RUN"),
                fixed_weight_status=summary.get(name, {}).get("fixed_weight", "NOT_RUN"),
                dataset=dataset,
                recorded_optimizer_steps=len(steps),
                independently_verified_effective_updates=checked.get("completed_updates", NA),
                independent_verifier=checked.get("status", "NOT_RUN"),
            )
            for key in [
                "rollout_seconds",
                "rollout_tokens_per_second",
                "rollout_tokens_per_second_per_active_gpu",
                "rollout_tokens_per_second_per_machine_gpu",
                "actor_logprob_seconds",
                "actor_train_seconds",
                "actor_train_tokens_per_second",
                "actor_train_tokens_per_second_per_active_gpu",
                "actor_train_tokens_per_second_per_machine_gpu",
                "synchronization_seconds",
                "unload_restore_seconds",
                "audit_seconds",
                "end_to_end_step_seconds",
                "response_tokens",
                "response_length",
                "reward",
                "kl",
                "expected_tokens",
                "compared_tokens",
                "mismatch_count",
                "mean_abs_diff",
                "max_abs_diff",
            ]:
                row[key] = NA
            if steps:
                # Totals and rates include completed records only. Incomplete audits never become PASS.
                generated = [read(base / f"v{s['update'] - 1}-generate/generated-batch.json") for s in steps]
                train = [read(base / f"update-{s['update']}/updated-rank-1.json") for s in steps]
                parity = [read(base / f"update-{s['update']}/training-parity-rank-1.json") for s in steps]
                tokens = sum(g["response_tokens"] for g in generated)
                seconds = sum(g["generation_seconds"] for g in generated)
                train_seconds = sum(t["actor_train_seconds"] for t in train)
                row.update(
                    rollout_seconds=seconds,
                    rollout_tokens_per_second=tokens / seconds,
                    rollout_tokens_per_second_per_active_gpu=tokens / seconds / p["rollout"]["gpus"],
                    rollout_tokens_per_second_per_machine_gpu=tokens / seconds / 8,
                    actor_logprob_seconds=sum(v["actor_logprob_seconds"] for v in parity),
                    actor_train_seconds=train_seconds,
                    actor_train_tokens_per_second=tokens / train_seconds,
                    actor_train_tokens_per_second_per_active_gpu=tokens / train_seconds / p["actor"]["gpus"],
                    actor_train_tokens_per_second_per_machine_gpu=tokens / train_seconds / 8,
                    response_tokens=tokens,
                    response_length=tokens / sum(len(g["samples"]) for g in generated),
                    reward=mean(c["raw_reward"] for g in generated for c in g["samples"]),
                    expected_tokens=sum(v["expected_tokens"] for v in parity),
                    compared_tokens=sum(v["compared_tokens"] for v in parity),
                    mismatch_count=sum(v["mismatch_count"] for v in parity),
                    mean_abs_diff=sum(
                        r["mean_abs_diff"] * r["expected_tokens"] for v in parity for r in v["rows"]
                    )
                    / tokens,
                    max_abs_diff=max(r["max_abs_diff"] for v in parity for r in v["rows"]),
                )
                if all("end_to_end_step_seconds" in s for s in steps):
                    row["end_to_end_step_seconds"] = sum(s["end_to_end_step_seconds"] for s in steps)
                # Separate load/recovery times need direct instrumentation; no residual-time attribution.
            rows.append(row)
        tables[dataset] = rows
        with (root / f"metrics-{dataset}.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (root / "metrics.json").write_text(json.dumps(tables, indent=2) + "\n")
    lines = [
        "# Revised N matrix retained evidence",
        "",
        "Original M matrix superseded; GPU acceptance requires independent evidence.",
        "",
    ]
    for dataset, rows in tables.items():
        lines += [
            f"Dataset: {dataset}",
            "",
            "| Metric | N1 | N2 | N3 | N4 | N5 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for key in rows[0]:
            lines.append("| " + key + " | " + " | ".join(str(r[key]) for r in rows) + " |")
        lines.append("")
    lines += [
        "Counts are logical response tokens, never rank-multiplied. Rates divide by actual active GPUs or the whole eight-card machine. Timing totals cover retained updates; end-to-end includes phase release/recovery, synchronization and all post-update audits. KL disabled = N/A. Native training metrics remain in each native-train-output.pt; unavailable component timings remain N/A.",
        "",
    ]
    (root / "RESULTS.md").write_text("\n".join(lines))
