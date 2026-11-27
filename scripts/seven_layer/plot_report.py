# SPDX-License-Identifier: MIT
"""PNG report (performance matrix, stage summary, trajectories, consistency) for the seven-layer matrix.

Same parsed data as wandb_report.py (report_data.py -> aggregate_matrix.py). Writes rounds.csv (one row per
group x task x step), summary.json and the PNGs. Values that evidence lacks are absent from plots and rows.
"""

import argparse
import csv
import json
import sys
import textwrap
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_data as rd  # noqa: E402

COLORS = {"N1": "#0072B2", "N2": "#E69F00", "N3": "#009E73", "N4": "#CC79A7", "N5": "#D55E00"}  # Okabe-Ito
DPI = 200  # high-DPI PNGs
STYLE = {"copy": "--", "dapo": "-"}
STAGES = [
    ("rollout_seconds", "Rollout generation"),
    ("rollout_setup_sync_seconds", "Rollout setup + weight sync"),
    ("actor_logprob_seconds", "Actor logprob"),
    ("actor_train_seconds", "Actor train"),
    ("offload_restore_seconds", "Offload/restore"),
    ("audit_seconds", "Audit"),
]
TABLE_KEYS = [
    "actor_topology", "rollout_tp", "fixed_consistency", "effective_steps", "mismatch_count", "mean_abs_diff",
    "rollout_seconds", "rollout_tok_per_active_gpu_s", "rollout_setup_sync_seconds", "offload_restore_seconds",
    "actor_logprob_seconds", "actor_train_seconds", "e2e_seconds", "response_length", "reward", "policy_loss",
    "target_workload", "independent_verifier",
]  # fmt: skip


def series(rows, group, task, field):
    """(steps, values, effective flags); a non-effective (attempted) step is kept and flagged."""
    pts = sorted((r["step"], r[field], r["effective"]) for r in rows if r["group"] == group and r["task"] == task and r[field] is not None)
    return [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]


def task_means(rows, group, task, field):
    """Mean over effective steps only: an attempted, refused step never enters a mean."""
    vals = [r[field] for r in rows if r["group"] == group and r["task"] == task and r["effective"] and r[field] is not None]
    return mean(vals) if vals else None


def write_rounds(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(rd.FIELDS)
        for r in sorted(rows, key=lambda r: (r["group"], rd.TASKS.index(r["task"]), r["step"])):
            w.writerow(["" if r[k] is None else r[k] for k in rd.FIELDS])


def write_summary(path, groups, rows):
    out = dict(
        scope="per-step values are real evidence only; missing values are listed, never filled",
        groups={
            g: dict(
                config=v["config"],
                notes=v["notes"],
                matrix={key: c["value"] for key, c in v["cells"].items()},
                task_means={
                    t: {f: task_means(rows, g, t, f) for f in rd.NUMERIC if f != "step" and task_means(rows, g, t, f) is not None}
                    for t in rd.TASKS
                },
                missing=sorted({f"{r['task']}/{f}" for r in v["rows"] for f in r["missing"]}),
            )
            for g, v in groups.items()
        },
    )
    Path(path).write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")




def save(plt, fig, out, name):
    fig.tight_layout()
    fig.savefig(Path(out) / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def matrix_png(plt, groups, out):
    ROWS, fmt = rd.agg.ROWS, rd.agg.fmt

    labels = {key: label.strip("*") for label, key, _ in ROWS}
    cols = list(groups)
    cells = [[labels[k]] + [rd.agg.show(groups[g]["cells"][k]) for g in cols] for k in TABLE_KEYS]
    notes = rd.matrix_footnotes(groups)
    fig, ax = plt.subplots(figsize=(2.0 + 1.6 * len(cols), 0.38 * len(cells) + 1.0 + 0.3 * len(notes)))
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=["Metric"] + cols, loc="center", cellLoc="center", colWidths=[0.4] + [0.6 / len(cols)] * len(cols))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
        if c == 0:
            cell.set_text_props(ha="left")
    ax.set_title("Seven-layer DeepSeek V4 - ROCm Performance & Consistency Matrix\n(DAPO target rows; N/A = no evidence)", fontsize=10)
    wrapped = "\n".join(textwrap.fill(n, 150) for n in notes)
    if wrapped:
        fig.text(0.02, 0.0, wrapped, fontsize=6, va="top", ha="left")
    save(plt, fig, out, "performance-matrix.png")


def summary_png(plt, groups, rows, out):
    tasks = [t for t in ("copy", "dapo") if any(r["task"] == t and r["effective"] for r in rows)]
    fig, axes = plt.subplots(1, max(len(tasks), 1), figsize=(6.5 * max(len(tasks), 1), 4.5), squeeze=False)
    shades = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#9D755D", "#BAB0AC"]
    for ax, task in zip(axes[0], tasks):
        names = [g for g in groups if any(r["group"] == g and r["task"] == task and r["effective"] for r in rows)]
        bottoms = [0.0] * len(names)
        for (field, label), color in zip(STAGES, shades):
            vals = [task_means(rows, g, task, field) for g in names]
            ax.bar(names, [v or 0.0 for v in vals], bottom=bottoms, label=label, color=color)
            bottoms = [b + (v or 0.0) for b, v in zip(bottoms, vals)]
        ax.set_title(f"{task}: mean stage seconds per step (absent stage = not drawn)")
        ax.set_ylabel("seconds")
    if tasks:
        axes[0][0].legend(fontsize=8)
    else:
        axes[0][0].text(0.5, 0.5, "no per-step evidence", ha="center")
    save(plt, fig, out, "performance-summary.png")


def lines(ax, rows, groups, field, title, tasks=("copy", "dapo")):
    drawn = False
    for g in groups:
        for t in tasks:
            x, y, eff = series(rows, g, t, field)
            if x:
                ax.plot(x, y, STYLE[t], marker="o", color=COLORS[g], label=f"{g} {t}")
                bad = [(a, b) for a, b, e in zip(x, y, eff) if not e]
                if bad:  # attempted step that was refused: not an effective GRPO update
                    ax.plot(*zip(*bad), linestyle="none", marker="X", markersize=11, markerfacecolor="white",
                            markeredgecolor=COLORS[g], markeredgewidth=1.8, label=f"{g} {t} step (non-effective)")  # fmt: skip
                drawn = True
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("GRPO update")
    ax.set_xticks([1, 2, 3])
    if not drawn:
        ax.text(0.5, 0.5, "no evidence", ha="center", transform=ax.transAxes)
    return drawn


def trajectories_png(plt, groups, rows, out):
    panels = [
        ("rollout_seconds", "Rollout generation (s)"), ("rollout_setup_sync_seconds", "Rollout setup + weight sync (s)"),
        ("actor_logprob_seconds", "Actor logprob (s)"), ("actor_train_seconds", "Actor train (s)"),
        ("rollout_tok_per_active_gpu_s", "Rollout tok/GPU/s"), ("e2e_seconds", "End-to-end step (s)"),
    ]  # fmt: skip
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    handles = False
    for ax, (field, title) in zip(axes.flat, panels):
        handles = lines(ax, rows, groups, field, title) or handles
    if handles:
        axes[0][0].legend(fontsize=7)
    save(plt, fig, out, "performance-trajectories.png")


def consistency_png(plt, groups, rows, out):
    panels = [
        ("reward", "Mean reward"), ("policy_loss", "Policy loss"), ("kl", "KL loss"),
        ("mismatch_count", "Mismatch count (train vs rollout)"),
    ]  # fmt: skip
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (field, title) in zip(axes.flat, panels):
        if not lines(ax, rows, groups, field, title) and field == "kl":
            ax.text(0.5, 0.4, rd.KL_REASON, ha="center", transform=ax.transAxes, fontsize=8)
    axes[0][0].legend(fontsize=7)
    save(plt, fig, out, "consistency-reward.png")


def logp_png(plt, groups, rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    lines(axes[0], rows, groups, "mean_abs_diff", "Mean |train - rollout| logprob diff")
    lines(axes[1], rows, groups, "max_abs_diff", "Max |train - rollout| logprob diff")
    axes[0].legend(fontsize=7)
    save(plt, fig, out, "mean-logp-diff.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", action="extend", required=True, type=Path, help="run roots (runs-N/<id>)")
    ap.add_argument("--prefer", help="run root (or its name) that wins when a group appears in several runs")
    ap.add_argument("--output", type=Path, required=True, help="output directory (created)")
    ap.add_argument("--runtime-lock", type=Path, help="runtime lock (supplies recipes_commit when the run provenance lacks it)")
    args = ap.parse_args()
    try:
        groups, warnings = rd.load_groups(args.runs, args.prefer, args.runtime_lock)
    except (ValueError, FileExistsError) as error:
        sys.exit(f"error: {error}")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.output.mkdir(parents=True, exist_ok=True)
    rows = rd.all_rows(groups)
    write_rounds(args.output / "rounds.csv", rows)
    write_summary(args.output / "summary.json", groups, rows)
    matrix_png(plt, groups, args.output)
    summary_png(plt, groups, rows, args.output)
    trajectories_png(plt, groups, rows, args.output)
    consistency_png(plt, groups, rows, args.output)
    logp_png(plt, groups, rows, args.output)
    for w in warnings:
        print("warning:", w, file=sys.stderr)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
