# SPDX-License-Identifier: MIT
"""Log the seven-layer N1-N5 matrix to Weights & Biases from retained run roots (real evidence only).

One run per group (stable id `dsv4-7l-<group>`, so a re-upload resumes/updates instead of duplicating) plus
one summary run (`dsv4-7l-summary`) holding the full matrix wandb.Table and any PNGs from --images.
Per-step values are logged under `fixed/`, `copy/` and `dapo/` with `<task>/step` (GRPO update index) as the
x axis (panels: <task>/<section>/<name>; sections consistency, reward, loss, timing, throughput, response). Missing values are omitted, never filled; they are listed in each run's `missing` summary entry.
Default mode is offline; sync later with `wandb sync <dir>/wandb/offline-run-*`.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_data as rd  # noqa: E402

DEFAULT_BASE_URL = "https://api.forge.coreweave.com"  # wandb 0.31 rejects https://forge.coreweave.com[/wandb]
DEFAULT_ENTITY = "RL-Kernel"
DEFAULT_PROJECT = "dsv4-rocm-single-node"
WANDB_GROUP = "dsv4-7l-matrix"
LOGGED = list(rd.SECTIONS)


def run_id(group):
    return f"dsv4-7l-{group}"


def history(rows):
    """[{<task>/step: n, <task>/<metric>: v, ...}] in task then step order; None values are omitted."""
    out = []
    for row in sorted(rows, key=lambda r: (rd.TASKS.index(r["task"]), r["step"])):
        item = {f"{row['task']}/step": row["step"], f"{row['task']}/effective": int(row["effective"])}
        item.update({f"{row['task']}/{rd.SECTIONS[f]}": row[f] for f in LOGGED if row[f] is not None})
        out.append(item)
    return out


def missing_list(group):
    """Sorted, human-readable `<task>/step<n>/<field>: reason` entries plus group-level notes."""
    items = list(group["notes"])
    for row in group["rows"]:
        items += [f"{row['task']}/step{row['step']}/{f}: {why}" for f, why in row["missing"].items()]
    return items


def payloads(groups):
    """Run payloads (pure data): name, config, per-step history, summary, matrix column."""
    runs = []
    for name, g in groups.items():
        cfg = g["config"]
        column = [
            [label.strip("*"), rd.agg.show(g["cells"][key]), definition] for label, key, definition in rd.agg.ROWS
        ]
        summary = {f"matrix/{key}": g["cells"][key]["value"] for _, key, _ in rd.agg.ROWS}
        summary["status"] = cfg["status"]
        if g["cells"]["dapo_rollout_sha256"].get("sha256"):
            summary["dapo/rollout_token_sha256"] = g["cells"]["dapo_rollout_sha256"]["sha256"]
        notes = [c["footnote"] for c in g["cells"].values() if c.get("footnote")]
        if notes:
            summary["dapo/attempted_step_note"] = notes[0]
        summary["missing"] = missing_list(g)
        runs.append(
            dict(id=run_id(name), name=run_id(name), group=name, config=cfg, history=history(g["rows"]), summary=summary, column=column)
        )
    return runs


def define_metrics(wandb, tasks):
    for task in tasks:
        wandb.define_metric(f"{task}/step")
        wandb.define_metric(f"{task}/*", step_metric=f"{task}/step")


def upload(wandb, runs, summary_table, images, *, footnotes=(), mode, entity, project, base_url, directory):
    """Log payloads through the given wandb module (injectable for tests). Returns the run dirs/ids."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if base_url:
        os.environ["WANDB_BASE_URL"] = base_url
    common = dict(project=project, entity=entity, mode=mode, dir=str(directory), resume="allow")
    done = []
    for payload in runs:
        run = wandb.init(
            id=payload["id"], name=payload["name"], group=WANDB_GROUP, job_type="group", config=payload["config"],
            tags=["dsv4", "seven-layer", payload["group"]], **common,
        )  # fmt: skip
        define_metrics(wandb, rd.TASKS)
        for item in payload["history"]:
            run.log(item)
        run.log({"matrix_column": wandb.Table(columns=["Metric", payload["group"], "Definition"], data=payload["column"])})
        for key, value in payload["summary"].items():
            run.summary[key] = value
        run.finish()
        done.append(payload["id"])
    columns, rows = summary_table
    run = wandb.init(id="dsv4-7l-summary", name="dsv4-7l-summary", group=WANDB_GROUP, job_type="summary",
                     config=dict(groups=columns, run_ids={p["group"]: p["config"]["run_id"] for p in runs}), **common)  # fmt: skip
    run.log({"performance_consistency_matrix": wandb.Table(columns=["Metric"] + columns, data=rows)})
    if footnotes:
        run.summary["matrix/footnotes"] = footnotes
    for png in images:
        run.log({png.stem: wandb.Image(str(png))})
    run.finish()
    return done + ["dsv4-7l-summary"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", action="extend", required=True, type=Path, help="run roots (runs-N/<id>)")
    ap.add_argument("--prefer", help="run root (or its name) that wins when a group appears in several runs")
    ap.add_argument("--dir", type=Path, required=True, help="wandb output dir, e.g. $WORK/results-N/wandb")
    ap.add_argument("--mode", choices=["offline", "online", "disabled"], default="offline")
    ap.add_argument("--entity", default=DEFAULT_ENTITY)
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL, help="W&B server base URL (WANDB_BASE_URL)")
    ap.add_argument("--images", type=Path, help="plot_report.py output dir; its *.png go to the summary run")
    ap.add_argument("--dry-run", action="store_true", help="print the payloads as JSON; do not import wandb")
    args = ap.parse_args()
    try:
        groups, warnings = rd.load_groups(args.runs, args.prefer)
    except (ValueError, FileExistsError) as error:
        sys.exit(f"error: {error}")
    for w in warnings:
        print("warning:", w, file=sys.stderr)
    runs = payloads(groups)
    if args.dry_run:
        print(json.dumps(runs, indent=2, default=str))
        return
    try:
        import wandb
    except ImportError:
        sys.exit("error: wandb is not installed (pip install wandb)")
    images = sorted(args.images.glob("*.png")) if args.images else []
    ids = upload(wandb, runs, rd.matrix_table(groups), images, footnotes=rd.matrix_footnotes(groups), mode=args.mode, entity=args.entity,
                 project=args.project, base_url=args.base_url, directory=args.dir)  # fmt: skip
    print("logged:", ", ".join(ids), f"({args.mode}) under {args.dir}")


if __name__ == "__main__":
    main()
