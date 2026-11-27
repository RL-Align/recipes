# SPDX-License-Identifier: MIT
"""Build one ZIP of everything the project owner needs to publish the seven-layer N1-N5 matrix to W&B.

Contents: offline W&B runs (one per group + summary), raw per-step CSV/JSON, the English Performance &
Consistency Matrix (Markdown + PNGs at dpi>=200), configuration and version info, the independent verifier
receipts, and SYNC.md with the exact `wandb sync` commands. The staged tree and the ZIP are scanned for
credential patterns and the build fails if any is found. No credentials are read or required.
"""

import argparse
import json
import platform
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_report as pr  # noqa: E402
import report_data as rd  # noqa: E402
import wandb_report as wr  # noqa: E402

SECRET = re.compile(
    rb"(?i)(api[_-]?key|\bnetrc\b|\bpasswd\b|\bpassword\b|\bsecret\b|(access|auth|bearer|hf|api)[_-]?token"
    rb"|\"token\"\s*:|authorization:|\bhf_[A-Za-z0-9]{20,}|\bghp_[A-Za-z0-9]{20,}|\bsk-[A-Za-z0-9]{20,})"
)
VERIFIER_NAMES = {"verification.json", "update-verification.json", "steps.json", "result.json", "ep4-vs-ep1-actor.json", "execution.json"}


def scan_secrets(root):
    """[(relative path, match)] for every file under root that matches a credential pattern."""
    hits = []
    for p in sorted(Path(root).rglob("*")):
        if p.is_file():
            m = SECRET.search(p.read_bytes())
            if m:
                hits.append((p.relative_to(root).as_posix(), m.group(0)[:40].decode("utf-8", "replace")))
    return hits


def verifier_files(run_root, group, scope=None):
    """Acceptance receipts (not raw .f32 dumps) below <run>/<group>[/<scope>]."""
    base = Path(run_root) / group
    base = base / scope if scope else base
    for p in sorted(base.rglob("*.json")):
        if p.name in VERIFIER_NAMES or p.name.startswith(("training-parity-rank-", "native-loss-rank")):
            yield p


def matrix_markdown(groups, note):
    columns, rows = rd.matrix_table(groups)
    lines = [
        "# Seven-layer DeepSeek V4 - ROCm Performance & Consistency Matrix",
        "",
        "| Metric | " + " | ".join(columns) + " |",
        "| --- |" + " --- |" * len(columns),
    ] + ["| " + " | ".join(r) + " |" for r in rows]
    lines += [""] + [f"{n}  " for n in rd.matrix_footnotes(groups)]
    lines += ["", "## Metric definitions", ""] + [f"- **{label.strip('*')}**: {d}" for label, _, d in rd.agg.ROWS]
    lines += [
        "",
        "## Notes",
        "",
        f"- {note}",
        "- N/A means the evidence does not contain the value (never zero-filled); per-step reasons are in raw/rounds.json.",
        "- Fixed-weight, copy-task and DAPO-Math-17k data are separate namespaces and are never mixed in one curve or mean.",
    ]
    return "\n".join(lines) + "\n"


def sync_md(ids, entity, project, base_url):
    runs = "\n".join(f"wandb sync wandb/wandb/offline-run-*-{i}" for i in ids)
    return f"""# Syncing the offline W&B runs

Unzip, then from the unzipped directory (the owner's own `wandb login` supplies credentials; none are included):

```bash
pip install wandb
export WANDB_BASE_URL={base_url}   # CoreWeave Forge API host (the wandb client rejects https://forge.coreweave.com/wandb)
export WANDB_ENTITY={entity}
export WANDB_PROJECT={project}
wandb login --host {base_url}
{runs}
```

Project UI: https://forge.coreweave.com/wandb/{entity}/{project}. Run ids are stable (`dsv4-7l-N1` .. `dsv4-7l-N5`, `dsv4-7l-summary`), so
re-syncing resumes the same run (sync each run once to avoid duplicated step rows).
Entity and project were stored in each offline run; the env vars above only restate them. Panels are named
`<task>/<section>/<name>` with task in fixed/copy/dapo and section in consistency, reward, loss, timing,
throughput, response; the x axis is `<task>/step` (GRPO update index).
"""


def build(run_roots, output, *, prefer=None, entity=wr.DEFAULT_ENTITY, project=wr.DEFAULT_PROJECT, base_url=wr.DEFAULT_BASE_URL, wandb=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"{output} exists; refusing to overwrite")
    groups, warnings = rd.load_groups(run_roots, prefer)
    stage = Path(tempfile.mkdtemp(prefix="dsv4-7l-report-"))
    try:
        root = stage / "dsv4-7l-report"
        rows = rd.all_rows(groups)
        raw = root / "raw"
        raw.mkdir(parents=True)
        pr.write_rounds(raw / "rounds.csv", rows)
        (raw / "rounds.json").write_text(json.dumps(rows, indent=2, default=str) + "\n", encoding="utf-8")
        pr.write_summary(raw / "summary.json", groups, rows, " ".join(sys.argv))
        note = "Performance rows are DAPO-Math-17k target values; consistency rows are cumulative over its updates."
        (root / "matrix.md").write_text(matrix_markdown(groups, note), encoding="utf-8")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figs = root / "figures"
        figs.mkdir()
        pr.matrix_png(plt, groups, figs)
        pr.summary_png(plt, groups, rows, figs)
        pr.trajectories_png(plt, groups, rows, figs)
        pr.consistency_png(plt, groups, rows, figs)
        pr.logp_png(plt, groups, rows, figs)
        cfg = root / "config"
        cfg.mkdir()
        versions = dict(
            python=platform.python_version(), platform=platform.platform(), matplotlib=matplotlib.__version__,
            wandb=getattr(wandb, "__version__", None), warnings=warnings,
        )  # fmt: skip
        (cfg / "versions.json").write_text(json.dumps(versions, indent=2) + "\n", encoding="utf-8")
        for g, v in groups.items():
            c = v["config"]
            (cfg / f"{g}-config.json").write_text(json.dumps(c, indent=2, default=str) + "\n", encoding="utf-8")
            rr = Path(c["run_root"])
            for name in ("provenance.json", "summary.json", "resource-policy.json"):  # provenance.json = sources lock
                if (rr / name).is_file():
                    shutil.copy(rr / name, cfg / f"{g}-{rr.name}-{name}")
            pieces = [(rr, None)]
            if c["correctness_from"]:
                src = Path(c["correctness_from"])
                pieces = [(rr, "dapo"), (rr, "fixed-rollout"), (src, "copy"), (src, "fixed-rollout")]
            for owner, scope in pieces:
                for p in verifier_files(owner, g, scope):
                    dest = root / "verifier" / owner.name / p.relative_to(owner)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(p, dest)
        ids = []
        if wandb is not None:
            ids = wr.upload(
                wandb, wr.payloads(groups), rd.matrix_table(groups), sorted(figs.glob("*.png")), footnotes=rd.matrix_footnotes(groups), mode="offline",
                entity=entity, project=project, base_url=base_url, directory=root / "wandb",
            )  # fmt: skip
        else:
            warnings.append("wandb not installed: offline runs not included")
        names = ids or [wr.run_id(g) for g in groups] + ["dsv4-7l-summary"]
        (root / "SYNC.md").write_text(sync_md(names, entity, project, base_url), encoding="utf-8")
        hits = scan_secrets(root)
        if hits:
            raise RuntimeError("credential-like content found, refusing to package: " + "; ".join(f"{p} ({m!r})" for p, m in hits))
        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(root.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(stage).as_posix())
        with tempfile.TemporaryDirectory() as check:  # scan what actually went into the archive
            with zipfile.ZipFile(output) as z:
                z.extractall(check)
            hits = scan_secrets(check)
        if hits:
            output.unlink()
            raise RuntimeError(f"credential-like content found in ZIP: {hits}")
        return dict(zip=str(output), runs=ids, warnings=warnings)
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", action="extend", required=True, type=Path, help="run roots (runs-N/<id>)")
    ap.add_argument("--prefer", help="run root (or its name) that wins when a group appears in several runs")
    ap.add_argument("--output", type=Path, required=True, help="ZIP file to create (must not exist)")
    ap.add_argument("--entity", default=wr.DEFAULT_ENTITY)
    ap.add_argument("--project", default=wr.DEFAULT_PROJECT)
    ap.add_argument("--base-url", default=wr.DEFAULT_BASE_URL)
    args = ap.parse_args()
    try:
        import wandb
    except ImportError:
        wandb = None
    try:
        result = build(args.runs, args.output, prefer=args.prefer, entity=args.entity, project=args.project, base_url=args.base_url, wandb=wandb)
    except (ValueError, FileExistsError, RuntimeError) as error:
        sys.exit(f"error: {error}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
