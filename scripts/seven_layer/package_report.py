# SPDX-License-Identifier: MIT
"""Build one ZIP of everything the project owner needs to publish the seven-layer N1-N5 matrix to W&B.

Contents: offline W&B runs (one per group + summary), raw per-step CSV/JSON, the English Performance &
Consistency Matrix (the very matrix.md/json/csv written by aggregate_matrix.py, plus PNGs at dpi>=200),
configuration and version info, the independent verifier receipts, and SYNC.md with the exact `wandb sync`
commands. The staged tree and the ZIP are scanned for credential patterns and the build fails if any is found.
No credentials are read or required.
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
import aggregate_matrix as agg  # noqa: E402
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


def sync_md(ids, entity, project, base_url):
    runs = "\n".join(f"wandb sync wandb/wandb/offline-run-*-{i}" for i in ids)
    return f"""# Syncing the offline W&B runs

Unzip, then from the unzipped directory (the owner's own `wandb login` supplies credentials; none are included).
This ZIP holds exactly one set of offline runs (`wandb/wandb/offline-run-*`); sync each run once, from this set only:

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


def source_commits(groups):
    """Source commits from the groups' provenance-derived configs: one value if all groups agree, else per group."""
    out = {}
    for key in ("recipes_commit", "rl_kernel_commit", "verl_commit"):
        values = {g: v["config"].get(key) for g, v in groups.items()}
        distinct = {json.dumps(x) for x in values.values()}
        out[key] = next(iter(values.values())) if len(distinct) == 1 else values
    return out


def build(
    run_roots, output, *, runtime_lock, prefer=None, entity=wr.DEFAULT_ENTITY, project=wr.DEFAULT_PROJECT,
    base_url=wr.DEFAULT_BASE_URL, wandb=None, matrix_dir=None, wandb_dir=None,
):  # fmt: skip
    """Build the ZIP.

    matrix_dir: an existing aggregate_matrix output whose matrix.md/json/csv are packaged as they are; without it
    aggregate_matrix.aggregate() runs into the staging area (same code path, byte-identical files).
    wandb_dir: where the offline W&B runs are written (the report directory's set); the ZIP copies that same set,
    so exactly one set exists. Without it the runs are written into the staging area only.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"{output} exists; refusing to overwrite")
    if wandb_dir is not None and list(Path(wandb_dir).glob("wandb/offline-run-*")):
        raise FileExistsError(f"{wandb_dir} already holds offline runs; refusing to create a second set")
    groups, warnings = rd.load_groups(run_roots, prefer, runtime_lock)
    stage = Path(tempfile.mkdtemp(prefix="dsv4-7l-report-"))
    try:
        root = stage / "dsv4-7l-report"
        rows = rd.all_rows(groups)
        raw = root / "raw"
        raw.mkdir(parents=True)
        pr.write_rounds(raw / "rounds.csv", rows)
        (raw / "rounds.json").write_text(json.dumps(rows, indent=2, default=str) + "\n", encoding="utf-8")
        pr.write_summary(raw / "summary.json", groups, rows)
        if matrix_dir is None:
            matrix_dir = stage / "matrix"
            agg.aggregate(run_roots, matrix_dir, runtime_lock, prefer)
        for name in ("matrix.md", "matrix.json", "matrix.csv"):  # the very files aggregate_matrix wrote, never re-rendered
            shutil.copyfile(Path(matrix_dir) / name, root / name)
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
        tooling = rd.tooling_commit()
        versions = dict(
            python=platform.python_version(), platform=platform.platform(), matplotlib=matplotlib.__version__,
            wandb=getattr(wandb, "__version__", None),
            source_commits=source_commits(groups),
            report_tooling_commit=tooling["commit"], report_tooling_commit_source=tooling["source"],
            warnings=warnings,
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
            wb = Path(wandb_dir) if wandb_dir is not None else root / "wandb"
            ids = wr.upload(
                wandb, wr.payloads(groups), rd.matrix_table(groups), sorted(figs.glob("*.png")), footnotes=rd.matrix_footnotes(groups), mode="offline",
                entity=entity, project=project, base_url=base_url, directory=wb,
            )  # fmt: skip
            if wandb_dir is not None:  # the same files as the report directory, copied once
                shutil.copytree(wb, root / "wandb", symlinks=True)  # latest-run stays a link: no second copy
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
    ap.add_argument("--runtime-lock", type=Path, required=True, help="runtime lock (dataset revision, recipes commit)")
    ap.add_argument("--matrix-dir", type=Path, help="existing aggregate_matrix.py output whose matrix.* files are packaged unchanged")
    ap.add_argument("--wandb-dir", type=Path, help="write the offline W&B runs here (report directory) and copy the same set into the ZIP")
    ap.add_argument("--entity", default=wr.DEFAULT_ENTITY)
    ap.add_argument("--project", default=wr.DEFAULT_PROJECT)
    ap.add_argument("--base-url", default=wr.DEFAULT_BASE_URL)
    args = ap.parse_args()
    try:
        import wandb
    except ImportError:
        wandb = None
    try:
        result = build(
            args.runs, args.output, runtime_lock=args.runtime_lock, prefer=args.prefer, entity=args.entity, project=args.project,
            base_url=args.base_url, wandb=wandb, matrix_dir=args.matrix_dir, wandb_dir=args.wandb_dir,
        )  # fmt: skip
    except (ValueError, FileExistsError, RuntimeError) as error:
        sys.exit(f"error: {error}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
