# SPDX-License-Identifier: MIT
"""Build from immutable clean commits and save exact source/image provenance."""

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BOOK = ROOT / "models/deepseek-v4/rocm/seven-layer"


def output(argv):
    return subprocess.check_output(argv, text=True).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, required=True)
    args = ap.parse_args()
    if output(["git", "-C", str(ROOT), "status", "--porcelain"]):
        raise ValueError("commit recipe changes before rebuilding")
    pins = json.loads((BOOK / "sources.lock.json").read_text())
    recipe_commit = output(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    tag = "rl-align/dsv4-seven-layer:actor-tp1-" + recipe_commit[:12]
    values = dict(
        BASE_IMAGE=pins["base_image"]["reference"],
        RLK_SHA=pins["kernel"]["commit"],
        RLK_TREE=pins["kernel"]["tree"],
        VERL_SHA=pins["verl"]["commit"],
        VERL_TREE=pins["verl"]["tree"],
        RECIPE_COMMIT=recipe_commit,
    )
    command = ["docker", "build", "--pull=false", "--file", str(BOOK / "Dockerfile"), "--tag", tag]
    for key, value in values.items():
        command += ["--build-arg", key + "=" + value]
    command.append(str(ROOT))
    subprocess.run(command, check=True)
    image_id = output(["docker", "image", "inspect", "--format={{.Id}}", tag])
    info = dict(
        image=tag,
        image_id=image_id,
        recipe_commit=recipe_commit,
        sources=pins,
        build_command=command,
        scope="Experimental actor-TP1 real scoring and native three-update phase execution; GPU acceptance pending; N5 EP4 experimental, GPU acceptance pending",
    )
    runtime = args.work / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    path = runtime / ("runtime-" + recipe_commit[:12] + ".lock.json")
    if path.exists():
        raise FileExistsError("retain prior build lock: " + str(path))
    path.write_text(json.dumps(info, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
