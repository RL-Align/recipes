# SPDX-License-Identifier: MIT
"""Materialize exact paired sources for CPU checks or independent review."""

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recipe_common import BOOK, lock


def sources(output):
    pins = lock()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for project, base, tree, patches in (
        (
            "RL-Kernel",
            pins["kernel_base"],
            pins["kernel_tree"],
            ["eight-gpu-rl-kernel.patch", "recipe-rl-kernel.patch"],
        ),
        ("verl", pins["verl"], pins["verl_tree"], ["eight-gpu-verl.patch", "recipe-verl.patch"]),
    ):
        repo = output / project

        def git(*args, _repo=repo, **kwargs):
            return subprocess.check_output(["git", "-C", str(_repo), *args], text=True, **kwargs).strip()

        repo.mkdir()
        git("init", "-q")
        git("remote", "add", "origin", f"https://github.com/RL-Align/{project}.git")
        git("fetch", "--depth=1", "origin", base)
        git("checkout", "--detach", "-q", "FETCH_HEAD")
        for patch in patches:
            path = BOOK / "patches" / patch
            if hashlib.sha256(path.read_bytes()).hexdigest() != pins["patches"][patch]:
                raise ValueError("patch SHA256 differs: " + patch)
            git("apply", "--check", str(path))
            git("apply", str(path))
        git("add", "-A")
        if git("write-tree") != tree:
            raise ValueError("effective source tree differs: " + project)
        env = {
            **os.environ,
            "GIT_AUTHOR_DATE": "2026-11-25T00:00:00Z",
            "GIT_COMMITTER_DATE": "2026-11-25T00:00:00Z",
        }
        git(
            "-c",
            "user.name=RL-Align Recipes",
            "-c",
            "user.email=recipes@users.noreply.github.com",
            "commit",
            "-qm",
            "Apply locked DSV4 recipe patches",
            env=env,
        )
        print(project, git("rev-parse", "HEAD"), tree)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    sources(parser.parse_args().output)
