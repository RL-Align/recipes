# SPDX-License-Identifier: MIT
"""Recipe helpers must win over dependency packages with the same name."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_report_import_with_dependency_scripts_package(tmp_path):
    dependency = tmp_path / "dependency"
    package = dependency / "scripts"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "raise RuntimeError('dependency scripts package shadowed the recipe')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", "from scripts.report import report; assert callable(report)"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": os.pathsep.join((str(ROOT), str(dependency)))},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
