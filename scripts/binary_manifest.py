# SPDX-License-Identifier: MIT
"""Record rebuilt binary hashes independently of source commits."""

import hashlib
import json
from pathlib import Path

roots = [
    Path("/opt/dsv4/vllm/vllm"),
    Path("/opt/dsv4/TransformerEngine"),
    Path("/opt/dsv4-venv/lib/python3.12/site-packages/transformer_engine"),
]
files = {}
for root in roots:
    for path in sorted(root.rglob("*.so*")):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                    digest.update(chunk)
            files[str(path)] = dict(bytes=path.stat().st_size, sha256=digest.hexdigest())
if not files:
    raise RuntimeError("no vLLM/TransformerEngine libraries were found")
print(json.dumps(files, indent=2))
