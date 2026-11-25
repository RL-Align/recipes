# Recipe contract

Every recipe must provide:

1. **Scope:** model revision/layers, frozen/trainable state, hardware and exact
   actor/rollout resources. Node count is not GPU count.
2. **Immutable inputs:** registry digest, paired framework commits/patch hashes,
   effective source trees, model and dataset revisions/checksums.
3. **Preparation:** accessible assets, schema/reward mapping, shared/local storage
   requirements and an editable allocation file containing no credentials.
4. **Execution:** one common entrypoint, a non-mutating plan, new output per run,
   allocated devices only, isolated runtime lifecycle and retained failed runs.
5. **Acceptance:** actual optimizer updates, full response-token checks, no
   tolerance relaxation, original tensor readback, live worker continuity and an
   independent verifier. Performance tables never substitute for acceptance.
6. **Status:** measured, CPU-checked, hardware-pending and failed are distinct.
   A short-run result does not establish a long-run or cross-node claim.

Recipes should primarily contain instructions, configuration and version locks.
Small preparation, launch and verification helpers are appropriate when needed.
Training algorithms, kernels and framework integrations belong in RL-Kernel or
the integration framework; prefer pinned upstream commits over bundled patches.
Temporary patches must retain their provenance and checksums until integrated.

The repository README is a model/backend index. Keep image downloads, model-specific
setup commands and validation records inside each recipe directory. Model/backend
details belong under `models/`; reusable orchestration belongs in
`scripts/` and `recipe_common.py`. Functional smoke data and real workload data
must be clearly separated. Commit executable configuration and references,
not weights, caches, private hostnames, raw traces, images or credentials.

## CPU review

```bash
python3 -m venv .venv
.venv/bin/pip install pytest hydra-core numpy ray ruff==0.12.2
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 scripts/sources.py --output /tmp/dsv4-recipe-sources-NEW
RLK_RECIPE_TEST_SOURCE=/tmp/dsv4-recipe-sources-NEW .venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

The source bootstrap fetches pinned public commits, checks patch hashes and
effective trees and makes clean local commits. Tests inspect the actual patched
contracts, including missing-rank/replica/placement and raw-bit negative controls.
They do not simulate a hardware acceptance result.
