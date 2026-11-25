# RL-Align Recipes

**Train. Sync. Audit. Measure.**

Reproducible reinforcement-learning recipes for **verl + RL-Kernel**. Each recipe
connects a model and hardware allocation to locked software, prepared data, a
launcher and independent acceptance checks. The deliverable is a repeatable run
with retained evidence.

| Model | Hardware | Allocation | Entry | Validation status |
| --- | --- | --- | --- | --- |
| DeepSeek-V4, four original layers | AMD ROCm, MI300X / MI350X | One node, 8 GPUs | [Recipe](models/deepseek-v4/rocm/README.md) · [AMD 中文操作手册](docs/amd-quickstart.zh-CN.md) | Prior 3-step MI350X result exists; this assembled recipe needs a fresh GPU run |
| DeepSeek-V4, four original layers | AMD ROCm, homogeneous nodes | **4 nodes × 8 GPUs = 32 GPUs** | Same recipe; separate allocation file | **Experimental: CPU contracts checked; multi-node GPU acceptance pending** |

## Start here

```bash
git clone https://github.com/RL-Align/recipes.git
cd recipes
cp models/deepseek-v4/rocm/configs/single-node-8gpu.json site.json
# Or: cp models/deepseek-v4/rocm/configs/four-node-32gpu.json site.json
# Edit node IPs/SSH targets, paths and the network interface in site.json.
python3 run.py plan --config site.json
python3 run.py build --config site.json
python3 run.py prepare --config site.json --download
python3 run.py run --config site.json --output /shared/dsv4-recipe/runs/smoke-001
```

The example output path must be changed if `work_dir` is changed. Run commands
from the first allocated node. The four-node recipe requires a genuinely shared
filesystem, SSH access from the first node, and Docker on every node. See the
[prerequisites and complete math-test sequence](docs/amd-quickstart.zh-CN.md).

The default is **three real optimizer updates on the functional copy task**.
DAPO-Math-17k and GSM8K are separate math profiles. A three-step math run with the
target load must pass before starting 200 updates. Preparation/building requires
no GPUs; the training launcher refuses occupied devices.

## What this repository contains

| Location | Purpose |
| --- | --- |
| `models/<model>/<backend>/` | Model recipe, immutable input references, allocation templates and scoped patches |
| `run.py` | Common `plan`, `build`, `prepare`, `run`, `verify`, `report` interface |
| `scripts/` | Node lifecycle, asset preparation, runtime checks and evidence reporting |
| `docs/` | Operator handoff and contribution contract |
| `tests/` | CPU contracts; GPU acceptance is reported separately |

Inspired by the model-oriented navigation of
[vLLM Recipes](https://github.com/vllm-project/recipes), this project emphasizes
the RL update cycle: **real rollout and rewards → native optimizer update →
complete policy synchronization → raw train/rollout logprob verification**.

Images, model weights, dataset downloads and run artifacts are **referenced, not
committed**. Tags help humans find an image; digests, source trees, dataset
revisions and SHA256 values identify what was actually used. Do not compare
performance from different workloads or call an unexecuted configuration a pass.

## Contribute

Read the [recipe contract](docs/recipe-contract.md). Keep new model recipes
self-contained and keep framework changes in explicit, checksummed patches.
Never add tokens, private host information, weights, generated policies or traces.
The repository's own code is MIT; bundled framework patches retain their
[upstream notices](NOTICE.md).

See the [current validation record](docs/validation.md) for CPU checks and the
remaining hardware validation boundary.
