# RL-Align Recipes

Reproducible RL training, train–rollout consistency checks and performance runs
with **verl + RL-Kernel**.

## Available recipes

| Model | Hardware | Configuration |
| --- | --- | --- |
| DeepSeek-V4, four layers | AMD ROCm, 8 GPUs | [Single node](models/deepseek-v4/rocm/configs/single-node-8gpu.json) |
| DeepSeek-V4, four layers | AMD ROCm, 32 GPUs | [Four nodes](models/deepseek-v4/rocm/configs/four-node-32gpu.json) |

## Quick start

Run from the first GPU node. You need Docker, Python 3.10+ and eight idle GPUs
per node. Four-node runs also require SSH access and a shared working directory.

**1. Get the recipe and choose a configuration.**

```bash
git clone https://github.com/RL-Align/recipes.git
cd recipes
cp models/deepseek-v4/rocm/configs/single-node-8gpu.json site.json
```

For four nodes, copy `four-node-32gpu.json` instead.

**2. Edit `site.json`.** Set node IPs, SSH targets, working/model/data paths and
the network interface. Use `gfx942` for MI300X or `gfx950` for MI350X.

**3. Build the runtime and download the pinned model/data bundle.**

```bash
python3 run.py build --config site.json
python3 run.py prepare --config site.json --download
```

**4. Run the three-step correctness check.**

```bash
python3 run.py run --config site.json
```

The command prints the new run directory, verifies the results and writes
`report.md`. It stops only the containers created for that run.

For dataset selection, 200-step runs and result files, see the
[DeepSeek-V4 recipe](models/deepseek-v4/rocm/README.md).

CPU checks pass. The assembled images and GPU runs still require hardware
validation; the 32-GPU configuration is experimental.
[Validation record](docs/validation.md) · [Contributing](docs/recipe-contract.md)

Images, models and datasets are external references with pinned versions and
checksums. Large artifacts are not stored in this repository.
