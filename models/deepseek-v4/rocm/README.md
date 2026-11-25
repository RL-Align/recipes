# DeepSeek-V4 on AMD ROCm

Native GRPO with **verl + RL-Kernel**, using the original four-layer checkpoint.
Nonquantized parameters are trained; packed FP4/FP8 weights stay frozen.

[Requirements](#requirements) · [Image](#container-image) · [Quick start](#quick-start) · [Math workload](#math-workload) · [Results](#results)

## Configurations

| Configuration | Actor | Rollout |
| --- | --- | --- |
| [Single node · 8 GPUs](configs/single-node-8gpu.json) | GPUs 0–3: PP2 × DP2 | GPUs 4–7: two PP2 instances |
| [Four nodes · 32 GPUs](configs/four-node-32gpu.json) | Nodes 0–1: PP2 × DP8, 16 GPUs | Nodes 2–3: eight PP2 instances, 16 GPUs |

Original layers are split `[0, 1, 2] / [3]`; TP, EP and CP are 1.
This is a reduced-depth model, not a full-model training run.

**Validation:** CPU checks pass. The assembled image and GPU runs still need
hardware validation; the four-node configuration is experimental.
[Validation record](validation.md).

## Requirements

- Linux with a compatible ROCm driver, Docker, Git, Python 3.10+ and `rocm-smi`.
- Eight idle GPUs, physical IDs 0–7, on each node. Use the same architecture across nodes.
- For four nodes: noninteractive SSH from the first node and the same shared
  filesystem mounted at `work_dir` on every node.
- Reachable node IPs and the cluster's Ray/RCCL network access. Enable RDMA only
  when it is configured on the cluster.

Run all commands below on the first GPU node. Keep work files outside this Git
checkout and allow sufficient disk space for the image, model and run evidence.

## Container image

Download the base image from [Docker Hub](https://hub.docker.com/r/vensenmu/rl-kernel/tags?name=pr711-grpo-stress):

```bash
docker pull vensenmu/rl-kernel:pr711-grpo-stress
```

This is the PR #711 dependency image. The recipe builds a derived runtime with
the paired 8-/32-GPU source changes. The manual pull is optional: the `build`
command uses the pinned base digest automatically. The derived image is built
locally and distributed to all participating nodes.

For `gfx950`, the build also recompiles the required libraries from their pinned
sources. Use `gfx942` for MI300X and `gfx950` for MI350X.
[Exact image, model and dataset versions](sources.lock.json).

## Quick start

### 1. Get the recipe

```bash
git clone https://github.com/RL-Align/recipes.git
cd recipes
```

Choose **one** configuration:

```bash
# Single node, 8 GPUs
cp models/deepseek-v4/rocm/configs/single-node-8gpu.json site.json

# For four nodes, use this instead:
# cp models/deepseek-v4/rocm/configs/four-node-32gpu.json site.json
```

### 2. Configure your nodes

Edit `site.json`:

| Field | Set to |
| --- | --- |
| `nodes[].ip` | Each node's reachable IP address |
| `nodes[].ssh` | `local` for the first node; SSH alias or `user@host` for peers |
| `work_dir` | Writable work directory; a shared filesystem for four nodes |
| `model_dir` | Model location; keep the template path for automatic download |
| `baseline_data_dir` | Baseline data location; keep the template path for automatic download |
| `gpu_arch` | `gfx942` for MI300X or `gfx950` for MI350X |
| `network.ifname` | Communication interface, or empty for automatic selection |
| `network.rdma` | `true` only if the cluster has RDMA configured |

If you change `work_dir`, update the model/data paths under it as well.
The default `smoke` profile runs three real optimizer updates on the copy task.

### 3. Prepare the runtime and assets

```bash
python3 run.py build --config site.json
python3 run.py prepare --config site.json --download
```

Already have the delivered four-layer model and baseline data? Set their paths
and omit `--download`. Preparation verifies the pinned checksums.
Do not prepare the same dataset directory twice.

### 4. Run and check the result

```bash
python3 run.py run --config site.json
```

The launcher prints a new directory under `work_dir/runs/`, runs independent
verification and writes `report.md`. It stops only the containers it created.
A successful run must have `"status": "PASS"` in `verification.json`.

## Math workload

After the correctness run passes, use real DAPO-Math-17k or GSM8K data.

| Profile | Updates | Prompts × responses | Response cap |
| --- | --- | --- | --- |
| `smoke` | 3 | 2 × 2 on 8 GPUs; 8 × 2 on 32 GPUs | 16 tokens |
| `math-smoke` | 3 | 8 × 16 | 7,168 tokens |
| `math-200` | 200 | 8 × 16 | 7,168 tokens |

### Test the target load

Change these fields in your existing `site.json`, keeping the node and path
settings unchanged:

| Field | Set to |
| --- | --- |
| `profile` | `math-smoke` |
| `dataset` | `dapo` or `gsm8k` |
| `gates.correctness_run` | Directory of the passed correctness run |

```bash
python3 run.py prepare --config site.json
python3 run.py run --config site.json
```

### Run 200 updates

Once the math short run passes, change:

| Field | Set to |
| --- | --- |
| `profile` | `math-200` |
| `gates.math_smoke_run` | Directory of the passed math short run |

Reuse the prepared dataset:

```bash
python3 run.py run --config site.json
```

Both short runs must use the same image, model and allocation as the long run.
The math short run must also use the same dataset and load. This requests 200
optimizer updates, not 200 epochs.

Math prompts are capped at 1,024 tokens. Actual response lengths and token
counts are recorded. Zero advantages, invalid gradients or any raw FP32
train/rollout logprob bit mismatch fail the run.

## Results

Use the run directory printed by the launcher:

| File | Purpose |
| --- | --- |
| `verification.json` | Independent acceptance; must report `PASS` |
| `report.md` | Single-path performance summary |
| `steps.csv` | Per-step timings, tokens, rewards and consistency |
| `driver.log` | Training output and errors |
| `runtime.lock.json` | Exact image and source versions |

To recheck an existing run or regenerate its report, replace `RUN_DIR` below
with its actual path:

```bash
python3 run.py verify --config site.json --output RUN_DIR
python3 run.py report --config site.json --output RUN_DIR
```

Timings include evidence collection and auditing. KL and reference-policy
logprob are disabled. No native-verl speedup comparison is included.

## Provenance

The recipe builds on the [PR #711](https://github.com/RL-Align/RL-Kernel/pull/711)
baseline, the [#717 paired eight-GPU patches](https://github.com/RL-Align/RL-Kernel/pull/717#issuecomment-7059960424),
and explicit 32-GPU recipe extensions. Refer to [sources.lock.json](sources.lock.json)
for pins and [validation.md](validation.md) for the tested scope.
