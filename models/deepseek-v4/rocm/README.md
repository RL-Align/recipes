# DeepSeek-V4 on AMD ROCm

Run native GRPO with verl + RL-Kernel using the original four-layer checkpoint.
Layers are split `[0, 1, 2] / [3]`. Packed FP4/FP8 weights stay frozen;
nonquantized parameters are trained. This is a reduced-depth model.

**Status:** CPU checks pass. The assembled runtime needs GPU validation.
Four-node execution is experimental.

## Download the image

Base image: [vensenmu/rl-kernel:pr711-grpo-stress](https://hub.docker.com/r/vensenmu/rl-kernel/tags?name=pr711-grpo-stress).

```bash
docker pull vensenmu/rl-kernel:pr711-grpo-stress
```

The base image does not contain all 8-/32-GPU changes. After configuring your
nodes, build the required runtime:

```bash
python3 run.py build --config site.json
```

This also pulls the base image automatically, so the manual pull is optional.

## Choose an allocation

| Configuration | Actor | Rollout |
| --- | --- | --- |
| [Single node, 8 GPUs](configs/single-node-8gpu.json) | GPUs 0–3: PP2 × DP2 | GPUs 4–7: two PP2 instances |
| [Four nodes, 32 GPUs](configs/four-node-32gpu.json) | Nodes 0–1: PP2 × DP8, 16 GPUs | Nodes 2–3: eight PP2 instances, 16 GPUs |

TP, EP and CP are 1. All selected GPUs must be idle and use the same
architecture. Run the commands on the first node.

## Configure

Follow the [quick start](../../../README.md#quick-start), then edit these fields
in `site.json`:

| Field | Value |
| --- | --- |
| `nodes[].ip` | Reachable node IP |
| `nodes[].ssh` | `local` for the first node; SSH alias or `user@host` for peers |
| `work_dir` | Writable work directory; the same shared filesystem on all four nodes |
| `model_dir` | Model directory; keep the template path to download automatically |
| `baseline_data_dir` | Baseline data directory; keep the template path to download automatically |
| `gpu_arch` | `gfx942` for MI300X; `gfx950` for MI350X |
| `network.ifname` | Cluster network interface, or empty for automatic selection |
| `network.rdma` | `true` only when RDMA is configured on the cluster |

Docker and Python 3.10+ are required on each node. Peer SSH must work without
interactive prompts. Use an unused Ray port and permit the cluster's Ray/RCCL
traffic. Keep work files outside the Git checkout.

The base image is `vensenmu/rl-kernel:pr711-grpo-stress`. `build` adds the paired
source changes and distributes the same derived image to every node. For gfx950
it also rebuilds the required libraries from their pinned sources.
[Image, model and dataset versions](sources.lock.json).

Already have the delivered model and baseline data? Set their paths and omit
`--download` from the preparation command. Do not prepare the same dataset twice.

## Run real math data

The default `smoke` profile is a functional copy-task check. After it passes,
use DAPO-Math-17k (`dapo`) or GSM8K (`gsm8k`) for the real workload.

| Profile | Updates | Prompts × responses | Maximum response length |
| --- | --- | --- | --- |
| `smoke` | 3 | 2 × 2 on 8 GPUs; 8 × 2 on 32 GPUs | 16 |
| `math-smoke` | 3 | 8 × 16 | 7,168 |
| `math-200` | 200 | 8 × 16 | 7,168 |

**1. Set these fields in your existing `site.json`:**

```json
"profile": "math-smoke",
"dataset": "dapo",
"gates": {
  "correctness_run": "/path/to/passed-smoke-run",
  "math_smoke_run": ""
}
```

Use the run directory printed by the previous correctness check. Keep the other
configuration fields unchanged.

**2. Prepare the math dataset and test the target load.**

```bash
python3 run.py prepare --config site.json
python3 run.py run --config site.json
```

**3. After that run passes**, set `profile` to `math-200` and
`gates.math_smoke_run` to its run directory, then run:

```bash
python3 run.py run --config site.json
```

This reuses the prepared data and requests **200 optimizer updates**. Both short
runs must use the same image, model and allocation as the long run. The math
short run must also use the same dataset and load.

Math prompt length is capped at 1,024 tokens. Actual response lengths and
generated token counts are recorded. Zero advantages, invalid gradients or any
FP32 train/rollout logprob bit mismatch fail the run.

## Results

Each run uses a new directory under `work_dir/runs/`. The launcher prints its
path. Check these files:

| File | Purpose |
| --- | --- |
| `verification.json` | Independent acceptance; must report `PASS` |
| `report.md` | Single-path performance summary |
| `steps.csv` | Per-step timings, tokens, rewards and consistency |
| `driver.log` | Training output and errors |
| `runtime.lock.json` | Exact image and source versions |

To recheck or regenerate a report, replace `RUN_DIR` with the printed path:

```bash
python3 run.py verify --config site.json --output RUN_DIR
python3 run.py report --config site.json --output RUN_DIR
```

Timings include evidence collection and auditing. KL and reference-policy
logprob are disabled. No native-verl speedup comparison is included.

Version lineage: [PR #711](https://github.com/RL-Align/RL-Kernel/pull/711) baseline,
[#717 paired eight-GPU patches](https://github.com/RL-Align/RL-Kernel/pull/717#issuecomment-7059960424),
and explicit recipe extensions for 32 GPUs. See the
[validation record](../../../docs/validation.md) for the tested scope.
