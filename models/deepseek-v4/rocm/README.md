# DeepSeek-V4 / ROCm / native GRPO

**Scope:** the existing four-layer DeepSeek-V4 checkpoint, original layers
`[0,1,2] / [3]`, `rocm_triton/split`, native verl/Megatron updates and persistent
vLLM PP2 rollout workers. Packed FP4/FP8 bases and scales remain frozen;
nonquantized parameters are trained and synchronized. This is neither a
full-depth model run nor LoRA training.

[AMD 操作手册（中文）](../../../docs/amd-quickstart.zh-CN.md) ·
[Version and artifact lock](sources.lock.json) ·
[Single-node allocation](configs/single-node-8gpu.json) ·
[Four-node allocation](configs/four-node-32gpu.json)

## Topologies

| Allocation | Actor | Rollout | Parallel axes |
| --- | --- | --- | --- |
| Single node, 8 GPUs | GPUs 0–3, PP2 × DP2 | GPUs 4–7, two independent PP2 instances | TP=EP=CP=1 |
| Four nodes, 32 GPUs | Nodes 0–1, 8 GPUs each, PP2 × DP8 | Nodes 2–3, four PP2 instances per node, eight total | TP=EP=CP=1 |

The 32-GPU configuration extends actor DP and rollout replica placement. It does
not enable TP/EP/CP expansion, PP4 or full-model training. Cross-node actor DP
uses native Megatron/RCCL collectives. Actual PP/rank/PCI ownership is recorded
and checked. Policy sync remains checked, atomic **shared-file** export/load and
full original-dtype readback, not an RDMA weight-transfer implementation.

Each rollout instance reserves its own two GPUs on its declared host. Actor and
rollout nodes are disjoint in the 32-GPU layout. The verifier requires all 16
actor ranks and all eight rollout instances; omitted replicas cannot be counted
as a pass.

## Version lineage

| Layer | Pin / provenance | Meaning |
| --- | --- | --- |
| Correctness baseline | [RL-Kernel #711](https://github.com/RL-Align/RL-Kernel/pull/711), merge `93cb9714e223e17e06b63fe721da6a8204751c04` | Historical 4-GPU / 3-update result; not an 8/32-GPU runtime |
| Base dependency image | `vensenmu/rl-kernel:pr711-grpo-stress` | Public image; immutable registry reference is in the lock |
| Eight-GPU paired extension | [#717 review artifacts](https://github.com/RL-Align/RL-Kernel/pull/717#issuecomment-7059960424) | Complete RL-Kernel and verl patches based on `71c7cb2d…` / `d154f5aa…` |
| Recipe extension | `patches/recipe-*.patch` | 32-GPU placement/sync and workload-aware independent verification |

The old image alone does **not** include these eight-/32-GPU source additions.
`build` produces a source-only derived image with both projects at the locked
trees, preserving the base vLLM/MCore dependency stack. It records a new Docker
image ID and exports that same image for all nodes. The new image is **not** the
historically measured image. Registry index/manifest digests and local image IDs
are different identifiers; the lock records them separately.

Set `gpu_arch=gfx942` for MI300X. For gfx950, the Dockerfile runs the documented
same-source vLLM-extension and Transformer Engine common-library rebuilds from
the #717 handoff; simply reusing gfx942 binary code objects is insufficient.
Binary hashes are captured independently of Git source identities. Optional
process-local NUMA interleaving is explicit in the allocation config.

The original eight-GPU tested commits were local source heads. Rather than
assuming those SHAs are fetchable, this repository includes the complete
published paired patches, their hashes and the expected resulting Git tree IDs.
The source build fails if patch application or the resulting tree differs.

The observer's `None`-gradient handling is included from the eight-GPU review
bundle: it preserves Tensor/None and records real gradients. Required effective
gradients, mHC backward coverage and native DDP reduction checks remain enabled.
The original parameter-specific source of the observed None was not established.

## Workload profiles

| Profile | Dataset | Updates | Prompts × responses | Response cap | Purpose |
| --- | --- | --- | --- | --- | --- |
| `smoke` / 8 GPUs | Functional exact-copy pairs | 3 | 2 × 2 | 16 | Runtime, real updates, complete sync and audit |
| `smoke` / 32 GPUs | Same copy pairs, explicitly repeated to 24 prompt rows | 3 | 8 × 2 | 16 | Exercise DP8 without dropping ranks; not a math benchmark |
| `math-smoke` | DAPO-Math-17k or GSM8K | 3 | 8 × 16 | 7,168 | Test the exact long-run load and native math reward |
| `math-200` | Same prepared math dataset | 200 | 8 × 16 | 7,168 | End-to-end long run; requires both short-run gates |

Math prompt cap is 1,024 and model length is 8,192. Long prompts are filtered
with the pinned model's chat template; the retained row count is recorded. DAPO
uses its supplied prompts and `math_dapo` reward routing; GSM8K is converted to
verl's native `openai/gsm8k` rule-reward schema. No reference-policy KL is enabled.
The response cap is a maximum, not evidence that 917,504 response tokens were
actually generated per step. Actual masked token counts are retained.

The four-layer model may produce uniformly incorrect answers or long-context
train/rollout mismatches. Zero advantages, empty/nonfinite scores, incomplete
updates, missing DDP evidence or any raw FP32 word mismatch stop the run.
**Do not substitute synthetic rewards, relax comparisons, bypass training or
quietly shrink the load to produce a green report.**

## Acceptance and performance

1. Initial v0 audit and post-update v1…vN audits: actual B1/B2, AB/BA order,
   fixed and fresh sequences, live-worker continuity and complete readback.
2. Every update: all effective response tokens, native actor vs rollout raw
   FP32 logprob bit equality, finite/nonzero effective gradients and advantages.
3. Native PP transfers, DDP reduction across all declared DP ranks, optimizer
   state/counter continuity and changed/synchronized original tensors.
4. Separate independent verifier, successful driver exit and exact declared
   topology/source/input preflight. A process exit of 0 alone does not pass.

`steps.csv`, `summary.json` and `report.md` summarize the single RLK path in the
spirit of [PR #377](https://github.com/RL-Align/RL-Kernel/pull/377). Timings include
the retained verification/evidence work; rollout includes reward-loop overhead.
Actor throughput uses **response tokens**, which differs from total train-token
throughput. Reference logprob, disabled KL and uninstrumented longest-sample
decode throughput are N/A. No original-verl baseline or speedup is fabricated.

## Validation status

This repository was prepared in a CPU environment. Patch application, source
tree locks, configuration, rank placement contracts and strict-comparison
negative controls can be checked here. **The derived Docker build, new
single-node GPU run, 32-GPU end-to-end run and math-200 run are not yet hardware
validated.** Prior #711/#717 results apply only to their recorded environments.
The historical v3 mismatch and standalone GPU setup failures documented in #711
must not be erased by this recipe. Consult the new run's actual acceptance.
