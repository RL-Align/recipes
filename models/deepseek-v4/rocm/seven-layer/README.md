# Seven-layer DSV4 actor-TP1 / rollout-TP1/2/4/8 validation

Experimental implementation, GPU acceptance pending. The original M1–M5 matrix
was revised on 2026-11-25: native DSV4 Hybrid Attention and this actor integration
require TP1. Historical configurations remain in `legacy-M-configs`; historical
CPU/startup evidence does not qualify the revised matrix. Never remove the actor
TP restriction or use Qwen3 support as DSV4 evidence.

| Group | Actor PP/TP/EP/ETP | Dense DP / expert DP | Actor GPUs | Rollout PP/TP/EP | Rollout GPUs |
| --- | --- | --- | ---: | --- | ---: |
| N1 | 2/1/1/1 | 1/1 | 2 | 1/1/1 | 1 |
| N2 | 2/1/1/1 | 1/1 | 2 | 1/2/1 | 2 |
| N3 | 2/1/1/1 | 1/1 | 2 | 1/4/1 | 4 |
| N4 | 2/1/1/1 | 1/1 | 2 | 1/8/1 | 8 |
| N5 | 2/1/4/1 | **4/1** | 8 | 1/8/1 | 8 |

All groups use original continuous layers 0–6, CP1, one rollout instance and
serial phase reuse of the same eight-card machine. Actor PP2 uses 4/3 original
layers: embedding on PP0; final norm, output head and mHC contraction on PP1.
N5 is a later expert partition/sync extension, gated on N1–N4 correctness. Its
EP4 compute/communication path is **not implemented or qualified here**. The
executor records BLOCKED and never substitutes EP1 or blocks primary groups.

## Implementation and acceptance boundaries

Implemented experimental paths include native PP2 actor execution, PP1 rollout
TP1/2/4/8 group projection and original-vocabulary FP32 score gathering, actor
policy export and original-tensor rollout shard readback, native verl generation,
copy/math rewards, GRPO advantage, PPO loss and Megatron backward/optimizer.
Cross-TP floating-point arithmetic still requires GPU bit qualification; a failed
comparison stops that group. CPU success establishes no numerical GPU result.

Every GPU phase is a new labelled private container. Exit releases its GPU
contexts before the next phase. Actor reconstruction restores full model state
including packed quantized buffers, optimizer masters/moments, scheduler, weight
version and RNG. Restored values are checked before a new optimizer identity is
recorded; this does not claim workers stayed resident.

The independent verifier checks actual seven-layer PP ownership, rank count and
PCI mapping, every exported nonquantized tensor and actual rollout shard bytes,
all valid response-token raw FP32 bits, finite values and a one-bit negative
control. A separate three-update verifier replays actual update-forward token
coverage, checkpoint bytes, native gradient/parameter/optimizer evidence, PP
transfers, exact recovery, and fresh/fixed post-update audits including version 3.

## Reproduce the existing experiment

Keep the current persistent WORK directory and verified assets. Do not download
or extract them again when their locks already match. The original source is
`deepseek-ai/DeepSeek-V4-Flash` revision
`60d8d70770c6776ff598c94bb586a859a38244f1`; compression ratios are
`[0,0,4,128,4,128,4]`. Original weights, scales, layer config, hash routing and
metadata are retained; MTP is excluded. `checkpoint.lock.json` records every
original tensor's dtype, shape and SHA256. DAPO-Math-17k is fixed at revision
`65877096c24ffa7abc4e4fa5edb95cf3413a5674`.

```bash
python3 scripts/seven_layer/matrix.py prepare --work "$WORK"
python3 scripts/seven_layer/build.py --work "$WORK"
# CPU request preparation uses the committed runtime and original native tokenizer;
# reuse the existing data/requests-N files after checking requests.lock.json.
python3 scripts/seven_layer/execute.py --work "$WORK" --image sha256:FINAL_IMAGE_ID
```

The executor runs common fixed actor weights against N1–N4 rollout configurations,
then each passing group's three-update copy correctness trajectory, then a separate
fresh three-update DAPO trajectory at 8 prompts ×16 responses, prompt cap1024 and
response cap7168. It never launches 200 steps or multiple nodes. Equal rewards,
zero advantages, no parameter change, OOM and missing evidence fail with artifacts
retained. A reduced load cannot substitute for the target load.

Source checkouts must be clean committed revisions before building. The image
records source trees and an immutable `.Id` in its runtime lock. Only RL-Kernel,
RL-Align/verl and this recipe change; external framework source remains pinned.

## Resource gate and continuation

The revised watcher uses `evidence/watch-N` and **ready-N.json**. The old M ready
file was archived and its watcher stopped. Never re-arm the old `ready.json`.

```bash
python3 scripts/seven_layer/watch.py \
  --output "$WORK/evidence/watch-N" --ready "$WORK/ready-N.json"
```

The gate checks immutable prerequisite hashes, all eight GPU processes, memory
and utilization every 30 seconds, then repeats idle checks immediately before
launch. The command in ready-N must invoke `execute.py`, never the old diagnostic
coordinator. Locks prevent duplicate self-launch only; they do not reserve other
users' GPUs. The executor rechecks before each phase, logs physical device
allocation and each container lifecycle, and only stops its own labelled container
on timeout. GPU utilization zero alone is insufficient. It never stops another
user's task, shared Ray/Docker service, resets a GPU or restarts the node.

## Results and components

Maintain separate implementation, CPU, fixed-weight GPU, three-update and target
load states. All unmeasured values are N/A. Count response tokens once per logical
sample, never once per rank; report throughput per active GPU and per whole eight
GPU machine. End-to-end step timing includes generation, scoring, update,
synchronization, release/recovery and audits. Native metrics remain saved in
`native-train-output.pt`; raw phase/execution JSONs retain individual timings.

Long inputs are prepared around compression/window boundaries and indexer Top-k
candidate counts. Attention/compression state, indexer/Top-k, hash/later routing,
routed/shared experts, mHC, embedding/head/norm, effective backward, PP transfers
and N5 EP communication stay **uncovered until actual execution evidence**. Seven
layers alone never establish coverage. Reports retain failed and partial steps
and independent verifier results; no native baseline or speedup is inferred.
