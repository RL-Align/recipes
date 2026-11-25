# Seven original DSV4 layers on one eight-GPU AMD node

**Experimental and incomplete. No M1–M5 GPU consistency or GRPO PASS is claimed.**
This separate profile preserves the historical four-layer recipe. It adds original
checkpoint extraction, a recoverable resource gate, five exact target profiles,
a fixed-token input suite, real model startup diagnostics, and evidence replay.
The remaining strict TP/EP kernels, weight resharding and phased GRPO lifecycle
are not implemented by these helpers. Do not run the four-layer trainer with these
JSON files or interpret a successful startup as acceptance.

| Group | Actor PP/TP/EP/ETP | Actor GPUs | Rollout PP/TP/EP | Rollout GPUs | Peak physical GPUs |
| --- | --- | ---: | --- | ---: | ---: |
| M1 | 1/1/1/1 | 1 | 1/2/1 | 2 | 2 |
| M2 | 1/2/1/2 | 2 | 1/4/1 | 4 | 4 |
| M3 | 1/4/1/4 | 4 | 1/8/1 | 8 | 8 |
| M4 | 1/8/1/8 | 8 | 1/1/1 | 1 | 8 |
| M5 | 2/4/4/1 | 8 | 1/8/1 | 8 | 8 |

All ordinary DP, expert DP and CP are 1. Each rollout has one instance. The
experiment requests all eight GPUs and reuses them across serial phases. Actor
and rollout counts are never added. Sequence parallelism is requested for TP>1;
its numerical path still requires implementation/qualification. M5's planned
layer layout is `[0,1,2,3] / [4,5,6]`, with embedding on stage zero and final norm,
output head and mHC contraction on stage one. EP4 assigns 64/256 experts per
expert rank. Planned mappings are not evidence of live rank placement.

## Preparation

Choose a new persistent `WORK` directory. Do not overwrite an older extraction.
The source is `deepseek-ai/DeepSeek-V4-Flash` revision
`60d8d70770c6776ff598c94bb586a859a38244f1`.

```bash
python3 scripts/seven_layer/prepare_checkpoint.py \
  --reuse-cache /path/to/verified/original-source-shards \
  --output "$WORK/models/DeepSeek-V4-Flash-7L"
python3 scripts/seven_layer/matrix.py prepare --work "$WORK"
python3 scripts/seven_layer/build.py --work "$WORK"
```

The extractor checks each original shard against its pinned LFS size and SHA256,
keeps original tensor bytes and scales, excludes MTP, and writes the complete
parameter shape/dtype/hash/ownership manifest. The seven compression ratios are
`[0,0,4,128,4,128,4]`; layers 0–2 retain their hash-routing tables. BF16/FP32
trainability candidates are recorded; actual runtime `requires_grad` and frozen
state still need auditing. No QAT/FP8/BF16 semantic switch is made.

The source/image lock explicitly records the new RL-Align/verl main revision
rather than reusing the old recipe's PASS. The historical paired patches are not
silently inherited. No external vLLM, Megatron or AITER files are modified.

## Recoverable resource gate

```bash
python3 scripts/seven_layer/watch.py \
  --output "$WORK/evidence/watch" --ready "$WORK/ready.json"
```

Use tmux or the host's persistent task service. Only one watcher may hold its
`monitor.lock`. It stores PID, status, 30-second timestamped GPU process/memory/
utilization samples, disk usage, process commands, parents, start times and
container cgroups. Zero utilization alone is never idle. It never kills tasks.
A local lock prevents duplicate self-launch; it does not reserve GPUs from other
users. No scheduler was present on the measured host.

`ready.json` must contain a real command argv and a nonempty list of immutable
prerequisites (`path`, `sha256`). After hashing them, the gate requires all eight
GPUs idle in two checks separated by five seconds and checks again immediately
before starting. The diagnostic coordinator repeats the resource guard between
stages and exposes only each phase's requested devices through DRM/KFD mapping.
It stops only its own labelled container on timeout. Re-running a watcher whose
`launched.json` exists is rejected to prevent duplicate execution.

The current executable continuation is:

```bash
python3 scripts/seven_layer/matrix.py diagnose --work "$WORK" --image sha256:IMAGE_ID
```

It attempts real GPU startup/model calls under each unchanged target topology,
records original errors, and separates fixed-TP PP/EP diagnostics after M5
failure. It **does not run optimizer updates**. Full fixed-weight validation,
three effective GRPO updates and the target load remain blocked until their
missing compute/sync/lifecycle paths are implemented and independently verified.
No 200-step or multi-node command is enabled here.

## Independent raw-score replay

RL-Kernel provides a separate-process verifier:

```bash
python -m rl_engine.integrations.common.response_bitwise \
  --suite "$WORK/evidence/fixed-input-suite.json" \
  --left /path/to/actor.json --right /path/to/rollout.json \
  --snapshot-sha256 CHECKPOINT_SHA256 --output verification.json
```

The manifests require original little-endian FP32 files, identical snapshot and
logical input-suite hashes, temperature 1, top-k -1 and top-p 1. Every expected
response token is compared, without tolerance, rounding or dtype conversion.
Missing/duplicate samples, stale snapshot, nonfinite scores and shared score-file
inodes fail. Signed zero differences and a one-bit perturbation are detected.
This verifies numerical artifacts; it cannot establish real engine execution,
rank placement, optimizer continuity or synchronization by itself.

The prepared suite includes single samples, multi-sample AB/BA, repetition,
lengths around C4/C128 and recent-window boundaries, and >512 compressed-key
candidates. None of these branches is covered until the actual model executes it.

## Metrics and status

Report all M1–M5 fixed-weight, three-update and target-load outcomes separately.
Use N/A for unexecuted metrics. Response tokens are counted once per logical
sample; TP/PP/EP ranks never multiply the numerator. Report throughput both per
active GPU and per reserved eight-card machine, and include actual synchronization
and stage switching in end-to-end time. Disabled reference/KL fields are N/A.
Three-step means are short-run observations, not steady-state performance.
No baseline or speedup claim is made.
