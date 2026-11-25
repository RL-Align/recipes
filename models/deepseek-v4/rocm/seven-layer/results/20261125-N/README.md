# Seven-layer DSV4 actor-TP1 / rollout-TP1/2/4/8 validation

Status: **BLOCKED — no revised-matrix GPU phase or optimizer update executed.**
This continues `/scratch/rl-kernel/vensen/dsv4-7l-matrix-20261125T1052Z`.
The original actor-TP>1 M matrix was withdrawn, its watcher stopped, and its ready
file archived. Original plans, model/data checksums and CPU results remain intact.
No old result or startup diagnostic is counted as N acceptance.

| Item | N1 | N2 | N3 | N4 | N5 |
| --- | --- | --- | --- | --- | --- |
| Actor PP/TP/EP/ETP | 2/1/1/1 | 2/1/1/1 | 2/1/1/1 | 2/1/1/1 | 2/1/4/1 |
| Dense DP / expert DP | 1/1 | 1/1 | 1/1 | 1/1 | **4/1** |
| Actor GPUs | 2 | 2 | 2 | 2 | 8 |
| Rollout PP/TP/EP | 1/1/1 | 1/2/1 | 1/4/1 | 1/8/1 | 1/8/1 |
| Rollout GPUs | 1 | 2 | 4 | 8 | 8 |
| Implementation | experimental | experimental | experimental | experimental | EP4 unimplemented |
| GPU fixed-weight score | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Actual effective correctness updates | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| Actual target-load updates | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| Independent GPU verifier | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Current status | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED / later gate |

Hardware: one eight-card AMD Instinct MI300X VF gfx942 node; each card reports
205,822,885,888 bytes VRAM. CP1 and one rollout instance throughout. Seven original
continuous layers 0–6, compression `[0,0,4,128,4,128,4]`, PP2 ownership 4/3,
embedding on PP0 and final norm/head/mHC contraction on PP1. Actor and rollout
serially reuse the same physical cards; workers are rebuilt with explicit complete
recovery, not described as resident. MTP and multi-node execution are excluded.

## Fixed source and runtime

- RL-Kernel `94f09c3b948bf59d396b5492d20853c1c52bfbad`, tree `075353e8f208d8a49cfaae32ef18404365c08b96`.
- RL-Align/verl `507f46e1e1a8fe9cb4c505bda95df63f0d525b2b`, tree `460f533c2e764ce7f05107d83e22c6e77f656c06`.
- Recipe code `73015732f8083fa2104c3ec061e09be18d8d319c`, tree `bff89b4d9b084d06388ba9983508cb2f58780f50`.
- Image `rl-align/dsv4-seven-layer:actor-tp1-73015732f808`; actual Docker image ID `sha256:8f1f79c09146d86474f2568a22c61bb1fe60fabe621ce1399a40ef18c9243932`.
- vLLM `c07d58dc82d75f7d1a947b3264b460384ec3e18d`; Megatron `97b50bdd936022ef58031c35db6d10581c181be0`; external framework source unchanged and clean.
- Torch `2.13.0+rocm10.0.0`, HIP `7.15.26333`, Bridge `0.7.0`, Transformer Engine `2.21.0.dev0`, AITER `0.1.29`, Triton `3.8.0+git669b31ac`, Ray `2.60.0`.
- Original model revision `60d8d70770c6776ff598c94bb586a859a38244f1`; extracted seven-layer SHA256 `62cbd4407a498f44552b1d2216b1f1a6b7cc5fac105669b4c0a3f8f2e73bebfe`, 27,164,886,180 bytes / 11,002 tensors.
- DAPO-Math-17k revision `65877096c24ffa7abc4e4fa5edb95cf3413a5674`; data SHA256 `534375d6bb8630d22ab46a56e11f2ffec1d288d8f7d04099bc82d68948705941`.

Original BF16/F32 tensors, packed FP4 weights, FP8 weights/scales and I64 routing
metadata are retained. Score storage/comparison uses original FP32 bit patterns,
temperature 1, top-k -1, top-p 1; no tolerance, rounding or reduced precision is
accepted. CPU extraction/readback does not prove GPU loader/compute semantics.

## Implemented code versus verified behavior

Experimental code now connects native actor forward, native PP1 rollout TP shard
loading, original-vocabulary FP32 head gathering, actor policy export and actual
rollout tensor readback. Phase training calls native verl generation, unchanged
copy/DAPO reward, GRPO advantage, PPO loss, Megatron backward and optimizer.
Training actor logprobs are independently recomputed; actual PPO forward scores
and every valid response token are retained and compared. Zero advantages or no
parameter change fail. Each update restores/retains optimizer, scheduler, model
buffers, RNG and version; every update including the third gets fresh generation
and fixed-input audits. Separate independent verifiers replay the raw evidence.
These are implemented paths awaiting GPU execution, not completed GPU training.

N1–N4 use one common original weight snapshot for fixed scoring. Their subsequent
copy and DAPO trajectories are separate; weights after different trajectories
are never conflated. Target load remains 8 prompts ×16 responses, prompt cap1024,
response cap7168 and exactly three valid updates. OOM or flat rewards retain a
failure; no silent load reduction or reward alteration is enabled.

Final fixed-image CPU checks: **34 RLK +3 verl +3 recipe =40 passed**, with no GPU
devices exposed. RLK includes the pinned native Bridge uneven PP layout, real
CPU/Gloo state transfer, TP8 original-vocabulary partition/padding/corruption and
strict signed-zero/one-bit/nonfinite score controls. verl includes three actual
CPU Adam updates with exact reconstruction of state and scheduler; this is not
DSV4 GRPO acceptance. Recipe checks N1–N5 topology and occupied-GPU rejection.
The real executor CLI is separately checked. Prior failed CPU attempts remain
under the experiment logs: ROCm platform auto-probe, cross-repository pytest
package collision, and a generic report-module import collision. The import
collision was minimally repaired and the committed image rebuilt.

## Resource blocker and automatic continuation

The measured independent GPU0 task is root-owned `profile_operator.py`, PID
3131740, started 2026-11-12 14:33:23 UTC in system Docker container
`badec208aec1442118231db8e03f9c8c40bff4060189880fba6d93a563bc5fd9`.
The saved cleanup attempt first rediscovered current GPU mappings and identity,
confirmed it uses GPU0 only, then sent SIGTERM; the OS returned EPERM. `sudo -n`
requires a password; the system Docker socket is inaccessible to this account.
A privileged controlled session is required to release this task. GPU utilization
zero is not treated as idle. No other user's GPU task, shared Ray/Docker service,
GPU reset or reboot was used.

The new watcher in tmux socket `dsv4-7l-20261125`, session `resource-watch-N`,
continues timestamped process/memory/utilization sampling every 30 seconds.
`ready-N.json` pins the final image and immutable assets/scripts; the retired
`ready.json` is absent. Once all eight cards pass repeated idle checks, it launches
`execute.py`, not a startup-only diagnostic. Per-phase guards and local locks
prevent duplicate self-launch, but do not reserve GPUs against other users.
N5 remains a separate EP4 gate and never blocks N1–N4.

## Evidence and metric definitions

[Full N1–N5 metric table](RESULTS.md), [JSON](metrics.json),
[copy CSV](metrics-copy.csv), [DAPO CSV](metrics-dapo.csv), [steps](steps.csv),
[components](component-coverage.csv), [runtime lock](runtime.lock.json),
[runtime audit](runtime-audit.json), [model readback](model-readback.json),
[original parameter manifest](model-parameters.json.gz),
[data lock](data.lock.json), [request lock](requests.lock.json),
[latest GPU sample](gpu-latest.json), [cleanup attempt](GPU0-cleanup-N-attempt.json),
[old-launcher stop receipt](disable-old-launcher.json).

Unmeasured times, throughputs, raw scores, expected/compared tokens, mismatch and
mean/max difference, reward and KL are N/A, never invented zeros. Actual executed
updates are zero. Numerators count logical response tokens once, without rank
multiplication. Throughput denominators distinguish active phase GPUs and the
whole eight-card machine. End-to-end includes switching, recovery, sync and all
post-update audits. Native metrics remain in each future native-train-output.pt.
The report records timing totals across retained updates, not 200-step means.
All GPU components remain NOT_COVERED; prepared long inputs are not execution
proof. Cross-TP arithmetic/reduction bit invariance and N5 EP4 communication remain
unqualified. No baseline, speedup, 200-step result or significance is claimed.

Reproduce using the existing validated assets and committed image:

```bash
WORK=/scratch/rl-kernel/vensen/dsv4-7l-matrix-20261125T1052Z
python3 "$WORK/recipes/scripts/seven_layer/execute.py" --work "$WORK" \
  --image sha256:8f1f79c09146d86474f2568a22c61bb1fe60fabe621ce1399a40ef18c9243932
```

Use the existing resource watcher rather than starting a duplicate invocation.
Draft fixes: [RLK #863](https://github.com/RL-Align/RL-Kernel/pull/863),
[verl #5](https://github.com/RL-Align/verl/pull/5),
[recipes #3](https://github.com/RL-Align/recipes/pull/3). No PR is auto-merged.
