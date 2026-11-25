> Original M matrix revised on 2026-11-25. Historical CPU/startup-only evidence; not N1–N5 GPU acceptance. See ../20261125-N.

# Seven-layer DSV4 single-node cross-TP/PP/EP validation

Status snapshot: 2026-11-25T11:13:36.255142+00:00 (UTC). **BLOCKED; no GPU model execution in this run.**

GPU0 is held by a root-owned single-card process; SIGTERM returned EPERM, sudo requires a password, and system Docker is inaccessible. Strict TP/EP computation, resharding and phased GRPO remain incomplete.

The requested work is incomplete. CPU checks, model extraction and startup scaffolding do not establish the requested five-group numerical or training acceptance. Historical #711 is already merged and does not include these new draft changes. Its old four-layer results and failures are preserved.

## Target versus actual topology

| Group | Actor PP/TP/EP/ETP | Actor GPUs | Rollout PP/TP/EP | Rollout GPUs | Peak physical | Actual execution | Fixed weights | Effective updates | Target 8×16 load |
| --- | --- | ---: | --- | ---: | ---: | --- | --- | --- | --- |
| M1 | 1/1/1/1 | 1 | 1/2/1 | 2 | 2 | NOT RUN | BLOCKED | 0/3 | NOT RUN |
| M2 | 1/2/1/2 | 2 | 1/4/1 | 4 | 4 | NOT RUN | BLOCKED | 0/3 | NOT RUN |
| M3 | 1/4/1/4 | 4 | 1/8/1 | 8 | 8 | NOT RUN | BLOCKED | 0/3 | NOT RUN |
| M4 | 1/8/1/8 | 8 | 1/1/1 | 1 | 8 | NOT RUN | BLOCKED | 0/3 | NOT RUN |
| M5 | 2/4/4/1 | 8 | 1/8/1 | 8 | 8 | NOT RUN | BLOCKED | 0/3 | NOT RUN |

All dense DP, expert DP and CP are 1. One rollout instance. The experiment requires all eight cards before launch; physical actor and rollout allocations are reused in serial phases and are never added. Planned mappings are in [planned-rank-map.json](planned-rank-map.json); no live rank/expert mapping has been observed.

M5 planned PP ownership is 0–3 / 4–6, embedding on PP0, output head/final norm/mHC contraction on PP1. Its EP4 ranks own expert ranges 0–63, 64–127, 128–191 and 192–255. Megatron ordinary/expert DP arithmetic gives 1 in all five targets. Hidden/head/indexer/group/vocabulary/expert dimensions divide the requested axes. SP is requested for TP>1. Packed quantized slicing, actual collectives, memory capacity and backward compatibility are not qualified.

## Hardware, versions and model

One AMD node, 8 × Instinct MI300X VF, gfx942, 205,822,885,888 bytes VRAM per card. PCI buses: 83:00.0, 8b:00.0, 93:00.0, 9b:00.0, a3:00.0, ab:00.0, b3:00.0, bb:00.0. Host Ubuntu 24.04.5, kernel 6.8.0-146-generic, ROCm 10.1.0, rootless Docker. About 1.8 TiB host RAM and 19 TiB free /scratch at preparation. Host /dev/shm is about 945 GiB; planned diagnostic containers use private 16 GiB shm. No scheduler was found; the local flock is not a cross-user reservation. Existing containers and shared services were not stopped.

| Source | Actual clean commit |
| --- | --- |
| RL-Kernel | `a275b3facfbb020eb0fcc062d93c89f9eefff9d7` |
| verl | `53c9dd1d659390b63186ae53825872842e7addf2` |
| vllm | `c07d58dc82d75f7d1a947b3264b460384ec3e18d` |
| Megatron-LM | `97b50bdd936022ef58031c35db6d10581c181be0` |

Built recipe code commit: `237d6cb215411794c182ac84a680bb6e1ee73f58`. Derived image ID from `docker image inspect`: `sha256:dee9aba34bb3a622e2b875fa8896519ae8cf6da18be961822247d073a304958d`. Base digest: `vensenmu/rl-kernel@sha256:f3bf76c2d3863bf5b3f6a912d106a6eb2042ad9898c54f60b1e9015d6aeb225f`. The new verl pin explicitly includes the current fork-main delta from historical d154f5; old paired recipe patches are not silently applied. Full source trees, actual import paths and package versions are in [runtime-audit.json](runtime-audit.json) and [runtime.lock.json](runtime.lock.json). No external dependency source was modified; all four runtime source trees are clean.

Original source: `deepseek-ai/DeepSeek-V4-Flash` revision `60d8d70770c6776ff598c94bb586a859a38244f1`, original continuous layers 0–6. Extracted 11,002 tensors / 27,164,886,180 bytes. Whole file SHA256: `62cbd4407a498f44552b1d2216b1f1a6b7cc5fac105669b4c0a3f8f2e73bebfe`. Every source shard passed pinned LFS hashes; every output tensor was independently read back and hashed. Original scales, FP4 codes, FP8 tensors, hash tables and tokenizer remain intact. Compression ratios: `[0,0,4,128,4,128,4]`. No repeated layers, random weights or dtype conversion. MTP is excluded.

Original packed FP4/FP8 weights/scales remain intended frozen state. The 129 floating trainability candidates require a live `requires_grad`/optimizer audit; this metadata is not evidence of effective training. No QAT/FP8/BF16 training-mode switch or trainability expansion is claimed. Full metadata: [model-parameters.json.gz](model-parameters.json.gz); extraction/readback: [model-readback.json](model-readback.json).

## Performance and consistency

The table borrows the metric structure of #377. It contains no measured performance values because no GPU model computation ran.

| Metric | M1 | M2 | M3 | M4 | M5 |
| --- | --- | --- | --- | --- | --- |
| Rollout time (s) | N/A | N/A | N/A | N/A | N/A |
| Rollout throughput, active GPUs (response tok/GPU/s) | N/A | N/A | N/A | N/A | N/A |
| Rollout throughput, reserved 8 GPUs (response tok/GPU/s) | N/A | N/A | N/A | N/A | N/A |
| Longest-sample throughput (response tok/s) | N/A | N/A | N/A | N/A | N/A |
| Actor logprob time (s) | N/A | N/A | N/A | N/A | N/A |
| Reference logprob time (s) | N/A | N/A | N/A | N/A | N/A |
| Actor train time (s) | N/A | N/A | N/A | N/A | N/A |
| Actor train throughput (prompt+response tok/s) | N/A | N/A | N/A | N/A | N/A |
| Actor train throughput, active GPUs (prompt+response tok/GPU/s) | N/A | N/A | N/A | N/A | N/A |
| Actor train throughput, reserved 8 GPUs (prompt+response tok/GPU/s) | N/A | N/A | N/A | N/A | N/A |
| Total train time (s) | N/A | N/A | N/A | N/A | N/A |
| Weight synchronization time (s) | N/A | N/A | N/A | N/A | N/A |
| GPU offload/restore and stage switch (s) | N/A | N/A | N/A | N/A | N/A |
| Independent audit time (s) | N/A | N/A | N/A | N/A | N/A |
| End-to-end step time (s) | N/A | N/A | N/A | N/A | N/A |
| Mean response length (tokens) | N/A | N/A | N/A | N/A | N/A |
| Mean raw reward | N/A | N/A | N/A | N/A | N/A |
| Mean KL loss | N/A | N/A | N/A | N/A | N/A |
| Expected response tokens | N/A | N/A | N/A | N/A | N/A |
| Compared response tokens | N/A | N/A | N/A | N/A | N/A |
| FP32 word mismatch count | N/A | N/A | N/A | N/A | N/A |
| Mean absolute logprob difference | N/A | N/A | N/A | N/A | N/A |
| Max absolute logprob difference | N/A | N/A | N/A | N/A | N/A |
| Effective optimizer updates | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| Independent verifier on model scores | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |

Throughput definitions: rollout uses response tokens only; actor training uses nonpadding prompt+response tokens once per logical sample. Report both tokens/(stage time × active GPUs) and tokens/(stage time × reserved 8 GPUs); ranks never multiply tokens. Longest-sample throughput uses maximum response length / rollout time. End-to-end must include admission closure, offload/restore or process recovery, synchronization/readback, auditing and training. Reference/KL are not measured and remain N/A. No baseline, speedup, stable throughput or statistical-significance claim. Future three-step means must be labelled short tests.

Pinned real dataset acquired and LFS-verified: DAPO-Math-17k revision `65877096c24ffa7abc4e4fa5edb95cf3413a5674`, 299,363,855 bytes, SHA256 `534375d6bb8630d22ab46a56e11f2ffec1d288d8f7d04099bc82d68948705941`. Seven-layer training preprocessing and GPU use are NOT RUN. See [data.lock.json](data.lock.json). Copy-task and real-math training results are both NOT RUN.

## Component coverage

| Component | Original asset / CPU preparation | Actual GPU execution |
| --- | --- | --- |
| Embedding, output head and normalization | Assets/layout inspected; see metadata | NOT COVERED |
| Attention and RoPE | Assets/layout inspected; see metadata | NOT COVERED |
| C4/C128 compression and state boundaries | Assets/layout inspected; see metadata | NOT COVERED |
| Indexer and actual Top-512 pruning | Assets/layout inspected; see metadata | NOT COVERED |
| Hash routing (0–2) and learned routing (3–6) | Assets/layout inspected; see metadata | NOT COVERED |
| Routed/shared experts | Assets/layout inspected; see metadata | NOT COVERED |
| mHC | Assets/layout inspected; see metadata | NOT COVERED |
| Effective backward and parameter updates | Assets/layout inspected; see metadata | NOT COVERED |
| PP forward/backward transport | Assets/layout inspected; see metadata | NOT COVERED |
| EP dispatch and combine | Assets/layout inspected; see metadata | NOT COVERED |

Fixed input suite includes short/long inputs, C4/C128/recent-window boundaries, more than 512 compressed-key candidates, single samples, AB/BA and repeats. These are prepared inputs, not executed coverage. No same-weight cross-configuration replay, model-score perturbation control, training-batch token comparison, optimizer step, policy synchronization, post-v3 validation or target-load attempt has run.

## Failures, repairs and remaining work

| Finding | Action and recheck | Status |
| --- | --- | --- |
| Command sandbox failed before execution (`bwrap` RTM_NEWADDR) | Approved out-of-sandbox execution used; commands and original failure retained | Work continued |
| GPU0 process has 0% utilization but still owns GPU memory/device | Current identity/start time/container and single-card mapping rechecked; SIGTERM returns EPERM; sudo needs password; system Docker socket denies access | BLOCKED; no task killed |
| Four-layer 3/1 PP ownership was hard-coded | Add native seven-layer 4/3 layout, disjoint endpoints and global pair ownership; pinned Bridge and historical layout regressions pass | CPU checked only |
| Independent raw-score replay needed full coverage/bit checks | Add separate verifier; detect one-bit perturbation, signed zero, missing samples, wrong snapshot and same-file cache | CPU controls pass; model evidence NOT RUN |
| First final-image check used OCI config digest instead of daemon image ID | Preserve exit-125 logs; use actual inspect ID from runtime lock and rerun | Final image: 29 + 3 CPU tests pass |
| Strict model TP/EP paths are incomplete | Keep fail-closed guards in packed actor loader/mHC and rollout output/group projection; no TP1 substitution | Unresolved implementation work |
| Cross-topology export/resharding/readback and phased GRPO lifecycle are incomplete | No cached rollout-score substitution or optimizer-history reset is introduced; real startup probes and serial resource guards added separately | Unresolved implementation work |
| Quantized vLLM sleep/wake capability | Inspected pinned level-1 backend suspend/resume and level-2 discard logic; runtime probe prepared | NOT GPU TESTED; no residency claim |
| M5 requires isolated PP/EP diagnostics if it fails | Prepare separate TP4 control, TP4/PP2 and TP4/EP4 actor diagnostics, distinct from M1–M5 | NOT RUN |

No tolerances, resource gates, rewards or advantages were relaxed. The target real-data load remains 8 prompts × 16 responses, prompt cap 1024, response cap 7168, three effective updates after correctness. No smaller diagnostic would count as that load. No 200-step or multi-node run was started.

## Persistent continuation and reproduction

Experiment directory: `/scratch/rl-kernel/vensen/dsv4-7l-matrix-20261125T1052Z`. Watcher PID: `3086958`. Tmux socket/session: `dsv4-7l-20261125` / `resource-watch`. Status: `evidence/watch/status.json`; timestamped samples: `evidence/watch/gpu-samples.jsonl`; watcher log: `logs/watch.log`. A hashed `ready.json` is armed. GPU0 still blocks the eight-card gate at this snapshot.

The executable continuation is genuine GPU model startup diagnostics only. It is not a complete fixed-weight/three-update GRPO runner. Resource release alone does not complete the missing numerical and lifecycle implementation. The watcher makes two checks five seconds apart, then an immediate final check; the coordinator guards all eight cards again between stages, uses each target phase scale, and retains actual process errors. A local lock cannot reserve against other users.

```bash
tmux -L dsv4-7l-20261125 attach -t resource-watch
cat /scratch/rl-kernel/vensen/dsv4-7l-matrix-20261125T1052Z/evidence/watch/status.json
python3 /scratch/rl-kernel/vensen/dsv4-7l-matrix-20261125T1052Z/recipes/scripts/seven_layer/matrix.py diagnose \
  --work /scratch/rl-kernel/vensen/dsv4-7l-matrix-20261125T1052Z \
  --image sha256:dee9aba34bb3a622e2b875fa8896519ae8cf6da18be961822247d073a304958d
```

Do not invoke the diagnostic command manually while its watcher may launch it. Use the single watcher and inspect its launch/status receipts. If a prior launch exists, retain it and use a new run identity for an intentional repeat.

Rebuild and final-image CPU validation:

```bash
python3 scripts/seven_layer/build.py --work NEW_WORK
docker run --rm -w /opt/dsv4/RL-Kernel --entrypoint /opt/dsv4-venv/bin/python IMAGE_ID \
  -m pytest -q -p no:cacheprovider --confcutdir=tests/integrations/common \
  tests/integrations/common/test_native_pipeline_partition.py \
  tests/integrations/common/test_response_bitwise.py
```

Draft changes: [RL-Kernel #863](https://github.com/RL-Align/RL-Kernel/pull/863), [RL-Align/verl #5](https://github.com/RL-Align/verl/pull/5), plus the companion recipe branch. Neither framework PR is merged. No human-review claim; developed with Codex assistance. CSV/JSON use null/blank for unmeasured metrics; the known completed optimizer-update count is zero.

Large weights, original data, failed logs, checkpoint manifests and the running resource monitor remain in the persistent experiment directory. The files in this report directory are small evidence/metadata, not weights or a historical PASS transfer.
