# Seven-layer DSV4: Q-up repair and remaining indexer failure

N1 full fixed-weight verifier remains **FAIL**, with57/57 response tokens compared,
12 mismatches, mean_abs_diff0.048596432334498355 and max_abs_diff0.33298683166503906.
Original rank/tensor readback now PASS, including preserved I64 hash tables.
Q-up repair reduced the prior30 failing tokens to12: all2047/2048/2049 prompt
boundary cases and mixed-batch short rows now match exactly. The four remaining
long rows (prompt2177, total2180, including reorder/repeat) fail. No valid native
GRPO update completed; all N1–N5 updates0/3 and all DAPO target loads NOT_RUN.

A real GPU replay proved identical original QNorm FP8 codes/scales and0 differences
across67,174,400 fixed-CK Q-up output elements against actor; the prior dynamically
selected native GEMM had4,367 BF16 differences. This is operator replay, not group
acceptance. Actual native local TP weights remain used by the rollout adapter.

A new targeted2177 trace proved layers0/1 match, then layer2 first differs at
position2051, where513 compressed keys first exceed Top512. Actual selected key
sets differ in129 query rows; first actor-only key276 versus rollout-only key19.
Raw BF16 indexer head weights match exactly. The eager actor BF16/Hadamard score
profile differs from native inference FP8 Q/K scoring. The new actor integration
independently reuses pinned production FP8 Q/K RoPE/C4 state/MQA/Top-k operators;
original BF16/F32 weights, scales, masks and metadata remain unchanged. Zero
configured indexer auxiliary loss remains zero, and main attention backward is
retained. No external source or actor TP restriction was changed. This new profile
is explicitly versioned in indexer-runtime-under-test.lock.json; GPU replay and
full-model acceptance are pending, never inferred from CPU branch checks.

Component JSON distinguishes observed forward module execution from numerical,
backward and EP acceptance. Both C4 and C128, actual Top-k overflow, hash and later
routing, routed/shared FFN, mHC, embedding/head/norm and PP forward paths have
saved execution evidence. Effective backward/PP backward, three valid updates,
post-update continuity and N5 EP4 remain NOT_RUN/incomplete. No MTP scope.

Logical response counts exclude prompts/padding/rank copies. Generation/training
performance/reward/KL are N/A because no training step completed. No baseline,
speedup,200-step or significance claims. Any later rate must distinguish active
GPU and whole8-card denominators, with lifecycle/sync/audit in end-to-end time.
The specifically verified GPU0 inactive root context remains a recorded shared
exception, not an exclusive performance environment. Resource checks remain on.

The sole current entry (do not duplicate the existing task-locked tmux job):

```bash
WORK=/scratch/rl-kernel/vensen/dsv4-7l-matrix-20261125T1052Z
RLK_TRACE_AUDIT_PATTERN=single-2177 python3 "$WORK/runtime/drivers/bfecbf40d9bf2d13c2bd36d7e06d3de0f70ffda7/scripts/seven_layer/orchestrate.py" \
 --work "$WORK" --runtime-lock "$WORK/runtime/runtime-bfecbf40d9bf.lock.json" \
 --resource-policy "$WORK/evidence/shared-GPU0-evaluation/policy.json" --groups N1 \
 --indexer-trace-run "$WORK/runs-N/20261125T150855Z-d07afa9a"
```

Existing drafts: RLK#863, verl#5, recipes#3. Current English conversation comment:
https://github.com/RL-Align/RL-Kernel/pull/711#issuecomment-7165721547
Large tensors/checkpoints remain in the experiment; hashes and raw score files
are published, without overwriting the earlier failed attempts.
