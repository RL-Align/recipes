# Seven-layer DSV4: first completed paired GPU scoring failure

This preserves the first actual actor PP2/TP1 → rollout PP1/TP1 scoring attempt,
not a passing run. Both loaded original continuous layers0–6; actor ownership4/3.
N1 full independent verification **FAIL**: original I64 hash routing tables were
silently loaded/exported as I32. A separately labelled raw-score diagnostic
compares every response token:57 expected/57 compared,30 mismatches,
mean_abs_diff0.0609356729607833,max_abs_diff0.4053459167480469, all finite.
Short single inputs matched; long and mixed-length inputs differed. The raw
score bytes, input suite, per-case bit differences, manifests and logs are saved.
They are never promoted to full metadata or group acceptance.

| Outcome | N1 | N2 | N3 | N4 | N5 |
|---|---|---|---|---|---|
| Fixed-score independent verifier |FAIL|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|
| Valid GRPO updates |0/3|0/3|0/3|0/3|0/3|
| DAPO8×16,1024/7168 |NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|NOT_RUN|

Logical response token counts exclude prompt/padding tokens and rank duplicates.
Timing/throughput, reward and KL remain N/A here; no training step completed.
Future throughput must give both active-GPU and whole-eight-card denominators,
with lifecycle/sync in end-to-end time. No baseline/speedup/200-step claims.
The environment shares a specifically verified inactive GPU0 root context;
unknown/active selected occupants still fail the resource gate.

The original dtype retention fix and selective module-boundary traces are now
committed in existing drafts RLK#863 and verl#5. New-runtime-health.json records
actual GPU0 matmul and eight-rank RCCL collective PASS in the rebuilt image;
this does not qualify model or N5 EP4 correctness. The new N1 tracing attempt is
ongoing, not covered by this preserved failed attempt. No external framework
sources were edited. N5 has only a CPU-tested local packed-expert partition/load
component; real EP4 dispatch/compute/backward/synchronization remains incomplete.

Exact image/source pins are in each provenance file. The sole active entry is:

```bash
WORK=/scratch/rl-kernel/vensen/dsv4-7l-matrix-20261125T1052Z
RLK_TRACE_AUDIT_PATTERN=single-2047 python3 "$WORK/runtime/drivers/16696ee307d03ce1a87c1928ca9e80a04467301f/scripts/seven_layer/orchestrate.py" \
 --work "$WORK" --runtime-lock "$WORK/runtime/runtime-16696ee307d0.lock.json" \
 --resource-policy "$WORK/evidence/shared-GPU0-evaluation/policy.json" --groups N1
```

Do not run a duplicate while the existing tmux job holds the task lock. Check
experiment logs/runs-N for the live attempt. Large weights remain on the node.
