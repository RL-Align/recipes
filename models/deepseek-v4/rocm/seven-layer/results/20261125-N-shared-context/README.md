# Seven-layer DSV4 actor-TP1 / rollout-TP1/2/4/8 validation — shared-context progress

This supersedes the resource-blocked status, without rewriting historical results.
The exact GPU0 process identity, device mapping and three 30-second-spaced samples
qualify a narrow user-authorized inactive context exception. VRAM was stable at
502,554,624 bytes total /201,367,552 process bytes, CU occupancy0 and SDMA836.
The declared per-card peak budget is128GiB plus8GiB reserve. Unknown/active selected
GPU tasks still block; identity, memory/SDMA/compute changes stop our own tasks.
This environment is shared, not an exclusive performance run.

GPU0 actual BF16 compute and eight-rank RCCL all_reduce/broadcast/reduce_scatter/
all_to_all (three rounds each) passed in the pinned formal image and device
isolation. The current actor PP2/TP1 loaded original layers0–6 in4/3 ownership and
saved19 real fixed samples /57 response-token raw FP32 scores. That is actor-only
scoring, not an N-group PASS. A separately saved cross-device actor repeat control
compared57/57 tokens with0 mismatches; it is not train/rollout acceptance.

| Outcome | N1 | N2 | N3 | N4 | N5 |
|---|---|---|---|---|---|
| Current phase |RUNNING|PLANNED|PLANNED|PLANNED|IMPLEMENTING|
| Fixed actor/rollout scoring |pending|pending|pending|pending|not run|
| Valid GRPO updates |0/3|0/3|0/3|0/3|0/3|
| DAPO target load |not run|not run|not run|not run|not run|
| Dense DP / expert DP |1/1|1/1|1/1|1/1|4/1|

All times/throughputs/reward/KL and full-comparison token/mismatch/error metrics
remain N/A until actual qualified evidence exists. No baseline, speedup, 200-step
or significance claim. Token counts are logical responses, not rank multiplied;
active-GPU and whole8-card throughput denominators must remain separate.

Preserved real failures: private Ray socket path exceeded107 bytes; fixed short
per-container Ray paths now persist logs under each phase's ray-runtime. Native
8193-query sparse warmup exceeded the old1GiB workspace allowance, not GPU OOM;
explicit bounded4GiB phase allowance retains checks and arithmetic, with65 CPU
contract tests passed. Zero-GPU KFD contexts caused two ROCm APIs to disagree;
normalization plus five bounded observation attempts retains activity checks
and has10 CPU resource regression tests passed. No external framework source
was changed. New errors are actively investigated, not silently retried.

N1 is prioritized for its complete fixed-score and three-update chain before
N2/N3/N4. N5 local packed expert ownership/loading (64 of256 per EP rank) has2
CPU component tests passed; native EP4 communication/compute/backward/sync remains
unqualified and the engine EP1 guard remains. Generic eight-card all-to-all does
not qualify N5. Component execution/independent coverage remains pending the
full audit; initialized or warmup-only paths are not accepted model validation.

The version lock, exact raw actor scores, actual rank/source fingerprints,
health results, exception policy and stable samples, failures and regressions
are included here. Large policy/checkpoint weights remain in the experiment,
never in Git. Original seven-layer and dataset assets are reused without download.

Current reproducible entry (use the one existing tmux job, not a duplicate):

```bash
WORK=/scratch/rl-kernel/vensen/dsv4-7l-matrix-20261125T1052Z
python3 "$WORK/runtime/drivers/beda416e674d7faca7738b7fc0a17b3738dc7ea4/scripts/seven_layer/orchestrate.py" \
 --work "$WORK" --runtime-lock "$WORK/runtime/runtime-beda416e674d.lock.json" \
 --resource-policy "$WORK/evidence/shared-GPU0-evaluation/policy.json"
```

Existing draft PRs: [RLK#863](https://github.com/RL-Align/RL-Kernel/pull/863),
[verl#5](https://github.com/RL-Align/verl/pull/5),
[recipes#3](https://github.com/RL-Align/recipes/pull/3). None merged.
