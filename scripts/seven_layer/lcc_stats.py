# SPDX-License-Identifier: MIT
"""Per-update long-context (lcc) statistics and cross-group identity for finished run roots.

Usage: python3 lcc_stats.py <run_root>:<group> [<run_root>:<group> ...]

Stdlib only; reads the recorded evidence and never synthesizes a value. For each update it reports the
generated batch the update trained on (prompt/response lengths, truncation at the response cap, sha of all
response ids), raw rewards and GRPO advantages from the actor receipt, phase seconds and the post-update
fresh/fixed audits. With several groups it states whether the per-update policy digest, reward sha,
response-token sha and token count are identical across them.
"""

import hashlib
import json
import statistics
import sys
from pathlib import Path

RESPONSE_CAP = 7168


def _sha(value):
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()[:12]


def update_rows(run_root, group, dataset="lcc", cap=RESPONSE_CAP):
    base = Path(run_root) / group / dataset
    rows = []
    for step in json.loads((base / "steps.json").read_text()):
        update = step["update"]
        batch = json.loads((base / f"v{update - 1}-generate" / "generated-batch.json").read_text())
        samples = batch["samples"]
        lengths = [len(s["response_ids"]) for s in samples]
        prompts = [len(s["prompt_ids"]) for s in samples]
        receipt = json.loads((base / f"update-{update}" / "reward-advantage-rank-0.json").read_text())
        rewards, advantages = receipt["raw_rewards"], receipt["advantages"]
        fresh = step["fresh_verification"]
        rows.append(dict(
            update=update,
            samples=len(samples),
            prompt_len=[min(prompts), max(prompts)],
            response_tokens=step["response_tokens"],
            response_len_mean=round(statistics.mean(lengths), 1),
            response_len_range=[min(lengths), max(lengths)],
            truncated=f"{sum(n >= cap for n in lengths)}/{len(samples)}",
            response_token_sha=_sha([s["response_ids"] for s in samples]),
            reward_sha=_sha(rewards),
            reward_mean=round(statistics.mean(rewards), 6),
            reward_std=round(statistics.pstdev(rewards), 6),
            advantage_nonzero=f"{sum(a != 0 for a in advantages)}/{len(advantages)}",
            advantage_mean_abs=round(receipt["advantage_mean_abs"], 4),
            advantage_std=round(receipt["advantage_std"], 4),
            advantage_range=[round(min(advantages), 4), round(max(advantages), 4)],
            generation_seconds=round(step["generation"]["seconds"], 1),
            actor_seconds=round(step["actor"]["seconds"], 1),
            policy_digest=fresh["policy_state_digest"][:16],
            fresh=fresh["status"],
            fresh_compared_tokens=fresh["compared_tokens"],
            fresh_mismatch=fresh["mismatch_count"],
            fixed=step.get("fixed_verification", {}).get("status"),
        ))
    return rows


IDENTITY_KEYS = ("update", "policy_digest", "reward_sha", "response_token_sha", "response_tokens")


def identity(groups):
    """True when every group has the same per-update identity tuple; None for fewer than two groups."""
    if len(groups) < 2:
        return None
    signatures = [[tuple(row[k] for k in IDENTITY_KEYS) for row in rows] for rows in groups.values()]
    return all(sig == signatures[0] for sig in signatures[1:])


def main(argv):
    if not argv:
        raise SystemExit(__doc__)
    groups = {}
    for item in argv:
        root, group = item.rsplit(":", 1)
        groups[f"{group}@{Path(root).name}"] = rows = update_rows(root, group)
        print(f"== {group} {Path(root).name}")
        for row in rows:
            print(json.dumps(row))
    verdict = identity(groups)
    if verdict is not None:
        print("cross-group identity:", "IDENTICAL" if verdict else "DIFFER")


if __name__ == "__main__":
    main(sys.argv[1:])
