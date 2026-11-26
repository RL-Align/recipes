# SPDX-License-Identifier: MIT
"""Byte-compare actor and rollout boundary traces; report the first differing boundary.

Runs inside the runtime image (needs torch and examples.operators.dsv4_4l.trace_store).
Diagnostic only: never acceptance evidence.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def leaves(value, path=""):
    import torch

    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, (tuple, list)):
        for i, v in enumerate(value):
            yield from leaves(v, f"{path}/{i}")
    elif isinstance(value, dict):
        for key, v in value.items():
            yield from leaves(v, f"{path}/{key}")


def rows_of(tensor):
    return tensor.reshape(tensor.shape[0], -1) if tensor.dim() else tensor.reshape(1, 1)


def scalar(value):
    try:
        return value.item()
    except (RuntimeError, TypeError):
        return float(value.float().item())  # dtypes without .item(); bits are reported separately


def compare_leaf(a, b):
    """Return (status, detail) for one tensor pair; bytes decide, never values."""
    import torch

    detail = dict(
        actor_dtype=str(a.dtype), rollout_dtype=str(b.dtype), actor_shape=list(a.shape), rollout_shape=list(b.shape)
    )
    if a.dtype != b.dtype:
        return "dtype-mismatch", detail
    x, y = rows_of(a.detach().cpu().contiguous()), rows_of(b.detach().cpu().contiguous())
    if x.shape[1:] != y.shape[1:]:
        return "shape-mismatch", detail
    status = "equal"
    if x.shape[0] != y.shape[0]:
        # Different token counts: compare the common leading rows only, flagged as such.
        n = min(x.shape[0], y.shape[0])
        x, y = x[:n], y[:n]
        status = "equal-common-rows"
        detail["compared_rows"] = n
    size = x.element_size()
    xb = x.reshape(-1).view(torch.uint8).reshape(-1, size)
    yb = y.reshape(-1).view(torch.uint8).reshape(-1, size)
    bad = (xb != yb).any(-1)
    detail["elements"] = int(bad.numel())
    detail["mismatch_count"] = int(bad.sum())
    if detail["mismatch_count"]:
        i = int(bad.nonzero()[0])
        cols = x.shape[1]
        detail.update(
            first_index=i,
            first_row=i // cols,
            first_col=i % cols,
            actor=scalar(x.reshape(-1)[i]),
            rollout=scalar(y.reshape(-1)[i]),
            actor_bits=bytes(xb[i].tolist())[::-1].hex(),
            rollout_bits=bytes(yb[i].tolist())[::-1].hex(),
        )
        status = "differs"
    return status, detail


def pair_events(actor_events, rollout_events):
    """Pair by (layer, name, occurrence) in actor forward order; return pairs and unmatched sets."""
    right = defaultdict(list)
    for e in rollout_events:
        right[(e["layer"], e["name"])].append(e)
    used = defaultdict(int)
    pairs, missing_in_rollout = [], defaultdict(int)
    for e in actor_events:
        key = (e["layer"], e["name"])
        n = used[key]
        used[key] += 1
        if n < len(right.get(key, [])):
            pairs.append((key, n, e, right[key][n]))
        else:
            missing_in_rollout[key] += 1
    missing_in_actor = {k: len(v) - used.get(k, 0) for k, v in right.items() if len(v) > used.get(k, 0)}
    return pairs, dict(missing_in_rollout), missing_in_actor


def compare(actor_dirs, rollout_dir, load):
    actor_events = []
    for d in actor_dirs:
        for e in json.loads((d / "manifest.json").read_text()):
            actor_events.append(dict(e, _dir=d))
    rollout_events = [dict(e, _dir=rollout_dir) for e in json.loads((rollout_dir / "manifest.json").read_text())]
    pairs, miss_r, miss_a = pair_events(actor_events, rollout_events)
    records = []
    for (layer, name), occurrence, a, r in pairs:
        xl, yl = dict(leaves(load(a["_dir"] / a["file"]))), dict(leaves(load(r["_dir"] / r["file"])))
        for path in xl.keys() & yl.keys():
            status, detail = compare_leaf(xl[path], yl[path])
            records.append(
                dict(
                    order=len(records),
                    layer=layer,
                    name=name,
                    occurrence=occurrence,
                    leaf=path,
                    actor_dir=a["_dir"].name,
                    status=status,
                    **detail,
                )
            )
        for path in sorted(xl.keys() ^ yl.keys()):
            records.append(
                dict(order=len(records), layer=layer, name=name, occurrence=occurrence, leaf=path, status="unmatched-leaf")
            )
    fmt = lambda d: [dict(layer=k[0], name=k[1], count=v) for k, v in d.items()]  # noqa: E731
    return dict(
        scope="diagnostic trace comparison only; never acceptance evidence",
        paired_events=len(pairs),
        first_difference=next((r for r in records if r["status"] == "differs"), None),
        first_non_equal=next((r for r in records if not r["status"].startswith("equal")), None),
        unmatched_actor_names=fmt(miss_r),
        unmatched_rollout_names=fmt(miss_a),
        records=records,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--actor", type=Path, nargs="+", required=True, help="actor trace dirs in PP/chunk order")
    ap.add_argument("--rollout", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    from examples.operators.dsv4_4l.trace_store import load_capture

    result = compare(args.actor, args.rollout, load_capture)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("FIRST_DIFFERENCE", json.dumps(result["first_difference"]), flush=True)


if __name__ == "__main__":
    main()
