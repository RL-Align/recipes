# SPDX-License-Identifier: MIT
"""Extract original layers 0-6 byte-for-byte, with immutable source and tensor hashes."""

import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import re
import shutil
import struct
import urllib.request
from pathlib import Path


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=120) as f:
        return json.load(f)


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def header(path):
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), n + 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-repo", default="deepseek-ai/DeepSeek-V4-Flash")
    ap.add_argument("--revision", default="60d8d70770c6776ff598c94bb586a859a38244f1")
    ap.add_argument("--reuse-cache", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError("never overwrite an earlier extraction: " + str(args.output))
    args.output.mkdir(parents=True)
    cache = args.output.parent / "source-shards"
    cache.mkdir(exist_ok=True)
    base = f"https://huggingface.co/{args.source_repo}/resolve/{args.revision}/"
    meta = fetch_json(
        f"https://huggingface.co/api/models/{args.source_repo}/revision/{args.revision}?blobs=true"
    )
    if meta["sha"] != args.revision:
        raise ValueError("resolved source revision differs")
    remote = {p["rfilename"]: p for p in meta["siblings"]}
    (args.output / "source-api.json").write_text(json.dumps(meta, indent=2) + "\n")
    index = fetch_json(base + "model.safetensors.index.json")
    original_config = fetch_json(base + "config.json")
    expected_globals = {
        "embed.weight",
        "head.weight",
        "norm.weight",
        "hc_head_fn",
        "hc_head_base",
        "hc_head_scale",
    }
    selected = {}
    for name, shard in index["weight_map"].items():
        m = re.match(r"^layers\.(\d+)\.", name)
        if m and int(m[1]) < 7 or name in expected_globals:
            selected[name] = shard
        elif not m and not name.startswith("mtp."):
            raise ValueError("unknown global checkpoint state: " + name)
    if set(k for k in selected if not k.startswith("layers.")) != expected_globals:
        raise ValueError("missing embedding/head/normalization/mHC global state")
    if {int(re.match(r"^layers\.(\d+)\.", k)[1]) for k in selected if k.startswith("layers.")} != set(
        range(7)
    ):
        raise ValueError("source lacks a complete original seven-layer range")
    shards = sorted(set(selected.values()))
    need = sum(remote[n]["lfs"]["size"] for n in shards if not (args.reuse_cache / n).exists())
    # Upper bound selected storage with complete shard sizes plus new downloads.
    capacity = need + sum(remote[n]["lfs"]["size"] for n in shards)
    if shutil.disk_usage(args.output).free < capacity + 10 * 1024**3:
        raise OSError("insufficient free disk for source shards, extraction, and 10 GiB headroom")
    (args.output / "extraction-plan.json").write_text(
        json.dumps(
            dict(
                shards=shards,
                selected_tensors=len(selected),
                new_download_bytes=need,
                upper_bound_bytes=capacity,
            ),
            indent=2,
        )
        + "\n"
    )

    def obtain(name):
        info = remote[name]["lfs"]
        candidates = [args.reuse_cache / name, cache / name]
        for path in candidates:
            if path.exists():
                if path.stat().st_size != info["size"] or sha(path) != info["sha256"]:
                    raise ValueError("cached original shard failed LFS hash: " + str(path))
                print("REUSE_VERIFIED", name, flush=True)
                return name, dict(path=str(path), sha256=info["sha256"], bytes=info["size"])
        path = cache / name
        tmp = path.with_suffix(".partial")
        if tmp.exists():
            raise FileExistsError("retained incomplete prior download: " + str(tmp))
        h = hashlib.sha256()
        n = 0
        with (
            urllib.request.urlopen(base + name + "?download=true", timeout=120) as src,
            tmp.open("xb") as dst,
        ):
            if src.status != 200:
                raise ValueError("unexpected partial source response")
            while chunk := src.read(8 * 1024**2):
                dst.write(chunk)
                h.update(chunk)
                n += len(chunk)
        if n != info["size"] or h.hexdigest() != info["sha256"]:
            raise ValueError("download LFS size/hash mismatch: " + name)
        tmp.rename(path)
        print("DOWNLOADED_VERIFIED", name, n, flush=True)
        return name, dict(path=str(path), sha256=h.hexdigest(), bytes=n)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        sources = dict(ex.map(obtain, shards))
    locations = {}
    out_header = {}
    offset = 0
    for shard, info in sources.items():
        h, start = header(Path(info["path"]))
        for name in sorted(k for k in h if selected.get(k) == shard):
            entry = h[name]
            a, b = entry["data_offsets"]
            if not 0 <= a <= b <= info["bytes"] - start:
                raise ValueError("invalid source tensor offset: " + name)
            out_header[name] = dict(
                shape=entry["shape"], dtype=entry["dtype"], data_offsets=[offset, offset + b - a]
            )
            locations[name] = (shard, start + a, b - a)
            offset += b - a
    if set(locations) != set(selected):
        raise ValueError("index/header tensor coverage mismatch")
    out_header["__metadata__"] = dict(
        format="pt",
        source_repository=args.source_repo,
        source_revision=args.revision,
        layers="0,1,2,3,4,5,6",
        storage="original bytes; no conversion",
    )
    raw = json.dumps(out_header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    tensors = {}
    total_hash = hashlib.sha256()
    target = args.output / "model.safetensors.partial"
    with contextlib.ExitStack() as stack, target.open("xb") as dst:
        files = {k: stack.enter_context(Path(v["path"]).open("rb")) for k, v in sources.items()}
        prefix = struct.pack("<Q", len(raw)) + raw
        dst.write(prefix)
        total_hash.update(prefix)
        for name, (shard, start, size) in locations.items():
            src = files[shard]
            src.seek(start)
            remaining = size
            th = hashlib.sha256()
            while remaining:
                chunk = src.read(min(8 * 1024**2, remaining))
                if not chunk:
                    raise EOFError(name)
                dst.write(chunk)
                th.update(chunk)
                total_hash.update(chunk)
                remaining -= len(chunk)
            entry = out_header[name]
            layer = re.match(r"^layers\.(\d+)\.", name)
            # Eligibility reflects existing importer semantics; requires_grad must also be audited at runtime.
            eligible = entry["dtype"] in ("BF16", "F32") and not name.endswith(".scale")
            tensors[name] = dict(
                shape=entry["shape"],
                dtype=entry["dtype"],
                bytes=size,
                sha256=th.hexdigest(),
                layer=int(layer[1]) if layer else None,
                source_shard=shard,
                source_absolute_offset=start,
                trainable_candidate=eligible,
            )
    target.rename(args.output / "model.safetensors")
    config = dict(original_config)
    config.update(
        num_hidden_layers=7,
        num_nextn_predict_layers=0,
        compress_ratios=original_config["compress_ratios"][:7],
    )
    (args.output / "original-config.json").write_text(json.dumps(original_config, indent=2) + "\n")
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    asset_hashes = {}
    for name in ["tokenizer.json", "tokenizer_config.json"]:
        with urllib.request.urlopen(base + name, timeout=120) as src:
            raw_asset = src.read()
        (args.output / name).write_bytes(raw_asset)
        asset_hashes[name] = hashlib.sha256(raw_asset).hexdigest()
    manifest = dict(
        schema_version=1,
        source_repository=args.source_repo,
        source_revision=args.revision,
        original_layers=list(range(7)),
        compress_ratios=config["compress_ratios"],
        mtp_in_scope=False,
        sources=sources,
        tensors=tensors,
        tensor_count=len(tensors),
        safetensors_sha256=total_hash.hexdigest(),
        safetensors_bytes=(args.output / "model.safetensors").stat().st_size,
        tokenizer_sha256=asset_hashes,
        config_sha256=sha(args.output / "config.json"),
        freeze_policy="Original packed FP4/FP8 weights and scales frozen; original floating parameters eligible. Runtime requires_grad audit still required.",
        pp2_proposed_layers=[[0, 1, 2, 3], [4, 5, 6]],
        gpu_validation="NOT_RUN",
    )
    (args.output / "checkpoint.lock.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("SEVEN_LAYER_READY", args.output, manifest["safetensors_sha256"], flush=True)


if __name__ == "__main__":
    main()
