# SPDX-License-Identifier: MIT
"""Prepare pinned assets inside the dependency image, without GPU access."""

import argparse
import hashlib
import json
import re
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, "/opt/recipes")
from recipe_common import KERNEL, config, data_dir, lock, plan, write_json


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_bundle(archive, destination):
    import zstandard

    # No archive member may escape the staging directory or create links/devices.
    with archive.open("rb") as source, zstandard.ZstdDecompressor().stream_reader(source) as stream:
        with tarfile.open(fileobj=stream, mode="r|") as tar:
            for member in tar:
                path = Path(member.name)
                if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
                    raise ValueError(f"unsupported/unsafe bundle member: {member.name}")
                dest = destination / path
                if dest.is_symlink():
                    raise ValueError("staging destination contains a symlink")
                if member.isdir():
                    dest.mkdir(parents=True, exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as src, dest.open("xb") as dst:
                        shutil.copyfileobj(src, dst, 8 * 1024**2)


def fetch_bundle(c):
    pinned = lock()["checkpoint"]
    downloads = Path(c["work_dir"]) / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    archive = downloads / pinned["archive"]
    if not archive.exists():
        pending = archive.with_suffix(archive.suffix + ".partial")
        print("Downloading pinned model/data bundle (large file)", flush=True)
        with urllib.request.urlopen(pinned["download_url"], timeout=120) as src, pending.open("wb") as dst:
            shutil.copyfileobj(src, dst, 8 * 1024**2)
        if sha(pending) != pinned["archive_sha256"]:
            raise ValueError("model bundle SHA256 differs; partial file retained")
        pending.rename(archive)
    if sha(archive) != pinned["archive_sha256"]:
        raise ValueError("existing bundle SHA256 differs")
    stage = downloads / "unpacked"
    if not stage.exists():
        partial = downloads / "unpacked.partial"
        partial.mkdir(exist_ok=False)
        extract_bundle(archive, partial)
        partial.rename(stage)
    model_candidates = []
    for cfg in stage.rglob("config.json"):
        if json.loads(cfg.read_text()).get("num_hidden_layers") == 4 and list(
            cfg.parent.glob("*.safetensors")
        ):
            model_candidates.append(cfg.parent)
    data_candidates = [
        p.parent for p in stage.rglob("train.parquet") if (p.parent / "replay-fixed-sequences.json").exists()
    ]
    for candidates, name in ((model_candidates, "model_dir"), (data_candidates, "baseline_data_dir")):
        target = Path(c[name])
        if target.exists() and any(target.iterdir()):
            continue
        if len(candidates) != 1:
            raise ValueError(
                f"bundle has {len(candidates)} {name} candidates; set the unpacked path explicitly"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.mkdir(exist_ok=True)

        # Hard-link immutable assets on the same filesystem; copy across filesystems.
        def link_or_copy(src, dst):
            try:
                Path(dst).hardlink_to(src)
            except OSError:
                shutil.copy2(src, dst)

        shutil.copytree(candidates[0], target, dirs_exist_ok=True, copy_function=link_or_copy)


def gsm_row(row, index, split):
    found = re.search(r"####\s*(-?[0-9.,]+)", row["answer"])
    if not found:
        raise ValueError("GSM8K answer has no canonical #### answer")
    return dict(
        data_source="openai/gsm8k",
        prompt=[
            dict(
                role="user",
                content=row["question"]
                + '\nLet\'s think step by step and output the final answer after "####".',
            )
        ],
        ability="math",
        reward_model=dict(style="rule", ground_truth=found.group(1).replace(",", "")),
        extra_info=dict(index=index, split=split),
    )


def native_prompt_builder(model, tokenizer):
    """Use the same model-family encoder and default kwargs as native AgentLoop."""
    from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

    model_type = json.loads((Path(model) / "config.json").read_text())["model_type"]
    return create_continuous_token_builder(tokenizer, hf_model_type=model_type, chat_template_kwargs={})


def pinned_dapo(c, pinned):
    """Retain and verify the published parquet at its locked commit, without transforming rows."""
    from datasets import Dataset
    from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url

    filename = "data/dapo-math-17k.parquet"
    kwargs = dict(repo_id=pinned["repo"], filename=filename, repo_type="dataset", revision=pinned["revision"])
    metadata = get_hf_file_metadata(hf_hub_url(**kwargs))
    if metadata.commit_hash != pinned["revision"] or not re.fullmatch(r"[0-9a-f]{64}", metadata.etag):
        raise ValueError("DAPO revision or LFS content digest differs from the locked public source")
    path = Path(hf_hub_download(**kwargs, cache_dir=str(Path(c["work_dir"]) / "cache/huggingface-hub")))
    digest = sha(path)
    if digest != metadata.etag:
        raise ValueError("downloaded DAPO parquet differs from its locked-revision LFS SHA256")
    receipt = dict(file=filename, revision=metadata.commit_hash, bytes=path.stat().st_size, sha256=digest)
    write_json(Path(c["work_dir"]) / "downloads/dapo-source.json", receipt)
    return Dataset.from_parquet(str(path)), receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    c = config(args.config)
    p = plan(c)
    if args.download:
        fetch_bundle(c)
    model = Path(c["model_dir"])
    baseline = Path(c["baseline_data_dir"])
    weights = list(model.glob("*.safetensors"))
    if (
        len(weights) != 1
        or weights[0].stat().st_size != lock()["checkpoint"]["bytes"]
        or sha(weights[0]) != lock()["checkpoint"]["sha256"]
    ):
        raise ValueError("requires the exact original four-layer safetensors checkpoint")
    recorded = Path(KERNEL) / "examples/operators/dsv4_4l/results/native_grpo_pp2_dp1_20261117"
    for name, metadata in json.loads((recorded / "checkpoint-assets.json").read_text()).items():
        if sha(model / name) != metadata["sha256"]:
            raise ValueError(f"checkpoint asset differs: {name}")
    from datasets import Dataset, load_dataset

    output = data_dir(c)
    output.mkdir(parents=True, exist_ok=False)
    for name in ("checkpoint-inventory.json", "replay-fixed-sequences.json", "extended-audit-prompts.json"):
        shutil.copyfile(baseline / name, output / name)
    if c["dataset"] == "copy":
        train = Dataset.from_parquet(str(baseline / "train.parquet"))
        val = Dataset.from_parquet(str(baseline / "val.parquet"))
        original_rows = len(train)
        rows = [dict(train[i % len(train)]) for i in range(p["prompts"] * 3)]
        train = Dataset.from_list(rows)
        source = dict(type="functional-copy", original_rows=original_rows)
    else:
        pinned = lock()["datasets"][c["dataset"]]
        if c["dataset"] == "gsm8k":
            raw = load_dataset(pinned["repo"], "main", revision=pinned["revision"])
            train = Dataset.from_list([gsm_row(r, i, "train") for i, r in enumerate(raw["train"])])
            val = Dataset.from_list([gsm_row(r, i, "test") for i, r in enumerate(raw["test"])])
        else:
            raw, raw_receipt = pinned_dapo(c, pinned)
            # No invented answers or replacement reward for the DAPO source.
            for row in raw:
                if (
                    row.get("data_source") not in ("math_dapo", "math", "math_dapo_reasoning")
                    or not row.get("prompt")
                    or not row.get("reward_model", {}).get("ground_truth")
                ):
                    raise ValueError(
                        "DAPO schema/reward routing differs from the expected native verl format"
                    )
            val, train = raw.select(range(128)), raw.select(range(128, len(raw)))
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
        builder = native_prompt_builder(model, tokenizer)
        before = len(train)
        train = train.filter(
            lambda r: len(builder.build_initial_tokens(r["prompt"])) <= p["max_prompt_length"]
        )
        source = dict(
            **pinned,
            raw_parquet=raw_receipt if c["dataset"] == "dapo" else None,
            prompt_encoding=dict(
                builder=type(builder).__module__ + "." + type(builder).__name__,
                model_type=json.loads((model / "config.json").read_text())["model_type"],
                chat_template_kwargs={},
                encoder_sha256=sha(Path(__import__("inspect").getfile(type(builder)))),
            ),
            rows_before_length_filter=before,
            rows_after_length_filter=len(train),
            validation="first 128 held-out DAPO rows or official GSM8K test",
        )
        # Both math profiles prepare enough data for 200 updates without cycling.
        if len(train) < 200 * p["prompts"]:
            raise ValueError("fewer than 200 full prompt batches remain after filtering")
    train.to_parquet(str(output / "train.parquet"))
    val.to_parquet(str(output / "val.parquet"))
    write_json(
        output / "manifest.json",
        dict(
            source=source,
            train_rows=len(train),
            val_rows=len(val),
            profile=c["profile"],
            checkpoint_sha256=sha(weights[0]),
            files={f.name: sha(f) for f in output.iterdir() if f.is_file()},
        ),
    )
    print(f"DATA_PREPARED {output}", flush=True)


if __name__ == "__main__":
    main()
