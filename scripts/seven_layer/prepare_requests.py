# SPDX-License-Identifier: MIT
"""Prepare immutable three-update copy/math requests with native DSV4 encoding."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--copy-parquet", type=Path, required=True)
    args = ap.parse_args()
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

    model = args.work / "models/DeepSeek-V4-Flash-7L"
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    builder = create_continuous_token_builder(tokenizer, hf_model_type="deepseek_v4", chat_template_kwargs={})
    out = args.work / "data/requests-N"
    out.mkdir(exist_ok=False)

    def prompt(row, index):
        ids = builder.build_initial_tokens(row["prompt"])
        if not ids or len(ids) > 1024:
            return None
        return dict(
            prompt_ids=ids,
            group_id=str(index),
            source_row=index,
            data_source=row["data_source"],
            ground_truth=row["reward_model"]["ground_truth"],
            extra_info=row.get("extra_info", {}),
        )

    copies = pq.read_table(args.copy_parquet).to_pylist()
    copy_prompts = [prompt(row, i) for i, row in enumerate(copies)]
    if not copy_prompts or any(r is None for r in copy_prompts):
        raise ValueError("functional copy prompt schema/length differs")
    dapo = args.work / "data/dapo-raw/dapo-math-17k.parquet"
    pinned = json.loads((args.work / "evidence/data.lock.json").read_text())
    if sha(dapo) != pinned["sha256"]:
        raise ValueError("pinned original DAPO parquet hash differs")
    math_prompts = []
    index = 0
    for batch in pq.ParquetFile(dapo).iter_batches(batch_size=128):
        for row in batch.to_pylist():
            current = index
            index += 1
            if current < 128:
                continue
            if row["data_source"] not in ("math_dapo", "math", "math_dapo_reasoning"):
                raise ValueError("real data reward routing changed")
            item = prompt(row, current)
            if item is not None:
                math_prompts.append(item)
            if len(math_prompts) == 24:
                break
        if len(math_prompts) == 24:
            break
    if len(math_prompts) != 24:
        raise ValueError("insufficient original real-data prompts within the 1024-token cap")
    for dataset, rows, n, count, cap in [
        ("copy", copy_prompts, 2, 2, 16),
        ("dapo", math_prompts, 8, 16, 7168),
    ]:
        for step in range(1, 5):
            selected = [rows[((step - 1) * n + i) % len(rows)] for i in range(n)]
            request = dict(
                dataset=dataset,
                prompts=selected,
                samples_per_prompt=count,
                max_tokens=cap,
                prompt_cap=1024,
                purpose="post-v3 audit only; no optimizer update"
                if step == 4
                else f"optimizer update {step}/3",
                source=pinned
                if dataset == "dapo"
                else dict(type="functional public exact-copy task", parquet_sha256=sha(args.copy_parquet)),
            )
            (out / f"{dataset}-step{step}.json").write_text(json.dumps(request, indent=2) + "\n")
    lock = dict(
        model_sha256=json.loads((model / "checkpoint.lock.json").read_text())["safetensors_sha256"],
        tokenizer_sha256=sha(model / "tokenizer.json"),
        builder=type(builder).__module__ + "." + type(builder).__name__,
        builder_source_sha256=sha(inspect.getfile(type(builder))),
        files={p.name: sha(p) for p in out.glob("*.json")},
        scope="CPU input preparation only; no GPU/reward/update acceptance",
    )
    (out / "requests.lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
