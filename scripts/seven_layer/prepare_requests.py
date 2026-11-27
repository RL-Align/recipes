# SPDX-License-Identifier: MIT
"""Prepare immutable three-update copy/math/long-context-copy requests with native DSV4 encoding."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path

LCC_SOURCE = "rlk/long-context-copy"
LCC_INSTRUCTION = "Copy the following text exactly, without any explanation:\n\n"
LCC_START_ROW = 1024  # DAPO prompts use rows >= 128 and stay far below this; checked at build time
LCC_PASSAGES = 24
LCC_RULE = (
    "Passage = consecutive DAPO-Math-17k user-message contents (file order, from the first row after the previous "
    "passage's last row; start row 1024), joined by a blank line, taking the longest character prefix whose prompt "
    f"(user message = {LCC_INSTRUCTION!r} + passage) has at most 1024 native DSV4 tokens; trailing whitespace "
    "stripped. ground_truth = passage; extra_info.target_ids = tokenizer.encode(passage, add_special_tokens=False). "
    "Reward = 1 - token_edit_distance(response_ids without one trailing EOS, target_ids) / max(len, len, 1)."
)
# dataset -> (prompts per update, responses per prompt, max response tokens, label)
LCC_DATASETS = {
    "lcc": (8, 16, 7168, "Long-Context Verifiable Sequence GRPO Benchmark (formal matrix target; not math reasoning)"),
    "lccsmoke": (2, 4, 2048, "DIAGNOSTIC/FEASIBILITY ONLY; never part of the formal matrix"),
}


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def lcc_message(passage):
    return [dict(role="user", content=LCC_INSTRUCTION + passage)]


def build_passages(rows, fits, count=LCC_PASSAGES):
    """Greedy longest-prefix passages from (row_index, text) pairs; fits(passage) -> prompt within the cap."""
    rows = iter(rows)
    passages = []
    while len(passages) < count:
        text, used, partial = "", [], False
        for index, row_text in rows:
            candidate = row_text if not text else text + "\n\n" + row_text
            if fits(candidate.rstrip()):
                text, used = candidate, used + [index]
                continue
            low, high = len(text), len(candidate)  # candidate[:high] does not fit
            while high - low > 1:
                middle = (low + high) // 2
                if fits(candidate[:middle].rstrip()):
                    low = middle
                else:
                    high = middle
            if candidate[:low].rstrip() != text.rstrip():
                used.append(index)
                partial = True
            text = candidate[:low]
            break
        else:
            raise ValueError("source rows exhausted before the requested number of passages")
        text = text.rstrip()
        if not text or not fits(text):
            raise ValueError("a passage does not fit the prompt cap")
        passages.append(dict(text=text, source_rows=used, partial_last_row=partial))
    return passages


def lcc_requests(passages, parquet_sha, eos_token_id, encode_prompt, encode_target, dataset):
    """{filename: request} for steps 1..4 of dataset ('lcc' or 'lccsmoke')."""
    n, count, cap, label = LCC_DATASETS[dataset]
    rows = []
    for i, passage in enumerate(passages):
        ids = encode_prompt(passage["text"])
        if not ids or len(ids) > 1024:
            raise ValueError("long-context prompt exceeds the 1024-token cap")
        rows.append(
            dict(
                prompt_ids=ids,
                group_id=str(i),
                source_row=i,
                data_source=LCC_SOURCE,
                ground_truth=passage["text"],
                extra_info=dict(
                    passage=i,
                    source_rows=passage["source_rows"],
                    partial_last_row=passage["partial_last_row"],
                    target_ids=encode_target(passage["text"]),
                    eos_token_id=eos_token_id,
                ),
            )
        )
    result = {}
    for step in range(1, 5):
        picked = [(step - 1) * n + i for i in range(n)]
        picked = [p % len(rows) for p in picked]
        result[f"{dataset}-step{step}.json"] = dict(
            dataset=dataset,
            prompts=[rows[p] for p in picked],
            samples_per_prompt=count,
            max_tokens=cap,
            prompt_cap=1024,
            purpose=("post-v3 audit only; no optimizer update" if step == 4 else f"optimizer update {step}/3"),
            label=label,
            source=dict(
                type="long-context copy (deterministic token edit-distance reward)",
                dataset_key=dataset,
                parquet_sha256=parquet_sha,
                start_row=LCC_START_ROW,
                construction_rule=LCC_RULE,
                reward_source="examples/operators/dsv4_4l/long_context_copy.py",
                passages=[
                    dict(
                        passage=p,
                        source_rows=passages[p]["source_rows"],
                        partial_last_row=passages[p]["partial_last_row"],
                        prompt_tokens=len(rows[p]["prompt_ids"]),
                        target_tokens=len(rows[p]["extra_info"]["target_ids"]),
                    )
                    for p in picked
                ],
            ),
        )
    return result


def user_text(row):
    users = [m["content"] for m in row["prompt"] if m["role"] == "user"]
    if len(users) != 1:
        raise ValueError("source row has no single user message")
    return users[0]


def lcc_files(work, builder, tokenizer, dapo, pinned, math_rows):
    import pyarrow.parquet as pq

    if max(math_rows) >= LCC_START_ROW:
        raise ValueError("DAPO prompt rows overlap the long-context source range")
    if sha(dapo) != pinned["sha256"]:
        raise ValueError("pinned original DAPO parquet hash differs")

    def source_rows():
        index = 0
        for batch in pq.ParquetFile(dapo).iter_batches(batch_size=128):
            for row in batch.to_pylist():
                if index >= LCC_START_ROW:
                    yield index, user_text(row)
                index += 1

    def fits(passage):
        ids = builder.build_initial_tokens(lcc_message(passage))
        return bool(ids) and len(ids) <= 1024

    passages = build_passages(source_rows(), fits)
    encode_prompt = lambda text: builder.build_initial_tokens(lcc_message(text))  # noqa: E731
    encode_target = lambda text: tokenizer.encode(text, add_special_tokens=False)  # noqa: E731
    files = {}
    for dataset in LCC_DATASETS:
        files.update(lcc_requests(passages, sha(dapo), tokenizer.eos_token_id, encode_prompt, encode_target, dataset))
    return files


def read_lock(out):
    """requests.lock.json of `out`, after checking that every locked file still has its recorded sha256."""
    lock = json.loads((Path(out) / "requests.lock.json").read_text())
    for name, digest in lock["files"].items():
        if sha(Path(out) / name) != digest:
            raise ValueError(f"existing request {name} differs from requests.lock.json")
    return lock


def add_to_lock(out, files):
    """Write new request files next to the existing ones (never overwriting) and add them to requests.lock.json."""
    out = Path(out)
    lock = read_lock(out)
    if any((out / name).exists() or name in lock["files"] for name in files):
        raise ValueError("long-context request files already exist; never overwritten")
    for name, request in files.items():
        (out / name).write_text(json.dumps(request, indent=2) + "\n")
    lock["files"].update({name: sha(out / name) for name in sorted(files)})
    lock["long_context_scope"] = "lcc/lccsmoke files added after the original lock; existing hashes unchanged"
    (out / "requests.lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    return lock


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--copy-parquet", type=Path)
    ap.add_argument("--out", type=Path, help="request directory (default <work>/data/requests-N)")
    ap.add_argument(
        "--lcc-only",
        action="store_true",
        help="add lcc/lccsmoke files to an existing request directory and extend its requests.lock.json; "
        "existing files must still match the lock and are never rewritten",
    )
    args = ap.parse_args()
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

    if not args.lcc_only and not args.copy_parquet:
        ap.error("--copy-parquet is required unless --lcc-only")
    model = args.work / "models/DeepSeek-V4-Flash-7L"
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    builder = create_continuous_token_builder(tokenizer, hf_model_type="deepseek_v4", chat_template_kwargs={})
    out = args.out or args.work / "data/requests-N"
    dapo = args.work / "data/dapo-raw/dapo-math-17k.parquet"
    pinned = json.loads((args.work / "evidence/data.lock.json").read_text())
    if args.lcc_only:
        lock = read_lock(out)
        math_rows = [
            p["source_row"] for name in lock["files"] if name.startswith("dapo-") for p in json.loads((out / name).read_text())["prompts"]
        ]
        lock = add_to_lock(out, lcc_files(args.work, builder, tokenizer, dapo, pinned, math_rows))
        print(json.dumps(lock, indent=2))
        return
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
    for name, request in lcc_files(args.work, builder, tokenizer, dapo, pinned, [p["source_row"] for p in math_prompts]).items():
        (out / name).write_text(json.dumps(request, indent=2) + "\n")
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
