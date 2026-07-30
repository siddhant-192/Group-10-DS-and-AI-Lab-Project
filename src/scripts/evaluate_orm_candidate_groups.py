#!/usr/bin/env python3
"""Use a trained GradeSQL-style ORM adapter to rerank SQL result groups."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import tarfile
import time
from typing import Any

from tqdm.auto import tqdm

import evaluate_text2sql_models as core


SYSTEM_MESSAGE = (
    "You are a text-to-SQL verifier. Determine whether the candidate SQL correctly "
    "answers the question using the supplied database schema. Respond only Yes or No."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-groups", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--model", default="qwen2.5-coder-1.5b-instruct")
    parser.add_argument("--model-root", type=Path, default=Path("models/text2sql-eval"))
    parser.add_argument(
        "--model-source", choices=("auto", "local", "huggingface"), default="auto"
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("/content/huggingface-cache"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--archive", type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verifier_messages(row: dict[str, Any], sql: str) -> list[dict[str, str]]:
    user = (
        f"Question: {str(row['question']).strip()}\n\n"
        f"Database schema:\n{str(row['schema']).strip()}\n\n"
        f"Candidate SQL:\n{sql.strip()}\n\n"
        "Is the SQL correct?"
    )
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user},
    ]


def first_answer_token(tokenizer: Any, messages: list[dict[str, str]], answer: str) -> int:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full = tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": answer}],
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids or len(full_ids) == len(prompt_ids):
        raise ValueError(f"Cannot isolate the {answer!r} assistant target from the chat template")
    return int(full_ids[len(prompt_ids)])


def select_candidate(candidates: list[dict[str, Any]]) -> int:
    if not candidates:
        raise ValueError("Every example must contain at least one candidate group")
    return max(
        range(len(candidates)),
        key=lambda index: (
            float(candidates[index]["orm_logit_margin"]),
            int(candidates[index].get("votes", 0)),
            -index,
        ),
    )


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.max_input_tokens < 1:
        raise ValueError("Batch size and token limit must be positive")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "predictions.jsonl"
    status_path = args.output_dir / "status.json"
    logger = core.configure_logging(args.output_dir)

    rows = read_jsonl(args.candidate_groups.resolve())
    if args.limit is not None:
        rows = rows[: args.limit]
    if len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("Candidate-group IDs must be unique")
    completed = read_jsonl(prediction_path) if prediction_path.exists() else []
    completed_ids = {str(row["id"]) for row in completed}
    pending = [row for row in rows if str(row["id"]) not in completed_ids]

    specs = core.load_specs(args.config.resolve(), args.manifest.resolve(), {args.model})
    if len(specs) != 1:
        raise ValueError(f"Expected exactly one ORM backbone, found {len(specs)}")
    spec = specs[0]
    args.adapter_dir = args.adapter_dir.resolve()
    args.model_root = args.model_root.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.adapter_label = "gradesql-orm"
    core.update_status(
        status_path,
        phase="loading_model",
        model=args.model,
        completed_examples=len(completed),
        total_examples=len(rows),
    )
    model, tokenizer, attention = core.load_model_and_tokenizer(spec, args, logger)

    sample_messages = verifier_messages(
        pending[0] if pending else rows[0],
        str(((pending[0] if pending else rows[0])["candidate_groups"])[0]["sql"]),
    ) if rows else []
    if not sample_messages:
        raise ValueError("Candidate-group input is empty")
    yes_token = first_answer_token(tokenizer, sample_messages, "Yes")
    no_token = first_answer_token(tokenizer, sample_messages, "No")
    if yes_token == no_token:
        raise ValueError("Yes and No map to the same first assistant token")
    logger.info("ORM label tokens: Yes=%d No=%d", yes_token, no_token)

    flat: list[tuple[dict[str, Any], int, str]] = []
    expected: dict[str, int] = {}
    scored: dict[str, list[dict[str, Any] | None]] = {}
    source_rows: dict[str, dict[str, Any]] = {}
    for row in pending:
        item = str(row["id"])
        groups = row.get("candidate_groups") or []
        if not groups:
            raise ValueError(f"{item}: no candidate groups")
        expected[item] = len(groups)
        scored[item] = [None] * len(groups)
        source_rows[item] = row
        for index, candidate in enumerate(groups):
            sql = str(candidate.get("sql") or "").strip()
            if not sql:
                raise ValueError(f"{item}: candidate {index} has empty SQL")
            prompt = tokenizer.apply_chat_template(
                verifier_messages(row, sql), tokenize=False, add_generation_prompt=True
            )
            flat.append((row, index, prompt))

    import torch

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    progress = tqdm(total=len(flat), desc="ORM candidates", unit="candidate")
    started = time.monotonic()
    for offset in range(0, len(flat), args.batch_size):
        batch = flat[offset : offset + args.batch_size]
        prompts = [entry[2] for entry in batch]
        token_counts = [
            len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) for prompt in prompts
        ]
        if max(token_counts, default=0) > args.max_input_tokens:
            raise ValueError(
                f"ORM prompt has {max(token_counts)} tokens, exceeding "
                f"max_input_tokens={args.max_input_tokens}; schemas are never silently truncated"
            )
        encoded = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        device = next(model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded, use_cache=False).logits[:, -1, [yes_token, no_token]]
            probabilities = torch.softmax(logits.float(), dim=-1).cpu()
            logits = logits.float().cpu()
        del encoded
        for position, (row, candidate_index, _prompt) in enumerate(batch):
            item = str(row["id"])
            candidate = dict(row["candidate_groups"][candidate_index])
            yes_logit = float(logits[position, 0])
            no_logit = float(logits[position, 1])
            candidate.update(
                {
                    "candidate_index": candidate_index,
                    "orm_yes_probability": round(float(probabilities[position, 0]), 8),
                    "orm_no_probability": round(float(probabilities[position, 1]), 8),
                    "orm_logit_margin": round(yes_logit - no_logit, 8),
                }
            )
            scored[item][candidate_index] = candidate
        progress.update(len(batch))

        completed_items = []
        for item, values in scored.items():
            if values and all(value is not None for value in values):
                candidates = [value for value in values if value is not None]
                selected_index = select_candidate(candidates)
                selected = candidates[selected_index]
                source = source_rows[item]
                output = {
                    "id": item,
                    "db_id": source.get("db_id"),
                    "question": source.get("question"),
                    "predicted_sql": selected["sql"],
                    "selected_candidate_index": selected_index,
                    "selected_orm_yes_probability": selected["orm_yes_probability"],
                    "selected_orm_logit_margin": selected["orm_logit_margin"],
                    "candidate_groups": candidates,
                }
                append_jsonl(prediction_path, output)
                completed.append(output)
                completed_ids.add(item)
                completed_items.append(item)
        for item in completed_items:
            del scored[item]
            del expected[item]
            del source_rows[item]
        core.update_status(
            status_path,
            phase="evaluating",
            completed_examples=len(completed),
            total_examples=len(rows),
            completed_candidates=min(offset + len(batch), len(flat)),
            total_candidates=len(flat),
            elapsed_seconds=round(time.monotonic() - started, 1),
        )
    progress.close()
    if scored:
        raise AssertionError(f"Incomplete candidate scores remain for {len(scored)} examples")

    completed_by_id = {str(row["id"]): row for row in completed}
    ordered = [completed_by_id[str(row["id"])] for row in rows]
    temporary = prediction_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(prediction_path)
    metrics = {
        "examples": len(ordered),
        "candidate_groups": sum(len(row["candidate_groups"]) for row in ordered),
        "mean_candidate_groups": round(
            sum(len(row["candidate_groups"]) for row in ordered) / len(ordered), 3
        ),
        "selected_candidate_indices": dict(
            Counter(str(row["selected_candidate_index"]) for row in ordered)
        ),
        "model": args.model,
        "repo_id": spec["repo_id"],
        "revision": spec["revision"],
        "adapter_dir": str(args.adapter_dir),
        "attention": attention,
        "yes_token_id": yes_token,
        "no_token_id": no_token,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    core.update_status(status_path, phase="complete", metrics=metrics)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if args.archive is not None:
        archive = args.archive.resolve()
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "w:gz") as handle:
            handle.add(args.output_dir, arcname="results")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
