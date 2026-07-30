#!/usr/bin/env python3
"""Tokenize the finalized SFT package locally and enforce sequence-length limits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
from typing import Any, Sequence

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "finetuning" / "spider_sft_v1"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "text2sql_eval_models.json"
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "models" / "text2sql-eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument(
        "--max-seq-length",
        type=int,
        help="Override model-specific sft_max_seq_length values from the model config.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def percentile(values: Sequence[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def length_summary(values: Sequence[int]) -> dict[str, int]:
    return {
        "min": min(values),
        "median": int(statistics.median(values)),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
        "total": sum(values),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))["models"]
    if args.models:
        selected = set(args.models)
        config = [spec for spec in config if str(spec["slug"]) in selected]
        missing = selected - {str(spec["slug"]) for spec in config}
        if missing:
            raise ValueError(f"Unknown model(s): {', '.join(sorted(missing))}")
    split_paths = {
        "train_base": data_dir / "train_base.jsonl",
        "train_curriculum": data_dir / "train_curriculum.jsonl",
        "validation": data_dir / "validation.jsonl",
    }
    splits = {name: read_jsonl(path) for name, path in split_paths.items()}
    report: dict[str, Any] = {
        "max_seq_length_override": args.max_seq_length,
        "data_dir": str(data_dir),
        "models": {},
    }
    violations = []
    for spec in config:
        slug = str(spec["slug"])
        model_dir = args.model_root.resolve() / slug
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=bool(spec.get("trust_remote_code", False)),
        )
        max_seq_length = int(args.max_seq_length or spec.get("sft_max_seq_length", 4096))
        model_report = {
            "tokenizer": str(model_dir),
            "max_seq_length": max_seq_length,
            "splits": {},
        }
        for split, rows in splits.items():
            full_lengths: list[int] = []
            prompt_lengths: list[int] = []
            target_lengths: list[int] = []
            max_record: tuple[int, str] = (-1, "")
            for row in rows:
                messages = row["messages"]
                roles = [message["role"] for message in messages]
                if roles not in (["system", "user", "assistant"], ["user", "assistant"]):
                    raise ValueError(f"{row['id']}: invalid message roles")
                full_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                )
                prompt_ids = tokenizer.apply_chat_template(
                    messages[:-1],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                target_ids = tokenizer(str(messages[-1]["content"]), add_special_tokens=False)["input_ids"]
                full_length = len(full_ids)
                full_lengths.append(full_length)
                prompt_lengths.append(len(prompt_ids))
                target_lengths.append(len(target_ids))
                if full_length > max_record[0]:
                    max_record = (full_length, str(row["id"]))
            over_limit = sum(value > max_seq_length for value in full_lengths)
            item = {
                "examples": len(rows),
                "full_sequence_tokens": length_summary(full_lengths),
                "prompt_tokens": length_summary(prompt_lengths),
                "target_tokens": length_summary(target_lengths),
                "over_max_seq_length": over_limit,
                "over_4096_tokens": sum(value > 4096 for value in full_lengths),
                "longest_example_id": max_record[1],
            }
            model_report["splits"][split] = item
            print(
                f"{slug} {split}: examples={len(rows)} median={item['full_sequence_tokens']['median']} "
                f"p95={item['full_sequence_tokens']['p95']} max={item['full_sequence_tokens']['max']} "
                f"limit={max_seq_length} "
                f"over_limit={over_limit}"
            )
            if over_limit:
                violations.append(f"{slug}/{split}={over_limit}")
        report["models"][slug] = model_report

    output = data_dir / "tokenization_report.json"
    atomic_json(output, report)
    checksums_path = data_dir / "checksums.json"
    checksums = json.loads(checksums_path.read_text(encoding="utf-8")) if checksums_path.exists() else {}
    checksums[output.name] = sha256_file(output)
    atomic_json(checksums_path, checksums)
    if violations:
        raise RuntimeError(
            f"SFT sequences exceed a configured model limit: {', '.join(violations)}"
        )
    print(f"Tokenization preflight passed: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
