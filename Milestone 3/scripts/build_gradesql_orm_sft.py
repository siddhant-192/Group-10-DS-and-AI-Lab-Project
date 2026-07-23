#!/usr/bin/env python3
"""Build a leakage-safe, deduplicated GradeSQL Spider ORM chat-SFT package."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "raw" / "gradesql" / "spider-balanced.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "finetuning" / "gradesql_orm_spider_v1"
SOURCE_REPO = "sisinflab-ai/GradeSQL-training-dataset-spider-balanced"
SOURCE_REVISION = "44cbee9732352a98cc2088005acd0839c3c266aa"
SYSTEM_MESSAGE = (
    "You are a text-to-SQL verifier. Determine whether the candidate SQL correctly "
    "answers the question using the supplied database schema. Respond only Yes or No."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-percent", type=int, default=10)
    parser.add_argument(
        "--max-prompt-characters",
        type=int,
        default=11_500,
        help="Drop whole long-schema groups that cannot fit the 4,096-token Qwen contract.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized(value: str) -> str:
    return " ".join(value.split())


def group_key(row: dict[str, Any]) -> str:
    payload = normalized(str(row["question"])) + "\n" + normalized(str(row["schema"]))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def exact_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        normalized(str(row["question"])),
        normalized(str(row["schema"])),
        normalized(str(row["sql"])),
        int(row["label"]),
    )


def user_prompt(row: dict[str, Any]) -> str:
    return (
        f"Question: {str(row['question']).strip()}\n\n"
        f"Database schema:\n{str(row['schema']).strip()}\n\n"
        f"Candidate SQL:\n{str(row['sql']).strip()}\n\n"
        "Is the SQL correct?"
    )


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    temporary = path.with_name(path.name + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    os.replace(temporary, path)
    return count


def sft_row(row: dict[str, Any], split: str, occurrence: int) -> dict[str, Any]:
    label = int(row["label"])
    if label not in (0, 1):
        raise ValueError(f"Unexpected GradeSQL label: {label}")
    identity = "\n".join(map(str, exact_key(row)))
    source_hash = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    user = user_prompt(row)
    # GradeSQL's published dataset builder assigns 1 to execution-equivalent
    # positive candidates and 0 to negative candidates. Keep this explicit:
    # reversing it would train an apparently well-behaved anti-verifier.
    answer = "Yes" if label == 1 else "No"
    return {
        "id": f"gradesql-{source_hash[:20]}",
        "source_id": source_hash,
        "dataset": "gradesql-spider-orm",
        "split": split,
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ],
        "orm": {
            "candidate_sql": str(row["sql"]),
            "candidate_result": str(row.get("data", "")),
            "grade_sql_label": label,
            "answer": answer,
            "label_contract": "0=incorrect/No, 1=correct/Yes",
            "source_duplicate_count": occurrence,
        },
    }


def main() -> int:
    args = parse_args()
    if not 1 <= args.validation_percent <= 50:
        raise ValueError("--validation-percent must be between 1 and 50")
    source = args.source.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_rows = pq.read_table(source).to_pylist()
    required = {"question", "schema", "sql", "data", "label"}
    if not raw_rows or not required.issubset(raw_rows[0]):
        raise ValueError(f"GradeSQL parquet is empty or missing columns: {required}")
    labels = Counter(int(row["label"]) for row in raw_rows)
    if labels[0] != labels[1] or set(labels) != {0, 1}:
        raise ValueError(f"Expected the published balanced binary labels, found {labels}")

    occurrences = Counter(exact_key(row) for row in raw_rows)
    unique_by_key: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for row in raw_rows:
        unique_by_key.setdefault(exact_key(row), row)
    deduplicated_rows = list(unique_by_key.values())

    long_groups = {
        group_key(row)
        for row in deduplicated_rows
        if len(user_prompt(row)) > args.max_prompt_characters
    }
    unique_rows = [row for row in deduplicated_rows if group_key(row) not in long_groups]

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in unique_rows:
        groups[group_key(row)].append(row)
    if any({int(row["label"]) for row in rows} != {0, 1} for rows in groups.values()):
        raise ValueError("Every GradeSQL question/schema group must contain both labels")

    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    train_groups: set[str] = set()
    validation_groups: set[str] = set()
    threshold = args.validation_percent * 10
    for key, rows in sorted(groups.items()):
        # Use a second independent hash so the content hash itself is not treated
        # as a pseudorandom number with an arbitrary prefix convention.
        split_value = int(hashlib.sha256(("split-v1:" + key).encode()).hexdigest()[:8], 16) % 1000
        if split_value < threshold:
            validation_rows.extend(rows)
            validation_groups.add(key)
        else:
            train_rows.extend(rows)
            train_groups.add(key)
    if train_groups & validation_groups:
        raise AssertionError("Question/schema group leakage detected")
    if not train_groups or not validation_groups:
        raise ValueError("Deterministic group split produced an empty partition")
    observed_validation = 100.0 * len(validation_groups) / len(groups)
    if abs(observed_validation - args.validation_percent) > 3.0:
        raise ValueError(
            f"Observed validation group share {observed_validation:.2f}% is unexpectedly far "
            f"from requested {args.validation_percent}%"
        )

    def convert(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
        ordered = sorted(rows, key=lambda row: (group_key(row), int(row["label"]), normalized(str(row["sql"]))))
        return [sft_row(row, split, occurrences[exact_key(row)]) for row in ordered]

    train = convert(train_rows, "train")
    validation = convert(validation_rows, "validation")
    write_jsonl(output / "train_base.jsonl", train)
    write_jsonl(output / "train_curriculum.jsonl", train)
    write_jsonl(output / "validation.jsonl", validation)
    checksums = {
        name: sha256_file(output / name)
        for name in ("train_base.jsonl", "train_curriculum.jsonl", "validation.jsonl")
    }
    write_json(output / "checksums.json", checksums)
    manifest = {
        "format_version": 1,
        "task": "autoregressive_outcome_reward_model",
        "source": {
            "repo_id": SOURCE_REPO,
            "revision": SOURCE_REVISION,
            "path": str(source),
            "sha256": sha256_file(source),
            "rows": len(raw_rows),
        },
        "label_contract": {"0": "incorrect/No", "1": "correct/Yes"},
        "exact_duplicates_removed": len(raw_rows) - len(deduplicated_rows),
        "long_schema_groups_removed": len(long_groups),
        "long_schema_rows_removed": len(deduplicated_rows) - len(unique_rows),
        "max_prompt_characters": args.max_prompt_characters,
        "unique_rows": len(unique_rows),
        "question_schema_groups": len(groups),
        "split_method": "deterministic SHA-256 question+schema group split",
        "validation_percent_requested": args.validation_percent,
        "group_overlap": len(train_groups & validation_groups),
        "train": {
            "rows": len(train),
            "groups": len(train_groups),
            "answers": dict(Counter(row["messages"][-1]["content"] for row in train)),
        },
        "validation": {
            "rows": len(validation),
            "groups": len(validation_groups),
            "answers": dict(Counter(row["messages"][-1]["content"] for row in validation)),
        },
        "checksums": checksums,
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
