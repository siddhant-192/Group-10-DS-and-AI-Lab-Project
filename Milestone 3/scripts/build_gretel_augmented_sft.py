#!/usr/bin/env python3
"""Build a leakage-checked Spider + execution-filtered Gretel SFT package."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sqlite3
import time
from typing import Any, Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARQUET = (
    PROJECT_ROOT / "data" / "raw" / "gretel" / "gretel-synthetic-text2sql-train.parquet"
)
DEFAULT_BASE = PROJECT_ROOT / "data" / "finetuning" / "spider_sft_v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "finetuning" / "spider_gretel_exec_v1"
READ_ONLY = re.compile(r"^\s*(?:SELECT|WITH)\b", re.IGNORECASE)
SYSTEM = (
    "You convert natural-language questions into one read-only SQLite query. "
    "Use only the supplied database schema. Return SQL only, with no Markdown "
    "fence, explanation, or alternative query."
)
DENIED_QUERY_ACTIONS = frozenset(
    value
    for name in (
        "SQLITE_ALTER_TABLE", "SQLITE_ANALYZE", "SQLITE_ATTACH", "SQLITE_CREATE_INDEX",
        "SQLITE_CREATE_TABLE", "SQLITE_CREATE_TEMP_INDEX", "SQLITE_CREATE_TEMP_TABLE",
        "SQLITE_CREATE_TEMP_TRIGGER", "SQLITE_CREATE_TEMP_VIEW", "SQLITE_CREATE_TRIGGER",
        "SQLITE_CREATE_VIEW", "SQLITE_CREATE_VTABLE", "SQLITE_DELETE", "SQLITE_DETACH",
        "SQLITE_DROP_INDEX", "SQLITE_DROP_TABLE", "SQLITE_DROP_TEMP_INDEX",
        "SQLITE_DROP_TEMP_TABLE", "SQLITE_DROP_TEMP_TRIGGER", "SQLITE_DROP_TEMP_VIEW",
        "SQLITE_DROP_TRIGGER", "SQLITE_DROP_VIEW", "SQLITE_DROP_VTABLE", "SQLITE_INSERT",
        "SQLITE_PRAGMA", "SQLITE_REINDEX", "SQLITE_TRANSACTION", "SQLITE_UPDATE",
    )
    if (value := getattr(sqlite3, name, None)) is not None
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--base-package", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-synthetic", type=int, default=5_000)
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 4))
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--query-timeout", type=float, default=0.5)
    return parser.parse_args()


def normalized(value: str) -> str:
    return " ".join(value.casefold().strip().split())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(path)
    return count


def validate_one(item: dict[str, Any], timeout: float) -> tuple[str, dict[str, Any] | None]:
    question = str(item.get("sql_prompt") or "").strip()
    context = str(item.get("sql_context") or "").strip()
    sql = str(item.get("sql") or "").strip()
    if not question or not context or not READ_ONLY.match(sql):
        return "not_read_only_or_missing", None
    connection = sqlite3.connect(":memory:")
    deadline = time.monotonic() + timeout
    connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 2_000)
    try:
        try:
            connection.executescript(context)
        except sqlite3.DatabaseError:
            return "context_execution_error", None
        schema_rows = connection.execute(
            "SELECT type, name, sql FROM sqlite_master "
            "WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL "
            "ORDER BY type, name"
        ).fetchall()
        if not schema_rows:
            return "empty_schema", None
        schema = ";\n\n".join(str(row[2]).strip().rstrip(";") for row in schema_rows) + ";"

        def authorizer(action: int, *_args: Any) -> int:
            return sqlite3.SQLITE_DENY if action in DENIED_QUERY_ACTIONS else sqlite3.SQLITE_OK

        connection.set_authorizer(authorizer)
        try:
            cursor = connection.execute(sql)
            cursor.fetchmany(2)
            if cursor.description is None:
                return "non_query_target", None
        except sqlite3.DatabaseError as exc:
            if "interrupted" in str(exc).lower():
                return "target_timeout", None
            return "target_execution_error", None
    finally:
        connection.close()

    user = f"Database dialect: SQLite\n\nDatabase schema:\n{schema}\n\nQuestion: {question}"
    return "accepted", {
        "id": f"gretel-{int(item['id']):06d}",
        "source_id": f"gretel-{int(item['id']):06d}",
        "dataset": "gretel-synthetic-text-to-sql-exec-filtered",
        "split": "train",
        "db_id": f"gretel_{int(item['id']):06d}",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": sql},
        ],
        "sampling": {
            "sample_origin": "gretel_execution_filtered",
            "source_domain": str(item.get("domain") or "unknown"),
            "source_complexity": str(item.get("sql_complexity") or "unknown"),
        },
        "_question_key": normalized(question),
        "_sql_key": normalized(sql.rstrip(";")),
        "_schema_key": normalized(schema),
    }


def select_balanced(rows: list[dict[str, Any]], maximum: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["sampling"]["source_complexity"])].append(row)
    for values in groups.values():
        rng.shuffle(values)
    selected: list[dict[str, Any]] = []
    ordered_groups = sorted(groups)
    while len(selected) < maximum and ordered_groups:
        next_groups = []
        for name in ordered_groups:
            if groups[name] and len(selected) < maximum:
                selected.append(groups[name].pop())
            if groups[name]:
                next_groups.append(name)
        ordered_groups = next_groups
    rng.shuffle(selected)
    return selected


def main() -> int:
    args = parse_args()
    parquet = args.parquet.resolve()
    base_package = args.base_package.resolve()
    output = args.output_dir.resolve()
    if args.max_synthetic < 1 or args.workers < 1 or args.query_timeout <= 0:
        raise ValueError("max-synthetic, workers, and query-timeout must be positive")
    base_train_path = base_package / "train_base.jsonl"
    validation_path = base_package / "validation.jsonl"
    for path in (parquet, base_train_path, validation_path):
        if not path.exists():
            raise FileNotFoundError(path)

    base_train = read_jsonl(base_train_path)
    validation = read_jsonl(validation_path)
    protected_questions = {
        normalized(str(message["content"]).rsplit("Question:", 1)[-1])
        for row in [*base_train, *validation]
        for message in row["messages"]
        if message["role"] == "user"
    }
    frame = pd.read_parquet(
        parquet,
        columns=["id", "domain", "sql_complexity", "sql_task_type", "sql_prompt", "sql_context", "sql"],
    )
    source_rows = frame.to_dict(orient="records")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        outcomes = list(executor.map(lambda row: validate_one(row, args.query_timeout), source_rows))

    rejection_counts = Counter(status for status, _row in outcomes)
    candidates = [row for status, row in outcomes if status == "accepted" and row is not None]
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    leakage_rejections = 0
    for row in candidates:
        if row["_question_key"] in protected_questions:
            leakage_rejections += 1
            continue
        key = (row["_question_key"], row["_sql_key"], row["_schema_key"])
        unique.setdefault(key, row)
    deduplicated = list(unique.values())
    selected = select_balanced(deduplicated, min(args.max_synthetic, len(deduplicated)), args.seed)
    for row in selected:
        for key in ("_question_key", "_sql_key", "_schema_key"):
            row.pop(key, None)

    combined = [*base_train, *selected]
    output.mkdir(parents=True, exist_ok=True)
    counts = {
        "source_rows": len(source_rows),
        "execution_accepted": len(candidates),
        "deduplicated_accepted": len(deduplicated),
        "validation_question_rejections": leakage_rejections,
        "selected_synthetic": len(selected),
        "spider_base": len(base_train),
        "combined_train": len(combined),
        "validation": len(validation),
    }
    atomic_jsonl(output / "train_base.jsonl", combined)
    atomic_jsonl(output / "train_curriculum.jsonl", combined)
    atomic_jsonl(output / "validation.jsonl", validation)
    manifest = {
        "package": "spider_gretel_exec_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "filter": {
            "read_only_targets_only": True,
            "sqlite_context_and_target_execution_required": True,
            "schema_reconstructed_from_sqlite_catalog": True,
            "exact_normalized_question_isolation_against_spider_train_and_validation": True,
            "balanced_round_robin_by_source_complexity": True,
            "query_timeout_seconds": args.query_timeout,
        },
        "sources": {
            "gretel_parquet": {"path": str(parquet), "sha256": sha256(parquet)},
            "spider_base_train": {"path": str(base_train_path), "sha256": sha256(base_train_path)},
            "spider_validation": {"path": str(validation_path), "sha256": sha256(validation_path)},
        },
        "counts": counts,
        "outcomes": dict(sorted(rejection_counts.items())),
        "selected_complexity": dict(sorted(Counter(
            str(row["sampling"]["source_complexity"]) for row in selected
        ).items())),
        "selected_domains": len({str(row["sampling"]["source_domain"]) for row in selected}),
    }
    atomic_json(output / "manifest.json", manifest)
    checksums = {
        name: sha256(output / name)
        for name in ("train_base.jsonl", "train_curriculum.jsonl", "validation.jsonl", "manifest.json")
    }
    atomic_json(output / "checksums.json", checksums)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
