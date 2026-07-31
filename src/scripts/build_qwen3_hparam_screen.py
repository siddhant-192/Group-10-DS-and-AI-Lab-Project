#!/usr/bin/env python3
"""Create a fixed stratified screening budget from the full Qwen3 HPO split."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "finetuning" / "qwen3_hparam_mschema_v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "finetuning" / "qwen3_hparam_mschema_screen_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-rows", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def complexity(row: dict[str, Any]) -> str:
    return str(row.get("metadata", {}).get("query_features", {}).get("complexity_proxy", "unknown"))


def stable_key(seed: int, row: dict[str, Any]) -> str:
    return hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def proportional_quotas(counts: Counter[str], total: int) -> dict[str, int]:
    raw = {label: total * count / sum(counts.values()) for label, count in counts.items()}
    quotas = {label: math.floor(value) for label, value in raw.items()}
    remainder = total - sum(quotas.values())
    for label in sorted(raw, key=lambda item: (-(raw[item] - quotas[item]), item))[:remainder]:
        quotas[label] += 1
    return quotas


def round_robin_database_sample(
    rows: list[dict[str, Any]], quota: int, seed: int
) -> list[dict[str, Any]]:
    by_database: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_database[str(row["db_id"])].append(row)
    for database_rows in by_database.values():
        database_rows.sort(key=lambda row: stable_key(seed, row))
    databases = sorted(by_database, key=lambda db_id: hashlib.sha256(f"{seed}:{db_id}".encode()).hexdigest())
    selected: list[dict[str, Any]] = []
    cursor = 0
    while len(selected) < quota:
        made_progress = False
        for db_id in databases:
            database_rows = by_database[db_id]
            if cursor < len(database_rows):
                selected.append(database_rows[cursor])
                made_progress = True
                if len(selected) == quota:
                    break
        if not made_progress:
            raise RuntimeError("Quota exceeds available rows")
        cursor += 1
    return selected


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    training = read_jsonl(source / "train_base.jsonl")
    validation = read_jsonl(source / "validation.jsonl")
    if not 1 <= args.train_rows <= len(training):
        raise ValueError("train-rows must be between 1 and the full training size")
    counts = Counter(complexity(row) for row in training)
    quotas = proportional_quotas(counts, args.train_rows)
    selected: list[dict[str, Any]] = []
    for offset, label in enumerate(sorted(quotas)):
        label_rows = [row for row in training if complexity(row) == label]
        selected.extend(round_robin_database_sample(label_rows, quotas[label], args.seed + offset))
    selected.sort(key=lambda row: str(row["id"]))
    if len({str(row["id"]) for row in selected}) != len(selected):
        raise AssertionError("Screening sample contains duplicate IDs")

    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "train_base.jsonl", selected)
    write_jsonl(output / "train_curriculum.jsonl", selected)
    write_jsonl(output / "validation.jsonl", validation)
    train_dbs = {str(row["db_id"]) for row in selected}
    validation_dbs = {str(row["db_id"]) for row in validation}
    manifest = {
        "format_version": 1,
        "package": "qwen3_hparam_mschema_screen_v1",
        "purpose": "successive-halving screening budget",
        "source": str(source.relative_to(PROJECT_ROOT)),
        "seed": args.seed,
        "train_rows": len(selected),
        "train_databases": len(train_dbs),
        "train_complexity": dict(sorted(Counter(complexity(row) for row in selected).items())),
        "validation_rows": len(validation),
        "validation_databases": len(validation_dbs),
        "database_overlap": sorted(train_dbs & validation_dbs),
        "selection": "proportional complexity quotas with database round-robin sampling",
        "official_spider_validation_used": False,
    }
    if manifest["database_overlap"]:
        raise AssertionError("Database leakage in screening package")
    atomic_write(output / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    checksums = {
        name: sha256(output / name)
        for name in ("train_base.jsonl", "train_curriculum.jsonl", "validation.jsonl", "manifest.json")
    }
    atomic_write(output / "checksums.json", json.dumps(checksums, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
