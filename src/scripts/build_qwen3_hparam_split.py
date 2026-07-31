#!/usr/bin/env python3
"""Build a deterministic database-disjoint QLoRA tuning split from Spider train."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "finetuning" / "spider_mschema_sft_v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "finetuning" / "qwen3_hparam_mschema_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-databases", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--search-trials", type=int, default=50_000)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def complexity(row: dict[str, Any]) -> str:
    return str(row.get("metadata", {}).get("query_features", {}).get("complexity_proxy", "unknown"))


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "databases": len({str(row["db_id"]) for row in rows}),
        "complexity": dict(sorted(Counter(complexity(row) for row in rows).items())),
    }


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    rows = read_jsonl(source / "train_base.jsonl")
    by_database: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_database[str(row["db_id"])].append(row)
    database_ids = sorted(by_database)
    if not 1 <= args.validation_databases < len(database_ids):
        raise ValueError("validation-databases must leave at least one training database")
    if args.search_trials < 1:
        raise ValueError("search-trials must be positive")

    labels = sorted({complexity(row) for row in rows})
    target_fraction = args.validation_databases / len(database_ids)
    total_by_label = Counter(complexity(row) for row in rows)
    target_rows = len(rows) * target_fraction
    target_labels = {label: total_by_label[label] * target_fraction for label in labels}

    def objective(selected: tuple[str, ...]) -> float:
        selected_rows = [row for db_id in selected for row in by_database[db_id]]
        counts = Counter(complexity(row) for row in selected_rows)
        row_error = abs(len(selected_rows) - target_rows) / max(1.0, target_rows)
        label_error = sum(
            abs(counts[label] - target_labels[label]) / max(1.0, target_labels[label])
            for label in labels
        ) / max(1, len(labels))
        return row_error + label_error

    rng = random.Random(args.seed)
    best: tuple[float, tuple[str, ...]] | None = None
    for _ in range(args.search_trials):
        selected = tuple(sorted(rng.sample(database_ids, args.validation_databases)))
        score = objective(selected)
        if best is None or (score, selected) < best:
            best = (score, selected)
    assert best is not None
    validation_ids = set(best[1])
    training_rows = [row for row in rows if str(row["db_id"]) not in validation_ids]
    validation_rows = [row for row in rows if str(row["db_id"]) in validation_ids]
    train_ids = {str(row["db_id"]) for row in training_rows}
    dev_ids = {str(row["db_id"]) for row in validation_rows}
    if train_ids & dev_ids:
        raise AssertionError("Database leakage in generated tuning split")
    if len(training_rows) + len(validation_rows) != len(rows):
        raise AssertionError("Generated split lost rows")

    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "train_base.jsonl", training_rows)
    # The generic bundle builder requires this filename. HPO deliberately uses
    # only the natural distribution, so curriculum is an identical alias.
    write_jsonl(output / "train_curriculum.jsonl", training_rows)
    write_jsonl(output / "validation.jsonl", validation_rows)
    manifest = {
        "format_version": 1,
        "package": "qwen3_hparam_mschema_v1",
        "purpose": "database-disjoint hyperparameter selection from Spider training databases",
        "source": str(source.relative_to(PROJECT_ROOT)),
        "source_train_sha256": sha256(source / "train_base.jsonl"),
        "seed": args.seed,
        "search_trials": args.search_trials,
        "selection_objective": "minimize row-count and complexity-distribution deviation",
        "objective_value": round(best[0], 12),
        "training": summary(training_rows),
        "validation": summary(validation_rows),
        "training_database_ids": sorted(train_ids),
        "validation_database_ids": sorted(dev_ids),
        "database_overlap": sorted(train_ids & dev_ids),
        "official_spider_validation_used": False,
    }
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
