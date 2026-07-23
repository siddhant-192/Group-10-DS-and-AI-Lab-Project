#!/usr/bin/env python3
"""Score saved predictions with strict and official MAC-SQL Spider EX semantics."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from evaluate_text2sql_models import execution_match, macsql_execution_match


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", action="append", required=True, help="LABEL=predictions.jsonl")
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--query-timeout", type=float, default=3.0)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentage(count: int, total: int) -> float:
    return round(100.0 * count / total, 3) if total else 0.0


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    strict = sum(bool(record["strict_execution_match"]) for record in records)
    official = sum(bool(record["macsql_execution_match"]) for record in records)
    promoted = sum(
        bool(record["macsql_execution_match"] and not record["strict_execution_match"])
        for record in records
    )
    by_complexity: dict[str, dict[str, int]] = defaultdict(lambda: {"examples": 0, "strict": 0, "macsql": 0})
    for record in records:
        bucket = by_complexity[str(record["complexity"])]
        bucket["examples"] += 1
        bucket["strict"] += int(bool(record["strict_execution_match"]))
        bucket["macsql"] += int(bool(record["macsql_execution_match"]))
    return {
        "examples": total,
        "strict_execution": {"count": strict, "pct": percentage(strict, total)},
        "macsql_execution": {"count": official, "pct": percentage(official, total)},
        "column_permutation_or_distinct_promotions": {
            "count": promoted,
            "pct": percentage(promoted, total),
        },
        "prediction_statuses": dict(Counter(record["prediction_execution_status"] for record in records)),
        "by_complexity": {
            complexity: {
                **counts,
                "strict_pct": percentage(counts["strict"], counts["examples"]),
                "macsql_pct": percentage(counts["macsql"], counts["examples"]),
            }
            for complexity, counts in sorted(by_complexity.items())
        },
    }


def main() -> int:
    args = parse_args()
    validation_rows = read_jsonl(args.validation.resolve())
    if args.limit is not None:
        validation_rows = validation_rows[: args.limit]
    validation = {str(row["id"]): row for row in validation_rows}
    ids = list(validation)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: dict[str, Any] = {}

    for source in args.predictions:
        if "=" not in source:
            raise ValueError(f"Expected LABEL=PATH, received {source!r}")
        label, raw_path = source.split("=", 1)
        predictions = {str(row["id"]): row for row in read_jsonl(Path(raw_path).resolve())}
        missing = [item for item in ids if item not in predictions]
        if missing:
            raise ValueError(f"{label}: missing {len(missing)} validation predictions")
        strict_cache = {}
        macsql_cache = {}
        records = []
        for item in ids:
            validation_row = validation[item]
            prediction_row = predictions[item]
            sql = str(prediction_row.get("predicted_sql") or "")
            strict, _strict_gold, strict_predicted = execution_match(
                args.project_root.resolve(), validation_row, sql, args.query_timeout, strict_cache
            )
            official, _official_gold, _official_predicted = macsql_execution_match(
                args.project_root.resolve(), validation_row, sql, args.query_timeout, macsql_cache
            )
            records.append(
                {
                    "id": item,
                    "db_id": validation_row["db_id"],
                    "complexity": validation_row.get("metadata", {}).get("query_features", {}).get("complexity_proxy", "unknown"),
                    "predicted_sql": sql,
                    "strict_execution_match": strict,
                    "macsql_execution_match": official,
                    "prediction_execution_status": strict_predicted.status,
                    "prediction_execution_error": strict_predicted.error,
                }
            )
        label_dir = output_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        with (label_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        metrics = summarize(records)
        metrics["source"] = str(Path(raw_path).resolve())
        (label_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        all_metrics[label] = metrics

    (output_dir / "metrics.json").write_text(
        json.dumps(all_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(all_metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
