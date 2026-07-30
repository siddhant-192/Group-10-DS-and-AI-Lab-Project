#!/usr/bin/env python3
"""Build distinct executable FINER/core result groups for ORM reranking."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from evaluate_text2sql_models import (
    ORDER_BY,
    QueryResult,
    execute_query,
    execution_signature,
    macsql_postprocess,
    macsql_result_equal,
    resolve_database,
    strip_distinct_sql,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finer", type=Path, required=True)
    parser.add_argument("--core-ensemble", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--query-timeout", type=float, default=3.0)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentage(value: int, total: int) -> float:
    return round(100.0 * value / total, 3) if total else 0.0


def strict_equal(gold: QueryResult, predicted: QueryResult) -> bool:
    return bool(
        gold.status == "ok"
        and predicted.status == "ok"
        and gold.column_count == predicted.column_count
        and gold.rows == predicted.rows
    )


def enriched_schema(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    user = next((str(message.get("content", "")) for message in messages if message.get("role") == "user"), "")
    marker = "Database Schema:"
    if marker in user and "\n\nQuestion:" in user:
        return user.split(marker, 1)[1].rsplit("\n\nQuestion:", 1)[0].strip()
    return str(row.get("schema", "")).strip()


def main() -> int:
    args = parse_args()
    validation_rows = read_jsonl(args.validation.resolve())
    if args.limit is not None:
        validation_rows = validation_rows[: args.limit]
    ids = [str(row["id"]) for row in validation_rows]
    finer = {str(row["id"]): row for row in read_jsonl(args.finer.resolve())}
    core = {str(row["id"]): row for row in read_jsonl(args.core_ensemble.resolve())}
    for label, source in (("FINER", finer), ("core", core)):
        missing = set(ids) - set(source)
        if missing:
            raise ValueError(f"{label} is missing {len(missing)} selected IDs")

    project_root = args.project_root.resolve()
    output_rows: list[dict[str, Any]] = []
    oracle_strict = 0
    oracle_macsql = 0
    candidate_counts: list[int] = []
    for row in validation_rows:
        item = str(row["id"])
        database = resolve_database(project_root, row)
        order_matters = bool(ORDER_BY.search(str(row["sql"])))
        result_cache: dict[str, QueryResult] = {}

        def run(sql: str) -> QueryResult:
            if sql not in result_cache:
                result_cache[sql] = execute_query(database, sql, args.query_timeout, order_matters)
            return result_cache[sql]

        official_cache: dict[str, QueryResult] = {}

        def run_official(sql: str) -> QueryResult:
            processed = strip_distinct_sql(macsql_postprocess(sql))
            if processed not in official_cache:
                official_cache[processed] = execute_query(
                    database, processed, args.query_timeout, order_matters
                )
            return official_cache[processed]

        gold = run(str(row["sql"]))
        official_gold = run_official(str(row["sql"]))
        candidates: list[dict[str, str]] = []
        finer_candidates = finer[item].get("candidates") or [finer[item]]
        candidates.extend(
            {
                "source": "finer",
                "sql": str(candidate.get("predicted_sql") or ""),
            }
            for candidate in finer_candidates
        )
        core_candidates = core[item].get("candidates") or [core[item]]
        candidates.extend(
            {
                "source": str(candidate.get("label") or "core"),
                "sql": str(candidate.get("predicted_sql") or ""),
            }
            for candidate in core_candidates
        )

        # The published MAC-SQL scorer removes DISTINCT and normalizes spaced
        # operators before execution. Two candidates can therefore share the
        # same raw result but differ under the official metric (or vice versa).
        # Keep both signatures in the gold-blind grouping key so reranking does
        # not silently discard an officially correct candidate.
        grouped: dict[tuple[Any, Any], list[dict[str, str]]] = defaultdict(list)
        for candidate in candidates:
            raw_signature = execution_signature(run(candidate["sql"]))
            official_signature = execution_signature(run_official(candidate["sql"]))
            if raw_signature is not None or official_signature is not None:
                grouped[(raw_signature, official_signature)].append(candidate)
        candidate_groups: list[dict[str, Any]] = []
        for (raw_signature, official_signature), values in grouped.items():
            sql_counts = Counter(candidate["sql"] for candidate in values)
            representative = max(
                sql_counts,
                key=lambda sql: (sql_counts[sql], -len(sql), sql),
            )
            sources = Counter(candidate["source"] for candidate in values)
            result = run(representative)
            official_result = run_official(representative)
            candidate_groups.append(
                {
                    "sql": representative,
                    "votes": len(values),
                    "distinct_sqls": len(sql_counts),
                    "sources": dict(sorted(sources.items())),
                    "signature": {
                        "raw": raw_signature,
                        "macsql": official_signature,
                    },
                    "audit_strict_match": strict_equal(gold, result),
                    "audit_macsql_match": macsql_result_equal(
                        official_gold, official_result, order_matters
                    ),
                }
            )
        candidate_groups.sort(
            key=lambda candidate: (candidate["votes"], candidate["sql"]), reverse=True
        )
        if not candidate_groups:
            fallback_sql = str(core[item].get("predicted_sql") or finer[item].get("predicted_sql") or "")
            candidate_groups = [
                {
                    "sql": fallback_sql,
                    "votes": 0,
                    "distinct_sqls": 1,
                    "sources": {"fallback": 1},
                    "signature": None,
                    "audit_strict_match": strict_equal(gold, run(fallback_sql)),
                    "audit_macsql_match": macsql_result_equal(
                        official_gold, run_official(fallback_sql), order_matters
                    ),
                }
            ]
        oracle_strict += int(any(candidate["audit_strict_match"] for candidate in candidate_groups))
        oracle_macsql += int(any(candidate["audit_macsql_match"] for candidate in candidate_groups))
        candidate_counts.append(len(candidate_groups))
        output_rows.append(
            {
                "id": item,
                "db_id": row["db_id"],
                "question": row["question"],
                "schema": enriched_schema(row),
                "candidate_groups": candidate_groups,
            }
        )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "candidate_groups.jsonl").open("w", encoding="utf-8") as handle:
        for row in output_rows:
            safe = {
                **row,
                "candidate_groups": [
                    {
                        key: value
                        for key, value in candidate.items()
                        if not key.startswith("audit_")
                    }
                    for candidate in row["candidate_groups"]
                ],
            }
            handle.write(json.dumps(safe, sort_keys=True) + "\n")
    with (output / "candidate_groups_audit.jsonl").open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    total = len(output_rows)
    metrics = {
        "examples": total,
        "total_candidate_groups": sum(candidate_counts),
        "candidate_groups_per_example": {
            "minimum": min(candidate_counts),
            "mean": round(sum(candidate_counts) / total, 3),
            "maximum": max(candidate_counts),
        },
        "candidate_oracle": {
            "strict": {"count": oracle_strict, "pct": percentage(oracle_strict, total)},
            "macsql": {"count": oracle_macsql, "pct": percentage(oracle_macsql, total)},
        },
        "sources": {
            "finer": str(args.finer.resolve()),
            "core_ensemble": str(args.core_ensemble.resolve()),
            "validation": str(args.validation.resolve()),
        },
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
