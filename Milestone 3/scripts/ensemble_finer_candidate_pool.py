#!/usr/bin/env python3
"""Blend FINER sampled execution clusters with weighted independent model votes."""

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
    finer_published_signature,
    finer_published_signature_is_all_zero,
    macsql_postprocess,
    macsql_result_equal,
    resolve_database,
    strip_distinct_sql,
    value_aware_result_is_all_zero,
    value_aware_signature,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finer", type=Path, required=True)
    parser.add_argument("--core", action="append", required=True, help="LABEL=predictions.jsonl")
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--core-weight-grid", default="0,1,2,3,4,5,6,7,8,9,10,11,12")
    parser.add_argument("--query-timeout", type=float, default=3.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--signature-mode", choices=("published", "robust"), default="published")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentage(count: int, total: int) -> float:
    return round(100.0 * count / total, 3) if total else 0.0


def signature_key(signature: Any) -> str:
    return json.dumps(signature, sort_keys=True)


def choose_signature(
    finer_signatures: list[Any | None],
    core_signatures: list[Any | None],
    core_weight: int,
    signature_mode: str = "robust",
) -> tuple[Any | None, int]:
    counts: Counter[Any] = Counter(
        signature for signature in finer_signatures if signature is not None
    )
    if core_weight:
        for signature in core_signatures:
            if signature is not None:
                counts[signature] += core_weight
    if not counts:
        return None, 0
    if signature_mode == "published":
        preferred = {
            signature: votes
            for signature, votes in counts.items()
            if signature and not finer_published_signature_is_all_zero(signature)
        }
    else:
        preferred = {
            signature: votes
            for signature, votes in counts.items()
            if signature and not value_aware_result_is_all_zero(signature)
        }
    eligible = preferred or counts
    return max(eligible.items(), key=lambda item: (item[1], signature_key(item[0])))


def strict_match(gold: QueryResult, predicted: QueryResult) -> bool:
    return bool(
        gold.status == "ok"
        and predicted.status == "ok"
        and gold.column_count == predicted.column_count
        and gold.rows == predicted.rows
    )


def main() -> int:
    args = parse_args()
    weights = sorted({int(value) for value in args.core_weight_grid.split(",")})
    if not weights or min(weights) < 0:
        raise ValueError("Core weights must be non-negative integers")
    validation_rows = read_jsonl(args.validation.resolve())
    if args.limit is not None:
        validation_rows = validation_rows[: args.limit]
    ids = [str(row["id"]) for row in validation_rows]
    validation = {str(row["id"]): row for row in validation_rows}
    finer = {str(row["id"]): row for row in read_jsonl(args.finer.resolve())}
    core_inputs: list[tuple[str, Path]] = []
    for value in args.core:
        if "=" not in value:
            raise ValueError(f"Expected LABEL=PATH, received {value!r}")
        label, raw_path = value.split("=", 1)
        core_inputs.append((label, Path(raw_path).resolve()))
    core_rows = {
        label: {str(row["id"]): row for row in read_jsonl(path)}
        for label, path in core_inputs
    }
    expected = set(ids)
    if not expected.issubset(finer):
        raise ValueError("FINER predictions do not align to validation")
    for label, rows in core_rows.items():
        if not expected.issubset(rows):
            raise ValueError(f"{label} predictions do not align to validation")

    project_root = args.project_root.resolve()
    outcomes: list[dict[str, Any]] = []
    for item in ids:
        row = validation[item]
        database = resolve_database(project_root, row)
        order_sensitive = bool(ORDER_BY.search(str(row["sql"])))
        gold = execute_query(database, str(row["sql"]), args.query_timeout, order_sensitive)
        official_gold_sql = strip_distinct_sql(macsql_postprocess(str(row["sql"])))
        official_gold = execute_query(
            database, official_gold_sql, args.query_timeout, order_sensitive
        )
        result_cache: dict[str, QueryResult] = {}
        official_result_cache: dict[str, QueryResult] = {}

        def run(sql: str) -> QueryResult:
            if sql not in result_cache:
                result_cache[sql] = execute_query(database, sql, args.query_timeout, order_sensitive)
            return result_cache[sql]

        def run_official(sql: str) -> QueryResult:
            official_sql = strip_distinct_sql(macsql_postprocess(sql))
            if official_sql not in official_result_cache:
                official_result_cache[official_sql] = execute_query(
                    database, official_sql, args.query_timeout, order_sensitive
                )
            return official_result_cache[official_sql]

        finer_candidates = finer[item].get("candidates") or []
        finer_sqls = [str(candidate.get("predicted_sql") or "") for candidate in finer_candidates]
        finer_results = [run(sql) for sql in finer_sqls]
        signature_function = finer_published_signature if args.signature_mode == "published" else value_aware_signature
        finer_signatures = [signature_function(result) for result in finer_results]
        core_sqls = [str(core_rows[label][item]["predicted_sql"]) for label, _path in core_inputs]
        core_results = [run(sql) for sql in core_sqls]
        core_signatures = [signature_function(result) for result in core_results]
        candidate_oracle = any(strict_match(gold, result) for result in finer_results + core_results)
        official_candidate_oracle = any(
            macsql_result_equal(official_gold, run_official(sql), order_sensitive)
            for sql in finer_sqls + core_sqls
        )
        by_weight: dict[str, Any] = {}
        for weight in weights:
            winning, votes = choose_signature(
                finer_signatures, core_signatures, weight, args.signature_mode
            )
            selected_source = "none"
            selected_sql = ""
            selected_result = QueryResult("error", None, None, 0.0, "no executable candidate")
            if winning is not None:
                if winning in finer_signatures:
                    index = finer_signatures.index(winning)
                    selected_source = "finer"
                    selected_sql = finer_sqls[index]
                    selected_result = finer_results[index]
                else:
                    index = core_signatures.index(winning)
                    selected_source = core_inputs[index][0]
                    selected_sql = core_sqls[index]
                    selected_result = core_results[index]
            by_weight[str(weight)] = {
                "correct": strict_match(gold, selected_result),
                "strict_correct": strict_match(gold, selected_result),
                "macsql_correct": macsql_result_equal(
                    official_gold, run_official(selected_sql), order_sensitive
                ) if selected_sql else False,
                "selected_source": selected_source,
                "predicted_sql": selected_sql,
                "weighted_votes": votes,
            }
        outcomes.append(
            {
                "id": item,
                "db_id": row["db_id"],
                "complexity": row.get("metadata", {}).get("query_features", {}).get("complexity_proxy", "unknown"),
                "candidate_oracle": candidate_oracle,
                "macsql_candidate_oracle": official_candidate_oracle,
                "by_weight": by_weight,
            }
        )

    scores = {
        weight: sum(bool(row["by_weight"][str(weight)]["correct"]) for row in outcomes)
        for weight in weights
    }
    macsql_scores = {
        weight: sum(bool(row["by_weight"][str(weight)]["macsql_correct"]) for row in outcomes)
        for weight in weights
    }
    best_weight = max(weights, key=lambda weight: (scores[weight], -weight))
    best_macsql_weight = max(weights, key=lambda weight: (macsql_scores[weight], -weight))

    # Leave-one-database-out: choose the weight only on other databases, then
    # score the unseen database. This avoids per-example validation-label routing.
    by_database: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in outcomes:
        by_database[str(row["db_id"])].append(row)
    cv_predictions: list[dict[str, Any]] = []
    macsql_cv_predictions: list[dict[str, Any]] = []
    selected_weights: dict[str, int] = {}
    macsql_selected_weights: dict[str, int] = {}
    for held_out, held_rows in sorted(by_database.items()):
        training_rows = [row for row in outcomes if str(row["db_id"]) != held_out]
        training_scores = {
            weight: sum(bool(row["by_weight"][str(weight)]["correct"]) for row in training_rows)
            for weight in weights
        }
        selected_weight = max(weights, key=lambda weight: (training_scores[weight], -weight))
        selected_weights[held_out] = selected_weight
        macsql_training_scores = {
            weight: sum(
                bool(row["by_weight"][str(weight)]["macsql_correct"])
                for row in training_rows
            )
            for weight in weights
        }
        macsql_selected_weight = max(
            weights, key=lambda weight: (macsql_training_scores[weight], -weight)
        )
        macsql_selected_weights[held_out] = macsql_selected_weight
        for row in held_rows:
            cv_predictions.append({**row, "selected_weight": selected_weight, **row["by_weight"][str(selected_weight)]})
            macsql_cv_predictions.append(
                {
                    **row,
                    "selected_weight": macsql_selected_weight,
                    **row["by_weight"][str(macsql_selected_weight)],
                }
            )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "global_best_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in outcomes:
            handle.write(
                json.dumps(
                    {
                        **row,
                        "selected_weight": best_weight,
                        **row["by_weight"][str(best_weight)],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    with (output / "global_best_macsql_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in outcomes:
            handle.write(
                json.dumps(
                    {
                        **row,
                        "selected_weight": best_macsql_weight,
                        **row["by_weight"][str(best_macsql_weight)],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    with (output / "leave_one_database_out_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in cv_predictions:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (output / "leave_one_database_out_macsql_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in macsql_cv_predictions:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    total = len(outcomes)
    cv_correct = sum(bool(row["correct"]) for row in cv_predictions)
    macsql_cv_correct = sum(bool(row["macsql_correct"]) for row in macsql_cv_predictions)
    oracle = sum(bool(row["candidate_oracle"]) for row in outcomes)
    macsql_oracle = sum(bool(row["macsql_candidate_oracle"]) for row in outcomes)
    metrics = {
        "examples": total,
        "signature_mode": args.signature_mode,
        "weights": {
            str(weight): {
                "strict_count": scores[weight],
                "strict_pct": percentage(scores[weight], total),
                "macsql_count": macsql_scores[weight],
                "macsql_pct": percentage(macsql_scores[weight], total),
            }
            for weight in weights
        },
        "global_best_exploratory": {
            "weight": best_weight,
            "count": scores[best_weight],
            "pct": percentage(scores[best_weight], total),
        },
        "global_best_macsql_exploratory": {
            "weight": best_macsql_weight,
            "count": macsql_scores[best_macsql_weight],
            "pct": percentage(macsql_scores[best_macsql_weight], total),
        },
        "leave_one_database_out": {
            "count": cv_correct,
            "pct": percentage(cv_correct, total),
            "selected_weights": selected_weights,
        },
        "leave_one_database_out_macsql": {
            "count": macsql_cv_correct,
            "pct": percentage(macsql_cv_correct, total),
            "selected_weights": macsql_selected_weights,
        },
        "combined_candidate_oracle": {"count": oracle, "pct": percentage(oracle, total)},
        "combined_macsql_candidate_oracle": {
            "count": macsql_oracle,
            "pct": percentage(macsql_oracle, total),
        },
        "sources": {
            "finer": str(args.finer.resolve()),
            **{label: str(path) for label, path in core_inputs},
        },
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
