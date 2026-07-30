#!/usr/bin/env python3
"""Confidence-gate FINER execution voting against an independent core ensemble."""

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


def strict_equal(gold: QueryResult, prediction: QueryResult) -> bool:
    return bool(
        gold.status == "ok"
        and prediction.status == "ok"
        and gold.column_count == prediction.column_count
        and gold.rows == prediction.rows
    )


def select_cluster(results: list[QueryResult]) -> tuple[int, int, int, Any | None]:
    """Return representative index, top votes, runner-up votes, and signature."""
    signatures = [finer_published_signature(result) for result in results]
    executable = [signature for signature in signatures if signature is not None]
    if not executable:
        return 0, 0, 0, None
    preferred = [
        signature
        for signature in executable
        if signature and not finer_published_signature_is_all_zero(signature)
    ]
    counts = Counter(preferred or executable)
    ranked = sorted(counts.items(), key=lambda item: (item[1], str(item[0])), reverse=True)
    winner, top_votes = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0
    return signatures.index(winner), top_votes, runner_up, winner


def parameters() -> list[tuple[int, int, int]]:
    # Route to core only when it has this many agreeing independent models,
    # FINER's top cluster is no larger than the cap, and its top-vs-second
    # margin is no larger than the margin cap.
    return [
        (minimum_core, maximum_finer, maximum_margin)
        for minimum_core in range(1, 5)
        for maximum_finer in range(0, 31)
        for maximum_margin in (0, 1, 2, 3, 4, 6, 8, 12, 30)
    ]


def main() -> int:
    args = parse_args()
    validation_rows = read_jsonl(args.validation.resolve())
    if args.limit is not None:
        validation_rows = validation_rows[: args.limit]
    validation = {str(row["id"]): row for row in validation_rows}
    finer = {str(row["id"]): row for row in read_jsonl(args.finer.resolve())}
    core = {str(row["id"]): row for row in read_jsonl(args.core_ensemble.resolve())}
    ids = list(validation)
    for label, source in (("FINER", finer), ("core", core)):
        missing = set(ids) - set(source)
        if missing:
            raise ValueError(f"{label} is missing {len(missing)} validation IDs")

    project_root = args.project_root.resolve()
    outcomes: list[dict[str, Any]] = []
    for item in ids:
        row = validation[item]
        database = resolve_database(project_root, row)
        order_matters = bool(ORDER_BY.search(str(row["sql"])))
        cache: dict[str, QueryResult] = {}

        def run(sql: str) -> QueryResult:
            if sql not in cache:
                cache[sql] = execute_query(database, sql, args.query_timeout, order_matters)
            return cache[sql]

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
        finer_candidates = finer[item].get("candidates") or []
        finer_sqls = [str(candidate.get("predicted_sql") or "") for candidate in finer_candidates]
        if not finer_sqls:
            finer_sqls = [str(finer[item].get("predicted_sql") or "")]
        finer_results = [run(sql) for sql in finer_sqls]
        finer_index, finer_votes, runner_up_votes, finer_signature = select_cluster(finer_results)
        finer_sql = finer_sqls[finer_index]

        core_candidates = core[item].get("candidates") or []
        core_sqls = [str(candidate.get("predicted_sql") or "") for candidate in core_candidates]
        if not core_sqls:
            core_sqls = [str(core[item].get("predicted_sql") or "")]
        core_results = [run(sql) for sql in core_sqls]
        core_index, core_votes, _core_runner_up, core_signature = select_cluster(core_results)
        core_sql = core_sqls[core_index]

        def scores(sql: str) -> dict[str, bool]:
            return {
                "strict": strict_equal(gold, run(sql)),
                "macsql": macsql_result_equal(official_gold, run_official(sql), order_matters),
            }

        outcomes.append(
            {
                "id": item,
                "db_id": row["db_id"],
                "finer_sql": finer_sql,
                "core_sql": core_sql,
                "finer_votes": finer_votes,
                "finer_margin": finer_votes - runner_up_votes,
                "core_votes": core_votes,
                "signatures_agree": finer_signature is not None and finer_signature == core_signature,
                "finer_scores": scores(finer_sql),
                "core_scores": scores(core_sql),
            }
        )

    grid = parameters()

    def select(row: dict[str, Any], parameter: tuple[int, int, int]) -> str:
        minimum_core, maximum_finer, maximum_margin = parameter
        use_core = bool(
            not row["signatures_agree"]
            and row["core_votes"] >= minimum_core
            and row["finer_votes"] <= maximum_finer
            and row["finer_margin"] <= maximum_margin
        )
        return "core" if use_core else "finer"

    def score(rows: list[dict[str, Any]], parameter: tuple[int, int, int], metric: str) -> int:
        return sum(bool(row[f"{select(row, parameter)}_scores"][metric]) for row in rows)

    global_scores = {
        metric: {parameter: score(outcomes, parameter, metric) for parameter in grid}
        for metric in ("strict", "macsql")
    }
    global_best = {
        metric: max(grid, key=lambda parameter: (global_scores[metric][parameter], parameter))
        for metric in global_scores
    }

    by_database: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in outcomes:
        by_database[str(row["db_id"])].append(row)
    cv_records: dict[str, list[dict[str, Any]]] = {"strict": [], "macsql": []}
    cv_parameters: dict[str, dict[str, tuple[int, int, int]]] = {
        "strict": {},
        "macsql": {},
    }
    for held_database, held_rows in sorted(by_database.items()):
        train_rows = [row for row in outcomes if str(row["db_id"]) != held_database]
        for metric in ("strict", "macsql"):
            chosen = max(grid, key=lambda parameter: (score(train_rows, parameter, metric), parameter))
            cv_parameters[metric][held_database] = chosen
            for row in held_rows:
                source = select(row, chosen)
                cv_records[metric].append(
                    {
                        **row,
                        "selected_source": source,
                        "predicted_sql": row[f"{source}_sql"],
                        "selected_parameter": chosen,
                        "correct": row[f"{source}_scores"][metric],
                    }
                )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for metric, rows in cv_records.items():
        with (output / f"leave_one_database_out_{metric}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    metrics = {
        "examples": len(outcomes),
        "gold_blind_features": ["core_votes", "finer_votes", "finer_vote_margin", "signature_agreement"],
        "finer_only": {
            metric: {
                "count": sum(bool(row["finer_scores"][metric]) for row in outcomes),
                "pct": percentage(sum(bool(row["finer_scores"][metric]) for row in outcomes), len(outcomes)),
            }
            for metric in ("strict", "macsql")
        },
        "core_only": {
            metric: {
                "count": sum(bool(row["core_scores"][metric]) for row in outcomes),
                "pct": percentage(sum(bool(row["core_scores"][metric]) for row in outcomes), len(outcomes)),
            }
            for metric in ("strict", "macsql")
        },
        "global_best_exploratory": {
            metric: {
                "parameter": global_best[metric],
                "count": global_scores[metric][global_best[metric]],
                "pct": percentage(global_scores[metric][global_best[metric]], len(outcomes)),
            }
            for metric in global_best
        },
        "leave_one_database_out": {
            metric: {
                "count": sum(bool(row["correct"]) for row in cv_records[metric]),
                "pct": percentage(sum(bool(row["correct"]) for row in cv_records[metric]), len(outcomes)),
                "selected_parameters": cv_parameters[metric],
            }
            for metric in cv_records
        },
        "sources": {
            "finer": str(args.finer.resolve()),
            "core_ensemble": str(args.core_ensemble.resolve()),
        },
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
