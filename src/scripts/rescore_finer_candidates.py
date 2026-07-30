#!/usr/bin/env python3
"""Reselect saved FINER candidates with the authors' exact published VAV rule."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import evaluate_text2sql_models as core


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--query-timeout", type=float, default=3.0)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def exact_mcnemar(corrected: int, regressed: int) -> float:
    discordant = corrected + regressed
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, value) for value in range(min(corrected, regressed) + 1)) / 2**discordant
    return min(1.0, 2.0 * tail)


def main() -> int:
    args = parse_args()
    source_rows = core.read_jsonl(args.predictions.resolve())
    source = {str(row["id"]): row for row in source_rows}
    validation_rows = core.read_jsonl(args.validation.resolve())
    if args.limit is not None:
        validation_rows = validation_rows[: args.limit]
    expected = {str(row["id"]) for row in validation_rows}
    if not expected.issubset(source):
        raise ValueError("Prediction and validation IDs do not align exactly")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rescored: list[dict[str, Any]] = []
    corrected = 0
    regressed = 0
    prediction_path = output / "predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for row in validation_rows:
            item = str(row["id"])
            original = source[item]
            candidates = original.get("candidates") or []
            database = core.resolve_database(args.project_root.resolve(), row)
            order_sensitive = bool(core.ORDER_BY.search(str(row["sql"])))
            gold = core.execute_query(database, str(row["sql"]), args.query_timeout, order_sensitive)
            cache: dict[str, core.QueryResult] = {}

            def execute(sql: str) -> core.QueryResult:
                if sql not in cache:
                    cache[sql] = core.execute_query(database, sql, args.query_timeout, order_sensitive)
                return cache[sql]

            results = [execute(str(candidate.get("predicted_sql") or "")) for candidate in candidates]
            selected_index, votes = core.select_finer_published_vav(results)
            selected = candidates[selected_index] if candidates else {
                "raw_prediction": "",
                "predicted_sql": "",
                "output_tokens": 0,
            }
            predicted_sql = str(selected.get("predicted_sql") or "")
            predicted_result = results[selected_index] if results else core.QueryResult(
                "error", None, None, 0.0, "no candidates"
            )
            matched = bool(
                gold.status == "ok"
                and predicted_result.status == "ok"
                and gold.column_count == predicted_result.column_count
                and gold.rows == predicted_result.rows
            )
            predicted_canonical, syntax_error = core.canonical_sql(predicted_sql)
            gold_canonical, gold_syntax_error = core.canonical_sql(str(row["sql"]))
            raw = str(selected.get("raw_prediction") or "")
            record = dict(original)
            record.update(
                {
                    "raw_prediction": raw,
                    "predicted_sql": predicted_sql,
                    "raw_exact_match": raw.strip() == str(row["sql"]).strip(),
                    "normalized_exact_match": core.normalize_sql(predicted_sql) == core.normalize_sql(str(row["sql"])),
                    "canonical_exact_match": predicted_canonical is not None and predicted_canonical == gold_canonical,
                    "syntax_valid": predicted_canonical is not None,
                    "syntax_error": syntax_error,
                    "gold_syntax_error": gold_syntax_error,
                    "execution_match": matched,
                    "gold_execution_status": gold.status,
                    "prediction_execution_status": predicted_result.status,
                    "prediction_execution_error": predicted_result.error,
                    "format_compliant": raw.strip() == predicted_sql and bool(core.READ_ONLY_PREFIX.match(raw.strip())),
                    "output_tokens": int(selected.get("output_tokens") or 0),
                    "selected_candidate_index": selected_index,
                    "execution_consensus_votes": votes,
                    "candidate_selection": "finer-published-vav",
                }
            )
            original_correct = bool(original["execution_match"])
            corrected += int(not original_correct and matched)
            regressed += int(original_correct and not matched)
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            rescored.append(record)

    metrics = core.summarize(rescored)
    metrics.update(
        {
            "candidate_selection": "finer-published-vav",
            "paired_vs_input_selector": {
                "corrected": corrected,
                "regressed": regressed,
                "net": corrected - regressed,
                "net_percentage_points": round(100.0 * (corrected - regressed) / len(rescored), 3),
                "exact_mcnemar_p": exact_mcnemar(corrected, regressed),
            },
            "source": str(args.predictions.resolve()),
        }
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
