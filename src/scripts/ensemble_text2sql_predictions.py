#!/usr/bin/env python3
"""Select among aligned model predictions by database-result consensus."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

from evaluate_text2sql_models import ORDER_BY, execute_query, execution_signature, resolve_database


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", action="append", required=True,
        help="Priority-ordered LABEL=predictions.jsonl input; repeat for every model.",
    )
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--query-timeout", type=float, default=3.0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentage(count: int, total: int) -> float:
    return round(100.0 * count / total, 3) if total else 0.0


def exact_mcnemar(corrected: int, regressed: int) -> float:
    discordant = corrected + regressed
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, value) for value in range(min(corrected, regressed) + 1)) / 2**discordant
    return min(1.0, 2.0 * tail)


def main() -> int:
    args = parse_args()
    if args.query_timeout <= 0:
        raise ValueError("--query-timeout must be positive")
    ordered_inputs = []
    for value in args.input:
        if "=" not in value:
            raise ValueError(f"Expected LABEL=PATH, received {value!r}")
        label, raw_path = value.split("=", 1)
        if not label or any(label == previous for previous, _path in ordered_inputs):
            raise ValueError(f"Input labels must be non-empty and unique: {label!r}")
        ordered_inputs.append((label, Path(raw_path).resolve()))
    validation_rows = read_jsonl(args.validation.resolve())
    ids = [str(row["id"]) for row in validation_rows]
    validation = {str(row["id"]): row for row in validation_rows}
    models = {
        label: {str(row["id"]): row for row in read_jsonl(path)}
        for label, path in ordered_inputs
    }
    expected = set(ids)
    for label, rows in models.items():
        if set(rows) != expected:
            raise ValueError(f"{label} predictions do not align exactly with validation")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "ensemble_predictions.jsonl"
    selected_records = []
    priority_label = ordered_inputs[0][0]
    corrected = 0
    regressed = 0
    with prediction_path.open("w", encoding="utf-8") as handle:
        for item in ids:
            row = validation[item]
            database = resolve_database(args.project_root.resolve(), row)
            gold_sql = str(row["sql"])
            order_sensitive = bool(ORDER_BY.search(gold_sql))
            gold = execute_query(database, gold_sql, args.query_timeout, order_sensitive)
            candidates = []
            signatures = []
            for label, _path in ordered_inputs:
                source = models[label][item]
                result = execute_query(
                    database, str(source["predicted_sql"]), args.query_timeout, order_sensitive
                )
                signature = execution_signature(result)
                signatures.append(signature)
                recomputed_match = bool(
                    gold.status == "ok" and result.status == "ok"
                    and gold.column_count == result.column_count and gold.rows == result.rows
                )
                if recomputed_match != bool(source["execution_match"]):
                    raise RuntimeError(f"Recomputed execution mismatch for {label}/{item}")
                candidates.append({
                    "label": label,
                    "predicted_sql": source["predicted_sql"],
                    "execution_status": result.status,
                    "execution_error": result.error,
                    "execution_match": recomputed_match,
                })
            counts = Counter(signature for signature in signatures if signature is not None)
            if counts:
                winning_votes = max(counts.values())
                winning_signatures = {signature for signature, votes in counts.items() if votes == winning_votes}
                selected_index = next(
                    index for index, signature in enumerate(signatures) if signature in winning_signatures
                )
            else:
                selected_index = 0
                winning_votes = 0
            selected = candidates[selected_index]
            oracle = any(candidate["execution_match"] for candidate in candidates)
            priority_correct = bool(candidates[0]["execution_match"])
            ensemble_correct = bool(selected["execution_match"])
            corrected += int(not priority_correct and ensemble_correct)
            regressed += int(priority_correct and not ensemble_correct)
            record = {
                "id": item,
                "db_id": row["db_id"],
                "complexity": row.get("metadata", {}).get("query_features", {}).get("complexity_proxy", "unknown"),
                "selected_label": selected["label"],
                "predicted_sql": selected["predicted_sql"],
                "execution_match": ensemble_correct,
                "execution_consensus_votes": winning_votes,
                "model_oracle_match": oracle,
                "candidates": candidates,
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            selected_records.append(record)

    total = len(ids)
    model_scores = {
        label: {
            "count": sum(bool(rows[item]["execution_match"]) for item in ids),
            "pct": percentage(sum(bool(rows[item]["execution_match"]) for item in ids), total),
        }
        for label, rows in models.items()
    }
    ensemble_count = sum(bool(row["execution_match"]) for row in selected_records)
    oracle_count = sum(bool(row["model_oracle_match"]) for row in selected_records)
    complexity = {}
    for name in ("simple", "moderate", "complex"):
        selected = [row for row in selected_records if row["complexity"] == name]
        count = sum(bool(row["execution_match"]) for row in selected)
        complexity[name] = {"examples": len(selected), "count": count, "pct": percentage(count, len(selected))}
    metrics = {
        "examples": total,
        "priority_order": [label for label, _path in ordered_inputs],
        "individual_models": model_scores,
        "ensemble": {"count": ensemble_count, "pct": percentage(ensemble_count, total)},
        "model_oracle": {"count": oracle_count, "pct": percentage(oracle_count, total)},
        "paired_vs_priority": {
            "priority_label": priority_label,
            "corrected": corrected,
            "regressed": regressed,
            "net": corrected - regressed,
            "net_percentage_points": round(100.0 * (corrected - regressed) / total, 3),
            "exact_mcnemar_p": exact_mcnemar(corrected, regressed),
        },
        "selection_counts": dict(Counter(row["selected_label"] for row in selected_records)),
        "consensus_vote_counts": dict(Counter(str(row["execution_consensus_votes"]) for row in selected_records)),
        "complexity": complexity,
        "sources": {label: str(path) for label, path in ordered_inputs},
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
