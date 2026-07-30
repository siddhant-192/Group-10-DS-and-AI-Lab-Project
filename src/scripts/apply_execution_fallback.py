#!/usr/bin/env python3
"""Use a secondary model only when a core ensemble has no executable SQL.

The routing decision is gold-blind: it inspects only the core ensemble's
execution-consensus vote count and the fallback prediction's execution status.
Gold execution labels are used solely to report the resulting accuracy.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True, help="Core ensemble_predictions.jsonl")
    parser.add_argument("--fallback", type=Path, required=True, help="Fallback model predictions.jsonl")
    parser.add_argument("--fallback-label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    core_path = args.core.resolve()
    fallback_path = args.fallback.resolve()
    core_rows = read_jsonl(core_path)
    fallback_rows = {str(row["id"]): row for row in read_jsonl(fallback_path)}
    core_ids = [str(row["id"]) for row in core_rows]
    if len(core_ids) != len(set(core_ids)):
        raise ValueError("Core ensemble IDs are not unique")
    if set(core_ids) != set(fallback_rows):
        raise ValueError("Core and fallback prediction IDs do not align exactly")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "fallback_predictions.jsonl"
    selected_rows: list[dict[str, Any]] = []
    corrected = 0
    regressed = 0
    attempted = 0
    selected = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for core in core_rows:
            item = str(core["id"])
            fallback = fallback_rows[item]
            if str(core.get("db_id")) != str(fallback.get("db_id")):
                raise ValueError(f"Database mismatch for {item}")
            no_core_execution = int(core.get("execution_consensus_votes", 0)) == 0
            fallback_executable = fallback.get("prediction_execution_status") == "ok"
            use_fallback = no_core_execution and fallback_executable
            attempted += int(no_core_execution)
            selected += int(use_fallback)

            core_correct = bool(core["execution_match"])
            result_correct = bool(fallback["execution_match"]) if use_fallback else core_correct
            corrected += int(not core_correct and result_correct)
            regressed += int(core_correct and not result_correct)
            record = dict(core)
            record.update(
                {
                    "core_selected_label": core.get("selected_label"),
                    "fallback_attempted": no_core_execution,
                    "fallback_executable": fallback_executable,
                    "fallback_label": args.fallback_label,
                    "fallback_selected": use_fallback,
                    "selected_label": args.fallback_label if use_fallback else core.get("selected_label"),
                    "predicted_sql": fallback["predicted_sql"] if use_fallback else core["predicted_sql"],
                    "execution_match": result_correct,
                }
            )
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            selected_rows.append(record)

    total = len(selected_rows)
    core_count = sum(bool(row["execution_match"]) for row in core_rows)
    result_count = sum(bool(row["execution_match"]) for row in selected_rows)
    complexity: dict[str, dict[str, float | int]] = {}
    for name in ("simple", "moderate", "complex"):
        rows = [row for row in selected_rows if row.get("complexity") == name]
        count = sum(bool(row["execution_match"]) for row in rows)
        complexity[name] = {"examples": len(rows), "count": count, "pct": percentage(count, len(rows))}

    metrics = {
        "examples": total,
        "routing_rule": "select fallback iff core execution_consensus_votes == 0 and fallback execution status is ok",
        "gold_blind_routing": True,
        "core": {"count": core_count, "pct": percentage(core_count, total)},
        "fallback_opportunities": attempted,
        "fallback_selected": selected,
        "result": {"count": result_count, "pct": percentage(result_count, total)},
        "paired_vs_core": {
            "corrected": corrected,
            "regressed": regressed,
            "net": corrected - regressed,
            "net_percentage_points": round(100.0 * (corrected - regressed) / total, 3),
            "exact_mcnemar_p": exact_mcnemar(corrected, regressed),
        },
        "selection_counts": dict(Counter(row.get("selected_label") for row in selected_rows)),
        "complexity": complexity,
        "sources": {"core": str(core_path), "fallback": str(fallback_path)},
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
