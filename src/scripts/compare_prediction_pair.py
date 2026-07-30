#!/usr/bin/env python3
"""Paired execution comparison for two aligned text-to-SQL prediction files."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--before-label", default="before")
    parser.add_argument("--after-label", default="after")
    parser.add_argument("--title", default="Paired text-to-SQL comparison")
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


def metrics(rows: dict[str, dict[str, Any]], ids: list[str]) -> dict[str, Any]:
    fields = (
        "execution_match", "normalized_exact_match", "canonical_exact_match",
        "syntax_valid", "format_compliant",
    )
    return {
        "examples": len(ids),
        **{
            field: {
                "count": (count := sum(bool(rows[item].get(field)) for item in ids)),
                "pct": percentage(count, len(ids)),
            }
            for field in fields
        },
        "execution_status": dict(Counter(str(rows[item].get("prediction_execution_status")) for item in ids)),
    }


def comparison(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]], ids: list[str]) -> dict[str, Any]:
    corrected = [item for item in ids if not before[item]["execution_match"] and after[item]["execution_match"]]
    regressed = [item for item in ids if before[item]["execution_match"] and not after[item]["execution_match"]]
    return {
        "corrected": len(corrected),
        "regressed": len(regressed),
        "net": len(corrected) - len(regressed),
        "net_percentage_points": round(100.0 * (len(corrected) - len(regressed)) / len(ids), 3),
        "discordant": len(corrected) + len(regressed),
        "exact_mcnemar_p": exact_mcnemar(len(corrected), len(regressed)),
        "corrected_ids": corrected,
        "regressed_ids": regressed,
    }


def main() -> int:
    args = parse_args()
    before = {str(row["id"]): row for row in read_jsonl(args.before.resolve())}
    after = {str(row["id"]): row for row in read_jsonl(args.after.resolve())}
    validation_rows = read_jsonl(args.validation.resolve())
    validation = {str(row["id"]): row for row in validation_rows}
    ids = [str(row["id"]) for row in validation_rows]
    expected = set(ids)
    if set(before) != expected or set(after) != expected:
        raise ValueError("Prediction IDs do not align exactly with validation")

    before_metrics = metrics(before, ids)
    after_metrics = metrics(after, ids)
    paired = comparison(before, after, ids)
    slices = []
    for complexity in ("simple", "moderate", "complex"):
        selected = [
            item for item in ids
            if validation[item].get("metadata", {}).get("query_features", {}).get("complexity_proxy") == complexity
        ]
        before_correct = sum(bool(before[item]["execution_match"]) for item in selected)
        after_correct = sum(bool(after[item]["execution_match"]) for item in selected)
        slices.append({
            "slice": complexity,
            "examples": len(selected),
            "before_correct": before_correct,
            "before_pct": percentage(before_correct, len(selected)),
            "after_correct": after_correct,
            "after_pct": percentage(after_correct, len(selected)),
            "delta_pp": round(100.0 * (after_correct - before_correct) / len(selected), 3),
        })

    result = {
        "title": args.title,
        "labels": {"before": args.before_label, "after": args.after_label},
        "sources": {"before": str(args.before.resolve()), "after": str(args.after.resolve())},
        "metrics": {"before": before_metrics, "after": after_metrics},
        "paired_execution": paired,
        "complexity_slices": slices,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output / "complexity_slices.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(slices[0]))
        writer.writeheader()
        writer.writerows(slices)
    report = [
        f"# {args.title}", "",
        "| Variant | Execution | Normalized exact | Syntax valid |",
        "|---|---:|---:|---:|",
        f"| {args.before_label} | {before_metrics['execution_match']['pct']:.3f}% "
        f"({before_metrics['execution_match']['count']}/{len(ids)}) | "
        f"{before_metrics['normalized_exact_match']['pct']:.3f}% | {before_metrics['syntax_valid']['pct']:.3f}% |",
        f"| {args.after_label} | {after_metrics['execution_match']['pct']:.3f}% "
        f"({after_metrics['execution_match']['count']}/{len(ids)}) | "
        f"{after_metrics['normalized_exact_match']['pct']:.3f}% | {after_metrics['syntax_valid']['pct']:.3f}% |",
        "", "## Paired change", "",
        f"{paired['corrected']} corrected, {paired['regressed']} regressed, net {paired['net']:+d} "
        f"({paired['net_percentage_points']:+.3f} pp); exact McNemar p={paired['exact_mcnemar_p']:.6g}.",
        "", "## Complexity", "",
        "| Slice | N | Before | After | Delta |", "|---|---:|---:|---:|---:|",
    ]
    report.extend(
        f"| {row['slice']} | {row['examples']} | {row['before_pct']:.3f}% | "
        f"{row['after_pct']:.3f}% | {row['delta_pp']:+.3f} pp |"
        for row in slices
    )
    report.extend(["", "Execution is result equivalence on the supplied read-only SQLite databases.", ""])
    (output / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "paired_execution": paired}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
