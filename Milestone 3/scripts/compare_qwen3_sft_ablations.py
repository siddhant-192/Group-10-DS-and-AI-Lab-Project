#!/usr/bin/env python3
"""Compare matched Qwen3 zero-shot, base-SFT, and curriculum-SFT predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Callable


FEATURES: dict[str, Callable[[dict[str, Any]], bool]] = {
    "aggregate": lambda f: int(f.get("aggregate_count", 0)) > 0,
    "where": lambda f: bool(f.get("has_where")),
    "join": lambda f: int(f.get("join_count", 0)) > 0,
    "multi_join": lambda f: int(f.get("join_count", 0)) >= 2,
    "group_by": lambda f: bool(f.get("has_group_by")),
    "having": lambda f: bool(f.get("has_having")),
    "order_by": lambda f: bool(f.get("has_order_by")),
    "limit": lambda f: bool(f.get("has_limit")),
    "distinct": lambda f: bool(f.get("has_distinct")),
    "subquery": lambda f: bool(f.get("has_subquery")),
    "set_operation": lambda f: int(f.get("set_operation_count", 0)) > 0,
    "join_and_subquery": lambda f: int(f.get("join_count", 0)) > 0 and bool(f.get("has_subquery")),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zero-shot", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--curriculum", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pct(count: int, total: int) -> float:
    return round(100.0 * count / total, 3) if total else 0.0


def exact_mcnemar_p(corrected: int, regressed: int) -> float:
    discordant = corrected + regressed
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(corrected, regressed) + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def metric_summary(rows: dict[str, dict[str, Any]], ids: list[str]) -> dict[str, Any]:
    total = len(ids)
    fields = {
        "execution": "execution_match",
        "normalized_exact": "normalized_exact_match",
        "canonical_exact": "canonical_exact_match",
        "raw_exact": "raw_exact_match",
        "syntax_valid": "syntax_valid",
        "format_compliant": "format_compliant",
    }
    result: dict[str, Any] = {"examples": total}
    for label, field in fields.items():
        count = sum(bool(rows[item][field]) for item in ids)
        result[label] = {"count": count, "pct": pct(count, total)}
    result["execution_errors"] = sum(rows[item]["prediction_execution_status"] != "ok" for item in ids)
    return result


def paired(a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]], ids: list[str], field: str) -> dict[str, Any]:
    corrected = sum(not bool(a[item][field]) and bool(b[item][field]) for item in ids)
    regressed = sum(bool(a[item][field]) and not bool(b[item][field]) for item in ids)
    both_correct = sum(bool(a[item][field]) and bool(b[item][field]) for item in ids)
    both_wrong = len(ids) - corrected - regressed - both_correct
    return {
        "corrected": corrected,
        "regressed": regressed,
        "net": corrected - regressed,
        "net_percentage_points": round(100.0 * (corrected - regressed) / len(ids), 3),
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "discordant": corrected + regressed,
        "exact_mcnemar_p": exact_mcnemar_p(corrected, regressed),
    }


def slice_row(
    label: str,
    ids: list[str],
    models: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    values = {name: pct(sum(bool(rows[item]["execution_match"]) for item in ids), len(ids)) for name, rows in models.items()}
    base_pair = paired(models["zero_shot"], models["base"], ids, "execution_match")
    curriculum_pair = paired(models["base"], models["curriculum"], ids, "execution_match")
    return {
        "slice": label,
        "examples": len(ids),
        **{f"{name}_execution_pct": value for name, value in values.items()},
        "base_vs_zero_pp": round(values["base"] - values["zero_shot"], 3),
        "curriculum_vs_zero_pp": round(values["curriculum"] - values["zero_shot"], 3),
        "curriculum_vs_base_pp": round(values["curriculum"] - values["base"], 3),
        "base_corrected_vs_zero": base_pair["corrected"],
        "base_regressed_vs_zero": base_pair["regressed"],
        "base_vs_zero_mcnemar_p": base_pair["exact_mcnemar_p"],
        "curriculum_corrected_vs_base": curriculum_pair["corrected"],
        "curriculum_regressed_vs_base": curriculum_pair["regressed"],
        "curriculum_vs_base_mcnemar_p": curriculum_pair["exact_mcnemar_p"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    paths = {"zero_shot": args.zero_shot, "base": args.base, "curriculum": args.curriculum}
    lists = {name: read_jsonl(path.resolve()) for name, path in paths.items()}
    models = {name: {str(row["id"]): row for row in rows} for name, rows in lists.items()}
    validation_rows = read_jsonl(args.validation.resolve())
    validation = {str(row["id"]): row for row in validation_rows}
    ids = [str(row["id"]) for row in validation_rows]
    expected = set(ids)
    for name, rows in models.items():
        if set(rows) != expected or len(rows) != len(ids):
            raise ValueError(f"{name} predictions do not align with validation")

    metrics = {name: metric_summary(rows, ids) for name, rows in models.items()}
    pairings = {
        "base_vs_zero_shot": {
            field: paired(models["zero_shot"], models["base"], ids, field)
            for field in ("execution_match", "normalized_exact_match")
        },
        "curriculum_vs_zero_shot": {
            field: paired(models["zero_shot"], models["curriculum"], ids, field)
            for field in ("execution_match", "normalized_exact_match")
        },
        "curriculum_vs_base": {
            field: paired(models["base"], models["curriculum"], ids, field)
            for field in ("execution_match", "normalized_exact_match")
        },
    }

    feature_rows = []
    for label, predicate in FEATURES.items():
        selected = [
            item
            for item in ids
            if predicate(dict(validation[item].get("metadata", {}).get("query_features", {})))
        ]
        feature_rows.append(slice_row(label, selected, models))

    complexity_rows = []
    for label in ("simple", "moderate", "complex"):
        selected = [
            item
            for item in ids
            if validation[item].get("metadata", {}).get("query_features", {}).get("complexity_proxy") == label
        ]
        complexity_rows.append(slice_row(label, selected, models))

    database_rows = []
    for db_id in sorted({str(validation[item]["db_id"]) for item in ids}):
        selected = [item for item in ids if str(validation[item]["db_id"]) == db_id]
        database_rows.append({"db_id": db_id, **slice_row(db_id, selected, models)})

    transition_rows = []
    for comparison, fields in pairings.items():
        for metric, values in fields.items():
            transition_rows.append({"comparison": comparison, "metric": metric, **values})

    changed_examples = {}
    for comparison, source, target in (
        ("base_vs_zero_shot", "zero_shot", "base"),
        ("curriculum_vs_zero_shot", "zero_shot", "curriculum"),
        ("curriculum_vs_base", "base", "curriculum"),
    ):
        changed_examples[comparison] = {
            "corrected_ids": [item for item in ids if not models[source][item]["execution_match"] and models[target][item]["execution_match"]],
            "regressed_ids": [item for item in ids if models[source][item]["execution_match"] and not models[target][item]["execution_match"]],
        }

    result = {
        "examples": len(ids),
        "sources": {name: str(path.resolve()) for name, path in paths.items()},
        "metrics": metrics,
        "paired": pairings,
        "complexity_slices": complexity_rows,
        "feature_slices": feature_rows,
        "database_slices": database_rows,
        "changed_examples": changed_examples,
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(output_dir / "paired_transitions.csv", transition_rows)
    write_csv(output_dir / "complexity_slices.csv", complexity_rows)
    write_csv(output_dir / "feature_slices.csv", feature_rows)
    write_csv(output_dir / "database_slices.csv", database_rows)

    execution = {name: values["execution"] for name, values in metrics.items()}
    base_pair = pairings["base_vs_zero_shot"]["execution_match"]
    curriculum_pair = pairings["curriculum_vs_base"]["execution_match"]
    report = [
        "# Qwen3 SFT ablation comparison",
        "",
        "| Variant | Execution | Normalized exact | Execution errors |",
        "|---|---:|---:|---:|",
    ]
    for name in ("zero_shot", "base", "curriculum"):
        values = metrics[name]
        report.append(
            f"| {name} | {values['execution']['pct']:.3f}% ({values['execution']['count']}/{len(ids)}) | "
            f"{values['normalized_exact']['pct']:.3f}% | {values['execution_errors']} |"
        )
    report.extend(
        [
            "",
            "## Paired execution changes",
            "",
            f"- Base vs zero-shot: {base_pair['corrected']} corrected, {base_pair['regressed']} regressed, "
            f"net {base_pair['net']:+d} ({base_pair['net_percentage_points']:+.3f} pp), exact McNemar p={base_pair['exact_mcnemar_p']:.6g}.",
            f"- Curriculum vs base: {curriculum_pair['corrected']} corrected, {curriculum_pair['regressed']} regressed, "
            f"net {curriculum_pair['net']:+d} ({curriculum_pair['net_percentage_points']:+.3f} pp), exact McNemar p={curriculum_pair['exact_mcnemar_p']:.6g}.",
            "",
            "## Complexity execution accuracy",
            "",
            "| Slice | N | Zero-shot | Base | Curriculum | Curriculum - base |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in complexity_rows:
        report.append(
            f"| {row['slice']} | {row['examples']} | {row['zero_shot_execution_pct']:.3f}% | "
            f"{row['base_execution_pct']:.3f}% | {row['curriculum_execution_pct']:.3f}% | "
            f"{row['curriculum_vs_base_pp']:+.3f} pp |"
        )
    report.extend(
        [
            "",
            "The execution metric is result equivalence on the supplied read-only SQLite instances, not the official Spider test-suite evaluator.",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "execution": execution, "paired_execution": {k: v["execution_match"] for k, v in pairings.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
