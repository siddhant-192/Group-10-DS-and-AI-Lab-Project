#!/usr/bin/env python3
"""Analyze per-example text-to-SQL baseline predictions without rerunning models."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Sequence

from sqlglot import exp, parse_one


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "artifacts" / "zero-shot-eval" / "runs"
DEFAULT_VALIDATION = PROJECT_ROOT / "data" / "processed" / "spider" / "validation.jsonl"
DEFAULT_TRAIN = PROJECT_ROOT / "data" / "processed" / "spider" / "train.jsonl"
SQL_START = re.compile(r"\b(?:SELECT|WITH)\b", re.IGNORECASE)


@dataclass(frozen=True)
class GoldFeature:
    key: str
    label: str
    predicate: Callable[[dict[str, Any]], bool]


GOLD_FEATURES = (
    GoldFeature("aggregate", "Aggregate", lambda f: int(f.get("aggregate_count", 0)) > 0),
    GoldFeature("where", "WHERE", lambda f: bool(f.get("has_where"))),
    GoldFeature("join", "Any join", lambda f: int(f.get("join_count", 0)) > 0),
    GoldFeature("multi_join", "2+ joins", lambda f: int(f.get("join_count", 0)) >= 2),
    GoldFeature("group_by", "GROUP BY", lambda f: bool(f.get("has_group_by"))),
    GoldFeature("having", "HAVING", lambda f: bool(f.get("has_having"))),
    GoldFeature("order_by", "ORDER BY", lambda f: bool(f.get("has_order_by"))),
    GoldFeature("limit", "LIMIT", lambda f: bool(f.get("has_limit"))),
    GoldFeature("distinct", "DISTINCT", lambda f: bool(f.get("has_distinct"))),
    GoldFeature("subquery", "Subquery", lambda f: bool(f.get("has_subquery"))),
    GoldFeature("set_operation", "Set operation", lambda f: int(f.get("set_operation_count", 0)) > 0),
    GoldFeature(
        "join_and_subquery",
        "Join + subquery",
        lambda f: int(f.get("join_count", 0)) > 0 and bool(f.get("has_subquery")),
    ),
    GoldFeature("complex", "Complex proxy", lambda f: f.get("complexity_proxy") == "complex"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="Directory containing comparison.json and one predictions.jsonl per model.",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--representatives", type=int, default=8)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def latest_complete_results(runs_root: Path) -> Path:
    candidates = sorted(
        (
            path
            for path in runs_root.glob("*/downloaded/results")
            if (path / "comparison.json").exists()
        ),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No completed evaluation results found under {runs_root}")
    return candidates[0]


def percentage(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 3) if denominator else 0.0


def normalize_sql(sql: str) -> str:
    return " ".join(sql.strip().rstrip(";").casefold().split())


def query_features(row: dict[str, Any]) -> dict[str, Any]:
    return dict(row.get("metadata", {}).get("query_features", {}))


def parse_sql(sql: str) -> exp.Expression | None:
    try:
        return parse_one(sql, read="sqlite")
    except Exception:
        return None


def sql_signature(sql: str) -> dict[str, Any] | None:
    root = parse_sql(sql)
    if root is None:
        return None
    selects = list(root.find_all(exp.Select))
    aggregate_functions = sorted(node.key.lower() for node in root.find_all(exp.AggFunc))
    literals = sorted(
        f"{'string' if node.is_string else 'number'}:{str(node.this).casefold()}"
        for node in root.find_all(exp.Literal)
    )
    return {
        "tables": sorted({node.name.casefold() for node in root.find_all(exp.Table)}),
        "columns": sorted({node.name.casefold() for node in root.find_all(exp.Column)}),
        "aggregate_functions": aggregate_functions,
        "literals": literals,
        "join_count": sum(1 for _ in root.find_all(exp.Join)),
        "select_count": len(selects),
        "projection_count": sum(len(node.expressions) for node in selects),
        "subquery_count": sum(1 for _ in root.find_all(exp.Subquery)),
        "set_operation_count": sum(
            1 for node in root.walk() if isinstance(node, (exp.Union, exp.Intersect, exp.Except))
        ),
        "has_where": root.find(exp.Where) is not None,
        "has_group_by": root.find(exp.Group) is not None,
        "has_having": root.find(exp.Having) is not None,
        "has_order_by": root.find(exp.Order) is not None,
        "has_limit": root.find(exp.Limit) is not None,
        "has_distinct": any(node.args.get("distinct") is not None for node in selects),
    }


SIGNATURE_LABELS = {
    "tables": "table_selection_mismatch",
    "columns": "column_reference_mismatch",
    "aggregate_functions": "aggregation_mismatch",
    "literals": "literal_or_value_mismatch",
    "join_count": "join_structure_mismatch",
    "select_count": "select_or_subquery_count_mismatch",
    "projection_count": "projection_count_mismatch",
    "subquery_count": "subquery_structure_mismatch",
    "set_operation_count": "set_operation_mismatch",
    "has_where": "where_clause_mismatch",
    "has_group_by": "group_by_mismatch",
    "has_having": "having_mismatch",
    "has_order_by": "order_by_mismatch",
    "has_limit": "limit_mismatch",
    "has_distinct": "distinct_mismatch",
}


def structural_mismatches(gold_sql: str, predicted_sql: str) -> list[str]:
    gold = sql_signature(gold_sql)
    predicted = sql_signature(predicted_sql)
    if gold is None or predicted is None:
        return []
    mismatches = [label for key, label in SIGNATURE_LABELS.items() if gold[key] != predicted[key]]
    return mismatches or ["other_semantic_mismatch"]


def runtime_error_category(record: dict[str, Any]) -> str:
    status = str(record.get("prediction_execution_status", "unknown"))
    message = str(record.get("prediction_execution_error") or "").casefold()
    if status == "unsafe":
        return "non_sql_or_unsafe"
    if status == "timeout":
        return "timeout"
    if status == "too_many_rows":
        return "too_many_rows"
    patterns = (
        ("no such column", "unknown_column"),
        ("ambiguous column name", "ambiguous_column"),
        ("no such table", "unknown_table"),
        ("no such function", "unknown_function"),
        ("misuse of aggregate", "aggregate_misuse"),
        ("aggregate functions are not allowed", "aggregate_misuse"),
        ("selects to the left and right", "set_operation_shape_mismatch"),
        ("syntax error", "sqlite_syntax_error"),
        ("circular reference", "circular_reference"),
    )
    for needle, label in patterns:
        if needle in message:
            return label
    return "other_database_error"


def primary_outcome(record: dict[str, Any]) -> str:
    if bool(record.get("execution_match")):
        return "correct"
    if record.get("prediction_execution_status") == "unsafe":
        return "non_sql_or_unsafe"
    if not bool(record.get("syntax_valid")):
        return "invalid_sql"
    if record.get("prediction_execution_status") != "ok":
        return "database_execution_error"
    return "executable_wrong_result"


def format_issue(record: dict[str, Any]) -> str:
    if bool(record.get("format_compliant")):
        return "compliant"
    raw = str(record.get("raw_prediction") or "").strip()
    predicted = str(record.get("predicted_sql") or "").strip()
    if "```" in raw:
        return "markdown_fence"
    match = SQL_START.search(raw)
    if match is None:
        return "no_sql_in_response"
    if match.start() > 0:
        return "prose_before_sql"
    if raw != predicted:
        return "trailing_text_or_multiple_statements"
    return "other_format_issue"


def bin_value(value: int, boundaries: Sequence[tuple[int, str]], final_label: str) -> str:
    for upper, label in boundaries:
        if value <= upper:
            return label
    return final_label


def row_bins(data_row: dict[str, Any], prediction: dict[str, Any]) -> dict[str, str]:
    features = query_features(data_row)
    sql_tokens = int(features.get("token_count", len(str(data_row["sql"]).split())))
    schema_tables = len(re.findall(r"(?im)^CREATE TABLE\b", str(data_row.get("schema", ""))))
    question_words = len(str(data_row.get("question", "")).split())
    input_tokens = int(prediction.get("input_tokens", 0))
    return {
        "gold_sql_tokens": bin_value(sql_tokens, ((10, "01-10"), (20, "11-20"), (30, "21-30")), "31+"),
        "schema_tables": bin_value(schema_tables, ((3, "01-03"), (6, "04-06")), "07+"),
        "question_words": bin_value(question_words, ((8, "01-08"), (15, "09-15")), "16+"),
        "input_tokens": bin_value(input_tokens, ((256, "001-256"), (512, "257-512")), "513+"),
    }


def counter_rows(model: str, category: str, counter: Counter[str], total: int) -> list[dict[str, Any]]:
    return [
        {
            "model": model,
            "category": category,
            "value": key,
            "count": count,
            "pct_of_all_examples": percentage(count, total),
        }
        for key, count in counter.most_common()
    ]


def metric_by_subset(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    correct = sum(bool(row.get("execution_match")) for row in rows)
    return {"support": len(rows), "correct": correct, "execution_accuracy_pct": percentage(correct, len(rows))}


def unique_sql_metrics(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[bool]] = defaultdict(list)
    for row in records:
        groups[normalize_sql(str(row["gold_sql"]))].append(bool(row.get("execution_match")))
    group_accuracies = [sum(values) / len(values) for values in groups.values()]
    return {
        "unique_normalized_gold_sql": len(groups),
        "macro_execution_accuracy_pct": round(100.0 * sum(group_accuracies) / len(group_accuracies), 3),
        "all_paraphrases_correct_count": sum(all(values) for values in groups.values()),
        "any_paraphrase_correct_count": sum(any(values) for values in groups.values()),
    }


def load_models(results_dir: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    comparison = read_json(results_dir / "comparison.json")
    predictions: dict[str, list[dict[str, Any]]] = {}
    for item in comparison:
        slug = str(item["slug"])
        path = results_dir / slug / "predictions.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        predictions[slug] = read_jsonl(path)
    return comparison, predictions


def validate_alignment(
    data: list[dict[str, Any]], predictions: dict[str, list[dict[str, Any]]]
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    ids = [str(row["id"]) for row in data]
    if len(ids) != len(set(ids)):
        raise ValueError("Validation data has duplicate IDs")
    data_by_id = {str(row["id"]): row for row in data}
    prediction_maps: dict[str, dict[str, dict[str, Any]]] = {}
    for slug, rows in predictions.items():
        model_ids = [str(row["id"]) for row in rows]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError(f"{slug} has duplicate prediction IDs")
        if set(model_ids) != set(ids):
            missing = len(set(ids) - set(model_ids))
            extra = len(set(model_ids) - set(ids))
            raise ValueError(f"{slug} prediction IDs do not align: missing={missing}, extra={extra}")
        prediction_maps[slug] = {str(row["id"]): row for row in rows}
    return ids, data_by_id, prediction_maps


def analyze_model(
    slug: str,
    ids: Sequence[str],
    data_by_id: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows = [predictions[row_id] for row_id in ids]
    primary = Counter(primary_outcome(row) for row in rows)
    runtime = Counter(
        runtime_error_category(row)
        for row in rows
        if row.get("prediction_execution_status") not in (None, "ok")
    )
    formats = Counter(format_issue(row) for row in rows)
    mismatches: Counter[str] = Counter()
    per_example_mismatches: dict[str, list[str]] = {}
    for row in rows:
        if row.get("execution_match") or not row.get("syntax_valid"):
            continue
        values = structural_mismatches(str(row["gold_sql"]), str(row["predicted_sql"]))
        per_example_mismatches[str(row["id"])] = values
        mismatches.update(values)

    complexity = {}
    for name in ("simple", "moderate", "complex"):
        complexity[name] = metric_by_subset(row for row in rows if row.get("complexity") == name)

    features = {}
    for feature in GOLD_FEATURES:
        subset = [
            predictions[row_id]
            for row_id in ids
            if feature.predicate(query_features(data_by_id[row_id]))
        ]
        features[feature.key] = {"label": feature.label, **metric_by_subset(subset)}

    databases = {}
    for db_id in sorted({str(row["db_id"]) for row in rows}):
        databases[db_id] = metric_by_subset(row for row in rows if row.get("db_id") == db_id)

    bins: dict[str, dict[str, Any]] = defaultdict(dict)
    grouped_bins: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row_id in ids:
        prediction = predictions[row_id]
        for dimension, label in row_bins(data_by_id[row_id], prediction).items():
            grouped_bins[dimension][label].append(prediction)
    for dimension, groups in grouped_bins.items():
        for label, group in sorted(groups.items()):
            bins[dimension][label] = metric_by_subset(group)

    return {
        "slug": slug,
        "examples": len(rows),
        "execution": metric_by_subset(rows),
        "unique_sql": unique_sql_metrics(rows),
        "primary_outcomes": dict(primary.most_common()),
        "runtime_errors": dict(runtime.most_common()),
        "format_issues": dict(formats.most_common()),
        "structural_mismatches": dict(mismatches.most_common()),
        "per_example_mismatches": per_example_mismatches,
        "complexity": complexity,
        "features": features,
        "databases": databases,
        "bins": dict(bins),
    }


def disagreement_analysis(
    ids: Sequence[str], model_order: Sequence[str], predictions: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    patterns: Counter[str] = Counter()
    correct_sets: dict[str, list[str]] = {}
    for row_id in ids:
        correct = [slug for slug in model_order if predictions[slug][row_id].get("execution_match")]
        correct_sets[row_id] = correct
        patterns[" + ".join(correct) if correct else "none"] += 1
    any_correct = sum(bool(values) for values in correct_sets.values())
    all_correct = sum(len(values) == len(model_order) for values in correct_sets.values())
    return {
        "examples": len(ids),
        "any_model_correct_count": any_correct,
        "any_model_correct_pct": percentage(any_correct, len(ids)),
        "all_models_correct_count": all_correct,
        "all_models_correct_pct": percentage(all_correct, len(ids)),
        "all_models_wrong_count": len(ids) - any_correct,
        "all_models_wrong_pct": percentage(len(ids) - any_correct, len(ids)),
        "correct_model_patterns": dict(patterns.most_common()),
        "correct_models_by_id": correct_sets,
    }


def training_priorities(
    strong_analysis: dict[str, Any], train_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    priorities = []
    for feature in GOLD_FEATURES:
        validation = strong_analysis["features"][feature.key]
        support = int(validation["support"])
        accuracy = float(validation["execution_accuracy_pct"])
        failures = support - int(validation["correct"])
        train_count = sum(feature.predicate(query_features(row)) for row in train_rows)
        if accuracy < 50:
            multiplier = 3.0
        elif accuracy < 60:
            multiplier = 2.5
        elif accuracy < 70:
            multiplier = 2.0
        elif accuracy < 80:
            multiplier = 1.5
        else:
            multiplier = 1.0
        impact_score = round(failures * math.log2(support + 1), 2)
        priorities.append(
            {
                "feature": feature.key,
                "label": feature.label,
                "validation_support": support,
                "strong_correct": int(validation["correct"]),
                "strong_failures": failures,
                "strong_execution_accuracy_pct": accuracy,
                "training_examples": train_count,
                "suggested_max_sampling_multiplier": multiplier,
                "impact_score": impact_score,
            }
        )
    return sorted(priorities, key=lambda row: (-float(row["impact_score"]), str(row["feature"])))


def representative_examples(
    ids: Sequence[str],
    strong_slug: str,
    data_by_id: dict[str, dict[str, Any]],
    prediction_maps: dict[str, dict[str, dict[str, Any]]],
    model_analyses: dict[str, dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    strong = prediction_maps[strong_slug]
    candidates = [row_id for row_id in ids if not strong[row_id].get("execution_match")]
    complexity_rank = {"complex": 0, "moderate": 1, "simple": 2}
    candidates.sort(
        key=lambda row_id: (
            complexity_rank.get(str(strong[row_id].get("complexity")), 3),
            -int(query_features(data_by_id[row_id]).get("token_count", 0)),
            row_id,
        )
    )
    chosen: list[str] = []
    seen_primary: Counter[str] = Counter()
    seen_mismatch: Counter[str] = Counter()
    for row_id in candidates:
        record = strong[row_id]
        primary = primary_outcome(record)
        mismatches = model_analyses[strong_slug]["per_example_mismatches"].get(row_id, [])
        first_mismatch = mismatches[0] if mismatches else "none"
        if seen_primary[primary] >= 3 or seen_mismatch[first_mismatch] >= 2:
            continue
        chosen.append(row_id)
        seen_primary[primary] += 1
        seen_mismatch[first_mismatch] += 1
        if len(chosen) >= limit:
            break
    if len(chosen) < limit:
        chosen.extend(row_id for row_id in candidates if row_id not in chosen)
        chosen = chosen[:limit]

    examples = []
    for row_id in chosen:
        record = strong[row_id]
        examples.append(
            {
                "id": row_id,
                "db_id": record["db_id"],
                "complexity": record["complexity"],
                "question": record["question"],
                "gold_sql": record["gold_sql"],
                "predicted_sql": record["predicted_sql"],
                "raw_prediction": record["raw_prediction"],
                "primary_outcome": primary_outcome(record),
                "runtime_error": (
                    runtime_error_category(record)
                    if record.get("prediction_execution_status") not in (None, "ok")
                    else None
                ),
                "structural_mismatches": model_analyses[strong_slug]["per_example_mismatches"].get(row_id, []),
                "other_models_correct": [
                    slug
                    for slug, records in prediction_maps.items()
                    if slug != strong_slug and records[row_id].get("execution_match")
                ],
            }
        )
    return examples


def write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(columns or rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(map(cell, headers)) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return lines


def report_markdown(
    result: dict[str, Any], comparison_by_slug: dict[str, dict[str, Any]]
) -> str:
    order = result["model_order"]
    analyses = result["models"]
    strong_slug = result["strong_model"]
    lines = [
        "# Zero-shot Text-to-SQL Error Analysis",
        "",
        f"Source results: `{result['source_results_dir']}`",
        "",
        "## Executive summary",
        "",
        f"The strongest baseline is **{strong_slug}** at "
        f"**{analyses[strong_slug]['execution']['execution_accuracy_pct']:.2f}% execution accuracy**. "
        "Execution match is the primary metric; textual exact match understates semantically equivalent SQL.",
        "",
    ]
    lines.extend(
        markdown_table(
            ("Tier", "Model", "Execution", "Unique-SQL macro", "Syntax valid", "Format compliant", "Failures"),
            [
                (
                    comparison_by_slug[slug].get("tier", ""),
                    slug,
                    f"{analyses[slug]['execution']['execution_accuracy_pct']:.2f}%",
                    f"{analyses[slug]['unique_sql']['macro_execution_accuracy_pct']:.2f}%",
                    f"{float(comparison_by_slug[slug].get('syntax_valid_pct', 0)):.2f}%",
                    f"{float(comparison_by_slug[slug].get('format_compliant_pct', 0)):.2f}%",
                    analyses[slug]["examples"] - analyses[slug]["execution"]["correct"],
                )
                for slug in order
            ],
        )
    )
    strong = analyses[strong_slug]
    long_sql = strong["bins"]["gold_sql_tokens"]["31+"]
    join_subquery = strong["features"]["join_and_subquery"]
    lines.extend(
        [
            "",
            "Key observations:",
            "",
            f"- The strongest model has {strong['primary_outcomes'].get('executable_wrong_result', 0)} executable-but-wrong "
            f"queries versus {strong['primary_outcomes'].get('database_execution_error', 0)} database errors, so its main "
            "remaining problem is semantic construction rather than output validity.",
            f"- Its accuracy falls to **{long_sql['execution_accuracy_pct']:.2f}%** on gold SQL longer than 30 tokens and "
            f"**{join_subquery['execution_accuracy_pct']:.2f}%** on queries combining joins with subqueries.",
            f"- The nearly unchanged unique-SQL macro score ({strong['unique_sql']['macro_execution_accuracy_pct']:.2f}%) "
            "shows that duplicated/paraphrased Spider queries are not materially inflating the headline score.",
            "- Some failures expose annotation ambiguity: a model can follow the natural-language request more literally than "
            "the provided gold SQL. Those cases should be audited before using aggressive hard-example training.",
            "",
        ]
    )

    lines.extend(["", "## Accuracy by gold-query complexity", ""])
    lines.extend(
        markdown_table(
            ("Complexity", "Support", *order),
            [
                (
                    complexity,
                    analyses[order[0]]["complexity"][complexity]["support"],
                    *(f"{analyses[slug]['complexity'][complexity]['execution_accuracy_pct']:.2f}%" for slug in order),
                )
                for complexity in ("simple", "moderate", "complex")
            ],
        )
    )

    lines.extend(["", "## Accuracy by SQL structure", ""])
    lines.append("These categories overlap; each row conditions on a feature in the gold query.")
    lines.append("")
    lines.extend(
        markdown_table(
            ("Feature", "Support", *order),
            [
                (
                    analyses[order[0]]["features"][feature.key]["label"],
                    analyses[order[0]]["features"][feature.key]["support"],
                    *(
                        f"{analyses[slug]['features'][feature.key]['execution_accuracy_pct']:.2f}%"
                        for slug in order
                    ),
                )
                for feature in GOLD_FEATURES
            ],
        )
    )

    lines.extend(["", "## Failure composition", ""])
    failure_labels = (
        "executable_wrong_result",
        "database_execution_error",
        "invalid_sql",
        "non_sql_or_unsafe",
    )
    lines.extend(
        markdown_table(
            ("Model", *failure_labels),
            [
                (slug, *(analyses[slug]["primary_outcomes"].get(label, 0) for label in failure_labels))
                for slug in order
            ],
        )
    )

    lines.extend(["", "### Output-format behavior", ""])
    format_rows = []
    for slug in order:
        for label, count in analyses[slug]["format_issues"].items():
            format_rows.append((slug, label, count))
    lines.extend(markdown_table(("Model", "Format category", "Count"), format_rows))

    lines.extend(["", "### Most frequent structural mismatches among wrong parseable SQL", ""])
    lines.append("Mismatch labels are heuristic and non-mutually-exclusive; they localize likely error sources rather than prove causality.")
    lines.append("")
    top_mismatches = []
    for slug in order:
        for label, count in list(analyses[slug]["structural_mismatches"].items())[:8]:
            top_mismatches.append((slug, label, count))
    lines.extend(markdown_table(("Model", "Mismatch", "Count"), top_mismatches))

    lines.extend(["", "### Database execution errors", ""])
    runtime_rows = []
    for slug in order:
        for label, count in analyses[slug]["runtime_errors"].items():
            runtime_rows.append((slug, label, count))
    lines.extend(markdown_table(("Model", "Error", "Count"), runtime_rows))

    lines.extend(["", f"## Difficulty slices for {strong_slug}", ""])
    difficulty_rows = []
    dimension_labels = {
        "gold_sql_tokens": "Gold SQL tokens",
        "question_words": "Question words",
        "schema_tables": "Schema tables",
        "input_tokens": "Model input tokens",
    }
    for dimension in ("gold_sql_tokens", "question_words", "schema_tables", "input_tokens"):
        for label, values in strong["bins"][dimension].items():
            difficulty_rows.append(
                (
                    dimension_labels[dimension],
                    label,
                    values["support"],
                    f"{values['execution_accuracy_pct']:.2f}%",
                )
            )
    lines.extend(markdown_table(("Dimension", "Bin", "Support", "Execution"), difficulty_rows))

    lines.extend(["", "## Cross-model disagreement", ""])
    disagreement = result["disagreement"]
    lines.extend(
        [
            f"- At least one model succeeds on **{disagreement['any_model_correct_count']}/{disagreement['examples']} "
            f"({disagreement['any_model_correct_pct']:.2f}%)** examples. This is the three-model oracle ceiling.",
            f"- All models succeed on **{disagreement['all_models_correct_count']} "
            f"({disagreement['all_models_correct_pct']:.2f}%)** examples.",
            f"- All models fail on **{disagreement['all_models_wrong_count']} "
            f"({disagreement['all_models_wrong_pct']:.2f}%)** examples.",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ("Correct model set", "Examples"),
            list(disagreement["correct_model_patterns"].items()),
        )
    )

    lines.extend(["", "## Hardest validation databases", ""])
    for slug in order:
        hardest = sorted(
            (
                item
                for item in analyses[slug]["databases"].items()
                if item[1]["support"] >= 20
            ),
            key=lambda item: (item[1]["execution_accuracy_pct"], -item[1]["support"], item[0]),
        )[:5]
        lines.extend([f"### {slug}", ""])
        lines.extend(
            markdown_table(
                ("Database", "Support", "Execution"),
                [
                    (db_id, values["support"], f"{values['execution_accuracy_pct']:.2f}%")
                    for db_id, values in hardest
                ],
            )
        )
        lines.append("")

    lines.extend(["## Fine-tuning priorities", ""])
    lines.append(
        "Priorities are ranked by the strongest model's observed failure volume and feature support. "
        "Sampling multipliers are caps, not additive weights, because examples belong to multiple categories."
    )
    lines.append("")
    lines.extend(
        markdown_table(
            ("Priority", "Feature", "Val support", "Strong failures", "Strong accuracy", "Train examples", "Max multiplier"),
            [
                (
                    index,
                    row["label"],
                    row["validation_support"],
                    row["strong_failures"],
                    f"{row['strong_execution_accuracy_pct']:.2f}%",
                    row["training_examples"],
                    f"{row['suggested_max_sampling_multiplier']:.1f}x",
                )
                for index, row in enumerate(result["training_priorities"][:10], start=1)
            ],
        )
    )
    lines.extend(
        [
            "",
            "Recommended training contract:",
            "",
            "1. Train only on the 6,997 executable training examples; never mix validation databases into training.",
            "2. Use feature-aware sampling with the maximum applicable multiplier per row, then cap repeated duplicates.",
            "3. Keep the SQL-only assistant target and explicitly penalize prose or Markdown output.",
            "4. Track execution accuracy overall and by feature; do not select checkpoints on text exact match alone.",
            "5. Retain a natural-distribution validation score alongside the hard-feature slices so oversampling does not hide regressions.",
            "",
            f"## Representative {strong_slug} failures",
            "",
        ]
    )
    for example in result["representative_examples"]:
        issue = example["runtime_error"] or ", ".join(example["structural_mismatches"]) or example["primary_outcome"]
        lines.extend(
            [
                f"### {example['id']} — {example['db_id']} / {example['complexity']}",
                "",
                f"Question: {example['question']}",
                "",
                f"Issue: `{issue}`",
                "",
                "Gold:",
                "",
                "```sql",
                str(example["gold_sql"]),
                "```",
                "",
                "Prediction:",
                "",
                "```sql",
                str(example["predicted_sql"]),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Metric caveats",
            "",
            "This analysis uses the run's local read-only SQLite result-equivalence metric. It is stronger than string exact match "
            "for semantically equivalent SQL, but it is not the official Spider test-suite evaluator and can admit accidental "
            "matches on a particular database instance. The retained predictions can be scored with an official evaluator later "
            "without regenerating model outputs.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    results_dir = (args.results_dir or latest_complete_results(DEFAULT_RUNS_ROOT)).resolve()
    data_path = args.data.resolve()
    train_path = args.train_data.resolve()
    output_dir = (args.output_dir or results_dir / "error-analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison, prediction_lists = load_models(results_dir)
    validation = read_jsonl(data_path)
    train = read_jsonl(train_path)
    ids, data_by_id, prediction_maps = validate_alignment(validation, prediction_lists)
    model_order = [str(item["slug"]) for item in comparison]
    comparison_by_slug = {str(item["slug"]): item for item in comparison}
    strong_slug = next(
        (str(item["slug"]) for item in comparison if item.get("tier") == "strong"),
        max(model_order, key=lambda slug: float(comparison_by_slug[slug]["execution_match_pct"])),
    )

    model_analyses = {
        slug: analyze_model(slug, ids, data_by_id, prediction_maps[slug]) for slug in model_order
    }
    disagreement = disagreement_analysis(ids, model_order, prediction_maps)
    priorities = training_priorities(model_analyses[strong_slug], train)
    representatives = representative_examples(
        ids,
        strong_slug,
        data_by_id,
        prediction_maps,
        model_analyses,
        args.representatives,
    )
    result = {
        "source_results_dir": str(results_dir),
        "validation_data": str(data_path),
        "train_data": str(train_path),
        "model_order": model_order,
        "strong_model": strong_slug,
        "models": model_analyses,
        "disagreement": disagreement,
        "training_priorities": priorities,
        "representative_examples": representatives,
    }

    (output_dir / "analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "REPORT.md").write_text(
        report_markdown(result, comparison_by_slug) + "\n", encoding="utf-8"
    )

    feature_rows = []
    database_rows = []
    failure_rows = []
    bin_rows = []
    for slug in model_order:
        analysis = model_analyses[slug]
        for key, values in analysis["features"].items():
            feature_rows.append({"model": slug, "feature": key, **values})
        for db_id, values in analysis["databases"].items():
            database_rows.append({"model": slug, "db_id": db_id, **values})
        failure_rows.extend(counter_rows(slug, "primary_outcome", Counter(analysis["primary_outcomes"]), len(ids)))
        failure_rows.extend(counter_rows(slug, "runtime_error", Counter(analysis["runtime_errors"]), len(ids)))
        failure_rows.extend(counter_rows(slug, "format_issue", Counter(analysis["format_issues"]), len(ids)))
        failure_rows.extend(
            counter_rows(slug, "structural_mismatch", Counter(analysis["structural_mismatches"]), len(ids))
        )
        for dimension, groups in analysis["bins"].items():
            for label, values in groups.items():
                bin_rows.append({"model": slug, "dimension": dimension, "bin": label, **values})

    write_csv(output_dir / "feature_accuracy.csv", feature_rows)
    write_csv(output_dir / "database_accuracy.csv", database_rows)
    write_csv(output_dir / "failure_counts.csv", failure_rows)
    write_csv(output_dir / "difficulty_bins.csv", bin_rows)
    write_csv(output_dir / "fine_tuning_priorities.csv", priorities)

    print(f"Analyzed {len(ids)} examples for {len(model_order)} models")
    print(f"Strong model: {strong_slug}")
    print(f"Report: {output_dir / 'REPORT.md'}")
    print(f"Machine-readable analysis: {output_dir / 'analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
