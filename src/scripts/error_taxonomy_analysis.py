#!/usr/bin/env python3
"""
Milestone 5 error-taxonomy and feature-conditioned analysis.

Reads predictions.jsonl produced by evaluate_text2sql_models.py and produces:
  1. Error taxonomy: syntax errors, hallucinated columns, hallucinated tables,
     ambiguous columns, other execution errors, semantic mismatches (executes
     fine but wrong answer), and correct predictions.
  2. Feature-conditioned accuracy: JOIN, GROUP BY, subqueries, multiple joins,
     JOIN+subquery combined, long SQL (>31 tokens) -- based on gold SQL
     structure, since that reflects the true difficulty of each question.
  3. Representative failure examples per category, for citing directly in
     the report's error analysis section.
  4. Most frequently hallucinated column/table names, useful evidence for
     discussing WHY errors occur (e.g. name collisions across tables).

Usage:
    python error_taxonomy_analysis.py \
        --predictions evidence/milestone5/evaluation_test_v5/milestone4_frozen/predictions.jsonl \
        --examples-per-category 3
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


NO_SUCH_COLUMN = re.compile(r"no such column:\s*([\w.\"]+)", re.IGNORECASE)
NO_SUCH_TABLE = re.compile(r"no such table:\s*([\w.\"]+)", re.IGNORECASE)
AMBIGUOUS_COLUMN = re.compile(r"ambiguous column name:\s*([\w.\"]+)", re.IGNORECASE)
JOIN_RE = re.compile(r"\bJOIN\b", re.IGNORECASE)
GROUP_BY_RE = re.compile(r"\bGROUP\s+BY\b", re.IGNORECASE)
SUBQUERY_RE = re.compile(r"\(\s*SELECT\b", re.IGNORECASE)


def categorize(row):
    """
    Returns one of:
      "correct"              -- execution_match True
      "syntax_error"          -- predicted SQL doesn't parse
      "hallucinated_column"   -- references a column that doesn't exist
      "hallucinated_table"    -- references a table that doesn't exist
      "ambiguous_column"      -- column exists but is ambiguous across joined tables
      "other_execution_error" -- executes-fails for some other reason (timeout, etc.)
      "semantic_mismatch"     -- executes fine, but result doesn't match gold
    """
    if row.get("execution_match"):
        return "correct"

    if not row.get("syntax_valid", True):
        return "syntax_error"

    error = row.get("prediction_execution_error") or ""

    if NO_SUCH_COLUMN.search(error):
        return "hallucinated_column"
    if NO_SUCH_TABLE.search(error):
        return "hallucinated_table"
    if AMBIGUOUS_COLUMN.search(error):
        return "ambiguous_column"
    if row.get("prediction_execution_status") == "ok":
        return "semantic_mismatch"

    return "other_execution_error"


def sql_features(gold_sql):
    """Structural features of the GOLD query, used to bucket difficulty."""
    join_count = len(JOIN_RE.findall(gold_sql))
    has_group_by = bool(GROUP_BY_RE.search(gold_sql))
    has_subquery = bool(SUBQUERY_RE.search(gold_sql))
    token_count = len(gold_sql.split())

    return {
        "has_join": join_count >= 1,
        "multiple_joins": join_count >= 2,
        "has_group_by": has_group_by,
        "has_subquery": has_subquery,
        "join_and_subquery": join_count >= 1 and has_subquery,
        "long_sql": token_count > 31,
        "join_count": join_count,
        "token_count": token_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--examples-per-category", type=int, default=3)
    parser.add_argument("--output-json", type=Path, default=None,
                         help="Optional: write the full analysis as JSON for the report appendix.")
    args = parser.parse_args()

    with open(args.predictions, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    n = len(rows)
    print(f"Total predictions analyzed: {n}\n")

    # ---------------------------------------------------------------
    # 1. Error taxonomy
    # ---------------------------------------------------------------
    for row in rows:
        row["_category"] = categorize(row)

    category_counts = Counter(row["_category"] for row in rows)

    print("=" * 60)
    print("ERROR TAXONOMY")
    print("=" * 60)
    for cat, count in category_counts.most_common():
        pct = 100.0 * count / n
        print(f"  {cat:24s} {count:4d}  ({pct:5.1f}%)")

    # ---------------------------------------------------------------
    # 2. Hallucinated column/table frequency -- evidence for "why"
    # ---------------------------------------------------------------
    hallucinated_columns = Counter()
    hallucinated_tables = Counter()

    for row in rows:
        error = row.get("prediction_execution_error") or ""
        col_match = NO_SUCH_COLUMN.search(error)
        if col_match:
            hallucinated_columns[col_match.group(1)] += 1
        table_match = NO_SUCH_TABLE.search(error)
        if table_match:
            hallucinated_tables[table_match.group(1)] += 1

    print()
    print("=" * 60)
    print("TOP HALLUCINATED COLUMNS (model referenced, doesn't exist)")
    print("=" * 60)
    for name, count in hallucinated_columns.most_common(10):
        print(f"  {name:30s} {count:3d} occurrences")

    print()
    print("=" * 60)
    print("TOP HALLUCINATED TABLES")
    print("=" * 60)
    for name, count in hallucinated_tables.most_common(10):
        print(f"  {name:30s} {count:3d} occurrences")

    # ---------------------------------------------------------------
    # 3. Feature-conditioned accuracy (based on GOLD sql structure)
    # ---------------------------------------------------------------
    for row in rows:
        row["_features"] = sql_features(row.get("gold_sql", ""))

    feature_flags = [
        "has_join",
        "multiple_joins",
        "has_group_by",
        "has_subquery",
        "join_and_subquery",
        "long_sql",
    ]

    print()
    print("=" * 60)
    print("FEATURE-CONDITIONED ACCURACY (gold SQL structure)")
    print("=" * 60)
    print(f"  {'feature':20s} {'n':>5s}  {'accuracy':>9s}   vs. rest")
    for flag in feature_flags:
        with_feature = [r for r in rows if r["_features"][flag]]
        without_feature = [r for r in rows if not r["_features"][flag]]

        acc_with = (
            sum(r["execution_match"] for r in with_feature) / len(with_feature)
            if with_feature else float("nan")
        )
        acc_without = (
            sum(r["execution_match"] for r in without_feature) / len(without_feature)
            if without_feature else float("nan")
        )

        print(
            f"  {flag:20s} {len(with_feature):5d}  {acc_with:8.1%}   "
            f"(without: {acc_without:.1%}, n={len(without_feature)})"
        )

    # Join-count granularity, since "multiple joins" hides a gradient
    print()
    print("  --- accuracy by exact join count (gold SQL) ---")
    by_join_count = defaultdict(list)
    for row in rows:
        by_join_count[row["_features"]["join_count"]].append(row)
    for count in sorted(by_join_count):
        group = by_join_count[count]
        acc = sum(r["execution_match"] for r in group) / len(group)
        print(f"  {count} JOIN(s): n={len(group):4d}  accuracy={acc:.1%}")

    # ---------------------------------------------------------------
    # 4. Representative examples per error category
    # ---------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"REPRESENTATIVE EXAMPLES (up to {args.examples_per_category} per category)")
    print("=" * 60)

    by_category = defaultdict(list)
    for row in rows:
        by_category[row["_category"]].append(row)

    examples_output = {}

    for cat in ("syntax_error", "hallucinated_column", "hallucinated_table",
                "ambiguous_column", "semantic_mismatch", "other_execution_error"):
        group = by_category.get(cat, [])
        if not group:
            continue
        print(f"\n--- {cat} ({len(group)} total) ---")
        examples_output[cat] = []
        for row in group[: args.examples_per_category]:
            print(f"  id={row['id']}  db={row['db_id']}")
            print(f"    question: {row['question'][:120]}")
            print(f"    gold_sql: {row['gold_sql'][:120]}")
            print(f"    pred_sql: {row['predicted_sql'][:120]}")
            if row.get("prediction_execution_error"):
                print(f"    error:    {row['prediction_execution_error'][:120]}")
            examples_output[cat].append({
                "id": row["id"],
                "db_id": row["db_id"],
                "question": row["question"],
                "gold_sql": row["gold_sql"],
                "predicted_sql": row["predicted_sql"],
                "error": row.get("prediction_execution_error"),
            })

    # ---------------------------------------------------------------
    # 5. Optional JSON dump for report appendix
    # ---------------------------------------------------------------
    if args.output_json:
        summary = {
            "total_examples": n,
            "error_taxonomy_counts": dict(category_counts),
            "error_taxonomy_pct": {
                k: round(100.0 * v / n, 2) for k, v in category_counts.items()
            },
            "top_hallucinated_columns": hallucinated_columns.most_common(20),
            "top_hallucinated_tables": hallucinated_tables.most_common(20),
            "feature_conditioned_accuracy": {
                flag: {
                    "n_with_feature": len([r for r in rows if r["_features"][flag]]),
                    "accuracy_with_feature": (
                        sum(r["execution_match"] for r in rows if r["_features"][flag])
                        / max(1, len([r for r in rows if r["_features"][flag]]))
                    ),
                }
                for flag in feature_flags
            },
            "accuracy_by_join_count": {
                str(count): sum(r["execution_match"] for r in group) / len(group)
                for count, group in by_join_count.items()
            },
            "representative_examples": examples_output,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nFull analysis written to: {args.output_json}")


if __name__ == "__main__":
    main()