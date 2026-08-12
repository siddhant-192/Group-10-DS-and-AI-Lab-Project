#!/usr/bin/env python3
"""
Schema-linking precision: across ALL table and column references the model
made in its predicted SQL (not just the ones that caused a failure), what
fraction actually point to real elements of the target database's schema?

This complements the error taxonomy's hallucinated_column/hallucinated_table
RATE (what % of examples failed due to a bad reference) with a PRECISION
view (of every reference the model made, correct predictions included, how
many were grounded in the real schema). A model could have a low
hallucination-failure rate while still frequently "getting lucky" with
coincidentally-valid-but-wrong references, or vice versa -- this metric is
computed independently of whether the query executed successfully.

Does NOT re-run the model -- reads existing predicted_sql from
predictions.jsonl and checks it against real schemas. Pure parsing +
schema lookup, no GPU needed.

Usage:
    python schema_linking_precision.py \
        --predictions evidence/milestone5/evaluation_mschema/milestone4_frozen/predictions.jsonl \
        --bird-eval evidence/milestone5/bird/bird_eval_mschema.jsonl \
        --output evidence/milestone5/schema_linking_precision.json
"""

import argparse
import json
import sqlite3
from pathlib import Path

import sqlglot
from sqlglot import exp


def get_schema(db_path):
    """Returns {table_name_lower: set(column_names_lower)} and the flat
    set of ALL column names across ALL tables (for column-reference checks,
    since a column reference's correct table isn't always resolvable from
    the query alone -- consistent with how hallucinated_column errors are
    defined elsewhere in this project's error taxonomy)."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        tables = {}
        for (table_name,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ):
            cols = conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
            tables[table_name.lower()] = {c[1].lower() for c in cols}
        all_columns = set()
        for cols in tables.values():
            all_columns |= cols
        return tables, all_columns
    finally:
        conn.close()


def extract_references(sql):
    """Returns (table_names, column_names) referenced in the SQL, or
    (None, None) if the SQL doesn't parse (syntax errors are already
    counted separately in the error taxonomy -- this metric only
    describes queries that were at least syntactically well-formed)."""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None, None

    tables = [t.name.lower() for t in tree.find_all(exp.Table)]
    columns = [c.name.lower() for c in tree.find_all(exp.Column) if c.name]
    return tables, columns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--bird-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    db_path_by_id = {}
    with open(args.bird_eval, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                db_path_by_id[str(row["id"])] = Path(row["metadata"]["database_path"])

    with open(args.predictions, encoding="utf-8") as f:
        preds = [json.loads(line) for line in f if line.strip()]

    schema_cache = {}

    total_table_refs = 0
    real_table_refs = 0
    total_column_refs = 0
    real_column_refs = 0

    n_parseable = 0
    n_fully_grounded = 0  # examples where EVERY table AND column reference was real
    n_skipped_unparseable = 0

    # Split by correctness, to see if grounding differs for correct vs wrong predictions
    by_correctness = {
        True: {"table_total": 0, "table_real": 0, "column_total": 0, "column_real": 0, "n": 0},
        False: {"table_total": 0, "table_real": 0, "column_total": 0, "column_real": 0, "n": 0},
    }

    for row in preds:
        row_id = str(row["id"])
        db_path = db_path_by_id.get(row_id)
        if db_path is None:
            continue

        if db_path not in schema_cache:
            schema_cache[db_path] = get_schema(db_path)
        tables_schema, all_columns_schema = schema_cache[db_path]

        predicted_sql = row.get("predicted_sql") or ""
        tables, columns = extract_references(predicted_sql)

        if tables is None:
            n_skipped_unparseable += 1
            continue

        n_parseable += 1
        is_correct = bool(row.get("execution_match"))

        example_table_real = [t in tables_schema for t in tables]
        example_column_real = [c in all_columns_schema for c in columns]

        total_table_refs += len(tables)
        real_table_refs += sum(example_table_real)
        total_column_refs += len(columns)
        real_column_refs += sum(example_column_real)

        bucket = by_correctness[is_correct]
        bucket["n"] += 1
        bucket["table_total"] += len(tables)
        bucket["table_real"] += sum(example_table_real)
        bucket["column_total"] += len(columns)
        bucket["column_real"] += sum(example_column_real)

        if all(example_table_real) and all(example_column_real):
            n_fully_grounded += 1

    n = len(preds)
    table_precision = real_table_refs / total_table_refs if total_table_refs else float("nan")
    column_precision = real_column_refs / total_column_refs if total_column_refs else float("nan")

    print("=" * 60)
    print("SCHEMA-LINKING PRECISION")
    print("=" * 60)
    print(f"Total predictions:            {n}")
    print(f"Parseable (syntax valid):     {n_parseable}  ({n_parseable/n:.1%})")
    print(f"Skipped (syntax error):       {n_skipped_unparseable}")
    print()
    print(f"Table-reference precision:    {table_precision:.1%}  ({real_table_refs}/{total_table_refs} references)")
    print(f"Column-reference precision:   {column_precision:.1%}  ({real_column_refs}/{total_column_refs} references)")
    print()
    print(f"Examples with ALL references grounded (fully correct schema-linking): "
          f"{n_fully_grounded}/{n_parseable}  ({n_fully_grounded/n_parseable:.1%} of parseable predictions)")

    print()
    print("--- Precision by prediction correctness ---")
    for is_correct, bucket in by_correctness.items():
        label = "Correct predictions" if is_correct else "Incorrect predictions"
        tp = bucket["table_real"] / bucket["table_total"] if bucket["table_total"] else float("nan")
        cp = bucket["column_real"] / bucket["column_total"] if bucket["column_total"] else float("nan")
        print(f"  {label} (n={bucket['n']}): table precision={tp:.1%}  column precision={cp:.1%}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({
                "total_predictions": n,
                "parseable": n_parseable,
                "skipped_unparseable": n_skipped_unparseable,
                "table_reference_precision": table_precision,
                "table_references_total": total_table_refs,
                "table_references_real": real_table_refs,
                "column_reference_precision": column_precision,
                "column_references_total": total_column_refs,
                "column_references_real": real_column_refs,
                "fully_grounded_examples": n_fully_grounded,
                "fully_grounded_pct_of_parseable": n_fully_grounded / n_parseable if n_parseable else None,
                "by_correctness": {
                    ("correct" if k else "incorrect"): {
                        "n": v["n"],
                        "table_precision": v["table_real"] / v["table_total"] if v["table_total"] else None,
                        "column_precision": v["column_real"] / v["column_total"] if v["column_total"] else None,
                    }
                    for k, v in by_correctness.items()
                },
            }, f, indent=2)
        print(f"\nFull report written to: {args.output}")


if __name__ == "__main__":
    main()
