#!/usr/bin/env python3
"""
Post-hoc repair for hallucinated-column errors, run against EXISTING
predictions.jsonl -- does NOT re-run the model, no GPU needed.

For every prediction that failed with "no such column: X", this script:
  1. Parses the predicted SQL to find which table alias X was queried on.
  2. Looks up the real database schema to find which table(s) actually
     contain a column named X.
  3. If exactly one other table (already present in the query's FROM/JOIN)
     has that column, rewrites the column's table qualifier to point there
     and re-executes the repaired query.
  4. Checks whether the repaired query now executes successfully, and
     separately whether its result now matches gold (a stricter,
     more useful signal than "it runs without erroring").

This never touches predictions.jsonl or your reported headline accuracy --
it's a standalone, read-only analysis answering "how many hallucinated-
column errors are mechanically recoverable without retraining."

Usage:
    python repair_hallucinated_columns.py \
        --predictions evidence/milestone5/evaluation_mschema/milestone4_frozen/predictions.jsonl \
        --bird-eval evidence/milestone5/bird/bird_eval_mschema.jsonl \
        --output evidence/milestone5/hallucinated_column_repair_report.json
"""

import argparse
import json
import re
import sqlite3
from pathlib import Path

import sqlglot
from sqlglot import exp


NO_SUCH_COLUMN = re.compile(r"no such column:\s*([\w.\"]+)", re.IGNORECASE)


def get_table_aliases(sql):
    """Returns {alias: real_table_name} for every table referenced in the query."""
    aliases = {}
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return aliases
    for table in tree.find_all(exp.Table):
        real_name = table.name
        alias = table.alias or real_name
        aliases[alias] = real_name
    return aliases


def get_schema_columns(db_path):
    """Returns {table_name: set(column_names)} for every table in the database."""
    schema = {}
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        tables = [
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        for table in tables:
            cols = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
            schema[table] = {c[1].lower() for c in cols}
    finally:
        conn.close()
    return schema


def find_owning_tables(column_name, schema, exclude_table=None):
    """Which real tables actually contain this column (case-insensitive)?"""
    owners = []
    for table, cols in schema.items():
        if table == exclude_table:
            continue
        if column_name.lower() in cols:
            owners.append(table)
    return owners


def attempt_repair(sql, error_message, aliases, schema):
    """
    Returns (repaired_sql, reason) if a confident single-candidate repair
    was found, else (None, reason_for_skipping).
    """
    match = NO_SUCH_COLUMN.search(error_message)
    if not match:
        return None, "error message did not match expected pattern"

    raw_ref = match.group(1).strip('"')
    if "." in raw_ref:
        wrong_alias, column_name = raw_ref.split(".", 1)
    else:
        wrong_alias, column_name = None, raw_ref

    wrong_table = aliases.get(wrong_alias) if wrong_alias else None

    owners = find_owning_tables(column_name, schema, exclude_table=wrong_table)

    # Only repair when exactly one OTHER table (already present as an alias
    # in this query) owns the column -- anything more ambiguous is left
    # alone rather than guessing.
    candidate_aliases = [a for a, t in aliases.items() if t in owners]

    if len(owners) != 1 or len(candidate_aliases) != 1:
        return None, f"ambiguous or no owner found (owners={owners}, candidate_aliases={candidate_aliases})"

    correct_alias = candidate_aliases[0]

    if wrong_alias:
        # Replace "wrong_alias.column" with "correct_alias.column"
        # (word-boundary safe, case-preserving on the column name).
        pattern = re.compile(
            rf"\b{re.escape(wrong_alias)}\.{re.escape(column_name)}\b", re.IGNORECASE
        )
        repaired = pattern.sub(f"{correct_alias}.{column_name}", sql)
    else:
        # Bare column reference (no alias at all) -- qualify it with the
        # correct alias. Only safe when the bare column name is unambiguous
        # in the query text (word-boundary match).
        pattern = re.compile(rf"\b{re.escape(column_name)}\b", re.IGNORECASE)
        repaired = pattern.sub(f"{correct_alias}.{column_name}", sql, count=1)

    if repaired == sql:
        return None, "substitution made no change (pattern not found in SQL text)"

    return repaired, f"rewrote {raw_ref} -> {correct_alias}.{column_name}"


def execute_readonly(db_path, sql, timeout=10):
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=timeout)
        try:
            conn.execute("PRAGMA query_only = ON")
            rows = conn.execute(sql).fetchall()
            return {"status": "ok", "rows": rows, "error": None}
        finally:
            conn.close()
    except Exception as e:
        return {"status": "error", "rows": None, "error": f"{type(e).__name__}: {e}"}


def rows_match(rows_a, rows_b):
    if rows_a is None or rows_b is None:
        return False

    def sort_key(row):
        # (is_none, str_value) per cell avoids comparing None to str/int
        # directly, which raises TypeError under Python 3's ordering rules.
        return tuple((value is None, str(value)) for value in row)

    a = sorted((tuple(row) for row in rows_a), key=sort_key)
    b = sorted((tuple(row) for row in rows_b), key=sort_key)
    return a == b


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

    hallucinated_column_rows = [
        r for r in preds
        if not r.get("execution_match")
        and r.get("prediction_execution_error")
        and NO_SUCH_COLUMN.search(r["prediction_execution_error"])
    ]

    print(f"Found {len(hallucinated_column_rows)} hallucinated-column errors to attempt repair on.\n")

    schema_cache = {}
    results = []
    attempted = 0
    repaired_and_executes = 0
    repaired_and_matches_gold = 0

    for row in hallucinated_column_rows:
        row_id = str(row["id"])
        db_path = db_path_by_id.get(row_id)
        if db_path is None:
            continue

        if db_path not in schema_cache:
            schema_cache[db_path] = get_schema_columns(db_path)
        schema = schema_cache[db_path]

        predicted_sql = row["predicted_sql"]
        aliases = get_table_aliases(predicted_sql)

        repaired_sql, reason = attempt_repair(
            predicted_sql, row["prediction_execution_error"], aliases, schema
        )

        record = {
            "id": row["id"], "db_id": row["db_id"],
            "original_sql": predicted_sql, "error": row["prediction_execution_error"],
            "repair_reason": reason, "repaired_sql": None,
            "repaired_executes": False, "repaired_matches_gold": False,
        }

        if repaired_sql is not None:
            attempted += 1
            record["repaired_sql"] = repaired_sql

            repaired_result = execute_readonly(db_path, repaired_sql)
            if repaired_result["status"] == "ok":
                repaired_and_executes += 1
                record["repaired_executes"] = True

                gold_result = execute_readonly(db_path, row["gold_sql"])
                if gold_result["status"] == "ok" and rows_match(gold_result["rows"], repaired_result["rows"]):
                    repaired_and_matches_gold += 1
                    record["repaired_matches_gold"] = True

        results.append(record)

    n = len(hallucinated_column_rows)
    print("=" * 60)
    print("REPAIR RESULTS")
    print("=" * 60)
    print(f"Total hallucinated-column errors:        {n}")
    print(f"Confident single-candidate repair found:  {attempted}  ({attempted/n:.1%})" if n else "n=0")
    print(f"Repaired query executes successfully:     {repaired_and_executes}  ({repaired_and_executes/n:.1%})" if n else "")
    print(f"Repaired query MATCHES GOLD:               {repaired_and_matches_gold}  ({repaired_and_matches_gold/n:.1%})" if n else "")

    if repaired_and_matches_gold:
        print(f"\nIf applied, this would raise correct count from X to X+{repaired_and_matches_gold} "
              f"(a same-model, post-hoc repair -- not a change to the reported single-shot number).")

    print("\nSample repairs:")
    shown = 0
    for r in results:
        if r["repaired_sql"] and shown < 5:
            print(f"  id={r['id']}  {r['repair_reason']}  "
                  f"executes={r['repaired_executes']}  matches_gold={r['repaired_matches_gold']}")
            shown += 1

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({
                "total_hallucinated_column_errors": n,
                "repair_attempted": attempted,
                "repaired_and_executes": repaired_and_executes,
                "repaired_and_matches_gold": repaired_and_matches_gold,
                "details": results,
            }, f, indent=2, ensure_ascii=False)
        print(f"\nFull report written to: {args.output}")


if __name__ == "__main__":
    main()