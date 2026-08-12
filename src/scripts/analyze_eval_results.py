#!/usr/bin/env python3
"""
Post-hoc analysis for Milestone 5 eval results.
Reads predictions.jsonl produced by evaluate_text2sql_models.py and reports
execution accuracy both overall and restricted to examples where gold SQL
executed successfully -- so a data problem never masquerades as a model
problem in your reported numbers.

Usage:
    python analyze_eval_results.py --predictions evidence/milestone5/evaluation_test_v3/milestone4_frozen/predictions.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()

    with open(args.predictions, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    n = len(rows)
    print(f"Total examples: {n}")

    gold_status = Counter(r["gold_execution_status"] for r in rows)
    pred_status = Counter(r["prediction_execution_status"] for r in rows)
    print(f"gold_execution_status: {dict(gold_status)}")
    print(f"prediction_execution_status: {dict(pred_status)}")

    valid = [r for r in rows if r["gold_execution_status"] == "ok"]
    invalid = [r for r in rows if r["gold_execution_status"] != "ok"]

    acc_all = sum(r["execution_match"] for r in rows) / n if n else 0
    acc_valid = sum(r["execution_match"] for r in valid) / len(valid) if valid else 0

    print()
    print("===================================")
    print("Execution accuracy")
    print("===================================")
    print(f"Over ALL {n} examples:                {acc_all:.1%}")
    print(f"Over {len(valid)} examples w/ valid gold:  {acc_valid:.1%}")
    print(f"Excluded (gold failed to execute):    {len(invalid)}")

    if invalid:
        print("\nFirst few gold-execution failures:")
        for r in invalid[:5]:
            print(f"  id={r['id']} db={r['db_id']}")

    # Complexity breakdown, if present
    if any(r.get("complexity") and r["complexity"] != "unknown" for r in valid):
        print()
        print("===================================")
        print("Accuracy by complexity (valid-gold subset)")
        print("===================================")
        groups = {}
        for r in valid:
            groups.setdefault(r.get("complexity", "unknown"), []).append(r)
        for name, group_rows in sorted(groups.items()):
            acc = sum(r["execution_match"] for r in group_rows) / len(group_rows)
            print(f"  {name}: {acc:.1%}  (n={len(group_rows)})")


if __name__ == "__main__":
    main()
