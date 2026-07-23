from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_zero_shot_errors", PROJECT_ROOT / "scripts" / "analyze_zero_shot_errors.py"
)
assert SPEC and SPEC.loader
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


class ErrorAnalysisTests(unittest.TestCase):
    def test_structural_mismatches_detect_schema_and_query_shape_changes(self) -> None:
        gold = (
            "SELECT p.name FROM person AS p JOIN pet AS x ON p.id = x.owner_id "
            "WHERE x.kind = 'dog'"
        )
        predicted = "SELECT name FROM person WHERE age = 5"
        mismatches = analysis.structural_mismatches(gold, predicted)
        self.assertIn("table_selection_mismatch", mismatches)
        self.assertIn("column_reference_mismatch", mismatches)
        self.assertIn("join_structure_mismatch", mismatches)
        self.assertIn("literal_or_value_mismatch", mismatches)

    def test_primary_outcome_separates_semantic_and_execution_failures(self) -> None:
        semantic = {
            "execution_match": False,
            "syntax_valid": True,
            "prediction_execution_status": "ok",
        }
        database_error = {
            "execution_match": False,
            "syntax_valid": True,
            "prediction_execution_status": "error",
        }
        self.assertEqual(analysis.primary_outcome(semantic), "executable_wrong_result")
        self.assertEqual(analysis.primary_outcome(database_error), "database_execution_error")

    def test_format_issue_detects_markdown_and_non_sql_output(self) -> None:
        fenced = {
            "format_compliant": False,
            "raw_prediction": "```sql\nSELECT 1;\n```",
            "predicted_sql": "SELECT 1;",
        }
        prose = {
            "format_compliant": False,
            "raw_prediction": "There is one matching row.",
            "predicted_sql": "There is one matching row.",
        }
        self.assertEqual(analysis.format_issue(fenced), "markdown_fence")
        self.assertEqual(analysis.format_issue(prose), "no_sql_in_response")

    def test_unique_sql_macro_avoids_paraphrase_weighting(self) -> None:
        rows = [
            {"gold_sql": "SELECT 1", "execution_match": True},
            {"gold_sql": " select  1; ", "execution_match": False},
            {"gold_sql": "SELECT 2", "execution_match": True},
        ]
        metrics = analysis.unique_sql_metrics(rows)
        self.assertEqual(metrics["unique_normalized_gold_sql"], 2)
        self.assertEqual(metrics["macro_execution_accuracy_pct"], 75.0)


if __name__ == "__main__":
    unittest.main()
