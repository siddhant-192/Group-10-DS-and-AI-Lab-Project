from __future__ import annotations

import importlib.util
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_text2sql_models", PROJECT_ROOT / "scripts" / "evaluate_text2sql_models.py"
)
assert SPEC and SPEC.loader
evaluator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluator
SPEC.loader.exec_module(evaluator)


class ZeroShotEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database = Path(self.temporary_directory.name) / "tiny.sqlite"
        connection = sqlite3.connect(self.database)
        connection.executescript(
            """
            CREATE TABLE person (id INTEGER PRIMARY KEY, name TEXT, age INTEGER);
            INSERT INTO person VALUES (1, 'Ada', 36), (2, 'Lin', 29);
            """
        )
        connection.commit()
        connection.close()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_extracts_fenced_sql_and_canonicalizes(self) -> None:
        extracted = evaluator.extract_sql("Here is the query:\n```sql\nSELECT name FROM person;\n```")
        canonical, error = evaluator.canonical_sql(extracted)
        self.assertEqual(extracted, "SELECT name FROM person;")
        self.assertIsNone(error)
        self.assertEqual(canonical, "SELECT name FROM person")

    def test_extracts_sql_after_reasoning_trace(self) -> None:
        raw = (
            "<think>We should select every name. Maybe SELECT bad FROM nowhere;</think>\n"
            "SELECT name FROM person;"
        )
        self.assertEqual(evaluator.extract_sql(raw), "SELECT name FROM person;")

    def test_execution_comparison_ignores_order_without_order_by(self) -> None:
        ascending = evaluator.execute_query(
            self.database, "SELECT name FROM person ORDER BY id", 1.0, order_sensitive=False
        )
        descending = evaluator.execute_query(
            self.database, "SELECT name FROM person ORDER BY id DESC", 1.0, order_sensitive=False
        )
        self.assertEqual(ascending.status, "ok")
        self.assertEqual(ascending.rows, descending.rows)

    def test_macsql_result_comparison_allows_global_column_permutation(self) -> None:
        gold = evaluator.execute_query(
            self.database, "SELECT name, age FROM person ORDER BY id", 1.0, order_sensitive=True
        )
        reordered = evaluator.execute_query(
            self.database, "SELECT age, name FROM person ORDER BY id", 1.0, order_sensitive=True
        )
        self.assertNotEqual(gold.rows, reordered.rows)
        self.assertTrue(evaluator.macsql_result_equal(gold, reordered, order_matters=True))

    def test_macsql_result_comparison_preserves_row_order_for_order_by(self) -> None:
        ascending = evaluator.execute_query(
            self.database, "SELECT name FROM person ORDER BY id", 1.0, order_sensitive=True
        )
        descending = evaluator.execute_query(
            self.database, "SELECT name FROM person ORDER BY id DESC", 1.0, order_sensitive=True
        )
        self.assertFalse(evaluator.macsql_result_equal(ascending, descending, order_matters=True))
        self.assertTrue(evaluator.macsql_result_equal(ascending, descending, order_matters=False))

    def test_macsql_result_comparison_matches_python_numeric_equality(self) -> None:
        integer = evaluator.QueryResult("ok", ("[1970]",), 1, 1.0)
        floating = evaluator.QueryResult("ok", ("[1970.0]",), 1, 1.0)
        self.assertTrue(evaluator.macsql_result_equal(integer, floating, order_matters=False))

    def test_strip_distinct_does_not_change_quoted_literals(self) -> None:
        sql = "SELECT DISTINCT name, 'distinct', \"distinct\" FROM person"
        self.assertEqual(
            evaluator.strip_distinct_sql(sql),
            "SELECT  name, 'distinct', \"distinct\" FROM person",
        )

    def test_macsql_postprocess_repairs_spaced_operators(self) -> None:
        self.assertEqual(
            evaluator.macsql_postprocess("SELECT * FROM t WHERE a > = 1 AND b ! = 2"),
            "SELECT * FROM t WHERE a >= 1 AND b != 2",
        )

    def test_execution_blocks_writes(self) -> None:
        result = evaluator.execute_query(
            self.database, "DELETE FROM person", 1.0, order_sensitive=False
        )
        self.assertEqual(result.status, "unsafe")
        connection = sqlite3.connect(self.database)
        count = connection.execute("SELECT count(*) FROM person").fetchone()[0]
        connection.close()
        self.assertEqual(count, 2)

    def test_execution_consensus_prefers_largest_executable_cluster(self) -> None:
        agreeing = evaluator.QueryResult("ok", ('["Ada"]',), 1, 1.0)
        other = evaluator.QueryResult("ok", ('["Lin"]',), 1, 1.0)
        failed = evaluator.QueryResult("error", None, None, 1.0, "no such column")
        index, votes = evaluator.select_execution_consensus([other, agreeing, failed, agreeing])
        self.assertEqual(index, 1)
        self.assertEqual(votes, 2)

    def test_execution_consensus_falls_back_when_every_candidate_fails(self) -> None:
        failed = evaluator.QueryResult("error", None, None, 1.0, "bad SQL")
        index, votes = evaluator.select_execution_consensus([failed, failed])
        self.assertEqual((index, votes), (0, 0))

    def test_value_aware_voting_ignores_column_order(self) -> None:
        reordered = evaluator.QueryResult("ok", ('["Ada", 36]',), 2, 1.0)
        agreeing = evaluator.QueryResult("ok", ('[36, "Ada"]',), 2, 1.0)
        other = evaluator.QueryResult("ok", ('[29, "Lin"]',), 2, 1.0)
        index, votes = evaluator.select_value_aware_voting([other, reordered, agreeing])
        self.assertEqual((index, votes), (1, 2))

    def test_value_aware_voting_skips_empty_and_all_zero_groups(self) -> None:
        empty = evaluator.QueryResult("ok", (), 1, 1.0)
        zero = evaluator.QueryResult("ok", ('[0]',), 1, 1.0)
        useful = evaluator.QueryResult("ok", ('["Ada"]',), 1, 1.0)
        index, votes = evaluator.select_value_aware_voting([empty, empty, zero, zero, useful])
        self.assertEqual((index, votes), (4, 1))

    def test_finer_published_vav_matches_value_groups_and_zero_filter(self) -> None:
        zero = evaluator.QueryResult("ok", ('[0]',), 1, 1.0)
        reordered = evaluator.QueryResult("ok", ('["Ada", 36]',), 2, 1.0)
        agreeing = evaluator.QueryResult("ok", ('[36, "Ada"]',), 2, 1.0)
        index, votes = evaluator.select_finer_published_vav([zero, zero, reordered, agreeing])
        self.assertEqual((index, votes), (2, 2))


if __name__ == "__main__":
    unittest.main()
