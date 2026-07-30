from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_sft_dataset", PROJECT_ROOT / "scripts" / "build_sft_dataset.py"
)
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def example(index: int, sql: str, *, join_count: int = 0, has_subquery: bool = False) -> dict:
    return {
        "id": f"spider-train-{index:05d}",
        "dataset": "spider",
        "split": "train",
        "db_id": "tiny",
        "question": f"Question {index}?",
        "sql": sql,
        "messages": [
            {"role": "system", "content": "Return SQL only."},
            {"role": "user", "content": f"Question {index}?"},
            {"role": "assistant", "content": sql},
        ],
        "metadata": {
            "execution_validation": {"status": "ok"},
            "query_features": {
                "aggregate_count": 0,
                "join_count": join_count,
                "set_operation_count": 0,
                "has_where": False,
                "has_group_by": False,
                "has_having": False,
                "has_order_by": False,
                "has_limit": False,
                "has_distinct": False,
                "has_subquery": has_subquery,
                "complexity_proxy": "complex" if has_subquery else "simple",
            },
        },
    }


class SftDatasetBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = {
            "combine_rule": "maximum",
            "seed": 17,
            "supplement_fraction": 0.5,
            "max_supplement_per_normalized_sql": 1,
            "feature_multipliers": {
                "join": 2.0,
                "multi_join": 2.5,
                "subquery": 2.5,
                "join_and_subquery": 3.0,
                "complex": 2.5,
            },
        }

    def test_feature_weights_use_maximum_not_sum(self) -> None:
        row = example(1, "SELECT x FROM a JOIN b ON a.id = b.id WHERE a.id IN (SELECT id FROM c)", join_count=1, has_subquery=True)
        self.assertEqual(builder.priority_multiplier(row, self.policy), 3.0)

    def test_supplement_is_deterministic_unique_and_template_capped(self) -> None:
        rows = [
            example(1, "SELECT a FROM t1 JOIN t2 ON t1.id = t2.id", join_count=1),
            example(2, " select  a  from t1 join t2 on t1.id = t2.id; ", join_count=1),
            example(3, "SELECT b FROM u1 JOIN u2 ON u1.id = u2.id", join_count=1),
            example(4, "SELECT c FROM v1 JOIN v2 ON v1.id = v2.id", join_count=1),
        ]
        first, _audit = builder.select_supplement(rows, self.policy)
        second, _audit = builder.select_supplement(rows, self.policy)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertEqual(len(first), len(set(first)))
        selected_sql = [builder.normalize_sql(rows[int(row_id.rsplit("-", 1)[1]) - 1]["sql"]) for row_id in first]
        self.assertEqual(len(selected_sql), len(set(selected_sql)))

    def test_source_validation_rejects_non_sql_target(self) -> None:
        row = example(1, "SELECT 1")
        row["sql"] = "There is one row."
        row["messages"][-1]["content"] = row["sql"]
        with self.assertRaisesRegex(ValueError, "does not start with SELECT or WITH"):
            builder.validate_source_rows([row], "train")


if __name__ == "__main__":
    unittest.main()
