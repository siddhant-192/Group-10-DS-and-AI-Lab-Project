from __future__ import annotations

from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from text2sql_data.schema import inspect_database  # noqa: E402
from text2sql_data.validation import (  # noqa: E402
    query_features,
    validate_readonly_query,
)


class DataPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        db_dir = Path(self.temporary_directory.name) / "tiny"
        db_dir.mkdir()
        self.database = db_dir / "tiny.sqlite"
        connection = sqlite3.connect(self.database)
        connection.executescript(
            """
            CREATE TABLE parent (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            );
            CREATE TABLE child (
                id INTEGER PRIMARY KEY,
                parent_id INTEGER,
                FOREIGN KEY (parent_id) REFERENCES parent(id)
            );
            INSERT INTO parent VALUES (1, 'Ada');
            INSERT INTO child VALUES (1, 1);
            """
        )
        connection.commit()
        connection.close()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_schema_introspection_and_ddl(self) -> None:
        schema = inspect_database(self.database)
        self.assertEqual(schema.db_id, "tiny")
        self.assertEqual(schema.table_count, 2)
        self.assertEqual(schema.column_count, 4)
        self.assertEqual(schema.foreign_key_count, 1)
        self.assertIn('CREATE TABLE "parent"', schema.to_ddl())
        self.assertIn('FOREIGN KEY ("parent_id")', schema.to_ddl())

    def test_read_only_query_validation(self) -> None:
        valid = validate_readonly_query(
            self.database,
            "SELECT parent.name FROM parent JOIN child ON parent.id = child.parent_id",
        )
        unsafe = validate_readonly_query(
            self.database,
            "DELETE FROM parent",
        )
        self.assertEqual(valid.status, "ok")
        self.assertEqual(unsafe.status, "unsafe")

        connection = sqlite3.connect(self.database)
        count = connection.execute("SELECT count(*) FROM parent").fetchone()[0]
        connection.close()
        self.assertEqual(count, 1)

    def test_query_features(self) -> None:
        features = query_features(
            "SELECT p.name, count(*) FROM parent p JOIN child c ON p.id = c.parent_id "
            "GROUP BY p.name HAVING count(*) > 1"
        )
        self.assertEqual(features["join_count"], 1)
        self.assertTrue(features["has_group_by"])
        self.assertTrue(features["has_having"])
        self.assertGreaterEqual(features["aggregate_count"], 2)


if __name__ == "__main__":
    unittest.main()
