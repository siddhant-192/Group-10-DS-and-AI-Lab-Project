"""Structural schema grounding for generated SQL (rule-based, no extra LLM call).

The model sometimes reads a REAL column from the WRONG table -- in Chinook,
Artist.Title or Album.GenreId, where Title lives on Album/Employee and
GenreId lives on Genre/Track. SQLite reports only the first such reference
("no such column: T2.Title"), which does not tell the user which table was
wrong or where the column actually lives.

This module resolves table aliases on the PARSED statement (via sqlglot)
and checks every qualified column against the real SQLite catalog, so the
UI can say "reads 'Title' from 'Artist', but that column exists in Album,
Employee" instead of surfacing the raw driver error. It is a message-quality
layer in front of validation, not a replacement for it: src/validation.py
and the SQLite authorizer remain the execution-time backstop.

Unparseable SQL, CTE references, and derived-table aliases are deliberately
left alone -- this check only reports a problem it can prove from the
catalog, so it never blocks a query that would have run.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sqlite3

import sqlglot
from sqlglot import exp


@lru_cache(maxsize=16)
def _catalog(db_path: str, mtime: float) -> tuple[dict[str, frozenset[str]], dict[str, str]]:
    """Return (lowercased table -> column set, lowercased table -> real name).

    Keyed on mtime so a replaced database file invalidates the cache.
    """

    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        names = [
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') "
                "AND name NOT LIKE 'sqlite_%'"
            )
        ]
        columns = {
            name.lower(): frozenset(
                str(row[1]).lower()
                for row in connection.execute(f'PRAGMA table_info("{name}")')
            )
            for name in names
        }
        return columns, {name.lower(): name for name in names}
    finally:
        connection.close()


def schema_grounding_error(db_path: Path, sql: str) -> str | None:
    """Return a user-facing message for the first ungrounded table or column.

    Returns None when nothing can be proven wrong, including when the SQL
    does not parse -- callers should continue to normal validation in that
    case rather than treating None as "verified correct".
    """

    if not (sql or "").strip():
        return None

    try:
        schema, real_names = _catalog(str(db_path.resolve()), db_path.stat().st_mtime)
    except Exception:
        return None
    if not schema:
        return None

    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None

    # CTE names look like tables to sqlglot but are not in the catalog.
    cte_names = {
        (cte.alias_or_name or "").lower()
        for cte in tree.find_all(exp.CTE)
        if cte.alias_or_name
    }

    aliases: dict[str, str] = {}
    used_tables: list[str] = []
    for table in tree.find_all(exp.Table):
        name = (table.name or "").lower()
        if not name or name in cte_names:
            continue
        if name not in schema:
            return (
                f"The generated SQL queries table {table.name!r}, which is not in "
                "this database. Available tables: "
                f"{', '.join(sorted(real_names.values()))}."
            )
        used_tables.append(name)
        aliases[name] = name
        if table.alias:
            aliases[table.alias.lower()] = name

    # Column names introduced by the query itself (SELECT ... AS x, later
    # referenced in GROUP BY / ORDER BY) are not catalog columns.
    local_names = cte_names | {
        (alias.alias or "").lower()
        for alias in tree.find_all(exp.Alias)
        if alias.alias
    }
    # Derived tables expose their own output columns, so unqualified names
    # cannot be resolved against the catalog alone.
    has_derived_table = any(tree.find_all(exp.Subquery))

    for column in tree.find_all(exp.Column):
        column_name = (column.name or "").lower()
        if not column_name or column_name == "*":
            continue

        qualifier = (column.table or "").lower()
        if qualifier:
            table = aliases.get(qualifier)
            if table is None or column_name in schema[table]:
                continue
            elsewhere = sorted(
                real_names[other] for other in schema if column_name in schema[other]
            )
            hint = (
                f" That column exists in: {', '.join(elsewhere)}."
                if elsewhere
                else " No table in this database has that column."
            )
            return (
                f"The generated SQL reads {column.name!r} from "
                f"{real_names[table]!r}, but {real_names[table]!r} has no such "
                f"column.{hint}"
            )

        if (
            used_tables
            and not has_derived_table
            and column_name not in local_names
            and not any(column_name in schema[table] for table in used_tables)
        ):
            return (
                f"The generated SQL reads {column.name!r}, which does not exist in "
                "any table it queries "
                f"({', '.join(real_names[table] for table in used_tables)})."
            )

    return None
