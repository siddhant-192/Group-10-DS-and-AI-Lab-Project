"""Deterministic dialect-leakage repair (rule-based, no extra LLM call).

The base model's broader pretraining occasionally leaks non-SQLite SQL
functions into otherwise-correct generated queries -- most commonly
YEAR(x) / MONTH(x) / DAY(x), which are valid in MySQL/SQL Server but do
not exist in SQLite (this is documented in the Milestone 5 report,
Section 10.1, as "cross-dialect syntax leakage").

This module rewrites the PARSED AST (via sqlglot), not the raw text, so
the fix is precise and does not risk corrupting unrelated parts of the
query. It only ever touches the specific known-bad node types below;
anything else is left completely unchanged.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

# node type -> SQLite strftime format code
_DATE_PART_FORMATS = {
    exp.Year: "%Y",
    exp.Month: "%m",
    exp.Day: "%d",
}


def repair_dialect_leakage(sql: str) -> tuple[str, list[str]]:
    """Rewrite known non-SQLite dialect functions to SQLite equivalents.

    Returns (repaired_sql, changes). If nothing needed fixing (including
    if the SQL fails to parse), returns the ORIGINAL sql unchanged and an
    empty changes list -- callers should treat an empty changes list as
    "no repair was made", not as an error.
    """

    text = (sql or "").strip()
    if not text:
        return sql, []

    try:
        tree = sqlglot.parse_one(text, read="sqlite")
    except Exception:
        return sql, []

    changes: list[str] = []

    for node_type, fmt in _DATE_PART_FORMATS.items():
        for node in list(tree.find_all(node_type)):
            inner = node.this
            if inner is None:
                continue
            replacement = exp.Anonymous(
                this="STRFTIME", expressions=[exp.Literal.string(fmt), inner]
            )
            node.replace(replacement)
            changes.append(f"{node_type.__name__.upper()}(...) -> STRFTIME('{fmt}', ...)")

    if not changes:
        return sql, []

    try:
        repaired = tree.sql(dialect="sqlite")
    except Exception:
        # If re-serialization somehow fails, fall back to the original
        # rather than risk emitting broken SQL.
        return sql, []

    return repaired, changes
