"""Structural read-only check (rule-based, no extra LLM call, no DB access).

A naive "does the SQL text start with SELECT or WITH" check is not
sufficient: SQLite allows a WITH (CTE) clause to precede DELETE, UPDATE,
or INSERT, not just SELECT. This module checks the PARSED statement's
actual resolved type via sqlglot, so a construction like:

    WITH x AS (SELECT ...) DELETE FROM customer WHERE id IN (SELECT id FROM x)

is correctly identified as a DELETE, not waved through because the text
happens to start with "WITH". This is an additional, fast, code-level
safety layer -- complementary to (not a replacement for) the SQLite
authorizer callback in sql_utils.py and any checks in src/validation.py,
which remain the final, execution-time backstop.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

_WRITE_NODE_TYPES = (
    exp.Delete,
    exp.Update,
    exp.Insert,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.TruncateTable,
)

_READONLY_NODE_TYPES = (exp.Select, exp.Union, exp.Except, exp.Intersect)


def check_actually_readonly(sql: str) -> tuple[bool, str | None]:
    """Returns (is_readonly, reason_if_not).

    Checks the PARSED root node type, not the leading keyword or text
    prefix, so a WITH clause preceding a write statement is correctly
    caught. Unparseable SQL is treated as not read-only (fail closed).
    """

    text = (sql or "").strip()
    if not text:
        return False, "SQL is empty"

    try:
        tree = sqlglot.parse_one(text, read="sqlite")
    except Exception as exc:
        return False, f"SQL did not parse: {exc}"

    if isinstance(tree, _WRITE_NODE_TYPES):
        return False, (
            f"This query resolves to a {type(tree).__name__.upper()} operation, "
            "not a read-only query, even though it may start with WITH. "
            "This system only allows read-only (SELECT) queries."
        )

    if not isinstance(tree, _READONLY_NODE_TYPES):
        return False, (
            f"This query is not a recognized read-only statement "
            f"(parsed as {type(tree).__name__}). Only SELECT-style "
            "queries are allowed."
        )

    return True, None
