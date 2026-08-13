"""Plain-English SQL explanation (rule-based, no extra LLM call).

Parses the generated SQL structurally and describes what it actually does:
which table, which columns, filters, grouping, ordering, and limit. This is
especially useful when clarification was skipped -- it lets the user catch
a mismatch between what they asked ("top 5 albums") and what the model
actually generated (e.g. ordered alphabetically by name, rather than by a
meaningful ranking metric like sales or rating), without having to read
raw SQL themselves.
"""

from __future__ import annotations
import sqlglot
from sqlglot import exp


def _column_label(node: exp.Expression) -> str:
    """Best-effort human label for a SELECT/ORDER BY/GROUP BY expression."""

    if isinstance(node, exp.Column):
        return node.name
    if isinstance(node, exp.Alias):
        return node.alias or _column_label(node.this)
    if isinstance(node, exp.Count):
        return "the count of matching rows"
    if isinstance(node, exp.Sum):
        inner = node.this
        return f"the total {_column_label(inner)}" if inner else "the total"
    if isinstance(node, exp.Avg):
        inner = node.this
        return f"the average {_column_label(inner)}" if inner else "the average"
    if isinstance(node, exp.Max):
        inner = node.this
        return f"the maximum {_column_label(inner)}" if inner else "the maximum"
    if isinstance(node, exp.Min):
        inner = node.this
        return f"the minimum {_column_label(inner)}" if inner else "the minimum"
    return node.sql(dialect="sqlite")


def _condition_to_english(node: exp.Expression) -> str:
    """Best-effort plain-English rendering of a WHERE condition tree."""

    if isinstance(node, exp.And):
        return f"{_condition_to_english(node.left)} and {_condition_to_english(node.right)}"
    if isinstance(node, exp.Or):
        return f"{_condition_to_english(node.left)} or {_condition_to_english(node.right)}"
    if isinstance(node, exp.EQ):
        return f"{_column_label(node.left)} is {node.right.sql(dialect='sqlite')}"
    if isinstance(node, exp.NEQ):
        return f"{_column_label(node.left)} is not {node.right.sql(dialect='sqlite')}"
    if isinstance(node, exp.GT):
        return f"{_column_label(node.left)} is greater than {node.right.sql(dialect='sqlite')}"
    if isinstance(node, exp.GTE):
        return f"{_column_label(node.left)} is at least {node.right.sql(dialect='sqlite')}"
    if isinstance(node, exp.LT):
        return f"{_column_label(node.left)} is less than {node.right.sql(dialect='sqlite')}"
    if isinstance(node, exp.LTE):
        return f"{_column_label(node.left)} is at most {node.right.sql(dialect='sqlite')}"
    if isinstance(node, exp.Like):
        return f"{_column_label(node.left)} matches the pattern {node.right.sql(dialect='sqlite')}"
    if isinstance(node, exp.In):
        return f"{_column_label(node.this)} is one of {node.sql(dialect='sqlite').split(' IN ')[-1]}"
    # Fallback: raw SQL fragment, still better than nothing
    return node.sql(dialect="sqlite")


def explain_sql(sql: str) -> str | None:
    """Return a plain-English description of a single SELECT/WITH query.

    Returns None if the SQL can't be parsed (caller should fall back to
    showing raw SQL only in that case).
    """

    text = (sql or "").strip()
    if not text:
        return None
    try:
        tree = sqlglot.parse_one(text, read="sqlite")
    except Exception:
        return None

    select = tree.find(exp.Select)
    if select is None:
        return None

    parts: list[str] = []

    # --- what is being selected ---
    select_exprs = select.expressions
    if len(select_exprs) == 1 and isinstance(
        select_exprs[0], (exp.Count, exp.Sum, exp.Avg, exp.Max, exp.Min)
    ):
        parts.append(f"This query calculates {_column_label(select_exprs[0])}")
    elif any(isinstance(e, exp.Star) for e in select_exprs):
        parts.append("This query returns all columns")
    else:
        labels = [_column_label(e) for e in select_exprs[:6]]
        extra = "" if len(select_exprs) <= 6 else f" and {len(select_exprs) - 6} more column(s)"
        if len(labels) == 1:
            parts.append(f"This query returns {labels[0]}")
        elif len(labels) == 2:
            parts.append(f"This query returns {labels[0]} and {labels[1]}" + extra)
        else:
            parts.append("This query returns " + ", ".join(labels[:-1]) + f", and {labels[-1]}" + extra)

    # --- from which table(s) ---
    tables = list(select.find_all(exp.Table))
    if tables:
        table_names = [t.name for t in tables]
        if len(table_names) == 1:
            parts.append(f"from **{table_names[0]}**")
        else:
            joined = " joined with ".join(f"**{n}**" for n in table_names)
            parts.append(f"by combining {joined}")

    # --- filters ---
    where = select.find(exp.Where)
    if where is not None and where.this is not None:
        try:
            condition_text = _condition_to_english(where.this)
            parts.append(f"where {condition_text}")
        except Exception:
            parts.append("with a filter condition")

    # --- grouping ---
    group = select.find(exp.Group)
    if group is not None and group.expressions:
        group_labels = [_column_label(e) for e in group.expressions]
        parts.append("grouped by " + ", ".join(group_labels))

    # --- ordering (the part that matters most for "top N" ambiguity) ---
    order = select.find(exp.Order)
    if order is not None and order.expressions:
        order_bits = []
        for o in order.expressions:
            direction = "descending" if o.args.get("desc") else "ascending"
            order_bits.append(f"{_column_label(o.this)} ({direction})")
        parts.append("ranked by " + ", ".join(order_bits))

    # --- limit ---
    limit = select.find(exp.Limit)
    if limit is not None and limit.expression is not None:
        parts.append(f"showing only the top {limit.expression.sql(dialect='sqlite')} result(s)")

    return ", ".join(parts) + "."