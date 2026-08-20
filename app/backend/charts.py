"""Rule-based chart selection from result shape (Milestone 1 / 3 presentation contract).

Rules (deterministic, no LLM):
  - 1 row x 1 numeric col          → metric
  - 1 categorical + 1 numeric, n≤25 → bar 
  - 1 temporal + 1 numeric          → line
  - 2 numeric columns               → scatter
  - otherwise                       → table only
  - optional                        → pie
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any


CHART_TYPES = ("auto", "metric", "bar", "pie", "line", "scatter", "table")


@dataclass(frozen=True)
class ChartSpec:
    chart_type: str
    reason: str
    x: str | None = None
    y: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_temporal(value: Any) -> bool:
    if isinstance(value, (date, datetime)):
        return True
    if not isinstance(value, str) or not value.strip():
        return False
    text = value.strip()
    # Cheap ISO-ish / common date patterns
    if len(text) >= 8 and text[0:4].isdigit() and ("-" in text or "/" in text):
        return True
    return False


def _column_kind(rows: list[list[Any]], col_index: int) -> str:
    samples = [row[col_index] for row in rows if row[col_index] is not None][:40]
    if not samples:
        return "empty"
    if all(_is_number(v) for v in samples):
        return "numeric"
    if all(_is_temporal(v) for v in samples):
        return "temporal"
    return "categorical"


def select_chart(
    columns: list[str] | None,
    rows: list[list[Any]] | None,
    override: str = "auto",
) -> ChartSpec:
    """Pick a chart type from result shape, with optional manual override."""

    if not columns or rows is None:
        return ChartSpec("table", "no result rows")

    override = (override or "auto").strip().lower()
    if override not in CHART_TYPES:
        override = "auto"

    n_rows = len(rows)
    n_cols = len(columns)
    kinds = [_column_kind(rows, i) for i in range(n_cols)]

    def _pick_xy_for_bar_or_line() -> tuple[str | None, str | None, str]:
        cat_idxs = [i for i, k in enumerate(kinds) if k in {"categorical", "temporal"}]
        num_idxs = [i for i, k in enumerate(kinds) if k == "numeric"]
        if not num_idxs:
            return None, None, "no-numeric-y"
        # Prefer a non-ID numeric column for Y when possible
        preferred = [
            i
            for i in num_idxs
            if not columns[i].lower().endswith("_id") and columns[i].lower() != "id"
        ]
        y_idx = preferred[0] if preferred else num_idxs[0]
        if cat_idxs:
            return columns[cat_idxs[0]], columns[y_idx], "ok"
        if len(num_idxs) >= 2:
            other = num_idxs[0] if num_idxs[0] != y_idx else num_idxs[1]
            return columns[other], columns[y_idx], "two-numeric"
        return None, None, "no-numeric-y"

    def _forced(chart_type: str, reason: str) -> ChartSpec:
        x = y = None
        if chart_type in {"bar", "line", "pie"}:
            x, y, status = _pick_xy_for_bar_or_line()
            if status == "no-numeric-y":
                return ChartSpec(
                    "table",
                    f"override {chart_type} needs a numeric Y column — showing table",
                )
            return ChartSpec(chart_type, reason, x=x, y=y)
        if chart_type == "scatter":
            num_idxs = [i for i, k in enumerate(kinds) if k == "numeric"]
            if len(num_idxs) >= 2:
                return ChartSpec(
                    "scatter",
                    reason,
                    x=columns[num_idxs[0]],
                    y=columns[num_idxs[1]],
                )
            return ChartSpec("table", "override scatter needs two numeric columns")
        if chart_type == "metric":
            num_idxs = [i for i, k in enumerate(kinds) if k == "numeric"]
            if num_idxs:
                return ChartSpec("metric", reason, y=columns[num_idxs[0]])
            return ChartSpec("table", "override metric needs a numeric column")
        return ChartSpec(chart_type, reason, x=x, y=y)

    if override != "auto":
        return _forced(override, f"manual override: {override}")

    # Scalar metric
    if n_rows == 1 and n_cols == 1 and kinds[0] == "numeric":
        return ChartSpec("metric", "single numeric scalar", y=columns[0])

    if n_cols == 2:
        k0, k1 = kinds[0], kinds[1]
        # categorical + numeric → bar
        if k0 == "categorical" and k1 == "numeric" and n_rows <= 25:
            return ChartSpec(
                "bar",
                "categorical x + numeric y (<=25 rows)",
                x=columns[0],
                y=columns[1],
            )
        if k1 == "categorical" and k0 == "numeric" and n_rows <= 25:
            return ChartSpec(
                "bar",
                "numeric first, categorical second — swapped for bar",
                x=columns[1],
                y=columns[0],
            )
        # temporal + numeric → line
        if k0 == "temporal" and k1 == "numeric":
            return ChartSpec("line", "temporal x + numeric y", x=columns[0], y=columns[1])
        if k1 == "temporal" and k0 == "numeric":
            return ChartSpec("line", "numeric + temporal — swapped for line", x=columns[1], y=columns[0])
        # two numerics → scatter
        if k0 == "numeric" and k1 == "numeric" and n_rows >= 2:
            return ChartSpec("scatter", "two numeric columns", x=columns[0], y=columns[1])

    return ChartSpec("table", "shape not mapped to a chart — show table only")


def short_answer(
    question: str,
    columns: list[str] | None,
    rows: list[list[Any]] | None,
    error: str | None,
) -> str:
    """Template natural-language blurb (no extra LLM call)."""

    if error:
        return f"Could not answer: {error}"
    if columns is None or rows is None:
        return "No result rows to summarize."
    if len(rows) == 0:
        return "The query ran successfully but returned no rows."
    if len(rows) == 1 and len(columns) == 1:
        return f"Answer: **{rows[0][0]}** ({columns[0]})."
    if len(rows) == 1:
        pairs = ", ".join(f"{c}={v}" for c, v in zip(columns, rows[0]))
        return f"Single-row result: {pairs}."
    return (
        f"Returned **{len(rows)}** rows and **{len(columns)}** columns "
        f"for: “{question.strip()}”."
    )
