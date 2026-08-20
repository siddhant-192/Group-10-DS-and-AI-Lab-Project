"""Streamlit UI for Talk to Your Database (calls ask() only)."""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "app") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "app"))

from backend.ask import ask, get_backend  # noqa: E402
from backend.charts import CHART_TYPES, select_chart  # noqa: E402
from backend.clarify import (  # noqa: E402
    assess_clarification,
    schema_terms_from_db,
)
from backend.config import BACKEND_MOCK, UIConfig, load_ui_config  # noqa: E402
from backend.registry import list_databases, resolve_database  # noqa: E402


EXAMPLE_QUESTIONS = {
    "mini_music": [
        "How many singers are there?",
        "List all singers",
        "show top singers",
    ],
    "chinook": [
        "How many albums are there?",
        "Return only SQL. Which artist has the most albums?",
        "show best artists",
    ],
}


st.set_page_config(page_title="Talk to Your Database", layout="wide")
st.title("Talk to Your Database")
st.caption(
    "Natural language → (optional one clarification) → SQL → safe results → table + chart"
)


@st.cache_resource(show_spinner="Loading model backend…")
def cached_backend(
    backend: str,
    model_slug: str,
    adapter_dir: str,
    load_4bit: bool,
    max_new_tokens: int,
    models_config_path: str,
    demo_databases_dir: str,
    execute_timeout_seconds: float,
    max_result_rows: int,
    mschema_examples: int,
):
    """Load mock or HF backend once per Streamlit process."""
    cfg = UIConfig(
        backend=backend,
        model_slug=model_slug,
        adapter_dir=Path(adapter_dir) if adapter_dir else None,
        demo_databases_dir=Path(demo_databases_dir),
        models_config_path=Path(models_config_path),
        max_new_tokens=max_new_tokens,
        execute_timeout_seconds=execute_timeout_seconds,
        max_result_rows=max_result_rows,
        mschema_examples=mschema_examples,
        load_4bit=load_4bit,
    )
    return get_backend(cfg)


def _backend_for(config: UIConfig):
    adapter = str(config.adapter_dir) if config.adapter_dir else ""
    return cached_backend(
        config.backend,
        config.model_slug,
        adapter,
        config.load_4bit,
        config.max_new_tokens,
        str(config.models_config_path),
        str(config.demo_databases_dir),
        config.execute_timeout_seconds,
        config.max_result_rows,
        config.mschema_examples,
    )


# def _render_chart(frame, chart: dict) -> None:
#     chart_type = (chart or {}).get("chart_type") or "table"
#     reason = (chart or {}).get("reason") or ""
#     x_col = chart.get("x")
#     y_col = chart.get("y")

#     st.caption(f"Chart: **{chart_type}** — {reason}")

#     try:
#         if chart_type == "metric" and y_col and y_col in frame.columns and len(frame):
#             st.metric(y_col, frame[y_col].iloc[0])
#             return
#         if chart_type == "bar" and x_col and y_col and x_col in frame.columns and y_col in frame.columns:
#             plot = frame[[x_col, y_col]].copy()
#             plot[y_col] = pd.to_numeric(plot[y_col], errors="coerce")
#             plot = plot.dropna(subset=[y_col])
#             if plot.empty:
#                 st.warning("Bar chart needs numeric Y values — nothing to plot.")
#                 return
#             st.bar_chart(plot.set_index(x_col)[y_col], use_container_width=True)
#             return
#         if chart_type == "line" and x_col and y_col and x_col in frame.columns and y_col in frame.columns:
#             plot = frame[[x_col, y_col]].copy()
#             plot[y_col] = pd.to_numeric(plot[y_col], errors="coerce")
#             plot = plot.dropna(subset=[y_col])
#             if plot.empty:
#                 st.warning("Line chart needs numeric Y values — nothing to plot.")
#                 return
#             st.line_chart(plot.set_index(x_col)[y_col], use_container_width=True)
#             return
#         if chart_type == "scatter" and x_col and y_col and x_col in frame.columns and y_col in frame.columns:
#             st.scatter_chart(frame, x=x_col, y=y_col, use_container_width=True)
#             return
#     except Exception as exc:
#         st.warning(f"Could not render chart ({exc}). Showing table only.")
#         return
#     st.info("No chart for this result shape — table only.")

def _render_chart(frame, chart: dict, *, sort_ascending: bool = False) -> None:
    chart_type = (chart or {}).get("chart_type") or "table"
    reason = (chart or {}).get("reason") or ""
    x_col = chart.get("x")
    y_col = chart.get("y")
    sort_key = "y" if sort_ascending else "-y"

    st.caption(f"Chart: **{chart_type}** — {reason}")

    def _xy_plot():
        plot = frame[[x_col, y_col]].copy()
        plot[y_col] = pd.to_numeric(plot[y_col], errors="coerce")
        plot = plot.dropna(subset=[y_col])
        return plot.sort_values(y_col, ascending=sort_ascending)

    try:
        if chart_type == "metric" and y_col and y_col in frame.columns and len(frame):
            st.metric(y_col, frame[y_col].iloc[0])
            return

        if chart_type == "bar" and x_col and y_col and x_col in frame.columns and y_col in frame.columns:
            plot = _xy_plot()
            if plot.empty:
                st.warning("Bar chart needs numeric Y values — nothing to plot.")
                return
            # Altair sort="-y" is required; st.bar_chart ignores dataframe order.
            spec = (
                alt.Chart(plot)
                .mark_bar()
                .encode(
                    x=alt.X(x_col, type="nominal", sort=sort_key, title=x_col),
                    y=alt.Y(y_col, type="quantitative", title=y_col),
                    tooltip=[x_col, y_col],
                )
            )
            st.altair_chart(spec, use_container_width=True)
            return

        if chart_type == "pie" and x_col and y_col and x_col in frame.columns and y_col in frame.columns:
            plot = _xy_plot()
            if plot.empty:
                st.warning("Pie chart needs numeric values — nothing to plot.")
                return
            spec = (
                alt.Chart(plot)
                .mark_arc()
                .encode(
                    theta=alt.Theta(y_col, type="quantitative", sort=sort_key),
                    color=alt.Color(x_col, type="nominal"),
                    tooltip=[x_col, y_col],
                )
            )
            st.altair_chart(spec, use_container_width=True)
            return

        if chart_type == "line" and x_col and y_col and x_col in frame.columns and y_col in frame.columns:
            plot = frame[[x_col, y_col]].copy()
            plot[y_col] = pd.to_numeric(plot[y_col], errors="coerce")
            plot = plot.dropna(subset=[y_col])
            if plot.empty:
                st.warning("Line chart needs numeric Y values — nothing to plot.")
                return
            st.line_chart(plot.set_index(x_col)[y_col], use_container_width=True)
            return

        if chart_type == "scatter" and x_col and y_col and x_col in frame.columns and y_col in frame.columns:
            plot = frame[[x_col, y_col]].copy()
            plot[x_col] = pd.to_numeric(plot[x_col], errors="coerce")
            plot[y_col] = pd.to_numeric(plot[y_col], errors="coerce")
            plot = plot.dropna(subset=[x_col, y_col]).sort_values(
                y_col, ascending=sort_ascending
            )
            if plot.empty:
                st.warning("Scatter chart needs two numeric columns — nothing to plot.")
                return
            st.scatter_chart(plot, x=x_col, y=y_col, use_container_width=True)
            return
    except Exception as exc:
        st.warning(f"Could not render chart ({exc}). Showing table only.")
        return
    st.info("No chart for this result shape — table only.")


config = load_ui_config()
databases = list_databases(config.demo_databases_dir)
is_mock = config.backend.strip().lower() in {BACKEND_MOCK, "mock"}

with st.sidebar:
    st.header("Settings")
    if is_mock:
        st.warning("Backend: **mock** (offline / no GPU)")
    else:
        st.success(f"Backend: **live** `{config.backend}`")
    st.caption(f"Model: `{config.model_slug}`")
    if config.adapter_dir:
        st.caption(f"Adapter: `{config.adapter_dir}`")
    enable_clarification = st.checkbox(
        "Clarify vague questions",
        value=True,
        help="One clarifying turn before SQL when the question looks underspecified.",
    )
    chart_override = st.selectbox(
        "Chart",
        options=list(CHART_TYPES),
        index=0,
        help="auto = rule-based from result shape.",
    )
    chart_sort = st.selectbox(
        "Chart sort",
        options=["descending", "ascending"],
        index=0,
        help="Sort bar / pie / scatter by the numeric value.",
    )
    if not is_mock:
        if st.button("Warm up model"):
            with st.spinner("Loading model…"):
                backend = _backend_for(config)
                if hasattr(backend, "_ensure_loaded"):
                    backend._ensure_loaded()
            st.success("Model ready.")

if not databases:
    st.error(
        "No SQLite files in `demo_databases/`. Run:\n\n"
        "`python app/scripts/download_demo_databases.py`"
    )
    st.stop()

db_ids = sorted(databases.keys())
db_id = st.selectbox("Database", options=db_ids)

examples = EXAMPLE_QUESTIONS.get(
    db_id, ["How many rows are in the main table?"]
)
example = st.selectbox("Example question", options=examples)
if st.button("Load example into question box"):
    st.session_state["question_text"] = example
    st.session_state.pop("clarify_pending", None)
    st.rerun()

if "question_text" not in st.session_state:
    st.session_state["question_text"] = ""

question = st.text_area(
    "Question",
    key="question_text",
    height=100,
    placeholder='e.g. How many albums are there?',
)
run = st.button("Run", type="primary")


def _run_ask(
    *,
    clarification: str | None = None,
    clarification_skipped: bool = False,
) -> None:
    with st.spinner("Running ask()…"):
        backend = _backend_for(config)
        st.session_state["last_result"] = ask(
            question,
            db_id,
            config=config,
            backend=backend,
            chart_override=chart_override,
            clarification=clarification,
            clarification_skipped=clarification_skipped,
            clarification_gate=enable_clarification,
        )
        st.session_state["last_question"] = question
    st.session_state.pop("clarify_pending", None)


def _open_clarify(*, assessment_dict: dict, source: str) -> None:
    st.session_state["clarify_pending"] = {
        "question": question,
        "db_id": db_id,
        "assessment": assessment_dict,
        "source": source,
    }
    # Keep the failed result on screen for after-error repair.
    if source != "after_error":
        st.session_state.pop("last_result", None)


if run:
    try:
        db_path = resolve_database(config.demo_databases_dir, db_id)
        schema_terms = schema_terms_from_db(db_path)
    except Exception as exc:
        st.error(f"Could not read schema for clarification: {exc}")
        schema_terms = []
    assessment = assess_clarification(question, schema_terms=schema_terms)
    st.session_state["last_clarify_assessment"] = assessment.to_dict()
    if enable_clarification and assessment.needed:
        _open_clarify(assessment_dict=assessment.to_dict(), source="pre_ask")
    else:
        _run_ask()

pending = st.session_state.get("clarify_pending")
if pending and pending.get("question") == question and pending.get("db_id") == db_id:
    info = pending["assessment"]
    source = pending.get("source") or "pre_ask"
    st.subheader(
        "Clarification needed"
        if source == "pre_ask"
        else "Previous query failed — add detail"
    )
    st.warning(info.get("question_to_user") or "Please clarify your question.")
    if info.get("reasons"):
        st.caption("Why: " + "; ".join(info["reasons"]))
    if info.get("matched_schema_terms"):
        st.caption(
            "Schema terms already matched: "
            + ", ".join(info["matched_schema_terms"][:8])
        )
    if info.get("suggestions"):
        st.write("Suggestions:")
        for tip in info["suggestions"]:
            st.write(f"- {tip}")
    clarify_text = st.text_input(
        "Your clarification",
        key="clarify_reply",
        placeholder="e.g. Top 5 by number of albums",
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Submit clarification", type="primary"):
            if not (clarify_text or "").strip():
                st.error(
                    "Clarification is empty. Add a short plain-English detail "
                    "(what to count or list — table names are optional), "
                    "or use Continue without clarification."
                )
            else:
                _run_ask(
                    clarification=clarify_text.strip(),
                    clarification_skipped=False,
                )
                st.rerun()
    with c2:
        if st.button("Continue without clarification"):
            _run_ask(clarification=None, clarification_skipped=True)
            st.rerun()

result = st.session_state.get("last_result")

if result:
    meta = result.get("model_metadata") or {}
    col1, col2, col3 = st.columns(3)
    col1.metric("Latency (ms)", result.get("latency_ms"))
    col2.metric("Backend", meta.get("backend", config.backend))
    col3.metric("DB", result.get("db_id") or db_id)

    if result.get("answer"):
        st.subheader("Answer")
        st.markdown(result["answer"])

    if result.get("error"):
        st.error(result["error"])
        used_clarify = bool(meta.get("clarification")) or bool(
            meta.get("clarification_skipped")
        )
        if (
            enable_clarification
            and not used_clarify
            and not (
                pending
                and pending.get("question") == question
                and pending.get("db_id") == db_id
            )
        ):
            st.warning(
                "The previous query did not succeed. Add a short clarification and run again."
            )
            if st.button("Add clarification"):
                err = str(result.get("error") or "unknown error")
                _open_clarify(
                    assessment_dict={
                        "needed": True,
                        "question_to_user": (
                            f"The last attempt failed: {err}\n\n"
                            "Add what to count, a top-N, or a filter, then submit."
                        ),
                        "suggestions": [
                            "Top 5 by a numeric measure (state which measure).",
                            "How many rows for a specific entity?",
                            "List all matching rows without ranking.",
                        ],
                        "reasons": [err],
                    },
                    source="after_error",
                )
                st.rerun()

    if result.get("sql"):
        st.subheader("SQL")
        st.code(result["sql"], language="sql")
        if result.get("sql_explanation"):
            st.caption(f"In plain English: {result['sql_explanation']}")

    if result.get("columns") is not None and result.get("rows") is not None:
        frame = pd.DataFrame(result["rows"], columns=result["columns"])
        st.subheader("Result table")
        st.dataframe(frame, use_container_width=True)

        st.subheader("Chart")
        # Changing sidebar override re-runs the script — re-chart without new model call..
        chart = select_chart(
            result.get("columns"),
            result.get("rows"),
            override=chart_override,
        ).to_dict()
        _render_chart(frame, chart, sort_ascending=(chart_sort == "ascending"))
    elif not result.get("error"):
        st.info("Query returned no rows.")

    with st.expander("Raw model output / metadata"):
        st.json(
            {
                "raw_model_output": result.get("raw_model_output"),
                "model_metadata": meta,
                "chart": select_chart(
                    result.get("columns"),
                    result.get("rows"),
                    override=chart_override,
                ).to_dict()
                if result.get("columns") is not None
                else result.get("chart"),
                "db_path": result.get("db_path"),
                "execute_ms": result.get("execute_ms"),
            }
        )
