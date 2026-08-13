"""Orchestration: question + db_id → SQL + safe readonly results.."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, UIConfig, load_ui_config
from .charts import select_chart, short_answer
from .clarify import assess_clarification, compose_question, schema_terms_from_db
from .models import ModelBackend, build_backend
from .mschema import render_mschema
from .registry import resolve_database
from .sql_utils import execute_query, extract_sql
from .explain_query import explain_sql

def _ensure_src_on_path() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _validate_sql(db_path: Path, sql: str, timeout_seconds: float) -> tuple[bool, str | None]:
    _ensure_src_on_path()
    from src.validation import validate_readonly_query

    result = validate_readonly_query(db_path, sql, timeout_seconds=timeout_seconds)
    return result.ok, result.error


_BACKEND_CACHE: dict[str, ModelBackend] = {}


def get_backend(config: UIConfig) -> ModelBackend:
    key = f"{config.backend}|{config.model_slug}|{config.adapter_dir}|{config.load_4bit}"
    if key not in _BACKEND_CACHE:
        _BACKEND_CACHE[key] = build_backend(config)
    return _BACKEND_CACHE[key]

def _friendly_validation_error(validation_error: str | None) -> str:
    """Convert low-level SQL validation errors into useful UI messages."""

    message = (validation_error or "").strip()

    if not message:
        return "The generated SQL could not be safely validated."

    if "no such column" in message.lower():
        return (
            "I couldn't safely interpret the question using the available "
            "database schema. The generated query referenced a column that "
            "does not exist."
        )

    if "no such table" in message.lower():
        return (
            "I couldn't safely interpret the question using the available "
            "database schema. The generated query referenced a table that "
            "does not exist."
        )

    return f"SQL validation failed: {message}"

def ask(
    question: str,
    db_id: str,
    *,
    config: UIConfig | None = None,
    backend: ModelBackend | None = None,
    chart_override: str = "auto",
    clarification: str | None = None,
    clarification_skipped: bool = False,
    clarification_gate: bool = False,
) -> dict[str, Any]:
    """Run the M3-style pipeline and return a UI-friendly payload.

    Presentation fields ``answer`` and ``chart`` follow the Milestone docs
    (template summary + rule-based chart spec). Generation is still one model
    call; optional ``clarification`` is folded into the question text first.

    If ``clarification_gate`` is True and the question looks underspecified for
    this database, refuse to generate until the caller supplies clarification
    or sets ``clarification_skipped=True`` (prevents silent wrong-table guesses).
    """

    started = time.monotonic()
    cfg = config or load_ui_config()
    question = (question or "").strip()
    if not question:
        return _error_payload(cfg, "question is empty", started)

    try:
        db_path = resolve_database(cfg.demo_databases_dir, db_id)
    except KeyError as exc:
        return _error_payload(cfg, str(exc), started)

    if clarification_gate and not clarification:
        try:
            terms = schema_terms_from_db(db_path)
        except Exception:
            terms = []
        assessment = assess_clarification(question, schema_terms=terms)

        # Hard block, NOT bypassed by clarification_skipped: a question with
        # zero connection to this database's schema (e.g. "abc") should never
        # reach the model. This is distinct from ordinary vagueness -- the
        # "skip clarification" button is meant to accept the model's safest
        # interpretation of an underspecified-but-related question, not to
        # force an answer out of a question that names nothing in this
        # database at all.
        no_grounding_at_all = terms and not assessment.matched_schema_terms
        if assessment.needed and no_grounding_at_all:
            payload = {
                "sql": None,
                "columns": None,
                "rows": None,
                "error": (
                    "This question does not reference anything in the "
                    "selected database, so no query can be safely generated. "
                    "Please rephrase it to relate to this database's tables "
                    "or data."
                ),
                "latency_ms": round((time.monotonic() - started) * 1000, 3),
                "model_metadata": {
                    "backend": cfg.backend,
                    "clarification_required": True,
                    "blocked_reason": "no_schema_grounding",
                    "clarification_reasons": assessment.reasons,
                    "matched_schema_terms": assessment.matched_schema_terms,
                },
                "raw_model_output": None,
                "db_id": db_id,
                "db_path": str(db_path),
                "clarification_request": assessment.to_dict(),
            }
            return _with_presentation(payload, question, chart_override)

        # Ordinary vagueness (has some grounding, just underspecified) is
        # still skippable as before.
        if assessment.needed and not clarification_skipped:
            payload = {
                "sql": None,
                "columns": None,
                "rows": None,
                "error": (
                    "clarification required before SQL: "
                    + (assessment.question_to_user or "question is underspecified")
                ),
                "latency_ms": round((time.monotonic() - started) * 1000, 3),
                "model_metadata": {
                    "backend": cfg.backend,
                    "clarification_required": True,
                    "clarification_reasons": assessment.reasons,
                    "matched_schema_terms": assessment.matched_schema_terms,
                },
                "raw_model_output": None,
                "db_id": db_id,
                "db_path": str(db_path),
                "clarification_request": assessment.to_dict(),
            }
            return _with_presentation(payload, question, chart_override)

    effective_question = compose_question(
        question, clarification, skipped=clarification_skipped
    )

    try:
        schema_text = render_mschema(db_path, db_id, example_num=cfg.mschema_examples)
    except Exception as exc:  # pragma: no cover - surface to UI
        return _error_payload(cfg, f"schema render failed: {exc}", started)

    try:
        model = backend or get_backend(cfg)
        raw_text, model_meta = model.generate(schema_text, effective_question)
    except Exception as exc:
        return _error_payload(cfg, f"model generate failed: {exc}", started, model_metadata={"backend": cfg.backend})

    model_meta = {
        **model_meta,
        "effective_question": effective_question,
        "clarification": clarification,
        "clarification_skipped": clarification_skipped,
    }

    sql = extract_sql(raw_text)
    if not sql:
        payload = {
            "sql": None,
            "columns": None,
            "rows": None,
            "error": "could not extract SQL from model output",
            "latency_ms": round((time.monotonic() - started) * 1000, 3),
            "model_metadata": model_meta,
            "raw_model_output": raw_text,
            "db_id": db_id,
            "db_path": str(db_path),
        }
        return _with_presentation(payload, question, chart_override)

    ok, validation_error = _validate_sql(db_path, sql, cfg.execute_timeout_seconds)
    if not ok:
        payload = {
            "sql": sql,
            "sql_explanation": explain_sql(sql),
            "columns": None,
            "rows": None,
            "error": _friendly_validation_error(validation_error),            
            "latency_ms": round((time.monotonic() - started) * 1000, 3),
            "model_metadata": model_meta,
            "raw_model_output": raw_text,
            "db_id": db_id,
            "db_path": str(db_path),
        }
        return _with_presentation(payload, question, chart_override)

    executed = execute_query(
        db_path,
        sql,
        timeout_seconds=cfg.execute_timeout_seconds,
        max_result_rows=cfg.max_result_rows,
    )
    error = None if executed.status == "ok" else (executed.error or executed.status)
    payload = {
        "sql": sql,
        "sql_explanation": explain_sql(sql),
        "columns": executed.columns,
        "rows": executed.rows,
        "error": error,
        "latency_ms": round((time.monotonic() - started) * 1000, 3),
        "execute_ms": executed.elapsed_ms,
        "model_metadata": model_meta,
        "raw_model_output": raw_text,
        "db_id": db_id,
        "db_path": str(db_path),
    }
    return _with_presentation(payload, question, chart_override)


def _with_presentation(
    payload: dict[str, Any],
    question: str,
    chart_override: str,
) -> dict[str, Any]:
    columns = payload.get("columns")
    rows = payload.get("rows")
    error = payload.get("error")
    payload["answer"] = short_answer(question, columns, rows, error)
    payload["chart"] = select_chart(columns, rows, override=chart_override).to_dict()
    return payload


def _error_payload(
    cfg: UIConfig,
    message: str,
    started: float,
    model_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "sql": None,
        "columns": None,
        "rows": None,
        "error": message,
        "latency_ms": round((time.monotonic() - started) * 1000, 3),
        "model_metadata": model_metadata or {"backend": cfg.backend},
        "raw_model_output": None,
        "db_id": None,
        "db_path": None,
    }
    return _with_presentation(payload, "", "auto")