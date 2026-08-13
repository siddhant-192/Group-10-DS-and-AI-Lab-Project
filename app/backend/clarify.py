"""Bounded one-shot clarification (Milestone 1 style).

DB-agnostic: entity grounding comes from the *selected* SQLite catalog
(table/column names), not hard-coded Chinook/mini_music word lists.
Model-agnostic: no extra LLM call; works for mock / Qwen2.5 / Qwen3.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable, Sequence


CLARIFY_VERSION = "schema-aware-v2"

# Underspecified / vague cues (linguistic — no DB dependency).
_VAGUE = re.compile(
    r"\b("
    r"best|worst|top|bottom|recent|latest|oldest|main|important|interesting|"
    r"show\s+me|tell\s+me|give\s+me|what\s+about|how\s+about|"
    r"sales|revenue|performance|status|summary|overview|stats|analysis|"
    r"fast[- ]?moving|low\s+stock|popular|good|bad"
    r")\b",
    re.IGNORECASE,
)

_METRIC_WORDS = re.compile(
    r"\b(count|how\s+many|sum|average|avg|total|number\s+of|min|max)\b",
    re.IGNORECASE,
)

_STRIP_FORMAT = re.compile(
    r"(?i)\b("
    r"return\s+only\s+sql|"
    r"sql\s+only|"
    r"no\s+explanation|"
    r"without\s+explanation|"
    r"do\s+not\s+explain|"
    r"just\s+the\s+sql|"
    r"only\s+sql"
    r")\b[,.:;!]?"
)

_HAS_LIMIT_OR_ORDER = re.compile(
    r"\b("
    r"top[\s-]?\d+|bottom[\s-]?\d+|last[\s-]?\d+|first[\s-]?\d+|limit[\s-]?\d+|"
    r"top\s+(?:one|two|three|four|five|six|seven|eight|nine|ten)|"
    r"bottom\s+(?:one|two|three|four|five|six|seven|eight|nine|ten)"
    r")\b",
    re.IGNORECASE,
)

# Ranking *measure* (not just N). "top 3 albums" has N but no measure.
_HAS_MEASURE = re.compile(
    r"\b("
    r"by\s+[a-z]+|"
    r"count|sum|average|avg|total|sales|revenue|plays|streams|"
    r"duration|length|price|year|rating|score"
    r")\b",
    re.IGNORECASE,
)

_HAS_TIME = re.compile(
    r"\b("
    r"today|yesterday|week|month|year|quarter|daily|monthly|yearly|"
    r"\d{4}|january|february|march|april|may|june|july|august|"
    r"september|october|november|december"
    r")\b",
    re.IGNORECASE,
)

_STOP = {
    "a",
    "an",
    "the",
    "of",
    "to",
    "in",
    "on",
    "for",
    "and",
    "or",
    "is",
    "are",
    "what",
    "which",
    "who",
    "how",
    "many",
    "much",
    "all",
    "from",
    "with",
    "by",
    "id",
    "name",
    "type",
    "date",
    "time",
}


@dataclass(frozen=True)
class ClarificationRequest:
    needed: bool
    question_to_user: str
    suggestions: list[str]
    reasons: list[str]
    matched_schema_terms: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def schema_terms_from_db(db_path: Path) -> list[str]:
    """Collect table + column identifiers from a SQLite file (DB-agnostic)."""

    path = Path(db_path).resolve()
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    terms: list[str] = []
    try:
        tables = [
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        for table in tables:
            terms.append(table)
            quoted = '"' + table.replace('"', '""') + '"'
            for column in connection.execute(f"PRAGMA table_info({quoted})"):
                name = str(column[1])
                if name:
                    terms.append(name)
                    terms.append(f"{table}.{name}")
    finally:
        connection.close()
    return terms


def _split_ident(ident: str) -> list[str]:
    """Split CustomerId / customer_id / MediaType into matchable tokens."""

    s = (ident or "").strip()
    if not s or "." in s and s.count(".") == 1:
        # table.column → both sides
        if "." in s:
            left, right = s.split(".", 1)
            return _split_ident(left) + _split_ident(right)
    s = s.replace(".", " ").replace("-", " ")
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    s = s.replace("_", " ")
    parts = [p.lower() for p in s.split() if p]
    return parts


def _variants(token: str) -> set[str]:
    t = token.lower().strip()
    if len(t) < 2 or t in _STOP:
        return set()
    out = {t}
    if t.endswith("ies") and len(t) > 4:
        out.add(t[:-3] + "y")
    if t.endswith("s") and len(t) > 3:
        out.add(t[:-1])
    else:
        out.add(t + "s")
    return out


# Close typos only (alum≈album). Cheap: a few dozen schema tokens, no embeddings.
_FUZZY_MIN_LEN = 4
_FUZZY_RATIO = 0.82


def _schema_token_map(schema_terms: Sequence[str]) -> dict[str, list[str]]:
    """Map a normalized token → schema identifiers that contain it."""

    mapping: dict[str, list[str]] = {}
    for term in schema_terms:
        parts = _split_ident(term)
        useful = [p for p in parts if p not in _STOP and len(p) >= 3]
        if not useful:
            useful = [p for p in parts if p not in _STOP and len(p) >= 2]
        for part in useful:
            for variant in _variants(part) | {part}:
                mapping.setdefault(variant, [])
                if term not in mapping[variant]:
                    mapping[variant].append(term)
    return mapping


def _fuzzy_hits(q_tokens: set[str], token_map: dict[str, list[str]]) -> list[str]:
    catalog = [tok for tok in token_map if len(tok) >= _FUZZY_MIN_LEN]
    found: list[str] = []
    seen: set[str] = set()
    for word in q_tokens:
        if word in _STOP or len(word) < _FUZZY_MIN_LEN:
            continue
        if word in token_map:
            continue
        best_tok = None
        best_ratio = 0.0
        for tok in catalog:
            if abs(len(tok) - len(word)) > 2:
                continue
            ratio = SequenceMatcher(None, word, tok).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_tok = tok
        if best_tok is not None and best_ratio >= _FUZZY_RATIO:
            for term in token_map[best_tok]:
                key = term.lower()
                if key not in seen:
                    seen.add(key)
                    found.append(term)
    return found


def matched_schema_terms(question: str, schema_terms: Sequence[str]) -> list[str]:
    """Return schema identifiers named in the question (exact, plural, or close typo)."""

    q = (question or "").lower()
    q_tokens = set(re.findall(r"[a-z0-9]+", q))
    token_map = _schema_token_map(schema_terms)
    hits: list[str] = []
    seen: set[str] = set()
    for term in schema_terms:
        parts = _split_ident(term)
        useful = [p for p in parts if p not in _STOP and len(p) >= 3]
        if not useful:
            useful = [p for p in parts if p not in _STOP and len(p) >= 2]
        if not useful:
            continue
        ok = False
        for part in useful:
            for variant in _variants(part):
                if variant in q_tokens or re.search(rf"\b{re.escape(variant)}\b", q):
                    ok = True
                    break
            if ok:
                break
        if ok:
            key = term.lower()
            if key not in seen:
                seen.add(key)
                hits.append(term)
    for term in _fuzzy_hits(q_tokens, token_map):
        key = term.lower()
        if key not in seen:
            seen.add(key)
            hits.append(term)
    return hits


def assess_clarification(
    question: str,
    *,
    schema_terms: Sequence[str] | None = None,
) -> ClarificationRequest:
    """Decide whether to ask exactly one clarifying question before SQL generation.

    ``schema_terms`` should come from the *selected* database. If omitted, the
    checker cannot ground entities and treats missing entity mentions as vague
    for count/list/show-style asks (safer default).
    """

    raw = (question or "").strip()
    if not raw:
        return ClarificationRequest(False, "", [], ["empty"], [])

    q = _STRIP_FORMAT.sub(" ", raw)
    q = re.sub(r"\s+", " ", q).strip(" ,.;:")
    if not q:
        q = raw

    terms = list(schema_terms or [])
    hits = matched_schema_terms(q, terms) if terms else []
    has_schema_hit = bool(hits)

    reasons: list[str] = []
    tokens = q.split()
    vague = bool(_VAGUE.search(q))
    has_metric = bool(_METRIC_WORDS.search(q))
    has_limit = bool(_HAS_LIMIT_OR_ORDER.search(q))
    has_measure = bool(_HAS_MEASURE.search(q) or has_metric)
    has_time = bool(_HAS_TIME.search(q))
    asks_count = bool(re.search(r"(?i)\b(count|how\s+many|number\s+of)\b", q))
    asks_list = bool(re.match(r"(?i)^\s*(list|show|give|display)\b", q))
    asks_what = bool(re.match(r"(?i)^\s*what\b", q))
    is_ranking = bool(
        re.search(r"\b(top|best|worst|popular)\d*\b", q, re.I)
        or re.search(r"\btop[\s-]?\d+", q, re.I)
    )

    too_short = len(tokens) < 4
    if too_short:
        reasons.append("fewer than 4 words - treating as underspecified")
    if vague and not (has_metric and has_schema_hit) and not (is_ranking and has_schema_hit):
        reasons.append("vague wording without a clear metric + schema entity")
    if is_ranking and not has_measure:
        reasons.append("ranking without saying what to rank by (e.g. by sales, by year)")
    if is_ranking and not has_limit:
        reasons.append("ranking word without how many (e.g. top 5)")
    if re.search(r"\b(recent|latest|last)\b", q, re.I) and not has_time and not has_limit:
        reasons.append("time wording without a concrete period")
    if asks_count and not has_schema_hit:
        reasons.append("asks for a count but does not name a table/entity from this database")
    if (asks_list or asks_what) and not has_schema_hit and not has_limit:
        reasons.append("list/what question without naming a table/entity from this database")
    if terms and not has_schema_hit:
        reasons.append("does not mention any table or column from the selected database")
    if not terms:
        reasons.append("no schema terms supplied — treating entity grounding as unknown")

    # Never skip on < 4 words. After the user skips/junk-clarifies,
    # ask.py still generates using the named entity + a default measure.
    if too_short:
        pass
    elif (
        re.match(r"(?i)^\s*how\s+many\b", q)
        and has_schema_hit
        and len(tokens) >= 4
    ):
        return ClarificationRequest(
            False, "", [], ["clear count question"], hits
        )
    elif asks_list and has_schema_hit and len(tokens) >= 4 and not is_ranking:
        return ClarificationRequest(
            False, "", [], ["clear list question"], hits
        )
    elif is_ranking and has_schema_hit and has_measure:
        return ClarificationRequest(
            False, "", [], ["ranking with entity and measure"], hits
        )

    # If the only issue was "no schema terms", still clarify count/list/what.
    if reasons == ["no schema terms supplied — treating entity grounding as unknown"]:
        if asks_count or asks_list or asks_what or vague:
            pass  # keep needed
        else:
            return ClarificationRequest(False, "", [], reasons, hits)

    needed = bool(reasons)
    if not needed:
        return ClarificationRequest(False, "", [], reasons, hits)

    hint_tables = table_hints(terms, limit=6)
    suggestions = []
    if hint_tables:
        suggestions.append(
            "Name what to count/list, e.g. "
            + ", ".join(f"how many {t}?" for t in hint_tables[:3])
        )
    suggestions.extend(
        [
            "If this is a top-N list: say what to rank by (sales, year, count).",
            "Add a filter (name, country, year) if you have one.",
        ]
    )
    if is_ranking and has_schema_hit and not has_measure:
        question_to_user = (
            "You named what to list, but not what \"top\" means. "
            "Reply with the ranking measure in plain English "
            "(for example: by number of tracks, by sales, by year). "
            "You can continue without that and a default measure will be used."
        )
    elif too_short:
        question_to_user = (
            "That question is under 4 words, so it is too short to run as-is. "
            "Add what to count/list, a filter, or how many rows. "
            "You can continue without that and a default query will be used "
            "if you already named a business thing (for example albums)."
        )
    else:
        question_to_user = (
            "Your question looks underspecified for this database. "
            "Please say what to count/measure (which business thing), "
            "filters, or how many rows — one short reply. "
            "You do not need exact SQL table names; plain English is fine."
        )
    return ClarificationRequest(
        True, question_to_user, suggestions, reasons, hits
    )


def table_hints(schema_terms: Iterable[str], limit: int = 6) -> list[str]:
    """Prefer bare table names (no dots) for user-facing suggestions."""

    tables: list[str] = []
    seen: set[str] = set()
    for term in schema_terms:
        if "." in term:
            continue
        key = term.lower()
        if key in seen or key in _STOP or len(term) < 2:
            continue
        seen.add(key)
        tables.append(term)
        if len(tables) >= limit:
            break
    return tables


def compose_question(question: str, clarification: str | None, skipped: bool = False) -> str:
    """Build the single-turn question text sent to the model."""

    base = (question or "").strip()
    note = (clarification or "").strip()
    if skipped and not note:
        return base
    if note:
        return (
            f"{base}\n\n"
            f"Clarification from user: {note}\n\n"
            "Use this clarification. Return SQL only inside a ```sql fence."
        )
    return base


def is_grounded(question: str, schema_terms: Sequence[str] | None) -> bool:
    """True if the question uses a business word that maps to this database."""

    return bool(matched_schema_terms(question or "", list(schema_terms or [])))


def clarification_is_usable(note: str, schema_terms: Sequence[str] | None) -> bool:
    """True if the clarification names an entity or adds a real analytic constraint."""

    text = (note or "").strip()
    if not text:
        return False
    if is_grounded(text, schema_terms):
        return True
    return looks_like_analytics_question(text)


def looks_like_analytics_question(question: str) -> bool:
    """True if the text looks like a data ask (plain English is enough)."""

    q = (question or "").strip()
    if not q:
        return False
    if _METRIC_WORDS.search(q) or _VAGUE.search(q):
        return True
    if re.match(r"(?i)^\s*(list|show|give|display)\b", q):
        return True
    if re.match(r"(?i)^\s*(what|which|who)\b", q) and (
        _METRIC_WORDS.search(q) or _VAGUE.search(q)
    ):
        return True
    if re.search(r"(?i)\b(compare|filter|group by)\b", q):
        return True
    return False
