#!/usr/bin/env python3
"""Build a validated, feature-aware Spider chat-SFT training package."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = PROJECT_ROOT / "data" / "processed" / "spider" / "train.jsonl"
DEFAULT_VALIDATION = PROJECT_ROOT / "data" / "processed" / "spider" / "validation.jsonl"
DEFAULT_POLICY = PROJECT_ROOT / "configs" / "text2sql_sft_sampling.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "finetuning" / "spider_sft_v1"
READ_ONLY_SQL = re.compile(r"^\s*(?:SELECT|WITH)\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    os.replace(temporary, path)
    return count


def normalize_sql(sql: str) -> str:
    return " ".join(sql.strip().rstrip(";").casefold().split())


def normalize_question(question: str) -> str:
    return " ".join(question.casefold().split())


def query_features(row: dict[str, Any]) -> dict[str, Any]:
    return dict(row.get("metadata", {}).get("query_features", {}))


def matched_features(row: dict[str, Any]) -> list[str]:
    features = query_features(row)
    values: list[str] = []
    if int(features.get("aggregate_count", 0)) > 0:
        values.append("aggregate")
    if bool(features.get("has_where")):
        values.append("where")
    if int(features.get("join_count", 0)) > 0:
        values.append("join")
    if int(features.get("join_count", 0)) >= 2:
        values.append("multi_join")
    if bool(features.get("has_group_by")):
        values.append("group_by")
    if bool(features.get("has_having")):
        values.append("having")
    if bool(features.get("has_order_by")):
        values.append("order_by")
    if bool(features.get("has_limit")):
        values.append("limit")
    if bool(features.get("has_distinct")):
        values.append("distinct")
    if bool(features.get("has_subquery")):
        values.append("subquery")
    if int(features.get("set_operation_count", 0)) > 0:
        values.append("set_operation")
    if int(features.get("join_count", 0)) > 0 and bool(features.get("has_subquery")):
        values.append("join_and_subquery")
    if features.get("complexity_proxy") == "complex":
        values.append("complex")
    return values


def priority_multiplier(row: dict[str, Any], policy: dict[str, Any]) -> float:
    multipliers = {str(key): float(value) for key, value in policy["feature_multipliers"].items()}
    unknown = set(matched_features(row)) - set(multipliers)
    if unknown:
        raise ValueError(f"Policy does not define feature(s): {', '.join(sorted(unknown))}")
    return max((1.0, *(multipliers[key] for key in matched_features(row))))


def validate_policy(policy: dict[str, Any]) -> None:
    if policy.get("combine_rule") != "maximum":
        raise ValueError("Only combine_rule='maximum' is supported; additive weights can explode overlaps")
    fraction = float(policy.get("supplement_fraction", 0))
    if not 0 <= fraction <= 1:
        raise ValueError("supplement_fraction must be between 0 and 1")
    cap = int(policy.get("max_supplement_per_normalized_sql", 0))
    if cap < 1:
        raise ValueError("max_supplement_per_normalized_sql must be at least 1")
    multipliers = policy.get("feature_multipliers")
    if not isinstance(multipliers, dict) or not multipliers:
        raise ValueError("feature_multipliers must be a non-empty object")
    if any(float(value) < 1 for value in multipliers.values()):
        raise ValueError("feature multipliers must be at least 1")


def validate_source_rows(rows: Sequence[dict[str, Any]], split: str) -> None:
    ids = [str(row.get("id", "")) for row in rows]
    if not rows:
        raise ValueError(f"{split} source is empty")
    if any(not row_id for row_id in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{split} IDs are empty or duplicated")
    for row in rows:
        row_id = str(row["id"])
        if row.get("split") != split:
            raise ValueError(f"{row_id}: expected split={split}, found {row.get('split')}")
        if row.get("metadata", {}).get("execution_validation", {}).get("status") != "ok":
            raise ValueError(f"{row_id}: gold query is not execution-validated")
        messages = row.get("messages")
        if not isinstance(messages, list) or [item.get("role") for item in messages] != [
            "system",
            "user",
            "assistant",
        ]:
            raise ValueError(f"{row_id}: messages must have system/user/assistant roles")
        sql = str(row.get("sql", "")).strip()
        if str(messages[-1].get("content", "")).strip() != sql:
            raise ValueError(f"{row_id}: assistant target does not equal gold SQL")
        if not READ_ONLY_SQL.match(sql):
            raise ValueError(f"{row_id}: target does not start with SELECT or WITH")
        if "```" in sql:
            raise ValueError(f"{row_id}: target contains a Markdown fence")


def validate_split_isolation(train: Sequence[dict[str, Any]], validation: Sequence[dict[str, Any]]) -> dict[str, Any]:
    train_ids = {str(row["id"]) for row in train}
    validation_ids = {str(row["id"]) for row in validation}
    train_databases = {str(row["db_id"]) for row in train}
    validation_databases = {str(row["db_id"]) for row in validation}
    train_pairs = {
        (str(row["db_id"]), normalize_question(str(row["question"])), normalize_sql(str(row["sql"])))
        for row in train
    }
    validation_pairs = {
        (str(row["db_id"]), normalize_question(str(row["question"])), normalize_sql(str(row["sql"])))
        for row in validation
    }
    train_questions = {normalize_question(str(row["question"])) for row in train}
    validation_questions = {normalize_question(str(row["question"])) for row in validation}
    checks = {
        "source_id_overlap_count": len(train_ids & validation_ids),
        "database_overlap_count": len(train_databases & validation_databases),
        "exact_db_question_sql_overlap_count": len(train_pairs & validation_pairs),
        "normalized_question_overlap_count": len(train_questions & validation_questions),
        "train_databases": len(train_databases),
        "validation_databases": len(validation_databases),
    }
    if any(checks[key] for key in (
        "source_id_overlap_count",
        "database_overlap_count",
        "exact_db_question_sql_overlap_count",
    )):
        raise ValueError(f"Train/validation isolation failed: {checks}")
    return checks


def weighted_key(rng: random.Random, weight: float) -> float:
    """Efraimidis-Spirakis key; smaller values have higher selection priority."""
    return -math.log(max(rng.random(), 1e-15)) / weight


def select_supplement(
    rows: Sequence[dict[str, Any]], policy: dict[str, Any]
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    budget = round(len(rows) * float(policy["supplement_fraction"]))
    template_cap = int(policy["max_supplement_per_normalized_sql"])
    rng = random.Random(int(policy["seed"]))
    candidates: list[tuple[float, str, str]] = []
    audit: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = str(row["id"])
        template = normalize_sql(str(row["sql"]))
        multiplier = priority_multiplier(row, policy)
        extra_weight = multiplier - 1.0
        audit[row_id] = {
            "source_id": row_id,
            "db_id": row["db_id"],
            "complexity": query_features(row).get("complexity_proxy", "unknown"),
            "matched_features": matched_features(row),
            "priority_multiplier": multiplier,
            "supplement_selection_weight": extra_weight,
            "normalized_sql_sha256": sha256_text(template),
        }
        if extra_weight > 0:
            candidates.append((weighted_key(rng, extra_weight), row_id, template))

    candidates.sort(key=lambda item: (item[0], item[1]))
    selected: list[str] = []
    template_counts: Counter[str] = Counter()
    for _key, row_id, template in candidates:
        if template_counts[template] >= template_cap:
            continue
        selected.append(row_id)
        template_counts[template] += 1
        if len(selected) == budget:
            break
    if len(selected) != budget:
        raise ValueError(
            f"Sampling policy requested {budget} supplemental examples but its template cap allowed only "
            f"{len(selected)}; lower supplement_fraction or increase max_supplement_per_normalized_sql"
        )
    selected_set = set(selected)
    template_frequency = Counter(normalize_sql(str(row["sql"])) for row in rows)
    for row in rows:
        row_id = str(row["id"])
        template = normalize_sql(str(row["sql"]))
        audit[row_id].update(
            {
                "normalized_sql_source_frequency": template_frequency[template],
                "selected_for_hard_supplement": row_id in selected_set,
                "curriculum_copies": 2 if row_id in selected_set else 1,
            }
        )
    return selected, audit


def sft_record(
    row: dict[str, Any],
    audit: dict[str, Any],
    sample_origin: str,
    occurrence: int,
) -> dict[str, Any]:
    source_id = str(row["id"])
    return {
        "id": source_id if sample_origin != "hard_supplement" else f"{source_id}::hard-2",
        "source_id": source_id,
        "dataset": row.get("dataset", "spider"),
        "split": row["split"],
        "db_id": row["db_id"],
        "messages": row["messages"],
        "sampling": {
            "sample_origin": sample_origin,
            "curriculum_occurrence": occurrence,
            "matched_features": audit["matched_features"],
            "priority_multiplier": audit["priority_multiplier"],
        },
    }


def validation_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "source_id": row["id"],
        "dataset": row.get("dataset", "spider"),
        "split": row["split"],
        "db_id": row["db_id"],
        "messages": row["messages"],
        "evaluation": {
            "complexity": query_features(row).get("complexity_proxy", "unknown"),
            "query_features": matched_features(row),
        },
    }


def distribution(source_rows: Sequence[dict[str, Any]], occurrences: Sequence[str]) -> dict[str, Any]:
    by_id = {str(row["id"]): row for row in source_rows}
    features: Counter[str] = Counter()
    complexity: Counter[str] = Counter()
    databases: Counter[str] = Counter()
    templates: Counter[str] = Counter()
    for row_id in occurrences:
        row = by_id[row_id]
        features.update(matched_features(row))
        complexity[str(query_features(row).get("complexity_proxy", "unknown"))] += 1
        databases[str(row["db_id"])] += 1
        templates[normalize_sql(str(row["sql"]))] += 1
    total = len(occurrences)
    return {
        "examples": total,
        "unique_source_ids": len(set(occurrences)),
        "unique_normalized_sql": len(templates),
        "max_occurrences_per_source_id": max(Counter(occurrences).values(), default=0),
        "max_occurrences_per_normalized_sql": max(templates.values(), default=0),
        "databases": len(databases),
        "complexity": dict(sorted(complexity.items())),
        "feature_counts": dict(sorted(features.items())),
        "feature_pct": {
            key: round(100.0 * value / total, 3) for key, value in sorted(features.items())
        },
    }


def render_readme(manifest: dict[str, Any]) -> str:
    base = manifest["distributions"]["base_train"]
    curriculum = manifest["distributions"]["curriculum_train"]
    features = sorted(set(base["feature_counts"]) | set(curriculum["feature_counts"]))
    lines = [
        "# Spider chat-SFT package v1",
        "",
        "This package preserves every execution-validated Spider training example once, then appends a deterministic "
        "hard-example supplement selected from aggregate zero-shot error categories. The official validation split is "
        "kept unchanged and no validation row or database appears in training.",
        "",
        "## Files",
        "",
        "| File | Rows | Purpose |",
        "|---|---:|---|",
        f"| `train_base.jsonl` | {manifest['counts']['base_train']} | Natural Spider training distribution; one row per source example. |",
        f"| `train_curriculum.jsonl` | {manifest['counts']['curriculum_train']} | Base rows plus the capped hard supplement; recommended first SFT input. |",
        f"| `validation.jsonl` | {manifest['counts']['validation']} | Untouched official validation examples for loss/evaluation. |",
        f"| `sampling_weights.jsonl` | {manifest['counts']['sampling_weights']} | Per-source feature, weight, template-frequency, and selection audit. |",
        "| `manifest.json` | — | Checksums, provenance, policy, leakage checks, and distributions. |",
        "",
        "All training records use the standard `messages` field with exactly one system, user, and SQL-only assistant turn. "
        "Extra audit columns can be ignored by chat-SFT trainers.",
        "",
        "## Sampling behavior",
        "",
        f"- Seed: `{manifest['policy']['seed']}`",
        f"- Base examples retained: **{manifest['counts']['base_train']} / {manifest['counts']['base_train']}**",
        f"- Unique supplemental examples: **{manifest['counts']['hard_supplement']}**",
        f"- Maximum copies of a source example: **{curriculum['max_occurrences_per_source_id']}**",
        f"- Maximum supplemental selections per normalized SQL template: "
        f"**{manifest['policy']['max_supplement_per_normalized_sql']}**",
        "- Overlapping feature multipliers are combined with `max`, never added.",
        "",
        "| Feature | Base count | Base % | Curriculum count | Curriculum % |",
        "|---|---:|---:|---:|---:|",
    ]
    for feature in features:
        lines.append(
            f"| {feature} | {base['feature_counts'].get(feature, 0)} | "
            f"{base['feature_pct'].get(feature, 0):.2f}% | "
            f"{curriculum['feature_counts'].get(feature, 0)} | "
            f"{curriculum['feature_pct'].get(feature, 0):.2f}% |"
        )
    checks = manifest["leakage_checks"]
    lines.extend(
        [
            "",
            "## Integrity checks",
            "",
            f"- Training source rows: **{manifest['counts']['base_train']}**, all executable and SQL-only.",
            f"- Validation rows: **{manifest['counts']['validation']}**, unchanged and executable.",
            f"- Train/validation database overlap: **{checks['database_overlap_count']}**.",
            f"- Exact `(database, question, SQL)` overlap: **{checks['exact_db_question_sql_overlap_count']}**.",
            f"- Reused normalized question wording across disjoint databases: **{checks['normalized_question_overlap_count']}**.",
            f"- Curriculum unique source coverage: **{curriculum['unique_source_ids']} / {base['unique_source_ids']}**.",
            "",
            "The feature policy was selected using aggregate validation error slices. That is hyperparameter feedback, not "
            "row leakage; however, final claims should eventually be confirmed on an untouched test set.",
            "",
            "## Loading",
            "",
            "```python",
            "from datasets import load_dataset",
            "",
            "dataset = load_dataset(",
            "    \"json\",",
            "    data_files={",
            "        \"train\": \"data/finetuning/spider_sft_v1/train_curriculum.jsonl\",",
            "        \"validation\": \"data/finetuning/spider_sft_v1/validation.jsonl\",",
            "    },",
            ")",
            "```",
            "",
            "Use `train_base.jsonl` as the control run. This package's validation file is convenient for validation loss; use "
            "the original processed validation rows and SQLite databases for generation-time execution scoring. For model "
            "selection, track execution accuracy rather than validation loss or exact string match alone.",
            "",
            "Before allocating a GPU, verify all three local chat templates and sequence lengths with:",
            "",
            "```bash",
            ".venv-model-eval/bin/python scripts/preflight_sft_dataset.py",
            "```",
            "",
            "The verified limits are 4,096 tokens for both Qwen checkpoints and 5,120 for DeepSeek. DeepSeek's tokenizer "
            "expands 82 base examples beyond 4,096 tokens (maximum 4,609), so those rows must be length-bucketed or run "
            "with batch size 1 rather than silently truncated.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    train_path = args.train.resolve()
    validation_path = args.validation.resolve()
    policy_path = args.policy.resolve()
    output_dir = args.output_dir.resolve()
    policy = read_json(policy_path)
    validate_policy(policy)
    train = read_jsonl(train_path)
    validation = read_jsonl(validation_path)
    validate_source_rows(train, "train")
    validate_source_rows(validation, "validation")
    leakage = validate_split_isolation(train, validation)

    selected, audit = select_supplement(train, policy)
    by_id = {str(row["id"]): row for row in train}
    selected_set = set(selected)
    base_records = [sft_record(row, audit[str(row["id"])], "base", 1) for row in train]
    supplemental_records = [
        sft_record(by_id[row_id], audit[row_id], "hard_supplement", 2) for row_id in selected
    ]
    curriculum_records = [*base_records, *supplemental_records]
    validation_records = [validation_record(row) for row in validation]
    sampling_rows = [audit[str(row["id"])] for row in train]

    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {
        "base_train": write_jsonl(output_dir / "train_base.jsonl", base_records),
        "hard_supplement": len(supplemental_records),
        "curriculum_train": write_jsonl(output_dir / "train_curriculum.jsonl", curriculum_records),
        "validation": write_jsonl(output_dir / "validation.jsonl", validation_records),
        "sampling_weights": write_jsonl(output_dir / "sampling_weights.jsonl", sampling_rows),
    }
    base_occurrences = [str(row["id"]) for row in train]
    curriculum_occurrences = [*base_occurrences, *selected]
    if set(base_occurrences) != {str(record["source_id"]) for record in curriculum_records}:
        raise AssertionError("Curriculum lost one or more base source IDs")
    if len(selected) != len(selected_set):
        raise AssertionError("Supplement unexpectedly contains duplicate source IDs")

    manifest = {
        "package": "spider_sft_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "train": {"path": str(train_path), "rows": len(train), "sha256": sha256_file(train_path)},
            "validation": {
                "path": str(validation_path),
                "rows": len(validation),
                "sha256": sha256_file(validation_path),
            },
            "policy": {"path": str(policy_path), "sha256": sha256_file(policy_path)},
        },
        "policy": policy,
        "counts": counts,
        "leakage_checks": leakage,
        "distributions": {
            "base_train": distribution(train, base_occurrences),
            "curriculum_train": distribution(train, curriculum_occurrences),
            "validation": distribution(validation, [str(row["id"]) for row in validation]),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    write_text(output_dir / "README.md", render_readme(manifest))
    artifact_files = (
        "train_base.jsonl",
        "train_curriculum.jsonl",
        "validation.jsonl",
        "sampling_weights.jsonl",
        "manifest.json",
        "README.md",
    )
    checksums = {name: sha256_file(output_dir / name) for name in artifact_files}
    write_json(output_dir / "checksums.json", checksums)

    print(f"Built Spider SFT package: {output_dir}")
    print(f"Base train: {counts['base_train']}")
    print(f"Hard supplement: {counts['hard_supplement']}")
    print(f"Curriculum train: {counts['curriculum_train']}")
    print(f"Validation: {counts['validation']}")
    print("Leakage checks: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
