#!/usr/bin/env python3
"""Reproduce FINER-SQL multi-candidate generation with vLLM and local SQLite voting."""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path
import tarfile
import time
from typing import Any

from tqdm.auto import tqdm

import evaluate_text2sql_models as core


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("/content/huggingface-cache"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="finer-sql-3b-spider")
    parser.add_argument("--num-candidates", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--query-timeout", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--archive", type=Path)
    return parser.parse_args()


def download_snapshot(spec: dict[str, Any], cache_dir: Path, logger: logging.Logger) -> str:
    from huggingface_hub import snapshot_download

    revision = spec.get("revision")
    if not revision:
        raise ValueError(f"FINER model {spec.get('slug')} must have an immutable revision")
    delays = (0, 15, 45, 90)
    for attempt, delay in enumerate(delays, start=1):
        if delay:
            logger.warning("Retrying model download in %d seconds (%d/%d)", delay, attempt, len(delays))
            time.sleep(delay)
        try:
            return snapshot_download(
                repo_id=str(spec["repo_id"]),
                revision=str(revision),
                cache_dir=str(cache_dir),
                max_workers=1,
            )
        except Exception:
            if attempt == len(delays):
                raise
            logger.exception("Sequential snapshot download attempt %d failed", attempt)
    raise AssertionError("unreachable")


def extract_finer_sql(raw: str) -> str:
    """Match FINER's published extraction: only text after the final reasoning tag."""
    if "</think>" not in raw:
        return ""
    return core.extract_sql(raw.rsplit("</think>", 1)[1])


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        handle.flush()


def main() -> int:
    args = parse_args()
    if args.num_candidates < 1 or args.batch_size < 1 or args.max_new_tokens < 1:
        raise ValueError("candidate, batch, and token counts must be positive")
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("--gpu-memory-utilization must be in (0, 1)")

    args.project_root = args.project_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = core.configure_logging(args.output_dir)
    status_path = args.output_dir.parent / "status.json"
    examples = core.read_jsonl(args.data.resolve())
    if args.limit is not None:
        examples = examples[: args.limit]
    specs = core.load_specs(args.config.resolve(), args.manifest.resolve(), {args.model})
    spec = specs[0]
    result_dir = args.output_dir / str(spec["slug"])
    result_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = result_dir / "predictions.jsonl"
    completed = core.read_jsonl(prediction_path) if prediction_path.exists() else []
    completed_ids = {str(row["id"]) for row in completed}
    pending = [row for row in examples if str(row["id"]) not in completed_ids]

    core.update_status(
        status_path,
        phase="downloading_model",
        engine="vllm",
        current_model=spec["slug"],
        completed_examples=len(completed),
        total_examples=len(examples),
        num_candidates=args.num_candidates,
    )
    model_path = download_snapshot(spec, args.cache_dir.resolve(), logger)

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    prompts = [core.render_prompt(tokenizer, row) for row in pending]
    maximum_prompt = max((len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) for prompt in prompts), default=0)
    if maximum_prompt + args.max_new_tokens > args.max_model_len:
        logger.warning(
            "Requested prompt+generation can exceed max_model_len (%d + %d > %d); vLLM will stop at context capacity.",
            maximum_prompt,
            args.max_new_tokens,
            args.max_model_len,
        )

    core.update_status(status_path, phase="loading_model", max_prompt_tokens=maximum_prompt)
    llm = LLM(
        model=model_path,
        dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
        trust_remote_code=bool(spec.get("trust_remote_code", False)),
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )
    sampling = SamplingParams(
        n=args.num_candidates,
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        stop=["<|endoftext|>"],
        seed=args.seed,
    )

    gold_cache: dict[tuple[str, str, bool], core.QueryResult] = {}
    progress = tqdm(total=len(examples), initial=len(completed), desc=str(spec["slug"]), unit="example")
    started = time.monotonic()
    for offset in range(0, len(pending), args.batch_size):
        batch_rows = pending[offset : offset + args.batch_size]
        batch_prompts = prompts[offset : offset + args.batch_size]
        generation_started = time.monotonic()
        outputs = llm.generate(batch_prompts, sampling, use_tqdm=False)
        generation_ms = (time.monotonic() - generation_started) * 1000
        records: list[dict[str, Any]] = []
        for row, request_output in zip(batch_rows, outputs, strict=True):
            candidate_rows: list[dict[str, Any]] = []
            candidate_results: list[core.QueryResult] = []
            for index, candidate in enumerate(request_output.outputs):
                raw = candidate.text
                sql = extract_finer_sql(raw)
                matched, _gold, result = core.execution_match(
                    args.project_root, row, sql, args.query_timeout, gold_cache
                )
                candidate_results.append(result)
                candidate_rows.append(
                    {
                        "index": index,
                        "raw_prediction": raw,
                        "predicted_sql": sql,
                        "output_tokens": len(candidate.token_ids),
                        "execution_status": result.status,
                        "execution_error": result.error,
                        "execution_match": matched,
                    }
                )
            selected_index, votes = core.select_value_aware_voting(candidate_results)
            selected = candidate_rows[selected_index]
            predicted_sql = str(selected["predicted_sql"])
            canonical, syntax_error = core.canonical_sql(predicted_sql)
            gold_canonical, gold_syntax_error = core.canonical_sql(str(row["sql"]))
            matched, gold_result, prediction_result = core.execution_match(
                args.project_root, row, predicted_sql, args.query_timeout, gold_cache
            )
            records.append(
                {
                    "id": row["id"],
                    "db_id": row["db_id"],
                    "complexity": row.get("metadata", {}).get("query_features", {}).get("complexity_proxy", "unknown"),
                    "question": row["question"],
                    "gold_sql": row["sql"],
                    "raw_prediction": selected["raw_prediction"],
                    "predicted_sql": predicted_sql,
                    "raw_exact_match": str(selected["raw_prediction"]).strip() == str(row["sql"]).strip(),
                    "normalized_exact_match": core.normalize_sql(predicted_sql) == core.normalize_sql(str(row["sql"])),
                    "canonical_exact_match": canonical is not None and canonical == gold_canonical,
                    "syntax_valid": canonical is not None,
                    "syntax_error": syntax_error,
                    "gold_syntax_error": gold_syntax_error,
                    "execution_match": matched,
                    "gold_execution_status": gold_result.status,
                    "prediction_execution_status": prediction_result.status,
                    "prediction_execution_error": prediction_result.error,
                    "format_compliant": bool(predicted_sql),
                    "input_tokens": len(tokenizer(batch_prompts[len(records)], add_special_tokens=False)["input_ids"]),
                    "input_truncated": False,
                    "output_tokens": selected["output_tokens"],
                    "generation_ms_per_example": round(generation_ms / len(batch_rows), 3),
                    "num_candidates": args.num_candidates,
                    "selected_candidate_index": selected_index,
                    "execution_consensus_votes": votes,
                    "candidate_selection": "value-aware-voting",
                    "candidate_oracle_match": any(item["execution_match"] for item in candidate_rows),
                    "candidates": candidate_rows,
                }
            )
        append_rows(prediction_path, records)
        completed.extend(records)
        progress.update(len(records))
        core.update_status(
            status_path,
            phase="evaluating",
            completed_examples=len(completed),
            total_examples=len(examples),
            elapsed_seconds=round(time.monotonic() - started, 1),
        )
    progress.close()

    metrics = core.summarize(completed)
    metrics.update(
        {
            "slug": spec["slug"],
            "repo_id": spec["repo_id"],
            "revision": spec["revision"],
            "engine": "vllm",
            "num_candidates": args.num_candidates,
            "temperature": args.temperature,
            "candidate_selection": "value-aware-voting",
            "max_model_len": args.max_model_len,
        }
    )
    (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    core.write_comparison(args.output_dir, [metrics])
    (args.output_dir / "comparison.json").write_text(json.dumps([metrics], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    core.update_status(
        status_path,
        phase="complete",
        current_model=None,
        completed_models=1,
        model_count=1,
        metrics=[metrics],
    )

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    if args.archive is not None:
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.archive, "w:gz") as archive:
            for path in sorted(args.output_dir.rglob("*")):
                archive.add(path, arcname=path.relative_to(args.output_dir), recursive=False)
    logger.info("FINER vLLM complete: execution %.3f%% oracle %.3f%%", metrics["execution_match_pct"], metrics["candidate_oracle_match_pct"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
