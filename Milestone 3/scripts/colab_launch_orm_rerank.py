#!/usr/bin/env python3
"""Launch ORM candidate reranking inside an already-trained Colab session."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path("/content/text2sql_sft")
arguments = [
    sys.executable,
    "/content/evaluate_orm_candidate_groups.py",
    "--config",
    str(ROOT / "configs" / "text2sql_eval_models.json"),
    "--manifest",
    str(ROOT / "models" / "text2sql-eval" / "download_manifest.json"),
    "--candidate-groups",
    "/content/orm_candidate_groups.jsonl",
    "--adapter-dir",
    str(ROOT / "output" / "final_adapter"),
    "--model",
    "qwen2.5-coder-1.5b-instruct",
    "--model-source",
    "huggingface",
    "--cache-dir",
    "/content/huggingface-cache",
    "--output-dir",
    "/content/orm_rerank/results",
    "--batch-size",
    "16",
    "--max-input-tokens",
    "4096",
    "--archive",
    "/content/orm_rerank_results.tar.gz",
]
process = subprocess.Popen(
    arguments,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
assert process.stdout is not None
for line in process.stdout:
    print(line, end="", flush=True)
return_code = process.wait()
if return_code:
    raise subprocess.CalledProcessError(return_code, arguments)
