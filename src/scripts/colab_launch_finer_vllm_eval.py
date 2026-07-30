"""Launch FINER's bundled vLLM evaluator inside the active Colab kernel."""

from __future__ import annotations

import json
from pathlib import Path
import runpy
import sys


ROOT = Path("/content/text2sql_eval")
sys.path.insert(0, str(ROOT / "scripts"))
config = json.loads((ROOT / "launch_config.json").read_text(encoding="utf-8"))
models = config.get("models") or ["finer-sql-3b-spider"]
if len(models) != 1 or models[0] != "finer-sql-3b-spider":
    raise ValueError("The vLLM launcher supports only finer-sql-3b-spider")

arguments = [
    str(ROOT / "scripts" / "evaluate_finer_vllm.py"),
    "--config", str(ROOT / "configs" / "text2sql_eval_models.json"),
    "--manifest", str(ROOT / "models" / "text2sql-eval" / "download_manifest.json"),
    "--data", str(ROOT / "data" / "processed" / "spider" / "validation.jsonl"),
    "--project-root", str(ROOT),
    "--cache-dir", "/content/huggingface-cache",
    "--output-dir", str(ROOT / "results"),
    "--model", models[0],
    "--num-candidates", str(config.get("num_candidates", 30)),
    "--temperature", str(config.get("temperature", 1.0)),
    "--max-new-tokens", str(config.get("max_new_tokens", 2048)),
    "--batch-size", str(config.get("batch_size", 8)),
    "--archive", str(ROOT / "results.tar.gz"),
]
if config.get("limit") is not None:
    arguments.extend(("--limit", str(config["limit"])))

sys.argv = arguments
try:
    runpy.run_path(arguments[0], run_name="__main__")
except SystemExit as exc:
    if exc.code not in (None, 0):
        raise
