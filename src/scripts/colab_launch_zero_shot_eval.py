"""Launch the bundled evaluator inside the active Colab kernel."""

from __future__ import annotations

import json
from pathlib import Path
import runpy
import sys


ROOT = Path("/content/text2sql_eval")
config = json.loads((ROOT / "launch_config.json").read_text(encoding="utf-8"))
arguments = [
    str(ROOT / "scripts" / "evaluate_text2sql_models.py"),
    "--config", str(ROOT / "configs" / "text2sql_eval_models.json"),
    "--manifest", str(ROOT / "models" / "text2sql-eval" / "download_manifest.json"),
    "--data", str(ROOT / "data" / "processed" / "spider" / "validation.jsonl"),
    "--project-root", str(ROOT),
    "--model-root", str(ROOT / "models" / "text2sql-eval"),
    "--model-source", str(config["model_source"]),
    "--cache-dir", "/content/huggingface-cache",
    "--output-dir", str(ROOT / "results"),
    "--max-input-tokens", str(config["max_input_tokens"]),
    "--max-new-tokens", str(config["max_new_tokens"]),
    "--num-candidates", str(config.get("num_candidates", 1)),
    "--temperature", str(config.get("temperature", 0.7)),
    "--top-p", str(config.get("top_p", 0.95)),
    "--candidate-selection", str(config.get("candidate_selection", "execution-consensus")),
    "--archive", str(ROOT / "results.tar.gz"),
]
if config.get("limit") is not None:
    arguments.extend(("--limit", str(config["limit"])))
if config.get("batch_size") is not None:
    arguments.extend(("--batch-size", str(config["batch_size"])))
for model in config.get("models") or []:
    arguments.extend(("--model", str(model)))
if config.get("adapter_dir"):
    arguments.extend(("--adapter-dir", str(config["adapter_dir"])))
    arguments.extend(("--adapter-label", str(config["adapter_label"])))

sys.argv = arguments
try:
    runpy.run_path(arguments[0], run_name="__main__")
except SystemExit as exc:
    if exc.code not in (None, 0):
        raise
