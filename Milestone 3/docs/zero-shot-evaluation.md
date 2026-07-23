# Zero-shot text-to-SQL evaluation

This workflow evaluates the three untouched instruction checkpoints on the
1,034-example Spider validation split. It does not train, quantize, or modify
the models.

## 1. Download and pin the model snapshots locally

```bash
./scripts/download_eval_models.sh
```

The downloader is resumable. It writes normal model directories under
`models/text2sql-eval/` and records repository commit hashes, file sizes, and
SHA-256 checksums in `models/text2sql-eval/download_manifest.json`.

The repositories are public. No Hugging Face credential is normally required.
If anonymous requests are rate-limited, authenticate directly in the terminal
with a read-only token; never put a token in a script or committed file.

## 2. Run on one Colab L4

Run a small end-to-end pilot first:

```bash
bash scripts/run_colab_zero_shot_eval.sh --limit 12
```

Run all three models on all 1,034 validation examples:

```bash
bash scripts/run_colab_zero_shot_eval.sh
```

The launcher requests an L4, verifies that Colab actually allocated an L4,
installs pinned evaluation dependencies, and uploads only the code, validation
rows, and required SQLite databases. The runtime fetches the exact model commit
hashes recorded by the local downloader. This avoids relaying more than 12 GiB
through the Colab CLI contents endpoint, which base64-encodes complete uploaded
files in local memory.

Models are loaded and evaluated sequentially. After every model the runner
deletes the model and remaining CUDA tensors, runs Python garbage collection,
and empties the CUDA allocator before loading the next checkpoint. If a batch
still causes an out-of-memory error, it halves that model's batch size and
retries without losing completed predictions.

The L4 is stopped automatically after results are downloaded, including after
most failures or Ctrl-C interruptions. Avoid `--keep-session` unless the live
runtime is deliberately needed for debugging.

## 3. Monitor from another terminal

The launcher prints its run directory. Pass it to:

```bash
bash scripts/monitor_colab_eval.sh artifacts/zero-shot-eval/runs/YYYYMMDD-HHMMSS
```

With no path, the monitor selects the newest run directory. It polls the remote
heartbeat every 15 seconds and shows phase, current model, completed models,
examples, batch size, and allocated VRAM. Stopping the monitor does not stop the
evaluation.

The primary launcher terminal also streams `tqdm` progress bars and writes the
same output to `orchestrator.log`.

## Outputs and metrics

Every model produces a resumable `predictions.jsonl` and `metrics.json`.
`comparison.csv` and `comparison.json` summarize:

- execution accuracy against the read-only SQLite database;
- raw, whitespace-normalized, and SQL-AST-canonical exact match;
- syntactically valid SQL rate;
- SQL-only output-format compliance;
- execution error categories;
- accuracy by the dataset's simple/moderate/complex proxy;
- generation latency, final batch size, attention implementation, dtype, and
  peak CUDA memory.

Generated SQL is permitted to execute only when it starts with `SELECT` or
`WITH`. SQLite databases are opened immutable/read-only, write actions are
denied through the SQLite authorizer, and every query has a timeout.

Execution accuracy is the primary local baseline. It is not presented as the
official Spider test-suite-accuracy implementation; the untouched predictions
are retained so that evaluator can be added later without rerunning inference.

## Evaluate a trained adapter under identical conditions

Provide exactly one pinned base slug, the downloaded adapter, and a unique
result label:

```bash
bash scripts/run_colab_zero_shot_eval.sh \
  --model qwen3-4b-instruct-2507 \
  --adapter-dir artifacts/qlora-training/runs/RUN/downloaded/output/final_adapter \
  --adapter-label qwen3-base-sft
```

The evaluator verifies that `adapter_config.json` names the expected base model
and exact pinned revision before loading it. Generation, prompts, decoding,
SQLite execution, metrics, batch handling, and automatic L4 shutdown remain the
same as the untouched baseline.

## 4. Reproduce the error analysis locally

The analyzer automatically selects the newest completed run:

```bash
.venv-model-eval/bin/python scripts/analyze_zero_shot_errors.py
```

To analyze a specific run instead:

```bash
.venv-model-eval/bin/python scripts/analyze_zero_shot_errors.py \
  --results-dir artifacts/zero-shot-eval/runs/YYYYMMDD-HHMMSS/downloaded/results
```

It does not require a GPU or rerun inference. The generated `error-analysis/`
directory contains a Markdown report, the complete machine-readable analysis,
feature/database/difficulty CSVs, representative failures, cross-model
disagreement counts, and training-feature sampling priorities.
