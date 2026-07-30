# Spider chat-SFT package v1

This package preserves every execution-validated Spider training example once, then appends a deterministic hard-example supplement selected from aggregate zero-shot error categories. The official validation split is kept unchanged and no validation row or database appears in training.

## Files

| File | Rows | Purpose |
|---|---:|---|
| `train_base.jsonl` | 6997 | Natural Spider training distribution; one row per source example. |
| `train_curriculum.jsonl` | 9796 | Base rows plus the capped hard supplement; recommended first SFT input. |
| `validation.jsonl` | 1034 | Untouched official validation examples for loss/evaluation. |
| `sampling_weights.jsonl` | 6997 | Per-source feature, weight, template-frequency, and selection audit. |
| `manifest.json` | — | Checksums, provenance, policy, leakage checks, and distributions. |

All training records use the standard `messages` field with exactly one system, user, and SQL-only assistant turn. Extra audit columns can be ignored by chat-SFT trainers.

## Sampling behavior

- Seed: `17`
- Base examples retained: **6997 / 6997**
- Unique supplemental examples: **2799**
- Maximum copies of a source example: **2**
- Maximum supplemental selections per normalized SQL template: **1**
- Overlapping feature multipliers are combined with `max`, never added.

| Feature | Base count | Base % | Curriculum count | Curriculum % |
|---|---:|---:|---:|---:|
| aggregate | 3268 | 46.71% | 4702 | 48.00% |
| complex | 760 | 10.86% | 1133 | 11.57% |
| distinct | 721 | 10.30% | 1051 | 10.73% |
| group_by | 1773 | 25.34% | 2602 | 26.56% |
| having | 427 | 6.10% | 628 | 6.41% |
| join | 2770 | 39.59% | 3983 | 40.66% |
| join_and_subquery | 399 | 5.70% | 603 | 6.16% |
| limit | 1104 | 15.78% | 1585 | 16.18% |
| multi_join | 826 | 11.80% | 1228 | 12.54% |
| order_by | 1625 | 23.22% | 2324 | 23.72% |
| set_operation | 524 | 7.49% | 785 | 8.01% |
| subquery | 1017 | 14.54% | 1511 | 15.43% |
| where | 3502 | 50.05% | 4869 | 49.70% |

## Integrity checks

- Training source rows: **6997**, all executable and SQL-only.
- Validation rows: **1034**, unchanged and executable.
- Train/validation database overlap: **0**.
- Exact `(database, question, SQL)` overlap: **0**.
- Reused normalized question wording across disjoint databases: **6**.
- Curriculum unique source coverage: **6997 / 6997**.

The feature policy was selected using aggregate validation error slices. That is hyperparameter feedback, not row leakage; however, final claims should eventually be confirmed on an untouched test set.

## Loading

```python
from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files={
        "train": "data/finetuning/spider_sft_v1/train_curriculum.jsonl",
        "validation": "data/finetuning/spider_sft_v1/validation.jsonl",
    },
)
```

Use `train_base.jsonl` as the control run. This package's validation file is convenient for validation loss; use the original processed validation rows and SQLite databases for generation-time execution scoring. For model selection, track execution accuracy rather than validation loss or exact string match alone.

Before allocating a GPU, verify all three local chat templates and sequence lengths with:

```bash
.venv-model-eval/bin/python scripts/preflight_sft_dataset.py
```

The verified limits are 4,096 tokens for both Qwen checkpoints and 5,120 for DeepSeek. DeepSeek's tokenizer expands 82 base examples beyond 4,096 tokens (maximum 4,609), so those rows must be length-bucketed or run with batch size 1 rather than silently truncated.
