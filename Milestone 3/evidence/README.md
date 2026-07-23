# Compact experiment evidence

This directory preserves aggregate results needed to audit the report without
committing large prediction files.

| Directory | Contents |
|---|---|
| `baseline/` | Three-model zero-shot comparison, per-model metrics, and error-analysis tables |
| `qwen3-mschema/` | Selected adapter's DDL versus M-Schema comparison |
| `macsql/` | Benchmark-compatible rescoring across candidate systems |
| `finer/` | FINER 30-candidate and candidate-pool metrics |
| `ensembles/` | Consensus and fallback selector metrics |

[`results_summary.csv`](results_summary.csv) is the human-readable table used
for the Milestone 3 report. [`results_summary.json`](results_summary.json)
includes training facts, hashes, and explicit deployment decisions.

Prediction-level JSONL is excluded because it is large and reproducible from
the pinned evaluator. Paths retained inside copied metric files are
project-relative provenance labels; the referenced prediction files are not
part of this publication package.

Metrics:

- **Strict execution** compares ordered/unordered SQLite result sets under the
  project's conservative evaluator.
- **Compatible execution** additionally applies the documented
  MAC-SQL/FINER-compatible normalization.
- **Candidate oracle** asks whether any sampled candidate is correct; it is an
  upper bound, not a deployable score.

