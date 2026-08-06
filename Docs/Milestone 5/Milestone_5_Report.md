# Milestone 5 Report — Text-to-SQL Evaluation

**Group 10**

| # | Team Member | Roll Number |
|---|-------------|-------------|
| 1 | Siddhant Hitesh Mantri | 21f3002218 |
| 2 | Anirudh Komanduri | 22f1000522 |
| 3 | Vishal S | 23f2003089 |
| 4 | Walunila Aier | 21f3002564 |
| 5 | Sambhav Jha | 22f3003227 |

---

## 1. Introduction

**Project objective.** Build a Text-to-SQL system that translates a natural-language
question plus a database schema into a correct, read-only SQLite query, under an explicit
hardware/size constraint (single NVIDIA L4 GPU, ≤4B-parameter models).

**Scope of this evaluation.** Milestone 5 evaluates the *frozen*
`Qwen/Qwen3-4B-Instruct-2507` QLoRA system with an M-Schema prompt on the Spider 1.0
validation split. No hyperparameters were tuned on Spider validation for this report;
all metrics are taken from the project evidence package produced by the existing
evaluation harness.

**Models evaluated.**

- **Primary configuration:** Qwen3-4B-Instruct-2507 + QLoRA adapter + M-Schema prompt.
- **Ablation variant:** same adapter with a DDL prompt instead of M-Schema.
- **Baselines:** Qwen3-4B zero-shot (DDL), Qwen2.5-Coder-1.5B zero-shot,
  DeepSeek-Coder-1.3B zero-shot.
- **External reference systems:** XiYanSQL-3B, FINER-SQL-3B, and multi-model
  consensus/fallback configurations.

**Objectives of Milestone 5.**

1. Produce a defensible execution-accuracy measurement of the frozen system on a
   held-out-from-training split.
2. Quantify the effect of the M-Schema prompt design choice against a DDL baseline.
3. Characterise residual failure modes (error analysis) to guide future work.
4. Report accuracy stratified by query complexity and SQL feature, not just a single
   aggregate number.

> **Note on the status of the split.** Spider validation databases are disjoint from
> training, but the split was already used for architecture and prompt decisions in
> Milestone 3. Scores here therefore support **development analysis**, not an unbiased
> final claim. A database-disjoint holdout never used in Milestones 3–4, and/or BIRD,
> was **not** evaluated in this milestone.

### Executive summary of results

On Spider 1.0 validation (1,034 examples, 20 databases), the primary configuration
reaches **78.6%** strict execution accuracy (813/1,034), **83.2%** compatible execution
accuracy (860/1,034), and **100%** syntax validity. Replacing DDL with M-Schema on the
same adapter improves strict execution from 76.9% to 78.6% (+1.7 pp). Performance by
query type is 85.7% simple, 72.6% moderate, and 58.1% complex. Residual errors are
predominantly semantic — executable SQL with incorrect results, driven by column
linking, join structure, and nested logic rather than invalid syntax.

---

## 2. Experimental Setup

### 2.1 Evaluation dataset

**Source and size.** The evaluation set is the official Spider 1.0 validation split:
1,034 examples spanning 20 databases, all held out from training. It is sourced from
`data/raw/spider/validation.parquet` (1,034 rows), fetched from the Hugging Face
`xlangai/spider` dataset. During the Spider data-prep pipeline, every gold SQL query in
the 1,034-example validation split executed successfully against its SQLite database
(0 failures), while 3 gold SQL queries in the 7,000-example training split failed to
execute and were dropped.

**Composition.**

- **Complexity proxy buckets:** 593 simple / 336 moderate / 105 complex examples.
- **Query-structure features present:** 408 queries with joins, 159 with subqueries,
  80 with set operations, 551 aggregate queries, 493 with WHERE, 277 with GROUP BY,
  237 with ORDER BY, 189 with LIMIT, 87 with DISTINCT, 79 with HAVING, 88 with multiple
  joins, and 77 combining joins and subqueries.
- 550 unique normalized SQL statements across the 1,034 examples, with at most 4
  occurrences of any single normalized pattern.
- Strictly database-disjoint from training: 0 train/validation database overlap and
  0 exact (database, question, SQL) overlap, though 6 normalized question wordings recur
  across different, disjoint databases.
- All 166 local SQLite databases (including the 20 used for validation) pass SQLite
  integrity quick checks.

**Preprocessing / evaluation workflow.**

1. **Prompt construction** — each model uses its own native tokenizer/chat template;
   a fixed system instruction ("produce one read-only SQLite query, use only the supplied
   schema, SQL only") is combined with the database dialect, schema representation (DDL or
   M-Schema), and the question.
2. **Token-budget check** — prompts are tokenized and rejected (error, not silent
   truncation) if they exceed `--max-input-tokens` for that model.
3. **Batching** — examples are sorted/grouped by prompt token length before batched
   generation to improve throughput.
4. **Generation** — deterministic greedy decoding (`do_sample=False`), fixed seed 17,
   BF16 compute, SDPA attention with native fallback.
5. **Post-processing** — strip known reasoning artifacts, extract SQL from an optional
   Markdown fence, locate the first `SELECT`/`WITH`, keep one statement, canonicalize
   with sqlglot.
6. **Execution-based scoring** — generated SQL executes only if it starts with
   `SELECT`/`WITH`; the SQLite connection is opened read-only/immutable
   (`mode=ro&immutable=1`), an authorizer blocks mutation/DDL/ATTACH/PRAGMA/transaction
   statements, each query has a 3-second timeout, and result retrieval stops above
   100,000 rows. Correctness is computed both strictly (order-preserving) and via a
   MAC-SQL/FINER-compatible bag-semantics comparison with one global column permutation.
7. **Result persistence** — per-example predictions (raw response, extracted/canonical
   SQL, syntax/execution status, token counts, latency, GPU memory, model/adapter/decode
   metadata) are written to a resumable `predictions.jsonl`, independent of
   `metrics.json`.

The same 1,034-example set is reused across all models/prompt variants/ensembles for
direct paired comparison, with significance assessed via exact McNemar tests. Because it
is repeatedly reused for development decisions, the team explicitly treats it as
**development-set evidence**, not a held-out final test claim.

### 2.2 Evaluation environment

**Hardware.** Every evaluation runs on a single remote Google Colab NVIDIA L4 GPU with
~22.03 GiB usable VRAM, provisioned via the Colab CLI. The launcher verifies the L4
allocation and bfloat16 support before proceeding. Only one model is resident on the GPU
at a time — after each model, the runner deletes it and remaining CUDA tensors, runs
garbage collection, and empties the allocator before loading the next checkpoint, keeping
peak-memory measurements uncontaminated.

**Software stack.**

- **Framework layer:** PyTorch + Hugging Face Transformers (execution); PEFT (LoRA
  adapters); sqlglot (SQL parsing/canonicalization); SQLite (immutable read-only
  execution).
- **Pinned eval environment** (`src/scripts/colab-eval-requirements.txt`):
  `transformers==4.57.6`, `peft==0.19.1`, `accelerate>=1.10,<2`,
  `huggingface_hub>=0.34,<2`, `jinja2>=3.1,<4`, `safetensors>=0.5,<1`,
  `sentencepiece>=0.2,<1`, `sqlglot>=27,<29`, `tqdm>=4.66,<5`.
- **Observed production baseline:** PyTorch 2.11.0+cu128, Transformers 4.57.6.
- **FINER multi-candidate/vLLM ablation:** `torch==2.8.0`, `transformers==4.57.3`,
  `huggingface-hub==0.34.0`, `vllm==0.10.2`, `sqlglot>=27,<29`.
- **QLoRA training env:** `transformers==4.57.6`, `datasets==5.0.0`,
  `accelerate==1.14.0`, `peft==0.19.1`, `bitsandbytes==0.49.2`, `tensorboard==2.20.0`.
- **Local:** Python 3.10+ required; Python 3.12 used for the local pipeline.

**Compute/precision settings.** BF16 weights, deterministic greedy decoding
(`do_sample=False`), fixed seed 17, SDPA attention with documented native fallback.

**Reproducibility.** The Colab CLI creates the L4 session, installs pinned deps, uploads
only code/validation rows/required databases, streams logs, downloads results, and
terminates the runtime automatically. Model snapshots are pinned and downloaded locally
first (`scripts/download_eval_models.sh`), recording exact commit hashes, file sizes, and
SHA-256 checksums. Each run's `run_config.json`/logs capture the exact GPU name/VRAM,
`torch.__version__`, and `transformers.__version__`. CUDA is a hard requirement; a
`colab_smoke_test.py` script fingerprints the environment before any real run; automatic
shutdown is verified after each run. Persisted metrics (e.g.
`evidence/baseline/qwen3_metrics.json`) record attention, dtype, cuda_peak, final batch
size, repo_id/revision, and timing.

**Reproduction commands.** From the top-level README: `bash scripts/download_eval_models.sh`,
then `bash scripts/run_colab_zero_shot_eval.sh`, or evaluate a trained adapter with
`--adapter-dir` / `--adapter-label` under identical conditions.

---

## 3. Model Training Summary

The primary adapter is trained with **QLoRA** on the Qwen3-4B-Instruct-2507 base. The
configuration below is the finalist selected by the hyperparameter search
(`configs/hparam/qwen3/selected-full5996.json`, deriving from
`configs/text2sql_qlora_training.json`).

**Base model & architecture.**

- Base: `Qwen/Qwen3-4B-Instruct-2507` (decoder-only causal LM).
- Adapter: QLoRA (LoRA over a 4-bit-quantized base). Primary adapter binary ≈ 132 MB.
- **Quantization:** 4-bit **NF4**, double quantization enabled, compute dtype bfloat16.

**LoRA configuration.**

| Parameter | Value |
|---|---|
| Rank `r` | 16 |
| `lora_alpha` | 32 |
| `lora_dropout` | 0.05 |
| `target_modules` | `all-linear` |
| `bias` | none |
| `task_type` | `CAUSAL_LM` |

**Optimization.**

| Setting | Value |
|---|---|
| Objective / loss | Autoregressive cross-entropy over the SQL completion (causal-LM SFT) |
| Optimizer | `paged_adamw_8bit` |
| Learning rate | **3e-4** (finalist; base config used 2e-4 — see §8) |
| LR scheduler | cosine, `warmup_ratio` = 0.03 |
| Weight decay | 0.0 |
| Max grad norm | 0.3 |
| Epochs | **1.0** |
| Seed | 17 |
| Precision | bf16 (tf32 enabled), gradient checkpointing on, `pad_to_multiple_of` = 8 |

**Batch size (Qwen3-4B finalist).** `per_device_train_batch_size` = 2 ×
`gradient_accumulation_steps` = 8 → **effective batch size 16**; `max_seq_length` = 4096.
(The base config reached the same effective batch of 16 via 1 × 16; the finalist uses
2 × 8 for higher L4 utilization at identical effective batch.)

**Checkpointing & selection.** `eval_strategy` = epoch, `save_steps` = 50,
`save_total_limit` = 2. The final adapter is selected by **strict execution accuracy** on
the tuning set (see §8), not by training loss.

**Training data.** Finalist trained on the **full 5,996-example** M-Schema training set;
the screening phase used **2,048** examples (see §8). A two-epoch stability run was
included in the search plan but the longer run did not complete.

> A separate autoregressive-ORM variant exists in the repo
> (`configs/gradesql_orm_qlora_training.json`: r=16/α=64, lr=7e-5, 2 epochs, seed 29). It
> is a distinct experiment and **not** the primary M-Schema adapter reported here.

---

## 4. Evaluation Methodology

**Evaluation protocol.** Execution-result equivalence: generated SQL is run against a
read-only SQLite database and its result set compared to the gold query's result set,
rather than comparing raw SQL text. This is justified because Text-to-SQL admits many
syntactically different but semantically equivalent queries; string-matching would
unfairly penalise correct-but-differently-phrased queries.

**Test dataset & ground truth.** Spider 1.0 validation (§2.1); gold SQL queries verified
to execute against their SQLite databases during data prep. Execution is restricted to
`SELECT`/`WITH` against an immutable read-only connection with an authorizer denying
mutations and a per-query timeout.

**Baseline models.** Qwen3-4B zero-shot (DDL), Qwen2.5-Coder-1.5B zero-shot,
DeepSeek-Coder-1.3B zero-shot; external references XiYanSQL-3B and FINER-SQL-3B (§7).

**Cross-validation.** Not applicable — a fixed official validation split is used to
preserve comparability with published Spider numbers.

**Success criteria.** The operating target is competitive strict/compatible execution
accuracy against comparable ≤4B systems (XiYanSQL-3B, FINER-SQL-3B) under the single-L4
constraint, with 100% syntax validity and residual errors concentrated in
semantic/compositional rather than syntactic failures.

---

## 5. Performance Metrics and Justification

The system is evaluated with a battery of metrics, each targeting a different aspect of
correctness, robustness, or efficiency.

**Primary — Execution accuracy.** Two variants reported side by side:

- **Strict execution accuracy:** order-preserving comparison of result rows/columns.
- **MAC-SQL/FINER-compatible execution accuracy:** bag semantics on rows with one global
  column permutation allowed, matching the published Spider execution metric used by
  external systems (MAC-SQL, FINER-SQL) so the numbers are literature-comparable.

**Secondary correctness/robustness.**

- **Syntax validity rate** — fraction of generated queries that parse as valid SQL,
  isolating grammar failures from semantic/schema-linking failures.
- **Exact-match variants** (raw, whitespace-normalized, AST-canonical) — retained as
  secondary diagnostics, explicitly *not* implying semantic accuracy.
- **Format compliance** — whether the model outputs SQL only (no fences/explanations),
  since malformed output can break downstream parsing.
- **Execution error categories** (unknown-column, non-executable, etc.) — for qualitative
  error analysis.

**Sliced/stratified metrics.** Execution accuracy is broken down by difficulty slice
(simple/moderate/complex), by SQL feature (joins, multi-joins, GROUP BY, HAVING,
subqueries, set operations, join+subquery), and per-database — because a single aggregate
can mask systematic weaknesses (e.g. the large drop on multi-join and join+subquery
cases).

**Efficiency/resource.** Mean generation latency per example and peak allocated GPU VRAM,
because the project operates under an explicit hardware/size constraint and must be judged
on the accuracy–efficiency tradeoff.

**Ensemble/consensus.** Execution consensus accuracy (majority vote), model/candidate
oracle accuracy (upper bound), and the paired **exact McNemar** significance test —
appropriate because comparisons are paired (same 1,034 examples), unlike an unpaired test.

---

## 6. Experimental Results

### 6.1 Primary configuration (Spider validation, n=1,034)

| Metric | Value |
|---|---|
| Strict EX | 78.6% (813 / 1,034) |
| Compatible EX | 83.2% (860 / 1,034) |
| Syntax validity | 100% |
| Normalized EM | 48.7% |

Compatible EX exceeds strict EX by 4.6 pp because it uses bag semantics and allows one
global column permutation. Normalized EM remains far below execution accuracy, confirming
that many correct answers differ in surface SQL form.

### 6.2 Comparison with baselines and external systems

The primary greedy single-model system is competitive with XiYanSQL-3B. Ensemble and
multi-candidate methods improve accuracy at higher latency and are treated as optional
operating modes.

<img width="1189" height="657" alt="image" src="https://github.com/user-attachments/assets/3bd8d504-7801-42ec-a4df-d8e5cf68b014" />


| System | Strict EX | Compatible EX |
|---|---|---|
| Qwen3-4B zero-shot (DDL) | 72.3% | — |
| Qwen2.5-Coder-1.5B zero-shot | 56.6% | — |
| DeepSeek-Coder-1.3B zero-shot | 47.3% | — |
| Qwen3-4B QLoRA + DDL | 76.9% | — |
| **Qwen3-4B QLoRA + M-Schema (primary)** | **78.6%** | **83.2%** |
| XiYanSQL-3B (M-Schema) | 78.4% | 83.3% |
| FINER-SQL-3B (30 candidates) | 79.0% | 84.2% |
| Five-model consensus | 82.8% | 87.3% |
| Strict FINER fallback | 83.1% | 87.3% |

### 6.3 Performance by query complexity

M-Schema improves every slice, with the largest absolute lift on complex queries. Complex
accuracy remains the limiting factor for deployment quality.

<img width="1077" height="625" alt="image" src="https://github.com/user-attachments/assets/d2755d34-3ac9-4085-a077-9c2174d7b20c" />


| Query type | N | Zero-shot | QLoRA + DDL | QLoRA + M-Schema | Δ (DDL→M-Schema) |
|---|---|---|---|---|---|
| Simple | 593 | 82.1% | 84.5% | 85.7% | +1.2 pp |
| Moderate | 336 | 61.9% | 71.1% | 72.6% | +1.5 pp |
| Complex | 105 | 50.5% | 52.4% | 58.1% | +5.7 pp |

### 6.4 Latency and memory

For the full-data greedy evaluation on the internal tuning set: mean generation latency
≈ **1,047 ms/example**; peak allocated VRAM ≈ **10.7 GiB** on an NVIDIA L4. The primary
science adapter binary is ≈ **132 MB** (base weights loaded from Hugging Face).

---

## 7. Baseline Comparison

The comparison table and chart in §6.2 give the full picture; the key deltas:

- **QLoRA vs zero-shot:** fine-tuning lifts Qwen3-4B strict EX from **72.3% → 76.9%**
  (DDL) and to **78.6%** with M-Schema — a **+6.3 pp** gain over the untuned base.
- **vs comparable ≤4B systems:** the primary single-model greedy system (78.6%) is on
  par with **XiYanSQL-3B** (78.4%) and within ~0.4 pp of **FINER-SQL-3B** (79.0%), which
  uses 30 sampled candidates.
- **vs ensembles:** five-model consensus (82.8%) and strict FINER fallback (83.1%)
  set the internal ceiling, at materially higher latency/cost — retained as optional
  operating modes rather than the primary claim.

---

## 8. Hyperparameter Analysis

Hyperparameters were selected in Milestone 4 via a staged **successive-halving** search
on Qwen3-4B (`configs/qwen3_hparam_search_plan.json`), then locked for this milestone.

**Methodology.**

- **Selection metric:** strict execution accuracy. Secondary: compatible EX, syntax
  validity, complexity-slice accuracy, inference latency.
- **Budgeting (successive halving):** screen on **2,048** train / **1,001** validation
  examples; promote finalists to the **full 5,996** train set. Full M-Schema trials were
  projected to take hours per configuration, motivating the screen-then-promote design.
- **Screening batching:** `per_device_train_batch_size` = 2 ×
  `gradient_accumulation_steps` = 8 → effective batch 16 (same effective batch as the
  baseline, higher L4 utilization).
- **Parallelism:** an authorized cap of 6 concurrent sessions, but live allocation allowed
  only 2 simultaneous L4 sessions, so trials ran in two-session waves.

**Search waves (each wave fixes the winner of the previous).**

| Wave | Parameter(s) explored | Grid |
|---|---|---|
| 1 — Learning rate | `learning_rate` | {5e-5, 1e-4, 2e-4, **3e-4**} |
| 2 — Capacity & regularization | rank/alpha; dropout; target profile | rank/α ∈ {(8,16), **(16,32)**, (32,64)}; dropout ∈ {0.0, **0.05**, 0.1}; target ∈ {**all-linear**, attention-only} |
| 3 — Optimization & stability | scheduler; warmup; epochs; seed | scheduler ∈ {**cosine**, linear}; warmup ∈ {**0.03**, 0.1}; epochs ∈ {**1.0**, 2.0}; seed ∈ {**17**, 29, 41} |
| 4 — Decoding | candidates; temperature; selection | candidates ∈ {**1 (greedy)**, 4}; T ∈ {0.2, 0.5, 0.8, 1.0} (4-cand only); top_p 0.95; selection ∈ {execution-consensus, value-aware-voting} |

**Final selection** (`selected-full5996`): LoRA r=16 / α=32 / dropout=0.05, `all-linear`;
learning rate **3e-4**, cosine schedule, warmup 0.03; 1 epoch; seed 17; effective batch
16; greedy decoding as the deterministic primary. Per-trial result tables live in the
Milestone 4 evidence (`hparam/final_selection.json`, `evidence/milestone4/`).

**Data-scale sensitivity (observed).** On the internal set, the **2,048-row screening
adapter** outperforms the **5,996-row full-data checkpoint** by ~1.0 strict point.
Candidate explanations (distribution shift with more data, insufficient one-epoch budget
at larger scale, hyperparameters selected on the screen, seed-scale variation) are noted
but **not** isolated experimentally — see Limitations.

---

## 9. Ablation Study — Prompt Format (DDL vs M-Schema)

Identical natural-distribution QLoRA adapter; prompt format only.

<img width="1239" height="546" alt="image" src="https://github.com/user-attachments/assets/034dec58-b506-4c6c-8e49-9dcedc403b29" />


| Prompt | Strict EX | Normalized EM | Syntax |
|---|---|---|---|
| DDL | 76.9% (795/1,034) | 50.2% | 100% |
| M-Schema | 78.6% (813/1,034) | 48.7% | 100% |

Paired outcomes: **59 corrected, 41 regressed, net +18 (+1.7 pp)**; exact McNemar
**p ≈ 0.09**. The gain supports the M-Schema design choice; significance is borderline and
is reported as such.

> This is the primary ablation axis. The zero-shot vs QLoRA rows in §6.2 additionally
> isolate the contribution of fine-tuning at fixed prompt.

---

## 10. Error Analysis

Qualitative error analysis uses `src/scripts/analyze_zero_shot_errors.py`, which generates
per-example gold-vs-predicted comparisons for each model tier (strong: Qwen3-4B-Instruct;
middle: Qwen2.5-Coder-1.5B; weak: DeepSeek-Coder-1.3B). Representative failures are sampled
for the strongest model, diversified by complexity, outcome type, and structural-mismatch
category.

### 10.1 Outcome mix (zero-shot Qwen3, n=1,034)

Semantic mistakes dominate. Syntax validity is effectively saturated (~100%); format
compliance for Qwen3 is 99.9%. Most errors are **executable but wrong result**
(251 of 286 total errors), not outright execution errors (35) — confirming failures are
semantic/structural rather than syntactic.

<img width="603" height="630" alt="image" src="https://github.com/user-attachments/assets/2d66bf2a-03f6-44e1-8769-26a6269aea8f" />


| Outcome | Count | Share |
|---|---|---|
| Correct (strict EX) | 748 | 72.3% |
| Executable, wrong result | 251 | 24.3% |
| Database execution error | 35 | 3.4% |

### 10.2 Structural mismatches

Leading mismatch tags (overlapping heuristics) for zero-shot Qwen3. Among runtime
failures, unknown-column (24) is the most frequent.

<img width="1050" height="601" alt="image" src="https://github.com/user-attachments/assets/4636cd15-2d6e-43de-b535-5228e8a8b7e2" />



| Issue | Count | % of examples |
|---|---|---|
| Column reference mismatch | 191 | 18.5% |
| Literal / value mismatch | 99 | 9.6% |
| Join structure mismatch | 99 | 9.6% |
| Table selection mismatch | 88 | 8.5% |
| Projection count mismatch | 86 | 8.3% |
| Aggregation mismatch | 47 | 4.5% |
| Subquery structure mismatch | 46 | 4.4% |

### 10.3 Feature-conditioned accuracy

Accuracy falls as compositional complexity rises; **join+subquery** and long gold queries
are the weakest regimes. Lowest database-level scores include `car_1` (44.6%) and
`wta_1` (45.2%).

<img width="1049" height="601" alt="image" src="https://github.com/user-attachments/assets/fd90f98d-004f-4157-b526-4ed09664e8cd" />



| Feature | Support | Strict EX |
|---|---|---|
| Any join | 408 | 62.5% |
| GROUP BY | 277 | 57.8% |
| Subquery | 159 | 58.5% |
| Two or more joins | 88 | 53.4% |
| Join and subquery | 77 | 44.2% |
| Gold SQL ≥ 31 tokens | 67 | 43.3% |

### 10.4 Representative failure cases (Qwen3-4B-Instruct baseline)

- **spider-validation-00066 (pets_1 / complex)** — *"first name of every student who has a
  dog but does not have a cat?"* Gold selects `T1.fname, T1.age` via `NOT IN`; the model
  correctly implements the logic with `NOT EXISTS` but drops the `age` column and adds
  `DISTINCT`, triggering `projection_count_mismatch` and `distinct_mismatch` despite
  arguably correct semantics.
- **spider-validation-00175/00176 (car_1 / complex)** — makers producing ≥2 models and
  >3 cars. Gold uses `INTERSECT` of two grouped queries; the model nests a `HAVING`
  subquery, causing `join_structure_mismatch` and `set_operation_mismatch`.
- **spider-validation-00700 (voter_1 / complex)** — area codes voting for both X and Y.
  Gold uses `INTERSECT`; prediction uses two `IN` subqueries — semantically similar but
  flagged `set_operation_mismatch`.
- **spider-validation-00757 (world_1 / moderate)** — language used by the largest number
  of Asian nations. Model orders by `SUM(percentage)` instead of `COUNT(*)`: an
  `aggregate_misuse` error.
- **spider-validation-00177/00178 (car_1 / complex)** — countries with >3 car makers OR
  producing 'fiat'. Model substantially restructures with unrelated joins/`LIKE` clauses,
  producing multiple structural mismatches.

**Root cause.** After QLoRA + M-Schema, overall strict EX rises 72.3% → 78.6% and complex
EX 50.5% → 58.1%, but residual risk stays concentrated in the same compositional regimes:
column linking, join structure, set operations, and nested logic.

### 10.5 Invalid, ambiguous, and out-of-domain behaviour

| Category | Finding |
|---|---|
| Unsafe SQL | Non-`SELECT`/`WITH` queries rejected; databases open read-only with an authorizer |
| Invalid SQL | Syntax validity ≈100% on primary runs; remaining failures are mostly wrong-but-running queries |
| Ambiguity | Execution match does not guarantee alignment with every reasonable reading of an ambiguous question |
| Out-of-domain / new holdout / BIRD | Not evaluated in this milestone |

---

## 11. Model Robustness

- **Generalization within Spider:** the split is database-disjoint from training
  (0 database overlap), so 78.6% reflects performance on unseen *schemas*, not just unseen
  questions.
- **Edge cases / degradation:** accuracy degrades predictably with compositional
  complexity (§10.3) — join+subquery (44.2%) and long gold queries (43.3%) are the
  stress points.
- **Ambiguity tolerance:** execution-match cannot guarantee alignment with every
  reasonable reading of an ambiguous question (§10.5).
- **Not established (stated honestly):** cross-benchmark generalization (BIRD, enterprise
  schemas, other SQL dialects), noise tolerance, and adversarial robustness were not
  tested this milestone.

---

## 12. Computational Performance

| Aspect | Value |
|---|---|
| Inference latency (mean) | ≈ 1,047 ms/example (greedy, full-data adapter, internal set) |
| Peak allocated VRAM | ≈ 10.7 GiB (of ~22.03 GiB usable on L4) |
| Adapter size | ≈ 132 MB (base weights loaded separately) |
| Compute precision | BF16, SDPA attention |
| Hardware | Single NVIDIA L4 GPU |
| Training cost profile | Full-data M-Schema trials ≈ hours per configuration (per search-plan step timing); screening at 2,048 examples to bound cost |
| Multi-candidate mode | ~2.5× latency for ~+1 pp strict (internal M4) — optional |

*Exact training wall-clock, GPU-utilization %, and steady-state throughput
(examples/sec) are not recorded in the evidence bundle; report them from training logs if
the examiner requires precise figures.*

---

## 13. Limitations

- **No untouched holdout or BIRD evaluation.** A split never used for M3–4 decisions is
  still required for an unbiased final accuracy claim.
- **One epoch is not shown to be optimal.** The two-epoch screening run did not complete;
  absence of a finished longer run is not evidence against additional epochs.
- **Single-epoch loss curves are only weakly diagnostic.** Loss declines early then
  plateaus; end-of-run eval loss remains higher than late training loss, but execution
  accuracy was not tracked across multiple epochs.
- **Screening vs full-data gap.** The 2,048-row screen exceeds the 5,996-row full-data
  adapter by ~1.0 strict point on the internal set; the cause is not isolated
  experimentally.
- **Metric mismatch.** Strict EX, compatible EX, and normalized EM answer different
  questions and must not be quoted interchangeably.
- **Package limits.** Fine-tuned prediction files are not in the public evidence bundle;
  §10 combines baseline prediction-level analysis with fine-tuned aggregate metrics.
- **Benchmark scope.** Results are for English Spider SQLite; transfer to BIRD, enterprise
  schemas, or other SQL dialects is not established.
- **Safety vs authorization.** Read-only execution prevents mutation; it does not
  implement application-level access control or privacy policy for schema sample values.

---

## 14. Possible Improvements

| Priority | Action |
|---|---|
| 1 | Evaluate once on a frozen database-disjoint holdout and/or BIRD without further tuning |
| 2 | Complete a multi-epoch or early-stop study scored by execution accuracy |
| 3 | Retain simple/moderate/complex reporting in all subsequent evaluations |
| 4 | Target residual join, aggregation, and nested-query failures in data or decoding policy |
| 5 | Use multi-candidate execution consensus only where latency budget allows (~+1 pp strict on internal M4 tests at ~2.5× cost) |
| 6 | Re-validate hyperparameters at full-data scale |
| 7 | Surface SQL and clarification in the product UI for ambiguous questions |

---

## 15. Discussion

**Were the objectives achieved?** Yes for the milestone's stated scope: a defensible,
reproducible execution-accuracy measurement of the frozen system (78.6% strict / 83.2%
compatible), a quantified prompt-design result (M-Schema > DDL, +1.7 pp), and a
structured error analysis. The one objective deliberately deferred is the unbiased final
claim, which requires a held-out/BIRD run.

**Comparison with expectations.** The primary single-model system lands where a
well-tuned ≤4B system should — level with XiYanSQL-3B and within a fraction of a point of
a 30-candidate FINER system — without paying that system's sampling cost. The complex-query
lift from M-Schema (+5.7 pp) is larger than on simple/moderate, matching the hypothesis
that richer schema representation helps most when schema linking is hardest.

**Practical applicability.** Under the single-L4, ≤4B constraint, the system is
deployable as a read-only query assistant with strong safety guarantees (immutable
connection, authorizer, timeouts) and ~1 s/query latency. The limiting factor for
production quality is complex-query accuracy (58.1%), concentrated in compositional SQL.

**Key observations / lessons learned.**

- Execution accuracy, not exact match, is the metric that matters — normalized EM (48.7%)
  badly understates a 78.6%-correct system.
- Almost all failures are *semantic*, not syntactic (100% syntax validity), so future
  gains come from schema linking and compositional reasoning, not grammar.
- Reusing the validation split for development decisions is the single biggest threat to
  the headline number's interpretation, and is called out rather than hidden.

---

## 16. Conclusion

- The primary Qwen3-4B QLoRA + M-Schema system achieves **78.6%** strict and **83.2%**
  compatible execution accuracy on Spider validation with **100%** syntax validity.
- M-Schema improves over DDL by **+1.7 pp** strict, with the largest benefit on complex
  queries.
- Errors are mainly semantic — schema linking and compositional SQL — especially on
  complex and join-plus-subquery cases.
- Spider validation supports comparative analysis but is **not** a fully untouched final
  test; holdout/BIRD evaluation remains open.
- No further tuning was performed on Spider validation for this milestone.

**Readiness.** Suitable for continued development and a controlled read-only deployment;
**not** yet ready for an unbiased final performance claim until the held-out/BIRD
evaluation is run.

---

## Appendix A — Evidence Index

| Content | Location |
|---|---|
| Headline metrics and system comparison | `evidence/results_summary.json`, `evidence/results_summary.csv` |
| M-Schema ablation and complexity | `evidence/qwen3-mschema/REPORT.md`, `comparison.json` |
| Error taxonomy and examples | `evidence/baseline/REPORT.md`, `failure_counts.csv`, `feature_accuracy.csv` |
| Milestone 4 confirmation metrics | `evidence/milestone4/evaluation/`, `hparam/final_selection.json` |
| Training / HP configs | `configs/text2sql_qlora_training.json`, `configs/hparam/qwen3/selected-full5996.json`, `configs/qwen3_hparam_search_plan.json` |
| Adapter identities | `release/final_model.json`, `release/milestone4_final_model.json` |
| Metric definitions | `evidence/README.md` |

## Appendix B — Full-data adapter (separate artifact)

The results in §6 correspond to the selected M-Schema evaluation configuration. The
separately released full-data Milestone 4 checkpoint, for comparison:

| Evaluation | N | Strict EX | Compatible EX | Syntax | Norm. EM |
|---|---|---|---|---|---|
| Internal tuning | 1,001 | 85.5% | 89.0% | 100% | 51.3% |
| Spider validation (locked; not used for HPO) | 1,034 | 77.9% | 81.2% | 99.9% | 50.4% |
| Best 2,048-row screening adapter (internal) | 1,001 | 86.5% | 89.1% | 100% | 49.6% |

On the internal set, the 2,048-row screening adapter outperforms the 5,996-row full-data
checkpoint by ~1.0 strict point. Both scores are retained; the predeclared full-data
release remains the Milestone 4 production artifact.

Full-data adapter by complexity (internal tuning):

| Query type | N | Strict EX | Compatible EX |
|---|---|---|---|
| Simple | 572 | 88.1% | 92.3% |
| Moderate | 321 | 84.1% | 87.5% |
| Complex | 108 | 75.9% | 75.9% |

On locked Spider validation, compatible EX for that adapter is 88.7% simple, 72.9%
moderate, 65.7% complex.

---

## References

1. Yu, T. et al. *Spider: A Large-Scale Human-Labeled Dataset for Complex and
   Cross-Domain Semantic Parsing and Text-to-SQL Task.* EMNLP 2018.
2. Qwen3-4B-Instruct-2507 model card. <https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507>
3. Dettmers, T. et al. *QLoRA: Efficient Finetuning of Quantized LLMs.*
4. XGenerationLab M-Schema. <https://github.com/XGenerationLab/M-Schema>
5. Group 10 Milestone 3 and Milestone 4 reports and evidence package.

---

## Signature of Approval

Siddhant Hitesh Mantri · Anirudh Komanduri · Vishal S · Walunila Aier · Sambhav Jha
