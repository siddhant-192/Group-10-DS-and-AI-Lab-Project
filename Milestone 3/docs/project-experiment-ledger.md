# Text-to-SQL Project Experiment Ledger

Last updated: 2026-07-20

This is the living, report-oriented record of project decisions and completed
experiments. Exact machine-readable evidence remains in the linked manifests,
metrics, predictions, and checksums; this document explains how those pieces fit
together.

## Project objective

Adapt small open-source language models (at most 4B parameters) to generate
read-only SQLite queries from a natural-language question plus the complete
database schema. The target includes simple queries and structurally difficult
queries involving multiple joins, grouping, subqueries, set operations, ordering,
and limits. The practical compute target is Google Colab, primarily an NVIDIA L4.

## Core experimental rules

- Preserve Spider's database-disjoint official train/validation split.
- Never train on validation rows or validation databases.
- Require SQL-only assistant targets without Markdown or explanation.
- Validate gold SQL against immutable, read-only SQLite databases.
- Use execution result equivalence as the primary local metric.
- Retain exact match, syntax validity, output-format compliance, latency, and
  memory as secondary diagnostics.
- Load and evaluate only one model at a time, then release GPU memory.
- Pin model revisions and retain untouched predictions so alternative evaluators
  can be applied without rerunning inference.

## Data foundation

Dataset: Spider 1.0 annotations paired with the local `milestone3/database/`
SQLite payload.

| Split | Source rows | Usable rows | Databases | Complex proxy | Notes |
|---|---:|---:|---:|---:|---|
| Train | 7,000 | 6,997 | 140 | 762 source / 760 usable | Three non-executable gold annotations excluded. |
| Validation | 1,034 | 1,034 | 20 | 105 | Official split preserved. |

Quality and isolation results:

- 166/166 local databases pass SQLite quick checks.
- Train/validation database overlap: 0.
- Exact `(database, question, SQL)` overlap: 0.
- Six normalized question wordings recur across different, disjoint databases.
- Training targets are executable, read-only SQL and contain no Markdown fences.

Evidence:

- `data/processed/spider/EDA.md`
- `data/processed/spider/eda_report.json`
- `data/processed/spider/manifest.json`
- `data/README.md`

## Selected model tiers

| Tier | Model | Pinned revision | Local size | Role |
|---|---|---|---:|---|
| Strong | Qwen/Qwen3-4B-Instruct-2507 | `cdbee75f17c01a7cc42f958dc650907174af0554` | 7.51 GiB | Strong modern model at the 4B ceiling. |
| Middle | Qwen/Qwen2.5-Coder-1.5B-Instruct | `2e1fd397ee46e1388853d2af2c993145b0f1098a` | 2.89 GiB | Smaller code-specialized comparison. |
| Weak/old | deepseek-ai/deepseek-coder-1.3b-instruct | `e063262dac8366fc1f28a4da0ff3c50ea66259ca` | 2.51 GiB | Older small-model floor. |

The public checkpoints downloaded without Hugging Face authentication. Exact file
hashes and revisions are recorded in
`models/text2sql-eval/download_manifest.json`.

## Colab execution infrastructure

The Google Colab CLI was installed in `.venv-colab-cli` and authenticated with
Application Default Credentials. The workflow can create an L4 session, install
remote dependencies, upload the evaluation bundle, stream logs and progress,
download results, and terminate the runtime automatically.

The production baseline used:

- GPU: NVIDIA L4, 22.03 GiB usable VRAM.
- PyTorch: 2.11.0+cu128.
- Transformers: 4.57.6.
- Dtype: bfloat16.
- Attention: SDPA for all three models.
- Sequential model loading with post-model CUDA allocation returning to about
  0.009 GiB.
- Automatic shutdown verified; Colab reported no active sessions afterward.

Workflow documentation: `docs/zero-shot-evaluation.md`.

## Untuned zero-shot baseline

Run: `artifacts/zero-shot-eval/runs/20260720-012118/`

| Model | Execution | Syntax valid | Canonical exact | Format compliant | Mean generation ms/example | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B-Instruct | 72.34% (748/1,034) | 100.00% | 7.93% | 99.90% | 596.25 | 9.13 GiB |
| Qwen2.5-Coder-1.5B | 56.58% (585/1,034) | 99.90% | 24.95% | 100.00% | 197.73 | 4.25 GiB |
| DeepSeek-Coder-1.3B | 47.29% (489/1,034) | 98.07% | 8.90% | 38.49% | 817.71 | 5.00 GiB |

Execution accuracy confirms the intended strong/middle/weak ranking. Qwen2.5's
higher textual/canonical exact match does not imply better semantic accuracy;
Qwen3 frequently produces differently written SQL with matching results.

Primary evidence:

- `artifacts/zero-shot-eval/runs/20260720-012118/downloaded/results/comparison.csv`
- `artifacts/zero-shot-eval/runs/20260720-012118/downloaded/results/comparison.json`
- Per-model `metrics.json` and `predictions.jsonl` files in the same directory.
- `artifacts/zero-shot-eval/runs/20260720-012118/orchestrator.log`

## Error analysis

Qwen3 has 286 execution failures: 251 executable-but-wrong queries and 35
database execution errors. Its remaining problem is predominantly schema linking
and semantic query construction rather than syntax or response formatting.

Important Qwen3 slices:

| Slice | Support | Execution accuracy |
|---|---:|---:|
| Simple | 593 | 82.12% |
| Moderate | 336 | 61.91% |
| Complex | 105 | 50.48% |
| Gold SQL over 30 tokens | 67 | 43.28% |
| Join + subquery | 77 | 44.16% |
| Two or more joins | 88 | 53.41% |
| GROUP BY | 277 | 57.76% |
| Subquery | 159 | 58.49% |

Additional observations:

- Qwen2.5 has 212 unknown-column errors, commonly caused by skipping necessary
  joins or assigning a requested field to the wrong table.
- DeepSeek returns 617 Markdown-fenced responses and 15 responses containing no
  SQL, even though SQL extraction salvages many fenced queries for execution.
- At least one model succeeds on 825/1,034 examples (79.79% three-model oracle),
  while all models fail on 209 examples.
- Qwen3's unique-normalized-SQL macro execution accuracy is 72.27%, almost equal
  to its 72.34% row-level score; paraphrased duplicate SQL is not materially
  inflating the result.
- Some Spider rows contain question/gold ambiguity. For example, a question may
  request a first name while its gold query returns first name and age. Such rows
  should be audited before being emphasized as hard training examples.

Full evidence:

- `artifacts/zero-shot-eval/runs/20260720-012118/downloaded/results/error-analysis/REPORT.md`
- `artifacts/zero-shot-eval/runs/20260720-012118/downloaded/results/error-analysis/analysis.json`
- Feature, database, difficulty, failure, and priority CSVs in that directory.

## Finalized SFT package

Location: `data/finetuning/spider_sft_v1/`

| Artifact | Rows | Purpose |
|---|---:|---|
| `train_base.jsonl` | 6,997 | Natural-distribution control. |
| `train_curriculum.jsonl` | 9,796 | Recommended training input. |
| Hard supplement within curriculum | 2,799 | Unique feature-weighted repeats. |
| `validation.jsonl` | 1,034 | Official validation content in chat-SFT form. |
| `sampling_weights.jsonl` | 6,997 | Per-source weight and selection audit. |

Curriculum policy:

- Every usable training example is retained exactly once before supplementation.
- A source row appears at most twice.
- At most one supplemental row is selected per normalized SQL template.
- Overlapping feature multipliers combine with `max`, not addition.
- Highest priority is join-plus-subquery (3.0x selection weight cap), followed by
  complex, multi-join, subquery, set-operation, GROUP BY, and DISTINCT slices.
- Validation-derived feedback is aggregate feature-level hyperparameter feedback;
  no validation content enters training. Final claims still require an untouched
  test set.

Tokenizer preflight:

| Model | Maximum full sequence | Training limit | Truncated rows |
|---|---:|---:|---:|
| Qwen3-4B | 3,173 | 4,096 | 0 |
| Qwen2.5-Coder-1.5B | 3,173 | 4,096 | 0 |
| DeepSeek-Coder-1.3B | 4,609 | 5,120 | 0 |

DeepSeek expands 82 base examples beyond 4,096 tokens. They are retained because
the checkpoint supports the required context; training must length-bucket these
rows or use batch size 1 rather than truncate them.

Evidence:

- `data/finetuning/spider_sft_v1/README.md`
- `data/finetuning/spider_sft_v1/manifest.json`
- `data/finetuning/spider_sft_v1/tokenization_report.json`
- `data/finetuning/spider_sft_v1/checksums.json`
- `configs/text2sql_sft_sampling.json`

## Metric caveats for the final report

The current execution metric compares results on the supplied read-only SQLite
instance and respects gold-query ordering when `ORDER BY` is present. It is more
semantically meaningful than string exact match, but it is not the official
Spider test-suite-accuracy implementation and can accept accidental equivalence
on a particular database instance. The saved predictions permit later official
rescoring without new inference.

## Next experimental stage

Build and smoke-test a resumable parameter-efficient SFT training harness for an
L4. The recommended sequence is:

1. Train Qwen3-4B on a very small curriculum subset to validate token masking,
   checkpointing, memory use, logging, resume, artifact download, and automatic
   session termination.
2. Run a Qwen3 base-versus-curriculum ablation so the feature-aware sampler is
   tested rather than assumed beneficial.
3. Train the middle and weak/old models with the selected curriculum recipe.
4. Evaluate every resulting adapter/checkpoint with the unchanged 1,034-example
   generation-and-execution harness.
5. Compare overall execution accuracy, hard slices, regressions, latency, and
   memory against the frozen zero-shot baselines.

Because full optimizer-state fine-tuning of a 4B checkpoint is not practical on
a single L4, the training design should use a parameter-efficient method and
retain identical evaluation conditions across models. Exact hyperparameters and
library versions must be pinned in the training manifest after the smoke test.

## QLoRA infrastructure smoke result

Run: `artifacts/qlora-training/runs/20260720-082659/`

The Qwen3 curriculum smoke test completed successfully on one NVIDIA L4. It used
64 length-stratified training examples and 16 validation examples with no
truncation. The first process trained through step 2 and saved a complete
checkpoint; a fresh process restored `checkpoint-2` and continued through step
4. The downloaded `checkpoint-4` contains adapter, optimizer, scheduler, RNG,
trainer state, tokenizer, and training arguments.

| Property | Result |
|---|---:|
| QLoRA backbone | Pinned Qwen3-4B-Instruct-2507 revision |
| Trainable LoRA parameters | 33,030,144 (0.821141% of 4,022,468,096) |
| LoRA | rank 16, alpha 32, dropout 0.05, all linear projections |
| Quantization | 4-bit NF4, BF16 compute, double quantization |
| Sequence range exercised | 146–3,173 tokens |
| Explicit assistant-only labels | Yes |
| Truncated examples | 0 |
| Resume verified | Yes, checkpoint 2 to step 4 |
| Final smoke validation loss | 1.463103 |
| Overall peak allocated VRAM | 10.591 GiB |
| Overall peak reserved VRAM | 12.617 GiB |
| Final adapter weight size | 132,187,888 bytes |
| Adapter SHA-256 | `774f975d12b076fcccacafdccbfb17f780581ff08ffb3b9504f3e46165a8d6c1` |
| Colab active sessions after collection | 0 |

The saved final adapter weights match the newest checkpoint byte-for-byte. The
adapter metadata was locally amended to pin the exact base revision because the
initial PEFT save wrote a null revision; the training script now writes this
field automatically for all subsequent runs. The initial raw transfer archive
is retained unchanged as provenance.

Validation evidence:

- `artifacts/qlora-training/runs/20260720-082659/artifact-validation.json`
- `artifacts/qlora-training/runs/20260720-082659/downloaded/output/phase_history.jsonl`
- `artifacts/qlora-training/runs/20260720-082659/downloaded/output/tokenization_summary.json`
- `artifacts/qlora-training/runs/20260720-082659/downloaded/output/checkpoint-4/trainer_state.json`
- `docs/qlora-training.md`

## Colab runtime-loss hardening

The first full Qwen3 base-control attempt
(`artifacts/qlora-training/runs/20260720-083847/`) was terminated when Colab
deleted the active runtime at approximately 10:59 PDT, about 2 hours 20 minutes
after allocation. The last independently downloaded status showed normal
training and no OOM; the local collector received a backend 404 before final or
partial remote artifacts could be packaged. The attempt is invalid and produces
no model score. Colab reported zero active sessions afterward.

The workflow was hardened before retrying:

- every complete checkpoint is atomically archived remotely;
- a concurrent local monitor downloads checkpoints during training;
- each export is validated for size, SHA-256, safe members, adapter weights,
  optimizer, scheduler, RNG, and trainer state;
- a new L4 can resume from a verified local checkpoint without Google Drive;
- checkpoint-export copies are excluded from the compact final result archive.

Hardening smoke run: `artifacts/qlora-training/runs/20260720-180631/`.
It downloaded and validated checkpoint 2 while the fresh resume phase was still
running, then downloaded checkpoint 4, collected the final adapter, validated
all artifacts, and terminated the L4. Final smoke validation loss was 1.466529;
peak allocated VRAM was 10.591 GiB; Colab again reported zero active sessions.

## Full Qwen3 base-control QLoRA result

Final run: `artifacts/qlora-training/runs/20260720-213353/`.

Colab repeatedly reset otherwise healthy L4 VMs during the full epoch. The
incremental export design converted these from experiment-ending failures into
bounded replays. The successful lineage was:

1. `20260720-181944`: trained from step 0, exported checkpoint 100, then the VM
   reset after its last visible step 184.
2. `20260720-192730`: restored checkpoint 100, exported checkpoint 200, then the
   VM reset after its last visible step 286.
3. `20260720-203050`: restored checkpoint 200, exported checkpoint 300, then the
   VM reset after its last visible step 388.
4. `20260720-213353`: restored checkpoint 300, exported checkpoints 400 and 438,
   completed full validation, downloaded final artifacts, and terminated the L4.

An intermediate attempt (`20260720-192501`) allocated and immediately released
an L4 after a one-file 206 MiB checkpoint upload received HTTP 500. Resume uploads
were then changed to checksum-verified 32 MiB multipart transfer; every subsequent
resume reassembled and verified successfully.

| Property | Result |
|---|---:|
| Dataset | 6,997-example natural-distribution base control |
| Optimizer steps | 438/438 (one epoch) |
| Final validation loss | 0.255960 |
| Final checkpoint | `checkpoint-438`, complete and resumable |
| Final adapter weight size | 132,187,888 bytes |
| Adapter SHA-256 | `5274d4c15179b195443940d92f8caacf10f99bdfca106ee24324cd44a2fbe9bb` |
| Trainable parameters | 33,030,144 (0.821141% of logical base) |
| Peak allocated VRAM | 10.638 GiB |
| Peak reserved VRAM | 17.766 GiB |
| Explicit assistant-only labels | Yes |
| Truncated examples | 0 |
| Adapter equals final checkpoint | Yes, byte-for-byte |
| Colab active sessions after collection | 0 |

Validation evidence:

- `artifacts/qlora-training/runs/20260720-213353/artifact-validation.json`
- `artifacts/qlora-training/runs/20260720-213353/downloaded/output/run_manifest.json`
- `artifacts/qlora-training/runs/20260720-213353/downloaded/output/checkpoint-438/trainer_state.json`
- `artifacts/qlora-training/runs/20260720-213353/downloaded/output/eval_results.json`

The full curriculum experiment is recorded from
`artifacts/qlora-training/runs/20260720-222755/` onward.

## Full Qwen3 curriculum QLoRA result

Final run: `artifacts/qlora-training/runs/20260721-035111/`.

The curriculum epoch also encountered periodic Colab VM resets and completed
through the verified checkpoint lineage below:

1. `20260720-222755`: trained from step 0, exported checkpoint 100, then reset
   after the last visible step 186.
2. `20260720-233410`: restored checkpoint 100, exported checkpoint 200, then
   reset after the last visible step 285.
3. `20260721-004046`: restored checkpoint 200, exported checkpoint 300, then
   reset after the last visible step 351.
4. `20260721-014423`: restored checkpoint 300, exported checkpoint 400, then
   reset after the last visible step 476.
5. `20260721-024751`: restored checkpoint 400, exported checkpoint 500, then
   reset after the last visible step 576.
6. `20260721-035111`: restored checkpoint 500, exported checkpoints 600 and
   613, completed validation, downloaded all artifacts, and terminated the L4.

| Property | Result |
|---|---:|
| Dataset | 9,796-example feature-weighted curriculum |
| Optimizer steps | 613/613 (one epoch) |
| Final validation loss | 0.267661 |
| Final checkpoint | `checkpoint-613`, complete and resumable |
| Final adapter weight size | 132,187,888 bytes |
| Adapter SHA-256 | `4bf7f03b84abfee569fca877ab9b1443012139c37bd90716bd30303eec7599da` |
| Trainable parameters | 33,030,144 (0.821141% of logical base) |
| Peak allocated VRAM | 10.655 GiB |
| Peak reserved VRAM | 12.660 GiB |
| Explicit assistant-only labels | Yes |
| Truncated examples | 0 |
| Adapter equals final checkpoint | Yes, byte-for-byte |
| Colab active sessions after collection | 0 |

Validation evidence:

- `artifacts/qlora-training/runs/20260721-035111/artifact-validation.json`
- `artifacts/qlora-training/runs/20260721-035111/downloaded/output/run_manifest.json`
- `artifacts/qlora-training/runs/20260721-035111/downloaded/output/checkpoint-613/trainer_state.json`
- `artifacts/qlora-training/runs/20260721-035111/downloaded/output/eval_results.json`

## Qwen3 SFT ablation evaluation

The untuned checkpoint, base-control adapter, and curriculum adapter were scored
with the same pinned Qwen3 revision, prompt, deterministic generation settings,
20 SQLite databases, and all 1,034 Spider validation examples.

Runs:

- Untuned: `artifacts/zero-shot-eval/runs/20260720-012118/`
- Base SFT: `artifacts/zero-shot-eval/runs/20260721-073105/`
- Curriculum SFT: `artifacts/zero-shot-eval/runs/20260721-074952/`

| Variant | Execution | Normalized exact | Canonical exact | Raw exact | Syntax | Execution errors |
|---|---:|---:|---:|---:|---:|---:|
| Untuned Qwen3 | 72.340% (748/1,034) | 6.383% | 7.930% | 0.000% | 100.000% | 35 |
| Base-control SFT | **76.886% (795/1,034)** | **50.193%** | **51.257%** | 21.857% | 100.000% | 38 |
| Curriculum SFT | 74.371% (769/1,034) | 48.259% | 49.613% | **22.631%** | 100.000% | 63 |

Paired execution comparison on identical examples:

| Comparison | Corrected | Regressed | Net | Delta | Exact McNemar p |
|---|---:|---:|---:|---:|---:|
| Base SFT vs untuned | 105 | 58 | +47 | +4.545 pp | 0.000288 |
| Curriculum SFT vs untuned | 106 | 85 | +21 | +2.031 pp | 0.147662 |
| Curriculum SFT vs base SFT | 33 | 59 | -26 | -2.515 pp | 0.008781 |

The base-control gain over zero-shot is statistically credible under the paired
exact McNemar test. The curriculum adapter is significantly worse than the base
adapter overall, and its gain over zero-shot is not significant at 0.05.

Execution by complexity proxy:

| Slice | N | Untuned | Base SFT | Curriculum SFT | Curriculum - base |
|---|---:|---:|---:|---:|---:|
| Simple | 593 | 82.125% | 84.486% | 81.788% | -2.698 pp |
| Moderate | 336 | 61.905% | 71.131% | 67.262% | -3.869 pp |
| Complex | 105 | 50.476% | 52.381% | 55.238% | +2.857 pp |

The complex-slice curriculum improvement is directionally aligned with the
sampler's intent but is not significant (11 base failures corrected versus 8
base successes regressed; p=0.648). It does not offset significant simple and
moderate regressions. Multi-join accuracy is especially weak: 53.409% untuned,
44.318% base SFT, and 39.773% curriculum SFT. Curriculum improves join-plus-
subquery by 2.597 pp over base and set operations by 1.250 pp, but both changes
are small and non-significant. The curriculum's 63 execution errors include 56
unknown-column and 5 unknown-table errors, versus 34 and 0 for base SFT,
indicating worse schema linking rather than a syntax-formatting problem.

Decision: use the natural-distribution base-control recipe as the selected Qwen3
SFT configuration. Do not propagate the current 40% supplemental curriculum
unchanged to the smaller models. A future curriculum retry should reduce the
repeat fraction and/or use a short second-stage hard-example phase with an
untouched test set for model selection.

Full paired evidence:

- `artifacts/zero-shot-eval/qwen3-sft-ablation-20260721/REPORT.md`
- `artifacts/zero-shot-eval/qwen3-sft-ablation-20260721/comparison.json`
- `artifacts/zero-shot-eval/qwen3-sft-ablation-20260721/paired_transitions.csv`
- Feature, complexity, and database slice CSVs in the same directory.

## Adapter-evaluation transport hardening

Uploading a 157 MB adapter-inclusive bundle as one Colab CLI file returned HTTP
500 (`20260721-044156`). The evaluator now uses retryable 32 MiB multipart
uploads and verifies the reassembled byte count and SHA-256 remotely. A later
Colab kernel timeout during dependency installation (`20260721-044430`) led to
bounded install retries. The current Colab image also ships torchao 0.10.0,
which PEFT 0.19.1 rejects; remote preparation now removes that incompatible
optional package before adapter loading. Finally, the orchestrator downloads and
validates `status.json` and refuses to report success unless every requested
model reaches the `complete` phase. The failed torchao attempt
(`20260721-044625`) was correctly rejected by this validator during local tests.

Both successful adapter evaluations used one L4 at a time, loaded only one
model, downloaded results, and terminated automatically. A final server query
reported no active Colab sessions.

## Smaller-model base QLoRA results (2026-07-21)

The natural-distribution 6,997-example base recipe was then applied unchanged
to the middle and weak/old controls. Both completed one 438-step epoch with
assistant-only labels and no sequence truncation.

| Property | Qwen2.5-Coder 1.5B | DeepSeek-Coder 1.3B |
|---|---:|---:|
| Final run | `20260721-083451-qwen2.5-coder-1.5b-instruct-85814` | `20260721-095647-deepseek-coder-1.3b-instruct-39608` |
| Optimizer steps | 438/438 | 438/438 |
| Final validation loss | 0.240890 | 0.196816 |
| Adapter bytes | 73,911,112 | 60,010,048 |
| Adapter SHA-256 | `64958a46f37277b62b90c2c9d4e48e9de4d9d101616ab55c76f0c9805df5ce19` | `c20ba38b1b41fec4a2ea2c2e49502ed2732db94b47c14b9a1ade5253f588834b` |
| Trainable parameters | 18,464,768 (1.196126%) | 14,991,360 (1.113381%) |
| Peak allocated VRAM | 13.909 GiB | 3.874 GiB |
| Final checkpoint | Complete `checkpoint-438` | Complete `checkpoint-438` |
| Resume verified | Not required | Yes, from verified `checkpoint-300` |
| Adapter equals checkpoint | Yes | Yes |

Qwen2.5 completed in one remote run. DeepSeek's first healthy session
(`20260721-085410-deepseek-coder-1.3b-instruct-98215`) lost its remote workspace
after the last visible step 339. Its locally verified checkpoint 300 was
uploaded in three parts, checksum-verified remotely, and resumed by the final
run. The dead allocation was explicitly stopped before the resume L4 was
created. An earlier same-second parallel-launch collision exposed a local run
directory race; run IDs now include model slug and process ID.

Validation evidence:

- `artifacts/qlora-training/runs/20260721-083451-qwen2.5-coder-1.5b-instruct-85814/artifact-validation.json`
- `artifacts/qlora-training/runs/20260721-095647-deepseek-coder-1.3b-instruct-39608/artifact-validation.json`

## Smaller-model full execution evaluation

Both adapters were scored on the same 1,034 Spider validation examples and 20
read-only SQLite databases as their zero-shot controls.

| Model | Zero-shot execution | Base-SFT execution | Delta | Corrected / regressed | Exact McNemar p |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-Coder 1.5B | 56.576% (585/1,034) | **65.571% (678/1,034)** | **+8.994 pp** | 160 / 67 | 5.91e-10 |
| DeepSeek-Coder 1.3B | 47.292% (489/1,034) | **67.602% (699/1,034)** | **+20.309 pp** | 264 / 54 | 2.32e-34 |

The gains are broad rather than confined to simple queries:

| Model | Simple delta | Moderate delta | Complex delta |
|---|---:|---:|---:|
| Qwen2.5-Coder 1.5B | +8.094 pp | +7.143 pp | **+20.000 pp** |
| DeepSeek-Coder 1.3B | +15.008 pp | **+29.762 pp** | **+20.000 pp** |

Both tuned models produced 100% syntactically valid SQL. DeepSeek's result is
especially important: one epoch moved the nominally weak/old 1.3B control above
the tuned 1.5B middle model, demonstrating that backbone age and zero-shot rank
do not predict text-to-SQL adaptation efficiency.

Runs and paired reports:

- Qwen2.5 evaluation: `artifacts/zero-shot-eval/runs/20260721-093619-26197/`
- Qwen2.5 report: `artifacts/zero-shot-eval/qwen2.5-sft-comparison-20260721/REPORT.md`
- DeepSeek evaluation: `artifacts/zero-shot-eval/runs/20260721-102939-60712/`
- DeepSeek report: `artifacts/zero-shot-eval/deepseek-sft-comparison-20260721/REPORT.md`

## SQL-specialist checkpoint evaluation

The Apache-2.0 `XGenerationLab/XiYanSQL-QwenCoder-3B-2504` specialist was pinned
at revision `b883e58ed83f74bab037d6a7b90c4b8706d357d7` and evaluated without any
project-specific training. It scored **75.822% (784/1,034)** execution accuracy,
33.849% normalized exact match, and 100% syntax validity. Complexity execution
was 82.293% simple, 70.536% moderate, and 56.190% complex. At only 3B parameters,
it is 1.064 points below the selected tuned Qwen3 adapter and is the strongest
no-training deployment candidate tested here.

Run: `artifacts/zero-shot-eval/runs/20260721-104253-69109/`.

The upstream model card reports that the 2504 series combines supervised
fine-tuning and GRPO and supports SQLite, PostgreSQL, and MySQL. It also reports
stronger Spider results with its M-Schema representation than with raw DDL, so
an M-Schema ablation is a plausible next prompt-only improvement:
https://huggingface.co/XGenerationLab/XiYanSQL-QwenCoder-3B-2504

## Five-candidate execution-consensus pilot

The evaluator now supports sampled candidate generation with selection based
only on predicted database-result agreement; gold results are never consulted
by the selector. On the first 300 aligned validation cases, five candidates
from the selected Qwen3 base-SFT adapter produced:

| Method | Execution |
|---|---:|
| Greedy Qwen3 on the same 300 | 71.000% (213/300) |
| Five-candidate execution consensus | **74.000% (222/300)** |
| Five-candidate oracle | **83.000% (249/300)** |

Consensus corrected 14 greedy failures and regressed 5 successes, a net +3.000
points; the 300-case exact McNemar p-value is 0.0636. The pilot is promising but
does not justify a full five-times-cost run by itself. The 9-point gap between
consensus and oracle instead motivates a learned selector or a targeted refiner.

Run: `artifacts/zero-shot-eval/runs/20260721-104300-69219/`.

## Four-model execution ensemble

A zero-GPU offline selector combined the full predictions from Qwen3 base SFT,
XiYanSQL 3B, DeepSeek 1.3B SFT, and Qwen2.5 1.5B SFT. It groups executable
predictions by their actual SQLite result and selects the largest agreement
cluster, breaking ties by the fixed priority order above.

| System | Execution |
|---|---:|
| Best single model (Qwen3 base SFT) | 76.886% (795/1,034) |
| Four-model execution consensus | **79.304% (820/1,034)** |
| Four-model oracle | **87.041% (900/1,034)** |

Relative to Qwen3, ensemble consensus corrected 39 examples and regressed 14,
for a significant net +25 (+2.418 pp; exact McNemar p=0.000802). Complexity
execution is 87.015% simple, 73.810% moderate, and 53.333% complex. The ensemble
is the highest measured system in this project so far, while the 7.737-point
oracle gap quantifies the opportunity for better selection/refinement.

Evidence: `artifacts/zero-shot-eval/four-model-execution-ensemble-20260721/metrics.json`.

## Execution-filtered synthetic augmentation

The 100,000-row Gretel synthetic text-to-SQL training split was audited rather
than ingested blindly. The reproducible builder rejected non-read-only targets,
broken SQLite contexts, broken target queries, duplicates, and exact normalized
question overlap with Spider train/validation. It reconstructs prompt schemas
from the SQLite catalog after executing each context, preventing INSERT data
from leaking into prompts.

| Audit outcome | Rows |
|---|---:|
| Source rows | 100,000 |
| Context + read-only target execution accepted | 67,836 |
| Broken target execution rejected | 19,392 |
| Non-read-only/missing rejected | 10,243 |
| Broken context rejected | 2,529 |
| Balanced synthetic rows selected | 5,000 |
| Spider + synthetic combined train | 11,997 |

The selected set covers all 100 domains and is nearly uniform across basic SQL,
aggregation, single join, multiple joins, subqueries, set operations, and window
functions. All sequences pass model-specific tokenizer limits: Qwen maximum
3,173/4,096 and DeepSeek maximum 4,609/5,120.

Package: `data/finetuning/spider_gretel_exec_v1/`.
Builder: `scripts/build_gretel_augmented_sft.py`.

The 750-step Qwen3 augmented-data QLoRA completed through verified checkpoint
resumes after several Colab workspace resets. The final checkpoint and adapter
match exactly, no examples were truncated, validation loss was 0.245359, and
the adapter SHA-256 is
`2a0f22fc72595488e2207adba2591744d6a45a3182c4fe0179794b12ce8f05a6`.
Its standard-prompt execution result was **74.468% (770/1,034)**, however,
which is 2.418 points below Qwen3 base SFT. It corrected 34 base-SFT failures
but regressed 59 successes (exact McNemar p=0.0124), with declines on simple,
moderate, and complex examples. The 5,000-row synthetic mix is therefore
rejected as the primary checkpoint: execution filtering alone did not prevent
negative transfer from a synthetic distribution that comprised 41.7% of the
combined training set.

M-Schema improved this augmented adapter by 1.934 points to 76.402%
(790/1,034), but it remained 2.224 points below base SFT with the same M-Schema
prompt (39 corrected, 62 regressed; p=0.0281). Adding either augmented prompt
variant as a normal fifth ensemble voter also reduced accuracy, to 80.754% for
DDL and 81.721% for M-Schema. These controls confirm that the negative result is
not merely a prompt-format mismatch.

Evidence:

- Final training run: `artifacts/qlora-training/runs/20260721-154029-qwen3-4b-instruct-2507-59689/`
- Artifact validation: `artifacts/qlora-training/runs/20260721-154029-qwen3-4b-instruct-2507-59689/artifact-validation.json`
- Full evaluation: `artifacts/zero-shot-eval/runs/20260721-160749-77558/`
- M-Schema evaluation: `artifacts/zero-shot-eval/runs/20260721-161956-85257/`
- Paired report: `artifacts/zero-shot-eval/qwen3-gretel-augmentation-comparison-20260721/REPORT.md`

## XiYan M-Schema prompt ablation

The official M-Schema representation was reconstructed directly from each
read-only SQLite database, including table/column types, primary keys, foreign
keys, and up to three bounded distinct examples per column. The English prompt
follows the upstream M-Schema template and uses a user-only chat turn as in the
XiYan quickstart. All 1,034 prompts fit safely (maximum 1,426 Qwen-family
tokens; zero above 4,096).

| XiYan 3B input | Execution | Normalized exact | Syntax |
|---|---:|---:|---:|
| Generic project prompt + DDL | 75.822% (784/1,034) | 33.849% | 100.000% |
| Official-style prompt + M-Schema | **78.433% (811/1,034)** | **49.903%** | 100.000% |

The M-Schema variant corrected 77 DDL failures and regressed 50 successes, a
significant net +27 (+2.611 pp; exact McNemar p=0.02068). Gains were +1.518 pp
simple, +5.060 pp moderate, and +0.952 pp complex. This makes the untuned XiYan
3B specialist the strongest single evaluated model, 1.547 points above the
tuned Qwen3 adapter.

Evidence:

- Run: `artifacts/zero-shot-eval/runs/20260721-111750-91573/`
- Paired report: `artifacts/zero-shot-eval/xiyan-mschema-comparison-20260721/REPORT.md`
- Rendered data: `data/processed/spider/validation_xiyan_mschema.jsonl`
- Reproducible builder: `scripts/build_xiyan_mschema_eval_data.py`
- Upstream format: https://github.com/XGenerationLab/M-Schema

## Updated M-Schema four-model ensemble

Replacing generic-DDL XiYan with the stronger M-Schema predictions and making
it the fixed tie-break priority raises execution consensus again:

| System | Execution |
|---|---:|
| Best single (XiYan 3B M-Schema) | 78.433% (811/1,034) |
| Four-model execution consensus | **81.721% (845/1,034)** |
| Four-model oracle | **88.781% (918/1,034)** |

Relative to XiYan M-Schema, consensus corrected 62 and regressed 28 examples,
for net +34 (+3.288 pp; exact McNemar p=0.000438). Complexity execution is
87.352% simple, 79.464% moderate, and 57.143% complex. Relative to the original
untuned Qwen3 checkpoint, the final ensemble gain is +9.381 points; relative to
selected Qwen3 base SFT it is +4.835 points. This is the current best system.

Evidence: `artifacts/zero-shot-eval/four-model-mschema-execution-ensemble-20260721/metrics.json`.

## Qwen3 M-Schema prompt transfer

The same M-Schema/value-aware prompt was applied to the selected Qwen3 base-SFT
adapter without additional training. It improved execution from 76.886%
(795/1,034) to **78.627% (813/1,034)**, while normalized exact match decreased
from 50.193% to 48.743%. Execution gains were broad: +1.180 pp simple, +1.488 pp moderate,
and +5.714 pp complex. The paired change corrected 59 cases and regressed 41,
for net +18 (+1.741 pp; exact McNemar p=0.0886). Although this standalone
paired result is not significant at 0.05, its complementary predictions improve
the multi-model consensus materially.

Run: `artifacts/zero-shot-eval/runs/20260721-155144-67150/`.
Paired report: `artifacts/zero-shot-eval/qwen3-base-sft-mschema-comparison-20260721/REPORT.md`.

## Best four-model consensus and strict fallback

Replacing DDL-prompt Qwen3 with M-Schema Qwen3 in the four-model core, while
retaining XiYan M-Schema as the fixed tie priority, produced a new core result:

| System | Execution |
|---|---:|
| Qwen3 base SFT + M-Schema | 78.627% (813/1,034) |
| XiYan M-Schema | 78.433% (811/1,034) |
| Four-model core consensus | **82.398% (852/1,034)** |
| Four-model oracle | **89.168% (922/1,034)** |

The failed Gretel checkpoint must not join normal consensus voting: doing so
reduces the ensemble to 80.754%. It is useful as a strictly isolated fallback,
however. A gold-blind router selects it only when all four core predictions are
non-executable and the fallback itself executes. This condition occurred five
times; the fallback was executable twice, corrected both, and regressed none,
raising the final measured system to **82.592% (854/1,034)**. The fallback's
increment is too small for standalone statistical significance (p=0.5), so it
should be described as an exploratory validation-set result and verified on a
held-out test set before adoption.

The M-Schema augmented variant was non-executable on all five fallback
opportunities and made no change. The DDL augmented variant is therefore the
only useful fallback of the two.

Evidence:

- Four-model core: `artifacts/zero-shot-eval/four-model-all-mschema-execution-ensemble-20260721/metrics.json`
- Strict fallback: `artifacts/zero-shot-eval/four-model-mschema-with-gretel-fallback-20260721/metrics.json`
- Reproducible router: `scripts/apply_execution_fallback.py`

## Small-model prompt sweep and revised best ensemble

M-Schema did not transfer uniformly. Qwen2.5-Coder 1.5B fell from 65.571% to
43.520% execution with only 81.915% syntax validity, so its DDL prompt is
retained. DeepSeek-Coder 1.3B changed from 67.602% to 67.021% and is slightly
worse alone, but its changed error distribution is more complementary. Using
DeepSeek M-Schema while retaining Qwen2.5 DDL raises the four-model core to
**82.785% (856/1,034)** and its oracle to **89.458% (925/1,034)**. The strict
failed-execution fallback then corrects two cases without regressions, producing
the current deterministic best of **82.979% (858/1,034)**.

Evidence:

- DeepSeek M-Schema run: `artifacts/zero-shot-eval/runs/20260721-171802-22318/`
- Qwen2.5 M-Schema run: `artifacts/zero-shot-eval/runs/20260721-171802-22319/`
- Revised core: `artifacts/zero-shot-eval/four-model-deepseek-mschema-ensemble-20260721/metrics.json`
- Revised strict fallback: `artifacts/zero-shot-eval/four-model-deepseek-mschema-with-gretel-fallback-20260721/metrics.json`

## FINER-SQL 3B integration: DDL ablation

The GRPO-trained `griffith-bigdata/FINER-SQL-3B-Spider` checkpoint was pinned at
revision `e3b7cd2054920cf346ad4b7aedd57d8d4b949e9d`. Its reasoning output exposed
an evaluator issue: generic extraction selected the first English word
"select" inside `<think>` instead of the SQL after `</think>`. The extractor now
follows the model authors' published `split("</think>")[-1]` rule, covered by a
regression test. Corrected syntax validity is 99.903%.

With the published instruction but the project's plain DDL schema, greedy
FINER scores **73.114% (756/1,034)**. This is intentionally an ablation, not a
reproduction of the reported 85% result: the authors use a question-specific,
value-enriched schema. Adding this weaker DDL prediction as a normal fifth vote
reduces consensus from 82.785% to 81.818%, but expands the five-model oracle to
**90.135% (932/1,034)**. Thus the 90% target is already present in candidate
coverage; candidate selection, not raw model capacity, is now the primary
bottleneck.

Using the exact published value-enriched messages raises greedy FINER by 3.578
points to **76.692% (793/1,034)** with 99.903% syntax validity. Complexity
execution is 84.486% simple, 71.726% moderate, and 48.571% complex. Adding this
greedy output expands the five-model oracle further to **90.716% (938/1,034)**,
although ordinary majority voting remains 82.785%. As a strict fallback only
when all four core outputs fail execution, FINER corrects three examples with
no regressions, setting a new deterministic best of **83.075% (859/1,034)**.
This strict fallback gain is exploratory (three discordant cases, p=0.25).

The exact public prompt parquet was saved at
`data/raw/finer-sql/spider_dev_prompts.parquet` and aligned to local databases.
All **1,034/1,034** system/user messages in
`data/processed/spider/validation_finer_official.jsonl` match the published
dataset. The DDL control is retained separately as
`data/processed/spider/validation_finer_ddl.jsonl`.

Evidence:

- DDL run: `artifacts/zero-shot-eval/runs/20260721-180536-53811/`
- Five-model DDL analysis: `artifacts/zero-shot-eval/five-model-finer-ddl-ensemble-20260721/metrics.json`
- Official enriched greedy run: `artifacts/zero-shot-eval/runs/20260721-183453-72534/`
- Official greedy five-model analysis: `artifacts/zero-shot-eval/five-model-finer-official-greedy-ensemble-20260721/metrics.json`
- New strict fallback: `artifacts/zero-shot-eval/four-model-with-finer-official-fallback-20260721/metrics.json`
- Prompt builder: `scripts/build_finer_eval_data.py`
- Upstream repository and reported Spider result: https://github.com/thanhdath/finer-sql

## Metric audit: strict denotation vs published MAC-SQL Spider EX

The project's original execution scorer intentionally required exact result
column order. FINER-SQL reports the MAC-SQL Spider execution metric instead:
it removes `DISTINCT`, uses bag semantics unless the gold query has `ORDER BY`,
and accepts one global permutation of output columns. A local deterministic
implementation of those published semantics was added alongside (not in place
of) the stricter project metric and covered by regression tests. It was also
cross-checked against upstream `exec_eval.py` on all 1,034 selected five-model
predictions with **0 disagreements**.

| System | Project strict EX | MAC-SQL/FINER EX |
|---|---:|---:|
| FINER 3B enriched greedy | 76.692% | 81.431% |
| XiYanSQL 3B M-Schema | 78.433% | **83.269%** |
| Qwen3 4B base-SFT M-Schema | 78.627% | **83.172%** |
| Four-model core consensus | 82.785% | 87.041% |
| Five-model FINER consensus | 82.785% | **87.331%** |
| Strict FINER fallback | **83.075%** | **87.331%** |

Thus the requested 80--85% small-model goal is already met by two independent
3--4B models under the benchmark metric used by published systems. The current
system is 2.669 points short of the 90% system target. Both metrics will remain
in every subsequent report so column-permutation equivalence cannot be mistaken
for an unqualified model-quality gain.

Evidence:

- Dual-metric rescore: `artifacts/zero-shot-eval/macsql-rescore-20260721/metrics.json`
- Reproducible scorer: `scripts/score_predictions_macsql.py`
- MAC-SQL-compatible comparison: `scripts/evaluate_text2sql_models.py`
- Upstream evaluator: https://github.com/wbbeyourself/MAC-SQL/blob/main/evaluation/exec_eval.py

## Active test-time scaling and M-Schema training

FINER's published test-time configuration (30 samples, temperature 1.0,
2,048 output tokens, official enriched prompts, vLLM 0.10.2) is running on an
L4. Candidate-level artifacts are retained so published value-aware voting,
confidence routing, weighted multi-model voting, and candidate-oracle coverage
can all be compared without regenerating the 31,020 SQL samples.

The Qwen3 4B M-Schema QLoRA run was resumed from a locally verified full
checkpoint after a Colab workspace reset. Checkpoints 100, 200, and 300 are
locally verified. The independent
`griffith-bigdata/Qwen3-4B-SQL-Writer` checkpoint is pinned at revision
`a44db07c04eedcf745308e46e9bd61ce08e03f17` and queued for the first L4 released
by the two active jobs; a third simultaneous Colab allocation was rejected by
the account's assignment limit before any model work began.

At 23:34 PDT, Colab cleared both active workspaces while their CLI executions
still appeared BUSY. FINER was recovered from its validated 752/1,034 local
candidate checkpoint and therefore regenerates only the final 282 examples.
Qwen had reached step 336, but its most recent complete exported checkpoint is
step 300; that archive is checksum-validated locally and queued for resume.
Ghost allocations were explicitly stopped once the missing remote status files
confirmed the reset. Colab subsequently enforced one active assignment, so the
remaining jobs are serialized until the account restores parallel capacity.

An explicitly provisional audit of the first 456/1,034 completed FINER examples
found **85.526% MAC-SQL EX** (390/456) from published value-aware voting and a
**92.982% strict candidate oracle**. Combining the FINER pool with the four core
models reached 88.377% MAC-SQL EX at the best global candidate-pool weight. A
separate confidence router reached 89.474% at its best global setting, but only
87.281% under leave-one-database-out selection. These prefix results are useful
for monitoring, not final claims: the prefix is not a random or held-out sample,
and the global settings are exploratory validation tuning. Final full-set and
gold-blind results will supersede them.

Interim evidence:

- FINER prefix rescore: `artifacts/zero-shot-eval/finer-n30-partial-456-20260721/`
- Candidate-pool prefix analysis: `artifacts/zero-shot-eval/finer-core-partial-456-20260721/`
- Confidence-router prefix analysis: `artifacts/zero-shot-eval/finer-core-router-partial-456-20260721/`

## FINER-SQL 3B 30-sample final

The published 30-sample configuration completed all 1,034 validation examples
after resuming from the 752-example recovery checkpoint. Exact published VAV
selection scores **79.014% strict EX (817/1,034)** and **84.236% MAC-SQL EX
(871/1,034)**. Its strict per-example candidate oracle is 90.232% (933/1,034).
The four-core-plus-FINER pool has **95.551% MAC-SQL candidate-oracle coverage
(988/1,034)**, but simple weighted result voting reaches only 86.460%; it is
therefore rejected as a production selector. The full confidence router also
fails to beat the prior 87.331% completed baseline. This confirms that candidate
coverage is sufficient and learned semantic selection is the remaining
bottleneck, rather than more FINER sampling.

Evidence:

- Completed generation: `artifacts/zero-shot-eval/runs/20260721-233648-66023/`
- Published VAV + official score: `artifacts/zero-shot-eval/finer-n30-full-published-20260722/`
- Full weighted pool: `artifacts/zero-shot-eval/finer-core-full-20260722/`
- Full confidence router: `artifacts/zero-shot-eval/finer-core-router-full-20260722/`

## Prepared <=4B outcome-verifier path

The public balanced Spider candidate-label dataset from GradeSQL is pinned at
revision `44cbee9732352a98cc2088005acd0839c3c266aa`. Its original 17,834 rows
contain 10,155 exact normalized duplicates. The local verifier package removes
those duplicates, holds out entire question+schema groups (not random candidate
rows), and excludes 16 whole oversized-schema groups instead of truncating
them. The resulting package has 6,891 train examples across 1,416 groups and
706 validation examples across 146 disjoint groups. Qwen2.5-Coder 1.5B
tokenization is fully valid at 4,096 tokens (maximum 3,930; zero truncations).
The label mapping was cross-checked against both the authors' dataset-builder
source and local Spider gold SQL before launch: GradeSQL label 1 means a correct
candidate (`Yes`) and label 0 means an incorrect candidate (`No`). This check
caught and corrected a reversed local mapping before any verifier GPU run.

The active verifier is a 1.5B QLoRA model trained to output `Yes`/`No` for a
question, value-enriched schema, and candidate SQL. It will rank only distinct
execution groups from the FINER/core candidate pool. It is not included in any
reported score until training and gold-blind reranking both complete.

The finalized gold-blind inference file uses both exact raw-execution and exact
MAC-SQL-postprocessed signatures. This prevents result grouping from collapsing
queries that differ in column count, column order, multiplicity, or DISTINCT
behavior under the official scorer. It contains 2,635 groups for 1,034 examples
(mean 2.548, maximum 21) and preserves the full **95.551% MAC-SQL candidate
oracle**. The safe file contains no audit correctness fields; gold labels exist
only in the separate audit artifact used to report oracle coverage.

Evidence:

- Builder: `scripts/build_gradesql_orm_sft.py`
- Package: `data/finetuning/gradesql_orm_spider_v1/manifest.json`
- Token audit: `data/finetuning/gradesql_orm_spider_v1/tokenization_report.json`
- Training config: `configs/gradesql_orm_qlora_training.json`
- Inference/reranking implementation: `scripts/evaluate_orm_candidate_groups.py`
- Gold-blind candidate groups: `artifacts/zero-shot-eval/orm-candidate-groups-exact-dual-full-20260722/candidate_groups.jsonl`
- Same-session Colab reranker: `scripts/run_colab_orm_rerank_existing.sh`
- Public method/code: https://github.com/sisinflab/GradeSQL

## GradeSQL-style 1.5B verifier result

The Qwen2.5-Coder-1.5B GradeSQL-style verifier completed 862 optimizer steps
across two epochs after two checksum-verified Colab resumes. Final held-out
loss was **0.229596**. The downloaded adapter, final checkpoint, pinned base
revision, resume state, and assistant-only labels all passed local artifact
validation; the final adapter SHA-256 is
`937de00d7a9c6129d3923ee77852b3e8d9d5a24c4617cc5b912abda48bca4f3c`.

Gold-blind reranking then scored all 2,635 execution groups for all 1,034
validation examples on a fresh L4. The first retained-session attempt was
invalidated by Colab clearing `/content`; the fresh launcher uploads the
validated local adapter and always stops its session. Two fresh preflight
failures exposed and fixed missing `sqlglot` evaluation dependencies and
Colab's incompatible optional `torchao 0.10.0`; neither failure produced or
contributed predictions. The completed run used SDPA and selected a candidate
solely by maximum learned Yes-minus-No logit margin.

The selector scores **74.565% strict EX (771/1,034)** and **78.240% MAC-SQL EX
(809/1,034)**. Complexity MAC-SQL EX is 89.039% simple, 70.833% moderate, and
40.952% complex. This is substantially below the completed 87.331% baseline,
so the standalone ORM selector is rejected. The result shows that low
held-out loss on the public GradeSQL distribution does not transfer to semantic
selection among the project's FINER/core execution groups; future selector
training must use in-domain candidates or a gold-free system that preserves the
strong baseline rather than replacing it globally.

Evidence:

- Validated training run: `artifacts/qlora-training/runs/20260722-021145-qwen2.5-coder-1.5b-instruct-68749/`
- Fresh reranker: `scripts/run_colab_orm_rerank_fresh.sh`
- Completed reranking: `artifacts/orm-reranking/runs/20260722-072736-fresh/`
- Dual-metric score: `artifacts/orm-reranking/runs/20260722-072736-fresh/macsql-score/metrics.json`
