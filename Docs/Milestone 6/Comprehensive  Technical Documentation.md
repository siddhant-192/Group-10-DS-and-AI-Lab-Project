# Comprehensive Technical Documentation

## Model Development: Auditor Reproduction Protocol

### 1\. Scope, reference state, and result to reproduce

This section covers the team's Week 3–4 work: Spider preparation, prompt/SFT package construction, zero-shot baselines, QLoRA, early evaluation, rejected experiments, specialists and ensembles, leakage-safe hyperparameter search, decoding search, and final adapter evaluation. The UI, deployment, BIRD, and other members' work must be documented separately.

Reference state:

| Item | Recorded value |
| :---- | :---- |
| Repository | `Group-10-DS-and-AI-Lab-Project` |
| Branch / commit | `main` / `7fd469cc1a6c3fc8a97babb22a6cf20bbceeac39` |
| Documentation audit | 2026-08-23 |
| Main base model | `Qwen/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554` |
| Milestone 3 release | Natural Spider QLoRA \+ M-Schema: 78.627% strict / 83.172% compatible |
| Milestone 4 release | Checkpoint 375: 77.853% strict / 81.238% compatible on official Spider validation |

Do not mix the two selected adapters:

| Artifact | Training set | LR / steps | Weight SHA-256 | Purpose |
| :---- | ----: | ----: | :---- | :---- |
| M3 natural adapter | 6,997 | 2e-4 / 438 | `5274d4c15179b195443940d92f8caacf10f99bdfca106ee24324cd44a2fbe9bb` | Best Week 3 single adapter |
| M4 final adapter | 5,996 | 3e-4 / 375 | `63a51ff491a163c1433dd4ac56d936d969337de67279f852ea1ef966ac335e5c` | Final leakage-safe HPO artifact |

The release pointers are `release/final_model.json` and `release/milestone4_final_model.json`.

### 2\. Current fresh-clone blockers

The evidence is usable, but audited commit `7fd469...` is **not yet a one-command reproduction package**. Repository reorganization moved scripts into `src/scripts/` and flattened `src/text2sql_data/`, without updating every path.

Before running the commands below:

1. Fix shell scripts that calculate the root as `SCRIPT_DIR/..`; from `src/scripts/` the repository is two parents up.  
2. Fix Python defaults using `Path(__file__).resolve().parents[1]`; use the correct root or one tested root helper.  
3. Make `prepare_spider_data.py` and tests agree on either the flat `src/spider_pipeline.py` layout or a restored `src/text2sql_data/` package.  
4. Update README, tests, `requirements-dev.txt`, and `PACKAGE_MANIFEST.sha256` from historical top-level `scripts/` paths.  
5. Publish the excluded data, predictions, checkpoints, and M4 adapter using stable URLs and hashes.

Audit the drift with:

```shell
rg -n 'PROJECT_ROOT.*SCRIPT_DIR/\.\.|PROJECT_ROOT.*parents\[1\]|/scripts/' \
  README.md requirements-dev.txt src tests
```

All later commands use the intended `src/scripts/` locations **after those fixes**. Until a different team member completes the fresh-clone test in Section 11, describe this as a reconstruction protocol, not a verified self-contained package.

### 3\. Code map and environments

| Location | Responsibility |
| :---- | :---- |
| `src/schema.py`, `src/validation.py`, `src/spider_pipeline.py` | Immutable schema extraction, read-only gold execution, Spider validation/leakage audit |
| `src/scripts/build_sft_dataset.py` | Natural and difficulty-curriculum chat SFT |
| `src/scripts/build_mschema_sft_package.py` | DDL-to-M-Schema conversion |
| `src/scripts/train_text2sql_qlora.py` | Assistant-only QLoRA training/export |
| `src/scripts/evaluate_text2sql_models.py` | Inference, SQL extraction, safe execution, strict metric, resume |
| `src/scripts/score_predictions_macsql.py` | Compatible execution rescoring |
| `src/scripts/build_qwen3_hparam_split.py` | Database-disjoint M4 split |
| `src/scripts/build_qwen3_hparam_screen.py` | Fixed 2,048-row HPO screen |
| `configs/`, `evidence/`, `release/` | Exact configurations, compact results, selected artifact identities |

Use separate environments; the broad top-level `requirements.txt` is not the scientific lock.

```shell
python3.12 -m venv .venv-data
.venv-data/bin/python -m pip install --upgrade pip
.venv-data/bin/python -m pip install -r requirements-data.txt

python3.12 -m venv .venv-model-eval
.venv-model-eval/bin/python -m pip install --upgrade pip
.venv-model-eval/bin/python -m pip install \
  -r src/scripts/model-eval-local-requirements.txt
```

The final remote run used an NVIDIA L4 (22.034 GiB), PyTorch 2.11.0+cu128, Transformers 4.57.6, Datasets 5.0.0, Accelerate 1.14.0, PEFT 0.19.1, bitsandbytes 0.49.2, and TensorBoard 2.20.0. Quantization was NF4 with double quantization and BF16 compute; TF32 and SDPA were enabled. Remote locks are `src/scripts/colab-sft-requirements.txt` and `src/scripts/colab-eval-requirements.txt`.

Authenticate using the auditor's account, never committed credentials:

```shell
bash src/scripts/setup_colab_cli.sh --gpu L4
```

Allow roughly 1 GiB for Spider databases, more than 15 GiB for baseline snapshots, 132 MB per Qwen3 adapter, and substantially more for optimizer checkpoints.

### 4\. Rebuild and verify all model inputs

#### 4.1 Spider

Obtain the official Spider 1.0 SQLite payload under its license and preserve:

```
milestone3/database/<database_id>/<database_id>.sqlite
```

The reference payload had 166 databases, all passing SQLite quick checks. The annotations use 140 train and 20 validation databases. The pinned `xlangai/spider` Parquets recorded in `data/raw/spider/source_manifest.json` are:

| Split | Rows / bytes | SHA-256 |
| :---- | ----: | :---- |
| train | 7,000 / 831,359 | `cb4b681558f6f8f428e516fb94c5a1cb19c5a0a0c153c0618c8cc4a28115d4cb` |
| validation | 1,034 / 125,887 | `c3e2a46303899a2d4afe3f6a3a62e59f8d589f241b3cbfb52356479b1f054888` |

```shell
bash src/scripts/setup_spider_data.sh
```

Do not use `--skip-query-execution`. Three train annotations are deliberately rejected because gold SQL failed execution. Therefore `--fail-on-execution-errors` is a diagnostic that exits nonzero, not the normal build command.

Acceptance:

| Check | Expected |
| :---- | :---- |
| usable train / validation / rejected | 6,997 / 1,034 / 3 |
| database overlap / exact triple overlap | 0 / 0 |
| normalized question overlap | 6 |
| `train.jsonl` | `37eb49700718cb0f9689a89a2c7976f06f6336f107a69b0a0dcf7dc383b97e51` |
| `validation.jsonl` | `649394695ac0548f917b5391bc97f6e2a8f26421288fd05c83339f5a6441bcf7` |
| `schemas.json` | `959baddb570e6c76a51c67a8f1c2ec6b820f283cae362e04588cf8a26b806226` |

The source of truth is `data/processed/spider/manifest.json`. Stop if counts or hashes differ.

#### 4.2 Natural, curriculum, and M-Schema SFT packages

```shell
.venv-data/bin/python src/scripts/build_sft_dataset.py \
  --train data/processed/spider/train.jsonl \
  --validation data/processed/spider/validation.jsonl \
  --policy configs/text2sql_sft_sampling.json \
  --output-dir data/finetuning/spider_sft_v1

.venv-data/bin/python src/scripts/build_mschema_sft_package.py \
  --source-dir data/finetuning/spider_sft_v1 \
  --processed-dir data/processed/spider \
  --output-dir data/finetuning/spider_mschema_sft_v1 \
  --examples 3 --max-mschema-chars 10000
```

The natural package uses every accepted row once. Seed-17 weighted curriculum adds 2,799 hard rows: 6,997 base, 9,796 curriculum, 1,034 validation. Expected hashes are:

| Package/file | SHA-256 |
| :---- | :---- |
| DDL base | `0965358e425aee4fc43ac6671ea5750f307390b5cc7fb861b54efddb84143ee8` |
| DDL curriculum | `e7bcdac53a891c8910beb8ee5b0955ec57b3b42c6ee8e102c154091b978f0e36` |
| DDL validation | `4d88bd5d240375e14db542f099daa5f491dd1b1e93724b89004a334fddf3e9f9` |
| sampling weights | `08634d6acee4485123f9846188a6be7d9439d5b0764328b4c77cf6ba36e1bffb` |
| M-Schema base | `9afad6bec3ef677682b1df5ff84770d5dd1d7ae734edba70ccd4e6c08c03f588` |
| M-Schema curriculum | `e0fe982c617d6fa1f5d30038a791f96fc7e8087ffcea00cbc4a65e6a95a46082` |
| M-Schema validation | `d825177f892f5b455132be4e724ceb34a482801698b0c37af556c4acd05ddd42` |

M-Schema records tables, typed columns, keys, and up to three bounded values. Because `baseball_1` exceeds 10,000 characters, 82 base and 110 curriculum rows use DDL fallback.

Download pinned models, then ensure no target truncation:

```shell
bash src/scripts/download_eval_models.sh
.venv-model-eval/bin/python src/scripts/preflight_sft_dataset.py \
  --data-dir data/finetuning/spider_sft_v1 \
  --config configs/text2sql_eval_models.json \
  --model-root models/text2sql-eval
```

Maximum sequence lengths are Qwen3 3,173/4,096, Qwen2.5 3,173/4,096, and DeepSeek 4,609/5,120; truncated rows must be zero.

#### 4.3 Pinned models

| Model | Revision | Parameters |
| :---- | :---- | ----: |
| `Qwen/Qwen3-4B-Instruct-2507` | `cdbee75f17c01a7cc42f958dc650907174af0554` | 4,022,468,096 |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | `2e1fd397ee46e1388853d2af2c993145b0f1098a` | 1,543,714,304 |
| `deepseek-ai/deepseek-coder-1.3b-instruct` | `e063262dac8366fc1f28a4da0ff3c50ea66259ca` | 1,346,471,936 |
| `XGenerationLab/XiYanSQL-QwenCoder-3B-2504` | `b883e58ed83f74bab037d6a7b90c4b8706d357d7` | specialist |
| `griffith-bigdata/FINER-SQL-3B-Spider` | `e3b7cd2054920cf346ad4b7aedd57d8d4b949e9d` | specialist |
| `griffith-bigdata/Qwen3-4B-SQL-Writer` | `a44db07c04eedcf745308e46e9bd61ce08e03f17` | prepared, not scored |

The general-model downloader creates a file-level manifest. Specialists download their pinned revisions on Colab. Never replace a revision with “latest.”

### 5\. Evaluation contract and baseline

`evaluate_text2sql_models.py` renders the chat prompt without the answer, extracts the first SQL after reasoning/fences, parses exactly one SQLite expression with sqlglot, opens SQLite with `mode=ro&immutable=1`, denies writes/DDL/attach/pragma/transactions, interrupts long queries, and bounds results.

- **Strict execution:** same normalized rows and column count; column order remains significant; row order matters when gold uses `ORDER BY`.  
- **Compatible execution:** MAC-SQL-style normalization plus at most one global column permutation.  
- **Syntax valid:** parseable only, not necessarily correct.  
- **Oracle:** any sampled candidate is correct; an upper bound, never a deployable score.

The compatible implementation had zero disagreements with upstream MAC-SQL on all 1,034 selected predictions.

Run a 12-row pilot, then the frozen full baseline:

```shell
bash src/scripts/run_colab_zero_shot_eval.sh --limit 12
bash src/scripts/run_colab_zero_shot_eval.sh
```

Reference run `20260720-012118`:

| Model | Strict (correct/1,034) | Syntax | ms/example | VRAM |
| :---- | ----: | ----: | ----: | ----: |
| Qwen3-4B | 72.340% (748) | 100.000% | 596.25 | 9.13 GiB |
| Qwen2.5-1.5B | 56.576% (585) | 99.903% | 197.73 | 4.25 GiB |
| DeepSeek-1.3B | 47.292% (489) | 98.066% | 817.71 | 5.00 GiB |

Qwen3 had 286 failures: 251 executable-but-wrong and 35 execution errors. Accuracy was 82.125% simple, 61.905% moderate, and 50.476% complex, showing semantic/schema-linking—not syntax—as the main weakness.

Recompute any saved prediction file without GPU:

```shell
.venv-model-eval/bin/python src/scripts/score_predictions_macsql.py \
  --predictions MODEL=/path/to/predictions.jsonl \
  --validation data/processed/spider/validation.jsonl \
  --project-root . --output-dir artifacts/rescored/MODEL
```

### 6\. Week 3: QLoRA and all model experiments

#### 6.1 Training and valid resume

Main QLoRA settings: NF4 double quantization/BF16; LoRA r16, alpha32, dropout0.05, all linear layers; micro-batch 2 × accumulation 8 \= effective 16; paged AdamW 8-bit; gradient checkpointing; one epoch; seed 17; Week 3 LR 2e-4. Only assistant SQL tokens receive loss.

```shell
bash src/scripts/run_colab_qlora_sft.sh \
  --model qwen3-4b-instruct-2507 \
  --dataset base \
  --data-dir data/finetuning/spider_sft_v1 \
  --training-config configs/text2sql_qlora_training.json
```

A valid resume restores adapter, optimizer, scheduler, RNG, trainer state, and global step. Natural lineage was `181944(0→100) → 192730(100→200) → 203050(200→300) → 213353(300→438)`. Curriculum lineage was `222755(0→100) → 233410(100→200) → 004046(200→300) → 014423(300→400) → 024751(400→500) → 035111(500→613)`. A segment restarting at zero is a new run.

Pipeline smoke evidence:

| Run | Result |
| :---- | :---- |
| `20260720-082659` | resume step 2→4; loss 1.463103; adapter `774f975d12b076fcccacafdccbfb17f780581ff08ffb3b9504f3e46165a8d6c1`; 10.591/12.617 GiB |
| `20260720-180631` | hardened packaging/validator passed; loss 1.466529 |

#### 6.2 Core Week 3 results

All rows below use official Spider validation (N=1,034). DDL is strict-only where compatible scoring was not part of that run.

| Experiment (run) | Training result | Strict / compatible | Decision |
| :---- | :---- | ----: | :---- |
| Qwen3 natural (`073105`) | 6,997 rows, 438 steps, loss .255960, SHA `5274d4...` | 76.886% / — | selected |
| Qwen3 curriculum (`074952`) | 9,796 rows, 613 steps, loss .267661, SHA `4bf7f03b...` | 74.371% / — | rejected |
| Qwen2.5 natural (`083451...`; eval `093619...`) | loss .240890, SHA `64958a46...`, 73,911,112 bytes | 65.571% / — | \+8.994 pp control |
| DeepSeek natural (`095647...`; eval `102939...`) | loss .196816, SHA `c20ba38b...`, 60,010,048 bytes | 67.602% / — | \+20.309 pp control |
| Qwen3 natural \+ M-Schema (`155144...`) | same M3 weights | 78.627% / 83.172% | M3 release |

Natural corrected 47 baseline examples (`p=.000288`); curriculum was 26 examples below natural (`p=.008781`). M-Schema corrected 59 and regressed 41 versus DDL (`p=.0886`). It was retained for best observed accuracy, not because the paired difference crossed 0.05.

Evaluate an adapter and compare aligned predictions:

```shell
bash src/scripts/run_colab_zero_shot_eval.sh \
  --model qwen3-4b-instruct-2507 \
  --data data/processed/spider/validation.jsonl \
  --adapter-dir /path/to/adapter --adapter-label qwen3-natural

.venv-model-eval/bin/python src/scripts/compare_prediction_pair.py \
  --before /path/to/baseline.jsonl --after /path/to/finetuned.jsonl \
  --validation data/processed/spider/validation.jsonl \
  --before-label zero-shot --after-label qwen3-natural \
  --output-dir artifacts/comparisons/qwen3-natural-vs-baseline
```

The M3 adapter must be exactly 132,187,888 bytes with the full hash in Section 1\. Use `verify_final_adapter.py` and `smoke_final_model.py` against `release/final_model.json`; smoke generation proves loading, while read-only execution against a known database proves correctness.

#### 6.3 Rejected, specialist, ensemble, and reranker experiments

These are required audit history, even though they are not the final pipeline:

| Experiment | Reproduction input/config | Run and outcome |
| :---- | :---- | :---- |
| Gretel augmentation | Source Parquet SHA `2bee9ac07cf5057d36b5ea30fb47d948697e882f42bd1cc661185396287c0180`; `build_gretel_augmented_sft.py`, max 5,000, workers12, seed29 | 100k read; 67,836 accepted; 67,831 deduped; 5,000/100 domains selected; 11,997 combined. SFT `154029...`, 750 steps, loss .245359, adapter `2a0f22fc...`. DDL `160749...` 74.468%; M-Schema `161956...` 76.402%; rejected. |
| XiYan frozen specialist | Revision in Section 4; `build_xiyan_mschema_eval_data.py --examples 3` | DDL `104253...` 75.822%; M-Schema `111750...` 78.433% / 83.269%; ensemble input. |
| Execution consensus | `select_execution_consensus`, 5 candidates, first 300 rows | `104300...`: greedy 71, selected 74, oracle 83 correct. |
| Multi-model ensemble | `ensemble_text2sql_predictions.py`, fixed tie order XiYan, Qwen3, DeepSeek, Qwen2.5, FINER | Progression 79.304 → 81.721 → 82.398 → 82.785 strict; final five-model 82.785 / 87.331, oracle 90.716. Strict FINER fallback 83.075 / 87.331 (+3, −0). Research result, not release. |
| FINER frozen specialist | Revision in Section 4; `build_finer_eval_data.py`; enriched published prompts | DDL `180536...` 73.114%; enriched `183453...` 76.692 / 81.431%; n=30 `233648...` selected 79.014 / 84.236, oracle 90.232. Combined selector 86.460 vs compatible oracle 95.551; global routing rejected. |
| GradeSQL ORM | Dataset revision `44cbee9732352a98cc2088005acd0839c3c266aa`; Parquet SHA `9a162085076295a284e9093d89894c16fb71037bf5772ee720559cc0290b9343` | 17,834 raw; 10,155 duplicates; train 6,891/1,416 groups, val 706/146 groups, zero truncation. SFT `021145...`, 862 steps, loss .229596, adapter `937de00d...`. Fresh rerank `072736-fresh`: 74.565 / 78.240; rejected. |
| Qwen3 M-Schema SFT attempt | Main QLoRA config | Incomplete; visible step 336, verified checkpoint 300, no final score. |
| SQL Writer | Revision in Section 4 | Prepared only; no validated score. |

Full historical adapter SHA-256 values: Qwen2.5 natural `64958a46f37277b62b90c2c9d4e48e9de4d9d101616ab55c76f0c9805df5ce19`; DeepSeek natural `c20ba38b1b41fec4a2ea2c2e49502ed2732db94b47c14b9a1ade5253f588834b`; Gretel-augmented Qwen3 `2a0f22fc72595488e2207adba2591744d6a45a3182c4fe0179794b12ce8f05a6`; GradeSQL ORM `937de00d7a9c6129d3923ee77852b3e8d9d5a24c4617cc5b912abda48bca4f3c`.

For Gretel, rebuild with `build_gretel_augmented_sft.py`. For GradeSQL use `build_gradesql_orm_sft.py` and `configs/gradesql_orm_qlora_training.json` (Qwen2.5, r16/alpha64, dropout .05, LR 7e-5, cosine/3%, two epochs, batch16, seed29), then `build_orm_candidate_groups.py` and `run_colab_orm_rerank_fresh.sh`. The candidate file must contain no gold-correctness fields.

FINER n=30 uses vLLM, temperature 1.0, max 2,048 tokens, and its published value-aware voting. Its extractor must take SQL after the final `</think>`; earlier prefix scores are superseded. The FINER prompt Parquet URI/hash is currently missing, so exact enriched-prompt reproduction is blocked until published.

Operational failures remain failures, not results: `20260720-083847` lost session, `192501` failed upload, and `20260721-044156`/`044430`/`044625` failed adapter transfer/install/load. Full source prediction files are required to recreate ensemble metrics.

### 7\. Week 4: leakage-safe HPO and decoding

#### 7.1 Fixed split

Week 3 repeatedly used official validation, so Week 4 selected hyperparameters on 20 databases held out only from original Spider train.

```shell
.venv-model-eval/bin/python src/scripts/build_qwen3_hparam_split.py \
  --source data/finetuning/spider_mschema_sft_v1 \
  --output data/finetuning/qwen3_hparam_mschema_v1 \
  --validation-databases 20 --seed 20260726 --search-trials 50000

.venv-model-eval/bin/python src/scripts/build_qwen3_hparam_screen.py \
  --source data/finetuning/qwen3_hparam_mschema_v1 \
  --output data/finetuning/qwen3_hparam_mschema_screen_v1 \
  --train-rows 2048 --seed 20260726
```

| Partition | Rows / DBs | Simple / moderate / complex | SHA-256 |
| :---- | ----: | ----: | :---- |
| final internal train | 5,996 / 120 | 3,423 / 1,921 / 652 | `b597a64b91c8161ebe07bf4686a74807da1c27c82fb752b4ae2d5117a6579f65` |
| HPO screen | 2,048 / 120 | 1,169 / 656 / 223 | **not published: blocking gap** |
| internal tune | 1,001 / 20 | 572 / 321 / 108 | `c4ddf2d8ab131200af869c6eeb9a817e4095be640a92964f44b3f532a921f986` |

Require zero DB overlap and deterministic IDs. Official validation was outside Week 4 gradients/HPO, but was already used in Week 3, so it is not globally untouched.

#### 7.2 Run and score HPO

```shell
LABEL=lr3e4
bash src/scripts/run_colab_qlora_hparam.sh \
  --label "$LABEL" \
  --training-config "configs/hparam/qwen3/$LABEL.json" \
  --data-dir data/finetuning/qwen3_hparam_mschema_screen_v1

bash src/scripts/run_colab_zero_shot_eval.sh \
  --model qwen3-4b-instruct-2507 \
  --data data/finetuning/qwen3_hparam_mschema_v1/validation.jsonl \
  --adapter-dir artifacts/qlora-hparam/runs/<RUN>/downloaded/output/final_adapter \
  --adapter-label q3hp-lr3e4

.venv-model-eval/bin/python src/scripts/summarize_qwen3_hparam_search.py
```

Every completed trial used the same 2,048/1,001 rows, greedy decoding, one-factor changes, and strict execution as the declared primary metric.

| Label/change | Strict / compatible | Train run → eval run |
| :---- | ----: | :---- |
| `lr5e5` | 83.117 / 86.414 | `120357-lr5e5-23015` → `141200-9776` |
| `lr1e4` | 84.815 / 87.413 | `130021-lr1e4-59805` → `134651-90252` |
| `lr2e4` | 86.214 / 89.211 | `130439-lr2e4-62675` → `135136-94225` |
| `lr3e4` | **86.513 / 89.111** | `120811-lr3e4-25878` → `140757-6902` |
| r8/a16 | 86.114 / 88.811 | `160158-...-80206` → `164910-10367` |
| r32/a64 | 85.514 / 88.312 | `160312-...-81070` → `165119-11832` |
| dropout 0 | 85.814 / 88.611 | `171711-...-29223` → `180839-61718` |
| dropout .10 | 86.014 / 89.111 | `171711-...-29224` → `180405-58681` |
| attention-only | 85.514 / 88.312 | `182928-...-75496` → `190546-98544` |
| linear scheduler | 86.314 / 89.311 | `182928-...-75495` → `191559-5515` |
| 10% warmup | 84.815 / 87.712 | `192240-...-9987` → `200944-40010` |
| seed29 | 86.214 / 89.610 | `203056-...-53804` → `211915-84507` |
| seed41 | 85.514 / 88.711 | `204756-...-64539` → `213709-96116` |

The full identifiers and machine-readable table are in `evidence/milestone4/hparam/search_summary.csv`. Seed 17/29/41 strict mean was 86.080%, population SD .419 point, range .999 point. The two-epoch trial stopped at 163/256 without a comparable adapter; it is incomplete, not evidence that two epochs are worse. Selected: LR 3e-4, r16/alpha32/dropout.05/all-linear, cosine/3% warmup, one epoch, seed17.

#### 7.3 Decoding

```shell
.venv-model-eval/bin/python src/scripts/build_qwen3_decode_screen.py \
  --input data/finetuning/qwen3_hparam_mschema_v1/validation.jsonl \
  --output data/finetuning/qwen3_hparam_mschema_v1/validation_decode_screen300.jsonl \
  --size 300 --seed 2717
```

The screen must contain 172 simple, 96 moderate, 32 complex rows across all 20 databases. Using the `lr3e4` screen adapter, greedy has one candidate; sampled modes use four, top-p .95, and execution-consensus:

| Mode | N | Strict / compatible | Oracle | ms/example |
| :---- | ----: | ----: | ----: | ----: |
| greedy | 300 | 86.000 / 89.000 | — | — |
| t=.2 ×4 | 300 | 86.000 / 89.000 | — | — |
| t=.5 ×4 | 300 | 87.667 / 90.667 | 89.667 | 2,645 |
| t=.8 ×4 | 300 | 85.667 / 88.667 | 90.333 | 2,858 |
| greedy | 1,001 | 86.513 / 89.111 | — | 990 |
| t=.2 ×4 | 1,001 | 86.613 / 89.211 | 87.812 | 2,424 |
| t=.5 ×4 | 1,001 | **87.512 / 90.010** | 90.110 | 2,495 |

Temperature 1.0 failed three times before generation. Greedy remained default because t=.5 costs about 2.52× more; t=.5/top-p.95/four candidates is optional accuracy mode. It was not rerun on official Spider after locking.

### 8\. Final M4 training, verification, and locked result

```shell
bash src/scripts/run_colab_qlora_sft.sh \
  --model qwen3-4b-instruct-2507 \
  --dataset base \
  --data-dir data/finetuning/qwen3_hparam_mschema_v1 \
  --training-config configs/hparam/qwen3/selected-full5996.json
```

Reference run `20260727-083554-qwen3-4b-instruct-2507-18108` resumed verified checkpoints 100, 200, 300, 350 and completed 375\.

| Field | Expected |
| :---- | ----: |
| trainable parameters | 33,030,144 (0.821141%) |
| train / internal eval loss | .0194689 / .2224936 |
| train / eval runtime | 2,047.3876 / 272.338 seconds |
| peak allocated / reserved VRAM | 16.98 / 20.48 GiB |
| adapter bytes / SHA-256 | 132,187,888 / `63a51ff491a163c1433dd4ac56d936d969337de67279f852ea1ef966ac335e5c` |

```shell
.venv-model-eval/bin/python src/scripts/validate_qlora_artifacts.py \
  artifacts/qlora-training/runs/<FULL_RUN>

ADAPTER=artifacts/qlora-training/runs/<FULL_RUN>/downloaded/output/final_adapter/adapter_model.safetensors
wc -c "$ADAPTER"
shasum -a 256 "$ADAPTER"
shasum -a 256 \
  artifacts/qlora-training/runs/<FULL_RUN>/downloaded/output/checkpoint-375/adapter_model.safetensors
```

Final and checkpoint-375 weights must be byte-identical. Checkpoint 375 must contain weights/config, optimizer, scheduler, RNG, trainer state, and training args; global step=375, truncated examples=0, and base revision=`cdbee75...`.

Evaluate greedily:

```shell
bash src/scripts/run_colab_zero_shot_eval.sh \
  --model qwen3-4b-instruct-2507 \
  --data data/finetuning/qwen3_hparam_mschema_v1/validation.jsonl \
  --adapter-dir /path/to/final_adapter --adapter-label qwen3-final-internal

bash src/scripts/run_colab_zero_shot_eval.sh \
  --model qwen3-4b-instruct-2507 \
  --data data/finetuning/spider_mschema_sft_v1/validation.jsonl \
  --adapter-dir /path/to/final_adapter --adapter-label qwen3-final-spider
```

| Evaluation run | N | Strict / compatible | Syntax | Exact |
| :---- | ----: | ----: | ----: | ----: |
| internal `20260727-092353-48716` | 1,001 | 85.514 / 89.011 | 100.000 | 51.349 |
| official `20260727-095012-65804` | 1,034 | **77.853 / 81.238** | 99.903 | 50.387 |

The official strict score is the final single-adapter headline. The 87.512% result is a different screen adapter, internal data, and four-candidate decoding. Evidence is in `evidence/milestone4/final_training/`, `evidence/milestone4/evaluation/`, and `evidence/milestone4/hparam/final_selection.json`.

The checkout lacks `adapter_model.safetensors` and a working M4 URL. `verify_final_adapter.py` expects the older M3 manifest shape. Publish the M4 weights with stable URI, bytes, SHA-256, base revision, license, and retrieval command before claiming fresh-clone inference.

### 9\. One-page decision trail

| Stage | Strict / compatible | Decision |
| :---- | ----: | :---- |
| Qwen3/Qwen2.5/DeepSeek zero-shot DDL | 72.340 / 56.576 / 47.292 strict | Qwen3 base |
| Qwen3 natural vs curriculum DDL | 76.886 vs 74.371 | natural |
| Natural Qwen3 M-Schema | 78.627 / 83.172 | M3 release |
| Gretel augmented M-Schema | 76.402 | reject |
| XiYan M-Schema | 78.433 / 83.269 | ensemble input |
| Five-model / strict FINER fallback | 82.785 / 87.331; 83.075 / 87.331 | research only |
| FINER n=30 | 79.014 / 84.236 | global selector rejected |
| GradeSQL ORM | 74.565 / 78.240 | rejected |
| selected HPO screen greedy | 86.513 / 89.111 internal | select config |
| selected screen t=.5 ×4 | 87.512 / 90.010 internal | optional |
| final step375 greedy | 85.514 / 89.011 internal; **77.853 / 81.238 official** | final single adapter |

The final official score is lower than the HPO score because the databases, adapter, training pool, and decoding experiment differ. This is not a same-test regression. Official Spider was unseen by M4 gradients but had influenced Week 3\.

## Final Evaluation 

This section describes how to reproduce the evaluation results reported for checkpoint 375 on BIRD Mini-Dev. It covers environment setup, data acquisition, the commands used for each of the three prompt configurations, expected outputs at every stage, and troubleshooting notes for issues actually encountered during development. 

### 1\. What This Pipeline Evaluates

**Model under test**: checkpoint 375 \-  a QLoRA adapter for  Qwen/Qwen3-4B-Instruct-2507, frozen at the end of Milestone 4\. No training or checkpoint selection happens in this pipeline; it only  evaluates the already-frozen model.

**Benchmark**: BIRD Mini-Dev (500 examples, 11 SQLite databases), held out  from all training and selection decisions in Milestones 3–4. This is the  first, and only, held-out generalization evidence produced for this checkpoint (Spider validation was examined during Milestone 3 development and is explicitly not used for this milestone's evaluation).

**Reproduction workflow**: this pipeline is run from a fresh git clone of  this repository — see Section 4 for the exact clone command, matching notebooks/m5\_final\_evaluation.ipynb.

**Three configurations evaluated on the identical 500 questions:**

| Configuration | What differs | Result (strict EX) |
| :---- | :---- | :---- |
| Plain DDL | Schema format not seen in training | 16.8% (84/500) |
| M-Schema (primary) | Schema format matches training exactly | 25.8% (129/500) |
| M-Schema \+ evidence | Adds BIRD's natural-language evidence hints | 30.8% (154/500) |

 

### 2\. Environment Setup

**Platform / library versions actually observed during development**

| Component | Version | Source |
| :---- | :---- | :---- |
| Platform | Google Colab, Python 3.12–3.13 | Colab's default runtime image (see note below) |
| GPU | NVIDIA L4, 22.03 GiB VRAM | evaluate\_text2sql\_models.py startup log |
| PyTorch | 2.11.0+cu128 | Colab default, confirmed stable across runs |
| Transformers | 5.13.1–5.15.0 | Colab default (see note below) |
| PEFT | latest compatible with the above | for adapter loading |
| sqlglot | ≥30.0 | SQL parsing (error taxonomy, schema-linking precision, dialect repair) |
| huggingface\_hub | latest | model/adapter download |

### **Critical version pins**  

| Package | Required version | What breaks without it |
| :---- | :---- | :---- |
| torchao | **0.18.0** | Colab's default (0.10.0) is incompatible with the peft version needed to load the adapter — you will get an import/load error when attaching the adapter to the base model. |
| bitsandbytes | **0.48.1** | 4-bit quantized loading fails without a compatible version; error message will explicitly say bitsandbytes is required or incompatible. Confirmed required to be **re-installed every fresh Colab session** — it is not part of Colab's default image and does not persist across sessions. |

   
Verify your environment before running anything:

```py
import torch, transformers, torchao, bitsandbytes, peft	print(torch.__version__, transformers.__version__, torchao.__version__, bitsandbytes.__version__)	print("CUDA available:", torch.cuda.is_available())	
```

### 3\. Data Acquisition

#### Provenance

BIRD Mini-Dev's original source is the [bird-bench/mini\_dev](https://github.com/bird-bench/mini_dev) GitHub repository (official BIRD benchmark team). The `MINIDEV` subdirectory (500 questions, gold SQL, and the 11 SQLite databases) was downloaded from that repository, re-zipped, and re-hosted to [Google Drive](https://drive.google.com/file/d/1gI-9HDwxmY-39_JKqCzy3IBos6Zd4Cnk/view?usp=sharing) with public ("Anyone with the link") access for convenience — this is the copy this repository's evaluation pipeline actually downloads from, documented below. 

#### Download

 The pipeline's own notebook (notebooks/m5\_final\_evaluation.ipynb)  
 wraps this in a skip-if-already-present check, so re-running the same cell  
 mid-session does not re-download unnecessarily:    	  
 

```py
from pathlib import Path	      	MINIDEV_ROOT = Path("/content/MINIDEV")	if MINIDEV_ROOT.exists() and any(MINIDEV_ROOT.iterdir()):	  print(f"Already extracted at {MINIDEV_ROOT}, skipping download.")	else:	  !pip install gdown --quiet	  !gdown 1gI-9HDwxmY-39_JKqCzy3IBos6Zd4Cnk -O /content/MINIDEV.zip	  
  import zipfile	  with zipfile.ZipFile("/content/MINIDEV.zip") as z:	     z.extractall("/content/")    
  print(f"Extraction complete: {MINIDEV_ROOT}")	      	
assert MINIDEV_ROOT.exists(), f"{MINIDEV_ROOT} not found -- check /content/ contents"	
```

      	  
Expected contents after extraction:

```
     	     MINIDEV/	 	├── mini_dev_sqlite.json     	# 500 questions	 	├── mini_dev_sqlite_gold.sql 	# 500 gold SQL queries	 	└── dev_databases/           	# 11 SQLite database files	      	
```

**Extraction must target local disk** (/content/MINIDEV), not a Drive-mounted path — SQLite files read directly from a Drive FUSE mount  raise disk I/O error when queried during evaluation. This download and extraction must be repeated every fresh Colab session, since anything under /content/ (as opposed to /content/drive/) is wiped when the  runtime resets — this is by design, not a bug, and is why the pipeline's own notebook always includes this step near the top rather than assuming it persists.

### 4\. Running the Pipeline

#### Step 0 — Clone the repository and resolve the adapter 

```py
import os, shutil	
from pathlib import Path	      	REPO_URL = "https://github.com/siddhant-192/Group-10-DS-and-AI-Lab-Project.git"	BRANCH = "main"
os.chdir("/content")
	dest = Path("/content/repo")	if dest.exists():	   shutil.rmtree(dest)	      	!git clone --depth 1 -b {BRANCH} "{REPO_URL}" /content/repo	%cd /content/repo	ls src/scripts/evaluate_text2sql_models.py  # sanity check: confirms the clone succeeded	
```

Checkpoint 375's adapter weights are hosted on HuggingFace Hub under the account (`walz89/checkpoint-375-adapter`), uploaded there after training completed in Milestone 4, and are resolved to a local path here: 

```py
from huggingface_hub import snapshot_download   local_adapter_path = snapshot_download(repo_id="walz89/checkpoint-375-adapter")	print(local_adapter_path)
```

   
Resolving the adapter to a local path (above) cannot be skipped in favor of passing the repo ID directly — `evaluate_text2sql_models.py`'s `--adapter-dir` argument only accepts a real local directory (it checks `Path(...).is_dir()` before loading and raises `FileNotFoundError` otherwise); passing the raw HuggingFace repo ID string directly does not work with this script, even though `PeftModel.from_pretrained()` itself would accept either. Always resolve to a local path via `snapshot_download()` first, as shown above, and use the printed `local_adapter_path` value for every `--adapter-dir` argument below. 

 

**Stage by stage** 

If you want to inspect or modify intermediate outputs, run each stage directly. The commands below use `prepare_bird_mschema_eval.py` for the M-Schema (primary) configuration. For the other two configurations, swap in a different script/flag at Stage 1 only — Stages 2–4 are identical regardless of configuration, just pointed at that configuration's own files:

* **DDL configuration**: use `prepare_bird_eval.py` instead (no `--include-evidence` flag; it doesn't apply to this script).  
* **M-Schema \+ evidence configuration**: use `prepare_bird_mschema_eval.py` as shown below, with `--include-evidence` added.

#### Stage 1 — Prepare:

```py
python src/scripts/prepare_bird_mschema_eval.py \   	--bird-root /content/MINIDEV \	       --output-dir evidence/milestone5/bird \	       --render-script-dir src/scripts   
```

Output: evidence/milestone5/bird/bird\_eval\_mschema.jsonl (500 lines, one JSON object per example, containing rendered schema \+ question \+ gold SQL).  
 

#### Stage 2 — Evaluate (GPU required):

```py
python src/scripts/evaluate_text2sql_models.py \      	--config configs/text2sql_eval_models.json \	       --manifest models/download_manifest.json \	   	--data evidence/milestone5/bird/bird_eval_mschema.jsonl \	   	--model qwen3-4b-instruct-2507 \	       --adapter-dir {local_adapter_path} \          --adapter-label milestone4_frozen \          --cache-dir ~/.cache/huggingface \          --batch-size 16 \	       --output-dir evidence/milestone5/evaluation_mschema	
```

      	  
Output: evidence/milestone5/evaluation\_mschema/milestone4\_frozen/predictions.jsonl and metrics.json. Resumable — safe to interrupt and re-run with the same \--output-dir; already-completed examples are skipped, not redone.

 

#### Stage 3 — Score:

```py
python src/scripts/score_predictions_macsql.py \          --predictions milestone4_frozen=evidence/milestone5/evaluation_mschema/milestone4_frozen/predictions.jsonl \	       --validation evidence/milestone5/bird/bird_eval_mschema.jsonl \	       --project-root . \	       --output-dir evidence/milestone5/macsql_scoring_mschema
```

	  
      	  
Output: metrics.json with both strict and MAC-SQL/FINER-compatible accuracy, computed independently from the raw predictions (a second, separate scoring pass — not just reading back a stored field).

 

#### Stage 4 — Analyze:

`analyze_eval_results.py` prints the headline strict execution accuracy — the fraction of predictions whose result set exactly matches gold — along with gold/prediction execution status counts.

```py
python src/scripts/analyze_eval_results.py \	       --predictions evidence/milestone5/evaluation_mschema/milestone4_frozen/predictions.jsonl
```

`error_taxonomy_analysis.py` categorizes every prediction into one of six failure types (correct, semantic mismatch, hallucinated column, hallucinated table, syntax error, other execution error), reports the percentage breakdown, and saves representative examples per category to the output JSON — this is the source of the error taxonomy table in the Milestone 5 report (Section 10.1). 

```py
python src/scripts/error_taxonomy_analysis.py \          --predictions evidence/milestone5/evaluation_mschema/milestone4_frozen/predictions.jsonl \	       --examples-per-category 3 \	       --output-json evidence/milestone5/error_taxonomy_summary_mschema.json	
```

### **Additional analyses (M-Schema configuration only, matching the report**

```py
	# Post-hoc hallucinated-column repair experiment (Section 10.7)	 	python src/scripts/repair_hallucinated_columns.py \          --predictions evidence/milestone5/evaluation_mschema/milestone4_frozen/predictions.jsonl \	       --bird-eval evidence/milestone5/bird/bird_eval_mschema.jsonl \	   	--output evidence/milestone5/hallucinated_column_repair_report.json	      	 	# Schema-linking precision (Section 10.4)    	python src/scripts/schema_linking_precision.py \      	--predictions evidence/milestone5/evaluation_mschema/milestone4_frozen/predictions.jsonl \	       --bird-eval evidence/milestone5/bird/bird_eval_mschema.jsonl \	   	--output evidence/milestone5/schema_linking_precision.json	      	 	# Adversarial robustness test (Section 11.2) -- requires GPU, real inference	 	python src/scripts/test_robustness_prompts.py \          --eval-script-dir src/scripts \          --database /content/MINIDEV/dev_databases/debit_card_specializing/debit_card_specializing.sqlite \	   	--db-id debit_card_specializing \	       --adapter-dir {local_adapter_path} \      	--output evidence/milestone5/robustness_test_report.json	      	 	# Latency / VRAM summary (Section 12) -- reads existing run artifacts, no GPU needed	 	python src/scripts/summarize_latency_vram.py \          --predictions evidence/milestone5/evaluation_mschema/milestone4_frozen/predictions.jsonl \	       --metrics evidence/milestone5/evaluation_mschema/milestone4_frozen/metrics.json \	       --run-label "M-Schema (primary)"	
```

      	

### 5\. Expected Outputs — Verification Table

After running all three configurations in full, confirm your outputs match:

| File | Key field | Expected value |
| :---- | :---- | :---- |
| evidence/milestone5/evaluation\_ddl/milestone4\_frozen/predictions.jsonl | strict EX (via analyze\_eval\_results.py) | 16.8% (84/500) |
| evidence/milestone5/evaluation\_mschema/milestone4\_frozen/predictions.jsonl | strict EX | 25.8% (129/500) |
| evidence/milestone5/evaluation\_mschema\_evidence/milestone4\_frozen/predictions.jsonl | strict EX | 30.8% (154/500) |
| evidence/milestone5/error\_taxonomy\_summary\_mschema.json | hallucinated\_column % | 12.0% (60/500) |
| evidence/milestone5/error\_taxonomy\_summary\_mschema.json | semantic\_mismatch % | 54.8% (274/500) |
| evidence/milestone5/schema\_linking\_precision.json | table-reference precision | 100.0% (1029/1029) |
| evidence/milestone5/schema\_linking\_precision.json | column-reference precision | 82.8% (2466/2977) |
| evidence/milestone5/hallucinated\_column\_repair\_report.json | repaired\_and\_matches\_gold | 0 (out of 60\) |
| evidence/milestone5/robustness\_test\_report.json | prompts staying read-only | 18/18 |

 All gold SQL should execute successfully: gold\_execution\_status: {"ok": 500} in every analyze\_eval\_results.py run — if this is anything other than 500, the gold-SQL parsing step (Section 6 below) did not work correctly.  
Small numeric drift (±1 example, i.e. ±0.2 percentage points) between runs  can occur if batch size or hardware differs from the L4 GPU used to produce the reported numbers, since floating-point kernel behavior can vary slightly across hardware even under greedy decoding. Larger discrepancies indicate an environment or data problem, not expected noise.

### 6\. Troubleshooting — Real Issues Encountered During Development

These are documented from issues actually hit while building this  
 pipeline..

**torchao import/version error when loading the adapter**  
 → Pin torchao==0.18.0 explicitly; Colab's default (0.10.0) is incompatible with the peft version this pipeline uses.

**disk I/O error when executing any SQL against a BIRD database**  
 → The .sqlite files are being read directly from a Google Drive-mounted path. Copy MINIDEV to local disk (/content/MINIDEV, not /content/drive/...) before running any evaluation stage — see Section 3\.

**Gold SQL execution errors / gold\_execution\_status not showing 500/500**  
 → BIRD's mini\_dev\_sqlite\_gold.sql uses inconsistent tab/space delimiters between the SQL and the trailing db\_id. The prep scripts use a validity-driven parser (try the raw line as-is first; only strip a trailing db\_id if that specific change makes the query executable) rather than a fixed delimiter split. 

**CUDA out of memory during Stage 2**  
 → Reduce \--batch-size.. The evaluation script also has automatic OOM-triggered batch-size reduction built in; check the printed log for a "reducing batch size" message before assuming a hard failure.

**Evaluation appears to restart from zero after an interruption**  
 → Confirm \--output-dir is identical to the interrupted run. The pipeline is resumable by design (checks predictions.jsonl for already-completed IDs), but a different \--output-dir will start a fresh run instead of resuming.

 

**FileNotFoundError on \--adapter-dir when passing the raw HuggingFace repo ID**  
 → evaluate\_text2sql\_models.py and test\_robustness\_prompts.py both require a real local directory for \--adapter-dir — they check Path(...).is\_dir() before loading, which a bare repo ID string like walz89/checkpoint-375-adapter fails. Always resolve it first: from huggingface\_hub import snapshot\_download; local\_adapter\_path \= snapshot\_download(repo\_id="walz89/checkpoint-375-adapter"), then use the printed path. See Section 4, Step 0\.

 

### 7\. How This Maps to the Milestone 5 Report

| Report section | Produced by |
| :---- | :---- |
| Executive Summary table, Section 6.1 | Stages 3–4, all three configurations |
| Section 9 (Ablation Study) | Same data as above, compared across configurations |
| Section 10.1 (Error Taxonomy) | error\_taxonomy\_analysis.py |
| Section 10.3 (Mechanism Testing) | Custom pandas.crosstab analysis over predictions.jsonl (see notebooks/m5\_final\_evaluation.ipynb, "Testing the Predicted Mechanisms" section) |
| Section 10.4 (Schema-Linking Precision) | schema\_linking\_precision.py |
| Section 10.5 (3-JOIN Anomaly) | Custom analysis in notebooks/m5\_final\_evaluation.ipynb, "Detailed Analysis of Queries by Number of JOINs" section |
| Section 10.7 (Post-Hoc Repair) | repair\_hallucinated\_columns.py |
| Section 11.2 (Robustness) | test\_robustness\_prompts.py |
| Section 12 (Computational Performance) | summarize\_latency\_vram.py |

 

## Deployment

This section describes how to deploy and verify the "Talk to Your Database" app as specified in the final project report (Section 6, Milestone 6). The only accepted deployment method for this deliverable is a Google Colab demo notebook — there is no localhost, mock-backend, or Hugging Face Spaces deployment path for this milestone. Every step below matches the report's own documented process; troubleshooting entries come from issues actually encountered during development.

### 0\. Order of Operations

Opening the notebook (Colab's File → Open Notebook → GitHub, repository `siddhant-192/Group-10-DS-and-AI-Lab-Project`, path `app/scripts/colab_qwen3/Colab_UI_Qwen3.ipynb`) and obtaining the actual project code are two different actions, not one. Opening the notebook only loads that single `.ipynb` file into Colab's editor — none of `app/`, `src/`, or any other project folder exists on the Colab machine yet at that point. The real code transfer happens when you **run Cell 0**, which executes: 

```py
!git clone --depth 1 -b milestone-6-ui-2 \
    "https://github.com/siddhant-192/Group-10-DS-and-AI-Lab-Project.git" \
    /content/repo
```

Only after this command completes does `/content/repo/app/app.py` (and everything else the rest of this document references) actually exist on the Colab virtual machine's disk. 

| Step | What you obtain | Where it comes from |
| ----- | ----- | ----- |
| 1 | This repository's code | `git clone` inside Cell 0 (see above)  |
| 2 | Python package environment | `app/scripts/colab-ui-requirements.txt`, installed by the notebook's own cells |
| 3 | Demo databases (Chinook, mini\_music) | Downloaded by the notebook's own cells into `demo_databases/` |
| 4 | Model adapter (checkpoint 375\) | Downloaded from HuggingFace Hub by the notebook's own cells |
| 5 | Running app instance | Started by the notebook's Streamlit-launch cell |

### 1\. What This Deploys

**Purpose**: let a non-technical user query a SQLite database in plain English and inspect the SQL that was actually run.

**Architecture (data flow):**

```
Colab notebook (app/scripts/colab_qwen3/Colab_UI_Qwen3.ipynb)
  -> Streamlit on the Colab VM
  -> user opens the Colab proxy URL
  -> optional one-shot clarification (app/backend/clarify.py)
  -> ask(question, db_id)  # app/backend/ask.py
  -> resolve demo database
  -> render M-Schema
  -> Qwen3 + PEFT adapter
  -> extract SQL
  -> src.validation.validate_readonly_query  # SELECT/WITH only
  -> readonly SQLite execute (timeout, row cap)
  -> template answer + rule-based chart
```

**What is deployed, and where:**

| Component | Where it runs |
| ----- | ----- |
| Launch | Colab notebook under `app/scripts/colab_qwen3` |
| Frontend \+ orchestration | Streamlit on the Colab VM |
| Model | In-process HuggingFace/PEFT on the Colab GPU |
| Database | `demo_databases/*.sqlite` on the Colab disk |
| How to open the app | Colab proxy URL printed by the notebook |

No FastAPI service or `/predict` endpoint is provided \-- `ask()` in `app/backend/ask.py` is the single entry point, called directly by the Streamlit UI.

### 2\. Notebook used

|  | Official (primary) |
| ----- | ----- |
| Notebook | `app/scripts/colab_qwen3/Colab_UI_Qwen3.ipynb` |
| Model | Qwen3-4B-Instruct-2507 \+ checkpoint 375 adapter |
| GPU required | Colab **Pro**, \~8-12+ GB VRAM in 4-bit |
| Sidebar should show | `qwen3-4b+adapter` |

### 3\. Environment Setup

**Key pins** (from `app/scripts/colab-ui-requirements.txt`):

| Package | Version |
| ----- | ----- |
| `transformers` | 4.57.6 |
| `peft` | 0.19.1 |
| `bitsandbytes` | 0.49.2 |

These are installed automatically by the notebook's own install cell \-- no separate manual `pip install` step is part of the documented process for this milestone.

### 4\. Launch Steps (Official \-- Qwen3 \+ Adapter)

1. Open Colab. **File \-\> Open Notebook \-\> GitHub.**  
2. Repository: `siddhant-192/Group-10-DS-and-AI-Lab-Project`  
3. Path: `app/scripts/colab_qwen3/Colab_UI_Qwen3.ipynb`  
4. **Runtime \-\> Change runtime type \-\> GPU** (Colab Pro, for sufficient VRAM).  
5. Run cells in order:  
   * **Cell 0 (Clone repository)** clones the project (`BRANCH = "milestone-6-ui-2"`, locates `app/app.py` inside the clone, and verifies `app/backend/models.py` exists \-- raising `FileNotFoundError` immediately if the branch doesn't contain the Streamlit UI, rather than failing confusingly later.  
   * **Cell 1 (Dependencies and GPU)** installs `app/scripts/colab-ui-requirements.txt`, then explicitly asserts `torch.cuda.is_available()` and prints the GPU name and approximate VRAM \-- confirms a GPU is genuinely present, not just assumed.  
   * **Cell 2 (Adapter directory)** downloads the adapter via `snapshot_download(repo_id="walz89/checkpoint-375-adapter")` **and** writes `app/ui_config.json` in the same cell (`backend`, `model_slug`, `adapter_dir`, `load_4bit`, `max_new_tokens`).  
   * **Cell 3 (Write `ui_config.json`)** writes the same configuration a second time \-- intentional redundancy in the notebook (harmless, idempotent), not a separate required step.  
   * **Cell 4 (Demo databases and Streamlit)** downloads demo databases, **asserts** the config's `backend` is `qwen3-4b+adapter` and `adapter_dir` is set (fails loudly with a clear message if Cells 2-3 weren't run correctly), writes `.streamlit/config.toml`, kills any process already on port 8501, and launches Streamlit \-- polling for up to 90 seconds until the port opens.  
   * **Cell 5 (Colab proxy URL)** checks the port is actually open first (raises a clear error telling you to re-run Cell 4 if not), then prints the clickable **Colab proxy URL** via `google.colab.kernel.proxyPort(8501)`. Open this link, not `localhost:8501` \-- the model runs on Colab, not your laptop.  
6. Confirm the sidebar shows **`qwen3-4b+adapter`** before proceeding.

**Optional cleanup**: an **"Optional \-- stop Streamlit"** cell is provided at the end of the notebook (commented out by default) to kill the process on port 8501 \-- useful for a full app restart without restarting the whole Colab runtime.

**Note on Cell 4's port-kill step**: it uses `fuser -k 8501/tcp` followed by a fixed 1-second wait before launching a new Streamlit process. If a prior process doesn't fully release the port within that second, the new launch can end up talking to a stale process instead of the freshly started one \-- if code changes don't appear to take effect after restarting, this is the first thing to check (see Section 8).

### 5\. Interacting With the App

1. Click **"Warm up model"** in the sidebar and wait for **"Model ready."** \-- loads the model once so your first real question isn't slow.  
2. Select a database from the **Database** dropdown.  
3. Type your question in plain English (**at least four words** when possible \-- short or vague questions will trigger a clarification), or load an example question.  
4. Click **Run**.  
5. If prompted to clarify, either add detail or click **"Continue without clarification"** to proceed with the system's most reasonable interpretation.  
6. Review: **Answer**, generated **SQL**, its **"In plain English"** explanation, **Result table**, and **Chart**.  
7. **Read the SQL before trusting the numbers.**

### 6\. Verification \-- Defined Smoke Test and Examples

| Database | Question | Expected |
| ----- | ----- | ----- |
| `mini_music` | "How many singers are there?" | **3** |
| `mini_music` | "List all singers" | Table of names |
| `chinook` | "How many albums are there?" | About 347 |
| `chinook` | "Top three albums" | Clarification first (rank by what?), then a result if you continue |
| `chinook` | "Show me the best" | Clarification: "best" is undefined |

**The primary smoke test for confirming a working deployment is the first row**: `mini_music` \+ "How many singers are there?" must return **3**. This is the fastest way to confirm the app, model, and adapter are all correctly wired together.

gm