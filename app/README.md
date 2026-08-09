# Talk to Your Database — Application UI

Streamlit front end for the Group-10 text-to-SQL pipeline (Milestone 3 core:
M-Schema prompting, readonly validation, SQLite execution).

**Primary model (evaluation / release):** Qwen3-4B + QLoRA adapter  
**Secondary runtime (resource-constrained GPU):** Qwen2.5-Coder-1.5B-Instruct  

## Architecture

```text
Streamlit (app/app.py)
  → optional one-shot clarification
  → ask(question, db_id, …)          # app/backend/ask.py
      → demo database registry
      → M-Schema prompt
      → model backend (mock | qwen2.5-1.5b | qwen3-4b+adapter)
      → extract SQL → validate readonly → execute
      → template answer + rule-based chart
```

### In scope / out of scope

| Included | Not included |
|----------|--------------|
| One model generation per answer (after optional clarification) | Multi-turn chat memory |
| Schema-aware clarification heuristics | LLM ambiguity classifier |
| Rule-based charts + manual override | LLM-selected chart types |
| Template short answers | Free-form narrative summaries |

Clarification uses table and column names from the **selected** database (DB-agnostic).
Changing Qwen2.5 ↔ Qwen3 is a configuration change only.

## Run — local (CPU, mock generator)

From the **repository root** (directory that contains `app/`):

```bash
python -m pip install -r requirements-ui.txt
python app/scripts/download_demo_databases.py
streamlit run app/app.py
```

Default `app/ui_config.json` uses `"backend": "mock"` (deterministic SQL stub; execution still uses real SQLite files).

## Run — Google Colab (GPU, Qwen2.5)

1. Use notebook [`scripts/Colab_UI_Qwen25.ipynb`](scripts/Colab_UI_Qwen25.ipynb).
2. Runtime → GPU (T4).
3. Run cells **0 → 4** in order (Cell 0 clones or unpacks the repo, then locates `app/`).
4. Open the **Colab proxy URL** printed by cell 4.

Details: [`scripts/COLAB_UI_RUNBOOK.md`](scripts/COLAB_UI_RUNBOOK.md).

CLI smoke test (no UI):

```bash
python app/scripts/colab_qwen25_smoke.py --db-id mini_music \
  --question "How many singers are there?"
```

## Run — official model (Qwen3-4B + adapter)

Requires a GPU machine with sufficient VRAM for 4-bit Qwen3-4B + adapter.

1. Copy [`ui_config.qwen3.example.json`](ui_config.qwen3.example.json) → `ui_config.json`.
2. Set `adapter_dir` to the verified QLoRA directory.
3. From the repository root:

```bash
pip install -r app/scripts/colab-ui-requirements.txt
python app/scripts/download_demo_databases.py
streamlit run app/app.py
```

| Setting | Qwen2.5 demo | Qwen3 + adapter |
|---------|--------------|-----------------|
| `backend` | `qwen2.5-1.5b` | `qwen3-4b+adapter` |
| `model_slug` | `qwen2.5-coder-1.5b-instruct` | `qwen3-4b-instruct-2507` |
| `adapter_dir` | `null` | path to adapter |

See [`Docs/Milestone 6/Milestone_6_Deployment_Demo.md`](../Docs/Milestone%206/Milestone_6_Deployment_Demo.md)
and [`Docs/Milestone 6/Official_Model_GPU_Run.md`](../Docs/Milestone%206/Official_Model_GPU_Run.md).

## Deployment options

| Level | Description |
|-------|-------------|
| Local / lab GPU (CLI) | Official model or Qwen2.5 via `streamlit run` |
| Colab | Qwen2.5 on T4; Qwen3 + adapter on Pro / higher VRAM |
| Temporary reverse tunnel | Optional public URL from a GPU host |

Guide: [`Docs/Milestone 6/DEPLOY.md`](../Docs/Milestone%206/DEPLOY.md)

## Sample questions

1. `mini_music` — How many singers are there? → **3**
2. `mini_music` — List all singers
3. `chinook` — How many albums are there? → **~347**

## Demo databases

See [`demo_databases/README.md`](../demo_databases/README.md).

## Code reuse from the Milestone 3 tree

| Component | Origin |
|-----------|--------|
| M-Schema | `src/scripts/build_xiyan_mschema_eval_data.py` → `app/backend/mschema.py` |
| Prompt | `src/scripts/smoke_final_model.py` → `app/backend/prompt.py` |
| SQL extract / execute | `src/scripts/evaluate_text2sql_models.py` → `app/backend/sql_utils.py` |
| Safety | `src.validation.validate_readonly_query` |
| Model IDs | `configs/text2sql_eval_models.json` |

## Out of scope / future work

- FastAPI service layer  
- FAISS schema retrieval (large schemas)  
- Multi-retry execution self-correction  
- Always-on production hosting  

