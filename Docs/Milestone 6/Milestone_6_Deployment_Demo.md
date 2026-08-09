# Milestone 6 — Deployment and Demo UI

**Project:** Talk to Your Database (Group-10)  
**Scope:** Interactive application on the Milestone 3 text-to-SQL core  
**UI package:** `app/` (Streamlit, repository root)

## 1. Objective

Provide an end-to-end demo in which a user:

1. Selects a SQLite database  
2. Submits a natural-language question  
3. Optionally answers one clarifying question when the request is underspecified  
4. Receives validated SQL, a result table, a short answer, and a chart  

**Release model:** Qwen3-4B + QLoRA  
**Constrained-GPU runtime:** Qwen2.5-Coder-1.5B-Instruct  

## 2. Delivered

| Component | Status |
|-----------|--------|
| Streamlit UI and `ask()` orchestration | Complete |
| Demo database registry | Complete |
| M-Schema prompting | Complete |
| Readonly validation and execution | Complete |
| Rule-based charts and override | Complete |
| Template answers | Complete |
| Schema-aware one-shot clarification | Complete |
| Backends: mock / Qwen2.5 / Qwen3+adapter | Complete |
| Colab launch notebook (CORS-safe) | Complete |

## 3. Out of scope / future work

| Item | Rationale |
|------|-----------|
| FAISS retrieval | Needed when the full schema no longer fits the context |
| FastAPI `/ask` | Streamlit already runs the full pipeline |
| Multi-retry self-correction | Stretch goal from Milestone 1 |
| Multi-turn conversation memory | Not required for this milestone |
| Always-on cloud hosting | Localhost / Colab is enough for the demo |
| Retraining / Milestone 4 HPT | Handled on the model-training track |

## 4. Architecture

```text
Streamlit
  → optional clarification (terms from selected DB schema)
  → ask(question, db_id)
      → M-Schema
      → model (mock | Qwen2.5 | Qwen3+adapter)
      → extract SQL → validate readonly → execute
      → answer + chart
```

## 5. How to run

### Local mock generator (CPU)

```bash
cd <repository-root>   # directory that contains app/
python -m pip install -r requirements-ui.txt
python app/scripts/download_demo_databases.py
streamlit run app/app.py
```

### Colab + Qwen2.5 (GPU)

Follow `app/scripts/Colab_UI_Qwen25.ipynb` and `app/scripts/COLAB_UI_RUNBOOK.md`.

### Official model (Qwen3-4B + adapter)

See `Official_Model_GPU_Run.md`. Copy `app/ui_config.qwen3.example.json` to
`app/ui_config.json`, set `adapter_dir`, then start Streamlit on a suitable GPU host.

## 6. Suggested demonstration script

| # | Database | Question | Expected behaviour |
|---|----------|----------|--------------------|
| 1 | `mini_music` | How many singers are there? | Count **3** |
| 2 | `mini_music` | List all singers | Result table |
| 3 | `chinook` | How many albums are there? | About **347** |
| 4 | `chinook` | Underspecified ask (e.g. show best artists) | Clarification, then SQL |

## 7. Remaining ops

- Configure and smoke-test Qwen3 + adapter on a GPU host  
- Optional: temporary public tunnel from the GPU host (`DEPLOY.md`)
