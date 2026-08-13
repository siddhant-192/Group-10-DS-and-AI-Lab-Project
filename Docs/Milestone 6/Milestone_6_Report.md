# Milestone 6 Report: Deployment and Interactive Demo

**Talk to Your Database — A Natural-Language Analytics Copilot**

- **Course:** Data Science and AI Lab  
- **Milestone:** 6 — Deployment and Interactive Demo  
- **Submission date:** 13.08.2026

**Team members**

- Siddhant Hitesh Mantri (21f3002218)
- Anirudh Komanduri (22f1000522)
- Vishal S (23f2003089)
- Walunila Aier (21f3002564)
- Sambhav Jha (22f3003227)
- Smrutishikta Das (21f1006009)

---

## Executive Summary

Milestone 6 adds an interactive Streamlit application on top of the Milestone 3–5 text-to-SQL core. A user selects a SQLite database, asks a natural-language question, optionally answers one clarifying question when the request is underspecified, and receives validated SQL, a result table, a short answer, and a rule-based chart.

The **release model** remains **Qwen3-4B-Instruct + QLoRA** with an M-Schema prompt and readonly execution. A **secondary runtime**, **Qwen2.5-Coder-1.5B-Instruct**, supports demonstration on resource-constrained GPUs (for example Google Colab T4). Both runtimes share the same UI and orchestration code; only configuration differs.

This milestone does **not** retrain models or retune hyperparameters. Scope is the interactive UI: `ask` orchestration, clarification, charts, shared validation, and demo run paths (local, Colab, optional GPU tunnel).

---

## 1. Introduction

### 1.1 Project context

Prior milestones selected and evaluated a Qwen3-4B QLoRA system on Spider with M-Schema prompting and readonly SQLite execution. Milestone 6 puts that stack behind an interactive demo UI.

### 1.2 Objectives

1. Expose question → SQL → safe result as an interactive UI.  
2. Support multiple model backends without forking the pipeline.  
3. Add one-shot clarification for underspecified questions.  
4. Add rule-based charts and a short answer.  
5. Document how to run the demo on Colab (Qwen2.5) and on a GPU host (Qwen3 + adapter).  
6. Record out-of-scope items (FAISS, FastAPI, permanent hosting, multi-retry correction).

### 1.3 Relationship to earlier milestones

| Milestone | Contribution reused here |
|-----------|---------------------------|
| M1 | Clarification and visualization requirements |
| M3 | Architecture, M-Schema, readonly validation |
| M4 | Frozen QLoRA recipe and adapter |
| M5 | Evaluation evidence for the official model |

---

## 2. System design

### 2.1 Architecture

```text
Streamlit UI
  → optional one-shot clarification (schema terms from selected DB)
  → ask(question, db_id)
      → M-Schema serialization
      → model backend (mock | Qwen2.5 | Qwen3+adapter)
      → SQL extraction
      → validate_readonly_query (SELECT/WITH only)
      → readonly SQLite execute
      → template answer + rule-based chart
```

### 2.2 Module map

| Module | Location | Role |
|--------|----------|------|
| UI | `app/app.py` | Database picker, question, clarification, results |
| Orchestration | `app/backend/ask.py` | End-to-end request handling |
| Models | `app/backend/models.py` | Mock / Hugging Face backends |
| Clarification | `app/backend/clarify.py` | Underspecification heuristics (DB-agnostic) |
| Schema | `app/backend/mschema.py` | M-Schema from SQLite |
| Safety | `src/validation.py` | Readonly gate (shared with M3) |
| Charts | `app/backend/charts.py` | Metric / bar / line / scatter / table |
| Config | `app/ui_config.json` | Backend and adapter selection |

### 2.3 Safety

Generated SQL may include non-readonly statements (for example `ALTER`). Such outputs are rejected by validation before execution. Connections use readonly SQLite URIs. This behaviour is independent of which model backend is selected.

### 2.4 Clarification

Before generation, a lightweight heuristic uses table and column names from the **selected** database (not a hard-coded Chinook vocabulary). Underspecified questions trigger exactly one clarification turn; the user may supply detail or continue without it. A validation failure may offer a single repair clarification.

---

## 3. Model runtimes

### 3.1 Official runtime — Qwen3-4B + QLoRA

| Item | Value |
|------|--------|
| Base | Qwen3-4B-Instruct-2507 |
| Adaptation | Project QLoRA adapter |
| Config | `app/ui_config.qwen3.example.json` |
| Host | Lab GPU or Colab Pro with sufficient VRAM (≈8–12+ GB for 4-bit) |
| Procedure | `Docs/Milestone 6/Official_Model_GPU_Run.md` |

### 3.2 Demo runtime — Qwen2.5-Coder-1.5B

| Item | Value |
|------|--------|
| Base | Qwen2.5-Coder-1.5B-Instruct |
| Adapter | None |
| Config | `app/ui_config.qwen25.example.json` |
| Host | Google Colab T4 (typical) |
| Procedure | `app/scripts/Colab_UI_Qwen25.ipynb` |

### 3.3 Comparison (same UI)

| Aspect | Qwen2.5 | Qwen3 + adapter |
|--------|---------|-----------------|
| Role | Interactive demo under tight GPU limits | Release / evaluation model |
| `backend` | `qwen2.5-1.5b` | `qwen3-4b+adapter` |
| `adapter_dir` | `null` | Required |
| Typical launch | Colab notebook cells 0–4 | Config + `streamlit run` (local or Colab Pro) |
| Databases | Shared `demo_databases/` | Shared |

Switching models does **not** require a different UI codebase—only configuration and, for Qwen3, the adapter artifact.

---

## 4. Demonstration environment

### 4.1 Databases

| Database | Role |
|----------|------|
| `mini_music` | Tiny always-on smoke DB |
| `chinook` | Multi-table demo (albums, artists, customers, invoices, …) |

Spider remains the scientific evaluation corpus (Milestones 3–5). Chinook and `mini_music` are demonstration databases (Milestone 1 noted Chinook as a low-risk fallback).

### 4.2 Sample demonstration script

1. `mini_music` — How many singers are there? → 3  
2. `mini_music` — List all singers  
3. `chinook` — How many albums are there? → ≈347  
4. Underspecified question → clarification UI, then SQL  

### 4.3 Deployment levels

| Level | Description |
|-------|-------------|
| A | Localhost or Colab interactive session (primary) |
| B | Temporary reverse tunnel from a GPU host (optional) |
| C | Always-on cloud hosting | Out of scope for this milestone |

---

## 5. Out of scope / future work

| Item | Reason |
|------|--------|
| FAISS schema retrieval | Needed when full schema no longer fits context |
| FastAPI `/ask` service | Streamlit already runs the pipeline |
| Multi-retry self-correction | Stretch goal |
| Full multi-turn chat | Not required for this milestone |
| Permanent production hosting | Not required for the interactive demo |

---

## 6. Limitations

- Qwen2.5 demo accuracy is below the official Qwen3 adapter; it validates the product path, not Spider headline metrics.  
- Clarification is heuristic, not an LLM classifier.  
- Colab free-tier GPU quota and VRAM constrain which backend can be shown in a given session.  
- Demonstration databases are smaller than enterprise schemas; FAISS is not exercised here.

---

## 7. Conclusion

Milestone 6 connects the Milestone 3–5 text-to-SQL core to an interactive UI with safe SQL execution, charts, and one-shot clarification. Qwen3-4B + QLoRA remains the release model (GPU host or Colab Pro). Qwen2.5-Coder-1.5B is the constrained-GPU demo path on Colab T4.

---

## Appendix A — Key paths

- UI entry: `app/app.py`  
- Orchestration: `app/backend/ask.py`  
- Colab notebook: `app/scripts/Colab_UI_Qwen25.ipynb`  
- M6 operational notes: `Docs/Milestone 6/`  
- Adapter config example: `app/ui_config.qwen3.example.json`  

## Appendix B — How to produce the PDF

Export this Markdown file to PDF using one of:

- Pandoc: `pandoc Milestone_6_Report.md -o Milestone_6_Report.pdf`  
- Google Docs / Word: paste or import, then File → Download → PDF  
- VS Code Markdown PDF extension
