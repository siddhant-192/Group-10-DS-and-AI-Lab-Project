# Colab — Qwen3-4B + QLoRA adapter (official UI)

Use this folder on **Colab Pro** (or any GPU with ~8–12+ GB VRAM).

| File | Purpose |
|------|---------|
| `Colab_UI_Qwen3.ipynb` | Run cells in order: clone repo → upload adapter → config → Streamlit |

## What you must upload

The **adapter folder** from the team release (QLoRA weights), for example:

- `adapter_config.json`
- `adapter_model.safetensors` (or `.bin` shards)

Typical path after upload: `/content/final_adapter`

The **base model** (`Qwen/Qwen3-4B-Instruct-2507`) is **not** uploaded.  
Hugging Face downloads it when Streamlit starts (`load_4bit: true`).

## How the adapter is wired (in code)

1. Notebook writes `app/ui_config.json` with:
   - `"backend": "qwen3-4b+adapter"`
   - `"adapter_dir": "/content/final_adapter"` (your path)
2. `app/backend/models.py` loads the base CausalLM, then:
   `PeftModel.from_pretrained(base, adapter_dir)`
3. If `adapter_dir` is missing or wrong, startup fails with a clear error  
   (Qwen3 backend **requires** an adapter).

## Do not use the Qwen2.5 notebook for this

`Colab_UI_Qwen25.ipynb` Cell 3 always copies the Qwen2.5 config.  
Use **this** notebook for Qwen3.

## Repo branch

Cell 0 defaults to GitHub `main`. The full UI must already be on `main`  
(or change `BRANCH` to the branch that contains `app/backend/`).
