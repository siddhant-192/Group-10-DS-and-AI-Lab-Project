# Colab — Qwen3-4B + QLoRA

GPU: Colab Pro recommended (~8–12+ GB VRAM for 4-bit Qwen3-4B + adapter).

| File | Role |
|------|------|
| `Colab_UI_Qwen3.ipynb` | Clone repo, set adapter path, start Streamlit |

## Adapter

Provide a directory that contains PEFT artifacts (`adapter_config.json`,
`adapter_model.safetensors`). A zip such as
`final_checkpoint_375_adapter_upload-….zip` unpacks to
`final_checkpoint_375_adapter_upload/`; use that directory as `adapter_dir`.

The base weights (`Qwen/Qwen3-4B-Instruct-2507`) are downloaded from Hugging Face
at runtime. They are not part of the adapter archive.

Cell 2 can mount Google Drive, locate the zip or folder, and set `ADAPTER_DIR`.

## Runtime wiring

`app/ui_config.json` must set `backend` to `qwen3-4b+adapter` and `adapter_dir`
to that directory. `app/backend/models.py` loads the base model, then applies
the adapter with PEFT.

Use this notebook for Qwen3. `Colab_UI_Qwen25.ipynb` installs the Qwen2.5 config.

## Branch

Cell 0 defaults to `milestone-6-ui`. Switch to `main` after the UI is merged.
