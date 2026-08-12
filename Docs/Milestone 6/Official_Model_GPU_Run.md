# Official model — GPU inference (Qwen3-4B + QLoRA)

The Streamlit application is identical for Qwen2.5 and Qwen3. Only
`app/ui_config.json` (and the presence of an adapter directory) changes.

Qwen3-4B in 4-bit plus a QLoRA adapter typically needs about **8–12+ GB** VRAM.
That fits Colab **Pro** runtimes with a suitable GPU (often T4/L4); free-tier
T4 sessions are tighter and more likely to hit memory or quota limits.

The commands below are the portable procedure. The same steps apply inside
Colab (Terminal or notebook cells) after the project and adapter are on disk.

## Requirements

- Project tree: repository root containing `app/`
- Verified QLoRA adapter directory (for example `release/final_adapter`)
- GPU runtime with enough VRAM for 4-bit Qwen3-4B + adapter
- Python packages in `app/scripts/colab-ui-requirements.txt`

## Configuration (both local GPU and Colab)

```bash
cd <repository-root>
cp app/ui_config.qwen3.example.json app/ui_config.json   # Linux / Colab
# Windows: copy app\ui_config.qwen3.example.json app\ui_config.json
```

Edit `app/ui_config.json`:

- `"backend": "qwen3-4b+adapter"`
- `"model_slug": "qwen3-4b-instruct-2507"`
- `"adapter_dir": "<path-to-adapter>"` (must exist on that machine)
- `"max_new_tokens": 512`

## Path A — Local or lab GPU (CLI)

```bash
pip install -r app/scripts/colab-ui-requirements.txt
python app/scripts/download_demo_databases.py
streamlit run app/app.py --server.address 0.0.0.0 --server.port 8501
```

Open `http://localhost:8501`. Sidebar should show `qwen3-4b+adapter`.

Optional temporary public URL:

```bash
cloudflared tunnel --url http://127.0.0.1:8501
```

## Path B — Google Colab Pro (Qwen3 notebook)

Use the dedicated notebook (do **not** use the Qwen2.5 notebook Cell 3):

`app/scripts/colab_qwen3/Colab_UI_Qwen3.ipynb`

Comments in that notebook explain how the adapter is integrated:

1. Upload the QLoRA folder (e.g. `/content/final_adapter`).
2. Cell 3 writes `ui_config.json` with `backend: qwen3-4b+adapter` and `adapter_dir`.
3. `app/backend/models.py` loads HF base Qwen3-4B, then `PeftModel.from_pretrained(..., adapter_dir)`.

Folder README: `app/scripts/colab_qwen3/README.md`.

## Comparison — Qwen2.5 UI vs Qwen3 UI

| | Qwen2.5 demo | Qwen3 + adapter |
|--|--------------|-----------------|
| Purpose | UI / pipeline check on modest GPU | Official science model |
| Config file | `ui_config.qwen25.example.json` | `ui_config.qwen3.example.json` |
| `backend` | `qwen2.5-1.5b` | `qwen3-4b+adapter` |
| `adapter_dir` | `null` | required path |
| Typical host | Free Colab T4 | Lab GPU or Colab Pro |
| UI code | Same `app/app.py` | Same `app/app.py` |
| Databases | Same `demo_databases/` | Same |

## Notes

- One Streamlit process loads one backend. Restart after changing `ui_config.json`.
- Readonly validation (SELECT/WITH only) is identical for both models.
- Overview of CLI vs Colab: [`DEPLOY.md`](DEPLOY.md).
