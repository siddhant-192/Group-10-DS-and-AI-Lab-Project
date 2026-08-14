# Official model — GPU inference (Qwen3-4B + QLoRA)

The Streamlit UI is shared across backends. Switching to Qwen3 requires
`app/ui_config.json` and a local QLoRA adapter directory.

Qwen3-4B in 4-bit with an adapter typically needs about **8–12+ GB** VRAM
(Colab Pro / L4 or a roomy T4). Free-tier T4 sessions often run out of memory.

## Requirements

- Repository root containing `app/`
- QLoRA adapter directory (PEFT files)
- GPU with sufficient VRAM
- Packages from `app/scripts/colab-ui-requirements.txt`

## Configuration

```bash
cd <repository-root>
cp app/ui_config.qwen3.example.json app/ui_config.json   # Linux / Colab
# Windows: copy app\ui_config.qwen3.example.json app\ui_config.json
```

Edit `app/ui_config.json`:

- `"backend": "qwen3-4b+adapter"`
- `"model_slug": "qwen3-4b-instruct-2507"`
- `"adapter_dir": "<path-to-adapter>"`
- `"max_new_tokens": 512`

## Path A — Local or lab GPU

```bash
pip install -r app/scripts/colab-ui-requirements.txt
python app/scripts/download_demo_databases.py
streamlit run app/app.py --server.address 0.0.0.0 --server.port 8501
```

Open `http://localhost:8501`. Sidebar backend: `qwen3-4b+adapter`.

Optional tunnel:

```bash
cloudflared tunnel --url http://127.0.0.1:8501
```

## Path B — Colab Pro

Notebook: `app/scripts/colab_qwen3/Colab_UI_Qwen3.ipynb`

1. Runtime → GPU (Pro).
2. Cell 0 clones the repo (`BRANCH = milestone-6-ui` until merged to `main`).
3. Cell 2 sets `ADAPTER_DIR` to the unzipped adapter folder (Drive mount supported).
4. Remaining cells write `ui_config.json`, start Streamlit, and print the Colab proxy URL.

See `app/scripts/colab_qwen3/README.md`.

## Backend comparison

| | Qwen2.5 | Qwen3 + adapter |
|--|---------|-----------------|
| Config | `ui_config.qwen25.example.json` | `ui_config.qwen3.example.json` |
| `backend` | `qwen2.5-1.5b` | `qwen3-4b+adapter` |
| `adapter_dir` | `null` | required |
| Typical host | Colab T4 | Lab GPU or Colab Pro |
| UI | `app/app.py` | `app/app.py` |

## Notes

- Restart Streamlit after changing `ui_config.json`.
- Readonly validation is the same for both backends.
- CLI vs Colab overview: [`DEPLOY.md`](DEPLOY.md).
