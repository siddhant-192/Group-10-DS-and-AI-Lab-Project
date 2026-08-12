# Deployment options

Run Streamlit on a GPU host (lab machine or Colab). A reverse tunnel is optional
when a temporary public URL is needed.

| Path | Model | Notes |
|------|--------|------|
| Local / lab GPU (CLI) | Qwen3 + adapter (or Qwen2.5) | Release / official path |
| Colab (T4) | Qwen2.5 | `Colab_UI_Qwen25.ipynb` |
| Colab Pro | Qwen3 + adapter | `colab_qwen3/Colab_UI_Qwen3.ipynb` |

Qwen3-4B + QLoRA needs about **8–12+ GB** VRAM in 4-bit. Details:
[`Official_Model_GPU_Run.md`](Official_Model_GPU_Run.md).

## 1 — Local or lab GPU (CLI)

```bash
cd <repository-root>

cp app/ui_config.qwen3.example.json app/ui_config.json
# Windows: copy app\ui_config.qwen3.example.json app\ui_config.json

# Set adapter_dir in ui_config.json to the QLoRA directory on this machine.

pip install -r app/scripts/colab-ui-requirements.txt
python app/scripts/download_demo_databases.py
streamlit run app/app.py --server.address 0.0.0.0 --server.port 8501
```

Open `http://localhost:8501`. For Qwen2.5, copy `ui_config.qwen25.example.json`
instead and leave `adapter_dir` as `null`.

### Temporary public URL

```bash
cloudflared tunnel --url http://127.0.0.1:8501
# or: ngrok http 8501
```

## 2 — Google Colab

**Qwen2.5:** `app/scripts/Colab_UI_Qwen25.ipynb` — open the Colab proxy URL from the notebook.

**Qwen3 + adapter:** `app/scripts/colab_qwen3/Colab_UI_Qwen3.ipynb`  
(see [`Official_Model_GPU_Run.md`](Official_Model_GPU_Run.md)).

## Out of scope

Always-on hosted Spaces and permanent cloud hosting are outside Milestone 6.
Localhost and Colab cover the demo.
