# Deployment options

Keep this simple: run Streamlit on a GPU (lab machine or Colab). Optional
tunnel only if you need a temporary public link.

| Path | Model | Notes |
|------|--------|------|
| Local / lab GPU (CLI) | Qwen3 + adapter (or Qwen2.5) | Primary for the official model |
| Colab (free T4) | Qwen2.5 | Notebook + Colab proxy URL |
| Colab Pro / roomy GPU | Qwen3 + adapter | Same UI; different config + adapter |

Qwen3-4B + QLoRA needs enough VRAM (about **8–12+ GB** in 4-bit). Details:
[`Official_Model_GPU_Run.md`](Official_Model_GPU_Run.md).

## 1 — Local or lab GPU (CLI)

```bash
cd <repository-root>   # folder that contains app/

# Linux / Colab shell:
cp app/ui_config.qwen3.example.json app/ui_config.json
# Windows:
# copy app\ui_config.qwen3.example.json app\ui_config.json

# Edit ui_config.json: set adapter_dir to the QLoRA folder on this machine.

pip install -r app/scripts/colab-ui-requirements.txt
python app/scripts/download_demo_databases.py
streamlit run app/app.py --server.address 0.0.0.0 --server.port 8501
```

Open `http://localhost:8501`. Sidebar should show `qwen3-4b+adapter`.

For Qwen2.5 on the same machine, copy `ui_config.qwen25.example.json` instead
and leave `adapter_dir` as `null`.

### Optional temporary public URL

```bash
cloudflared tunnel --url http://127.0.0.1:8501
# or: ngrok http 8501
```

Stop Streamlit and the tunnel when the demo ends.

## 2 — Google Colab

**Qwen2.5 (typical free T4):**  
`app/scripts/Colab_UI_Qwen25.ipynb` and `app/scripts/COLAB_UI_RUNBOOK.md`.  
Open the **Colab proxy URL** from the notebook (not localhost on your laptop).

**Qwen3 + adapter:**  
Use a Pro / higher-VRAM runtime, upload the adapter, and follow
[`Official_Model_GPU_Run.md`](Official_Model_GPU_Run.md) Path B. Do not keep the
notebook step that copies the Qwen2.5 example config.

## What we are not using

Always-on hosted Spaces (CPU mock UI, paid GPU Spaces, etc.) are out of scope
for this milestone. Localhost / Colab is enough for the demo.
