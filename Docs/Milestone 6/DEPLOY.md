# Deployment options

## Overview

| Option | Public URL | Runs Qwen3-4B + adapter |
|--------|------------|-------------------------|
| GPU host + reverse tunnel | Temporary | Yes |
| Hugging Face Space (CPU) | Persistent | No (mock generator only) |
| Colab T4 session | Via Colab proxy | Qwen2.5 only (VRAM limits) |

Qwen3-4B with a QLoRA adapter requires a GPU host with adequate VRAM.
A free CPU Space can host the UI with the mock generator; it cannot load the
official 4B model.

**Recommended path for the official model:** run Streamlit on a GPU machine,
optionally expose it with Cloudflare Tunnel or ngrok. See
[`Official_Model_GPU_Run.md`](Official_Model_GPU_Run.md).

## Option A — GPU host and temporary tunnel

```bash
copy app\ui_config.qwen3.example.json app\ui_config.json
# set adapter_dir to the verified QLoRA directory

pip install -r app/scripts/colab-ui-requirements.txt
python app/scripts/download_demo_databases.py
streamlit run app/app.py --server.address 0.0.0.0 --server.port 8501
```

Second terminal:

```bash
cloudflared tunnel --url http://127.0.0.1:8501
# or: ngrok http 8501
```

Share the generated HTTPS URL for the session. Stop Streamlit and the tunnel
when finished.

## Option B — Hugging Face Space (mock UI)

Suitable for a persistent UI demonstration without a GPU.

1. Create a Streamlit Space on CPU hardware.  
2. Upload this project so the Space root contains `app/`, `src/`, and `configs/`.  
3. Keep `app/ui_config.json` with `"backend": "mock"`.  
4. Follow [`hf_space_README.md`](hf_space_README.md).

Loading Qwen3 on Spaces requires paid GPU hardware and uploading adapter weights.

## Option C — Colab with Qwen2.5

Use `app/scripts/Colab_UI_Qwen25.ipynb` when a T4 GPU is available.
This path is intended for the 1.5B demo runtime, not the official 4B adapter.
