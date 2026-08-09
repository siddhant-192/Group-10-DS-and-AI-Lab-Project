---
title: Talk to Your Database
emoji: 🗄️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.39.0"
app_file: app/app.py
pinned: false
---

# Talk to Your Database (hosted UI)

Intended configuration: `"backend": "mock"` on CPU hardware.
Demonstrates database selection, clarification, SQL display, results, and charts.

The release model (Qwen3-4B + QLoRA) requires a GPU host. See
`Docs/Milestone 6/Official_Model_GPU_Run.md`.

## Space setup

1. Space root must include `app/`, `src/`, and `configs/`.
2. Set `app/ui_config.json` to `"backend": "mock"`.
3. Populate demo databases:

```bash
python app/scripts/download_demo_databases.py
```

4. Root `requirements.txt` for a mock Space:

```text
streamlit>=1.32.0
pandas>=2.0.0
```

GPU stacks are not required for the mock backend.
