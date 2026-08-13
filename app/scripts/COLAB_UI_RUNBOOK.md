# Colab UI runbook (Qwen2.5)

## Procedure

1. Provide the project via **GitHub clone** or a zip of the repository root
   (the folder that contains `app/`).
2. Runtime → **GPU (T4)**.
3. Open `app/scripts/Colab_UI_Qwen25.ipynb`.
4. Run **Cell 0** (obtains the code, then discovers the project root by finding `app/app.py`).
   Confirm output mentions clarification module `schema-aware-v2`.
5. Run cells **1 → 2 → 3 → 4**.
6. Cell 3 must complete with a ready message (spinner stops).
7. Cell 4 prints a Colab proxy URL — open that link.

Sidebar should show live backend `qwen2.5-1.5b`.

## Cell 3 does not finish

1. Interrupt the cell.  
2. Prefer the notebook that launches Streamlit with `system_raw` and `nohup`
   (background process), not a blocking `!python` launcher.  
3. Re-open the updated notebook from the current repository.

## Blank page or WebSocket origin errors

Start Streamlit with:

```text
--server.enableCORS false --server.enableXsrfProtection false
```

Stop listeners on port 8501, re-run Cell 3, then Cell 4.
Prefer the proxy URL from Cell 4 over Streamlit’s “External URL”.

## Avoid

- Opening `localhost:8501` on the local Windows machine while expecting the Colab model  
- Relying on Streamlit External URL on Colab (often refused)

## Stop the server

```bash
fuser -k 8501/tcp
```
