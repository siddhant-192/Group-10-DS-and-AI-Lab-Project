#!/usr/bin/env python3
"""Start/stop Streamlit + Qwen2.5 for Colab.

Prefer the notebook Cell 3 (system_raw + nohup) — that is the reliable path.
This CLI uses nohup too so `!python` does not hang on Streamlit children.

  python app/scripts/run_streamlit_localhost_colab.py
  python app/scripts/run_streamlit_localhost_colab.py --stop
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PORT = 8501
PID_FILE = Path("/tmp/streamlit_ui_qwen25.pid")
LOG_FILE = Path("/tmp/streamlit_ui.log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--skip-config-copy", action="store_true")
    parser.add_argument("--stop", action="store_true")
    return parser.parse_args()


def port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def stop_server(port: int) -> None:
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text(encoding="utf-8").strip())
            os.kill(pid, signal.SIGTERM)
            print(f"Sent SIGTERM to PID {pid}")
        except (ValueError, ProcessLookupError, PermissionError) as exc:
            print(f"PID file stop skipped: {exc}")
        try:
            PID_FILE.unlink(missing_ok=True)
        except OSError:
            pass
    try:
        subprocess.run(
            ["fuser", "-k", f"{port}/tcp"],
            check=False,
            capture_output=True,
        )
        print(f"Freed port {port}.")
    except FileNotFoundError:
        pass


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()

    if args.stop:
        stop_server(args.port)
        return 0

    os.chdir(root)

    try:
        import torch
    except ImportError:
        print("ERROR: torch not installed.", file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Use GPU T4 runtime.", file=sys.stderr)
        return 1

    print(f"CUDA OK: {torch.cuda.get_device_name(0)}")

    if not any((root / "demo_databases").glob("*.sqlite")):
        print("ERROR: no demo DBs — run download_demo_databases.py first.", file=sys.stderr)
        return 1

    example = root / "app" / "ui_config.qwen25.example.json"
    if not args.skip_config_copy and example.exists():
        shutil.copy2(example, root / "app" / "ui_config.json")
        print("ui_config.json <- qwen25 example")

    stop_server(args.port)
    time.sleep(1)

    # nohup + background shell so this process can exit without waiting on Streamlit
    shell_cmd = (
        f"MODEL_BACKEND=qwen2.5-1.5b MODEL_SLUG=qwen2.5-coder-1.5b-instruct "
        f"nohup {sys.executable} -m streamlit run {root / 'app' / 'app.py'} "
        f"--server.port {args.port} --server.address 0.0.0.0 --server.headless true "
        f"--server.enableCORS false --server.enableXsrfProtection false "
        f"--browser.gatherUsageStats false "
        f"> {LOG_FILE} 2>&1 & echo $!"
    )
    print("Starting Streamlit via nohup…")
    proc = subprocess.run(
        shell_cmd,
        shell=True,
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    pid_text = (proc.stdout or "").strip().splitlines()
    if pid_text:
        try:
            PID_FILE.write_text(pid_text[-1].strip(), encoding="utf-8")
            print(f"Background PID {pid_text[-1].strip()}")
        except OSError:
            pass

    for i in range(60):
        if port_open(args.port):
            print()
            print("READY — launcher exiting (Streamlit stays in background).")
            print(f"  port={args.port} backend=qwen2.5-1.5b")
            print("  Prefer notebook Cell 3+4. Stop: --stop")
            return 0
        time.sleep(1)
        if i % 10 == 9:
            print(f"  waiting for port… {i + 1}s")

    print("ERROR: port did not open. Tail of log:", file=sys.stderr)
    if LOG_FILE.exists():
        print(LOG_FILE.read_text(encoding="utf-8", errors="replace")[-2000:], file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
