"""Launch one or two resumable QLoRA phases inside the active Colab runtime."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path("/content/text2sql_sft")
CONFIG = ROOT / "launch_config.json"
STATUS = ROOT / "status.json"
OUTPUT = ROOT / "output"
launch = json.loads(CONFIG.read_text(encoding="utf-8"))


def status(**values: object) -> None:
    payload = {}
    if STATUS.exists():
        try:
            payload = json.loads(STATUS.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    payload.update(values)
    payload["updated_at_epoch"] = time.time()
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(STATUS)


def command(max_steps: int, phase_label: str, resume: bool, final_phase: bool) -> list[str]:
    values = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "train_text2sql_qlora.py"),
        "--launch-config",
        str(CONFIG),
        "--output-dir",
        str(OUTPUT),
        "--cache-dir",
        "/content/huggingface-cache",
        "--status-path",
        str(STATUS),
        "--max-steps",
        str(max_steps),
        "--phase-label",
        phase_label,
        "--resume",
        "auto" if resume else "none",
    ]
    if resume:
        values.append("--require-checkpoint")
    if final_phase:
        values.append("--final-phase")
    return values


environment = os.environ.copy()
environment.update(
    {
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HOME": "/content/huggingface-cache",
        "TRANSFORMERS_CACHE": "/content/huggingface-cache",
    }
)
status(
    phase="launching",
    model=launch["model"]["slug"],
    dataset_variant=launch["dataset_variant"],
    smoke=launch["smoke"],
    started_at_utc=datetime.now(timezone.utc).isoformat(),
)
try:
    final_steps = int(launch["training"]["max_steps"])
    if launch.get("resume_smoke_test"):
        phase_one_steps = int(launch["resume_phase_one_steps"])
        subprocess.run(
            command(phase_one_steps, "smoke-checkpoint", resume=False, final_phase=False),
            check=True,
            env=environment,
        )
        status(phase="resume_test_loading", phase_label="smoke-resume", step=phase_one_steps)
        subprocess.run(
            command(final_steps, "smoke-resume", resume=True, final_phase=True),
            check=True,
            env=environment,
        )
    else:
        resume_full_run = any(OUTPUT.glob("checkpoint-*/trainer_state.json"))
        subprocess.run(
            command(final_steps, "main", resume=resume_full_run, final_phase=True),
            check=True,
            env=environment,
        )
except Exception as exc:
    status(phase="failed", error=f"{type(exc).__name__}: {exc}")
    raise
