"""Thin runtime config for the UI ask() pipeline.

Env vars override the JSON file when both are set.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "app" / "ui_config.json"
DEFAULT_DEMO_DIR = PROJECT_ROOT / "demo_databases"
DEFAULT_MODELS_CONFIG = PROJECT_ROOT / "configs" / "text2sql_eval_models.json"

# Backend names used by ask() / models.py
BACKEND_MOCK = "mock"
BACKEND_QWEN25 = "qwen2.5-1.5b"
BACKEND_QWEN3 = "qwen3-4b+adapter"

SLUG_QWEN25 = "qwen2.5-coder-1.5b-instruct"
SLUG_QWEN3 = "qwen3-4b-instruct-2507"


@dataclass(frozen=True)
class UIConfig:
    backend: str = BACKEND_MOCK
    model_slug: str = SLUG_QWEN25
    adapter_dir: Path | None = None
    demo_databases_dir: Path = DEFAULT_DEMO_DIR
    models_config_path: Path = DEFAULT_MODELS_CONFIG
    max_new_tokens: int = 512
    execute_timeout_seconds: float = 5.0
    max_result_rows: int = 200
    mschema_examples: int = 3
    load_4bit: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model_slug": self.model_slug,
            "adapter_dir": str(self.adapter_dir) if self.adapter_dir else None,
            "demo_databases_dir": str(self.demo_databases_dir),
            "models_config_path": str(self.models_config_path),
            "max_new_tokens": self.max_new_tokens,
            "execute_timeout_seconds": self.execute_timeout_seconds,
            "max_result_rows": self.max_result_rows,
            "mschema_examples": self.mschema_examples,
            "load_4bit": self.load_4bit,
        }


def _path_or_none(value: str | None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return Path(str(value)).expanduser()


def load_ui_config(config_path: Path | None = None) -> UIConfig:
    path = config_path or Path(os.environ.get("UI_CONFIG_PATH", DEFAULT_CONFIG_PATH))
    file_data: dict[str, Any] = {}
    if path.exists():
        file_data = json.loads(path.read_text(encoding="utf-8"))

    backend = os.environ.get("MODEL_BACKEND", file_data.get("backend", BACKEND_MOCK))
    model_slug = os.environ.get("MODEL_SLUG", file_data.get("model_slug", SLUG_QWEN25))
    adapter_raw = os.environ.get("ADAPTER_DIR", file_data.get("adapter_dir"))
    demo_dir = Path(
        os.environ.get(
            "DEMO_DATABASES_DIR",
            file_data.get("demo_databases_dir", str(DEFAULT_DEMO_DIR)),
        )
    )
    if not demo_dir.is_absolute():
        demo_dir = PROJECT_ROOT / demo_dir

    models_config = Path(
        file_data.get("models_config_path", str(DEFAULT_MODELS_CONFIG))
    )
    if not models_config.is_absolute():
        models_config = PROJECT_ROOT / models_config

    adapter_dir = _path_or_none(adapter_raw if adapter_raw is None else str(adapter_raw))
    if adapter_dir is not None and not adapter_dir.is_absolute():
        adapter_dir = PROJECT_ROOT / adapter_dir

    return UIConfig(
        backend=str(backend).strip(),
        model_slug=str(model_slug).strip(),
        adapter_dir=adapter_dir,
        demo_databases_dir=demo_dir,
        models_config_path=models_config,
        max_new_tokens=int(file_data.get("max_new_tokens", 512)),
        execute_timeout_seconds=float(file_data.get("execute_timeout_seconds", 5.0)),
        max_result_rows=int(file_data.get("max_result_rows", 200)),
        mschema_examples=int(file_data.get("mschema_examples", 3)),
        load_4bit=bool(file_data.get("load_4bit", True)),
    )


def load_model_spec(slug: str, models_config_path: Path) -> dict[str, Any]:
    payload = json.loads(models_config_path.read_text(encoding="utf-8"))
    for item in payload.get("models", []):
        if str(item.get("slug")) == slug:
            return dict(item)
    raise KeyError(f"Model slug {slug!r} not found in {models_config_path}")
