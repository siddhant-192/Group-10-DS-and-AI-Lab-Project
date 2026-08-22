"""Swappable text-to-SQL model backends: mock | qwen2.5 | qwen3+adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import re
import time
from typing import Any

from .config import (
    BACKEND_MOCK,
    BACKEND_QWEN25,
    BACKEND_QWEN3,
    UIConfig,
    load_model_spec,
)
from .prompt import mschema_prompt


class ModelBackend(ABC):
    name: str

    @abstractmethod
    def generate(self, schema: str, question: str, dialect: str = "sqlite") -> tuple[str, dict[str, Any]]:
        """Return (raw_model_text, metadata)."""


class MockBackend(ModelBackend):
    """CPU-only stub: invents simple SQL from the question + first table name."""

    name = BACKEND_MOCK

    def generate(self, schema: str, question: str, dialect: str = "sqlite") -> tuple[str, dict[str, Any]]:
        started = time.monotonic()
        tables = _table_names(schema)
        table = _pick_table(question, tables) or (tables[0] if tables else "sqlite_master")
        q = question.lower()
        if "how many" in q or "count" in q or "number of" in q:
            sql = f"SELECT COUNT(*) AS count FROM {table};"
        elif "list" in q or "show" in q or "all" in q or "what" in q:
            sql = f"SELECT * FROM {table} LIMIT 10;"
        else:
            sql = f"SELECT * FROM {table} LIMIT 5;"
        raw = f"```sql\n{sql}\n```"
        meta = {
            "backend": self.name,
            "model_slug": "mock",
            "generation_ms": round((time.monotonic() - started) * 1000, 3),
            "note": "Deterministic mock — no LLM loaded.",
        }
        return raw, meta


class HuggingFaceBackend(ModelBackend):
    """Load a HF causal LM (optional PEFT adapter) and greedy-generate SQL."""

    def __init__(
        self,
        *,
        name: str,
        repo_id: str,
        revision: str | None,
        trust_remote_code: bool,
        adapter_dir: Path | None,
        max_new_tokens: int,
        load_4bit: bool,
    ) -> None:
        self.name = name
        self.repo_id = repo_id
        self.revision = revision
        self.trust_remote_code = trust_remote_code
        self.adapter_dir = adapter_dir
        self.max_new_tokens = max_new_tokens
        self.load_4bit = load_4bit
        self._model = None
        self._tokenizer = None
        self._device = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError(
                "transformers/torch missing. On Colab install "
                "src/scripts/colab-sft-requirements.txt (or colab-eval-requirements.txt)."
            ) from exc

        if self.load_4bit and not torch.cuda.is_available():
            raise RuntimeError(
                "4-bit load requires CUDA. Set load_4bit=false for CPU "
                "(slow / may OOM), or use backend=mock locally."
            )
        if self.load_4bit:
            try:
                import bitsandbytes  # noqa: F401
            except Exception as exc:
                raise RuntimeError(
                    "4-bit load needs bitsandbytes. On Colab run: "
                    "pip install bitsandbytes==0.49.2 "
                    "(or: pip install -r app/scripts/colab-ui-requirements.txt)."
                ) from exc

        tokenizer_source = str(self.adapter_dir) if self.adapter_dir else self.repo_id
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=self.trust_remote_code,
            revision=None if self.adapter_dir else self.revision,
        )
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "device_map": "auto" if torch.cuda.is_available() else None,
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        }
        if self.revision and not self.adapter_dir:
            model_kwargs["revision"] = self.revision
        if self.load_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        base = AutoModelForCausalLM.from_pretrained(self.repo_id, **model_kwargs)
        if self.adapter_dir is not None:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(
                base,
                str(self.adapter_dir.resolve()),
                device_map="auto" if torch.cuda.is_available() else None,
            )
        else:
            self._model = base
        self._model.eval()
        self._device = next(self._model.parameters()).device

    def generate(self, schema: str, question: str, dialect: str = "sqlite") -> tuple[str, dict[str, Any]]:
        import torch

        self._ensure_loaded()
        assert self._tokenizer is not None and self._model is not None
        started = time.monotonic()
        messages = [{"role": "user", "content": mschema_prompt(schema, question, dialect=dialect)}]
        rendered = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = self._tokenizer(rendered, return_tensors="pt")
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = self._model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = output[0, encoded["input_ids"].shape[1] :]
        text = self._tokenizer.decode(generated, skip_special_tokens=True)
        meta = {
            "backend": self.name,
            "repo_id": self.repo_id,
            "adapter_dir": str(self.adapter_dir) if self.adapter_dir else None,
            "generation_ms": round((time.monotonic() - started) * 1000, 3),
            "max_new_tokens": self.max_new_tokens,
            "load_4bit": self.load_4bit,
        }
        return text, meta


_TABLE_RE = re.compile(r"# Table:\s*(\S+)")
_DDL_TABLE_RE = re.compile(r'CREATE\s+TABLE\s+"?([A-Za-z_][\w]*)"?', re.IGNORECASE)


def _table_names(schema: str) -> list[str]:
    names = _TABLE_RE.findall(schema)
    if names:
        return names
    return _DDL_TABLE_RE.findall(schema)


def _pick_table(question: str, tables: list[str]) -> str | None:
    q = question.lower()
    for table in tables:
        token = table.lower().rstrip("s")
        if table.lower() in q or token in q:
            return table
    return None


def _first_table_name(schema: str) -> str | None:
    names = _table_names(schema)
    return names[0] if names else None


def build_backend(config: UIConfig) -> ModelBackend:
    backend = config.backend.strip().lower()
    if backend in {BACKEND_MOCK, "mock"}:
        return MockBackend()

    if backend in {BACKEND_QWEN25, "qwen2.5", "qwen25"}:
        slug = config.model_slug or "qwen2.5-coder-1.5b-instruct"
        spec = load_model_spec(slug, config.models_config_path)
        return HuggingFaceBackend(
            name=BACKEND_QWEN25,
            repo_id=str(spec["repo_id"]),
            revision=str(spec.get("revision")) if spec.get("revision") else None,
            trust_remote_code=bool(spec.get("trust_remote_code", False)),
            adapter_dir=None,
            max_new_tokens=config.max_new_tokens,
            load_4bit=config.load_4bit,
        )

    if backend in {BACKEND_QWEN3, "qwen3", "qwen3-4b"}:
        slug = config.model_slug or "qwen3-4b-instruct-2507"
        spec = load_model_spec(slug, config.models_config_path)
        if config.adapter_dir is None:
            raise ValueError(
                "backend qwen3-4b+adapter requires adapter_dir "
                "(set ADAPTER_DIR or ui_config.json adapter_dir)."
            )
        if not config.adapter_dir.exists():
            raise FileNotFoundError(f"Adapter not found: {config.adapter_dir}")
        return HuggingFaceBackend(
            name=BACKEND_QWEN3,
            repo_id=str(spec["repo_id"]),
            revision=str(spec.get("revision")) if spec.get("revision") else None,
            trust_remote_code=bool(spec.get("trust_remote_code", False)),
            adapter_dir=config.adapter_dir,
            max_new_tokens=config.max_new_tokens,
            load_4bit=config.load_4bit,
        )

    raise ValueError(
        f"Unknown MODEL_BACKEND {config.backend!r}. "
        f"Use: {BACKEND_MOCK!r}, {BACKEND_QWEN25!r}, or {BACKEND_QWEN3!r}."
    )
