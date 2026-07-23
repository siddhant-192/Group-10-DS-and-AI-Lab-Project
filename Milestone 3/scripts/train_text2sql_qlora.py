#!/usr/bin/env python3
"""Train one pinned text-to-SQL model with explicit-mask QLoRA on one GPU."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import tarfile
import sys
import time
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("/content/huggingface-cache"))
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", default="none", help="none, auto, or a checkpoint path")
    parser.add_argument("--require-checkpoint", action="store_true")
    parser.add_argument("--phase-label", default="main")
    parser.add_argument("--final-phase", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id_hash(rows: Sequence[dict[str, Any]]) -> str:
    joined = "\n".join(str(row["id"]) for row in rows) + "\n"
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("text2sql-qlora")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "training.log", encoding="utf-8", mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def cuda_memory() -> dict[str, float]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        return {
            "allocated_gib": round(torch.cuda.memory_allocated() / 1024**3, 3),
            "reserved_gib": round(torch.cuda.memory_reserved() / 1024**3, 3),
            "max_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "max_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        }
    except ImportError:
        return {}


def update_status(path: Path, **values: Any) -> None:
    payload: dict[str, Any] = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
    payload.update(values)
    payload["updated_at_epoch"] = time.time()
    atomic_json(path, payload)


def conversation_character_length(row: dict[str, Any]) -> int:
    return sum(len(str(message.get("content", ""))) for message in row.get("messages", []))


def select_length_stratified(rows: Sequence[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    """Choose a deterministic smoke subset spanning the complete length range."""
    if limit is None or limit >= len(rows):
        return list(rows)
    if limit <= 0:
        raise ValueError("Dataset limit must be positive")
    ordered = sorted(rows, key=lambda row: (conversation_character_length(row), str(row["id"])))
    if limit == 1:
        return [ordered[-1]]
    positions = [round(index * (len(ordered) - 1) / (limit - 1)) for index in range(limit)]
    if len(set(positions)) != limit:
        raise AssertionError("Length-stratified selection produced duplicate positions")
    return [ordered[position] for position in positions]


def encode_conversation(tokenizer: Any, row: dict[str, Any], max_seq_length: int) -> dict[str, Any]:
    """Create causal labels with every non-assistant prompt token masked to -100."""
    messages = row.get("messages")
    roles = [message.get("role") for message in messages] if isinstance(messages, list) else []
    if roles not in (["system", "user", "assistant"], ["user", "assistant"]):
        raise ValueError(
            f"{row.get('id')}: expected system/user/assistant or user/assistant role order"
        )
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            f"{row.get('id')}: the full chat tokenization does not begin with the generation prompt; "
            "assistant-only masking would be unsafe"
        )
    if len(full_ids) > max_seq_length:
        raise ValueError(
            f"{row.get('id')}: sequence has {len(full_ids)} tokens, exceeding max_seq_length={max_seq_length}; "
            "training will not silently truncate schemas or SQL targets"
        )
    labels = [-100] * len(prompt_ids) + list(full_ids[len(prompt_ids) :])
    target_tokens = sum(label != -100 for label in labels)
    if target_tokens <= 0:
        raise ValueError(f"{row.get('id')}: assistant target has no trainable tokens")
    return {
        "input_ids": list(full_ids),
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "length": len(full_ids),
        "source_id": str(row["id"]),
        "target_tokens": target_tokens,
    }


def package_versions(names: Sequence[str]) -> dict[str, str]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def export_resume_checkpoint(
    output_dir: Path, checkpoint: Path, status_path: Path, phase_label: str
) -> dict[str, Any]:
    """Atomically expose one complete checkpoint for concurrent local download."""
    required = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "rng_state.pth",
    )
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise RuntimeError(f"Cannot export incomplete {checkpoint.name}: missing {missing}")
    try:
        step = int(checkpoint.name.rsplit("-", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid checkpoint directory name: {checkpoint.name}") from exc
    export_dir = output_dir / "checkpoint_exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    final_archive = export_dir / f"{checkpoint.name}.tar"
    temporary_archive = export_dir / f".{checkpoint.name}.tar.tmp"
    with tarfile.open(temporary_archive, "w") as archive:
        archive.add(checkpoint, arcname=checkpoint.name, recursive=True)
    os.replace(temporary_archive, final_archive)
    metadata = {
        "step": step,
        "archive": str(final_archive),
        "bytes": final_archive.stat().st_size,
        "sha256": sha256_file(final_archive),
    }
    atomic_json(export_dir / f"{checkpoint.name}.json", metadata)
    update_status(
        status_path,
        phase="checkpoint_exported",
        phase_label=phase_label,
        step=step,
        checkpoint_export=metadata,
        cuda=cuda_memory(),
    )
    return metadata


def resolve_resume_checkpoint(output_dir: Path, resume: str) -> Path | None:
    if resume == "none":
        return None
    if resume != "auto":
        candidate = Path(resume).resolve()
        if not candidate.is_dir():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {candidate}")
        return candidate
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        checkpoints.append((step, path))
    return max(checkpoints, default=(0, None), key=lambda item: item[0])[1]


def main() -> int:
    args = parse_args()
    launch_path = args.launch_config.resolve()
    output_dir = args.output_dir.resolve()
    status_path = args.status_path.resolve()
    cache_dir = args.cache_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output_dir)
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    model_spec = launch["model"]
    training = launch["training"]
    model_slug = str(model_spec["slug"])
    phase_history = output_dir / "phase_history.jsonl"
    metrics_history = output_dir / "trainer_metrics.jsonl"
    started_at = datetime.now(timezone.utc).isoformat()

    update_status(
        status_path,
        phase="initializing",
        phase_label=args.phase_label,
        model=model_slug,
        dataset_variant=launch["dataset_variant"],
        step=0,
        max_steps=args.max_steps if args.max_steps is not None else training["max_steps"],
        cuda=cuda_memory(),
        error=None,
    )
    try:
        import torch
        import transformers
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainerCallback,
            TrainingArguments,
            set_seed,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for QLoRA training")
        gpu_name = torch.cuda.get_device_name(0)
        if "L4" not in gpu_name.upper():
            raise RuntimeError(f"Expected an NVIDIA L4, found: {gpu_name}")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected GPU does not support bfloat16")
        seed = int(training["seed"])
        set_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = bool(training["tf32"])
        torch.backends.cudnn.allow_tf32 = bool(training["tf32"])
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        train_path = Path(launch["train_data"]).resolve()
        validation_path = Path(launch["validation_data"]).resolve()
        train_rows = select_length_stratified(read_jsonl(train_path), training.get("train_limit"))
        validation_rows = select_length_stratified(
            read_jsonl(validation_path), training.get("validation_limit")
        )
        if not train_rows or not validation_rows:
            raise ValueError("Training and validation selections must both be non-empty")

        update_status(
            status_path,
            phase="loading_tokenizer",
            train_examples=len(train_rows),
            validation_examples=len(validation_rows),
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_spec["repo_id"],
            revision=model_spec["revision"],
            trust_remote_code=bool(model_spec.get("trust_remote_code", False)),
            cache_dir=str(cache_dir),
        )
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        max_seq_length = int(training["max_seq_length"])
        encoded_train = [encode_conversation(tokenizer, row, max_seq_length) for row in train_rows]
        encoded_validation = [
            encode_conversation(tokenizer, row, max_seq_length) for row in validation_rows
        ]
        train_lengths = [row["length"] for row in encoded_train]
        validation_lengths = [row["length"] for row in encoded_validation]
        tokenization_summary = {
            "train_examples": len(encoded_train),
            "validation_examples": len(encoded_validation),
            "train_id_sha256": stable_id_hash(train_rows),
            "validation_id_sha256": stable_id_hash(validation_rows),
            "train_min_tokens": min(train_lengths),
            "train_max_tokens": max(train_lengths),
            "validation_min_tokens": min(validation_lengths),
            "validation_max_tokens": max(validation_lengths),
            "train_target_tokens": sum(row["target_tokens"] for row in encoded_train),
            "validation_target_tokens": sum(row["target_tokens"] for row in encoded_validation),
            "max_seq_length": max_seq_length,
            "truncated_examples": 0,
        }
        atomic_json(output_dir / "tokenization_summary.json", tokenization_summary)
        train_dataset = Dataset.from_list(encoded_train)
        validation_dataset = Dataset.from_list(encoded_validation)

        update_status(status_path, phase="loading_quantized_model", cuda=cuda_memory())
        quantization = launch["quantization"]
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=bool(quantization["load_in_4bit"]),
            bnb_4bit_quant_type=str(quantization["bnb_4bit_quant_type"]),
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=bool(quantization["bnb_4bit_use_double_quant"]),
        )
        model_kwargs = {
            "revision": model_spec["revision"],
            "trust_remote_code": bool(model_spec.get("trust_remote_code", False)),
            "cache_dir": str(cache_dir),
            "quantization_config": quantization_config,
            "torch_dtype": torch.bfloat16,
            "device_map": {"": 0},
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
        }
        attention = "sdpa"
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_spec["repo_id"], attn_implementation="sdpa", **model_kwargs
            )
        except (TypeError, ValueError, ImportError) as exc:
            if not any(term in str(exc).lower() for term in ("sdpa", "attn", "attention")):
                raise
            logger.warning("SDPA unavailable (%s); using the model's native attention", exc)
            model = AutoModelForCausalLM.from_pretrained(model_spec["repo_id"], **model_kwargs)
            attention = "native"
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(training["gradient_checkpointing"]),
            gradient_checkpointing_kwargs={
                "use_reentrant": bool(training["gradient_checkpointing_use_reentrant"])
            },
        )
        lora = launch["lora"]
        lora_config = LoraConfig(
            r=int(lora["r"]),
            lora_alpha=int(lora["lora_alpha"]),
            lora_dropout=float(lora["lora_dropout"]),
            bias=str(lora["bias"]),
            target_modules=lora["target_modules"],
            task_type=str(lora["task_type"]),
        )
        # PEFT otherwise records a null revision in adapter_config.json. The
        # adapter must always reload the exact backbone used for training.
        lora_config.revision = str(model_spec["revision"])
        model = get_peft_model(model, lora_config)
        trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        quantized_storage_numel = sum(parameter.numel() for parameter in model.parameters())
        logical_base_parameters = int(model_spec["base_parameter_count"])
        logger.info(
            "%s loaded with %s attention: trainable=%d total=%d (%.4f%%), CUDA=%s",
            model_slug,
            attention,
            trainable_parameters,
            logical_base_parameters,
            100.0 * trainable_parameters / logical_base_parameters,
            cuda_memory(),
        )

        effective_max_steps = int(
            args.max_steps if args.max_steps is not None else training.get("max_steps", -1)
        )
        training_arguments = TrainingArguments(
            output_dir=str(output_dir),
            overwrite_output_dir=False,
            do_train=True,
            do_eval=True,
            eval_strategy=str(training["eval_strategy"]),
            save_strategy=str(training["save_strategy"]),
            per_device_train_batch_size=int(training["per_device_train_batch_size"]),
            per_device_eval_batch_size=int(training["per_device_eval_batch_size"]),
            gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            max_grad_norm=float(training["max_grad_norm"]),
            num_train_epochs=float(training["num_train_epochs"]),
            max_steps=effective_max_steps,
            lr_scheduler_type=str(training["lr_scheduler_type"]),
            warmup_ratio=float(training["warmup_ratio"]),
            optim=str(training["optim"]),
            logging_steps=int(training["logging_steps"]),
            logging_first_step=True,
            eval_steps=int(training["eval_steps"]) if training.get("eval_steps") else None,
            save_steps=int(training["save_steps"]),
            save_total_limit=int(training["save_total_limit"]),
            bf16=bool(training["bf16"]),
            tf32=bool(training["tf32"]),
            gradient_checkpointing=bool(training["gradient_checkpointing"]),
            gradient_checkpointing_kwargs={
                "use_reentrant": bool(training["gradient_checkpointing_use_reentrant"])
            },
            group_by_length=True,
            length_column_name="length",
            report_to=["tensorboard"],
            run_name=str(launch["run_name"]),
            seed=seed,
            data_seed=seed,
            dataloader_num_workers=2,
            dataloader_pin_memory=True,
            remove_unused_columns=True,
            save_safetensors=True,
            disable_tqdm=False,
        )
        collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=None,
            padding=True,
            label_pad_token_id=-100,
            pad_to_multiple_of=int(training["pad_to_multiple_of"]),
            return_tensors="pt",
        )

        class StatusCallback(TrainerCallback):
            def on_step_end(self, args_: Any, state: Any, control: Any, **kwargs: Any) -> None:
                del args_, control, kwargs
                # Logging can be intentionally sparse on full runs. Keep a
                # lightweight per-step heartbeat so remote monitoring never
                # appears stalled between logging intervals.
                update_status(
                    status_path,
                    phase="training",
                    phase_label=args.phase_label,
                    step=state.global_step,
                    max_steps=state.max_steps,
                    epoch=state.epoch,
                    cuda=cuda_memory(),
                )

            def on_log(self, args_: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
                del args_, control, kwargs
                values = dict(logs or {})
                append_jsonl(
                    metrics_history,
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "phase_label": args.phase_label,
                        "step": state.global_step,
                        "epoch": state.epoch,
                        **values,
                        "cuda": cuda_memory(),
                    },
                )
                update_status(
                    status_path,
                    phase="training",
                    phase_label=args.phase_label,
                    step=state.global_step,
                    max_steps=state.max_steps,
                    epoch=state.epoch,
                    loss=values.get("loss", values.get("eval_loss")),
                    learning_rate=values.get("learning_rate"),
                    cuda=cuda_memory(),
                )

            def on_save(self, args_: Any, state: Any, control: Any, **kwargs: Any) -> None:
                del args_, control, kwargs
                export_resume_checkpoint(
                    output_dir,
                    output_dir / f"checkpoint-{state.global_step}",
                    status_path,
                    args.phase_label,
                )

        trainer = Trainer(
            model=model,
            args=training_arguments,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            data_collator=collator,
            callbacks=[StatusCallback()],
        )
        checkpoint = resolve_resume_checkpoint(output_dir, args.resume)
        if args.require_checkpoint and checkpoint is None:
            raise RuntimeError("This phase requires a resume checkpoint, but none was found")
        logger.info(
            "Starting phase=%s max_steps=%d resume=%s train=%d validation=%d",
            args.phase_label,
            effective_max_steps,
            checkpoint or "none",
            len(train_dataset),
            len(validation_dataset),
        )
        update_status(
            status_path,
            phase="training",
            phase_label=args.phase_label,
            resumed_from_checkpoint=str(checkpoint) if checkpoint else None,
            cuda=cuda_memory(),
        )
        phase_started = time.monotonic()
        train_result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
        train_metrics = dict(train_result.metrics)
        # Step-based saving does not guarantee that the last optimizer step is a
        # save boundary. With the pinned Transformers API, explicitly materialize
        # a complete final checkpoint so downloaded state always matches the
        # final adapter and is genuinely resumable.
        final_checkpoint = output_dir / f"checkpoint-{trainer.state.global_step}"
        required_final_state = ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json")
        if not all((final_checkpoint / name).exists() for name in required_final_state):
            trainer._save_checkpoint(trainer.model, trial=None)
        export_resume_checkpoint(output_dir, final_checkpoint, status_path, args.phase_label)
        trainer.save_metrics("train", train_metrics)
        trainer.save_state()
        update_status(status_path, phase="evaluating_loss", step=trainer.state.global_step, cuda=cuda_memory())
        eval_metrics = dict(trainer.evaluate())
        trainer.save_metrics("eval", eval_metrics)
        final_adapter = output_dir / "final_adapter"
        trainer.save_model(str(final_adapter))
        tokenizer.save_pretrained(final_adapter)
        phase_record = {
            "phase_label": args.phase_label,
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.monotonic() - phase_started, 3),
            "resumed_from_checkpoint": str(checkpoint) if checkpoint else None,
            "global_step": trainer.state.global_step,
            "max_steps": trainer.state.max_steps,
            "train_metrics": train_metrics,
            "eval_metrics": eval_metrics,
            "cuda_peak": cuda_memory(),
            "adapter_bytes": directory_bytes(final_adapter),
        }
        append_jsonl(phase_history, phase_record)
        all_phases = read_jsonl(phase_history)
        overall_cuda_peak = {
            key: max(
                float(phase.get("cuda_peak", {}).get(key, 0.0)) for phase in all_phases
            )
            for key in ("max_allocated_gib", "max_reserved_gib")
        }
        run_manifest = {
            "run_name": launch["run_name"],
            "model": model_spec,
            "dataset_variant": launch["dataset_variant"],
            "train_data": {
                "path": str(train_path),
                "sha256": sha256_file(train_path),
                "selected_examples": len(train_rows),
                "selected_id_sha256": stable_id_hash(train_rows),
            },
            "validation_data": {
                "path": str(validation_path),
                "sha256": sha256_file(validation_path),
                "selected_examples": len(validation_rows),
                "selected_id_sha256": stable_id_hash(validation_rows),
            },
            "method": "QLoRA",
            "quantization": quantization,
            "lora": lora,
            "training": training,
            "attention": attention,
            "explicit_assistant_only_labels": True,
            "truncated_examples": 0,
            "trainable_parameters": trainable_parameters,
            "logical_base_parameters": logical_base_parameters,
            "quantized_storage_parameter_numel": quantized_storage_numel,
            "trainable_parameter_pct_of_logical_base": round(
                100.0 * trainable_parameters / logical_base_parameters, 6
            ),
            "gpu": gpu_name,
            "gpu_total_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 3),
            "packages": package_versions(
                ("torch", "transformers", "datasets", "accelerate", "peft", "bitsandbytes", "tensorboard")
            ),
            "phase_count": len(all_phases),
            "resume_verified": any(phase.get("resumed_from_checkpoint") for phase in all_phases),
            "overall_cuda_peak": overall_cuda_peak,
            "last_phase": phase_record,
        }
        atomic_json(output_dir / "run_manifest.json", run_manifest)
        final_phase = "complete" if args.final_phase else "phase_complete"
        update_status(
            status_path,
            phase=final_phase,
            phase_label=args.phase_label,
            step=trainer.state.global_step,
            max_steps=trainer.state.max_steps,
            eval_loss=eval_metrics.get("eval_loss"),
            adapter_bytes=phase_record["adapter_bytes"],
            cuda=cuda_memory(),
        )
        logger.info("Phase %s complete: %s", args.phase_label, phase_record)
        return 0
    except Exception as exc:
        logger.exception("QLoRA phase failed")
        update_status(
            status_path,
            phase="failed",
            phase_label=args.phase_label,
            error=f"{type(exc).__name__}: {exc}",
            cuda=cuda_memory(),
        )
        raise
    finally:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
