#!/usr/bin/env python3
"""Materialize one auditable Qwen3 QLoRA hyperparameter configuration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = PROJECT_ROOT / "configs" / "text2sql_qlora_training.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--selection-dataset", default="qwen3_hparam_mschema_v1")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--scheduler", choices=("cosine", "linear"), default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument(
        "--compact-screening",
        action="store_true",
        help="Skip optimizer checkpoints and loss-only evaluation; generation EX is run separately.",
    )
    parser.add_argument(
        "--target-profile",
        choices=("all-linear", "attention-only"),
        default="all-linear",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.learning_rate <= 0 or args.rank <= 0 or args.alpha <= 0 or args.epochs <= 0:
        raise ValueError("learning rate, rank, alpha, and epochs must be positive")
    if min(args.train_batch_size, args.eval_batch_size, args.gradient_accumulation_steps) <= 0:
        raise ValueError("batch sizes and gradient accumulation must be positive")
    if not 0 <= args.dropout < 1 or not 0 <= args.warmup_ratio < 1 or args.weight_decay < 0:
        raise ValueError("dropout/warmup must be in [0,1); weight decay must be non-negative")
    config = json.loads(args.base.resolve().read_text(encoding="utf-8"))
    try:
        base_config = str(args.base.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        base_config = str(args.base.resolve())
    config["search_metadata"] = {
        "label": args.label,
        "base_config": base_config,
        "selection_dataset": args.selection_dataset,
    }
    config["lora"].update(
        {
            "r": args.rank,
            "lora_alpha": args.alpha,
            "lora_dropout": args.dropout,
            "target_modules": (
                "all-linear"
                if args.target_profile == "all-linear"
                else ["q_proj", "k_proj", "v_proj", "o_proj"]
            ),
        }
    )
    config["optimization"].update(
        {
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.epochs,
            "lr_scheduler_type": args.scheduler,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        }
    )
    config["models"]["qwen3-4b-instruct-2507"].update(
        {
            "per_device_train_batch_size": args.train_batch_size,
            "per_device_eval_batch_size": args.eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
        }
    )
    if args.compact_screening:
        config["optimization"]["skip_final_loss_eval"] = True
        config["optimization"]["skip_resume_checkpoints"] = True
        config["full"]["eval_strategy"] = "no"
        config["full"]["save_strategy"] = "no"
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
