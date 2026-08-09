#!/usr/bin/env python3
"""Colab / GPU smoke: Qwen2.5-Coder-1.5B → extract SQL → safety → execute on a demo DB.

Run from the project root (or upload this folder to Colab Drive and chdir there).

Example (Colab T4):
  !pip install -q -r src/scripts/colab-eval-requirements.txt
  !python app/scripts/download_demo_databases.py --skip-chinook
  !python app/scripts/colab_qwen25_smoke.py --db-id mini_music \\
      --question "How many singers are there?"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--db-id", default="mini_music")
    parser.add_argument("--question", default="How many singers are there?")
    parser.add_argument(
        "--backend",
        default="qwen2.5-1.5b",
        choices=["mock", "qwen2.5-1.5b", "qwen3-4b+adapter"],
    )
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--no-4bit", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "app"))

    from backend.config import UIConfig, load_ui_config
    from backend.ask import ask
    from backend.registry import list_databases

    base = load_ui_config(root / "app" / "ui_config.json")
    cfg = UIConfig(
        backend=args.backend,
        model_slug=base.model_slug
        if args.backend != "qwen3-4b+adapter"
        else "qwen3-4b-instruct-2507",
        adapter_dir=args.adapter_dir or base.adapter_dir,
        demo_databases_dir=root / "demo_databases",
        models_config_path=root / "configs" / "text2sql_eval_models.json",
        max_new_tokens=args.max_new_tokens,
        execute_timeout_seconds=base.execute_timeout_seconds,
        max_result_rows=base.max_result_rows,
        mschema_examples=base.mschema_examples,
        load_4bit=not args.no_4bit,
    )

    available = list_databases(cfg.demo_databases_dir)
    if args.db_id not in available:
        print(
            json.dumps(
                {
                    "error": f"db_id {args.db_id!r} missing",
                    "available": sorted(available),
                    "hint": "Run: python app/scripts/download_demo_databases.py",
                },
                indent=2,
            )
        )
        return 1

    # Fresh backend per smoke run (avoid stale mock when switching).
    from backend.models import build_backend

    backend = build_backend(cfg)
    result = ask(args.question, args.db_id, config=cfg, backend=backend)
    print(json.dumps(result, indent=2, default=str))
    return 0 if not result.get("error") else 2


if __name__ == "__main__":
    raise SystemExit(main())
