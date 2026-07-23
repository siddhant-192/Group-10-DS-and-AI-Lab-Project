#!/usr/bin/env python3
"""Load the selected base plus QLoRA adapter and run one M-Schema prompt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verify_final_adapter import DEFAULT_MANIFEST, verify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--schema-file", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Load without bitsandbytes 4-bit quantization (requires substantially more memory).",
    )
    return parser.parse_args()


def mschema_prompt(schema: str, question: str) -> str:
    return (
        "You are now a sqlite data analyst, and you are given a database schema as follows:\n\n"
        f"【Schema】\n{schema}\n\n"
        f"【Question】\n{question}\n\n"
        "【Evidence】\n\n"
        "Please read and understand the database schema carefully, and generate an executable SQL based "
        "on the user's question and evidence. The generated SQL is protected by ```sql and ```."
    )


def main() -> int:
    args = parse_args()
    verified = verify(args.adapter_dir, args.manifest)
    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    base = manifest["base_model"]

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "Model dependencies are missing. On Colab install "
            "`scripts/colab-sft-requirements.txt` first."
        ) from exc

    if not args.no_4bit and not torch.cuda.is_available():
        raise RuntimeError("The default 4-bit smoke test requires an NVIDIA CUDA GPU.")

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir.resolve())
    model_kwargs = {
        "revision": base["revision"],
        "trust_remote_code": bool(base["trust_remote_code"]),
        "device_map": "auto",
        "attn_implementation": "sdpa",
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }
    if not args.no_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    base_model = AutoModelForCausalLM.from_pretrained(base["repo_id"], **model_kwargs)
    model = PeftModel.from_pretrained(base_model, args.adapter_dir.resolve())
    model.eval()

    schema = args.schema_file.resolve().read_text(encoding="utf-8")
    messages = [{"role": "user", "content": mschema_prompt(schema, args.question)}]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(rendered, return_tensors="pt")
    input_device = next(model.parameters()).device
    encoded = {key: value.to(input_device) for key, value in encoded.items()}

    with torch.inference_mode():
        output = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output[0, encoded["input_ids"].shape[1] :]
    print(
        json.dumps(
            {
                "verification": verified,
                "question": args.question,
                "generated_text": tokenizer.decode(generated, skip_special_tokens=True).strip(),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

