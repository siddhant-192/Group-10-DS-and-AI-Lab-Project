from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


training = load_script("train_text2sql_qlora", "train_text2sql_qlora.py")
bundler = load_script("build_colab_sft_bundle", "build_colab_sft_bundle.py")
checkpoint_inspector = load_script(
    "inspect_sft_checkpoint_archive", "inspect_sft_checkpoint_archive.py"
)


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        if messages[0]["role"] == "system":
            prompt = "SYSTEM=" + messages[0]["content"] + "|USER=" + messages[1]["content"] + "|ASSISTANT="
            assistant_index = 2
        else:
            prompt = "USER=" + messages[0]["content"] + "|ASSISTANT="
            assistant_index = 1
        if add_generation_prompt:
            return prompt
        return prompt + messages[assistant_index]["content"] + "<eos>"

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [ord(character) for character in text]}


def row(index: int, size: int = 1):
    return {
        "id": f"row-{index}",
        "messages": [
            {"role": "system", "content": "SQL only"},
            {"role": "user", "content": "x" * size},
            {"role": "assistant", "content": "SELECT 1"},
        ],
    }


class QloraTrainingTests(unittest.TestCase):
    def test_explicit_labels_mask_every_prompt_token(self) -> None:
        tokenizer = FakeTokenizer()
        encoded = training.encode_conversation(tokenizer, row(1), max_seq_length=1000)
        prompt = tokenizer.apply_chat_template(row(1)["messages"][:-1], tokenize=False, add_generation_prompt=True)
        prompt_length = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        self.assertTrue(all(value == -100 for value in encoded["labels"][:prompt_length]))
        self.assertTrue(all(value != -100 for value in encoded["labels"][prompt_length:]))
        self.assertEqual(len(encoded["labels"]), len(encoded["input_ids"]))

    def test_user_assistant_prompt_uses_the_same_masking_contract(self) -> None:
        tokenizer = FakeTokenizer()
        item = {
            "id": "mschema-row",
            "messages": [
                {"role": "user", "content": "M-Schema and question"},
                {"role": "assistant", "content": "SELECT 1"},
            ],
        }
        encoded = training.encode_conversation(tokenizer, item, max_seq_length=1000)
        prompt = tokenizer.apply_chat_template(item["messages"][:-1], tokenize=False, add_generation_prompt=True)
        prompt_length = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        self.assertTrue(all(value == -100 for value in encoded["labels"][:prompt_length]))
        self.assertTrue(all(value != -100 for value in encoded["labels"][prompt_length:]))

    def test_encoding_refuses_silent_truncation(self) -> None:
        with self.assertRaisesRegex(ValueError, "will not silently truncate"):
            training.encode_conversation(FakeTokenizer(), row(1, size=100), max_seq_length=20)

    def test_smoke_subset_spans_shortest_and_longest(self) -> None:
        rows = [row(index, size=index + 1) for index in range(20)]
        selected = training.select_length_stratified(rows, 5)
        self.assertEqual(len(selected), 5)
        self.assertEqual(selected[0]["id"], "row-0")
        self.assertEqual(selected[-1]["id"], "row-19")
        self.assertEqual(len({item["id"] for item in selected}), 5)

    def test_auto_resume_uses_latest_numeric_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkpoint-2").mkdir()
            (root / "checkpoint-10").mkdir()
            (root / "checkpoint-bad").mkdir()
            self.assertEqual(training.resolve_resume_checkpoint(root, "auto"), root / "checkpoint-10")

    def test_smoke_config_overrides_model_batch_and_cli_limit(self) -> None:
        config = {
            "optimization": {"seed": 17},
            "models": {"model": {"per_device_train_batch_size": 8, "max_seq_length": 4096}},
            "smoke": {
                "per_device_train_batch_size": 1,
                "max_steps": 4,
                "resume_phase_one_steps": 2,
                "train_limit": 64,
            },
            "full": {"max_steps": -1},
        }
        args = argparse.Namespace(max_steps=6, train_limit=None, validation_limit=None)
        merged = bundler.merge_training(config, "model", True, args)
        self.assertEqual(merged["per_device_train_batch_size"], 1)
        self.assertEqual(merged["max_steps"], 6)
        self.assertEqual(merged["max_seq_length"], 4096)

    def test_checkpoint_export_is_complete_and_validatable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            checkpoint = output / "checkpoint-100"
            checkpoint.mkdir(parents=True)
            for name in checkpoint_inspector.REQUIRED:
                (checkpoint / name).write_bytes(name.encode("utf-8"))
            status = root / "status.json"
            exported = training.export_resume_checkpoint(output, checkpoint, status, "test")
            archive = Path(exported["archive"])
            validated = checkpoint_inspector.validate(
                archive,
                expected_sha256=str(exported["sha256"]),
                expected_bytes=int(exported["bytes"]),
            )
            self.assertEqual(validated["step"], 100)
            payload = json.loads(status.read_text(encoding="utf-8"))
            self.assertEqual(payload["phase"], "checkpoint_exported")
            self.assertEqual(payload["checkpoint_export"]["step"], 100)

    def test_checkpoint_inspector_rejects_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "payload"
            source.write_text("unsafe", encoding="utf-8")
            archive_path = root / "unsafe.tar"
            with tarfile.open(archive_path, "w") as archive:
                archive.add(source, arcname="../payload")
            with self.assertRaisesRegex(RuntimeError, "Unsafe archive member"):
                checkpoint_inspector.validate(archive_path)


if __name__ == "__main__":
    unittest.main()
