"""Pack compact QLoRA HPO outputs without base weights or optimizer checkpoints."""

from pathlib import Path
import tarfile


ROOT = Path("/content/text2sql_sft")
OUTPUT = ROOT / "output"
DESTINATION = ROOT / "hparam-results-transfer.tar.gz"
TOP_LEVEL = (
    "status.json",
    "launch_config.json",
    "bundle_manifest.json",
    "environment.json",
)
OUTPUT_FILES = (
    "training.log",
    "trainer_metrics.jsonl",
    "phase_history.jsonl",
    "train_results.json",
    "eval_results.json",
    "trainer_state.json",
    "run_manifest.json",
    "tokenization_summary.json",
)

with tarfile.open(DESTINATION, "w:gz") as archive:
    for name in TOP_LEVEL:
        path = ROOT / name
        if path.is_file():
            archive.add(path, arcname=name)
    for name in OUTPUT_FILES:
        path = OUTPUT / name
        if path.is_file():
            archive.add(path, arcname=f"output/{name}")
    adapter = OUTPUT / "final_adapter"
    if not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(adapter / "adapter_model.safetensors")
    for path in sorted(adapter.rglob("*")):
        if path.is_file():
            archive.add(path, arcname=path.relative_to(ROOT))

print(f"COLAB_HPARAM_RESULTS_ARCHIVE={DESTINATION} bytes={DESTINATION.stat().st_size}")
