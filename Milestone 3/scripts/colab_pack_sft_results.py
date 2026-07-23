"""Create a compact QLoRA result archive, including only the latest full checkpoint."""

from __future__ import annotations

from pathlib import Path
import tarfile


ROOT = Path("/content/text2sql_sft")
OUTPUT = ROOT / "output"
DESTINATION = ROOT / "sft-results-transfer.tar.gz"


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return -1


latest_checkpoint = max(
    (path for path in OUTPUT.glob("checkpoint-*") if checkpoint_step(path) >= 0),
    key=checkpoint_step,
    default=None,
)
with tarfile.open(DESTINATION, "w:gz") as archive:
    for relative in (
        Path("status.json"),
        Path("launch_config.json"),
        Path("bundle_manifest.json"),
        Path("environment.json"),
    ):
        path = ROOT / relative
        if path.exists():
            archive.add(path, arcname=relative)
    if OUTPUT.exists():
        for path in sorted(OUTPUT.rglob("*")):
            if not path.is_file():
                continue
            if OUTPUT / "checkpoint_exports" in path.parents:
                continue
            if any(parent.name.startswith("checkpoint-") for parent in path.parents):
                if latest_checkpoint is None or latest_checkpoint not in path.parents:
                    continue
            archive.add(path, arcname=path.relative_to(ROOT))
print(
    f"COLAB_SFT_RESULTS_ARCHIVE={DESTINATION} bytes={DESTINATION.stat().st_size} "
    f"latest_checkpoint={latest_checkpoint.name if latest_checkpoint else 'none'}"
)
