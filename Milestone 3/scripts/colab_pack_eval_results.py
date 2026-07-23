"""Create a downloadable archive even after a partially completed run."""

from __future__ import annotations

from pathlib import Path
import tarfile


ROOT = Path("/content/text2sql_eval")
DESTINATION = ROOT / "results-transfer.tar.gz"
with tarfile.open(DESTINATION, "w:gz") as archive:
    for relative in (Path("results"), Path("status.json"), Path("launch_config.json")):
        path = ROOT / relative
        if path.exists():
            archive.add(path, arcname=relative)
print(f"COLAB_RESULTS_ARCHIVE={DESTINATION} bytes={DESTINATION.stat().st_size}")
