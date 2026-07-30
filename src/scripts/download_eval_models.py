#!/usr/bin/env python3
"""Download and pin the three zero-shot evaluation checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

from huggingface_hub import HfApi, snapshot_download
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "text2sql_eval_models.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "text2sql-eval"
MANIFEST_NAME = "download_manifest.json"
RESERVED_FREE_BYTES = 3 * 1024**3
ALLOW_PATTERNS = (
    "*.json",
    "*.jinja",
    "*.md",
    "*.model",
    "*.py",
    "*.safetensors",
    "*.tiktoken",
    "*.txt",
    "LICENSE*",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Download only this slug; repeat to select multiple models.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_specs(path: Path, selected: set[str] | None) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    specs = payload.get("models")
    if not isinstance(specs, list) or not specs:
        raise ValueError(f"No models found in {path}")
    if selected:
        known = {str(spec["slug"]) for spec in specs}
        unknown = selected - known
        if unknown:
            raise ValueError(f"Unknown model slug(s): {', '.join(sorted(unknown))}")
        specs = [spec for spec in specs if str(spec["slug"]) in selected]
    return specs


def remote_snapshot_size(info: Any) -> int:
    total = 0
    for sibling in info.siblings:
        if not any(Path(sibling.rfilename).match(pattern) for pattern in ALLOW_PATTERNS):
            continue
        size = getattr(sibling, "size", None)
        if size is None and getattr(sibling, "lfs", None):
            size = sibling.lfs.get("size")
        total += int(size or 0)
    return total


def human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024
    raise AssertionError("unreachable")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    selected = set(args.models) if args.models else None
    specs = load_specs(args.config.resolve(), selected)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    resolved: list[tuple[dict[str, Any], Any, int]] = []
    print("Resolving public Hugging Face revisions and file sizes...")
    for spec in tqdm(specs, unit="model"):
        info = api.model_info(
            str(spec["repo_id"]), revision=str(spec.get("revision", "main")), files_metadata=True
        )
        resolved.append((spec, info, remote_snapshot_size(info)))

    required = sum(size for _, _, size in resolved)
    existing = 0
    for spec, _info, size in resolved:
        destination = output_dir / str(spec["slug"])
        present = sum(
            path.stat().st_size
            for path in destination.rglob("*")
            if path.is_file() and ".cache" not in path.parts
        ) if destination.exists() else 0
        existing += min(size, present)
    remaining = max(0, required - existing)
    free = shutil.disk_usage(output_dir).free
    print(f"Selected snapshot bytes: {human_bytes(required)}")
    print(f"Already present:         {human_bytes(existing)}")
    print(f"Estimated remaining:     {human_bytes(remaining)}")
    print(f"Local free space:        {human_bytes(free)}")
    if remaining and free < remaining + RESERVED_FREE_BYTES:
        raise RuntimeError(
            "Insufficient disk space for all checkpoints plus a 3 GiB safety reserve. "
            f"Need approximately {human_bytes(remaining + RESERVED_FREE_BYTES)}, "
            f"but only {human_bytes(free)} is free."
        )

    if args.dry_run:
        for spec, info, size in resolved:
            print(f"{spec['slug']}: {spec['repo_id']}@{info.sha} ({human_bytes(size)})")
        return 0

    manifest: dict[str, Any] = {"format_version": 1, "models": []}
    for index, (spec, info, _) in enumerate(resolved, start=1):
        destination = output_dir / str(spec["slug"])
        print(f"\n[{index}/{len(resolved)}] Downloading {spec['repo_id']}@{info.sha}")
        snapshot_download(
            repo_id=str(spec["repo_id"]),
            revision=info.sha,
            local_dir=destination,
            allow_patterns=list(ALLOW_PATTERNS),
            max_workers=max(1, args.max_workers),
        )

        weight_files = sorted(destination.glob("*.safetensors"))
        if not weight_files:
            raise RuntimeError(f"No safetensors weights found in {destination}")
        files = []
        for path in sorted(p for p in destination.rglob("*") if p.is_file()):
            if ".cache" in path.parts:
                continue
            files.append(
                {
                    "path": path.relative_to(destination).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
        manifest["models"].append(
            {
                **spec,
                "revision": info.sha,
                "local_dir": destination.relative_to(PROJECT_ROOT).as_posix(),
                "bytes": sum(item["bytes"] for item in files),
                "files": files,
            }
        )

    manifest_path = output_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nPinned download manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nDownload interrupted; rerun the same command to resume.", file=sys.stderr)
        raise SystemExit(130)
