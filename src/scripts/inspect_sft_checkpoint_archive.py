#!/usr/bin/env python3
"""Read checkpoint-export metadata and validate resumable tar archives."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


REQUIRED = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "rng_state.pth",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    export = payload.get("checkpoint_export")
    if not isinstance(export, dict):
        return 2
    print(
        "\t".join(
            (
                str(int(export["step"])),
                str(export["archive"]),
                str(int(export["bytes"])),
                str(export["sha256"]),
            )
        )
    )
    return 0


def validate(
    path: Path, expected_sha256: str | None = None, expected_bytes: int | None = None
) -> dict[str, object]:
    actual_bytes = path.stat().st_size
    if expected_bytes is not None and actual_bytes != expected_bytes:
        raise RuntimeError(f"Archive size mismatch: expected {expected_bytes}, got {actual_bytes}")
    actual_sha256 = sha256(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Archive checksum mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    with tarfile.open(path, "r") as archive:
        members = archive.getmembers()
        if not members:
            raise RuntimeError("Checkpoint archive is empty")
        names = []
        for member in members:
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or ".." in pure.parts:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"Checkpoint archive cannot contain links: {member.name}")
            names.append(pure)
    roots = {name.parts[0] for name in names if name.parts}
    if len(roots) != 1:
        raise RuntimeError(f"Expected one checkpoint root, found: {sorted(roots)}")
    root = next(iter(roots))
    if not root.startswith("checkpoint-"):
        raise RuntimeError(f"Invalid checkpoint root: {root}")
    try:
        step = int(root.rsplit("-", 1)[1])
    except ValueError as exc:
        raise RuntimeError(f"Invalid checkpoint step: {root}") from exc
    files = {name.name for name in names if len(name.parts) == 2}
    missing = sorted(REQUIRED - files)
    if missing:
        raise RuntimeError(f"Checkpoint archive is missing required files: {missing}")
    return {
        "archive": str(path.resolve()),
        "bytes": actual_bytes,
        "sha256": actual_sha256,
        "checkpoint": root,
        "step": step,
        "required_files_verified": sorted(REQUIRED),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    metadata_parser = subparsers.add_parser("metadata")
    metadata_parser.add_argument("status", type=Path)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("archive", type=Path)
    validate_parser.add_argument("--expected-sha256")
    validate_parser.add_argument("--expected-bytes", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "metadata":
        return metadata(args.status)
    result = validate(args.archive, args.expected_sha256, args.expected_bytes)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
