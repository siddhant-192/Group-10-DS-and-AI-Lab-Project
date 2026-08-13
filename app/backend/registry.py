"""DB-agnostic registry over demo_databases/*.sqlite."""

from __future__ import annotations

from pathlib import Path


def list_databases(demo_dir: Path) -> dict[str, Path]:
    """Map db_id (file stem) -> absolute sqlite path."""

    root = demo_dir.resolve()
    if not root.exists():
        return {}
    found: dict[str, Path] = {}
    for path in sorted(root.glob("*.sqlite")):
        found[path.stem] = path
    for path in sorted(root.glob("*.db")):
        found.setdefault(path.stem, path)
    return found


def resolve_database(demo_dir: Path, db_id: str) -> Path:
    registry = list_databases(demo_dir)
    if db_id not in registry:
        available = ", ".join(sorted(registry)) or "(none)"
        raise KeyError(f"Unknown db_id {db_id!r}. Available: {available}")
    return registry[db_id]
