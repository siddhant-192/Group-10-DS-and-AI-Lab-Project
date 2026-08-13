#!/usr/bin/env python3
"""Populate demo_databases/ with a tiny built-in DB and optional Chinook / Spider copies."""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEMO_DIR = PROJECT_ROOT / "demo_databases"

# Public Chinook SQLite mirror (same schema used widely for SQL demos).
CHINOOK_URL = (
    "https://github.com/lerocha/chinook-database/raw/master/"
    "ChinookDatabase/DataSources/Chinook_Sqlite.sqlite"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-dir", type=Path, default=DEFAULT_DEMO_DIR)
    parser.add_argument(
        "--skip-chinook",
        action="store_true",
        help="Do not download Chinook (still creates mini_music.sqlite).",
    )
    parser.add_argument(
        "--copy-spider-from",
        type=Path,
        default=None,
        help="Spider database root containing <db_id>/<db_id>.sqlite folders.",
    )
    parser.add_argument(
        "--spider-ids",
        nargs="*",
        default=["concert_singer", "cars_data"],
        help="Which Spider db_ids to copy when --copy-spider-from is set.",
    )
    return parser.parse_args()


def create_mini_music(path: Path) -> None:
    """Small always-available demo DB so mock/ask works offline."""

    if path.exists():
        path.unlink()
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE singer (
                Singer_ID INTEGER PRIMARY KEY,
                Name TEXT,
                Country TEXT,
                Age INTEGER
            );
            CREATE TABLE concert (
                concert_ID INTEGER PRIMARY KEY,
                concert_Name TEXT,
                Theme TEXT,
                Singer_ID INTEGER,
                FOREIGN KEY (Singer_ID) REFERENCES singer(Singer_ID)
            );
            INSERT INTO singer VALUES
                (1, 'Taylor Swift', 'USA', 34),
                (2, 'Adele', 'UK', 35),
                (3, 'Arijit Singh', 'India', 36);
            INSERT INTO concert VALUES
                (10, 'Night One', 'Pop', 1),
                (11, 'Hello Tour', 'Ballad', 2),
                (12, 'Soul Night', 'Bollywood', 3),
                (13, 'Encore', 'Pop', 1);
            """
        )
        connection.commit()
    finally:
        connection.close()


def download_chinook(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".download")
    print(f"Downloading Chinook -> {path}")
    urllib.request.urlretrieve(CHINOOK_URL, tmp)
    tmp.replace(path)


def copy_spider_dbs(spider_root: Path, demo_dir: Path, db_ids: list[str]) -> None:
    for db_id in db_ids:
        src = spider_root / db_id / f"{db_id}.sqlite"
        if not src.exists():
            # Also accept flat layout: spider_root/db_id.sqlite
            flat = spider_root / f"{db_id}.sqlite"
            src = flat if flat.exists() else src
        if not src.exists():
            print(f"Skip missing Spider DB: {db_id} (looked for {src})")
            continue
        dest = demo_dir / f"{db_id}.sqlite"
        print(f"Copy {src} -> {dest}")
        shutil.copy2(src, dest)


def main() -> int:
    args = parse_args()
    demo_dir = args.demo_dir.resolve()
    demo_dir.mkdir(parents=True, exist_ok=True)

    mini = demo_dir / "mini_music.sqlite"
    create_mini_music(mini)
    print(f"Wrote {mini}")

    if not args.skip_chinook:
        chinook = demo_dir / "chinook.sqlite"
        try:
            download_chinook(chinook)
            print(f"Wrote {chinook}")
        except Exception as exc:
            print(f"Chinook download failed ({exc}). mini_music.sqlite is still available.", file=sys.stderr)

    if args.copy_spider_from is not None:
        copy_spider_dbs(args.copy_spider_from.resolve(), demo_dir, list(args.spider_ids))

    print("Done. Registry db_ids:", sorted(p.stem for p in demo_dir.glob("*.sqlite")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
