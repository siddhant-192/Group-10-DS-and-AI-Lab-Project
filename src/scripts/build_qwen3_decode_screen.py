#!/usr/bin/env python3
"""Build a deterministic, complexity-stratified decoding-tuning subset."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2717)
    return parser.parse_args()


def allocate(counts: Counter[str], total: int) -> dict[str, int]:
    available = sum(counts.values())
    if total <= 0 or total > available:
        raise ValueError(f"size must be in [1, {available}]")
    exact = {key: total * value / available for key, value in counts.items()}
    result = {key: int(value) for key, value in exact.items()}
    remaining = total - sum(result.values())
    for key in sorted(counts, key=lambda item: (exact[item] - result[item], item), reverse=True):
        if remaining == 0:
            break
        result[key] += 1
        remaining -= 1
    return result


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    groups: dict[str, list[dict]] = defaultdict(list)
    def complexity(row: dict) -> str:
        return str(
            row.get("complexity")
            or row.get("metadata", {}).get("query_features", {}).get("complexity_proxy")
            or "unknown"
        )

    for row in rows:
        groups[complexity(row)].append(row)

    counts = Counter({key: len(value) for key, value in groups.items()})
    quotas = allocate(counts, args.size)
    rng = random.Random(args.seed)
    selected: list[dict] = []
    for key in sorted(groups):
        candidates = list(groups[key])
        rng.shuffle(candidates)
        selected.extend(candidates[: quotas[key]])
    rng.shuffle(selected)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in selected))
    selected_counts = Counter(complexity(row) for row in selected)
    databases = {str(row.get("db_id")) for row in selected}
    print(json.dumps({
        "input_rows": len(rows),
        "output_rows": len(selected),
        "seed": args.seed,
        "complexity": dict(sorted(selected_counts.items())),
        "databases": len(databases),
    }, indent=2))


if __name__ == "__main__":
    main()
