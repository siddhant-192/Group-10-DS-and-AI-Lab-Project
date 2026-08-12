#!/usr/bin/env python3
"""
Summarizes latency (generation_ms_per_example, already stored per-row in
predictions.jsonl) and peak VRAM usage (cuda_peak, already stored in
metrics.json) from your completed BIRD evaluation runs.

This directly answers M4's explicit "Milestone 5 should ... report
execution, syntax, exact match, latency, VRAM, and complexity slices"
requirement -- the data was already captured during evaluation, this
script just aggregates and reports it. No re-run, no GPU needed.

Usage:
    python summarize_latency_vram.py \
        --predictions evidence/milestone5/evaluation_mschema/milestone4_frozen/predictions.jsonl \
        --metrics evidence/milestone5/evaluation_mschema/milestone4_frozen/metrics.json \
        --run-label "M-Schema (primary)"
"""

import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True,
                         help="metrics.json written by evaluate_text2sql_models.py for this run")
    parser.add_argument("--run-label", type=str, default="run")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    with open(args.predictions, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    latencies = [
        float(r["generation_ms_per_example"])
        for r in rows
        if r.get("generation_ms_per_example") is not None
    ]

    print(f"[{args.run_label}] {len(rows)} predictions, {len(latencies)} with latency recorded\n")

    print("=" * 60)
    print("LATENCY (ms per example, batched generation)")
    print("=" * 60)
    if latencies:
        print(f"  mean:    {statistics.mean(latencies):.1f} ms")
        print(f"  median:  {statistics.median(latencies):.1f} ms")
        print(f"  min:     {min(latencies):.1f} ms")
        print(f"  max:     {max(latencies):.1f} ms")
        if len(latencies) > 1:
            print(f"  stdev:   {statistics.stdev(latencies):.1f} ms")
        total_seconds = sum(latencies) / 1000
        print(f"  total generation time (sum): {total_seconds:.1f} s ({total_seconds/60:.1f} min)")
    else:
        print("  No latency data found in predictions.jsonl -- check that the eval run used "
              "a version of evaluate_text2sql_models.py that records generation_ms_per_example.")

    print()
    print("=" * 60)
    print("PEAK GPU MEMORY (from metrics.json cuda_peak)")
    print("=" * 60)
    with open(args.metrics, encoding="utf-8") as f:
        metrics = json.load(f)

    cuda_peak = metrics.get("cuda_peak", {})
    if cuda_peak:
        for key in ("allocated_gib", "reserved_gib", "max_allocated_gib", "max_reserved_gib"):
            if key in cuda_peak:
                print(f"  {key}: {cuda_peak[key]:.2f} GiB")
    else:
        print("  No cuda_peak field found in metrics.json.")

    for key in ("final_batch_size", "attention", "dtype", "load_seconds", "evaluation_seconds"):
        if key in metrics:
            print(f"  {key}: {metrics[key]}")

    if args.output_json:
        summary = {
            "run_label": args.run_label,
            "n_predictions": len(rows),
            "n_with_latency": len(latencies),
            "latency_ms": {
                "mean": statistics.mean(latencies) if latencies else None,
                "median": statistics.median(latencies) if latencies else None,
                "min": min(latencies) if latencies else None,
                "max": max(latencies) if latencies else None,
                "stdev": statistics.stdev(latencies) if len(latencies) > 1 else None,
            },
            "cuda_peak": cuda_peak,
            "final_batch_size": metrics.get("final_batch_size"),
            "load_seconds": metrics.get("load_seconds"),
            "evaluation_seconds": metrics.get("evaluation_seconds"),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary written to: {args.output_json}")


if __name__ == "__main__":
    main()
