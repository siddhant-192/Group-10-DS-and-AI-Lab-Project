#!/usr/bin/env python3
"""Render Milestone 4 training and hyperparameter evidence from saved artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "output" / "figures" / "milestone4"
FINAL_TRAINING = ROOT / "evidence" / "milestone4" / "final_training"
TRAINER_STATE = FINAL_TRAINING / "trainer_state.json"
EVAL_RESULTS = FINAL_TRAINING / "eval_results.json"
SEARCH_SUMMARY = ROOT / "evidence" / "milestone4" / "hparam" / "search_summary.csv"


def render_training_diagnostics() -> None:
    state = json.loads(TRAINER_STATE.read_text())
    eval_metrics = json.loads(EVAL_RESULTS.read_text())
    rows = [row for row in state["log_history"] if "loss" in row]

    steps = [row["step"] for row in rows]
    losses = [row["loss"] for row in rows]
    learning_rates = [row["learning_rate"] for row in rows]
    grad_norms = [row["grad_norm"] for row in rows]
    final_eval_loss = eval_metrics["eval_loss"]

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8), sharex=True)

    axes[0].plot(steps, losses, color="#2563eb", marker="o", markersize=3, linewidth=1.8)
    axes[0].scatter(
        [state["global_step"]],
        [final_eval_loss],
        color="#dc2626",
        marker="D",
        s=55,
        zorder=5,
        label=f"Final validation loss = {final_eval_loss:.4f} (measured once)",
    )
    for checkpoint in (100, 200, 300, 350):
        axes[0].axvline(checkpoint, color="#9ca3af", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Assistant-token cross-entropy")
    axes[0].set_title("Final Qwen3-4B QLoRA training diagnostics")
    axes[0].grid(alpha=0.2)
    axes[0].legend(loc="upper right", frameon=False)

    axes[1].plot(
        steps,
        learning_rates,
        color="#7c3aed",
        linewidth=1.8,
        label="Learning rate",
    )
    axes[1].set_ylabel("Learning rate", color="#7c3aed")
    axes[1].tick_params(axis="y", labelcolor="#7c3aed")
    axes[1].set_xlabel("Optimizer step")
    axes[1].grid(alpha=0.2)

    grad_axis = axes[1].twinx()
    grad_axis.plot(
        steps,
        grad_norms,
        color="#059669",
        linewidth=1.2,
        alpha=0.75,
        label="Gradient norm",
    )
    grad_axis.axhline(
        0.3,
        color="#ea580c",
        linestyle=":",
        linewidth=1.2,
        label="Configured clipping threshold",
    )
    grad_axis.set_ylabel("Pre-clipping gradient norm", color="#059669")
    grad_axis.tick_params(axis="y", labelcolor="#059669")

    lines_left, labels_left = axes[1].get_legend_handles_labels()
    lines_right, labels_right = grad_axis.get_legend_handles_labels()
    axes[1].legend(
        lines_left + lines_right,
        labels_left + labels_right,
        loc="upper right",
        frameon=False,
    )

    fig.text(
        0.5,
        0.01,
        "Dashed vertical lines mark locally verified resumable checkpoints. "
        "Validation loss was evaluated only at the final epoch.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(FIGURES / "final_training_diagnostics.png", dpi=180)
    plt.close(fig)


def render_hparam_comparison() -> None:
    with SEARCH_SUMMARY.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    scored = [
        row
        for row in rows
        if row.get("strict_execution_pct") not in (None, "", "pending")
        and row.get("compatible_execution_pct") not in (None, "", "pending")
    ]
    labels = [row["label"] for row in scored]
    strict = [float(row["strict_execution_pct"]) for row in scored]
    compatible = [float(row["compatible_execution_pct"]) for row in scored]

    y_positions = list(range(len(scored)))
    height = 0.38
    fig, axis = plt.subplots(figsize=(11, 7.5))
    axis.barh(
        [position + height / 2 for position in y_positions],
        strict,
        height=height,
        color="#2563eb",
        label="Strict execution",
    )
    axis.barh(
        [position - height / 2 for position in y_positions],
        compatible,
        height=height,
        color="#10b981",
        label="Compatible execution",
    )
    axis.set_yticks(y_positions)
    axis.set_yticklabels(labels)
    axis.invert_yaxis()
    axis.set_xlim(80, 91)
    axis.set_xlabel("Execution accuracy (%)")
    axis.set_title("Scored Qwen3 QLoRA hyperparameter trials (1,001-example tuning set)")
    axis.grid(axis="x", alpha=0.2)
    axis.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "hparam_execution_accuracy.png", dpi=180)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    render_training_diagnostics()
    render_hparam_comparison()
    print(FIGURES / "final_training_diagnostics.png")
    print(FIGURES / "hparam_execution_accuracy.png")


if __name__ == "__main__":
    main()
