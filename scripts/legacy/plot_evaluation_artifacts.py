from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve


METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "roc_auc"]
SCORE_FIELDS = ["anomaly_probability", "fused_anomaly_score", "anomaly_score", "score"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate report plots from a saved evaluation.json run.")
    parser.add_argument(
        "--eval",
        type=Path,
        required=True,
        help="Path to evaluation.json.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Optional predictions.csv. Defaults to predictions.csv next to evaluation.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/figures"),
        help="Directory where PNG/PDF plots are written.",
    )
    parser.add_argument(
        "--prefix",
        default="run",
        help="Filename prefix for generated plots.",
    )
    args = parser.parse_args()

    evaluation = json.loads(args.eval.read_text(encoding="utf-8"))
    predictions_path = args.predictions or args.eval.with_name("predictions.csv")
    predictions = read_predictions(predictions_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_metric_bar(evaluation, args.output_dir, args.prefix)
    write_label_bars(evaluation, args.output_dir, args.prefix)
    write_confusion(predictions, args.output_dir, args.prefix)
    write_roc_pr(predictions, args.output_dir, args.prefix)

    print(f"Wrote plots to {args.output_dir}")
    for path in sorted(args.output_dir.glob(f"{args.prefix}-*.png")):
        print(path)


def read_predictions(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"No predictions found in {path}")
    return rows


def score_field(predictions: list[dict[str, str]]) -> str:
    fields = set(predictions[0])
    for field in SCORE_FIELDS:
        if field in fields:
            return field
    raise SystemExit(f"No supported score column found. Expected one of: {', '.join(SCORE_FIELDS)}")


def write_metric_bar(evaluation: dict, output_dir: Path, prefix: str) -> None:
    test = evaluation["test"]
    values = [float(test[name]) for name in METRIC_NAMES]
    labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(labels, values, color=["#2f6fbb", "#4f9d69", "#d08c2f", "#8d5fbf", "#3c8c8c"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Held-Out Test Metrics")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", va="bottom")
    save_figure(fig, output_dir / f"{prefix}-metrics")


def write_label_bars(evaluation: dict, output_dir: Path, prefix: str) -> None:
    by_label = evaluation.get("by_label", {})
    if not by_label:
        return

    labels = list(by_label)
    flagged = [float(by_label[label]["flagged_rate"]) for label in labels]
    scores = [float(by_label[label]["mean_score"]) for label in labels]

    write_horizontal_bar(
        labels,
        flagged,
        title="Flagged Rate By Label",
        xlabel="Flagged Rate",
        output=output_dir / f"{prefix}-flag-rate-by-label",
        xlim=(0, 1.0),
        value_format="{:.1%}",
    )
    write_horizontal_bar(
        labels,
        scores,
        title="Mean Anomaly Score By Label",
        xlabel="Mean Anomaly Score",
        output=output_dir / f"{prefix}-mean-score-by-label",
        xlim=None,
        value_format="{:.3f}",
    )


def write_horizontal_bar(
    labels: list[str],
    values: list[float],
    title: str,
    xlabel: str,
    output: Path,
    xlim: tuple[float, float] | None,
    value_format: str,
) -> None:
    order = np.argsort(values)
    ordered_labels = [labels[index] for index in order]
    ordered_values = [values[index] for index in order]

    height = max(4.5, 0.45 * len(labels) + 1.8)
    fig, ax = plt.subplots(figsize=(9.0, height))
    bars = ax.barh(ordered_labels, ordered_values, color="#2f6fbb")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    if xlim is not None:
        ax.set_xlim(*xlim)
    right = ax.get_xlim()[1]
    for bar, value in zip(bars, ordered_values):
        ax.text(min(value + right * 0.015, right * 0.98), bar.get_y() + bar.get_height() / 2, value_format.format(value), va="center")
    save_figure(fig, output)


def write_confusion(predictions: list[dict[str, str]], output_dir: Path, prefix: str) -> None:
    y_true = np.asarray([int(row["y_true"]) for row in predictions])
    y_pred = np.asarray([int(row["flagged"]) for row in predictions])
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred Normal", "Pred Anomaly"])
    ax.set_yticks([0, 1], labels=["True Normal", "True Anomaly"])
    ax.set_title("Confusion Matrix")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > matrix.max() * 0.55 else "black"
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color=color, fontsize=13)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, output_dir / f"{prefix}-confusion-matrix")


def write_roc_pr(predictions: list[dict[str, str]], output_dir: Path, prefix: str) -> None:
    field = score_field(predictions)
    y_true = np.asarray([int(row["y_true"]) for row in predictions])
    scores = np.asarray([float(row[field]) for row in predictions])
    if len(set(y_true.tolist())) < 2:
        return

    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.plot(fpr, tpr, color="#2f6fbb", linewidth=2.2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1.0)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    save_figure(fig, output_dir / f"{prefix}-roc-curve")

    precision, recall, _ = precision_recall_curve(y_true, scores)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.plot(recall, precision, color="#4f9d69", linewidth=2.2, label=f"AUC = {pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.25)
    save_figure(fig, output_dir / f"{prefix}-precision-recall-curve")


def save_figure(fig: plt.Figure, path_without_suffix: Path) -> None:
    fig.tight_layout()
    fig.savefig(path_without_suffix.with_suffix(".png"), dpi=220)
    fig.savefig(path_without_suffix.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
