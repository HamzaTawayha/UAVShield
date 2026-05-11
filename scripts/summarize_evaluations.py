from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRIC_FIELDS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "normal_flag_rate",
    "anomaly_flag_rate",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create report-ready metric tables from saved evaluation.json files."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data"),
        help="Directory to scan recursively for evaluation.json files.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("data/evaluation_summary.md"),
        help="Markdown table output path.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/evaluation_summary.csv"),
        help="CSV table output path.",
    )
    parser.add_argument(
        "--include-by-label",
        action="store_true",
        help="Also write a per-label Markdown table next to the main Markdown output.",
    )
    args = parser.parse_args()

    evaluations = sorted(args.root.rglob("evaluation.json"))
    if not evaluations:
        raise SystemExit(f"No evaluation.json files found under {args.root}")

    rows = [load_summary(path, args.root) for path in evaluations]
    rows = sorted(rows, key=lambda row: (row["dataset"], row["experiment"]))

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    markdown = render_markdown(rows)
    args.output_md.write_text(markdown, encoding="utf-8")
    write_csv(args.output_csv, rows)

    print(markdown)
    print(f"\nWrote Markdown summary to {args.output_md}")
    print(f"Wrote CSV summary to {args.output_csv}")

    if args.include_by_label:
        by_label_rows = []
        for path in evaluations:
            by_label_rows.extend(load_by_label(path, args.root))
        by_label_path = args.output_md.with_name(f"{args.output_md.stem}_by_label.md")
        by_label_path.write_text(render_by_label_markdown(by_label_rows), encoding="utf-8")
        print(f"Wrote per-label Markdown summary to {by_label_path}")


def load_summary(path: Path, root: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    test = payload.get("test", {})
    parent = path.parent

    row: dict[str, Any] = {
        "dataset": infer_dataset(path),
        "experiment": parent.name,
        "path": str(path.relative_to(root.parent if root.parent != Path("") else Path("."))),
        "model": payload.get("model", ""),
        "threshold_strategy": payload.get("threshold_strategy", ""),
        "threshold": payload.get("threshold", ""),
        "windows": test.get("windows", ""),
        "train_windows": payload.get("train", {}).get("windows", ""),
        "calibration_windows": payload.get("calibration", {}).get("windows", ""),
    }
    for field in METRIC_FIELDS:
        row[field] = test.get(field, "")
    fill_missing_flag_rates(row, payload)
    return row


def infer_dataset(path: Path) -> str:
    parts = path.parts
    if "uav_sead" in parts:
        return "UAV-SEAD"
    if "px4_flight_review" in parts:
        return "PX4 Flight Review"
    if len(parts) >= 2 and parts[0] == "data":
        return parts[1]
    return "unknown"


def fill_missing_flag_rates(row: dict[str, Any], payload: dict[str, Any]) -> None:
    if row.get("normal_flag_rate") != "" and row.get("anomaly_flag_rate") != "":
        return

    predictions = payload.get("predictions", [])
    if predictions:
        normal_flags = []
        anomaly_flags = []
        for prediction in predictions:
            y_true = int(prediction.get("y_true", -1))
            flagged = int(prediction.get("flagged", 0))
            if y_true == 0:
                normal_flags.append(flagged)
            elif y_true == 1:
                anomaly_flags.append(flagged)
        if row.get("normal_flag_rate") == "" and normal_flags:
            row["normal_flag_rate"] = sum(normal_flags) / len(normal_flags)
        if row.get("anomaly_flag_rate") == "" and anomaly_flags:
            row["anomaly_flag_rate"] = sum(anomaly_flags) / len(anomaly_flags)
        return

    by_label = payload.get("by_label", {})
    if row.get("normal_flag_rate") == "":
        for label, stats in by_label.items():
            if str(label).lower().startswith("normal"):
                row["normal_flag_rate"] = stats.get("flagged_rate", "")
                break
    if row.get("anomaly_flag_rate") == "":
        flagged = 0
        total = 0
        for label, stats in by_label.items():
            if str(label).lower().startswith("normal"):
                continue
            n = int(stats.get("n", 0))
            flagged += int(stats.get("flagged_count", 0))
            total += n
        if total > 0:
            row["anomaly_flag_rate"] = flagged / total


def render_markdown(rows: list[dict[str, Any]]) -> str:
    headers = [
        "Dataset",
        "Experiment",
        "Model",
        "Strategy",
        "Threshold",
        "Windows",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
        "Normal Flag",
        "Anomaly Flag",
    ]
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row["dataset"],
                row["experiment"],
                compact_model_name(str(row["model"])),
                row["threshold_strategy"] or "-",
                fmt_float(row["threshold"], 4),
                fmt_int(row["windows"]),
                fmt_pct(row["accuracy"]),
                fmt_pct(row["precision"]),
                fmt_pct(row["recall"]),
                fmt_float(row["f1"], 4),
                fmt_float(row["roc_auc"], 4),
                fmt_pct(row["normal_flag_rate"]),
                fmt_pct(row["anomaly_flag_rate"]),
            ]
        )
    return markdown_table(headers, table_rows)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "dataset",
        "experiment",
        "model",
        "threshold_strategy",
        "threshold",
        "windows",
        "train_windows",
        "calibration_windows",
        *METRIC_FIELDS,
        "path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def load_by_label(path: Path, root: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for label, stats in payload.get("by_label", {}).items():
        rows.append(
            {
                "dataset": infer_dataset(path),
                "experiment": path.parent.name,
                "label": label,
                "n": stats.get("n", ""),
                "mean_score": stats.get("mean_score", ""),
                "flagged_rate": stats.get("flagged_rate", ""),
                "path": str(path.relative_to(root.parent if root.parent != Path("") else Path("."))),
            }
        )
    return sorted(rows, key=lambda row: (row["dataset"], row["experiment"], row["label"]))


def render_by_label_markdown(rows: list[dict[str, Any]]) -> str:
    headers = ["Dataset", "Experiment", "Label", "n", "Mean Score", "Flagged"]
    table_rows = [
        [
            row["dataset"],
            row["experiment"],
            row["label"],
            fmt_int(row["n"]),
            fmt_float(row["mean_score"], 4),
            fmt_pct(row["flagged_rate"]),
        ]
        for row in rows
    ]
    return markdown_table(headers, table_rows)


def compact_model_name(model: str) -> str:
    if "MOMENT-1-large embeddings" in model:
        return model.replace("AutonLab/MOMENT-1-large embeddings + ", "MOMENT embeddings + ")
    if "MOMENT-1-large reconstruction" in model:
        return "MOMENT reconstruction"
    return model or "-"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(escape_cell(str(cell)) for cell in row) + " |")
    return "\n".join(lines) + "\n"


def escape_cell(value: str) -> str:
    return value.replace("|", "\\|")


def fmt_int(value: Any) -> str:
    try:
        if value == "":
            return "-"
        return str(int(value))
    except (TypeError, ValueError):
        return "-"


def fmt_float(value: Any, digits: int) -> str:
    try:
        if value == "":
            return "-"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def fmt_pct(value: Any) -> str:
    try:
        if value == "":
            return "-"
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "-"


if __name__ == "__main__":
    main()
