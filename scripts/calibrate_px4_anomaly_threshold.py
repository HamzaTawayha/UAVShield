from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate a deployable anomaly-score threshold from normal PX4 windows."
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=Path("data/px4_flight_review/moment_windows/anomaly_scores.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/px4_flight_review/moment_windows/anomaly_threshold.json"),
    )
    parser.add_argument("--normal-label", default="normal_good")
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.95,
        help="Normal-score quantile used as threshold. 0.95 targets about 5% false positives.",
    )
    args = parser.parse_args()

    rows = read_scores(args.scores)
    normal_scores = [row["anomaly_score"] for row in rows if row["label"] == args.normal_label]
    if not normal_scores:
        raise SystemExit(f"No rows found for normal label {args.normal_label!r}")

    threshold = quantile(normal_scores, args.quantile)
    summary = build_summary(rows, threshold, args.normal_label)
    artifact = {
        "threshold": threshold,
        "normal_label": args.normal_label,
        "calibration_quantile": args.quantile,
        "interpretation": "score > threshold is anomalous",
        "source_scores": str(args.scores),
        "summary": summary,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    write_eval_csv(args.output.with_suffix(".evaluation.csv"), summary)

    print(f"Wrote threshold artifact to {args.output}")
    print(f"Threshold: {threshold:.6f} calibrated from {args.normal_label} q={args.quantile}")
    for label, stats in summary.items():
        print(
            f"  {label}: n={stats['n']}, mean={stats['mean_score']:.6f}, "
            f"p95={stats['p95_score']:.6f}, flagged={stats['flagged_rate']:.2%}"
        )


def read_scores(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [
            {
                "label": row["label"],
                "y": int(row["y"]),
                "file": row["file"],
                "window_start_s": float(row["window_start_s"]),
                "anomaly_score": float(row["anomaly_score"]),
                "backend": row["backend"],
            }
            for row in csv.DictReader(handle)
        ]


def build_summary(rows: list[dict[str, object]], threshold: float, normal_label: str) -> dict[str, dict[str, object]]:
    labels = sorted({str(row["label"]) for row in rows})
    summary: dict[str, dict[str, object]] = {}
    for label in labels:
        scores = [float(row["anomaly_score"]) for row in rows if row["label"] == label]
        flagged = [score > threshold for score in scores]
        summary[label] = {
            "n": len(scores),
            "mean_score": mean(scores) if scores else 0.0,
            "p50_score": quantile(scores, 0.50) if scores else 0.0,
            "p95_score": quantile(scores, 0.95) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "flagged_count": sum(flagged),
            "flagged_rate": sum(flagged) / len(flagged) if flagged else 0.0,
            "role": "false_positive_estimate" if label == normal_label else "detection_estimate",
        }
    return summary


def write_eval_csv(path: Path, summary: dict[str, dict[str, object]]) -> None:
    fields = ["label", "n", "mean_score", "p50_score", "p95_score", "max_score", "flagged_count", "flagged_rate", "role"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label, stats in summary.items():
            writer.writerow({"label": label, **stats})


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)
    sorted_values = sorted(values)
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


if __name__ == "__main__":
    main()
