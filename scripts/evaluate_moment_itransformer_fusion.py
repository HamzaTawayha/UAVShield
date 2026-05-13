from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_itransformer_classifier import (
    InvertedTransformerClassifier,
    ModelConfig,
    f1_optimal_threshold,
    score_indices,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed late fusion between MOMENT and iTransformer anomaly scores."
    )
    parser.add_argument(
        "--windows",
        type=Path,
        default=Path("data/uav_sead/moment_windows/uav_sead_precise_windows.npz"),
    )
    parser.add_argument(
        "--moment-dir",
        type=Path,
        default=Path("data/uav_sead/moment_windows/moment_precise_binary_head"),
    )
    parser.add_argument(
        "--itransformer-dir",
        type=Path,
        default=Path("data/uav_sead/moment_windows/itransformer_classifier_multiclass_unweighted"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/uav_sead/moment_windows/fusion_moment_itransformer_equal"),
    )
    parser.add_argument(
        "--alpha-moment",
        type=float,
        default=0.5,
        help="Fixed MOMENT weight. iTransformer receives 1 - alpha.",
    )
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--calibration-size", type=float, default=0.25)
    parser.add_argument(
        "--threshold-strategy",
        choices=["normal_quantile", "f1_calibration"],
        default="f1_calibration",
    )
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    add_wandb_args(parser)
    args = parser.parse_args()

    if not 0.0 <= args.alpha_moment <= 1.0:
        raise SystemExit("--alpha-moment must be between 0 and 1.")
    run_config = {
        "windows": str(args.windows),
        "moment_dir": str(args.moment_dir),
        "itransformer_dir": str(args.itransformer_dir),
        "output_dir": str(args.output_dir),
        "fusion": {
            "method": "fixed_weight_average",
            "alpha_moment": args.alpha_moment,
            "alpha_itransformer": 1.0 - args.alpha_moment,
        },
        "split": {
            "test_size": args.test_size,
            "calibration_size": args.calibration_size,
            "test_random_state": 11,
            "calibration_random_state": 13,
            "grouped_by": "files",
        },
        "threshold": {
            "strategy": args.threshold_strategy,
            "normal_quantile": args.threshold_quantile,
        },
    }
    wandb_run = init_wandb(args, run_config, job_type="fusion-eval")

    data = np.load(args.windows, allow_pickle=True)
    x = data["x"].astype(np.float32)
    y_multiclass = data["y"].astype(np.int64)
    y = (y_multiclass != 0).astype(np.int64)
    labels = data["labels"]
    files = data["files"]
    starts = data["window_start_s"]

    train_idx, cal_idx, test_idx = grouped_train_cal_test_split(
        x=x,
        y=y,
        files=files,
        test_size=args.test_size,
        calibration_size=args.calibration_size,
    )

    print("Scoring MOMENT...", flush=True)
    moment_cal_scores, moment_test_scores = score_moment(
        moment_dir=args.moment_dir,
        cal_idx=cal_idx,
        test_idx=test_idx,
    )
    print("Scoring iTransformer...", flush=True)
    itr_cal_scores, itr_test_scores, itr_metadata = score_itransformer(
        itransformer_dir=args.itransformer_dir,
        x=x,
        y_multiclass=y_multiclass,
        cal_idx=cal_idx,
        test_idx=test_idx,
        batch_size=args.batch_size,
        device=choose_device(args.device),
    )

    alpha_itransformer = 1.0 - args.alpha_moment
    cal_scores = args.alpha_moment * moment_cal_scores + alpha_itransformer * itr_cal_scores
    test_scores = args.alpha_moment * moment_test_scores + alpha_itransformer * itr_test_scores

    if args.threshold_strategy == "f1_calibration":
        threshold = f1_optimal_threshold(cal_scores, y[cal_idx])
    else:
        threshold = normal_quantile_threshold(cal_scores, y[cal_idx], args.threshold_quantile)

    eval_payload = evaluate(
        y_true=y[test_idx],
        scores=test_scores,
        threshold=threshold,
        labels=labels[test_idx],
        files=files[test_idx],
        starts=starts[test_idx],
        normal_threshold_quantile=args.threshold_quantile,
    )
    eval_payload["train"] = split_summary(y[train_idx])
    eval_payload["calibration"] = split_summary(y[cal_idx])
    eval_payload["test"]["windows"] = int(len(test_idx))
    eval_payload["model"] = "Fixed late fusion: MOMENT + iTransformer"
    eval_payload["threshold"] = threshold
    eval_payload["threshold_strategy"] = args.threshold_strategy
    eval_payload["interpretation"] = "fused_anomaly_score > threshold is anomalous"
    eval_payload["fusion"] = {
        "method": "fixed_weight_average",
        "alpha_moment": args.alpha_moment,
        "alpha_itransformer": alpha_itransformer,
        "moment_dir": str(args.moment_dir),
        "itransformer_dir": str(args.itransformer_dir),
        "itransformer_metadata": itr_metadata,
    }
    eval_payload["hyperparameters"] = run_config

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(eval_payload, indent=2),
        encoding="utf-8",
    )
    write_predictions(args.output_dir / "predictions.csv", eval_payload["predictions"])
    log_final_to_wandb(
        wandb_run,
        eval_payload,
        artifact_paths=[
            args.output_dir / "run_config.json",
            args.output_dir / "evaluation.json",
            args.output_dir / "predictions.csv",
        ],
        artifact_name=f"{wandb_safe_name(args.output_dir.name)}-fusion",
        log_artifacts=args.wandb_log_artifacts,
    )
    print_result(threshold, eval_payload, args.output_dir)
    finish_wandb(wandb_run)


def choose_device(requested: str) -> torch.device:
    cuda_available = torch.cuda.is_available()
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested, but PyTorch cannot see a CUDA device.")
    if requested == "cpu":
        return torch.device("cpu")
    if cuda_available:
        print(f"CUDA available: {torch.cuda.get_device_name(0)}", flush=True)
        return torch.device("cuda")
    print("CUDA is not available to PyTorch; falling back to CPU.", flush=True)
    return torch.device("cpu")


def score_moment(moment_dir: Path, cal_idx: np.ndarray, test_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    embeddings_path = moment_dir / "moment_embeddings.npz"
    model_path = moment_dir / "moment_binary_head.joblib"
    if not embeddings_path.exists():
        raise SystemExit(f"Missing MOMENT embeddings: {embeddings_path}")
    if not model_path.exists():
        raise SystemExit(f"Missing MOMENT head: {model_path}")

    embeddings = np.load(embeddings_path, allow_pickle=True)["embeddings"]
    model = joblib.load(model_path)
    try:
        model.named_steps["rf"].verbose = 0
    except Exception:
        pass
    return model.predict_proba(embeddings[cal_idx])[:, 1], model.predict_proba(embeddings[test_idx])[:, 1]


def score_itransformer(
    itransformer_dir: Path,
    x: np.ndarray,
    y_multiclass: np.ndarray,
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    checkpoint_path = itransformer_dir / "itransformer_classifier.pt"
    if not checkpoint_path.exists():
        raise SystemExit(f"Missing iTransformer checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    task = checkpoint.get("args", {}).get("task", "binary")
    config_payload = dict(checkpoint["config"])
    config_payload.setdefault("n_outputs", 1 if task == "binary" else int(y_multiclass.max()) + 1)
    config = ModelConfig(**config_payload)

    model = InvertedTransformerClassifier(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    cal_scores = score_indices(model, x, cal_idx, batch_size, device, task)
    test_scores = score_indices(model, x, test_idx, batch_size, device, task)
    return cal_scores, test_scores, {"task": task, "config": config_payload}


def grouped_train_cal_test_split(
    x: np.ndarray,
    y: np.ndarray,
    files: np.ndarray,
    test_size: float,
    calibration_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=11)
    train_cal_idx, test_idx = next(splitter.split(x, y, groups=files))
    cal_splitter = GroupShuffleSplit(n_splits=1, test_size=calibration_size, random_state=13)
    train_rel_idx, cal_rel_idx = next(
        cal_splitter.split(
            x[train_cal_idx],
            y[train_cal_idx],
            groups=files[train_cal_idx],
        )
    )
    train_idx = train_cal_idx[train_rel_idx]
    cal_idx = train_cal_idx[cal_rel_idx]
    return train_idx, cal_idx, test_idx


def normal_quantile_threshold(scores: np.ndarray, y: np.ndarray, quantile: float) -> float:
    normal_scores = scores[y == 0]
    if normal_scores.size == 0:
        return float(np.quantile(scores, quantile))
    return float(np.quantile(normal_scores, quantile))


def split_summary(y: np.ndarray) -> dict[str, int]:
    return {
        "windows": int(y.size),
        "normal_windows": int((y == 0).sum()),
        "anomaly_windows": int((y == 1).sum()),
    }


def evaluate(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    labels: np.ndarray,
    files: np.ndarray,
    starts: np.ndarray,
    normal_threshold_quantile: float,
) -> dict[str, Any]:
    y_pred = (scores > threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    try:
        roc_auc = roc_auc_score(y_true, scores)
    except ValueError:
        roc_auc = 0.0

    normal_mask = y_true == 0
    anomaly_mask = y_true == 1
    normal_flag_rate = float(np.mean(y_pred[normal_mask])) if np.any(normal_mask) else 0.0
    anomaly_flag_rate = float(np.mean(y_pred[anomaly_mask])) if np.any(anomaly_mask) else 0.0

    predictions = [
        {
            "label": str(label),
            "y_true": int(true),
            "file": str(file),
            "window_start_s": float(start),
            "fused_anomaly_score": float(score),
            "flagged": int(flagged),
        }
        for label, true, file, start, score, flagged in zip(labels, y_true, files, starts, scores, y_pred)
    ]

    by_label: dict[str, dict[str, float]] = {}
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        label_scores = scores[mask]
        label_pred = y_pred[mask]
        by_label[str(label)] = {
            "n": int(mask.sum()),
            "mean_score": float(np.mean(label_scores)) if label_scores.size else 0.0,
            "p95_score": float(np.quantile(label_scores, 0.95)) if label_scores.size else 0.0,
            "flagged_count": int(label_pred.sum()),
            "flagged_rate": float(label_pred.mean()) if label_pred.size else 0.0,
        }

    return {
        "test": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "normal_flag_rate": normal_flag_rate,
            "anomaly_flag_rate": anomaly_flag_rate,
            "normal_threshold_quantile": normal_threshold_quantile,
        },
        "by_label": by_label,
        "predictions": predictions,
    }


def write_predictions(path: Path, predictions: list[dict[str, Any]]) -> None:
    fields = ["label", "y_true", "file", "window_start_s", "fused_anomaly_score", "flagged"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)


def print_result(threshold: float, eval_payload: dict[str, Any], output_dir: Path) -> None:
    test = eval_payload["test"]
    print(f"\nWrote fusion evaluation to {output_dir / 'evaluation.json'}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Test ROC-AUC: {test['roc_auc']:.4f}")
    print(f"Test accuracy: {test['accuracy']:.4f}")
    print(f"Test precision: {test['precision']:.4f}")
    print(f"Test recall: {test['recall']:.4f}")
    print(f"Test F1: {test['f1']:.4f}")
    print(f"Test normal flag rate: {test['normal_flag_rate']:.2%}")
    print(f"Test anomaly flag rate: {test['anomaly_flag_rate']:.2%}")
    for label, stats in eval_payload["by_label"].items():
        print(
            f"  {label}: n={stats['n']}, "
            f"flagged={stats['flagged_rate']:.2%}, mean_score={stats['mean_score']:.4f}"
        )


def add_wandb_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wandb-project", default="", help="Enable W&B logging under this project.")
    parser.add_argument("--wandb-entity", default="", help="Optional W&B entity/team.")
    parser.add_argument("--wandb-run-name", default="", help="Optional W&B run name.")
    parser.add_argument("--wandb-group", default="", help="Optional W&B run group.")
    parser.add_argument("--wandb-tags", nargs="*", default=[], help="Optional W&B tags.")
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
        help="W&B mode used only when --wandb-project is set.",
    )
    parser.add_argument(
        "--wandb-log-artifacts",
        action="store_true",
        help="Upload run_config/evaluation/predictions files to W&B artifacts.",
    )


def init_wandb(args: argparse.Namespace, config: dict[str, Any], job_type: str):
    if not args.wandb_project or args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit(
            "W&B logging requested, but wandb is not installed. "
            "Run: python -m pip install wandb"
        ) from exc

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or None,
        group=args.wandb_group or None,
        tags=args.wandb_tags or None,
        job_type=job_type,
        mode=args.wandb_mode,
        config=config,
    )
    print(f"W&B logging enabled: project={args.wandb_project}, run={run.name}", flush=True)
    return run


def log_final_to_wandb(
    wandb_run,
    eval_payload: dict[str, Any],
    artifact_paths: list[Path],
    artifact_name: str,
    log_artifacts: bool,
) -> None:
    if wandb_run is None:
        return
    import wandb

    metrics = {
        "test/accuracy": eval_payload["test"]["accuracy"],
        "test/precision": eval_payload["test"]["precision"],
        "test/recall": eval_payload["test"]["recall"],
        "test/f1": eval_payload["test"]["f1"],
        "test/roc_auc": eval_payload["test"]["roc_auc"],
        "test/normal_flag_rate": eval_payload["test"]["normal_flag_rate"],
        "test/anomaly_flag_rate": eval_payload["test"]["anomaly_flag_rate"],
        "threshold": eval_payload["threshold"],
        "eval_step": 1,
    }
    for label, stats in eval_payload["by_label"].items():
        prefix = f"by_label/{wandb_safe_name(label)}"
        metrics[f"{prefix}/n"] = stats["n"]
        metrics[f"{prefix}/flagged_rate"] = stats["flagged_rate"]
        metrics[f"{prefix}/mean_score"] = stats["mean_score"]
        metrics[f"{prefix}/p95_score"] = stats["p95_score"]
    wandb_run.log(metrics)
    wandb_run.summary.update(metrics)

    final_metrics_table = wandb.Table(
        columns=["metric", "value"],
        data=[
            ["accuracy", eval_payload["test"]["accuracy"]],
            ["precision", eval_payload["test"]["precision"]],
            ["recall", eval_payload["test"]["recall"]],
            ["f1", eval_payload["test"]["f1"]],
            ["roc_auc", eval_payload["test"]["roc_auc"]],
            ["normal_flag_rate", eval_payload["test"]["normal_flag_rate"]],
            ["anomaly_flag_rate", eval_payload["test"]["anomaly_flag_rate"]],
            ["threshold", eval_payload["threshold"]],
        ],
    )
    by_label_table = wandb.Table(
        columns=["label", "n", "flagged_count", "flagged_rate", "mean_score", "p95_score"],
        data=[
            [
                label,
                stats["n"],
                stats["flagged_count"],
                stats["flagged_rate"],
                stats["mean_score"],
                stats["p95_score"],
            ]
            for label, stats in eval_payload["by_label"].items()
        ],
    )
    predictions = eval_payload.get("predictions", [])
    prediction_columns = list(predictions[0].keys()) if predictions else []
    prediction_table = wandb.Table(
        columns=prediction_columns,
        data=[[row[column] for column in prediction_columns] for row in predictions],
    )
    wandb_run.log(
        {
            "tables/final_metrics": final_metrics_table,
            "tables/by_label": by_label_table,
            "tables/predictions": prediction_table,
            "charts/flagged_rate_by_label": wandb.plot.bar(
                by_label_table,
                "label",
                "flagged_rate",
                title="Flagged Rate by Label",
            ),
            "charts/mean_score_by_label": wandb.plot.bar(
                by_label_table,
                "label",
                "mean_score",
                title="Mean Anomaly Score by Label",
            ),
        }
    )

    if log_artifacts:
        artifact = wandb.Artifact(artifact_name, type="model-evaluation")
        for path in artifact_paths:
            if path.exists():
                artifact.add_file(str(path))
        wandb_run.log_artifact(artifact)


def finish_wandb(wandb_run) -> None:
    if wandb_run is not None:
        wandb_run.finish()


def wandb_safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in value)


if __name__ == "__main__":
    main()
