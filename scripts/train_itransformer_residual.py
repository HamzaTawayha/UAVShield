from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class ModelConfig:
    n_channels: int
    context_len: int
    horizon: int
    hidden_size: int
    n_heads: int
    n_layers: int
    ff_multiplier: int
    dropout: float
    use_instance_norm: bool


class WindowForecastDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, x: np.ndarray, indices: np.ndarray, context_len: int, horizon: int) -> None:
        self.x = x
        self.indices = indices.astype(np.int64)
        self.context_len = context_len
        self.horizon = horizon

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.x[self.indices[item]]
        context = window[:, : self.context_len]
        target = window[:, self.context_len : self.context_len + self.horizon]
        return torch.from_numpy(context), torch.from_numpy(target)


class InvertedTransformerResidual(nn.Module):
    """iTransformer-style forecaster: sensor channels are tokens, not independent batches."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.value_embedding = nn.Linear(config.context_len, config.hidden_size)
        self.channel_embedding = nn.Parameter(
            torch.randn(1, config.n_channels, config.hidden_size) * 0.02
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.n_heads,
            dim_feedforward=config.hidden_size * config.ff_multiplier,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.forecast_head = nn.Linear(config.hidden_size, config.horizon)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if self.config.use_instance_norm:
            mean = context.mean(dim=-1, keepdim=True).detach()
            stdev = context.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-5).detach()
            context = (context - mean) / stdev
        else:
            mean = None
            stdev = None

        tokens = self.value_embedding(context) + self.channel_embedding
        encoded = self.encoder(tokens)
        forecast = self.forecast_head(encoded)

        if mean is not None and stdev is not None:
            forecast = forecast * stdev + mean
        return forecast


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train an iTransformer-style multivariate forecasting-residual detector "
            "on UAV-SEAD window datasets."
        )
    )
    parser.add_argument(
        "--windows",
        type=Path,
        default=Path("data/uav_sead/moment_windows/uav_sead_precise_windows.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/uav_sead/moment_windows/itransformer_residual"),
    )
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--calibration-size", type=float, default=0.25)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument(
        "--threshold-strategy",
        choices=["normal_quantile", "f1_calibration"],
        default="f1_calibration",
    )
    parser.add_argument("--context-len", type=int, default=48)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--ff-multiplier", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--no-instance-norm", action="store_true")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument(
        "--loss",
        choices=["mse", "mae", "huber"],
        default="huber",
        help="Training loss on normal windows.",
    )
    parser.add_argument(
        "--score-error",
        choices=["squared", "absolute"],
        default="squared",
        help="Residual used as anomaly score after forecasting.",
    )
    parser.add_argument(
        "--score-agg",
        choices=["mean", "max", "p95", "p99"],
        default="p95",
        help="How to collapse channel x horizon residuals into one window score.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="auto uses CUDA when PyTorch can see it.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-windows", type=int, default=0, help="Optional quick smoke-test limit.")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    set_seed(args.seed)
    data = np.load(args.windows, allow_pickle=True)
    x = data["x"].astype(np.float32)
    y_multiclass = data["y"].astype(np.int64)
    labels = data["labels"]
    files = data["files"]
    starts = data["window_start_s"]
    feature_names = data["feature_names"] if "feature_names" in data.files else np.arange(x.shape[1])
    y = (y_multiclass != 0).astype(np.int64)

    if args.limit_windows > 0:
        limit = min(args.limit_windows, x.shape[0])
        x = x[:limit]
        y = y[:limit]
        labels = labels[:limit]
        files = files[:limit]
        starts = starts[:limit]

    validate_window_shape(x, args.context_len, args.horizon)
    train_idx, cal_idx, test_idx = grouped_train_cal_test_split(
        x=x,
        y=y,
        files=files,
        test_size=args.test_size,
        calibration_size=args.calibration_size,
    )
    train_normal_idx = train_idx[y[train_idx] == 0]
    cal_normal_idx = cal_idx[y[cal_idx] == 0]
    if train_normal_idx.size == 0:
        raise SystemExit("iTransformer residual training requires normal training windows.")

    device = choose_device(args.device)
    config = ModelConfig(
        n_channels=int(x.shape[1]),
        context_len=args.context_len,
        horizon=args.horizon,
        hidden_size=args.hidden_size,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ff_multiplier=args.ff_multiplier,
        dropout=args.dropout,
        use_instance_norm=not args.no_instance_norm,
    )
    model = InvertedTransformerResidual(config).to(device)
    criterion = build_loss(args.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = make_loader(
        x=x,
        indices=train_normal_idx,
        context_len=args.context_len,
        horizon=args.horizon,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = None
    if cal_normal_idx.size > 0:
        val_loader = make_loader(
            x=x,
            indices=cal_normal_idx,
            context_len=args.context_len,
            horizon=args.horizon,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Training iTransformer residual detector on {len(train_normal_idx)} normal windows "
        f"({x.shape[1]} channels, context={args.context_len}, horizon={args.horizon}) on {device}",
        flush=True,
    )
    train_history, best_state = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
    )
    if best_state is not None:
        model.load_state_dict(best_state)

    cal_scores = score_indices(
        model=model,
        x=x,
        indices=cal_idx,
        context_len=args.context_len,
        horizon=args.horizon,
        batch_size=args.batch_size,
        device=device,
        score_error=args.score_error,
        score_agg=args.score_agg,
    )
    test_scores = score_indices(
        model=model,
        x=x,
        indices=test_idx,
        context_len=args.context_len,
        horizon=args.horizon,
        batch_size=args.batch_size,
        device=device,
        score_error=args.score_error,
        score_agg=args.score_agg,
    )

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
    eval_payload["model"] = "iTransformer-style multivariate forecasting residual"
    eval_payload["threshold"] = threshold
    eval_payload["threshold_strategy"] = args.threshold_strategy
    eval_payload["score_error"] = args.score_error
    eval_payload["score_aggregation"] = args.score_agg
    eval_payload["train_normal_windows_used"] = int(len(train_normal_idx))
    eval_payload["interpretation"] = "forecast_residual_score > threshold is anomalous"
    eval_payload["architecture"] = {
        **asdict(config),
        "note": "Sensor channels are Transformer tokens, matching the core iTransformer inversion.",
    }
    eval_payload["training_history"] = train_history

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "feature_names": [str(item) for item in feature_names.tolist()],
            "args": vars(args),
        },
        args.output_dir / "itransformer_residual.pt",
    )
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(eval_payload, indent=2),
        encoding="utf-8",
    )
    write_predictions(args.output_dir / "predictions.csv", eval_payload["predictions"])
    print_result(threshold, eval_payload, args.output_dir)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_window_shape(x: np.ndarray, context_len: int, horizon: int) -> None:
    if x.ndim != 3:
        raise SystemExit(f"Expected x with shape [windows, channels, time], got {x.shape}")
    if context_len + horizon > x.shape[2]:
        raise SystemExit(
            f"context_len + horizon must fit in window length {x.shape[2]}, "
            f"got {context_len} + {horizon}."
        )
    if not np.isfinite(x).all():
        raise SystemExit("Window tensor contains NaN or infinite values.")


def choose_device(requested: str) -> torch.device:
    cuda_available = torch.cuda.is_available()
    if requested == "cuda" and not cuda_available:
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot see a CUDA device. "
            "Check nvidia-smi, driver/container GPU access, and the installed torch build."
        )
    if requested == "cpu":
        return torch.device("cpu")
    if cuda_available:
        print(f"CUDA available: {torch.cuda.get_device_name(0)}", flush=True)
        return torch.device("cuda")
    print("CUDA is not available to PyTorch; falling back to CPU.", flush=True)
    return torch.device("cpu")


def make_loader(
    x: np.ndarray,
    indices: np.ndarray,
    context_len: int,
    horizon: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    dataset = WindowForecastDataset(x, indices, context_len, horizon)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_loss(name: str) -> nn.Module:
    if name == "mae":
        return nn.L1Loss()
    if name == "mse":
        return nn.MSELoss()
    return nn.SmoothL1Loss(beta=0.5)


def train_model(
    model: nn.Module,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    patience: int,
) -> tuple[list[dict[str, float]], dict[str, torch.Tensor] | None]:
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_metric = math.inf
    stale_epochs = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device) if val_loader is not None else train_loss
        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})

        improved = val_loss < best_metric - 1e-6
        if improved:
            best_metric = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs or stale_epochs >= patience:
            elapsed = time.time() - start_time
            print(
                f"  epoch {epoch:03d}/{epochs} "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} elapsed={elapsed:.1f}s",
                flush=True,
            )

        if patience > 0 and stale_epochs >= patience:
            print(f"  early stopping after {epoch} epochs", flush=True)
            break

    return history, best_state


def run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for context, target in loader:
        context = context.to(device=device, dtype=torch.float32, non_blocking=True)
        target = target.to(device=device, dtype=torch.float32, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(context)
        loss = criterion(prediction, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.detach().item()) * context.shape[0]
        total_count += int(context.shape[0])
    return total_loss / max(total_count, 1)


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    if loader is None:
        return 0.0
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for context, target in loader:
            context = context.to(device=device, dtype=torch.float32, non_blocking=True)
            target = target.to(device=device, dtype=torch.float32, non_blocking=True)
            prediction = model(context)
            loss = criterion(prediction, target)
            total_loss += float(loss.detach().item()) * context.shape[0]
            total_count += int(context.shape[0])
    return total_loss / max(total_count, 1)


def score_indices(
    model: nn.Module,
    x: np.ndarray,
    indices: np.ndarray,
    context_len: int,
    horizon: int,
    batch_size: int,
    device: torch.device,
    score_error: str,
    score_agg: str,
) -> np.ndarray:
    model.eval()
    scores: list[np.ndarray] = []
    total = len(indices)
    with torch.no_grad():
        for start in range(0, total, batch_size):
            batch_idx = indices[start : start + batch_size]
            batch = torch.from_numpy(x[batch_idx]).to(device=device, dtype=torch.float32)
            context = batch[:, :, :context_len]
            target = batch[:, :, context_len : context_len + horizon]
            prediction = model(context)
            if score_error == "absolute":
                residual = torch.abs(prediction - target)
            else:
                residual = torch.square(prediction - target)
            scores.append(aggregate_residual(residual, score_agg).detach().cpu().numpy())
            print(f"  scored {min(start + batch_size, total)}/{total}", flush=True)
    return np.concatenate(scores).astype(np.float64)


def aggregate_residual(residual: torch.Tensor, aggregate: str) -> torch.Tensor:
    flat = residual.reshape(residual.shape[0], -1)
    if aggregate == "max":
        return flat.max(dim=1).values
    if aggregate == "p99":
        return torch.quantile(flat, 0.99, dim=1)
    if aggregate == "p95":
        return torch.quantile(flat, 0.95, dim=1)
    return flat.mean(dim=1)


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


def f1_optimal_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    candidates = threshold_candidates(scores)
    if candidates.size == 0:
        return 0.0

    best_threshold = float(candidates[0])
    best_key = (-1.0, -1.0, -1.0)
    for threshold in candidates:
        y_pred = (scores > threshold).astype(np.int64)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y,
            y_pred,
            average="binary",
            zero_division=0,
        )
        key = (float(f1), float(recall), float(precision))
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def threshold_candidates(scores: np.ndarray) -> np.ndarray:
    unique_scores = np.unique(scores)
    if unique_scores.size == 0:
        return np.asarray([], dtype=np.float64)

    eps = 1e-12
    if unique_scores.size == 1:
        return np.asarray([unique_scores[0] - eps, unique_scores[0] + eps])
    midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0
    return np.concatenate(([unique_scores[0] - eps], midpoints, [unique_scores[-1] + eps]))


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
            "forecast_residual_score": float(score),
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
    fields = ["label", "y_true", "file", "window_start_s", "forecast_residual_score", "flagged"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)


def print_result(threshold: float, eval_payload: dict[str, Any], output_dir: Path) -> None:
    test = eval_payload["test"]
    print(f"\nWrote model to {output_dir / 'itransformer_residual.pt'}")
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
            f"flagged={stats['flagged_rate']:.2%}, mean_score={stats['mean_score']:.6f}"
        )


if __name__ == "__main__":
    main()
