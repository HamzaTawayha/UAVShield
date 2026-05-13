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
    seq_len: int
    n_outputs: int
    hidden_size: int
    n_heads: int
    n_layers: int
    ff_multiplier: int
    dropout: float
    instance_norm: bool
    pooling: str


class WindowClassificationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, x: np.ndarray, y: np.ndarray, indices: np.ndarray) -> None:
        self.x = x
        self.y = y
        self.indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = self.indices[item]
        return torch.from_numpy(self.x[index]), torch.tensor(self.y[index])


class InvertedTransformerClassifier(nn.Module):
    """iTransformer-style classifier: each UAV sensor channel is a Transformer token."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.value_embedding = nn.Linear(config.seq_len, config.hidden_size)
        self.channel_embedding = nn.Parameter(
            torch.randn(1, config.n_channels, config.hidden_size) * 0.02
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size) * 0.02)
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
        self.norm = nn.LayerNorm(config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.instance_norm:
            mean = x.mean(dim=-1, keepdim=True).detach()
            stdev = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-5).detach()
            x = (x - mean) / stdev

        tokens = self.value_embedding(x) + self.channel_embedding
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        encoded = self.encoder(torch.cat([cls, tokens], dim=1))
        if self.config.pooling == "mean":
            pooled = encoded[:, 1:, :].mean(dim=1)
        else:
            pooled = encoded[:, 0, :]
        logits = self.classifier(self.norm(pooled))
        if self.config.n_outputs == 1:
            return logits.squeeze(-1)
        return logits


def main() -> None:
    run_start = time.time()
    parser = argparse.ArgumentParser(
        description="Train a supervised iTransformer-style UAV anomaly classifier."
    )
    parser.add_argument(
        "--windows",
        type=Path,
        default=Path("data/uav_sead/moment_windows/uav_sead_precise_windows.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/uav_sead/moment_windows/itransformer_classifier"),
    )
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--calibration-size", type=float, default=0.25)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument(
        "--threshold-strategy",
        choices=["normal_quantile", "f1_calibration"],
        default="f1_calibration",
    )
    parser.add_argument(
        "--task",
        choices=["binary", "multiclass"],
        default="binary",
        help=(
            "binary trains normal-vs-anomaly; multiclass trains each UAV-SEAD class "
            "and scores anomaly as 1 - P(Normal)."
        ),
    )
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--ff-multiplier", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--instance-norm", action="store_true")
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default="balanced",
        help="balanced uses neg/pos as BCE pos_weight on the train split.",
    )
    parser.add_argument(
        "--early-stop-metric",
        choices=["val_loss", "val_f1", "val_auc"],
        default="val_f1",
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
    add_wandb_args(parser)
    args = parser.parse_args()

    set_seed(args.seed)
    checkpoint("loading window dataset", run_start)
    data = np.load(args.windows, allow_pickle=True)
    x = data["x"].astype(np.float32)
    y_multiclass = data["y"].astype(np.int64)
    labels = data["labels"]
    files = data["files"]
    starts = data["window_start_s"]
    feature_names = data["feature_names"] if "feature_names" in data.files else np.arange(x.shape[1])
    class_names = data["class_names"] if "class_names" in data.files else np.asarray(["Normal", "Anomaly"])
    y = (y_multiclass != 0).astype(np.int64)

    if args.limit_windows > 0:
        limit = min(args.limit_windows, x.shape[0])
        x = x[:limit]
        y = y[:limit]
        y_multiclass = y_multiclass[:limit]
        labels = labels[:limit]
        files = files[:limit]
        starts = starts[:limit]

    validate_window_shape(x)
    checkpoint(
        f"dataset windows={x.shape[0]}, channels={x.shape[1]}, seq_len={x.shape[2]}, "
        f"normal={int((y == 0).sum())}, anomaly={int((y == 1).sum())}",
        run_start,
    )

    train_idx, cal_idx, test_idx = grouped_train_cal_test_split(
        x=x,
        y=y,
        files=files,
        test_size=args.test_size,
        calibration_size=args.calibration_size,
    )
    print_split_summary("train", y[train_idx])
    print_split_summary("calibration", y[cal_idx])
    print_split_summary("test", y[test_idx])

    device = choose_device(args.device)
    config = ModelConfig(
        n_channels=int(x.shape[1]),
        seq_len=int(x.shape[2]),
        n_outputs=1 if args.task == "binary" else int(y_multiclass.max()) + 1,
        hidden_size=args.hidden_size,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ff_multiplier=args.ff_multiplier,
        dropout=args.dropout,
        instance_norm=args.instance_norm,
        pooling=args.pooling,
    )
    train_targets = training_targets(args.task, y, y_multiclass)
    hyperparameters = build_hyperparameters(args, config, y[train_idx], y_multiclass[train_idx], class_names)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(hyperparameters, indent=2),
        encoding="utf-8",
    )
    print("\niTransformer classifier hyperparameters:")
    print(json.dumps(hyperparameters, indent=2), flush=True)
    wandb_run = init_wandb(args, hyperparameters, job_type="itransformer-train")
    if wandb_run is not None:
        wandb_run.summary["train/windows"] = int(len(train_idx))
        wandb_run.summary["calibration/windows"] = int(len(cal_idx))
        wandb_run.summary["test/windows"] = int(len(test_idx))

    train_loader = make_loader(
        x,
        train_targets,
        train_idx,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    cal_loader = make_loader(
        x,
        train_targets,
        cal_idx,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = InvertedTransformerClassifier(config).to(device)
    criterion = build_loss(args.task, args.class_weight, y[train_idx], y_multiclass[train_idx], device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    checkpoint(f"training iTransformer classifier on {device}", run_start)
    train_history, best_state = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=cal_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        early_stop_metric=args.early_stop_metric,
        task=args.task,
        wandb_run=wandb_run,
        wandb_log_batches_every=args.wandb_log_batches_every,
    )
    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint("scoring calibration split", run_start)
    cal_scores = score_indices(model, x, cal_idx, args.batch_size, device, args.task)
    checkpoint("scoring test split", run_start)
    test_scores = score_indices(model, x, test_idx, args.batch_size, device, args.task)

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
    eval_payload["model"] = "iTransformer-style supervised multivariate classifier"
    eval_payload["threshold"] = threshold
    eval_payload["threshold_strategy"] = args.threshold_strategy
    eval_payload["interpretation"] = "anomaly_probability > threshold is anomalous"
    eval_payload["architecture"] = {
        **asdict(config),
        "note": "Sensor channels are Transformer tokens, matching the core iTransformer inversion.",
    }
    eval_payload["hyperparameters"] = hyperparameters
    eval_payload["training_history"] = train_history

    checkpoint("saving model and evaluation artifacts", run_start)
    torch.save(
        {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "feature_names": [str(item) for item in feature_names.tolist()],
        "class_names": [str(item) for item in class_names.tolist()],
        "args": vars(args),
        },
        args.output_dir / "itransformer_classifier.pt",
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
            args.output_dir / "itransformer_classifier.pt",
        ],
        artifact_name=f"{wandb_safe_name(args.output_dir.name)}-itransformer",
        log_artifacts=args.wandb_log_artifacts,
    )
    print_result(threshold, eval_payload, args.output_dir)
    finish_wandb(wandb_run)
    checkpoint("run complete", run_start)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint(message: str, start_time: float) -> None:
    elapsed = time.time() - start_time
    print(f"[checkpoint +{elapsed:.1f}s] {message}", flush=True)


def validate_window_shape(x: np.ndarray) -> None:
    if x.ndim != 3:
        raise SystemExit(f"Expected x with shape [windows, channels, time], got {x.shape}")
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


def training_targets(task: str, y_binary: np.ndarray, y_multiclass: np.ndarray) -> np.ndarray:
    if task == "multiclass":
        return y_multiclass.astype(np.int64)
    return y_binary.astype(np.float32)


def build_hyperparameters(
    args: argparse.Namespace,
    config: ModelConfig,
    train_y: np.ndarray,
    train_y_multiclass: np.ndarray,
    class_names: np.ndarray,
) -> dict[str, Any]:
    pos = int((train_y == 1).sum())
    neg = int((train_y == 0).sum())
    pos_weight = float(neg / max(pos, 1)) if args.class_weight == "balanced" else 1.0
    return {
        "windows": str(args.windows),
        "output_dir": str(args.output_dir),
        "task": args.task,
        "architecture": asdict(config),
        "optimization": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "class_weight": args.class_weight,
            "pos_weight": pos_weight,
            "early_stop_metric": args.early_stop_metric,
        },
        "classes": {
            str(index): {
                "name": str(class_names[index]) if index < len(class_names) else str(index),
                "train_windows": int((train_y_multiclass == index).sum()),
            }
            for index in range(config.n_outputs if args.task == "multiclass" else len(class_names))
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


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    dataset = WindowClassificationDataset(x, y, indices)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_loss(
    task: str,
    class_weight: str,
    train_y: np.ndarray,
    train_y_multiclass: np.ndarray,
    device: torch.device,
) -> nn.Module:
    if task == "multiclass":
        if class_weight == "none":
            return nn.CrossEntropyLoss()
        n_classes = int(train_y_multiclass.max()) + 1
        counts = np.bincount(train_y_multiclass, minlength=n_classes).astype(np.float32)
        weights = counts.sum() / np.maximum(counts, 1.0)
        weights = weights / weights.mean()
        print(f"Using CE class weights={np.round(weights, 4).tolist()}", flush=True)
        return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    if class_weight == "none":
        return nn.BCEWithLogitsLoss()
    positives = int((train_y == 1).sum())
    negatives = int((train_y == 0).sum())
    pos_weight = negatives / max(positives, 1)
    print(f"Using BCE pos_weight={pos_weight:.4f} ({negatives} normal / {positives} anomaly)", flush=True)
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))


def train_model(
    model: nn.Module,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epochs: int,
    patience: int,
    early_stop_metric: str,
    task: str,
    wandb_run: Any | None = None,
    wandb_log_batches_every: int = 0,
) -> tuple[list[dict[str, float]], dict[str, torch.Tensor] | None]:
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_metric = -math.inf
    stale_epochs = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch=epoch,
            wandb_run=wandb_run,
            wandb_log_batches_every=wandb_log_batches_every,
        )
        scheduler.step()
        val_loss, val_scores, val_targets = evaluate_validation(model, val_loader, criterion, device, task)
        val_threshold = f1_optimal_threshold(val_scores, val_targets)
        val_pred = (val_scores > val_threshold).astype(np.int64)
        val_accuracy = accuracy_score(val_targets, val_pred)
        precision, recall, val_f1, _ = precision_recall_fscore_support(
            val_targets,
            val_pred,
            average="binary",
            zero_division=0,
        )
        try:
            val_auc = roc_auc_score(val_targets, val_scores)
        except ValueError:
            val_auc = 0.0

        metric_value = {
            "val_loss": -val_loss,
            "val_f1": float(val_f1),
            "val_auc": float(val_auc),
        }[early_stop_metric]
        improved = metric_value > best_metric + 1e-6
        if improved:
            best_metric = metric_value
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_accuracy": float(val_accuracy),
                "val_precision": float(precision),
                "val_recall": float(recall),
                "val_f1": float(val_f1),
                "val_auc": float(val_auc),
                "val_threshold": float(val_threshold),
                "learning_rate": float(scheduler.get_last_lr()[0]),
            }
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": float(train_loss),
                    "val/loss": float(val_loss),
                    "val/accuracy": float(val_accuracy),
                    "val/precision": float(precision),
                    "val/recall": float(recall),
                    "val/f1": float(val_f1),
                    "val/roc_auc": float(val_auc),
                    "val/threshold": float(val_threshold),
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                }
            )

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs or stale_epochs >= patience:
            elapsed = time.time() - start_time
            print(
                f"  epoch {epoch:03d}/{epochs} "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"val_acc={val_accuracy:.4f} val_f1={val_f1:.4f} val_auc={val_auc:.4f} "
                f"elapsed={elapsed:.1f}s",
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
    epoch: int,
    wandb_run: Any | None = None,
    wandb_log_batches_every: int = 0,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for batch_index, (batch_x, batch_y) in enumerate(loader, start=1):
        batch_x = batch_x.to(device=device, dtype=torch.float32, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        if logits.ndim == 1:
            batch_y = batch_y.to(device=device, dtype=torch.float32, non_blocking=True)
        else:
            batch_y = batch_y.to(device=device, dtype=torch.long, non_blocking=True)
        loss = criterion(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        batch_loss = float(loss.detach().item())
        total_loss += batch_loss * batch_x.shape[0]
        total_count += int(batch_x.shape[0])
        if (
            wandb_run is not None
            and wandb_log_batches_every > 0
            and (batch_index == 1 or batch_index % wandb_log_batches_every == 0 or batch_index == len(loader))
        ):
            batch_step = (epoch - 1) * len(loader) + batch_index
            wandb_run.log(
                {
                    "batch_step": batch_step,
                    "epoch": epoch,
                    "train/batch_loss": batch_loss,
                    "train/batch_progress": batch_index / max(len(loader), 1),
                }
            )
    return total_loss / max(total_count, 1)


def evaluate_validation(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
    task: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    scores: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device=device, dtype=torch.float32, non_blocking=True)
            if task == "multiclass":
                batch_y = batch_y.to(device=device, dtype=torch.long, non_blocking=True)
            else:
                batch_y = batch_y.to(device=device, dtype=torch.float32, non_blocking=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += float(loss.detach().item()) * batch_x.shape[0]
            total_count += int(batch_x.shape[0])
            scores.append(logits_to_anomaly_score(logits, task).detach().cpu().numpy())
            targets.append((batch_y != 0).detach().cpu().numpy())
    return (
        total_loss / max(total_count, 1),
        np.concatenate(scores).astype(np.float64),
        np.concatenate(targets).astype(np.int64),
    )


def score_indices(
    model: nn.Module,
    x: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
    task: str,
) -> np.ndarray:
    model.eval()
    scores: list[np.ndarray] = []
    total = len(indices)
    with torch.no_grad():
        for start in range(0, total, batch_size):
            batch_idx = indices[start : start + batch_size]
            batch = torch.from_numpy(x[batch_idx]).to(device=device, dtype=torch.float32)
            scores.append(logits_to_anomaly_score(model(batch), task).detach().cpu().numpy())
            print(f"  scored {min(start + batch_size, total)}/{total}", flush=True)
    return np.concatenate(scores).astype(np.float64)


def logits_to_anomaly_score(logits: torch.Tensor, task: str) -> torch.Tensor:
    if task == "multiclass":
        probabilities = torch.softmax(logits, dim=-1)
        return 1.0 - probabilities[:, 0]
    return torch.sigmoid(logits)


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
        return 0.5

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


def print_split_summary(name: str, y: np.ndarray) -> None:
    print(
        f"{name.capitalize()} split: windows={int(y.size)}, "
        f"normal={int((y == 0).sum())}, anomaly={int((y == 1).sum())}",
        flush=True,
    )


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
            "anomaly_probability": float(score),
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
    fields = ["label", "y_true", "file", "window_start_s", "anomaly_probability", "flagged"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)


def print_result(threshold: float, eval_payload: dict[str, Any], output_dir: Path) -> None:
    test = eval_payload["test"]
    print(f"\nWrote model to {output_dir / 'itransformer_classifier.pt'}")
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
        "--wandb-log-batches-every",
        type=int,
        default=10,
        help="When W&B is enabled, log training batch loss every N batches. Use 0 to disable.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
        help="W&B mode used only when --wandb-project is set.",
    )
    parser.add_argument(
        "--wandb-log-artifacts",
        action="store_true",
        help="Upload run_config/evaluation/predictions/model files to W&B artifacts.",
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
    run.define_metric("epoch")
    run.define_metric("train/loss", step_metric="epoch")
    run.define_metric("val/*", step_metric="epoch")
    run.define_metric("learning_rate", step_metric="epoch")
    run.define_metric("batch_step")
    run.define_metric("train/batch_loss", step_metric="batch_step")
    run.define_metric("train/batch_progress", step_metric="batch_step")
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
        "accuracy": eval_payload["test"]["accuracy"],
        "precision": eval_payload["test"]["precision"],
        "recall": eval_payload["test"]["recall"],
        "f1": eval_payload["test"]["f1"],
        "roc_auc": eval_payload["test"]["roc_auc"],
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
            "charts/final_metrics": wandb.plot.bar(
                final_metrics_table,
                "metric",
                "value",
                title="Final Test Metrics",
            ),
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
