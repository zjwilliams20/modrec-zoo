import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import mlflow
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, n_classes: int, desc: str = "eval") -> Dict:
    model.eval()
    y_true, y_pred, y_conf = [], [], []
    with torch.no_grad():
        for xb, yb in tqdm(loader, desc=desc, unit="batch", leave=False):
            logits = model(xb.to(device))
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            y_conf.extend(conf.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
            y_true.extend(yb.tolist())
    y_true_np = np.asarray(y_true, dtype=int)
    y_pred_np = np.asarray(y_pred, dtype=int)
    y_conf_np = np.asarray(y_conf, dtype=np.float32)
    labels = np.arange(n_classes)
    return {
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)) if len(y_true_np) else 0.0,
        "confusion": confusion_matrix(y_true_np, y_pred_np, labels=labels),
        "y_true": y_true_np,
        "y_pred": y_pred_np,
        "confidence": y_conf_np,
    }


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> pl.DataFrame:
    label_ids = np.arange(len(labels))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_ids,
        zero_division=0,
    )
    return pl.DataFrame(
        {
            "class_id": label_ids,
            "modulation": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )


def log_prf_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> None:
    label_ids = np.arange(len(labels))
    for average in ("macro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=label_ids,
            average=average,
            zero_division=0,
        )
        mlflow.log_metric(f"{average}_precision", float(precision))
        mlflow.log_metric(f"{average}_recall", float(recall))
        mlflow.log_metric(f"{average}_f1", float(f1))


def accuracy_by_snr(
    metadata: pl.DataFrame,
    test_idx: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bin_width: float,
) -> pl.DataFrame:
    snr = metadata[test_idx]["snr_db"].to_numpy()
    bins = np.floor(snr / bin_width) * bin_width
    rows = []
    for bin_start in sorted(np.unique(bins)):
        mask = bins == bin_start
        rows.append(
            {
                "snr_bin_db": float(bin_start),
                "snr_bin_end_db": float(bin_start + bin_width),
                "n": int(np.sum(mask)),
                "accuracy": float(accuracy_score(y_true[mask], y_pred[mask])),
            }
        )
    return pl.DataFrame(rows)


def calibration_by_snr(
    metadata: pl.DataFrame,
    test_idx: np.ndarray,
    y_true: np.ndarray,
    y_conf: np.ndarray,
    y_pred: np.ndarray,
    bin_width: float,
) -> pl.DataFrame:
    snr = metadata[test_idx]["snr_db"].to_numpy()
    bins = np.floor(snr / bin_width) * bin_width
    rows = []
    for bin_start in sorted(np.unique(bins)):
        mask = bins == bin_start
        n = int(mask.sum())
        acc = float((y_pred[mask] == y_true[mask]).mean()) if n > 0 else float("nan")
        mean_conf = float(y_conf[mask].mean()) if n > 0 else float("nan")
        ece = abs(acc - mean_conf) if n > 0 else float("nan")
        rows.append(
            {
                "snr_bin_db": float(bin_start),
                "snr_bin_end_db": float(bin_start + bin_width),
                "n": n,
                "accuracy": acc,
                "mean_confidence": mean_conf,
                "ece": ece,
            }
        )
    return pl.DataFrame(rows)


def calibration_stats(
    y_true: np.ndarray,
    y_conf: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> Tuple[pl.DataFrame, float, float]:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = float(bin_edges[i]), float(bin_edges[i + 1])
        mask = (y_conf >= lo) & (y_conf <= hi if i == n_bins - 1 else y_conf < hi)
        n = int(mask.sum())
        rows.append(
            {
                "bin_lower": lo,
                "bin_upper": hi,
                "bin_midpoint": (lo + hi) / 2,
                "accuracy": float((y_pred[mask] == y_true[mask]).mean()) if n > 0 else float("nan"),
                "mean_confidence": float(y_conf[mask].mean()) if n > 0 else float("nan"),
                "count": n,
            }
        )
    df = pl.DataFrame(rows)
    occupied = df.filter(pl.col("count") > 0)
    n_total = max(len(y_true), 1)
    weights = occupied["count"].to_numpy() / n_total
    gaps = np.abs(occupied["accuracy"].to_numpy() - occupied["mean_confidence"].to_numpy())
    ece = float(np.dot(weights, gaps))
    mce = float(np.max(gaps)) if len(gaps) > 0 else 0.0
    return df, ece, mce


def write_summary(
    path: Path,
    run_id: str,
    args: argparse.Namespace,
    labels: List[str],
    results: List[Dict],
) -> None:
    lines = [
        "ModRec supervised baseline summary",
        f"MLflow run id: {run_id}",
        f"Training dataset: {args.dataset_dir}",
        f"Validation dataset: {args.val_dataset_dir_effective}",
        f"Test dataset: {args.test_dataset_dir_effective}",
        f"Test source: {args.test_dataset_source}",
        f"Models: {', '.join(result['model'] for result in results)}",
        f"Labels: {', '.join(labels)}",
        f"Train dataset examples available: {getattr(args, 'n_train_dataset_examples_available', 'unknown')}",
        f"Train dataset examples used: {getattr(args, 'n_train_dataset_examples_used', 'unknown')}",
        f"Test dataset examples available: {getattr(args, 'n_test_dataset_examples_available', 'unknown')}",
        f"Test dataset examples used: {getattr(args, 'n_test_dataset_examples_used', 'unknown')}",
        f"Split sizes: train={getattr(args, 'n_train_examples', 'unknown')}, val={getattr(args, 'n_val_examples', 'unknown')}, test={getattr(args, 'n_test_examples', 'unknown')}",
        f"Epochs: {args.epochs}",
        f"Batch size: {args.batch_size}",
        "",
    ]
    for result in results:
        lines.extend(
            [
                f"Model: {result['model']}",
                f"Representation: {result['representation']}",
                f"Best val accuracy: {result['best_val_accuracy']:.4f}",
                f"Test accuracy: {result['test_accuracy']:.4f}",
                f"ECE: {result['test_ece']:.4f}",
                f"MCE: {result['test_mce']:.4f}",
                "Accuracy versus SNR:",
                result["accuracy_by_snr"].write_csv(),
                "Confusion matrix counts:",
                matrix_to_text(result["confusion"], labels),
                "Per-class precision/recall/F1:",
                result["per_class_metrics"].write_csv(),
                f"Confusion plot: {result['confusion_path']}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def matrix_to_text(matrix: np.ndarray, labels: List[str]) -> str:
    header = ["true\\pred", *labels]
    rows = ["\t".join(header)]
    for label, values in zip(labels, matrix):
        rows.append("\t".join([label, *[str(int(v)) for v in values]]))
    return "\n".join(rows)
