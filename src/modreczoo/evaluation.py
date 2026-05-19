import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import mlflow
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, n_classes: int, desc: str = "eval") -> Dict:
    model.eval()
    y_true, y_pred, y_conf, nll_bits = [], [], [], []
    with torch.no_grad():
        for xb, yb in tqdm(loader, desc=desc, unit="batch", leave=False):
            yb_device = yb.to(device)
            logits = model(xb.to(device))
            batch_nll_bits = F.cross_entropy(logits, yb_device, reduction="none") / np.log(2.0)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            nll_bits.extend(batch_nll_bits.cpu().tolist())
            y_conf.extend(conf.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
            y_true.extend(yb.tolist())
    y_true_np = np.asarray(y_true, dtype=int)
    y_pred_np = np.asarray(y_pred, dtype=int)
    y_conf_np = np.asarray(y_conf, dtype=np.float32)
    nll_bits_np = np.asarray(nll_bits, dtype=np.float32)
    labels = np.arange(n_classes)
    return {
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)) if len(y_true_np) else 0.0,
        "confusion": confusion_matrix(y_true_np, y_pred_np, labels=labels),
        "y_true": y_true_np,
        "y_pred": y_pred_np,
        "confidence": y_conf_np,
        "nll_bits": nll_bits_np,
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


def log_f1_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> None:
    label_ids = np.arange(len(labels))
    _, _, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_ids,
        average="macro",
        zero_division=0,
    )
    mlflow.log_metric("macro_f1", float(f1))


def bootstrap_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Dict[str, float]:
    correct = (y_true == y_pred).astype(np.float32)
    n = len(correct)
    if n == 0:
        return {
            "n": 0,
            "n_bootstrap": n_bootstrap,
            "confidence": confidence,
            "accuracy": float("nan"),
            "bootstrap_mean": float("nan"),
            "bootstrap_std": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
        }

    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot_acc = correct[sample_idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "n": n,
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
        "accuracy": float(correct.mean()),
        "bootstrap_mean": float(boot_acc.mean()),
        "bootstrap_std": float(boot_acc.std(ddof=1)),
        "ci_lower": float(np.quantile(boot_acc, alpha)),
        "ci_upper": float(np.quantile(boot_acc, 1.0 - alpha)),
    }


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


def accuracy_by_ebw(
    metadata: pl.DataFrame,
    test_idx: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bin_width: float = 0.1,
) -> pl.DataFrame:
    ebw = metadata[test_idx]["ebw"].to_numpy()
    bins = np.round(np.floor(ebw / bin_width) * bin_width, 10)
    rows = []
    for bin_start in sorted(np.unique(bins)):
        mask = bins == bin_start
        rows.append(
            {
                "ebw_bin": float(bin_start),
                "ebw_bin_end": float(bin_start + bin_width),
                "n": int(np.sum(mask)),
                "accuracy": float(accuracy_score(y_true[mask], y_pred[mask])),
            }
        )
    return pl.DataFrame(rows)


def accuracy_by_osr_snr_levels(
    metadata: pl.DataFrame,
    test_idx: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bin_width: float,
    n_levels: int = 3,
) -> pl.DataFrame:
    snr = metadata[test_idx]["snr_db"].to_numpy()
    osr = metadata[test_idx]["osr"].to_numpy()
    snr_bins = np.floor(snr / bin_width) * bin_width
    unique_bins = np.asarray(sorted(np.unique(snr_bins)), dtype=float)
    if len(unique_bins) > n_levels:
        pick = np.rint(np.linspace(0, len(unique_bins) - 1, n_levels)).astype(int)
        selected_bins = unique_bins[pick]
    else:
        selected_bins = unique_bins

    rows = []
    for bin_start in selected_bins:
        snr_mask = snr_bins == bin_start
        for value in sorted(np.unique(osr[snr_mask])):
            mask = snr_mask & (osr == value)
            rows.append(
                {
                    "snr_bin_db": float(bin_start),
                    "snr_bin_end_db": float(bin_start + bin_width),
                    "osr": float(value),
                    "n": int(np.sum(mask)),
                    "accuracy": float(accuracy_score(y_true[mask], y_pred[mask])),
                }
            )
    return pl.DataFrame(rows)


def information_summary(confusion: np.ndarray, nll_bits: np.ndarray) -> Dict[str, float]:
    counts = np.asarray(confusion, dtype=np.float64)
    n = float(counts.sum())
    mean_nll_bits = float(np.mean(nll_bits)) if len(nll_bits) else float("nan")
    if n <= 0:
        return {
            "n": 0,
            "label_entropy_bits": float("nan"),
            "pred_label_mi_bits": float("nan"),
            "pred_label_mi_fraction": float("nan"),
            "conditional_entropy_true_given_pred_bits": float("nan"),
            "nll_bits": mean_nll_bits,
            "mi_nll_lower_bound_bits": float("nan"),
        }

    joint = counts / n
    true_prior = joint.sum(axis=1)
    pred_prior = joint.sum(axis=0)
    nonzero_true = true_prior > 0
    label_entropy_bits = float(-np.sum(true_prior[nonzero_true] * np.log2(true_prior[nonzero_true])))

    nz = joint > 0
    expected = true_prior[:, None] * pred_prior[None, :]
    pred_label_mi_bits = float(np.sum(joint[nz] * np.log2(joint[nz] / expected[nz])))
    pred_label_mi_fraction = (
        pred_label_mi_bits / label_entropy_bits if label_entropy_bits > 0 else float("nan")
    )
    conditional_entropy = max(0.0, label_entropy_bits - pred_label_mi_bits)
    mi_nll_lower_bound = (
        max(0.0, label_entropy_bits - mean_nll_bits) if np.isfinite(mean_nll_bits) else float("nan")
    )
    return {
        "n": int(n),
        "label_entropy_bits": label_entropy_bits,
        "pred_label_mi_bits": pred_label_mi_bits,
        "pred_label_mi_fraction": pred_label_mi_fraction,
        "conditional_entropy_true_given_pred_bits": conditional_entropy,
        "nll_bits": mean_nll_bits,
        "mi_nll_lower_bound_bits": mi_nll_lower_bound,
    }


def information_by_snr(
    metadata: pl.DataFrame,
    test_idx: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nll_bits: np.ndarray,
    n_classes: int,
    bin_width: float,
) -> pl.DataFrame:
    snr = metadata[test_idx]["snr_db"].to_numpy()
    bins = np.floor(snr / bin_width) * bin_width
    labels = np.arange(n_classes)
    rows = []
    for bin_start in sorted(np.unique(bins)):
        mask = bins == bin_start
        summary = information_summary(confusion_matrix(y_true[mask], y_pred[mask], labels=labels), nll_bits[mask])
        rows.append(
            {
                "snr_bin_db": float(bin_start),
                "snr_bin_end_db": float(bin_start + bin_width),
                **summary,
            }
        )
    return pl.DataFrame(rows)


def _make_constellations() -> dict[str, np.ndarray]:
    c: dict[str, np.ndarray] = {}
    c["2PSK"] = np.array([1.0, -1.0], dtype=complex)
    c["4PSK"] = np.exp(1j * np.pi * np.array([1, 3, 5, 7]) / 4)
    c["8PSK"] = np.exp(1j * 2 * np.pi * np.arange(8) / 8)
    c["pi/4-DQPSK"] = np.exp(1j * np.pi * np.array([0, 2, 4, 6]) / 4)
    for n, label in [(4, "16QAM"), (8, "64QAM"), (16, "256QAM")]:
        levels = np.arange(-(n - 1), n, 2, dtype=float)
        re, im = np.meshgrid(levels, levels)
        pts = (re + 1j * im).ravel()
        pts /= np.sqrt(np.mean(np.abs(pts) ** 2))
        c[label] = pts
    c["MSK"] = c["2PSK"].copy()  # approximated as BPSK (1 bit/symbol, binary phase)
    return c


def union_bound_accuracy_by_snr(
    snr_bins: np.ndarray,
    modulation_names: List[str],
    n_mc: int = 5000,
    seed: int = 0,
) -> pl.DataFrame:
    """Single-sample ML floor on classification accuracy via Monte Carlo.

    Samples n_mc symbols from each constellation, adds AWGN, and runs the ML
    decision (log-likelihood with uniform symbol prior) across all K classes.
    Averaging the per-class error gives the union-bound floor on accuracy.

    This is a FLOOR for N-sample classifiers — neural models that see thousands
    of samples will exceed it. It correctly handles overlapping PSK constellations
    where the erfc/d_min approach is degenerate (d_min = 0 since QPSK ⊂ 8-PSK).
    The prior factor log(1/|C_k|) breaks the symmetry for overlapping constellations.
    """
    from scipy.special import logsumexp as scipy_logsumexp

    rng = np.random.default_rng(seed)
    constellations = _make_constellations()
    names = [n for n in modulation_names if n in constellations]
    K = len(names)
    constels = [constellations[n] for n in names]

    rows = []
    for snr_db in snr_bins:
        snr_lin = 10 ** (float(snr_db) / 10)
        sigma = 1.0 / np.sqrt(2 * snr_lin)  # std per real/imag component

        per_class_err = np.zeros(K)
        for i, si in enumerate(constels):
            symbols = si[rng.integers(0, len(si), size=n_mc)]
            noise = sigma * (rng.standard_normal(n_mc) + 1j * rng.standard_normal(n_mc))
            y = symbols + noise
            lls = np.array([
                -np.log(len(sk)) + scipy_logsumexp(
                    -np.abs(y[:, None] - sk[None, :]) ** 2 / (2 * sigma ** 2), axis=1
                )
                for sk in constels
            ])  # (K, n_mc)
            per_class_err[i] = float(np.mean(np.argmax(lls, axis=0) != i))

        rows.append({
            "snr_bin_db": float(snr_db),
            "union_bound_accuracy": float(np.clip(1.0 - per_class_err.mean(), 0.0, 1.0)),
        })
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
        f"Spectrogram size: {getattr(args, 'spectrogram_size', 'n/a')}",
        f"Spectrogram freq bins: {getattr(args, 'spectrogram_freq_bins', None) or getattr(args, 'spectrogram_size', 'n/a')}",
        f"Spectrogram time bins: {getattr(args, 'spectrogram_time_bins', None) or getattr(args, 'spectrogram_size', 'n/a')}",
        f"Spectrogram nperseg: {getattr(args, 'spectrogram_nperseg', 'n/a')}",
        f"Spectrogram noverlap: {getattr(args, 'spectrogram_noverlap', 'n/a')}",
        f"Spectrogram window: {getattr(args, 'spectrogram_window', 'n/a')}",
        f"Spectrogram window beta: {getattr(args, 'spectrogram_window_beta', 'n/a')}",
        f"Spectrogram base channels: {getattr(args, 'spectrogram_base_channels', 'n/a')}",
        f"Spectrogram kernel size: {getattr(args, 'spectrogram_kernel_size', 'n/a')}",
        "",
    ]
    for result in results:
        lines.extend(
            [
                f"Model: {result['model']}",
                f"Representation: {result['representation']}",
                f"Best val accuracy: {result['best_val_accuracy']:.4f}",
                f"Test accuracy: {result['test_accuracy']:.4f}",
                (
                    "Test accuracy bootstrap "
                    f"{100 * result['accuracy_bootstrap']['confidence']:.0f}% CI: "
                    f"[{result['accuracy_bootstrap']['ci_lower']:.4f}, "
                    f"{result['accuracy_bootstrap']['ci_upper']:.4f}], "
                    f"std={result['accuracy_bootstrap']['bootstrap_std']:.4f}, "
                    f"n_bootstrap={result['accuracy_bootstrap']['n_bootstrap']}"
                ),
                f"ECE: {result['test_ece']:.4f}",
                f"MCE: {result['test_mce']:.4f}",
                "Information diagnostics:",
                result["information_summary"].write_csv(),
                "Accuracy versus SNR:",
                result["accuracy_by_snr"].write_csv(),
                "Accuracy versus OSR:",
                result["accuracy_by_osr"].write_csv(),
                "Information diagnostics versus SNR:",
                result["information_by_snr"].write_csv(),
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
