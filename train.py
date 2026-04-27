import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import mlflow
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import make_model
from plotting import plot_accuracy_by_snr, plot_confusion_matrix
from preprocessing import get_data_loader, ordered_modulation_labels
from simulator import load_dataset


CFO_ESTIMATORS = ("lag_correlation", "phase_slope", "spectral_centroid")
CFO_SWEEP_MODES = ("raw", *CFO_ESTIMATORS)
CHANNEL_FORMATS = ("real_imag", "mag_phase", "mag_inst_freq")
MODEL_NAMES = ("time_cnn", "frequency_cnn", "spectrogram_cnn", "feature_mlp")
MLFLOW_PROFILES = ("local", "runpod")


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def new_profile_stats(limit: int) -> Dict[str, float]:
    return {
        "limit": float(limit),
        "batches": 0.0,
        "samples": 0.0,
        "data_wait_sec": 0.0,
        "host_to_device_sec": 0.0,
        "compute_sec": 0.0,
        "mlflow_metric_log_sec": 0.0,
        "artifact_log_sec": 0.0,
    }


def log_profile_stats(stats: Dict[str, float], prefix: str = "profile") -> None:
    batches = int(stats["batches"])
    if batches <= 0:
        return

    samples = max(stats["samples"], 1.0)
    total_batch_sec = stats["data_wait_sec"] + stats["host_to_device_sec"] + stats["compute_sec"]
    metrics = {
        f"{prefix}_batches": batches,
        f"{prefix}_samples": int(stats["samples"]),
        f"{prefix}_data_wait_sec": stats["data_wait_sec"],
        f"{prefix}_host_to_device_sec": stats["host_to_device_sec"],
        f"{prefix}_compute_sec": stats["compute_sec"],
        f"{prefix}_mlflow_metric_log_sec": stats["mlflow_metric_log_sec"],
        f"{prefix}_artifact_log_sec": stats["artifact_log_sec"],
        f"{prefix}_data_wait_ms_per_batch": 1000.0 * stats["data_wait_sec"] / batches,
        f"{prefix}_host_to_device_ms_per_batch": 1000.0 * stats["host_to_device_sec"] / batches,
        f"{prefix}_compute_ms_per_batch": 1000.0 * stats["compute_sec"] / batches,
        f"{prefix}_samples_per_sec": samples / max(total_batch_sec, np.finfo(float).eps),
    }
    mlflow.log_metrics(metrics)
    print(
        "Profile "
        f"batches={batches} "
        f"data_wait={metrics[f'{prefix}_data_wait_ms_per_batch']:.2f}ms/batch "
        f"h2d={metrics[f'{prefix}_host_to_device_ms_per_batch']:.2f}ms/batch "
        f"compute={metrics[f'{prefix}_compute_ms_per_batch']:.2f}ms/batch "
        f"samples_per_sec={metrics[f'{prefix}_samples_per_sec']:.0f}"
    )


def stratified_split(labels: np.ndarray, train_frac: float, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    temp_frac = 1.0 - train_frac
    if temp_frac <= 0 or val_frac < 0 or val_frac >= temp_frac:
        raise ValueError("Expected train_frac + val_frac to be less than 1.0.")

    train_idx, temp_idx, _, temp_labels = train_test_split(
        indices,
        labels,
        test_size=temp_frac,
        random_state=seed,
        stratify=labels,
    )
    val_relative_frac = val_frac / temp_frac
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1.0 - val_relative_frac,
        random_state=seed,
        stratify=temp_labels,
    )
    return train_idx.astype(np.int64), val_idx.astype(np.int64), test_idx.astype(np.int64)


def dataset_sample_indices(labels: np.ndarray, sample_frac: float, max_examples: int | None, seed: int) -> np.ndarray:
    if not 0 < sample_frac <= 1:
        raise ValueError("Expected sample_frac to be in (0, 1].")
    if max_examples is not None and max_examples <= 0:
        raise ValueError("Expected max_examples to be positive.")

    n_examples = len(labels)
    frac_target = int(np.ceil(n_examples * sample_frac))
    target = min(frac_target, max_examples if max_examples is not None else n_examples)
    if target >= n_examples:
        return np.arange(n_examples, dtype=np.int64)
    return stratified_subset_indices(labels, target, seed)


def stratified_subset_indices(labels: np.ndarray, target: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    unique_labels, inverse, class_counts = np.unique(labels, return_inverse=True, return_counts=True)
    if target < len(unique_labels):
        raise ValueError(f"Expected at least {len(unique_labels)} examples to keep one example per class.")

    expected = class_counts * target / len(labels)
    selected_counts = np.maximum(1, np.floor(expected).astype(int))
    selected_counts = np.minimum(selected_counts, class_counts)
    remainders = expected - np.floor(expected)

    while selected_counts.sum() < target:
        capacity = class_counts - selected_counts
        candidates = np.flatnonzero(capacity > 0)
        chosen = candidates[np.argmax(remainders[candidates])]
        selected_counts[chosen] += 1

    while selected_counts.sum() > target:
        candidates = np.flatnonzero(selected_counts > 1)
        chosen = candidates[np.argmin(remainders[candidates])]
        selected_counts[chosen] -= 1

    selected = []
    for class_id, count in enumerate(selected_counts):
        class_indices = np.flatnonzero(inverse == class_id)
        selected.extend(rng.choice(class_indices, size=int(count), replace=False).tolist())
    selected = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected)
    return selected


def train_one_model(
    args: argparse.Namespace,
    model_name: str,
    signals: np.ndarray,
    metadata: pl.DataFrame,
    labels: np.ndarray,
    label_to_id: Dict[str, int],
    id_to_label: Dict[int, str],
    splits: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Dict:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, representation = make_model(model_name, len(label_to_id), signals.shape[1])
    model.to(device)
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("representation", representation)

    train_idx, val_idx, test_idx = splits
    train_loader = get_data_loader(
        signals,
        metadata,
        train_idx,
        label_to_id,
        model_name=model_name,
        channel_format=args.channel_format,
        remove_cfo=args.remove_cfo,
        cfo_estimator=args.cfo_estimator,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = get_data_loader(
        signals,
        metadata,
        val_idx,
        label_to_id,
        model_name=model_name,
        channel_format=args.channel_format,
        remove_cfo=args.remove_cfo,
        cfo_estimator=args.cfo_estimator,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = get_data_loader(
        signals,
        metadata,
        test_idx,
        label_to_id,
        model_name=model_name,
        channel_format=args.channel_format,
        remove_cfo=args.remove_cfo,
        cfo_estimator=args.cfo_estimator,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = None
    best_val_acc = -1.0
    profile_stats = new_profile_stats(args.profile_batches) if args.profile_batches > 0 else None

    epoch_bar = tqdm(range(1, args.epochs + 1), desc=model_name, unit="epoch")
    for epoch in epoch_bar:
        epoch_start = time.perf_counter()
        train_start = time.perf_counter()
        model.train()
        total_loss = 0.0
        train_bar = tqdm(
            train_loader,
            desc=f"{model_name} train {epoch}/{args.epochs}",
            unit="batch",
            leave=False,
        )
        next_batch_start = time.perf_counter()
        for xb, yb in train_bar:
            batch_ready = time.perf_counter()
            should_profile = profile_stats is not None and profile_stats["batches"] < profile_stats["limit"]
            if should_profile:
                profile_stats["data_wait_sec"] += batch_ready - next_batch_start

            sync_if_cuda(device)
            h2d_start = time.perf_counter()
            xb, yb = xb.to(device), yb.to(device)
            sync_if_cuda(device)
            compute_start = time.perf_counter()

            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()
            sync_if_cuda(device)
            compute_end = time.perf_counter()

            if should_profile:
                profile_stats["batches"] += 1
                profile_stats["samples"] += len(yb)
                profile_stats["host_to_device_sec"] += compute_start - h2d_start
                profile_stats["compute_sec"] += compute_end - compute_start

            total_loss += loss.item() * len(yb)
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
            next_batch_start = time.perf_counter()

        train_duration = time.perf_counter() - train_start
        val_start = time.perf_counter()
        val_metrics = evaluate(model, val_loader, device, len(label_to_id), desc="val")
        val_duration = time.perf_counter() - val_start
        train_loss = total_loss / max(len(train_loader.dataset), 1)
        epoch_duration = time.perf_counter() - epoch_start
        train_samples_per_sec = len(train_loader.dataset) / max(train_duration, np.finfo(float).eps)
        val_samples_per_sec = len(val_loader.dataset) / max(val_duration, np.finfo(float).eps)
        metric_log_start = time.perf_counter()
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_metrics["accuracy"], step=epoch)
        mlflow.log_metric("epoch_duration_sec", epoch_duration, step=epoch)
        mlflow.log_metric("train_duration_sec", train_duration, step=epoch)
        mlflow.log_metric("val_duration_sec", val_duration, step=epoch)
        mlflow.log_metric("train_samples_per_sec", train_samples_per_sec, step=epoch)
        mlflow.log_metric("val_samples_per_sec", val_samples_per_sec, step=epoch)
        if profile_stats is not None:
            profile_stats["mlflow_metric_log_sec"] += time.perf_counter() - metric_log_start
        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_acc=f"{val_metrics['accuracy']:.4f}",
            epoch_sec=f"{epoch_duration:.1f}",
            train_sps=f"{train_samples_per_sec:.0f}",
        )
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test_start = time.perf_counter()
    test_metrics = evaluate(model, test_loader, device, len(label_to_id), desc="test")
    test_duration = time.perf_counter() - test_start
    test_samples_per_sec = len(test_loader.dataset) / max(test_duration, np.finfo(float).eps)
    snr_summary = accuracy_by_snr(metadata, test_idx, test_metrics["y_true"], test_metrics["y_pred"], args.snr_bin_width)
    labels_ordered = [id_to_label[i] for i in range(len(id_to_label))]
    class_summary = per_class_metrics(test_metrics["y_true"], test_metrics["y_pred"], labels_ordered)
    artifact_dir = Path(args.artifact_staging_dir) / mlflow.active_run().info.run_id
    confusion_path = artifact_dir / "confusion_matrix.png"
    snr_plot_path = artifact_dir / "accuracy_vs_snr.png"
    snr_csv_path = artifact_dir / "accuracy_vs_snr.csv"
    class_csv_path = artifact_dir / "per_class_metrics.csv"
    plot_confusion_matrix(test_metrics["confusion"], labels_ordered, confusion_path, model_name)
    plot_accuracy_by_snr(snr_summary, snr_plot_path, model_name)
    snr_summary.write_csv(snr_csv_path)
    class_summary.write_csv(class_csv_path)
    artifact_log_start = time.perf_counter()
    mlflow.log_artifact(str(confusion_path), artifact_path="plots")
    mlflow.log_artifact(str(snr_plot_path), artifact_path="plots")
    mlflow.log_artifact(str(snr_csv_path), artifact_path="tables")
    mlflow.log_artifact(str(class_csv_path), artifact_path="tables")
    if profile_stats is not None:
        profile_stats["artifact_log_sec"] += time.perf_counter() - artifact_log_start

    mlflow.log_metric("test_accuracy", test_metrics["accuracy"])
    mlflow.log_metric("best_val_accuracy", best_val_acc)
    mlflow.log_metric("test_duration_sec", test_duration)
    mlflow.log_metric("test_samples_per_sec", test_samples_per_sec)
    log_prf_metrics(test_metrics["y_true"], test_metrics["y_pred"], labels_ordered)
    if profile_stats is not None:
        log_profile_stats(profile_stats)
    return {
        "model": model_name,
        "representation": representation,
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_metrics["accuracy"],
        "confusion": test_metrics["confusion"],
        "accuracy_by_snr": snr_summary,
        "per_class_metrics": class_summary,
        "confusion_path": str(confusion_path),
    }


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, n_classes: int, desc: str = "eval") -> Dict:
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in tqdm(loader, desc=desc, unit="batch", leave=False):
            logits = model(xb.to(device))
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(pred.tolist())
            y_true.extend(yb.numpy().tolist())
    y_true_np = np.asarray(y_true, dtype=int)
    y_pred_np = np.asarray(y_pred, dtype=int)
    labels = np.arange(n_classes)
    return {
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)) if len(y_true_np) else 0.0,
        "confusion": confusion_matrix(y_true_np, y_pred_np, labels=labels),
        "y_true": y_true_np,
        "y_pred": y_pred_np,
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

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_ids,
        zero_division=0,
    )


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
        f"Dataset: {args.dataset_dir}",
        f"Models: {', '.join(result['model'] for result in results)}",
        f"Labels: {', '.join(labels)}",
        f"Examples available: {getattr(args, 'n_examples_available', 'unknown')}",
        f"Examples used: {getattr(args, 'n_examples_used', 'unknown')}",
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train simple supervised ModRec baselines.")
    parser.add_argument("command", nargs="?", choices=("train", "sweep"), default="train")
    parser.add_argument("--dataset-dir", default="data/awgn_sobol")
    parser.add_argument("--mlflow-profile", choices=MLFLOW_PROFILES, default=os.getenv("MODREC_MLFLOW_PROFILE", "local"))
    parser.add_argument("--mlflow-tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    parser.add_argument("--mlflow-dir", default=None)
    parser.add_argument("--mlflow-db", default=None)
    parser.add_argument("--artifact-staging-dir", default=None)
    parser.add_argument("--experiment", default="modrec-supervised-baselines")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["time_cnn", "frequency_cnn", "feature_mlp"],
        choices=MODEL_NAMES,
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sample-frac", type=float, default=1.0)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--channel-format", choices=CHANNEL_FORMATS, default="real_imag")
    parser.add_argument("--remove-cfo", action="store_true")
    parser.add_argument(
        "--cfo-estimator",
        choices=CFO_SWEEP_MODES,
        default="lag_correlation",
    )
    parser.add_argument("--sweep-channel-formats", nargs="+", choices=CHANNEL_FORMATS, default=list(CHANNEL_FORMATS))
    parser.add_argument("--sweep-cfo-estimators", nargs="+", choices=CFO_SWEEP_MODES, default=list(CFO_SWEEP_MODES))
    parser.add_argument("--sweep-batch-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--snr-bin-width", type=float, default=4.0)
    parser.add_argument("--profile-batches", type=int, default=0)
    parser.add_argument("--system-metrics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--system-metrics-interval", type=int, default=10)
    return parser


def default_mlflow_paths(profile: str) -> Tuple[Path, Path, Path]:
    if profile == "runpod":
        root = Path("/workspace/mlflow")
    else:
        root = Path("mlflow")
    return root / "artifacts", root / "mlflow.db", root / "staging"


def configure_mlflow(args: argparse.Namespace) -> None:
    default_artifact_root, default_db_path, default_staging_dir = default_mlflow_paths(args.mlflow_profile)
    artifact_root = Path(args.mlflow_dir or default_artifact_root).absolute()
    db_path = Path(args.mlflow_db or default_db_path).absolute()
    args.mlflow_dir = str(artifact_root)
    args.mlflow_db = str(db_path)
    args.artifact_staging_dir = str(Path(args.artifact_staging_dir or default_staging_dir).absolute())

    artifact_root.mkdir(parents=True, exist_ok=True)
    Path(args.artifact_staging_dir).mkdir(parents=True, exist_ok=True)

    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    else:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")

    client = MlflowClient()
    experiment = client.get_experiment_by_name(args.experiment)
    if experiment is None:
        if args.mlflow_tracking_uri:
            client.create_experiment(args.experiment)
        else:
            client.create_experiment(args.experiment, artifact_location=artifact_root.as_uri())

    mlflow.set_experiment(args.experiment)
    mlflow.set_experiment_tag("mlflow_profile", args.mlflow_profile)
    mlflow.set_experiment_tag("tracking_uri", mlflow.get_tracking_uri())
    mlflow.set_experiment_tag("artifact_root", str(artifact_root))
    if args.system_metrics:
        mlflow.set_system_metrics_sampling_interval(args.system_metrics_interval)
        mlflow.enable_system_metrics_logging()


def cfo_label(args: argparse.Namespace) -> str:
    return f"cfo_{args.cfo_estimator}" if args.remove_cfo and args.cfo_estimator != "raw" else "cfo_raw"


def run_name_for(args: argparse.Namespace, model_name: str) -> str:
    return f"{model_name}-{args.channel_format}-bs{args.batch_size}-{cfo_label(args)}"


def effective_cfo_estimator(args: argparse.Namespace) -> str:
    return args.cfo_estimator if args.remove_cfo and args.cfo_estimator != "raw" else "raw"


def iter_sweep_args(args: argparse.Namespace) -> List[argparse.Namespace]:
    batch_sizes = args.sweep_batch_sizes if args.sweep_batch_sizes else [args.batch_size]
    cfo_modes = list(args.sweep_cfo_estimators)

    configs = []
    seen = set()
    for model_name in args.models:
        for channel_format in args.sweep_channel_formats:
            for batch_size in batch_sizes:
                for cfo_mode in cfo_modes:
                    remove_cfo = cfo_mode != "raw"
                    key = (model_name, channel_format, batch_size, cfo_mode)
                    if key in seen:
                        continue
                    seen.add(key)
                    cfg = copy.copy(args)
                    cfg.models = [model_name]
                    cfg.channel_format = channel_format
                    cfg.batch_size = batch_size
                    cfg.remove_cfo = remove_cfo
                    cfg.cfo_estimator = cfo_mode
                    configs.append(cfg)
    return configs


def validate_args(args: argparse.Namespace) -> None:
    if args.mlflow_profile not in MLFLOW_PROFILES:
        raise ValueError(f"--mlflow-profile must be one of: {', '.join(MLFLOW_PROFILES)}.")
    if args.profile_batches < 0:
        raise ValueError("--profile-batches must be non-negative.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if args.command == "train" and args.remove_cfo and args.cfo_estimator == "raw":
        raise ValueError("--cfo-estimator raw means no CFO removal; use a real estimator with --remove-cfo.")
    if args.command == "sweep" and not args.sweep_cfo_estimators:
        raise ValueError("Expected at least one CFO sweep mode.")


def log_common_params(args: argparse.Namespace, signals: np.ndarray, labels: List[str], sweep_index: int, sweep_total: int) -> None:
    mlflow.log_params(
        {
            "dataset_dir": args.dataset_dir,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "sample_frac": args.sample_frac,
            "max_examples": args.max_examples if args.max_examples is not None else "none",
            "profile_batches": args.profile_batches,
            "seed": args.seed,
            "n_samples": signals.shape[1],
            "n_examples": args.n_examples_used,
            "n_examples_available": args.n_examples_available,
            "n_train_examples": args.n_train_examples,
            "n_val_examples": args.n_val_examples,
            "n_test_examples": args.n_test_examples,
            "channel_format": args.channel_format,
            "remove_cfo": args.remove_cfo,
            "cfo_estimator": effective_cfo_estimator(args),
            "labels": json.dumps(labels),
            "sweep_index": sweep_index,
            "sweep_total": sweep_total,
        }
    )


def run_config(
    args: argparse.Namespace,
    model_name: str,
    signals: np.ndarray,
    metadata: pl.DataFrame,
    label_values: np.ndarray,
    label_to_id: Dict[str, int],
    id_to_label: Dict[int, str],
    labels: List[str],
    splits: Tuple[np.ndarray, np.ndarray, np.ndarray],
    sweep_index: int,
    sweep_total: int,
) -> None:
    with mlflow.start_run(run_name=run_name_for(args, model_name)) as run:
        log_common_params(args, signals, labels, sweep_index, sweep_total)
        result = train_one_model(args, model_name, signals, metadata, label_values, label_to_id, id_to_label, splits)
        summary_path = Path(args.mlflow_dir) / f"performance_summary_{run.info.run_id}.txt"
        write_summary(summary_path, run.info.run_id, args, labels, [result])
        mlflow.log_artifact(str(summary_path), artifact_path="summaries")
        print(summary_path)


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    signals, metadata = load_dataset(args.dataset_dir)
    observed_labels = metadata["modulation"].unique().to_list()
    labels = ordered_modulation_labels(observed_labels)
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    full_label_values = metadata["modulation"].to_numpy()
    dataset_idx = dataset_sample_indices(full_label_values, args.sample_frac, args.max_examples, args.seed)
    label_values = full_label_values[dataset_idx]
    relative_splits = stratified_split(label_values, args.train_frac, args.val_frac, args.seed)
    splits = tuple(dataset_idx[idx] for idx in relative_splits)
    args.n_examples_available = int(signals.shape[0])
    args.n_examples_used = int(len(dataset_idx))
    args.n_train_examples = int(len(splits[0]))
    args.n_val_examples = int(len(splits[1]))
    args.n_test_examples = int(len(splits[2]))
    print(
        f"Using {args.n_examples_used}/{args.n_examples_available} examples: "
        f"train={args.n_train_examples}, val={args.n_val_examples}, test={args.n_test_examples}."
    )

    configure_mlflow(args)

    configs = iter_sweep_args(args) if args.command == "sweep" else [args]
    print(f"Running {len(configs)} configuration(s).")
    for sweep_index, cfg in enumerate(configs, start=1):
        for model_name in cfg.models:
            print(
                f"[{sweep_index}/{len(configs)}] {run_name_for(cfg, model_name)} "
                f"epochs={cfg.epochs} lr={cfg.lr:g} seed={cfg.seed}"
            )
            run_config(
                cfg,
                model_name,
                signals,
                metadata,
                label_values,
                label_to_id,
                id_to_label,
                labels,
                splits,
                sweep_index,
                len(configs),
            )


if __name__ == "__main__":
    main()
