import argparse
import copy
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import mlflow
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from modreczoo.data import get_data_loader
from modreczoo.evaluation import (
    accuracy_by_snr,
    calibration_by_snr,
    calibration_stats,
    evaluate,
    log_prf_metrics,
    per_class_metrics,
    write_summary,
)
from modreczoo.models import make_model, representation_for_model


CFO_ESTIMATORS = ("lag_correlation", "phase_slope", "spectral_centroid")
CFO_SWEEP_MODES = ("raw", *CFO_ESTIMATORS)
CHANNEL_FORMATS = ("real_imag", "mag", "mag_phase", "mag_inst_freq")
MODEL_NAMES = ("time_cnn", "frequency_cnn", "spectrogram_cnn", "feature_mlp", "resnet_1d", "complex_cnn_1d", "dilated_cnn_1d")
SNR_BIN_WIDTH = 4.0
MLFLOW_DIR = Path("mlflow").absolute()
MLFLOW_DB = MLFLOW_DIR / "mlflow.db"
MLFLOW_ARTIFACTS = MLFLOW_DIR / "artifacts"
MLFLOW_STAGING = MLFLOW_DIR / "staging"
EXPERIMENT_NAME = "modrec"


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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


def stratified_train_val_split(labels: np.ndarray, train_frac: float, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    total_frac = train_frac + val_frac
    if train_frac <= 0 or val_frac <= 0 or total_frac <= 0:
        raise ValueError("Expected positive train and validation fractions.")

    indices = np.arange(len(labels))
    val_relative_frac = val_frac / total_frac
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_relative_frac,
        random_state=seed,
        stratify=labels,
    )
    return train_idx.astype(np.int64), val_idx.astype(np.int64)


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
    train_signals: np.ndarray,
    train_metadata: pl.DataFrame,
    val_signals: np.ndarray,
    val_metadata: pl.DataFrame,
    test_signals: np.ndarray,
    test_metadata: pl.DataFrame,
    label_to_id: Dict[str, int],
    id_to_label: Dict[int, str],
    splits: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Dict:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    representation = representation_for_model(model_name)
    model, representation = make_model(
        model_name,
        len(label_to_id),
        train_signals.shape[1],
        in_channels=input_channels_for(representation, args.channel_format),
    )
    model.to(device)
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("representation", representation)

    train_idx, val_idx, test_idx = splits
    train_loader = get_data_loader(
        train_signals,
        train_metadata,
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
        val_signals,
        val_metadata,
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
        test_signals,
        test_metadata,
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
        for xb, yb in train_bar:
            sync_if_cuda(device)
            xb, yb = xb.to(device), yb.to(device)
            sync_if_cuda(device)

            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()
            sync_if_cuda(device)

            total_loss += loss.item() * len(yb)
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_duration = time.perf_counter() - train_start
        val_metrics = evaluate(model, val_loader, device, len(label_to_id), desc="val")
        train_loss = total_loss / max(len(train_loader.dataset), 1)
        epoch_duration = time.perf_counter() - epoch_start
        train_samples_per_sec = len(train_loader.dataset) / max(train_duration, np.finfo(float).eps)
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_metrics["accuracy"], step=epoch)
        mlflow.log_metric("epoch_duration_sec", epoch_duration, step=epoch)
        mlflow.log_metric("train_samples_per_sec", train_samples_per_sec, step=epoch)
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
    test_metrics = evaluate(model, test_loader, device, len(label_to_id), desc="test")
    snr_summary = accuracy_by_snr(test_metadata, test_idx, test_metrics["y_true"], test_metrics["y_pred"], SNR_BIN_WIDTH)
    labels_ordered = [id_to_label[i] for i in range(len(id_to_label))]
    class_summary = per_class_metrics(test_metrics["y_true"], test_metrics["y_pred"], labels_ordered)
    artifact_dir = MLFLOW_STAGING / mlflow.active_run().info.run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    confusion_path = artifact_dir / "confusion_matrix.png"
    snr_plot_path = artifact_dir / "accuracy_vs_snr.png"
    snr_csv_path = artifact_dir / "accuracy_vs_snr.csv"
    class_csv_path = artifact_dir / "per_class_metrics.csv"
    from modreczoo.plotting import (
        plot_accuracy_by_snr,
        plot_calibration_by_snr,
        plot_confusion_matrix,
        plot_reliability_diagram,
    )

    plot_confusion_matrix(test_metrics["confusion"], labels_ordered, confusion_path, model_name)
    plot_accuracy_by_snr(snr_summary, snr_plot_path, model_name)
    snr_summary.write_csv(snr_csv_path)
    class_summary.write_csv(class_csv_path)
    calib_df, ece, mce = calibration_stats(
        test_metrics["y_true"], test_metrics["confidence"], test_metrics["y_pred"]
    )
    calib_snr_df = calibration_by_snr(
        test_metadata, test_idx, test_metrics["y_true"], test_metrics["confidence"], test_metrics["y_pred"], SNR_BIN_WIDTH
    )
    reliability_path = artifact_dir / "reliability_diagram.png"
    calib_csv_path = artifact_dir / "calibration_stats.csv"
    calib_snr_plot_path = artifact_dir / "calibration_by_snr.png"
    calib_snr_csv_path = artifact_dir / "calibration_by_snr.csv"
    plot_reliability_diagram(calib_df, ece, mce, reliability_path, model_name)
    plot_calibration_by_snr(calib_snr_df, calib_snr_plot_path, model_name)
    calib_df.write_csv(calib_csv_path)
    calib_snr_df.write_csv(calib_snr_csv_path)
    mlflow.log_artifact(str(confusion_path), artifact_path="plots")
    mlflow.log_artifact(str(snr_plot_path), artifact_path="plots")
    mlflow.log_artifact(str(reliability_path), artifact_path="plots")
    mlflow.log_artifact(str(calib_snr_plot_path), artifact_path="plots")
    mlflow.log_artifact(str(snr_csv_path), artifact_path="tables")
    mlflow.log_artifact(str(class_csv_path), artifact_path="tables")
    mlflow.log_artifact(str(calib_csv_path), artifact_path="tables")
    mlflow.log_artifact(str(calib_snr_csv_path), artifact_path="tables")
    mlflow.log_metric("test_accuracy", test_metrics["accuracy"])
    mlflow.log_metric("best_val_accuracy", best_val_acc)
    mlflow.log_metric("test_ece", ece)
    mlflow.log_metric("test_mce", mce)
    log_prf_metrics(test_metrics["y_true"], test_metrics["y_pred"], labels_ordered)
    return {
        "model": model_name,
        "representation": representation,
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_metrics["accuracy"],
        "test_ece": ece,
        "test_mce": mce,
        "confusion": test_metrics["confusion"],
        "accuracy_by_snr": snr_summary,
        "per_class_metrics": class_summary,
        "calibration_stats": calib_df,
        "confusion_path": str(confusion_path),
    }


def input_channels_for(representation: str, channel_format: str) -> int:
    if representation == "features":
        return 1
    if channel_format == "mag":
        if representation == "spectrogram":
            raise ValueError("--channel-format mag is not supported for spectrogram models.")
        return 1
    return 2


def configure_mlflow() -> None:
    MLFLOW_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    MLFLOW_STAGING.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.absolute()}")
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        client.create_experiment(EXPERIMENT_NAME, artifact_location=MLFLOW_ARTIFACTS.as_uri())
    else:
        desired = MLFLOW_ARTIFACTS.absolute().as_uri()
        if getattr(experiment, "artifact_location", None) != desired:
            import sqlite3
            with sqlite3.connect(str(MLFLOW_DB.absolute())) as con:
                con.execute(
                    "update experiments set artifact_location=? where experiment_id=?",
                    (desired, experiment.experiment_id),
                )
    mlflow.set_experiment(EXPERIMENT_NAME)


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
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if args.command == "train" and args.remove_cfo and args.cfo_estimator == "raw":
        raise ValueError("--cfo-estimator raw means no CFO removal; use a real estimator with --remove-cfo.")
    if args.command == "sweep" and not args.sweep_cfo_estimators:
        raise ValueError("Expected at least one CFO sweep mode.")


def validate_known_labels(metadata: pl.DataFrame, labels: List[str], dataset_dir: str, role: str) -> None:
    unknown = sorted(set(metadata["modulation"].unique().to_list()) - set(labels))
    if unknown:
        raise ValueError(
            f"{role} dataset {dataset_dir} contains labels not present in the training dataset: "
            f"{', '.join(unknown)}."
        )


def log_common_params(
    args: argparse.Namespace,
    train_signals: np.ndarray,
    test_signals: np.ndarray,
    labels: List[str],
    sweep_index: int,
    sweep_total: int,
) -> None:
    mlflow.log_params(
        {
            "dataset_dir": args.dataset_dir,
            "train_dataset_dir": args.dataset_dir,
            "val_dataset_dir": args.val_dataset_dir_effective,
            "test_dataset_dir": args.test_dataset_dir_effective,
            "test_dataset_source": args.test_dataset_source,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "sample_frac": args.sample_frac,
            "max_examples": args.max_examples if args.max_examples is not None else "none",
            "seed": args.seed,
            "n_samples": train_signals.shape[1],
            "train_n_samples": train_signals.shape[1],
            "test_n_samples": test_signals.shape[1],
            "n_examples": args.n_train_dataset_examples_used,
            "n_examples_available": args.n_train_dataset_examples_available,
            "n_train_dataset_examples_available": args.n_train_dataset_examples_available,
            "n_train_dataset_examples_used": args.n_train_dataset_examples_used,
            "n_test_dataset_examples_available": args.n_test_dataset_examples_available,
            "n_test_dataset_examples_used": args.n_test_dataset_examples_used,
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
    train_signals: np.ndarray,
    train_metadata: pl.DataFrame,
    val_signals: np.ndarray,
    val_metadata: pl.DataFrame,
    test_signals: np.ndarray,
    test_metadata: pl.DataFrame,
    label_to_id: Dict[str, int],
    id_to_label: Dict[int, str],
    labels: List[str],
    splits: Tuple[np.ndarray, np.ndarray, np.ndarray],
    sweep_index: int,
    sweep_total: int,
) -> None:
    with mlflow.start_run(run_name=run_name_for(args, model_name)) as run:
        log_common_params(args, train_signals, test_signals, labels, sweep_index, sweep_total)
        result = train_one_model(
            args,
            model_name,
            train_signals,
            train_metadata,
            val_signals,
            val_metadata,
            test_signals,
            test_metadata,
            label_to_id,
            id_to_label,
            splits,
        )
        summary_path = MLFLOW_ARTIFACTS / f"performance_summary_{run.info.run_id}.txt"
        write_summary(summary_path, run.info.run_id, args, labels, [result])
        mlflow.log_artifact(str(summary_path), artifact_path="summaries")
        print(summary_path)
