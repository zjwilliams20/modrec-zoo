import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from modreczoo.data import get_data_loader
from modreczoo.evaluation import (
    accuracy_by_snr,
    bootstrap_accuracy,
    calibration_by_snr,
    calibration_stats,
    evaluate,
    information_by_snr,
    information_summary,
    log_f1_metrics,
    per_class_metrics,
    union_bound_accuracy_by_snr,
    write_summary,
)
from modreczoo.models import make_model, representation_for_model, required_channel_format_for


CFO_ESTIMATORS = ("lag_correlation", "phase_slope", "spectral_centroid")
CFO_SWEEP_MODES = ("raw", *CFO_ESTIMATORS)
CHANNEL_FORMATS = (
    "real_imag", "mag", "mag_phase", "mag_inst_freq", "differential_complex",
    "apf", "complex_powers", "multilag", "cyclic_caf", "scf",
)
MODEL_NAMES = (
    "time_cnn", "frequency_cnn", "spectrogram_cnn", "spectrogram_resnet",
    "feature_mlp", "resnet_1d", "complex_cnn_1d", "dilated_cnn_1d",
    "patch_transformer_1d", "multiscale_pyramid_1d", "multi_stream_1d", "apf_net_1d",
    "multilag_net_1d", "cyclic_caf_1d", "scf_resnet",
)
SNR_BIN_WIDTH = 4.0
MLFLOW_DIR = Path("mlflow").absolute()
MLFLOW_DB = MLFLOW_DIR / "mlflow.db"
MLFLOW_ARTIFACTS = MLFLOW_DIR / "artifacts"
MLFLOW_STAGING = MLFLOW_DIR / "staging"
EXPERIMENT_NAME = "modrec"


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def model_parameter_counts(model: torch.nn.Module) -> Dict[str, int]:
    params = list(model.parameters())
    return {
        "model_num_parameters": int(sum(p.numel() for p in params)),
        "model_trainable_parameters": int(sum(p.numel() for p in params if p.requires_grad)),
    }


def model_io_info(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    id_to_label: Dict[int, str],
    representation: str,
    channel_format: str,
    device: torch.device,
) -> Tuple[Dict[str, Any], np.ndarray, Any]:
    sample_x, _ = train_loader.dataset[0]
    input_example = sample_x.unsqueeze(0).cpu().numpy()
    model.eval()
    with torch.no_grad():
        output_example = model(sample_x.unsqueeze(0).to(device)).detach().cpu().numpy()
    return (
        {
            "input_shape": list(input_example.shape),
            "input_dtype": str(input_example.dtype),
            "output_shape": list(output_example.shape),
            "output_dtype": str(output_example.dtype),
            "representation": representation,
            "channel_format": channel_format,
            "labels": [id_to_label[i] for i in range(len(id_to_label))],
        },
        input_example,
        infer_signature(input_example, output_example),
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
    channel_format = effective_channel_format_for(model_name, args.channel_format)
    model, representation = make_model(
        model_name,
        len(label_to_id),
        train_signals.shape[1],
        in_channels=input_channels_for(representation, channel_format),
        spectrogram_base_channels=args.spectrogram_base_channels,
        spectrogram_freq_kernel=args.spectrogram_freq_kernel,
        spectrogram_time_kernel=args.spectrogram_time_kernel,
        transformer_patch_size=args.transformer_patch_size,
        transformer_d_model=args.transformer_d_model,
        transformer_n_heads=args.transformer_n_heads,
        transformer_n_layers=args.transformer_n_layers,
    )
    model.to(device)
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("representation", representation)

    train_idx, val_idx, test_idx = splits
    loader_kwargs = dict(
        model_name=model_name,
        channel_format=channel_format,
        remove_cfo=args.remove_cfo,
        cfo_estimator=args.cfo_estimator,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        spectrogram_freq_bins=args.spectrogram_freq_bins,
        spectrogram_time_bins=args.spectrogram_time_bins,
        spectrogram_nperseg=args.spectrogram_nperseg,
        spectrogram_noverlap=args.spectrogram_noverlap,
        spectrogram_window=args.spectrogram_window,
    )
    train_loader = get_data_loader(train_signals, train_metadata, train_idx, label_to_id, shuffle=True, **loader_kwargs)
    val_loader = get_data_loader(val_signals, val_metadata, val_idx, label_to_id, shuffle=False, **loader_kwargs)
    test_loader = get_data_loader(test_signals, test_metadata, test_idx, label_to_id, shuffle=False, **loader_kwargs)

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
    parameter_counts = model_parameter_counts(model)
    model_info, input_example, signature = model_io_info(
        model, train_loader, id_to_label, representation, channel_format, device
    )
    mlflow.log_params(parameter_counts)
    mlflow.log_dict({**parameter_counts, **model_info}, "model_info.json")
    mlflow.pytorch.log_model(
        model,
        artifact_path="model",
        registered_model_name=model_name,
        input_example=input_example,
        signature=signature,
        metadata={**parameter_counts, **model_info},
        params=parameter_counts,
    )
    test_metrics = evaluate(model, test_loader, device, len(label_to_id), desc="test")
    accuracy_bootstrap = bootstrap_accuracy(test_metrics["y_true"], test_metrics["y_pred"], seed=args.seed)
    snr_summary = accuracy_by_snr(test_metadata, test_idx, test_metrics["y_true"], test_metrics["y_pred"], SNR_BIN_WIDTH)
    info_summary = information_summary(test_metrics["confusion"], test_metrics["nll_bits"])
    info_snr_summary = information_by_snr(
        test_metadata,
        test_idx,
        test_metrics["y_true"],
        test_metrics["y_pred"],
        test_metrics["nll_bits"],
        len(label_to_id),
        SNR_BIN_WIDTH,
    )
    labels_ordered = [id_to_label[i] for i in range(len(id_to_label))]
    class_summary = per_class_metrics(test_metrics["y_true"], test_metrics["y_pred"], labels_ordered)
    artifact_dir = MLFLOW_STAGING / mlflow.active_run().info.run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    confusion_path = artifact_dir / "confusion_matrix.png"
    snr_plot_path = artifact_dir / "accuracy_vs_snr.png"
    snr_csv_path = artifact_dir / "accuracy_vs_snr.csv"
    ub_csv_path = artifact_dir / "union_bound_by_snr.csv"
    class_csv_path = artifact_dir / "per_class_metrics.csv"
    accuracy_bootstrap_path = artifact_dir / "accuracy_bootstrap.csv"
    info_summary_path = artifact_dir / "information_summary.csv"
    info_snr_path = artifact_dir / "information_by_snr.csv"
    info_plot_path = artifact_dir / "information_by_snr.png"
    from modreczoo.plotting import (
        plot_accuracy_by_snr,
        plot_calibration_by_snr,
        plot_confusion_matrix,
        plot_information_by_snr,
        plot_input_examples,
        plot_reliability_diagram,
    )

    input_examples_path = artifact_dir / "input_examples.png"
    plot_input_examples(train_loader, id_to_label, representation, channel_format, input_examples_path)
    mlflow.log_artifact(str(input_examples_path), artifact_path="plots")

    snr_bins = snr_summary["snr_bin_db"].to_numpy()
    ub_summary = union_bound_accuracy_by_snr(snr_bins, labels_ordered)
    mi_fraction = np.clip(
        info_snr_summary["mi_nll_lower_bound_bits"].to_numpy()
        / info_snr_summary["label_entropy_bits"].to_numpy(),
        0.0, 1.0,
    )
    overlays = {
        "Union bound (erfc)": ub_summary["union_bound_accuracy"].to_numpy(),
        "MI fraction (NLL lower bound)": mi_fraction,
    }

    plot_confusion_matrix(test_metrics["confusion"], labels_ordered, confusion_path, model_name)
    plot_accuracy_by_snr(snr_summary, snr_plot_path, model_name, overlays=overlays)
    plot_information_by_snr(info_snr_summary, info_summary, info_plot_path, model_name)
    snr_summary.write_csv(snr_csv_path)
    ub_summary.write_csv(ub_csv_path)
    class_summary.write_csv(class_csv_path)
    pl.DataFrame([accuracy_bootstrap]).write_csv(accuracy_bootstrap_path)
    pl.DataFrame([info_summary]).write_csv(info_summary_path)
    info_snr_summary.write_csv(info_snr_path)
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
    mlflow.log_artifact(str(info_plot_path), artifact_path="plots")
    mlflow.log_artifact(str(reliability_path), artifact_path="plots")
    mlflow.log_artifact(str(calib_snr_plot_path), artifact_path="plots")
    mlflow.log_artifact(str(snr_csv_path), artifact_path="tables")
    mlflow.log_artifact(str(ub_csv_path), artifact_path="tables")
    mlflow.log_artifact(str(class_csv_path), artifact_path="tables")
    mlflow.log_artifact(str(accuracy_bootstrap_path), artifact_path="tables")
    mlflow.log_artifact(str(info_summary_path), artifact_path="tables")
    mlflow.log_artifact(str(info_snr_path), artifact_path="tables")
    mlflow.log_artifact(str(calib_csv_path), artifact_path="tables")
    mlflow.log_artifact(str(calib_snr_csv_path), artifact_path="tables")
    mlflow.log_metric("test_accuracy", test_metrics["accuracy"])
    mlflow.log_metric(
        "test_accuracy_ci_half_width",
        (accuracy_bootstrap["ci_upper"] - accuracy_bootstrap["ci_lower"]) / 2.0,
    )
    mlflow.log_metric("best_val_accuracy", best_val_acc)
    log_f1_metrics(test_metrics["y_true"], test_metrics["y_pred"], labels_ordered)
    return {
        "model": model_name,
        "representation": representation,
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_metrics["accuracy"],
        "accuracy_bootstrap": accuracy_bootstrap,
        "test_ece": ece,
        "test_mce": mce,
        "confusion": test_metrics["confusion"],
        "accuracy_by_snr": snr_summary,
        "information_summary": pl.DataFrame([info_summary]),
        "information_by_snr": info_snr_summary,
        "per_class_metrics": class_summary,
        "calibration_stats": calib_df,
        "confusion_path": str(confusion_path),
    }


def effective_channel_format_for(model_name: str, requested_format: str) -> str:
    """Return the channel format the DataLoader should use for this model.

    Priority: forced external format > user request.
    """
    forced = required_channel_format_for(model_name)
    if forced:
        return forced
    return requested_format


def input_channels_for(representation: str, channel_format: str) -> int:
    if representation == "features":
        return 1
    if channel_format == "apf":
        return 4
    if channel_format == "complex_powers":
        return 6
    if channel_format == "multilag":
        return 6
    if channel_format == "cyclic_caf":
        return 3
    if channel_format in ("mag", "scf"):
        return 1
    return 2


def configure_mlflow(experiment_name: str = EXPERIMENT_NAME) -> None:
    MLFLOW_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    MLFLOW_STAGING.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.absolute()}")
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        client.create_experiment(experiment_name, artifact_location=MLFLOW_ARTIFACTS.as_uri())
    else:
        desired = MLFLOW_ARTIFACTS.absolute().as_uri()
        if getattr(experiment, "artifact_location", None) != desired:
            import sqlite3
            with sqlite3.connect(str(MLFLOW_DB.absolute())) as con:
                con.execute(
                    "update experiments set artifact_location=? where experiment_id=?",
                    (desired, experiment.experiment_id),
                )
    mlflow.set_experiment(experiment_name)


def cfo_label(args: argparse.Namespace) -> str:
    return f"cfo_{args.cfo_estimator}" if args.remove_cfo and args.cfo_estimator != "raw" else "cfo_raw"


# Ordered param descriptors used to build sweep-aware run names.
# Only params present in _sweep_varying_params are included.
_SWEEP_PARAM_FORMATTERS: List[Tuple[str, Any]] = [
    ("channel_format",           lambda a: a.channel_format),
    ("batch_size",               lambda a: f"bs{a.batch_size}"),
    ("cfo_estimator",            lambda a: cfo_label(a)),
    ("spectrogram_freq_bins",    lambda a: f"f{a.spectrogram_freq_bins}"),
    ("spectrogram_time_bins",    lambda a: f"t{a.spectrogram_time_bins}"),
    ("spectrogram_nperseg",      lambda a: f"n{a.spectrogram_nperseg}"),
    ("spectrogram_noverlap",     lambda a: f"o{a.spectrogram_noverlap}"),
    ("spectrogram_window",       lambda a: a.spectrogram_window),
    ("spectrogram_base_channels",lambda a: f"c{a.spectrogram_base_channels}"),
    ("spectrogram_freq_kernel",  lambda a: f"fk{a.spectrogram_freq_kernel}"),
    ("spectrogram_time_kernel",  lambda a: f"tk{a.spectrogram_time_kernel}"),
]

_SWEEP_ATTR_TO_PARAM = {
    "sweep_channel_formats":           "channel_format",
    "sweep_batch_sizes":               "batch_size",
    "sweep_cfo_estimators":            "cfo_estimator",
    "sweep_spectrogram_freq_bins":     "spectrogram_freq_bins",
    "sweep_spectrogram_time_bins":     "spectrogram_time_bins",
    "sweep_spectrogram_base_channels": "spectrogram_base_channels",
    "sweep_spectrogram_freq_kernels":  "spectrogram_freq_kernel",
    "sweep_spectrogram_time_kernels":  "spectrogram_time_kernel",
}


def _swept_params(args: argparse.Namespace) -> frozenset:
    varying = set()
    if len(getattr(args, "models", [])) > 1:
        varying.add("model_name")
    for attr, param in _SWEEP_ATTR_TO_PARAM.items():
        vals = getattr(args, attr, None)
        if vals is not None and len(vals) > 1:
            varying.add(param)
    return frozenset(varying)


def run_name_for(args: argparse.Namespace, model_name: str) -> str:
    if getattr(args, "run_name", None):
        return args.run_name

    varying = getattr(args, "_sweep_varying_params", None)
    if varying:
        parts = [model_name]
        for param, fmt in _SWEEP_PARAM_FORMATTERS:
            if param in varying:
                parts.append(fmt(args))
        return "-".join(parts)

    name = f"{model_name}-{args.channel_format}-bs{args.batch_size}-{cfo_label(args)}"
    if representation_for_model(model_name) == "spectrogram":
        name += f"-f{args.spectrogram_freq_bins}-t{args.spectrogram_time_bins}"
    return name


def effective_cfo_estimator(args: argparse.Namespace) -> str:
    return args.cfo_estimator if args.remove_cfo and args.cfo_estimator != "raw" else "raw"


def iter_sweep_args(args: argparse.Namespace) -> List[argparse.Namespace]:
    batch_sizes = args.sweep_batch_sizes if args.sweep_batch_sizes else [args.batch_size]
    cfo_modes = list(args.sweep_cfo_estimators)
    spectrogram_freq_bins = args.sweep_spectrogram_freq_bins if args.sweep_spectrogram_freq_bins else [args.spectrogram_freq_bins]
    spectrogram_time_bins = args.sweep_spectrogram_time_bins if args.sweep_spectrogram_time_bins else [args.spectrogram_time_bins]
    spectrogram_base_channels = args.sweep_spectrogram_base_channels if args.sweep_spectrogram_base_channels else [args.spectrogram_base_channels]
    spectrogram_freq_kernels = args.sweep_spectrogram_freq_kernels if args.sweep_spectrogram_freq_kernels else [args.spectrogram_freq_kernel]
    spectrogram_time_kernels = args.sweep_spectrogram_time_kernels if args.sweep_spectrogram_time_kernels else [args.spectrogram_time_kernel]

    varying = _swept_params(args)
    configs = []
    seen = set()
    for model_name in args.models:
        channel_formats = (
            [effective_channel_format_for(model_name, args.sweep_channel_formats[0])]
            if required_channel_format_for(model_name)
            else args.sweep_channel_formats
        )
        for channel_format in channel_formats:
            for batch_size in batch_sizes:
                for cfo_mode in cfo_modes:
                    for freq_bins in spectrogram_freq_bins:
                        for time_bins in spectrogram_time_bins:
                            for base_channels in spectrogram_base_channels:
                                for freq_kernel in spectrogram_freq_kernels:
                                    for time_kernel in spectrogram_time_kernels:
                                        key = (model_name, channel_format, batch_size, cfo_mode, freq_bins, time_bins, base_channels, freq_kernel, time_kernel)
                                        if key in seen:
                                            continue
                                        seen.add(key)
                                        cfg = copy.copy(args)
                                        cfg.models = [model_name]
                                        cfg.channel_format = channel_format
                                        cfg.batch_size = batch_size
                                        cfg.remove_cfo = cfo_mode != "raw"
                                        cfg.cfo_estimator = cfo_mode
                                        cfg.spectrogram_freq_bins = freq_bins
                                        cfg.spectrogram_time_bins = time_bins
                                        cfg.spectrogram_base_channels = base_channels
                                        cfg.spectrogram_freq_kernel = freq_kernel
                                        cfg.spectrogram_time_kernel = time_kernel
                                        cfg._sweep_varying_params = varying
                                        configs.append(cfg)
    return configs


def validate_args(args: argparse.Namespace) -> None:
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if args.command == "train" and args.remove_cfo and args.cfo_estimator == "raw":
        raise ValueError("--cfo-estimator raw means no CFO removal; use a real estimator with --remove-cfo.")
    if args.command == "sweep" and not args.sweep_cfo_estimators:
        raise ValueError("Expected at least one CFO sweep mode.")
    if args.spectrogram_noverlap >= args.spectrogram_nperseg:
        raise ValueError("--spectrogram-noverlap must be less than --spectrogram-nperseg.")
    if args.spectrogram_freq_bins < args.spectrogram_nperseg:
        raise ValueError("--spectrogram-freq-bins must be at least --spectrogram-nperseg.")
    if args.spectrogram_time_bins < 1:
        raise ValueError("--spectrogram-time-bins must be positive.")
    if args.spectrogram_freq_kernel % 2 == 0 or args.spectrogram_freq_kernel < 1:
        raise ValueError("--spectrogram-freq-kernel must be a positive odd integer.")
    if args.spectrogram_time_kernel % 2 == 0 or args.spectrogram_time_kernel < 1:
        raise ValueError("--spectrogram-time-kernel must be a positive odd integer.")


def validate_known_labels(metadata: pl.DataFrame, labels: List[str], dataset_dir: str, role: str) -> None:
    unknown = sorted(set(metadata["modulation"].unique().to_list()) - set(labels))
    if unknown:
        raise ValueError(
            f"{role} dataset {dataset_dir} contains labels not present in the training dataset: "
            f"{', '.join(unknown)}."
        )


def log_common_params(
    args: argparse.Namespace,
    model_name: str,
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
            "channel_format": effective_channel_format_for(model_name, args.channel_format),
            "remove_cfo": args.remove_cfo,
            "cfo_estimator": effective_cfo_estimator(args),
            "spectrogram_freq_bins": args.spectrogram_freq_bins,
            "spectrogram_time_bins": args.spectrogram_time_bins,
            "spectrogram_nperseg": args.spectrogram_nperseg,
            "spectrogram_noverlap": args.spectrogram_noverlap,
            "spectrogram_window": args.spectrogram_window,
            "spectrogram_base_channels": args.spectrogram_base_channels,
            "spectrogram_freq_kernel": args.spectrogram_freq_kernel,
            "spectrogram_time_kernel": args.spectrogram_time_kernel,
            "transformer_patch_size": args.transformer_patch_size,
            "transformer_d_model": args.transformer_d_model,
            "transformer_n_heads": args.transformer_n_heads,
            "transformer_n_layers": args.transformer_n_layers,
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
    config_yaml: Optional[str] = None,
) -> None:
    with mlflow.start_run(run_name=run_name_for(args, model_name)) as run:
        log_common_params(args, model_name, train_signals, test_signals, labels, sweep_index, sweep_total)
        if config_yaml is not None:
            mlflow.log_text(config_yaml, "config.yaml")
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
