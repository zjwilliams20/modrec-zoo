import argparse
import copy
import gc
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

from modreczoo.auxiliary import (
    auxiliary_class_counts,
    auxiliary_target_info,
    build_metadata_target_encoders,
    unpack_batch,
)
from modreczoo.data import get_data_loader, load_dataset
from modreczoo.evaluation import (
    accuracy_by_ebw,
    accuracy_by_osr_snr_levels,
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
from modreczoo.models import MODEL_NAMES, make_model, representation_for_model, required_channel_format_for
from modreczoo.models.wrappers import ModelWithPreprocessor, MultiTaskModel, forward_all
from modreczoo.oracle import load_oracle_cache, oracle_cache_status
from modreczoo.preprocessing import PREPROCESSOR_NAMES, make_preprocessor
from modreczoo.reporting import build_prediction_table, error_slice_table, write_performance_explorer


CFO_ESTIMATORS = ("lag_correlation", "phase_slope", "spectral_centroid")
CFO_SWEEP_MODES = ("raw", *CFO_ESTIMATORS)
CHANNEL_FORMATS = (
    "real_imag", "mag_phase", "differential_complex",
    "apf", "complex_powers", "unit_phasor_powers", "multilag", "cyclic_caf", "scf",
)
SNR_BIN_WIDTH = 4.0
MLFLOW_DIR = Path("mlflow").absolute()
MLFLOW_DB = MLFLOW_DIR / "mlflow.db"
MLFLOW_ARTIFACTS = MLFLOW_DIR / "artifacts"
MLFLOW_STAGING = MLFLOW_DIR / "staging"
EXPERIMENT_NAME = "modrec"
SYSTEM_METRICS_SAMPLING_INTERVAL_SEC = 5
SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING = 1


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
    sample = train_loader.dataset[0]
    sample_x = sample[0]
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


def safe_metric_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name)


def log_auxiliary_accuracy_metrics(prefix: str, metrics: Dict, step: int | None = None) -> None:
    for name, values in metrics.get("auxiliary", {}).items():
        mlflow.log_metric(f"{prefix}_{safe_metric_name(name)}_accuracy", values["accuracy"], step=step)


def multitask_loss(
    model: torch.nn.Module,
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    auxiliary: dict[str, torch.Tensor] | None,
    aux_loss_weight: float,
    aux_loss_mode: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    losses = {"modulation": F.cross_entropy(outputs["modulation"], labels)}
    if auxiliary:
        for name, target in auxiliary.items():
            if name in outputs:
                losses[name] = F.cross_entropy(outputs[name], target)

    if aux_loss_mode == "uncertainty" and hasattr(model, "loss_log_vars"):
        total = losses["modulation"].new_tensor(0.0)
        for name, loss in losses.items():
            log_var = model.loss_log_vars[name]  # type: ignore[attr-defined, index]
            total = total + torch.exp(-log_var) * loss + 0.5 * log_var
        return total, losses

    auxiliary_losses = [loss for name, loss in losses.items() if name != "modulation"]
    total = losses["modulation"]
    if auxiliary_losses:
        total = total + aux_loss_weight * sum(auxiliary_losses) / len(auxiliary_losses)
    return total, losses


def validate_preprocessor_args(
    name: str,
    representation: str,
    channel_format: str,
    in_channels: int,
) -> None:
    if name == "none":
        return
    if representation not in {"time", "frequency"}:
        raise ValueError(
            f"--preprocessor {name} expects 1D time/frequency tensors, "
            f"but model representation is {representation!r}."
        )
    if name == "normalize" and channel_format not in {"real_imag", "differential_complex"}:
        raise ValueError(
            "--preprocessor normalize computes power by summing channels and "
            "requires a real/imag channel format (real_imag or differential_complex); "
            f"got channel_format={channel_format!r}."
        )
    if name == "radio_transform" and (representation != "time" or channel_format != "real_imag" or in_channels != 2):
        raise ValueError("--preprocessor radio_transform requires time-domain real_imag input with two channels.")


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


def pad_or_trim_signals(signals: np.ndarray, n_samples: int) -> np.ndarray:
    if signals.shape[1] == n_samples:
        return signals
    if signals.shape[1] > n_samples:
        return signals[:, :n_samples]
    pad_width = ((0, 0), (0, n_samples - signals.shape[1]))
    return np.pad(signals, pad_width, mode="constant")


def evaluate_and_log_test_set(
    name: str,
    model: torch.nn.Module,
    signals: np.ndarray,
    metadata: pl.DataFrame,
    idx: np.ndarray,
    label_to_id: Dict[str, int],
    id_to_label: Dict[int, str],
    loader_kwargs: Dict,
    artifact_dir: Path,
    dataset_dir: str,
    seed: int,
    device: torch.device,
    n_samples: Optional[int] = None,
    auxiliary_tasks: dict[str, int] | None = None,
) -> Dict:
    from modreczoo.plotting import (
        plot_accuracy_by_ebw,
        plot_accuracy_by_osr,
        plot_accuracy_by_snr,
        plot_calibration_by_snr,
        plot_confusion_matrix,
        plot_dataset_metadata,
        plot_information_by_snr,
        plot_reliability_diagram,
    )
    labels_ordered = [id_to_label[i] for i in range(len(id_to_label))]
    loader = get_data_loader(signals, metadata, idx, label_to_id, shuffle=False, n_samples=n_samples, **loader_kwargs)
    test_metrics = evaluate(model, loader, device, len(label_to_id), desc=name, auxiliary_tasks=auxiliary_tasks)
    del loader
    gc.collect()
    accuracy_bootstrap = bootstrap_accuracy(test_metrics["y_true"], test_metrics["y_pred"], seed=seed)
    snr_summary = accuracy_by_snr(metadata, idx, test_metrics["y_true"], test_metrics["y_pred"], SNR_BIN_WIDTH)
    osr_summary = accuracy_by_osr_snr_levels(metadata, idx, test_metrics["y_true"], test_metrics["y_pred"], SNR_BIN_WIDTH)
    ebw_summary = accuracy_by_ebw(metadata, idx, test_metrics["y_true"], test_metrics["y_pred"])
    info_summary_dict = information_summary(test_metrics["confusion"], test_metrics["nll_bits"])
    info_snr_summary = information_by_snr(
        metadata, idx, test_metrics["y_true"], test_metrics["y_pred"],
        test_metrics["nll_bits"], len(label_to_id), SNR_BIN_WIDTH,
    )
    oracle_metrics = load_oracle_cache(dataset_dir, metadata, idx, labels_ordered)
    if oracle_metrics is None:
        status = oracle_cache_status(dataset_dir, metadata, labels_ordered)
        print(f"Skipping oracle metrics for {dataset_dir}: {status}.")

    oracle_snr_summary = None
    oracle_osr_summary = None
    oracle_info_snr_summary = None
    if oracle_metrics is not None:
        oracle_snr_summary = accuracy_by_snr(
            metadata, idx, oracle_metrics["y_true"], oracle_metrics["y_pred"], SNR_BIN_WIDTH,
        ).rename({"accuracy": "oracle_accuracy"})
        oracle_osr_summary = accuracy_by_osr_snr_levels(
            metadata, idx, oracle_metrics["y_true"], oracle_metrics["y_pred"], SNR_BIN_WIDTH,
        ).rename({"accuracy": "oracle_accuracy"})
        oracle_info_snr_summary = information_by_snr(
            metadata, idx, oracle_metrics["y_true"], oracle_metrics["y_pred"],
            oracle_metrics["nll_bits"], len(label_to_id), SNR_BIN_WIDTH,
        )
    class_summary = per_class_metrics(test_metrics["y_true"], test_metrics["y_pred"], labels_ordered)

    name_dir = artifact_dir / name
    name_dir.mkdir(parents=True, exist_ok=True)
    metadata_plot_path = name_dir / "dataset_metadata.png"
    confusion_path = name_dir / "confusion_matrix.png"
    snr_plot_path = name_dir / "accuracy_vs_snr.png"
    snr_csv_path = name_dir / "accuracy_vs_snr.csv"
    osr_plot_path = name_dir / "accuracy_vs_osr.png"
    osr_csv_path = name_dir / "accuracy_vs_osr.csv"
    ebw_plot_path = name_dir / "accuracy_vs_ebw.png"
    ebw_csv_path = name_dir / "accuracy_vs_ebw.csv"
    ub_csv_path = name_dir / "union_bound_by_snr.csv"
    oracle_snr_csv_path = name_dir / "oracle_accuracy_by_snr.csv"
    oracle_osr_csv_path = name_dir / "oracle_accuracy_by_osr.csv"
    oracle_info_summary_path = name_dir / "oracle_information_summary.csv"
    oracle_info_snr_path = name_dir / "oracle_information_by_snr.csv"
    class_csv_path = name_dir / "per_class_metrics.csv"
    accuracy_bootstrap_path = name_dir / "accuracy_bootstrap.csv"
    info_summary_path = name_dir / "information_summary.csv"
    info_snr_path = name_dir / "information_by_snr.csv"
    info_plot_path = name_dir / "information_by_snr.png"
    reliability_path = name_dir / "reliability_diagram.png"
    calib_csv_path = name_dir / "calibration_stats.csv"
    calib_snr_plot_path = name_dir / "calibration_by_snr.png"
    calib_snr_csv_path = name_dir / "calibration_by_snr.csv"
    predictions_path = name_dir / "predictions.parquet"
    error_slices_path = name_dir / "error_slices.csv"
    explorer_path = name_dir / "performance_explorer.html"

    snr_bins = snr_summary["snr_bin_db"].to_numpy()
    ub_summary = union_bound_accuracy_by_snr(snr_bins, labels_ordered)
    mi_fraction = np.clip(
        info_snr_summary["mi_nll_lower_bound_bits"].to_numpy()
        / info_snr_summary["label_entropy_bits"].to_numpy(),
        0.0, 1.0,
    )
    overlays: Dict = {
        "Union bound (erfc)": ub_summary["union_bound_accuracy"].to_numpy(),
        "MI fraction (NLL lower bound)": mi_fraction,
    }
    osr_overlays: Dict = {}
    fraction_overlays = None
    if oracle_metrics is not None:
        oracle_accuracy = (
            snr_summary.select("snr_bin_db")
            .join(oracle_snr_summary.select(["snr_bin_db", "oracle_accuracy"]), on="snr_bin_db", how="left")
            ["oracle_accuracy"]
            .to_numpy()
        )
        oracle_mi_fraction = (
            info_snr_summary.select("snr_bin_db")
            .join(
                oracle_info_snr_summary.select(["snr_bin_db", "pred_label_mi_fraction"]),
                on="snr_bin_db",
                how="left",
            )
            ["pred_label_mi_fraction"]
            .to_numpy()
        )
        oracle_osr_overlay = (
            oracle_osr_summary
            .rename({"oracle_accuracy": "accuracy"})
            .select(["snr_bin_db", "snr_bin_end_db", "osr", "n", "accuracy"])
        )
        overlays["Oracle (known nuisance)"] = oracle_accuracy
        osr_overlays["Oracle (known nuisance)"] = oracle_osr_overlay
        fraction_overlays = {"Oracle MI fraction": oracle_mi_fraction}

    plot_dataset_metadata(metadata, idx, metadata_plot_path, name)
    plot_confusion_matrix(test_metrics["confusion"], labels_ordered, confusion_path, name)
    plot_accuracy_by_snr(snr_summary, snr_plot_path, name, overlays=overlays)
    plot_accuracy_by_osr(osr_summary, osr_plot_path, name, overlays=osr_overlays or None)
    plot_accuracy_by_ebw(ebw_summary, ebw_plot_path, name)
    plot_information_by_snr(info_snr_summary, info_summary_dict, info_plot_path, name, fraction_overlays=fraction_overlays)

    snr_summary.write_csv(snr_csv_path)
    osr_summary.write_csv(osr_csv_path)
    ebw_summary.write_csv(ebw_csv_path)
    ub_summary.write_csv(ub_csv_path)
    if oracle_metrics is not None:
        oracle_snr_summary.write_csv(oracle_snr_csv_path)
        oracle_osr_summary.write_csv(oracle_osr_csv_path)
        pl.DataFrame([information_summary(oracle_metrics["confusion"], oracle_metrics["nll_bits"])]).write_csv(oracle_info_summary_path)
        oracle_info_snr_summary.write_csv(oracle_info_snr_path)
    class_summary.write_csv(class_csv_path)
    pl.DataFrame([accuracy_bootstrap]).write_csv(accuracy_bootstrap_path)
    pl.DataFrame([info_summary_dict]).write_csv(info_summary_path)
    info_snr_summary.write_csv(info_snr_path)

    calib_df, ece, mce = calibration_stats(test_metrics["y_true"], test_metrics["confidence"], test_metrics["y_pred"])
    calib_snr_df = calibration_by_snr(
        metadata, idx, test_metrics["y_true"], test_metrics["confidence"], test_metrics["y_pred"], SNR_BIN_WIDTH,
    )
    plot_reliability_diagram(calib_df, ece, mce, reliability_path, name)
    plot_calibration_by_snr(calib_snr_df, calib_snr_plot_path, name)
    calib_df.write_csv(calib_csv_path)
    calib_snr_df.write_csv(calib_snr_csv_path)
    predictions = build_prediction_table(metadata, idx, test_metrics, id_to_label, oracle_metrics=oracle_metrics)
    min_slice_count = max(5, len(predictions) // 200)
    error_slices = error_slice_table(predictions, min_count=min_slice_count)
    predictions.write_parquet(predictions_path)
    error_slices.write_csv(error_slices_path)
    write_performance_explorer(
        explorer_path,
        name,
        predictions,
        error_slices,
        test_metrics["confusion"],
        labels_ordered,
        {"accuracy": test_metrics["accuracy"], "ece": ece, "mce": mce},
    )

    plots_path = f"plots/{name}"
    tables_path = f"tables/{name}"
    reports_path = f"reports/{name}"
    mlflow.log_artifact(str(metadata_plot_path), artifact_path=plots_path)
    mlflow.log_artifact(str(confusion_path), artifact_path=plots_path)
    mlflow.log_artifact(str(snr_plot_path), artifact_path=plots_path)
    mlflow.log_artifact(str(osr_plot_path), artifact_path=plots_path)
    mlflow.log_artifact(str(ebw_plot_path), artifact_path=plots_path)
    mlflow.log_artifact(str(info_plot_path), artifact_path=plots_path)
    mlflow.log_artifact(str(reliability_path), artifact_path=plots_path)
    mlflow.log_artifact(str(calib_snr_plot_path), artifact_path=plots_path)
    mlflow.log_artifact(str(snr_csv_path), artifact_path=tables_path)
    mlflow.log_artifact(str(osr_csv_path), artifact_path=tables_path)
    mlflow.log_artifact(str(ebw_csv_path), artifact_path=tables_path)
    mlflow.log_artifact(str(ub_csv_path), artifact_path=tables_path)
    if oracle_metrics is not None:
        mlflow.log_artifact(str(oracle_snr_csv_path), artifact_path=tables_path)
        mlflow.log_artifact(str(oracle_osr_csv_path), artifact_path=tables_path)
        mlflow.log_artifact(str(oracle_info_summary_path), artifact_path=tables_path)
        mlflow.log_artifact(str(oracle_info_snr_path), artifact_path=tables_path)
    mlflow.log_artifact(str(class_csv_path), artifact_path=tables_path)
    mlflow.log_artifact(str(accuracy_bootstrap_path), artifact_path=tables_path)
    mlflow.log_artifact(str(info_summary_path), artifact_path=tables_path)
    mlflow.log_artifact(str(info_snr_path), artifact_path=tables_path)
    mlflow.log_artifact(str(calib_csv_path), artifact_path=tables_path)
    mlflow.log_artifact(str(calib_snr_csv_path), artifact_path=tables_path)
    mlflow.log_artifact(str(predictions_path), artifact_path=tables_path)
    mlflow.log_artifact(str(error_slices_path), artifact_path=tables_path)
    mlflow.log_artifact(str(explorer_path), artifact_path=reports_path)

    mlflow.log_metric(f"{name}_accuracy", test_metrics["accuracy"])
    log_auxiliary_accuracy_metrics(name, test_metrics)
    mlflow.log_metric(
        f"{name}_accuracy_ci_half_width",
        (accuracy_bootstrap["ci_upper"] - accuracy_bootstrap["ci_lower"]) / 2.0,
    )
    if oracle_metrics is not None:
        mlflow.log_metric(f"{name}_oracle_accuracy", oracle_metrics["accuracy"])
    log_f1_metrics(test_metrics["y_true"], test_metrics["y_pred"], labels_ordered, prefix=f"{name}_")

    return {
        "accuracy": test_metrics["accuracy"],
        "accuracy_bootstrap": accuracy_bootstrap,
        "ece": ece,
        "mce": mce,
        "confusion": test_metrics["confusion"],
        "accuracy_by_snr": snr_summary,
        "accuracy_by_osr": osr_summary,
        "information_summary": pl.DataFrame([info_summary_dict]),
        "information_by_snr": info_snr_summary,
        "per_class_metrics": class_summary,
        "calibration_stats": calib_df,
        "confusion_path": str(confusion_path),
    }


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
    extra_test_dirs: Optional[List[str]] = None,
) -> Dict:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    representation = representation_for_model(model_name)
    channel_format = effective_channel_format_for(model_name, args.channel_format)
    base_input_channels = input_channels_for(representation, channel_format)
    validate_preprocessor_args(args.preprocessor, representation, channel_format, base_input_channels)
    preprocessor, model_input_channels = make_preprocessor(
        args.preprocessor,
        base_input_channels,
        out_channels=args.preprocessor_channels,
        kernel_size=args.preprocessor_kernel_size,
        max_time_shift=args.preprocessor_max_time_shift,
        max_frequency_shift=args.preprocessor_max_frequency_shift,
        max_phase_shift=args.preprocessor_max_phase_shift,
    )
    model, representation = make_model(
        model_name,
        len(label_to_id),
        train_signals.shape[1],
        in_channels=model_input_channels,
        spectrogram_base_channels=args.spectrogram_base_channels,
        spectrogram_freq_kernel=args.spectrogram_freq_kernel,
        spectrogram_time_kernel=args.spectrogram_time_kernel,
        transformer_patch_size=args.transformer_patch_size,
        transformer_d_model=args.transformer_d_model,
        transformer_n_heads=args.transformer_n_heads,
        transformer_n_layers=args.transformer_n_layers,
    )
    if preprocessor is not None:
        model = ModelWithPreprocessor(model, preprocessor)

    train_idx, val_idx, test_idx = splits
    auxiliary_encoders = build_metadata_target_encoders(
        train_metadata,
        train_idx,
        args.aux_targets,
        n_bins=args.aux_bins,
    )
    auxiliary_tasks = auxiliary_class_counts(auxiliary_encoders)
    if auxiliary_tasks:
        model = MultiTaskModel(
            model,
            auxiliary_tasks,
            hidden_dim=args.aux_head_hidden,
            uncertainty_weighting=args.aux_loss_mode == "uncertainty",
        )
    model.to(device)
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("representation", representation)

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
        auxiliary_encoders=auxiliary_encoders,
    )
    train_loader = get_data_loader(train_signals, train_metadata, train_idx, label_to_id, shuffle=True, **loader_kwargs)
    val_loader = get_data_loader(val_signals, val_metadata, val_idx, label_to_id, shuffle=False, **loader_kwargs)

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
        loss_totals: dict[str, float] = {}
        for batch in train_bar:
            xb, yb, auxiliary = unpack_batch(batch)
            sync_if_cuda(device)
            xb, yb = xb.to(device), yb.to(device)
            if auxiliary is not None:
                auxiliary = {name: target.to(device) for name, target in auxiliary.items()}
            sync_if_cuda(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = forward_all(model, xb) if auxiliary_tasks else {"modulation": model(xb)}
            loss, losses = multitask_loss(
                model,
                outputs,
                yb,
                auxiliary,
                args.aux_loss_weight,
                args.aux_loss_mode,
            )
            loss.backward()
            optimizer.step()
            sync_if_cuda(device)

            total_loss += loss.item() * len(yb)
            for name, value in losses.items():
                loss_totals[name] = loss_totals.get(name, 0.0) + value.item() * len(yb)
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_duration = time.perf_counter() - train_start
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            len(label_to_id),
            desc="val",
            auxiliary_tasks=auxiliary_tasks or None,
        )
        train_loss = total_loss / max(len(train_loader.dataset), 1)
        epoch_duration = time.perf_counter() - epoch_start
        train_samples_per_sec = len(train_loader.dataset) / max(train_duration, np.finfo(float).eps)
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        for name, loss_total in loss_totals.items():
            mlflow.log_metric(f"train_{safe_metric_name(name)}_loss", loss_total / max(len(train_loader.dataset), 1), step=epoch)
        mlflow.log_metric("val_accuracy", val_metrics["accuracy"], step=epoch)
        log_auxiliary_accuracy_metrics("val", val_metrics, step=epoch)
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
    del epoch_bar
    if "train_bar" in locals():
        del train_bar

    if best_state is not None:
        model.load_state_dict(best_state)
    parameter_counts = model_parameter_counts(model)
    model_info, input_example, signature = model_io_info(
        model, train_loader, id_to_label, representation, channel_format, device
    )
    model_info.update(
        {
            "preprocessor": args.preprocessor,
            "preprocessor_input_channels": base_input_channels,
            "preprocessor_output_channels": model_input_channels,
            "auxiliary_targets": auxiliary_target_info(auxiliary_encoders),
        }
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
    artifact_dir = MLFLOW_STAGING / mlflow.active_run().info.run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    from modreczoo.plotting import plot_input_examples
    input_examples_path = artifact_dir / "input_examples.png"
    plot_input_examples(train_loader, id_to_label, representation, channel_format, input_examples_path)
    if input_examples_path.exists():
        mlflow.log_artifact(str(input_examples_path), artifact_path="plots")
    del train_loader, val_loader
    gc.collect()

    eval_result = evaluate_and_log_test_set(
        "test", model, test_signals, test_metadata, test_idx,
        label_to_id, id_to_label, loader_kwargs, artifact_dir,
        args.test_dataset_dir_effective, args.seed, device,
        auxiliary_tasks=auxiliary_tasks or None,
    )
    labels_ordered = [id_to_label[i] for i in range(len(id_to_label))]
    for extra_dir in (extra_test_dirs or []):
        print(f"Loading extra test dataset {extra_dir}...")
        extra_signals, extra_metadata = load_dataset(extra_dir)
        print(f"Loaded {extra_dir}: signals={extra_signals.shape} dtype={extra_signals.dtype}.")
        validate_known_labels(extra_metadata, labels_ordered, extra_dir, "Extra test")
        extra_name = Path(extra_dir).stem
        extra_idx = np.arange(len(extra_metadata), dtype=np.int64)
        try:
            evaluate_and_log_test_set(
                extra_name, model, extra_signals, extra_metadata, extra_idx,
                label_to_id, id_to_label, loader_kwargs, artifact_dir,
                extra_dir, args.seed, device,
                n_samples=train_signals.shape[1],
                auxiliary_tasks=auxiliary_tasks or None,
            )
        finally:
            del extra_signals, extra_metadata, extra_idx
            gc.collect()
    mlflow.log_metric("best_val_accuracy", best_val_acc)
    return {
        "model": model_name,
        "representation": representation,
        "best_val_accuracy": best_val_acc,
        "test_accuracy": eval_result["accuracy"],
        "accuracy_bootstrap": eval_result["accuracy_bootstrap"],
        "test_ece": eval_result["ece"],
        "test_mce": eval_result["mce"],
        "confusion": eval_result["confusion"],
        "accuracy_by_snr": eval_result["accuracy_by_snr"],
        "accuracy_by_osr": eval_result["accuracy_by_osr"],
        "information_summary": eval_result["information_summary"],
        "information_by_snr": eval_result["information_by_snr"],
        "per_class_metrics": eval_result["per_class_metrics"],
        "calibration_stats": eval_result["calibration_stats"],
        "confusion_path": eval_result["confusion_path"],
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
    if representation in ("iq_features", "csp_features", "csp_canonical"):
        return 1
    if representation == "joint_csp":
        from modreczoo.features import N_CSP_EXPERT_FEATURES
        return 6 + N_CSP_EXPERT_FEATURES  # 113
    if channel_format == "apf":
        return 4
    if channel_format in ("complex_powers", "unit_phasor_powers"):
        return 6
    if channel_format == "multilag":
        return 6
    if channel_format == "cyclic_caf":
        return 3
    if channel_format == "scf":
        return 1
    return 2


def configure_mlflow(experiment_name: str = EXPERIMENT_NAME) -> None:
    MLFLOW_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    MLFLOW_STAGING.mkdir(parents=True, exist_ok=True)
    mlflow.enable_system_metrics_logging()
    mlflow.set_system_metrics_sampling_interval(SYSTEM_METRICS_SAMPLING_INTERVAL_SEC)
    mlflow.set_system_metrics_samples_before_logging(SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING)
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
    if getattr(args, "preprocessor", "none") != "none":
        name += f"-pp{args.preprocessor}"
    if getattr(args, "aux_targets", None):
        aux = "_".join(safe_metric_name(name) for name in args.aux_targets)
        name += f"-aux{aux}"
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
    if args.preprocessor not in PREPROCESSOR_NAMES:
        raise ValueError(f"--preprocessor must be one of {', '.join(PREPROCESSOR_NAMES)}.")
    if args.preprocessor_kernel_size % 2 == 0 or args.preprocessor_kernel_size < 1:
        raise ValueError("--preprocessor-kernel-size must be a positive odd integer.")
    if args.preprocessor_channels is not None and args.preprocessor_channels <= 0:
        raise ValueError("--preprocessor-channels must be positive.")
    if args.aux_bins < 2:
        raise ValueError("--aux-bins must be at least 2.")
    if args.aux_loss_weight < 0:
        raise ValueError("--aux-loss-weight must be non-negative.")
    if args.aux_head_hidden < 0:
        raise ValueError("--aux-head-hidden must be non-negative.")


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
            "test_n_samples": getattr(args, "test_n_samples_original", test_signals.shape[1]),
            "test_effective_n_samples": test_signals.shape[1],
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
            "preprocessor": getattr(args, "preprocessor", "none"),
            "preprocessor_channels": getattr(args, "preprocessor_channels", None) or "input",
            "preprocessor_kernel_size": getattr(args, "preprocessor_kernel_size", 31),
            "preprocessor_max_time_shift": getattr(args, "preprocessor_max_time_shift", 8.0),
            "preprocessor_max_frequency_shift": getattr(args, "preprocessor_max_frequency_shift", 0.02),
            "preprocessor_max_phase_shift": getattr(args, "preprocessor_max_phase_shift", float(np.pi)),
            "aux_targets": json.dumps(getattr(args, "aux_targets", None) or []),
            "aux_bins": getattr(args, "aux_bins", 8),
            "aux_loss_weight": getattr(args, "aux_loss_weight", 0.2),
            "aux_loss_mode": getattr(args, "aux_loss_mode", "fixed"),
            "aux_head_hidden": getattr(args, "aux_head_hidden", 0),
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
    extra_test_dirs: Optional[List[str]] = None,
) -> None:
    test_n_samples_original = test_signals.shape[1]
    test_signals = pad_or_trim_signals(test_signals, train_signals.shape[1])
    args.test_n_samples_original = test_n_samples_original
    if test_n_samples_original != test_signals.shape[1]:
        print(
            f"Adjusted test signal length from {test_n_samples_original} "
            f"to train length {test_signals.shape[1]}."
        )
    with mlflow.start_run(run_name=run_name_for(args, model_name), log_system_metrics=True) as run:
        log_common_params(args, model_name, train_signals, test_signals, labels, sweep_index, sweep_total)
        if config_yaml is not None:
            mlflow.log_text(config_yaml, "config.yaml")
        split_dir = MLFLOW_STAGING / run.info.run_id / "splits"
        split_dir.mkdir(parents=True, exist_ok=True)
        np.save(split_dir / "train_idx.npy", splits[0])
        np.save(split_dir / "val_idx.npy", splits[1])
        np.save(split_dir / "test_idx.npy", splits[2])
        mlflow.log_artifacts(str(split_dir), artifact_path="splits")
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
            extra_test_dirs=extra_test_dirs,
        )
        summary_path = MLFLOW_ARTIFACTS / f"performance_summary_{run.info.run_id}.txt"
        write_summary(summary_path, run.info.run_id, args, labels, [result])
        mlflow.log_artifact(str(summary_path), artifact_path="summaries")
        print(summary_path)
