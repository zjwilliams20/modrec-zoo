import argparse
import json
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
from preprocessing import ModrecDataset
from simulator import load_dataset


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
    train_ds = ModrecDataset(
        signals,
        metadata,
        train_idx,
        label_to_id,
        representation,
        args.channel_format,
        args.remove_cfo,
        args.cfo_estimator,
    )
    val_ds = ModrecDataset(
        signals,
        metadata,
        val_idx,
        label_to_id,
        representation,
        args.channel_format,
        args.remove_cfo,
        args.cfo_estimator,
    )
    test_ds = ModrecDataset(
        signals,
        metadata,
        test_idx,
        label_to_id,
        representation,
        args.channel_format,
        args.remove_cfo,
        args.cfo_estimator,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

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
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_duration = time.perf_counter() - train_start
        val_start = time.perf_counter()
        val_metrics = evaluate(model, val_loader, device, len(label_to_id), desc="val")
        val_duration = time.perf_counter() - val_start
        train_loss = total_loss / max(len(train_ds), 1)
        epoch_duration = time.perf_counter() - epoch_start
        train_samples_per_sec = len(train_ds) / max(train_duration, np.finfo(float).eps)
        val_samples_per_sec = len(val_ds) / max(val_duration, np.finfo(float).eps)
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_metrics["accuracy"], step=epoch)
        mlflow.log_metric("epoch_duration_sec", epoch_duration, step=epoch)
        mlflow.log_metric("train_duration_sec", train_duration, step=epoch)
        mlflow.log_metric("val_duration_sec", val_duration, step=epoch)
        mlflow.log_metric("train_samples_per_sec", train_samples_per_sec, step=epoch)
        mlflow.log_metric("val_samples_per_sec", val_samples_per_sec, step=epoch)
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
    test_samples_per_sec = len(test_ds) / max(test_duration, np.finfo(float).eps)
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
    mlflow.log_artifact(str(confusion_path), artifact_path="plots")
    mlflow.log_artifact(str(snr_plot_path), artifact_path="plots")
    mlflow.log_artifact(str(snr_csv_path), artifact_path="tables")
    mlflow.log_artifact(str(class_csv_path), artifact_path="tables")

    mlflow.log_metric("test_accuracy", test_metrics["accuracy"])
    mlflow.log_metric("best_val_accuracy", best_val_acc)
    mlflow.log_metric("test_duration_sec", test_duration)
    mlflow.log_metric("test_samples_per_sec", test_samples_per_sec)
    log_prf_metrics(test_metrics["y_true"], test_metrics["y_pred"], labels_ordered)
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
    parser.add_argument("--dataset-dir", default="data/awgn_sobol")
    parser.add_argument("--mlflow-dir", default="mlruns")
    parser.add_argument("--mlflow-db", default="mlflow.db")
    parser.add_argument("--artifact-staging-dir", default=".mlflow_artifact_staging")
    parser.add_argument("--experiment", default="modrec-supervised-baselines")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["time_cnn", "frequency_cnn", "feature_mlp"],
        choices=("time_cnn", "frequency_cnn", "spectrogram_cnn", "feature_mlp"),
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--channel-format", choices=("real_imag", "mag_phase"), default="real_imag")
    parser.add_argument("--remove-cfo", action="store_true")
    parser.add_argument(
        "--cfo-estimator",
        choices=("lag_correlation", "phase_slope", "spectral_centroid"),
        default="lag_correlation",
    )
    parser.add_argument("--snr-bin-width", type=float, default=4.0)
    parser.add_argument("--system-metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--system-metrics-interval", type=int, default=10)
    return parser


def label_order_from_manifest(dataset_dir: str, observed_labels: List[str]) -> List[str]:
    manifest_path = Path(dataset_dir) / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        manifest_order = manifest.get("modulations", [])
        labels = [label for label in manifest_order if label in observed_labels]
        labels.extend(label for label in observed_labels if label not in labels)
        return labels
    return observed_labels


def configure_mlflow(args: argparse.Namespace) -> None:
    artifact_root = Path(args.mlflow_dir).absolute()
    artifact_root.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{Path(args.mlflow_db).absolute()}")

    client = MlflowClient()
    experiment = client.get_experiment_by_name(args.experiment)
    if experiment is None:
        client.create_experiment(args.experiment, artifact_location=artifact_root.as_uri())

    mlflow.set_experiment(args.experiment)
    mlflow.set_experiment_tag("artifact_root", str(artifact_root))
    if args.system_metrics:
        mlflow.set_system_metrics_sampling_interval(args.system_metrics_interval)
        mlflow.enable_system_metrics_logging()


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    signals, metadata = load_dataset(args.dataset_dir)
    observed_labels = metadata["modulation"].unique().to_list()
    labels = label_order_from_manifest(args.dataset_dir, observed_labels)
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    label_values = metadata["modulation"].to_numpy()
    splits = stratified_split(label_values, args.train_frac, args.val_frac, args.seed)

    configure_mlflow(args)

    for model_name in args.models:
        cfo_suffix = f"-cfo-{args.cfo_estimator.replace('_', '-')}" if args.remove_cfo else "-raw-cfo"
        run_name = f"{model_name}-{args.channel_format}-bs{args.batch_size}{cfo_suffix}"
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(
                {
                    "dataset_dir": args.dataset_dir,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "seed": args.seed,
                    "n_samples": signals.shape[1],
                    "n_examples": signals.shape[0],
                    "channel_format": args.channel_format,
                    "remove_cfo": args.remove_cfo,
                    "cfo_estimator": args.cfo_estimator,
                    "labels": json.dumps(labels),
                }
            )
            result = train_one_model(args, model_name, signals, metadata, label_values, label_to_id, id_to_label, splits)
            summary_path = Path(args.mlflow_dir) / f"performance_summary_{run.info.run_id}.txt"
            write_summary(summary_path, run.info.run_id, args, labels, [result])
            mlflow.log_artifact(str(summary_path), artifact_path="summaries")
            print(summary_path)


if __name__ == "__main__":
    main()
