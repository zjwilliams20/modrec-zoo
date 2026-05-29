import json
import sqlite3
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable
from urllib.parse import urlparse

import mlflow.pytorch
import numpy as np
import polars as pl
import plotly.graph_objects as go
import plotly.io as pio
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from modreczoo.data import get_data_loader, load_dataset, ordered_modulation_labels
from modreczoo.training import dataset_sample_indices, stratified_split, stratified_train_val_split


MLFLOW_DB = Path("mlflow/mlflow.db")


@dataclass
class RunInfo:
    run_id: str
    run_name: str
    artifact_dir: Path
    model_uri: str
    params: Dict[str, str]
    config: Dict[str, Any]
    model_info: Dict[str, Any]


@dataclass
class DomainSpec:
    name: str
    dataset_dir: str | None
    split: str | None


def load_run_info(run_id: str, mlflow_db: Path = MLFLOW_DB) -> RunInfo:
    with sqlite3.connect(str(mlflow_db)) as con:
        row = con.execute(
            "select name, artifact_uri from runs where run_uuid=?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown MLflow run id: {run_id}")
        run_name, artifact_uri = row
        model_row = con.execute(
            "select artifact_location from logged_models where source_run_id=? and name='model'",
            (run_id,),
        ).fetchone()
        if model_row is None:
            raise ValueError(f"Run {run_id} has no logged PyTorch model named 'model'.")
        params = dict(con.execute("select key, value from params where run_uuid=?", (run_id,)))

    artifact_dir = _path_from_uri(artifact_uri)
    config = _read_simple_yaml(artifact_dir / "config.yaml")
    model_info = json.loads((artifact_dir / "model_info.json").read_text(encoding="utf-8"))
    return RunInfo(
        run_id=run_id,
        run_name=run_name or run_id,
        artifact_dir=artifact_dir,
        model_uri=model_row[0],
        params=params,
        config=config,
        model_info=model_info,
    )


def parse_domain_specs(items: Iterable[str]) -> list[DomainSpec]:
    specs = []
    for item in items:
        if item.endswith(":auto"):
            specs.append(DomainSpec(item[:-5], None, item[:-5]))
        elif "=" in item:
            name, dataset_dir = item.split("=", 1)
            specs.append(DomainSpec(name, dataset_dir, None))
        else:
            raise ValueError(f"Expected domain as name:auto or name=dataset_dir, got {item!r}.")
    if not specs:
        raise ValueError("Expected at least one domain.")
    return specs


def write_domain_shift_report(
    run_id: str,
    domain_specs: list[DomainSpec],
    output_dir: Path,
    source_domain: str | None = None,
    max_examples_per_domain: int = 5000,
    batch_size: int | None = None,
    num_workers: int = 0,
    device_name: str = "cuda",
    seed: int | None = None,
) -> dict[str, Path]:
    run = load_run_info(run_id)
    seed = int(seed if seed is not None else _param(run, "seed", 0, int))
    labels = _labels_for_run(run)
    label_to_id = {label: i for i, label in enumerate(labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    device = torch.device(device_name if device_name and torch.cuda.is_available() or device_name == "cpu" else "cpu")
    model = load_logged_model(run, device)
    layer_name, layer = _last_linear(model)

    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for spec in domain_specs:
        signals, metadata, indices = _load_domain(run, spec, labels, seed)
        indices = _sample_indices(metadata, indices, max_examples_per_domain, seed)
        rows = extract_domain_latents(
            model,
            signals,
            metadata,
            indices,
            run,
            spec.name,
            label_to_id,
            id_to_label,
            device,
            layer,
            batch_size=batch_size or _param(run, "batch_size", 128, int),
            num_workers=num_workers,
        )
        all_rows.append(rows)
        del signals, metadata

    latents = pl.concat(all_rows, how="diagonal")
    emb_cols = [c for c in latents.columns if c.startswith("emb_")]
    emb = latents.select(emb_cols).to_numpy()
    projection = _projection_table(latents, emb, seed)
    source_domain = source_domain or str(projection["domain"][0])
    if source_domain not in set(projection["domain"].unique().to_list()):
        raise ValueError(f"Source domain {source_domain!r} is not present in the extracted domains.")
    domain_summary = _domain_summary(projection, source_domain)
    class_shift = _class_shift(projection, source_domain)
    slice_shift = _slice_shift(projection, source_domain)

    latents_path = output_dir / "latents.parquet"
    projection_path = output_dir / "projection.parquet"
    domain_summary_path = output_dir / "domain_summary.csv"
    class_shift_path = output_dir / "class_shift.csv"
    slice_shift_path = output_dir / "slice_shift.csv"
    html_path = output_dir / "latent_explorer.html"
    latents.write_parquet(latents_path)
    projection.write_parquet(projection_path)
    domain_summary.write_csv(domain_summary_path)
    class_shift.write_csv(class_shift_path)
    slice_shift.write_csv(slice_shift_path)
    _write_html_report(html_path, run, projection, domain_summary, class_shift, slice_shift, source_domain, layer_name)
    return {
        "latents": latents_path,
        "projection": projection_path,
        "domain_summary": domain_summary_path,
        "class_shift": class_shift_path,
        "slice_shift": slice_shift_path,
        "html": html_path,
    }


def extract_domain_latents(
    model: torch.nn.Module,
    signals: np.ndarray,
    metadata: pl.DataFrame,
    indices: np.ndarray,
    run: RunInfo,
    domain_name: str,
    label_to_id: Dict[str, int],
    id_to_label: Dict[int, str],
    device: torch.device,
    layer: torch.nn.Module,
    batch_size: int,
    num_workers: int,
) -> pl.DataFrame:
    captured: list[torch.Tensor] = []

    def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captured.append(inputs[0].detach().cpu())

    handle = layer.register_forward_pre_hook(hook)
    y_true, y_pred, confidence, entropy_bits, nll_bits = [], [], [], [], []
    top2_pred, top2_conf = [], []
    loader = get_data_loader(
        signals,
        metadata,
        indices,
        label_to_id,
        model_name=_param(run, "model_name", ""),
        channel_format=_param(run, "channel_format", "real_imag"),
        remove_cfo=_as_bool(_param(run, "remove_cfo", "False")),
        cfo_estimator=_param(run, "cfo_estimator", "lag_correlation"),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        spectrogram_freq_bins=_param(run, "spectrogram_freq_bins", 64, int),
        spectrogram_time_bins=_param(run, "spectrogram_time_bins", 64, int),
        spectrogram_nperseg=_param(run, "spectrogram_nperseg", 64, int),
        spectrogram_noverlap=_param(run, "spectrogram_noverlap", 48, int),
        spectrogram_window=_param(run, "spectrogram_window", "hann"),
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        n_samples=int(run.model_info.get("input_shape", [None, None, _param(run, "train_n_samples", 0, int)])[-1]),
    )
    model.eval()
    try:
        with torch.no_grad():
            for xb, yb in tqdm(loader, desc=f"latents:{domain_name}", unit="batch", leave=False):
                logits = model(xb.to(device))
                probs = torch.softmax(logits, dim=1)
                conf, pred = probs.max(dim=1)
                top_probs, top_idx = torch.topk(probs, k=min(2, probs.shape[1]), dim=1)
                yb_device = yb.to(device)
                y_true.extend(yb.tolist())
                y_pred.extend(pred.cpu().tolist())
                confidence.extend(conf.cpu().tolist())
                entropy_bits.extend((-(probs * torch.log2(probs.clamp_min(1e-12))).sum(dim=1)).cpu().tolist())
                nll_bits.extend((F.cross_entropy(logits, yb_device, reduction="none") / np.log(2.0)).cpu().tolist())
                top2_pred.extend(top_idx[:, 1].cpu().tolist() if probs.shape[1] > 1 else pred.cpu().tolist())
                top2_conf.extend(top_probs[:, 1].cpu().tolist() if probs.shape[1] > 1 else torch.zeros_like(conf).cpu().tolist())
    finally:
        handle.remove()

    emb = torch.cat(captured, dim=0).numpy().astype(np.float32)
    true_labels = [id_to_label[int(i)] for i in y_true]
    pred_labels = [id_to_label[int(i)] for i in y_pred]
    rows = metadata[indices].clone().with_columns(
        pl.lit(domain_name).alias("domain"),
        pl.Series("row_index", indices.astype(np.int64)),
        pl.Series("true_label", true_labels),
        pl.Series("pred_label", pred_labels),
        pl.Series("correct", np.asarray(y_true) == np.asarray(y_pred)),
        pl.Series("confidence", np.asarray(confidence, dtype=np.float32)),
        pl.Series("entropy_bits", np.asarray(entropy_bits, dtype=np.float32)),
        pl.Series("nll_bits", np.asarray(nll_bits, dtype=np.float32)),
        pl.Series("top2_label", [id_to_label[int(i)] for i in top2_pred]),
        pl.Series("top2_confidence", np.asarray(top2_conf, dtype=np.float32)),
    )
    emb_frame = pl.DataFrame({f"emb_{i:04d}": emb[:, i] for i in range(emb.shape[1])})
    return pl.concat([rows, emb_frame], how="horizontal")


def load_logged_model(run: RunInfo, device: torch.device) -> torch.nn.Module:
    _install_legacy_model_aliases()
    return mlflow.pytorch.load_model(run.model_uri, map_location=device).to(device)


def reconstruct_split_indices(run: RunInfo, split: str, labels: np.ndarray) -> tuple[str, np.ndarray]:
    split = split.lower()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Can only reconstruct train/val/test, got {split!r}.")
    dataset_dir = _param(run, "dataset_dir", run.config.get("dataset_dir"))
    test_source = _param(run, "test_dataset_source", "heldout_split")
    split_path = run.artifact_dir / "splits" / f"{split}_idx.npy"
    if split_path.exists():
        if split == "test" and test_source == "external_dataset":
            return _param(run, "test_dataset_dir", dataset_dir), np.load(split_path)
        return dataset_dir, np.load(split_path)
    sample_frac = _param(run, "sample_frac", run.config.get("sample_frac", 1.0), float)
    max_examples_raw = _param(run, "max_examples", run.config.get("max_examples", None))
    max_examples = None if max_examples_raw in (None, "none", "null") else int(max_examples_raw)
    seed = _param(run, "seed", run.config.get("seed", 0), int)
    train_frac = float(run.config.get("train_frac", 0.7))
    val_frac = float(run.config.get("val_frac", 0.15))
    sampled = dataset_sample_indices(labels, sample_frac, max_examples, seed)
    sampled_labels = labels[sampled]
    if test_source == "external_dataset":
        if split == "test":
            test_dir = _param(run, "test_dataset_dir", None)
            _, test_metadata = load_dataset(test_dir)
            test_labels = test_metadata["modulation"].to_numpy()
            return test_dir, dataset_sample_indices(test_labels, sample_frac, max_examples, seed)
        train_rel, val_rel = stratified_train_val_split(sampled_labels, train_frac, val_frac, seed)
        return dataset_dir, sampled[train_rel if split == "train" else val_rel]
    train_rel, val_rel, test_rel = stratified_split(sampled_labels, train_frac, val_frac, seed)
    rel = {"train": train_rel, "val": val_rel, "test": test_rel}[split]
    return dataset_dir, sampled[rel]


def _load_domain(run: RunInfo, spec: DomainSpec, labels: list[str], seed: int) -> tuple[np.ndarray, pl.DataFrame, np.ndarray]:
    if spec.split is not None:
        train_signals, train_metadata = load_dataset(_param(run, "dataset_dir", run.config.get("dataset_dir")))
        dataset_dir, indices = reconstruct_split_indices(run, spec.split, train_metadata["modulation"].to_numpy())
        if dataset_dir == _param(run, "dataset_dir", run.config.get("dataset_dir")):
            return train_signals, train_metadata, indices
        signals, metadata = load_dataset(dataset_dir)
        return signals, metadata, indices
    signals, metadata = load_dataset(str(spec.dataset_dir))
    missing = sorted(set(metadata["modulation"].unique().to_list()) - set(labels))
    if missing:
        raise ValueError(f"Domain {spec.name} has unknown labels: {missing}")
    return signals, metadata, np.arange(len(metadata), dtype=np.int64)


def _projection_table(latents: pl.DataFrame, emb: np.ndarray, seed: int) -> pl.DataFrame:
    x = StandardScaler().fit_transform(emb)
    pca_dim = min(20, x.shape[1], x.shape[0])
    pca_full = PCA(n_components=pca_dim, random_state=seed).fit_transform(x)
    pca2 = pca_full[:, :2] if pca_dim >= 2 else np.column_stack([pca_full[:, 0], np.zeros(len(pca_full))])
    try:
        import umap
    except ImportError as exc:
        raise ImportError("Install umap-learn with `uv sync` before running this tool.") from exc
    n_neighbors = min(30, max(2, len(x) - 1))
    umap2 = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.08, metric="euclidean", random_state=seed).fit_transform(pca_full)
    keep = [c for c in latents.columns if not c.startswith("emb_")]
    return latents.select(keep).with_columns(
        pl.Series("pca_x", pca2[:, 0].astype(np.float32)),
        pl.Series("pca_y", pca2[:, 1].astype(np.float32)),
        pl.Series("umap_x", umap2[:, 0].astype(np.float32)),
        pl.Series("umap_y", umap2[:, 1].astype(np.float32)),
    )


def _domain_summary(df: pl.DataFrame, source_domain: str) -> pl.DataFrame:
    points = df.select(["domain", "umap_x", "umap_y"]).to_numpy()
    domain = points[:, 0]
    xy = points[:, 1:].astype(float)
    source_centroid = xy[domain == source_domain].mean(axis=0)
    purity = _knn_same_domain_purity(xy, domain.astype(str))
    rows = []
    for name in sorted(set(domain)):
        mask = domain == name
        centroid = xy[mask].mean(axis=0)
        rows.append({
            "domain": str(name),
            "n": int(mask.sum()),
            "accuracy": float(df.filter(pl.col("domain") == name)["correct"].mean()),
            "mean_confidence": float(df.filter(pl.col("domain") == name)["confidence"].mean()),
            "mean_entropy_bits": float(df.filter(pl.col("domain") == name)["entropy_bits"].mean()),
            "centroid_distance_to_source": float(np.linalg.norm(centroid - source_centroid)),
            "knn_same_domain_purity": float(purity[mask].mean()),
        })
    return pl.DataFrame(rows).sort("domain")


def _class_shift(df: pl.DataFrame, source_domain: str) -> pl.DataFrame:
    rows = []
    source = df.filter(pl.col("domain") == source_domain)
    for label in sorted(df["true_label"].unique().to_list()):
        src = source.filter(pl.col("true_label") == label)
        if len(src) == 0:
            continue
        src_centroid = src.select(["umap_x", "umap_y"]).to_numpy().mean(axis=0)
        src_acc = float(src["correct"].mean())
        for domain in sorted(df["domain"].unique().to_list()):
            cur = df.filter((pl.col("domain") == domain) & (pl.col("true_label") == label))
            if len(cur) == 0:
                continue
            cur_centroid = cur.select(["umap_x", "umap_y"]).to_numpy().mean(axis=0)
            acc = float(cur["correct"].mean())
            rows.append({
                "domain": domain,
                "true_label": label,
                "n": len(cur),
                "accuracy": acc,
                "source_accuracy": src_acc,
                "accuracy_delta": acc - src_acc,
                "centroid_distance_to_source": float(np.linalg.norm(cur_centroid - src_centroid)),
                "mean_confidence": float(cur["confidence"].mean()),
                "mean_entropy_bits": float(cur["entropy_bits"].mean()),
            })
    return pl.DataFrame(rows).sort(["centroid_distance_to_source", "n"], descending=[True, True])


def _slice_shift(df: pl.DataFrame, source_domain: str) -> pl.DataFrame:
    dims = [c for c in ["channel", "snr_bin", "osr", "symbol_period", "channel_n_taps"] if c in df.columns]
    work = df
    if "snr_db" in work.columns and "snr_bin" not in work.columns:
        snr = np.floor(work["snr_db"].to_numpy() / 4.0) * 4.0
        work = work.with_columns(pl.Series("snr_bin", [f"{v:g}-{v + 4:g}" for v in snr]))
        if "snr_bin" not in dims:
            dims.insert(0, "snr_bin")
    rows = []
    for dim in dims:
        for value in sorted(work[dim].cast(pl.String).unique().to_list()):
            src = work.filter((pl.col("domain") == source_domain) & (pl.col(dim).cast(pl.String) == value))
            if len(src) == 0:
                continue
            src_acc = float(src["correct"].mean())
            src_centroid = src.select(["umap_x", "umap_y"]).to_numpy().mean(axis=0)
            for domain in sorted(work["domain"].unique().to_list()):
                cur = work.filter((pl.col("domain") == domain) & (pl.col(dim).cast(pl.String) == value))
                if len(cur) < 10:
                    continue
                cur_centroid = cur.select(["umap_x", "umap_y"]).to_numpy().mean(axis=0)
                acc = float(cur["correct"].mean())
                rows.append({
                    "domain": domain,
                    "dimension": dim,
                    "slice": value,
                    "n": len(cur),
                    "accuracy": acc,
                    "source_accuracy": src_acc,
                    "accuracy_delta": acc - src_acc,
                    "centroid_distance_to_source": float(np.linalg.norm(cur_centroid - src_centroid)),
                })
    return pl.DataFrame(rows).sort(["centroid_distance_to_source", "n"], descending=[True, True]) if rows else pl.DataFrame()


def _write_html_report(path: Path, run: RunInfo, df: pl.DataFrame, domain_summary: pl.DataFrame, class_shift: pl.DataFrame, slice_shift: pl.DataFrame, source_domain: str, layer_name: str) -> None:
    fig = _scatter_figure(df)
    summary_html = domain_summary.to_pandas().to_html(index=False, float_format=lambda x: f"{x:.4f}")
    class_html = class_shift.head(20).to_pandas().to_html(index=False, float_format=lambda x: f"{x:.4f}")
    slice_html = slice_shift.head(20).to_pandas().to_html(index=False, float_format=lambda x: f"{x:.4f}") if len(slice_shift) else "<p>No slice shifts.</p>"
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{run.run_name} latent domain shift</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; background: #f8fafc; }}
h1 {{ margin: 0 0 8px; font-size: 28px; }}
h2 {{ margin: 28px 0 10px; font-size: 18px; }}
.plot, table {{ background: white; border: 1px solid #d8dee9; border-radius: 6px; }}
.plot {{ padding: 8px; margin-bottom: 18px; }}
table {{ border-collapse: collapse; font-size: 12px; }}
th, td {{ padding: 5px 8px; border-bottom: 1px solid #e5e9f0; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
p {{ color: #52606d; max-width: 920px; }}
</style>
</head>
<body>
<h1>{run.run_name} Latent Domain Shift</h1>
<p>Run {run.run_id}. Embeddings are captured from <code>{layer_name}</code>; source domain is <code>{source_domain}</code>.</p>
<div class="plot">{pio.to_html(fig, include_plotlyjs=True, full_html=False)}</div>
<h2>Domain Summary</h2>{summary_html}
<h2>Largest Class Shifts</h2>{class_html}
<h2>Largest Metadata Slice Shifts</h2>{slice_html}
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def _scatter_figure(df: pl.DataFrame) -> go.Figure:
    hover_cols = [c for c in ["domain", "row_index", "true_label", "pred_label", "correct", "confidence", "entropy_bits", "snr_db", "osr", "channel"] if c in df.columns]
    buttons = []
    traces = []
    visibility_groups = []
    color_dims = [c for c in ["domain", "true_label", "pred_label", "correct", "channel"] if c in df.columns]
    color_dims += [c for c in ["confidence", "entropy_bits", "snr_db"] if c in df.columns]
    custom = np.stack([df[c].cast(pl.String).to_numpy() for c in hover_cols], axis=1)
    hover = "<br>".join(f"{c}=%{{customdata[{i}]}}" for i, c in enumerate(hover_cols)) + "<extra></extra>"
    marker_size = 8
    for i, dim in enumerate(color_dims):
        values = df[dim].to_numpy()
        group_indices = []
        if df.schema[dim].is_numeric():
            marker = {"color": values, "colorscale": "Viridis", "showscale": True, "size": marker_size, "opacity": 0.78}
            group_indices.append(len(traces))
            traces.append(
                go.Scattergl(
                    x=df["umap_x"],
                    y=df["umap_y"],
                    mode="markers",
                    marker=marker,
                    customdata=custom,
                    hovertemplate=hover,
                    visible=i == 0,
                    name=dim,
                    showlegend=False,
                )
            )
        else:
            text_values = values.astype(str)
            categories = sorted(set(text_values), key=str)
            palette = _category_palette(len(categories))
            for cat_index, category in enumerate(categories):
                mask = text_values == category
                group_indices.append(len(traces))
                traces.append(
                    go.Scattergl(
                        x=df["umap_x"].to_numpy()[mask],
                        y=df["umap_y"].to_numpy()[mask],
                        mode="markers",
                        marker={
                            "color": palette[cat_index],
                            "size": marker_size,
                            "opacity": 0.78,
                        },
                        customdata=custom[mask],
                        hovertemplate=hover,
                        visible=i == 0,
                        name=str(category),
                        legendgroup=str(category),
                        showlegend=True,
                    )
                )
        visibility_groups.append(group_indices)
    for i, dim in enumerate(color_dims):
        visible = [False] * len(traces)
        for trace_index in visibility_groups[i]:
            visible[trace_index] = True
        buttons.append({
            "label": dim,
            "method": "update",
            "args": [
                {"visible": visible},
                {"title": f"UMAP colored by {dim}", "legend.title.text": dim},
            ],
        })
    fig = go.Figure(traces)
    fig.update_layout(
        title=f"UMAP colored by {color_dims[0]}",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        height=720,
        legend={
            "title": {"text": color_dims[0]},
            "itemsizing": "constant",
            "font": {"size": 13},
            "bgcolor": "rgba(255,255,255,0.9)",
            "bordercolor": "#d8dee9",
            "borderwidth": 1,
        },
        updatemenus=[{"buttons": buttons, "direction": "down", "x": 1.0, "y": 1.12}],
    )
    return fig


def _category_palette(n: int) -> list[str]:
    base = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
    ]
    if n <= len(base):
        return base[:n]
    return [base[i % len(base)] for i in range(n)]


def _install_legacy_model_aliases() -> None:
    from modreczoo.models.cnn import CNN1D, CNN2D
    from modreczoo.models.dilated import DilatedCNN1D, DilatedConvCell1D
    from modreczoo.models.multiscale import MultiScalePyramidNet, _ScaleEncoder
    from modreczoo.models.resnet import ResBlock1D, ResBlock2D, ResNet1D, ResNet2D
    from modreczoo.models.streams import APFNet, MultiStreamNet, _StreamEncoder
    from modreczoo.models.transformer import PatchTransformer1D

    advanced = types.ModuleType("modreczoo.models.advanced")
    for obj in [PatchTransformer1D, MultiScalePyramidNet, _ScaleEncoder, DilatedCNN1D, DilatedConvCell1D, MultiStreamNet, APFNet, _StreamEncoder]:
        setattr(advanced, obj.__name__, obj)
    baselines = types.ModuleType("modreczoo.models.baselines")
    for obj in [CNN1D, CNN2D, ResBlock1D, ResBlock2D, ResNet1D, ResNet2D]:
        setattr(baselines, obj.__name__, obj)
    sys.modules.setdefault("modreczoo.models.advanced", advanced)
    sys.modules.setdefault("modreczoo.models.baselines", baselines)


def _last_linear(model: torch.nn.Module) -> tuple[str, torch.nn.Linear]:
    if hasattr(model, "embedding_layer"):
        return model.embedding_layer()  # type: ignore[no-any-return, attr-defined]
    last = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            last = (name, module)
    if last is None:
        raise ValueError("Model has no nn.Linear layer for embedding extraction.")
    return last


def _sample_indices(metadata: pl.DataFrame, indices: np.ndarray, max_examples: int, seed: int) -> np.ndarray:
    if max_examples <= 0 or len(indices) <= max_examples:
        return indices
    labels = metadata[indices]["modulation"].to_numpy()
    rel = dataset_sample_indices(labels, 1.0, max_examples, seed)
    return indices[rel]


def _knn_same_domain_purity(xy: np.ndarray, domains: np.ndarray, k: int = 15) -> np.ndarray:
    if len(xy) <= 1:
        return np.ones(len(xy))
    n_neighbors = min(k + 1, len(xy))
    ind = NearestNeighbors(n_neighbors=n_neighbors).fit(xy).kneighbors(return_distance=False)
    neigh = ind[:, 1:]
    return (domains[neigh] == domains[:, None]).mean(axis=1)


def _labels_for_run(run: RunInfo) -> list[str]:
    raw = run.params.get("labels")
    if raw:
        return list(json.loads(raw))
    labels = run.model_info.get("labels")
    if labels:
        return list(labels)
    _, metadata = load_dataset(_param(run, "dataset_dir", run.config.get("dataset_dir")))
    return ordered_modulation_labels(metadata["modulation"].unique().to_list())


def _path_from_uri(uri: str) -> Path:
    parsed = urlparse(uri)
    return Path(parsed.path if parsed.scheme == "file" else uri)


def _param(run: RunInfo, key: str, default: Any = None, cast: Any = None) -> Any:
    value = run.params.get(key, run.config.get(key, default))
    if cast is not None and value is not None:
        return cast(value)
    return value


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def _read_simple_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    result: Dict[str, Any] = {}
    current_key = None
    current_list = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        if raw.startswith("- ") and current_key is not None and current_list is not None:
            current_list.append(_coerce_scalar(raw[2:].strip()))
            continue
        if ":" not in raw or raw.startswith(" "):
            continue
        key, value = raw.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "":
            current_list = []
            result[key] = current_list
        else:
            result[key] = _coerce_scalar(value)
            current_list = None
        current_key = key
    return result


def _coerce_scalar(value: str) -> Any:
    if value in {"null", "None", "none"}:
        return None
    if value in {"true", "false"}:
        return value == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
