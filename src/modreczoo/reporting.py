from html import escape
from pathlib import Path
import re
from typing import Dict, Iterable

import numpy as np
import polars as pl
import plotly.graph_objects as go
import plotly.io as pio


EXCLUDED_SLICE_COLUMNS = {
    "signal_id",
    "source_path",
    "source_zip",
    "source_member",
    "channel_tap_delays",
    "channel_tap_real",
    "channel_tap_imag",
}
EXACT_NUMERIC_COLUMNS = {
    "osr",
    "symbol_period",
    "upsample_factor",
    "downsample_factor",
    "n_samples",
    "n_symbols",
    "channel_n_taps",
    "channel_max_delay_samples",
}


def build_prediction_table(
    metadata: pl.DataFrame,
    idx: np.ndarray,
    metrics: Dict,
    id_to_label: Dict[int, str],
    oracle_metrics: Dict | None = None,
) -> pl.DataFrame:
    rows = metadata[idx].clone()
    y_true = np.asarray(metrics["y_true"], dtype=int)
    y_pred = np.asarray(metrics["y_pred"], dtype=int)
    true_labels = _labels(y_true, id_to_label)
    pred_labels = _labels(y_pred, id_to_label)
    top2_labels = _labels(np.asarray(metrics["top2_pred"], dtype=int), id_to_label)
    correct = y_true == y_pred
    rows = rows.with_columns(
        pl.Series("true_label", true_labels),
        pl.Series("pred_label", pred_labels),
        pl.Series("correct", correct),
        pl.Series("error_pair", ["correct" if ok else f"{t}->{p}" for ok, t, p in zip(correct, true_labels, pred_labels)]),
        pl.Series("confidence", np.asarray(metrics["confidence"], dtype=np.float32)),
        pl.Series("nll_bits", np.asarray(metrics["nll_bits"], dtype=np.float32)),
        pl.Series("true_probability", np.asarray(metrics["true_probability"], dtype=np.float32)),
        pl.Series("top2_label", top2_labels),
        pl.Series("top2_confidence", np.asarray(metrics["top2_confidence"], dtype=np.float32)),
        pl.Series(
            "pred_margin",
            np.asarray(metrics["confidence"], dtype=np.float32)
            - np.asarray(metrics["top2_confidence"], dtype=np.float32),
        ),
    )
    if oracle_metrics is not None:
        oracle_pred = np.asarray(oracle_metrics["y_pred"], dtype=int)
        oracle_labels = _labels(oracle_pred, id_to_label)
        rows = rows.with_columns(
            pl.Series("oracle_pred_label", oracle_labels),
            pl.Series("oracle_correct", oracle_pred == y_true),
            pl.Series("oracle_nll_bits", np.asarray(oracle_metrics["nll_bits"], dtype=np.float32)),
        )
    return rows


def error_slice_table(predictions: pl.DataFrame, min_count: int = 5, max_slices: int = 80) -> pl.DataFrame:
    total = len(predictions)
    if total == 0:
        return pl.DataFrame()
    overall_error_rate = float((~predictions["correct"]).mean())
    rows = []
    for column in _slice_columns(predictions):
        values = predictions[column].to_numpy()
        bins = _slice_bins(column, values)
        for label in sorted(set(bins), key=str):
            mask = bins == label
            n = int(mask.sum())
            if n < min_count:
                continue
            errors = int((~predictions["correct"].to_numpy()[mask]).sum())
            error_rate = errors / n
            rows.append(
                {
                    "dimension": column,
                    "slice": str(label),
                    "n": n,
                    "errors": errors,
                    "accuracy": 1.0 - error_rate,
                    "error_rate": error_rate,
                    "overall_error_rate": overall_error_rate,
                    "error_lift": error_rate / overall_error_rate if overall_error_rate > 0 else 0.0,
                }
            )
    if not rows:
        return pl.DataFrame()
    return (
        pl.DataFrame(rows)
        .sort(["error_lift", "errors", "n"], descending=[True, True, True])
        .head(max_slices)
    )


def write_performance_explorer(
    path: Path,
    name: str,
    predictions: pl.DataFrame,
    error_slices: pl.DataFrame,
    confusion: np.ndarray,
    labels: Iterable[str],
    summary: Dict[str, float],
) -> None:
    labels = list(labels)
    path.parent.mkdir(parents=True, exist_ok=True)
    figures = [
        _confusion_figure(predictions, labels),
        _slice_figure(error_slices),
        _dimension_accuracy_figure(predictions),
        _metadata_histogram_figure(predictions),
        _confidence_figure(predictions),
        _high_confidence_errors_figure(predictions),
    ]
    divs = [_plot_div(fig, include_plotlyjs=i == 0) for i, fig in enumerate(figures)]
    cards = "".join(
        f"<div class='card'><div class='value'>{escape(value)}</div><div class='label'>{escape(label)}</div></div>"
        for label, value in _summary_cards(predictions, summary)
    )
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{escape(name)} performance explorer</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; background: #f8fafc; }}
h1 {{ margin: 0 0 16px; font-size: 28px; }}
h2 {{ margin: 28px 0 10px; font-size: 18px; }}
.cards {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 24px; }}
.card {{ background: white; border: 1px solid #d8dee9; border-radius: 6px; padding: 12px 16px; min-width: 130px; }}
.value {{ font-size: 22px; font-weight: 700; }}
.label {{ color: #5f6b7a; font-size: 12px; margin-top: 4px; }}
.plot {{ background: white; border: 1px solid #d8dee9; border-radius: 6px; padding: 8px; margin-bottom: 18px; }}
p {{ max-width: 900px; color: #52606d; }}
</style>
</head>
<body>
<h1>{escape(name)} Performance Explorer</h1>
<p>Metadata joined to per-example predictions. Use Plotly hover, zoom, legend toggles, and dropdowns to inspect error modes.</p>
<div class="cards">{cards}</div>
{''.join(f"<div class='plot'>{div}</div>" for div in divs)}
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def _labels(ids: np.ndarray, id_to_label: Dict[int, str]) -> list[str]:
    return [id_to_label[int(i)] for i in ids]


def _slice_columns(df: pl.DataFrame) -> list[str]:
    excluded_prefixes = ("source_",)
    result = []
    for name, dtype in df.schema.items():
        if name in EXCLUDED_SLICE_COLUMNS or name.startswith(excluded_prefixes):
            continue
        if name in {"correct", "confidence", "nll_bits", "true_probability", "top2_confidence", "pred_margin"}:
            continue
        if dtype in (pl.String, pl.Boolean) or dtype.is_numeric():
            result.append(name)
    return result


def _slice_bins(column: str, values: np.ndarray) -> np.ndarray:
    if values.dtype.kind in "biufc":
        x = values.astype(float)
        finite = np.isfinite(x)
        unique = np.unique(x[finite])
        if column == "snr_db":
            bins = np.floor(x / 4.0) * 4.0
            return np.asarray([f"{b:g}-{b + 4:g}" if np.isfinite(b) else "missing" for b in bins], dtype=object)
        if column in EXACT_NUMERIC_COLUMNS or len(unique) <= 16:
            return np.asarray(["missing" if not np.isfinite(v) else f"{v:g}" for v in x], dtype=object)
        if finite.sum() < 2:
            return np.asarray(["missing"] * len(values), dtype=object)
        edges = np.unique(np.quantile(x[finite], np.linspace(0.0, 1.0, 11)))
        if len(edges) < 2:
            return np.asarray([f"{x[finite][0]:g}"] * len(values), dtype=object)
        bin_ids = np.searchsorted(edges, x, side="right") - 1
        bin_ids = np.clip(bin_ids, 0, len(edges) - 2)
        return np.asarray(
            [
                "missing" if not np.isfinite(v) else f"{edges[i]:.4g}-{edges[i + 1]:.4g}"
                for v, i in zip(x, bin_ids)
            ],
            dtype=object,
        )
    return np.asarray(["missing" if v is None else str(v) for v in values], dtype=object)


def _summary_cards(predictions: pl.DataFrame, summary: Dict[str, float]) -> list[tuple[str, str]]:
    n = len(predictions)
    errors = int((~predictions["correct"]).sum()) if n else 0
    return [
        ("Examples", str(n)),
        ("Accuracy", f"{float(summary['accuracy']):.3f}"),
        ("Errors", str(errors)),
        ("Mean NLL bits", f"{float(predictions['nll_bits'].mean()):.3f}" if n else "nan"),
        ("ECE", f"{float(summary['ece']):.3f}"),
        ("MCE", f"{float(summary['mce']):.3f}"),
    ]


def _plot_div(fig: go.Figure, include_plotlyjs: bool = False) -> str:
    return pio.to_html(fig, include_plotlyjs=include_plotlyjs, full_html=False)


def _confusion_figure(predictions: pl.DataFrame, labels: list[str]) -> go.Figure:
    ranges = _snr_ranges(predictions)
    traces = []
    buttons = []
    for i, (range_label, mask) in enumerate(ranges):
        counts = _confusion_counts(predictions.filter(pl.Series(mask)), labels)
        z = _row_normalized(counts)
        traces.append(
            go.Heatmap(
                z=z,
                x=labels,
                y=labels,
                customdata=counts,
                zmin=0.0,
                zmax=1.0,
                colorscale="Blues",
                visible=i == 0,
                colorbar={"title": "row frac"} if i == 0 else None,
                hovertemplate=(
                    "true=%{y}<br>pred=%{x}<br>"
                    "row-normalized=%{z:.3f}<br>n=%{customdata}<extra></extra>"
                ),
            )
        )
        buttons.append(
            {
                "label": range_label,
                "method": "update",
                "args": [
                    {"visible": [j == i for j in range(len(ranges))]},
                    {"title": f"Confusion Matrix ({range_label})"},
                ],
            }
        )
    fig = go.Figure(traces)
    fig.update_layout(
        title=f"Confusion Matrix ({ranges[0][0]})",
        xaxis_title="Predicted",
        yaxis_title="True",
        yaxis={"autorange": "reversed"},
        height=560,
        updatemenus=[{"buttons": buttons, "direction": "down", "x": 1.0, "y": 1.14}],
    )
    return fig


def _snr_ranges(predictions: pl.DataFrame) -> list[tuple[str, np.ndarray]]:
    n = len(predictions)
    ranges = [("All SNR", np.ones(n, dtype=bool))]
    if "snr_db" not in predictions.columns or n == 0:
        return ranges
    snr = predictions["snr_db"].to_numpy()
    finite = np.isfinite(snr)
    bins = np.floor(snr[finite] / 4.0) * 4.0
    for start in sorted(np.unique(bins)):
        mask = finite & (np.floor(snr / 4.0) * 4.0 == start)
        ranges.append((f"SNR {start:g}-{start + 4:g} dB", mask))
    return ranges


def _confusion_counts(predictions: pl.DataFrame, labels: list[str]) -> np.ndarray:
    label_to_id = {label: i for i, label in enumerate(labels)}
    counts = np.zeros((len(labels), len(labels)), dtype=int)
    for true_label, pred_label in zip(predictions["true_label"].to_list(), predictions["pred_label"].to_list()):
        if true_label in label_to_id and pred_label in label_to_id:
            counts[label_to_id[true_label], label_to_id[pred_label]] += 1
    return counts


def _row_normalized(counts: np.ndarray) -> np.ndarray:
    row_sum = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, row_sum, out=np.zeros_like(counts, dtype=float), where=row_sum > 0)


def _slice_figure(error_slices: pl.DataFrame) -> go.Figure:
    if len(error_slices) == 0:
        return go.Figure().update_layout(title="Ranked Error Slices")
    df = error_slices.head(30).with_columns((pl.col("dimension") + ": " + pl.col("slice")).alias("label"))
    fig = go.Figure(
        go.Bar(
            x=df["error_lift"].to_list(),
            y=df["label"].to_list(),
            orientation="h",
            customdata=np.stack([df["n"].to_numpy(), df["errors"].to_numpy(), df["error_rate"].to_numpy()], axis=1),
            hovertemplate="lift=%{x:.2f}<br>n=%{customdata[0]}<br>errors=%{customdata[1]}<br>error rate=%{customdata[2]:.3f}<extra></extra>",
        )
    )
    fig.update_layout(title="Highest-Lift Error Slices", xaxis_title="Error Lift", height=760, yaxis={"autorange": "reversed"})
    return fig


def _dimension_accuracy_figure(predictions: pl.DataFrame) -> go.Figure:
    dims = _metadata_dimensions(predictions)
    if not dims:
        return go.Figure().update_layout(title="Accuracy by Metadata")
    traces = []
    buttons = []
    for i, dim in enumerate(dims):
        grouped = _accuracy_by_dimension(predictions, dim)
        traces.append(
            go.Bar(
                x=grouped["slice"].to_list(),
                y=grouped["accuracy"].to_list(),
                customdata=np.stack([grouped["n"].to_numpy(), grouped["errors"].to_numpy()], axis=1),
                visible=i == 0,
                hovertemplate=f"{dim}=%{{x}}<br>accuracy=%{{y:.3f}}<br>n=%{{customdata[0]}}<br>errors=%{{customdata[1]}}<extra></extra>",
            )
        )
        buttons.append(
            {
                "label": dim,
                "method": "update",
                "args": [
                    {"visible": [j == i for j in range(len(dims))]},
                    {"xaxis": {"title": dim}},
                ],
            }
        )
    fig = go.Figure(traces)
    fig.update_layout(
        title="Accuracy by Metadata Dimension",
        yaxis_title="Accuracy",
        xaxis_title=dims[0],
        height=460,
        updatemenus=[{"buttons": buttons, "direction": "down", "x": 1.0, "y": 1.18}],
    )
    return fig


def _metadata_histogram_figure(predictions: pl.DataFrame) -> go.Figure:
    dims = _metadata_dimensions(predictions)
    if not dims:
        return go.Figure().update_layout(title="Metadata Histogram")
    traces = []
    buttons = []
    for i, dim in enumerate(dims):
        visible = i == 0
        correct = predictions.filter(pl.col("correct"))[dim].to_list()
        errors = predictions.filter(~pl.col("correct"))[dim].to_list()
        if predictions.schema[dim].is_numeric():
            traces.extend(
                [
                    go.Histogram(x=correct, name="Correct", opacity=0.7, nbinsx=30, visible=visible),
                    go.Histogram(x=errors, name="Errors", opacity=0.7, nbinsx=30, visible=visible),
                ]
            )
        else:
            correct_counts = _value_counts(correct)
            error_counts = _value_counts(errors)
            categories = sorted(set(correct_counts) | set(error_counts), key=str)
            traces.extend(
                [
                    go.Bar(x=categories, y=[correct_counts.get(c, 0) for c in categories], name="Correct", visible=visible),
                    go.Bar(x=categories, y=[error_counts.get(c, 0) for c in categories], name="Errors", visible=visible),
                ]
            )
        buttons.append(
            {
                "label": dim,
                "method": "update",
                "args": [
                    {"visible": [j // 2 == i for j in range(2 * len(dims))]},
                    {"xaxis": {"title": dim}},
                ],
            }
        )
    fig = go.Figure(traces)
    fig.update_layout(
        title="Metadata Histogram",
        xaxis_title=dims[0],
        yaxis_title="Count",
        barmode="overlay",
        height=460,
        updatemenus=[{"buttons": buttons, "direction": "down", "x": 1.0, "y": 1.18}],
    )
    return fig


def _accuracy_by_dimension(predictions: pl.DataFrame, dim: str) -> pl.DataFrame:
    bins = _slice_bins(dim, predictions[dim].to_numpy())
    sort_keys = _slice_sort_keys(bins, predictions.schema[dim].is_numeric())
    return (
        predictions.with_columns(pl.Series("slice", bins), pl.Series("slice_sort_key", sort_keys))
        .group_by("slice")
        .agg(
            pl.col("slice_sort_key").min(),
            pl.len().alias("n"),
            (~pl.col("correct")).sum().alias("errors"),
            pl.col("correct").mean().alias("accuracy"),
        )
        .sort(["slice_sort_key", "slice"])
        .drop("slice_sort_key")
    )


def _slice_sort_keys(labels: np.ndarray, numeric: bool) -> np.ndarray:
    if not numeric:
        return np.zeros(len(labels), dtype=float)
    keys = []
    for label in labels:
        text = str(label)
        if text == "missing":
            keys.append(float("inf"))
            continue
        match = re.match(r"^-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", text, re.IGNORECASE)
        keys.append(float(match.group(0)) if match else float("inf"))
    return np.asarray(keys, dtype=float)


def _metadata_dimensions(predictions: pl.DataFrame) -> list[str]:
    preferred = (
        "snr_db",
        "osr",
        "symbol_period",
        "ebw",
        "sto",
        "cfo",
        "cpo",
        "channel",
        "modulation",
        "upsample_factor",
        "downsample_factor",
        "channel_n_taps",
        "channel_rms_delay_samples",
    )
    return [c for c in preferred if c in predictions.columns]


def _value_counts(values: list) -> dict:
    counts: dict = {}
    for value in values:
        key = "missing" if value is None else value
        counts[key] = counts.get(key, 0) + 1
    return counts


def _confidence_figure(predictions: pl.DataFrame) -> go.Figure:
    correct = predictions.filter(pl.col("correct"))["confidence"].to_list()
    errors = predictions.filter(~pl.col("correct"))["confidence"].to_list()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=correct, name="Correct", opacity=0.75, nbinsx=20))
    fig.add_trace(go.Histogram(x=errors, name="Errors", opacity=0.75, nbinsx=20))
    fig.update_layout(title="Confidence Distribution", xaxis_title="Confidence", yaxis_title="Count", barmode="overlay", height=420)
    return fig


def _high_confidence_errors_figure(predictions: pl.DataFrame) -> go.Figure:
    cols = [c for c in ("signal_id", "true_label", "pred_label", "confidence", "nll_bits", "snr_db", "osr", "symbol_period", "ebw", "sto", "cfo") if c in predictions.columns]
    rows = predictions.filter(~pl.col("correct")).sort("confidence", descending=True).head(25)
    values = [rows[c].to_list() for c in cols]
    fig = go.Figure(data=[go.Table(header={"values": cols}, cells={"values": values})])
    fig.update_layout(title="Highest-Confidence Errors", height=520)
    return fig
