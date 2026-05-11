from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from modreczoo.simulation import (
    MODEMS,
    apply_srrc_filter,
    ber_sweep,
    generate_symbols,
    rng_from_seed,
)


def plot_modulation_summaries(
    modulations: Iterable[str],
    output_dir: str,
    k_symbols: int = 128,
    osr: int = 8,
    ebw: float = 0.35,
    ebn0_db: Sequence[float] = tuple(range(0, 21, 2)),
    ber_bits: int = 100_000,
    seed: Optional[int] = 0,
    show: bool = False,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = rng_from_seed(seed)

    for modulation in modulations:
        fig = plot_modulation_summary(
            modulation=modulation,
            k_symbols=k_symbols,
            osr=osr,
            ebw=ebw,
            ebn0_db=ebn0_db,
            ber_bits=ber_bits,
            rng=rng,
            seed=seed,
        )
        fig.savefig(output / f"{safe_filename(modulation)}_summary.png", dpi=160, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)


def plot_modulation_summary(
    modulation: str,
    k_symbols: int,
    osr: int,
    ebw: float,
    ebn0_db: Sequence[float],
    ber_bits: int,
    rng: np.random.Generator,
    seed: Optional[int],
) -> plt.Figure:
    symbols, _ = generate_symbols(modulation, k_symbols, rng)
    waveform = apply_srrc_filter(symbols, osr, ebw)
    waveform = waveform[: k_symbols * osr]
    freq, spectrum_db = normalized_spectrum(waveform)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"{modulation} waveform sanity view", fontsize=14)
    add_parameter_text(fig, k_symbols, osr, ebw, ebn0_db, ber_bits, seed)

    plot_constellation(axes[0, 0], modulation, symbols)
    plot_time_domain(axes[0, 1], waveform, osr, k_symbols)
    plot_frequency_domain(axes[1, 0], freq, spectrum_db)
    plot_ber(axes[1, 1], modulation, ebn0_db, ber_bits, seed)

    fig.tight_layout()
    return fig


def plot_constellation(ax: plt.Axes, modulation: str, symbols: np.ndarray) -> None:
    if modulation in MODEMS:
        points = MODEMS[modulation].points
        ax.scatter(np.real(points), np.imag(points), s=48, color="#1f77b4")
    else:
        ax.scatter(np.real(symbols), np.imag(symbols), s=16, alpha=0.7, color="#1f77b4")

    ax.axhline(0, color="0.75", linewidth=0.8)
    ax.axvline(0, color="0.75", linewidth=0.8)
    ax.set_title("Clean constellation")
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.set_aspect("equal", adjustable="box")
    set_equal_axis_limits(ax)
    ax.grid(True, alpha=0.25)


def plot_time_domain(ax: plt.Axes, waveform: np.ndarray, osr: int, k_symbols: int) -> None:
    t_symbols = np.arange(len(waveform)) / osr
    ax.plot(t_symbols, np.real(waveform), label="I", linewidth=1.0)
    ax.plot(t_symbols, np.imag(waveform), label="Q", linewidth=1.0)
    ax.set_title(f"Time domain, first {k_symbols} symbols")
    ax.set_xlabel("Time (symbols)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, min(k_symbols, t_symbols[-1] if len(t_symbols) else k_symbols))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")


def add_parameter_text(
    fig: plt.Figure,
    k_symbols: int,
    osr: int,
    ebw: float,
    ebn0_db: Sequence[float],
    ber_bits: int,
    seed: Optional[int],
) -> None:
    ebn0_values = list(ebn0_db)
    ebn0_text = f"{min(ebn0_values):g}-{max(ebn0_values):g} dB" if ebn0_values else "none"
    if len(ebn0_values) > 1:
        ebn0_text += f", {len(ebn0_values)} pts"
    text = (
        f"K symbols: {k_symbols}\n"
        f"OSR: {osr}\n"
        f"SRRC EBW: {ebw:g}\n"
        f"BER Eb/N0: {ebn0_text}\n"
        f"BER bits/point: {ber_bits}\n"
        f"Seed: {seed}"
    )
    fig.text(
        0.985,
        0.965,
        text,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.92},
    )


def set_equal_axis_limits(ax: plt.Axes, min_span: float = 2.4, padding: float = 0.18) -> None:
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    span = max(x_max - x_min, y_max - y_min, min_span)
    span *= 1 + padding
    half = span / 2
    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)


def plot_frequency_domain(ax: plt.Axes, freq: np.ndarray, spectrum_db: np.ndarray) -> None:
    ax.plot(freq, spectrum_db, linewidth=1.0)
    ax.set_title("Frequency domain")
    ax.set_xlabel("Normalized frequency (cycles/sample)")
    ax.set_ylabel("Magnitude (dB, normalized)")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(max(-100, float(np.nanmin(spectrum_db))), 5)
    ax.grid(True, alpha=0.25)


def plot_ber(
    ax: plt.Axes,
    modulation: str,
    ebn0_db: Sequence[float],
    ber_bits: int,
    seed: Optional[int],
) -> None:
    if modulation == "MSK":
        ax.axis("off")
        ax.text(0.5, 0.5, "MSK BER check not implemented", ha="center", va="center")
        return

    results = ber_sweep([modulation], ebn0_db, ber_bits, seed)
    x = results["ebn0_db"].to_numpy()
    empirical = np.maximum(results["empirical_ber"].to_numpy(), 0.5 / ber_bits)
    theory = np.maximum(results["theory_ber"].to_numpy(), 0.5 / ber_bits)

    ax.semilogy(x, empirical, marker="o", label="Empirical")
    ax.semilogy(x, theory, marker="s", label="Theory")
    ax.set_title("BER vs SNR")
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("BER")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right")


def normalized_spectrum(waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(waveform) == 0:
        return np.array([]), np.array([])

    window = np.hanning(len(waveform))
    spectrum = np.fft.fftshift(np.fft.fft(waveform * window, n=next_power_of_two(len(waveform))))
    magnitude = np.abs(spectrum)
    magnitude = magnitude / max(np.max(magnitude), np.finfo(float).eps)
    spectrum_db = 20 * np.log10(np.maximum(magnitude, np.finfo(float).eps))
    freq = np.fft.fftshift(np.fft.fftfreq(len(spectrum), d=1.0))
    return freq, spectrum_db


def next_power_of_two(n: int) -> int:
    return 1 << (max(1, n) - 1).bit_length()


def safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in value)


def plot_confusion_matrix(matrix: np.ndarray, labels: List[str], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = matrix / np.maximum(row_sums, 1)
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(f"{title} normalized confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_reliability_diagram(
    calib_df: pl.DataFrame,
    ece: float,
    mce: float,
    path: Path,
    title: str,
) -> None:
    lowers = calib_df["bin_lower"].to_numpy()
    uppers = calib_df["bin_upper"].to_numpy()
    mids = calib_df["bin_midpoint"].to_numpy()
    accuracy = calib_df["accuracy"].to_numpy(allow_copy=True)
    counts = calib_df["count"].to_numpy()
    bin_width = float(uppers[0] - lowers[0]) if len(uppers) > 0 else 0.1
    n_total = max(int(counts.sum()), 1)
    bar_w = bin_width * 0.8

    fig, (ax_cal, ax_hist) = plt.subplots(
        2, 1, figsize=(5, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    ax_cal.plot([0, 1], [0, 1], "--", color="0.5", linewidth=1.0)
    for mid, acc, n in zip(mids, accuracy, counts):
        if n == 0 or np.isnan(acc):
            continue
        ax_cal.bar(mid, acc, width=bar_w, color="#4C72B0", alpha=0.75, align="center")
        if acc < mid:
            ax_cal.bar(mid, mid - acc, width=bar_w, bottom=acc, color="#CC3333", alpha=0.4, align="center")
        elif acc > mid:
            ax_cal.bar(mid, acc - mid, width=bar_w, bottom=mid, color="#33AA55", alpha=0.4, align="center")

    ax_cal.set_ylabel("Accuracy")
    ax_cal.set_xlim(0, 1)
    ax_cal.set_ylim(0, 1)
    ax_cal.set_title(f"{title} reliability diagram\nECE={ece:.4f}  MCE={mce:.4f}")
    ax_cal.grid(True, alpha=0.25)
    ax_cal.legend(
        handles=[
            plt.Line2D([0], [0], linestyle="--", color="0.5", label="Perfect calibration"),
            mpatches.Patch(color="#4C72B0", alpha=0.75, label="Accuracy"),
            mpatches.Patch(color="#CC3333", alpha=0.6, label="Overconfident"),
            mpatches.Patch(color="#33AA55", alpha=0.6, label="Underconfident"),
        ],
        fontsize=8,
        loc="upper left",
    )

    ax_hist.bar(mids, counts / n_total, width=bar_w, color="#4C72B0", alpha=0.75, align="center")
    ax_hist.set_xlabel("Confidence")
    ax_hist.set_ylabel("Fraction")
    ax_hist.grid(True, alpha=0.25)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_calibration_by_snr(summary: pl.DataFrame, path: Path, title: str) -> None:
    x = summary["snr_bin_db"].to_numpy()
    acc = summary["accuracy"].to_numpy(allow_copy=True)
    conf = summary["mean_confidence"].to_numpy(allow_copy=True)
    ece = summary["ece"].to_numpy(allow_copy=True)

    fig, (ax_gap, ax_ece) = plt.subplots(2, 1, figsize=(7, 6), gridspec_kw={"height_ratios": [2, 1]}, sharex=True)

    ax_gap.plot(x, acc, marker="o", label="Accuracy", color="#4C72B0")
    ax_gap.plot(x, conf, marker="s", linestyle="--", label="Mean confidence", color="#DD8452")
    valid = ~(np.isnan(acc) | np.isnan(conf))
    over = valid & (conf > acc)
    under = valid & (acc >= conf)
    ax_gap.fill_between(x, acc, conf, where=over, interpolate=True, alpha=0.25, color="#CC3333", label="Overconfident")
    ax_gap.fill_between(x, acc, conf, where=under, interpolate=True, alpha=0.25, color="#33AA55", label="Underconfident")
    ax_gap.set_ylabel("Value")
    ax_gap.set_ylim(0, 1)
    ax_gap.set_title(f"{title} calibration by SNR")
    ax_gap.grid(True, alpha=0.25)
    ax_gap.legend(fontsize=8, loc="lower right")

    bar_w = (x[1] - x[0]) * 0.7 if len(x) > 1 else 2.0
    ax_ece.bar(x, np.where(np.isnan(ece), 0, ece), width=bar_w, color="#4C72B0", alpha=0.75, align="center")
    ax_ece.set_xlabel("SNR bin start (dB)")
    ax_ece.set_ylabel("|acc − conf|")
    ax_ece.set_ylim(0, None)
    ax_ece.grid(True, alpha=0.25)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_example_spectrograms(
    signals: np.ndarray,
    metadata: pl.DataFrame,
    output_dir: Path,
    n_per_class: int = 3,
    nperseg: int = 64,
    noverlap: int = 48,
    freq_bins: int = 64,
    time_bins: int = 64,
    window: str = "kaiser",
    window_beta: float = 15.0,
) -> None:
    from modreczoo.data import normalize_signal, spectrogram_channels

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    modulations = sorted(metadata["modulation"].unique().to_list())
    n_mods = len(modulations)

    fig_all, axes_all = plt.subplots(
        n_mods, n_per_class,
        figsize=(n_per_class * 2.5, n_mods * 2.2),
        squeeze=False,
    )
    fig_all.suptitle("Example spectrograms by modulation and SNR", fontsize=12, y=1.01)

    def _render(x: np.ndarray) -> np.ndarray:
        return spectrogram_channels(
            x, channel_format="mag",
            freq_bins=freq_bins, time_bins=time_bins,
            nperseg=nperseg, noverlap=noverlap,
            window=window, window_beta=window_beta,
        )[0]

    for row_idx, modulation in enumerate(modulations):
        mod_df = metadata.filter(pl.col("modulation") == modulation).sort("snr_db")
        n_total = len(mod_df)
        snr_vals = mod_df["snr_db"].to_numpy()
        signal_ids = mod_df["signal_id"].to_numpy()
        pick_idxs = [min(int(round(i * (n_total - 1) / max(n_per_class - 1, 1))), n_total - 1) for i in range(n_per_class)]

        fig_mod, axes_mod = plt.subplots(1, n_per_class, figsize=(n_per_class * 2.8, 3.0), squeeze=False)
        fig_mod.suptitle(modulation, fontsize=12)

        for col_idx, pick in enumerate(pick_idxs):
            snr = float(snr_vals[pick])
            spec = _render(normalize_signal(signals[int(signal_ids[pick])]))
            for ax in (axes_mod[0, col_idx], axes_all[row_idx, col_idx]):
                ax.imshow(spec, aspect="auto", origin="lower", cmap="inferno", interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"{snr:.1f} dB", fontsize=8)
            if col_idx == 0:
                axes_all[row_idx, 0].set_ylabel(modulation, fontsize=8)

        fig_mod.tight_layout()
        fig_mod.savefig(output_dir / f"{safe_filename(modulation)}.png", dpi=160, bbox_inches="tight")
        plt.close(fig_mod)

    fig_all.tight_layout()
    fig_all.savefig(output_dir / "overview.png", dpi=160, bbox_inches="tight")
    plt.close(fig_all)


_CHANNEL_NAMES = {
    "real_imag":           ["I", "Q"],
    "mag":                 ["Mag"],
    "mag_phase":           ["Mag", "Phase"],
    "mag_inst_freq":       ["Mag", "InstFreq"],
    "differential_complex":["Re(d)", "Im(d)"],
    "apf":                 ["LogMag", "cos(ph)", "sin(ph)", "InstFreq"],
    "complex_powers":      ["Re(x)", "Im(x)", "Re(x²)", "Im(x²)", "Re(x⁴)", "Im(x⁴)"],
    "multilag":            ["Re(lag1)", "Im(lag1)", "Re(lag4)", "Im(lag4)", "Re(lag16)", "Im(lag16)"],
    "cyclic_caf":          ["CAF lag1", "CAF lag4", "CAF lag16"],
    "scf":                 ["SCF"],
}

# Max samples to plot per 1-D channel (keeps subplots compact).
_MAX_PLOT_SAMPLES = 512


def plot_input_examples(
    loader,
    id_to_label: dict,
    representation: str,
    channel_format: str,
    path: Path,
) -> None:
    """Plot one example per class as the model receives it, log-scale clipped for spectrograms."""
    n_classes = len(id_to_label)
    examples: dict = {}
    for xb, yb in loader:
        for x, y in zip(xb, yb):
            cls = int(y.item())
            if cls not in examples:
                examples[cls] = x.cpu().numpy()
            if len(examples) == n_classes:
                break
        if len(examples) == n_classes:
            break

    present = [i for i in range(n_classes) if i in examples]
    n_channels = next(iter(examples.values())).shape[0]
    ch_names = _CHANNEL_NAMES.get(channel_format, [f"ch{i}" for i in range(n_channels)])
    # Pad or trim if channel count drifts from the table.
    ch_names = list(ch_names[:n_channels]) + [f"ch{i}" for i in range(len(ch_names), n_channels)]

    is_2d = representation == "spectrogram"
    n_rows, n_cols = len(present), n_channels
    cell_w, cell_h = (2.4, 2.0) if is_2d else (3.2, 1.2)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * cell_w, n_rows * cell_h), squeeze=False)
    fig.suptitle(f"Input examples — {channel_format}", fontsize=10)

    for row_idx, cls_id in enumerate(present):
        x = examples[cls_id]
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            ch = x[col_idx]
            if is_2d:
                ax.imshow(ch, aspect="auto", origin="lower", cmap="inferno", interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                sig = ch[:_MAX_PLOT_SAMPLES]
                ax.plot(sig, linewidth=0.5, color="#1f77b4")
                ax.set_xlim(0, len(sig) - 1)
                ax.tick_params(labelsize=5)
                ax.grid(True, alpha=0.2)
            if row_idx == 0:
                ax.set_title(ch_names[col_idx], fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(id_to_label[cls_id], fontsize=7, rotation=0, ha="right", labelpad=40)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_by_snr(summary: pl.DataFrame, path: Path, title: str) -> None:
    x = summary["snr_bin_db"].to_numpy()
    y = summary["accuracy"].to_numpy()
    n = summary["n"].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y, marker="o")
    for xi, yi, ni in zip(x, y, n):
        ax.text(xi, yi, str(int(ni)), fontsize=7, ha="center", va="bottom")
    ax.set_title(f"{title} accuracy versus SNR")
    ax.set_xlabel("SNR bin start (dB)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
