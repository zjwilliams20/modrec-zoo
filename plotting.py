from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from simulator import (
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
