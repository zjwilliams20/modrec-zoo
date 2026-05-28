from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import scipy.ndimage
import scipy.signal as signal
import torch
from torch.utils.data import DataLoader, Dataset

from modreczoo.models import representation_for_model


SIGNALS_FILE = "signals.npy"
EXTRAS_FILE = "extras.npz"
METADATA_FILE = "metadata.parquet"
README_MODULATION_ORDER = ("2PSK", "4PSK", "8PSK", "pi/4-DQPSK", "16QAM", "64QAM", "256QAM", "MSK")
CSP_LAGS = (1, 4, 16)


def save_dataset(
    output_dir: str,
    signals: np.ndarray,
    metadata: pl.DataFrame,
    extras: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / SIGNALS_FILE, signals)
    if extras:
        np.savez(output / EXTRAS_FILE, **extras)
    metadata.write_parquet(output / METADATA_FILE)


def load_dataset(output_dir: str) -> Tuple[np.ndarray, pl.DataFrame]:
    output = Path(output_dir)
    signals = np.load(output / SIGNALS_FILE, mmap_mode="r")
    metadata = pl.read_parquet(output / METADATA_FILE)
    metadata = ensure_symbol_metadata(metadata)
    return signals, metadata


def ensure_symbol_metadata(metadata: pl.DataFrame) -> pl.DataFrame:
    if "symbol_period" not in metadata.columns:
        metadata = metadata.with_columns(pl.lit(1).alias("symbol_period"))
    if "symbol_rate" not in metadata.columns and {"osr", "symbol_period"} <= set(metadata.columns):
        metadata = metadata.with_columns((1.0 / (pl.col("symbol_period").cast(pl.Float64) * pl.col("osr").cast(pl.Float64))).alias("symbol_rate"))
    elif "symbol_rate" not in metadata.columns and {"upsample_factor", "downsample_factor", "symbol_period"} <= set(metadata.columns):
        metadata = metadata.with_columns(
            (
                pl.col("downsample_factor").cast(pl.Float64)
                / (pl.col("symbol_period").cast(pl.Float64) * pl.col("upsample_factor").cast(pl.Float64))
            ).alias("symbol_rate")
        )
    return metadata


class ModrecDataset(Dataset):
    def __init__(
        self,
        signals: np.ndarray,
        metadata: pl.DataFrame,
        indices: np.ndarray,
        label_to_id: Dict[str, int],
        representation: str,
        channel_format: str,
        remove_cfo: bool,
        cfo_estimator: str,
        spectrogram_freq_bins: int = 64,
        spectrogram_time_bins: int = 64,
        spectrogram_nperseg: int = 64,
        spectrogram_noverlap: int = 48,
        spectrogram_window: str = "hann",
        n_samples: Optional[int] = None,
    ) -> None:
        self.signals = signals
        self.metadata = metadata
        self.indices = indices.astype(np.int64)
        self.label_to_id = label_to_id
        self.representation = representation
        self.channel_format = channel_format
        self.remove_cfo = remove_cfo
        self.cfo_estimator = cfo_estimator
        self.spectrogram_freq_bins = spectrogram_freq_bins
        self.spectrogram_time_bins = spectrogram_time_bins
        self.spectrogram_nperseg = spectrogram_nperseg
        self.spectrogram_noverlap = spectrogram_noverlap
        self.spectrogram_window = spectrogram_window
        self.n_samples = n_samples
        self.labels = metadata["modulation"].to_numpy()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = int(self.indices[item])
        x = normalize_signal(self.signals[idx])
        if self.n_samples is not None:
            if x.shape[0] > self.n_samples:
                x = x[: self.n_samples]
            elif x.shape[0] < self.n_samples:
                x = np.pad(x, (0, self.n_samples - x.shape[0]), mode="constant")
        if self.remove_cfo:
            x = remove_empirical_cfo(x, estimator=self.cfo_estimator)
        y = self.label_to_id[str(self.labels[idx])]

        if self.representation == "time":
            features = complex_channels(x, self.channel_format)
        elif self.representation == "frequency":
            features = frequency_channels(x, self.channel_format)
        elif self.representation == "spectrogram":
            features = spectrogram_channels(
                x,
                channel_format=self.channel_format,
                freq_bins=self.spectrogram_freq_bins,
                time_bins=self.spectrogram_time_bins,
                nperseg=self.spectrogram_nperseg,
                noverlap=self.spectrogram_noverlap,
                window=self.spectrogram_window,
            )
        elif self.representation == "iq_features":
            features = iq_features(x)
        elif self.representation == "csp_features":
            features = csp_expert_features(x)
        elif self.representation == "csp_canonical":
            features = csp_canonical_features(x)
        else:
            raise ValueError(f"Unsupported representation: {self.representation}")

        return torch.from_numpy(features).float(), torch.tensor(y, dtype=torch.long)


def get_data_loader(
    signals: np.ndarray,
    metadata: pl.DataFrame,
    indices: np.ndarray,
    label_to_id: Dict[str, int],
    model_name: str,
    channel_format: str = "real_imag",
    remove_cfo: bool = False,
    cfo_estimator: str = "lag_correlation",
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,
    spectrogram_freq_bins: int = 64,
    spectrogram_time_bins: int = 64,
    spectrogram_nperseg: int = 64,
    spectrogram_noverlap: int = 48,
    spectrogram_window: str = "hann",
    pin_memory: bool = True,
    persistent_workers: bool = True,
    n_samples: Optional[int] = None,
    **loader_kwargs,
) -> DataLoader:
    dataset = ModrecDataset(
        signals=signals,
        metadata=metadata,
        indices=indices,
        label_to_id=label_to_id,
        representation=representation_for_model(model_name),
        channel_format=channel_format,
        remove_cfo=remove_cfo,
        cfo_estimator=cfo_estimator,
        spectrogram_freq_bins=spectrogram_freq_bins,
        spectrogram_time_bins=spectrogram_time_bins,
        spectrogram_nperseg=spectrogram_nperseg,
        spectrogram_noverlap=spectrogram_noverlap,
        spectrogram_window=spectrogram_window,
        n_samples=n_samples,
    )
    if num_workers > 0:
        loader_kwargs.setdefault("persistent_workers", persistent_workers)
    loader_kwargs.setdefault("pin_memory", pin_memory)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **loader_kwargs)


def load_dataset_loader(
    dataset_dir: str | Path,
    model_name: str,
    indices: np.ndarray | None = None,
    label_to_id: Dict[str, int] | None = None,
    channel_format: str = "real_imag",
    remove_cfo: bool = False,
    cfo_estimator: str = "lag_correlation",
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,
    spectrogram_freq_bins: int = 64,
    spectrogram_time_bins: int = 64,
    spectrogram_nperseg: int = 64,
    spectrogram_noverlap: int = 48,
    spectrogram_window: str = "hann",
    pin_memory: bool = True,
    persistent_workers: bool = True,
    **loader_kwargs,
) -> Tuple[DataLoader, np.ndarray, pl.DataFrame, Dict[str, int]]:
    signals, metadata = load_dataset(str(dataset_dir))
    if indices is None:
        indices = np.arange(signals.shape[0], dtype=np.int64)
    if label_to_id is None:
        labels = ordered_modulation_labels(metadata["modulation"].unique().to_list())
        label_to_id = {label: idx for idx, label in enumerate(labels)}

    loader = get_data_loader(
        signals=signals,
        metadata=metadata,
        indices=indices,
        label_to_id=label_to_id,
        model_name=model_name,
        channel_format=channel_format,
        remove_cfo=remove_cfo,
        cfo_estimator=cfo_estimator,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        spectrogram_freq_bins=spectrogram_freq_bins,
        spectrogram_time_bins=spectrogram_time_bins,
        spectrogram_nperseg=spectrogram_nperseg,
        spectrogram_noverlap=spectrogram_noverlap,
        spectrogram_window=spectrogram_window,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        **loader_kwargs,
    )
    return loader, signals, metadata, label_to_id


def ordered_modulation_labels(observed_labels: List[str]) -> List[str]:
    labels = [label for label in README_MODULATION_ORDER if label in observed_labels]
    labels.extend(sorted(label for label in observed_labels if label not in labels))
    return labels


def normalize_signal(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.complex64, copy=False)
    x = x - np.mean(x)
    scale = np.sqrt(np.mean(np.abs(x) ** 2))
    return x / max(scale, np.finfo(np.float32).eps)


def remove_empirical_cfo(x: np.ndarray, estimator: str = "lag_correlation") -> np.ndarray:
    cfo = estimate_cfo(x, estimator=estimator)
    n = np.arange(len(x))
    return x * np.exp(-1j * 2 * np.pi * cfo * n)


def estimate_cfo(x: np.ndarray, estimator: str = "lag_correlation") -> float:
    if estimator == "lag_correlation":
        return estimate_cfo_lag_correlation(x)
    if estimator == "phase_slope":
        return estimate_cfo_phase_slope(x)
    if estimator == "spectral_centroid":
        return estimate_cfo_spectral_centroid(x)
    raise ValueError(f"Unsupported CFO estimator: {estimator}")


def estimate_cfo_lag_correlation(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    corr = np.sum(x[1:] * np.conj(x[:-1]))
    return float(np.angle(corr) / (2 * np.pi))


def estimate_cfo_phase_slope(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    phase = np.unwrap(np.angle(x))
    n = np.arange(len(x), dtype=np.float64)
    weights = np.abs(x).astype(np.float64)
    weights = weights / max(np.mean(weights), np.finfo(float).eps)
    n_centered = n - np.average(n, weights=weights)
    phase_centered = phase - np.average(phase, weights=weights)
    denom = np.sum(weights * n_centered**2)
    if denom <= np.finfo(float).eps:
        return 0.0
    slope = np.sum(weights * n_centered * phase_centered) / denom
    return float(slope / (2 * np.pi))


def estimate_cfo_spectral_centroid(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    window = np.hanning(len(x))
    spectrum = np.fft.fftshift(np.fft.fft(x * window))
    power = np.abs(spectrum) ** 2
    total_power = np.sum(power)
    if total_power <= np.finfo(float).eps:
        return 0.0
    freq = np.fft.fftshift(np.fft.fftfreq(len(spectrum), d=1.0))
    return float(np.sum(freq * power) / total_power)


def complex_channels(x: np.ndarray, channel_format: str) -> np.ndarray:
    if channel_format == "real_imag":
        return np.stack((np.real(x), np.imag(x))).astype(np.float32)
    if channel_format == "mag_phase":
        mag = normalized_log_magnitude(x)
        phase = np.angle(x) / np.pi
        return np.stack((mag, phase)).astype(np.float32)
    if channel_format == "differential_complex":
        return differential_complex_channels(x)
    if channel_format == "apf":
        return apf_channels(x)
    if channel_format == "complex_powers":
        return complex_powers_channels(x)
    if channel_format == "multilag":
        return multilag_channels(x)
    if channel_format == "cyclic_caf":
        return cyclic_caf_channels(x)
    raise ValueError(f"Unsupported channel format: {channel_format}")


def differential_complex_channels(x: np.ndarray) -> np.ndarray:
    d = x[1:] * np.conj(x[:-1])
    d = np.concatenate([d[:1], d])
    scale = max(np.sqrt(np.mean(np.abs(d) ** 2)), np.finfo(np.float32).eps)
    return np.stack((np.real(d) / scale, np.imag(d) / scale)).astype(np.float32)


def complex_powers_channels(x: np.ndarray) -> np.ndarray:
    """6-channel encoding of orders 1, 2, 4: [Re(x), Im(x), Re(x²), Im(x²), Re(x⁴), Im(x⁴)].

    Each pair is RMS-normalized independently. x^M removes phase modulation at order M,
    so BPSK/MSK features concentrate in x², and QAM-order cumulants appear in x⁴.
    """
    eps = np.finfo(np.float32).eps

    def _norm_pair(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        scale = max(np.sqrt(np.mean(np.abs(z) ** 2)), eps)
        return (np.real(z) / scale).astype(np.float32), (np.imag(z) / scale).astype(np.float32)

    r1, i1 = _norm_pair(x)
    r2, i2 = _norm_pair(x ** 2)
    r4, i4 = _norm_pair(x ** 4)
    return np.stack((r1, i1, r2, i2, r4, i4))


def lag_product(x: np.ndarray, lag: int) -> np.ndarray:
    delayed = np.zeros_like(x)
    delayed[lag:] = x[:-lag]
    return x * np.conj(delayed)


def multilag_channels(x: np.ndarray, lags: Tuple[int, ...] = CSP_LAGS) -> np.ndarray:
    """Multi-lag conjugate products as real/imag channel pairs.

    Extends differential-complex features to several lags. For each lag tau,
    computes z[n] * z*[n - tau], whose angle is the phase change over tau samples.
    """
    channels = []
    eps = np.finfo(np.float32).eps
    for lag in lags:
        prod = lag_product(x, lag)
        scale = max(np.sqrt(np.mean(np.abs(prod) ** 2)), eps)
        channels.extend((np.real(prod) / scale, np.imag(prod) / scale))
    return np.stack(channels).astype(np.float32)


def cyclic_caf_channels(x: np.ndarray, lags: Tuple[int, ...] = CSP_LAGS) -> np.ndarray:
    """CAF magnitude spectra for several lags.

    For each lag tau, computes the DFT magnitude of z[n] * z*[n - tau]. The
    spectra are independently max-normalized and stacked as channels.
    """
    spectra = []
    eps = np.finfo(np.float32).eps
    for lag in lags:
        prod = lag_product(x, lag)
        r_alpha = np.abs(np.fft.fft(prod))
        spectra.append(r_alpha / max(np.max(r_alpha), eps))
    return np.stack(spectra).astype(np.float32)


def apf_channels(x: np.ndarray) -> np.ndarray:
    mag = normalized_log_magnitude(x)
    cos_phase = np.cos(np.angle(x)).astype(np.float32)
    sin_phase = np.sin(np.angle(x)).astype(np.float32)
    inst_freq = instantaneous_frequency(x)
    return np.stack((mag, cos_phase, sin_phase, inst_freq))


def normalized_log_magnitude(x: np.ndarray) -> np.ndarray:
    mag = np.log1p(np.abs(x))
    return ((mag - np.mean(mag)) / max(np.std(mag), np.finfo(np.float32).eps)).astype(np.float32)


def instantaneous_frequency(x: np.ndarray) -> np.ndarray:
    phase = np.unwrap(np.angle(x))
    inst_freq = np.diff(phase, prepend=phase[0]) / np.pi
    return np.nan_to_num(inst_freq).astype(np.float32)


def frequency_channels(x: np.ndarray, channel_format: str) -> np.ndarray:
    spectrum = np.fft.fftshift(np.fft.fft(x))
    spectrum = spectrum / max(np.sqrt(np.mean(np.abs(spectrum) ** 2)), np.finfo(np.float32).eps)
    return complex_channels(spectrum, channel_format)


def _parse_window(window: str) -> str | tuple:
    if ":" in window:
        name, beta = window.split(":", 1)
        return (name, float(beta))
    return window


def spectrogram_channels(
    x: np.ndarray,
    channel_format: str,
    freq_bins: int = 64,
    time_bins: int = 64,
    nperseg: int = 64,
    noverlap: int = 48,
    window: str = "hann",
) -> np.ndarray:
    if freq_bins < nperseg:
        raise ValueError("spectrogram_freq_bins must be at least spectrogram_nperseg.")
    if time_bins < 1:
        raise ValueError("spectrogram_time_bins must be positive.")
    if noverlap >= nperseg:
        raise ValueError("spectrogram_noverlap must be less than spectrogram_nperseg.")
    scipy_window = _parse_window(window)
    _, _, zxx = signal.spectrogram(
        x,
        window=scipy_window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=freq_bins,
        detrend=False,
        return_onesided=False,
        scaling="spectrum",
        mode="complex",
    )
    zxx = np.fft.fftshift(zxx, axes=0)
    if channel_format == "real_imag":
        real = resize_2d(np.real(zxx), freq_bins, time_bins)
        imag = resize_2d(np.imag(zxx), freq_bins, time_bins)
        scale = max(np.sqrt(np.mean(real**2 + imag**2)), np.finfo(np.float32).eps)
        return np.stack((real / scale, imag / scale)).astype(np.float32)
    if channel_format == "mag_phase":
        mag = np.log1p(np.abs(zxx))
        phase = np.angle(zxx)
        mag = resize_2d(mag, freq_bins, time_bins)
        phase = resize_2d(phase, freq_bins, time_bins)
        mag = (mag - np.mean(mag)) / max(np.std(mag), np.finfo(np.float32).eps)
        phase = phase / np.pi
        return np.stack((mag, phase)).astype(np.float32)
    if channel_format == "complex_powers":
        # Apply power transforms in time domain, then compute per-channel mag spectrogram.
        # x^M removes phase modulation at order M (e.g. x² → BPSK carrier at 2f_c).
        eps = np.finfo(np.float32).eps
        out = []
        for power in (1, 2, 4):
            xp = x ** power
            _, _, zp = signal.spectrogram(
                xp,
                window=scipy_window,
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=freq_bins,
                detrend=False,
                return_onesided=False,
                scaling="spectrum",
                mode="complex",
            )
            zp = np.fft.fftshift(zp, axes=0)
            for part in (np.real(zp), np.imag(zp)):
                ch = resize_2d(part, freq_bins, time_bins).astype(np.float32)
                scale = max(np.sqrt(np.mean(ch ** 2)), eps)
                out.append(ch / scale)
        return np.stack(out)
    if channel_format == "scf":
        return scf_channels(x, n_alpha=time_bins, n_freq=freq_bins, nperseg=nperseg)
    raise ValueError(f"Unsupported channel format: {channel_format}")


def resize_2d(x: np.ndarray, rows: int, cols: int) -> np.ndarray:
    if x.shape == (rows, cols):
        return x
    zoom_factors = (rows / x.shape[0], cols / x.shape[1])
    return scipy.ndimage.zoom(x, zoom_factors, order=1)


def scf_channels(
    x: np.ndarray,
    n_alpha: int = 64,
    n_freq: int = 64,
    nperseg: int = 64,
) -> np.ndarray:
    """Spectral Correlation Function (SCF) as a (1, n_alpha, n_freq) image.

    Computes S^alpha(f) via STFT cross-correlation: for each cyclic-frequency
    offset Δk ∈ [0, n_alpha), S[Δk, k] = mean_t(X[k+Δk, t] · X*[k, t]).
    The magnitude |S| encodes the cyclostationary footprint of the modulation.

    Citation:
        Roberts, R.S., et al. "Computationally Efficient Algorithms for Cyclic
        Spectral Analysis." IEEE Signal Processing Magazine, vol. 8, no. 2,
        1991, pp. 38–49. https://doi.org/10.1109/79.81008
    """
    noverlap = nperseg * 3 // 4
    _, _, zxx = signal.spectrogram(
        x,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=n_freq,
        detrend=False,
        return_onesided=False,
        scaling="spectrum",
        mode="complex",
    )
    # zxx: (n_freq, n_time)
    n_f, n_t = zxx.shape
    scf = np.zeros((n_alpha, n_f), dtype=np.float32)
    for dk in range(n_alpha):
        shifted = np.roll(zxx, -dk, axis=0)
        scf[dk] = np.abs(np.mean(zxx * np.conj(shifted), axis=1))
    scf = (scf - np.mean(scf)) / max(np.std(scf), np.finfo(np.float32).eps)
    return scf[np.newaxis]


N_CSP_CANONICAL_FEATURES = 13
N_CSP_EXPERT_FEATURES = 107


def csp_canonical_features(x: np.ndarray) -> np.ndarray:
    """Canonical CSP feature vector (13 features) strictly from the AMC literature.

    Implements the normalized higher-order cumulants and differential-phase
    moments of Swami & Sadler (2000) plus amplitude moment statistics used in
    Dobre et al. (2007). These features have closed-form theoretical values
    per modulation type that are provably invariant to AWGN, unknown signal
    amplitude, and carrier phase offset.

    Features:
        [0]  |C₂₀| — conjugate 2nd cumulant; BPSK≈1, all others≈0
        [1]  |C₄₀| — BPSK 2.0, QPSK 1.0, 8PSK≈0, QAM 0.6–0.7
        [2]  C₄₂  — excess kurtosis proxy; PSK negative, more so for QAM
        [3]  |C₄₁| — near-0 for symmetric constellations
        [4]  M₄₂  — amplitude kurtosis (constant-envelope PSK/MSK: ≈ 1)
        [5]  σ(|x|)/μ(|x|) — amplitude variation coefficient (PSK/MSK ≈ 0)
        [6]  M₆₃  — 6th amplitude moment (monotone in QAM order)
        [7]  M₈₄  — 8th amplitude moment (separates 64/256-QAM)
        [8]  |E[d²]| — 2-fold phase symmetry; BPSK ≈ 1
        [9]  |E[d⁴]| — 4-fold; QPSK ≈ 1, π/4-DQPSK < 1
        [10] |E[d⁸]| — 8-fold; 8PSK & π/4-DQPSK high
        [11] σ(Δφ)/μ(|Δφ|) — IF regularity; MSK ≈ 0 (constant IF)
        [12] max|FFT(x²)|/mean — conjugate spectral peak (BPSK line at 2fₓ)

    Citations:
        Swami, A. & Sadler, B. M. "Hierarchical Digital Modulation Classification
            Using Cumulants." IEEE Trans. Commun., 2000.
        Dobre, O. A. et al. "Survey of automatic modulation classification
            techniques." IET Commun., 1(2), 137-156, 2007.
    """
    # CFO removal is required for conjugate cumulants: E[x^k] rotates at k×δ
    # cycles/sample and time-averaging with N=4096 causes severe attenuation.
    x = remove_empirical_cfo(x - np.mean(x))
    x = x / np.sqrt(max(float(np.mean(np.abs(x) ** 2)), 1e-10))

    # Moments (m21 == 1 after normalization above)
    m20 = np.mean(x ** 2)
    m40 = np.mean(x ** 4)
    m41 = np.mean(x ** 3 * x.conj())
    m42 = float(np.mean(np.abs(x) ** 4))
    c40 = m40 - 3 * m20 ** 2
    c41 = m41 - 3 * m20        # m21 == 1
    c42 = float((m42 - abs(m20) ** 2 - 2.0).real)

    amp = np.abs(x)
    raw_d = x[1:] * x[:-1].conj()
    d = raw_d / (np.abs(raw_d) + 1e-10)
    dphi = np.diff(np.unwrap(np.angle(x)))
    x2_spec = np.abs(np.fft.fft(x ** 2))
    half = x2_spec[1 : len(x2_spec) // 2]

    feats = np.array([
        float(abs(m20)),                                        # |C₂₀|
        float(abs(c40)),                                        # |C₄₀|
        c42,                                                    # C₄₂
        float(abs(c41)),                                        # |C₄₁|
        m42,                                                    # M₄₂
        float(amp.std() / (amp.mean() + 1e-10)),               # σ/μ amp
        float(np.mean(amp ** 6)),                               # M₆₃
        float(np.mean(amp ** 8)),                               # M₈₄
        float(abs(np.mean(d ** 2))),                            # |E[d²]|
        float(abs(np.mean(d ** 4))),                            # |E[d⁴]|
        float(abs(np.mean(d ** 8))),                            # |E[d⁸]|
        float(dphi.std() / (np.abs(dphi).mean() + 1e-10)),     # IF regularity
        float(half.max() / (half.mean() + 1e-10)),              # conjugate spectral peak
    ], dtype=np.float32)
    return np.nan_to_num(feats)


def _psd_bandwidth_T_rough(x: np.ndarray) -> tuple[float, float]:
    """Estimate rough symbol period from 90% power bandwidth of Welch PSD.

    Returns (T_rough, bw_90_onesided) where T_rough is in samples/symbol.
    The 90% bandwidth of an SRRC signal ≈ Rs*(1 + 0.5*beta), so
    T_rough = 1 / Rs_rough with ~1.25x error due to unknown beta in [0,1].
    """
    from scipy.signal import welch as _welch
    _, psd = _welch(x, fs=1.0, nperseg=512, return_onesided=False)
    psd = np.abs(psd)
    half_N = len(psd) // 2
    # Accumulate power from DC outward (pair symmetric bins)
    center_psd = np.zeros(half_N + 1)
    center_psd[0] = psd[0]
    for k in range(1, half_N + 1):
        center_psd[k] = psd[k] + (psd[-k] if k < len(psd) - k else psd[k])
    cumsum = np.cumsum(center_psd)
    idx_90 = int(np.searchsorted(cumsum, 0.90 * cumsum[-1]))
    bw_90 = max(idx_90, 1) / len(psd)                   # one-sided, normalized
    Rs_rough = 2.0 * bw_90 / 1.25                       # assume beta ≈ 0.5
    T_rough = float(np.clip(1.0 / max(Rs_rough, 1.0 / 128), 1.0, 128.0))
    return T_rough, float(bw_90)


def csp_expert_features(x: np.ndarray) -> np.ndarray:
    """Extended 70-feature set for modulation recognition.

    Augments csp_canonical_features() (Groups 1, 4, 5, part of 7) with
    domain-informed signal statistics that are NOT strictly derivable from
    cyclostationary theory:
      • Amplitude distribution: quantile ladder, fine-grained CDF thresholds,
        short-lag ACF (Groups 2)
      • Absolute phase concentration E[ph^k] (Group 3)
      • Signed differential-phase real parts Re(E[d^k]) (Group 4 extension)
      • IF kurtosis, lag-1 ACF, 8-bin raw histogram (Group 5 extension)
      • Symbol-rate-normalized IF: blind T estimate from 90% PSD bandwidth
        reduces OSR-induced variance (Group 6)
      • Spectral: signal + amplitude Wiener entropy (Group 7)
      • Blind symbol timing (BST): brute-force search over t0 ∈ [0, T_int)
        maximizing |E[d^4]| at lag T — captures PSK-order phase symmetry
        at the approximate symbol rate (Group 8)
      • High-order complex moments |E[x^6]|, |E[x^8]|: monotone with QAM
        order (16QAM < 64QAM < 256QAM) due to different symbol alphabet
        amplitude statistics (Group 9)

    The non-canonical additions are empirically important (amplitude lag-1 ACF
    is the #2 most discriminative feature by RF importance; the amplitude CDF
    thresholds capture QAM constellation inner-ring density) but their
    distributions shift with OSR and excess-bandwidth β.
    """
    from scipy.signal import welch as _welch
    from scipy.stats import kurtosis as _kurtosis

    # CFO removal + RMS normalization (m21 == 1 after this)
    x = remove_empirical_cfo(x - np.mean(x))
    x = x / np.sqrt(max(float(np.mean(np.abs(x) ** 2)), 1e-10))

    feats: list[float] = []

    # ── Group 1: Conjugate cumulants (4 features) ─────────────────────────
    # E[x^k] rotates at k×δ; CFO removal above makes these valid.
    m20 = np.mean(x ** 2)
    m40 = np.mean(x ** 4)
    m41 = np.mean(x ** 3 * x.conj())
    m42 = float(np.mean(np.abs(x) ** 4))
    c40 = m40 - 3 * m20 ** 2
    c41 = m41 - 3 * m20            # m21 == 1
    c42 = float((m42 - abs(m20) ** 2 - 2.0).real)
    feats += [
        float(abs(m20)),            # |C₂₀|: BPSK≈1, others≈0
        float(abs(c40)),            # |C₄₀|: BPSK 2.0, QPSK 1.0, 8PSK≈0, QAM 0.6-0.7
        c42,                        # C₄₂: PSK/QAM negative, more negative for higher QAM
        float(abs(c41)),            # |C₄₁|: near-0 for symmetric constellations
    ]

    # ── Group 2: Amplitude distribution (24 features) ─────────────────────
    amp = np.abs(x)
    m63 = float(np.mean(amp ** 6))
    m84 = float(np.mean(amp ** 8))
    feats += [
        m42,                                            # amplitude kurtosis
        float(amp.std() / (amp.mean() + 1e-10)),       # variation coeff (PSK/MSK ≈ 0)
        m63,                                            # 6th moment (monotone in QAM order)
        m84,                                            # 8th moment (separates 64/256-QAM)
    ]
    # Quantiles (6)
    feats += list(np.quantile(amp, [0.10, 0.25, 0.50, 0.75, 0.90, 0.95]).astype(float))
    # CDF thresholds: inner-ring density captures QAM order (9)
    # 16QAM inner: ≈0.45, 64QAM: ≈0.22, 256QAM: ≈0.11 (RMS-normalized)
    for thr in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]:
        feats.append(float(np.mean(amp < thr)))
    # High-amplitude tail (2)
    feats += [float(np.mean(amp > 1.4)), float(np.mean(amp > 1.8))]
    # Amplitude ACF at short lags (3): lag-1 is the #2 most important feature
    amp_c = amp - amp.mean()
    amp_var = float(np.mean(amp_c ** 2)) + 1e-10
    for lag in [1, 2, 4]:
        feats.append(float(np.mean(amp_c[lag:] * amp_c[:-lag]) / amp_var))

    # ── Group 3: Absolute phase concentration (2 features) ────────────────
    ph = x / (amp + 1e-10)                             # unit-magnitude phasor
    feats += [float(abs(np.mean(ph ** 2))), float(abs(np.mean(ph ** 4)))]

    # ── Group 4: Differential phase (6 features) ──────────────────────────
    raw_d = x[1:] * x[:-1].conj()
    d = raw_d / (np.abs(raw_d) + 1e-10)               # unit-normalized phase increment
    for k in [2, 4, 8]:
        dk = d ** k
        feats += [float(abs(np.mean(dk))), float(np.mean(dk).real)]
        # real part distinguishes QPSK (+1) from π/4-DQPSK (-1) via Re(E[d^4])

    # ── Group 5: Instantaneous frequency — raw (11 features) ──────────────
    dphi = np.diff(np.unwrap(np.angle(x)))
    dphi_c = dphi - dphi.mean()
    dphi_var = float(np.mean(dphi_c ** 2)) + 1e-10
    feats += [
        float(dphi.std() / (np.abs(dphi).mean() + 1e-10)),  # IF regularity (MSK→0)
        float(_kurtosis(dphi)),                               # excess kurtosis
    ]
    feats.append(float(np.mean(dphi_c[1:] * dphi_c[:-1]) / dphi_var))  # IF lag-1 ACF
    # 8-bin histogram of |IF| ∈ [0, π] (captures per-sample phase-jump distribution)
    hist_raw, _ = np.histogram(np.abs(dphi), bins=8, range=(0, np.pi), density=True)
    feats += list(hist_raw.astype(float))

    # ── Group 6: Symbol-rate-normalized IF (13 features) ──────────────────
    # Blind symbol period from 90% PSD bandwidth; ~1.25x error but sufficient
    # to reduce OSR-induced variance in IF statistics.
    T_rough, bw_90 = _psd_bandwidth_T_rough(x)
    feats += [T_rough, bw_90]
    dphi_norm = dphi * T_rough                         # ≈ phase jump per symbol
    feats += [
        float(np.std(dphi_norm) / (np.abs(dphi_norm).mean() + 1e-10)),
        float(_kurtosis(dphi_norm)),
    ]
    hist_norm, _ = np.histogram(np.abs(dphi_norm), bins=8, range=(0, np.pi), density=True)
    feats += list(hist_norm.astype(float))
    lag_T = max(1, min(int(round(T_rough)), len(dphi_c) - 1))
    feats.append(float(np.mean(dphi_c[lag_T:] * dphi_c[:-lag_T]) / dphi_var))

    # ── Group 7: Spectral features (4 features) ───────────────────────────
    _, psd = _welch(x, fs=1.0, nperseg=256, return_onesided=False)
    psd_a = np.abs(psd) + 1e-10
    feats.append(float(np.exp(np.mean(np.log(psd_a))) / np.mean(psd_a)))  # signal Wiener entropy

    _, amp_psd = _welch(amp, fs=1.0, nperseg=256, return_onesided=True)
    amp_psd_a = np.abs(amp_psd) + 1e-10
    feats += [
        float(amp_psd_a[1:].max() / amp_psd_a[1:].mean()),              # amplitude PSD peak/mean
        float(np.exp(np.mean(np.log(amp_psd_a))) / np.mean(amp_psd_a)), # amplitude Wiener entropy
    ]
    # Conjugate spectral peak-to-mean: BPSK has a sharp cyclostationary line at 2f_c
    x2_spec = np.abs(np.fft.fft(x ** 2))
    half = x2_spec[1 : len(x2_spec) // 2]
    feats.append(float(half.max() / (half.mean() + 1e-10)))

    # ── Group 8: Multi-scale phase concentration profile (8 features) ────────
    # Compute |E[d_T^k]| across T = 2..30 samples where d_T[n] = unit phase diff
    # at lag T. Profile shape encodes symbol period + PSK order without timing.
    # Key per-class signatures at large lags (T=20..30):
    #   2PSK ≈ 0.52, MSK ≈ 0.45, 4PSK ≈ 0.26, π/4-DQPSK ≈ 0.21,
    #   8PSK ≈ 0.02, QAM ≈ 0.02-0.04
    #
    # FFT optimization: E[d_T^k] = E[x_k_phase[n+T] * conj(x_k_phase[n])]
    # = autocorrelation of x_k_phase (= (x/|x|)^k) at lag T.
    # All 29 lags computed in one FFT: O(N log N) vs O(29N) Python loop.
    x_phase = x / (np.abs(x) + 1e-10)  # unit-magnitude complex samples
    n_fft = len(x) * 2                  # zero-pad → linear (not circular) correlation
    pc4_profile = np.zeros(29, dtype=np.float32)
    pc2_profile = np.zeros(29, dtype=np.float32)
    pc8_profile = np.zeros(29, dtype=np.float32)
    x4_phase = x_phase ** 4
    re4_acf: np.ndarray | None = None   # keep for Group 9 signed features
    for xp, profile_arr in [
        (x4_phase, pc4_profile),
        (x_phase ** 2, pc2_profile),
        (x_phase ** 8, pc8_profile),
    ]:
        fft_xp = np.fft.fft(xp, n=n_fft)
        # IFFT(|FFT|^2)[T] = sum_n xp[n]*conj(xp[n-T]); |.| = |R(T)| by stationarity
        acf = np.fft.ifft(np.abs(fft_xp) ** 2)
        profile_arr[:] = (np.abs(acf[2:31]) / len(x)).astype(np.float32)
        if re4_acf is None:
            re4_acf = acf   # first iteration = x^4; save for Group 9
    # Summary statistics from the magnitude profile
    pc4_early = float(pc4_profile[0:4].mean())   # T=2..5: initial level
    pc4_mid   = float(pc4_profile[8:14].mean())  # T=10..15: mid-range
    pc4_late  = float(pc4_profile[18:29].mean()) # T=20..30: plateau
    pc4_decay = float(pc4_late / (pc4_early + 1e-6))  # PSK: high, QAM/8PSK: low
    pc4_max   = float(pc4_profile.max())
    pc4_min   = float(pc4_profile.min())  # 8PSK: low (|E[d^4]|=0 at T_s); 0.96σ sep 8PSK/4PSK
    peak_lag_idx = int(pc4_profile.argmax())               # save 0-index for Group 9
    pc4_argmax   = float(peak_lag_idx + 2) / 30.0          # normalized T̂_s
    pc2_late  = float(pc2_profile[18:29].mean()) # T=20..30: 2PSK indicator
    pc8_late  = float(pc8_profile[18:29].mean()) # T=20..30: 8PSK indicator
    feats += [pc4_early, pc4_mid, pc4_late, pc4_decay, pc4_max, pc4_min, pc4_argmax,
              pc2_late, pc8_late]

    # ── Group 9: Signed Re(E[d^4]) summary + high-order moments (5 features) ──
    # Re(E[d_T^4]) is the SIGNED counterpart to the magnitude profile above.
    # At the symbol period T_s:
    #   4PSK:       d^4 = exp(j·4·π/2·k) = +1 → Re(E[d^4]) ≈ +1 at T_s
    #   π/4-DQPSK: d^4 = exp(j·4·π/4·k) = −1 → Re(E[d^4]) ≈ −1 at T_s
    #   8PSK:       d^4 = ±1 equally    → Re(E[d^4]) ≈  0 at T_s
    # re4_min is the minimum over ALL lags (good for π/4-DQPSK detection but
    # under-estimates the +1 signal for 4PSK due to ISI at off-peak lags).
    # re4_at_peak evaluates Re(E[d^4]) at the estimated T_s (from pc4_argmax):
    #   4PSK: ≈ +1,  π/4-DQPSK: ≈ −1,  8PSK: ≈ 0   → much stronger separation.
    assert re4_acf is not None
    re4_real = (re4_acf[2:31].real / len(x)).astype(np.float32)
    re4_min     = float(re4_real.min())
    re4_asym    = float(re4_real.max() + re4_real.min())   # sign asymmetry (1.14σ sep)
    re4_at_peak = float(re4_real[peak_lag_idx])            # signed value at T̂_s
    # |E[x^k]|: k-th complex moment of SRRC-filtered signal; monotone with QAM order.
    feats += [re4_min, re4_asym, re4_at_peak,
              float(abs(np.mean(x ** 6))), float(abs(np.mean(x ** 8)))]

    # ── Group 10: Full signed Re(E[d^4]) profile (29 features, T=2..30) ──────
    # Gives the model complete shape information to learn optimal statistics
    # (e.g., profile is positive+sustained for 4PSK, single negative dip at T_s
    # for π/4-DQPSK, flat near-zero for 8PSK/QAM).  Redundant with Group 9
    # summaries but lets the network discover higher-order patterns.
    feats += re4_real.tolist()

    return np.nan_to_num(np.array(feats, dtype=np.float32))


def iq_features(x: np.ndarray) -> np.ndarray:
    """Compact feature baseline loosely inspired by Azzouz & Nandi (1995).

    The reference uses decision-tree rules over a larger set of instantaneous
    amplitude, phase, and frequency statistics for AMC. This implementation is
    not a reproduction: it keeps only simple amplitude moments, I/Q spread,
    instantaneous-frequency moments, and coarse spectral centroid/spread for a
    lightweight learned baseline.

    Citation:
        Azzouz, E. E. & Nandi, A. K. "Automatic identification of digital
        modulation types." Signal Processing, 47(1), 55-69, 1995.
        https://doi.org/10.1016/0165-1684(95)00099-2
    """
    amp = np.abs(x)
    phase = np.unwrap(np.angle(x))
    inst_freq = np.diff(phase, prepend=phase[0])
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(x)))
    spectrum = spectrum / max(np.sum(spectrum), np.finfo(np.float32).eps)
    freqs = np.linspace(-0.5, 0.5, len(spectrum), endpoint=False)
    spectral_centroid = np.sum(freqs * spectrum)
    spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * spectrum))
    feats = np.array(
        [
            np.mean(amp),
            np.std(amp),
            np.mean(amp**2),
            np.mean(amp**4),
            np.std(np.real(x)),
            np.std(np.imag(x)),
            np.mean(np.abs(inst_freq)),
            np.std(inst_freq),
            spectral_centroid,
            spectral_spread,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(feats)
