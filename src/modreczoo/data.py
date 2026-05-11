from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import scipy.ndimage
import scipy.signal as signal
import torch
from torch.utils.data import DataLoader, Dataset

from modreczoo.models import representation_for_model


SIGNALS_FILE = "signals.npz"
METADATA_FILE = "metadata.parquet"
README_MODULATION_ORDER = ("2PSK", "4PSK", "8PSK", "pi/4-DQPSK", "16QAM", "64QAM", "256QAM", "MSK")


def save_dataset(
    output_dir: str,
    signals: np.ndarray,
    metadata: pl.DataFrame,
    extras: Optional[Dict[str, np.ndarray]] = None,
    compressed: bool = False,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    arrays = {"signals": signals}
    if extras:
        arrays.update(extras)
    save = np.savez_compressed if compressed else np.savez
    save(output / SIGNALS_FILE, **arrays)
    metadata.write_parquet(output / METADATA_FILE)


def load_dataset(output_dir: str) -> Tuple[np.ndarray, pl.DataFrame]:
    output = Path(output_dir)
    with np.load(output / SIGNALS_FILE) as data:
        signals = data["signals"]
    return signals, pl.read_parquet(output / METADATA_FILE)


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
        self.labels = metadata["modulation"].to_numpy()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = int(self.indices[item])
        x = normalize_signal(self.signals[idx])
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
        elif self.representation == "features":
            features = handcrafted_features(x)
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
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **loader_kwargs,
    )


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
    if channel_format == "mag":
        mag = normalized_log_magnitude(x)
        return mag.astype(np.float32)[np.newaxis, :]
    if channel_format == "mag_phase":
        mag = normalized_log_magnitude(x)
        phase = np.angle(x) / np.pi
        return np.stack((mag, phase)).astype(np.float32)
    if channel_format == "mag_inst_freq":
        mag = normalized_log_magnitude(x)
        inst_freq = instantaneous_frequency(x)
        return np.stack((mag, inst_freq)).astype(np.float32)
    if channel_format == "differential_complex":
        return differential_complex_channels(x)
    if channel_format == "apf":
        return apf_channels(x)
    if channel_format == "complex_powers":
        return complex_powers_channels(x)
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
    if channel_format == "mag":
        mag = np.log1p(np.abs(zxx))
        mag = resize_2d(mag, freq_bins, time_bins)
        mag = (mag - np.mean(mag)) / max(np.std(mag), np.finfo(np.float32).eps)
        return mag[np.newaxis].astype(np.float32)
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
    if channel_format == "mag_inst_freq":
        mag = np.log1p(np.abs(zxx))
        phase = np.unwrap(np.angle(zxx), axis=1)
        inst_freq = np.diff(phase, axis=1, prepend=phase[:, :1]) / np.pi
        mag = resize_2d(mag, freq_bins, time_bins)
        inst_freq = resize_2d(inst_freq, freq_bins, time_bins)
        mag = (mag - np.mean(mag)) / max(np.std(mag), np.finfo(np.float32).eps)
        return np.stack((mag, np.nan_to_num(inst_freq))).astype(np.float32)
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


def handcrafted_features(x: np.ndarray) -> np.ndarray:
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
