from typing import Dict, Tuple

import numpy as np
import polars as pl
import scipy.signal as signal
import torch
from torch.utils.data import Dataset


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
        spectrogram_size: int = 64,
    ) -> None:
        self.signals = signals
        self.metadata = metadata
        self.indices = indices.astype(np.int64)
        self.label_to_id = label_to_id
        self.representation = representation
        self.channel_format = channel_format
        self.remove_cfo = remove_cfo
        self.cfo_estimator = cfo_estimator
        self.spectrogram_size = spectrogram_size
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
            features = spectrogram_channels(x, channel_format=self.channel_format, size=self.spectrogram_size)
        elif self.representation == "features":
            features = handcrafted_features(x)
        else:
            raise ValueError(f"Unsupported representation: {self.representation}")

        return torch.from_numpy(features).float(), torch.tensor(y, dtype=torch.long)


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
        mag = np.abs(x)
        phase = np.angle(x) / np.pi
        mag = np.log1p(mag)
        mag = (mag - np.mean(mag)) / max(np.std(mag), np.finfo(np.float32).eps)
        return np.stack((mag, phase)).astype(np.float32)
    raise ValueError(f"Unsupported channel format: {channel_format}")


def frequency_channels(x: np.ndarray, channel_format: str) -> np.ndarray:
    spectrum = np.fft.fftshift(np.fft.fft(x))
    spectrum = spectrum / max(np.sqrt(np.mean(np.abs(spectrum) ** 2)), np.finfo(np.float32).eps)
    return complex_channels(spectrum, channel_format)


def spectrogram_channels(x: np.ndarray, channel_format: str, size: int = 64) -> np.ndarray:
    _, _, zxx = signal.stft(x, nperseg=64, noverlap=48, nfft=size, return_onesided=False, boundary=None)
    zxx = np.fft.fftshift(zxx, axes=0)
    if channel_format == "real_imag":
        real = resize_2d(np.real(zxx), size, size)
        imag = resize_2d(np.imag(zxx), size, size)
        scale = max(np.sqrt(np.mean(real**2 + imag**2)), np.finfo(np.float32).eps)
        return np.stack((real / scale, imag / scale)).astype(np.float32)
    if channel_format == "mag_phase":
        mag = np.log1p(np.abs(zxx))
        phase = np.angle(zxx)
        mag = resize_2d(mag, size, size)
        phase = resize_2d(phase, size, size)
        mag = (mag - np.mean(mag)) / max(np.std(mag), np.finfo(np.float32).eps)
        phase = phase / np.pi
        return np.stack((mag, phase)).astype(np.float32)
    raise ValueError(f"Unsupported channel format: {channel_format}")


def resize_2d(x: np.ndarray, rows: int, cols: int) -> np.ndarray:
    row_idx = np.linspace(0, x.shape[0] - 1, rows).round().astype(int)
    col_idx = np.linspace(0, x.shape[1] - 1, cols).round().astype(int)
    return x[row_idx][:, col_idx]


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
