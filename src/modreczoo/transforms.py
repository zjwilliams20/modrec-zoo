import numpy as np
import scipy.ndimage
import scipy.signal as signal


CSP_LAGS = (1, 4, 16)
FLOAT_EPS = np.finfo(np.float32).eps


def normalize_signal(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.complex64, copy=False)
    x = x - np.mean(x)
    scale = np.sqrt(np.mean(np.abs(x) ** 2))
    return x / max(scale, FLOAT_EPS)


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
    spectrum = np.fft.fftshift(np.fft.fft(x * np.hanning(len(x))))
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
        return np.stack((normalized_log_magnitude(x), np.angle(x) / np.pi)).astype(np.float32)
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
    scale = max(np.sqrt(np.mean(np.abs(d) ** 2)), FLOAT_EPS)
    return np.stack((np.real(d) / scale, np.imag(d) / scale)).astype(np.float32)


def complex_powers_channels(x: np.ndarray) -> np.ndarray:
    channels = []
    for power in (1, 2, 4):
        z = x ** power
        scale = max(np.sqrt(np.mean(np.abs(z) ** 2)), FLOAT_EPS)
        channels.extend((np.real(z) / scale, np.imag(z) / scale))
    return np.stack(channels).astype(np.float32)


def lag_product(x: np.ndarray, lag: int) -> np.ndarray:
    delayed = np.zeros_like(x)
    delayed[lag:] = x[:-lag]
    return x * np.conj(delayed)


def multilag_channels(x: np.ndarray, lags: tuple[int, ...] = CSP_LAGS) -> np.ndarray:
    channels = []
    for lag in lags:
        prod = lag_product(x, lag)
        scale = max(np.sqrt(np.mean(np.abs(prod) ** 2)), FLOAT_EPS)
        channels.extend((np.real(prod) / scale, np.imag(prod) / scale))
    return np.stack(channels).astype(np.float32)


def cyclic_caf_channels(x: np.ndarray, lags: tuple[int, ...] = CSP_LAGS) -> np.ndarray:
    spectra = []
    for lag in lags:
        r_alpha = np.abs(np.fft.fft(lag_product(x, lag)))
        spectra.append(r_alpha / max(np.max(r_alpha), FLOAT_EPS))
    return np.stack(spectra).astype(np.float32)


def apf_channels(x: np.ndarray) -> np.ndarray:
    phase = np.angle(x)
    return np.stack(
        (
            normalized_log_magnitude(x),
            np.cos(phase).astype(np.float32),
            np.sin(phase).astype(np.float32),
            instantaneous_frequency(x),
        )
    )


def normalized_log_magnitude(x: np.ndarray) -> np.ndarray:
    mag = np.log1p(np.abs(x))
    return ((mag - np.mean(mag)) / max(np.std(mag), FLOAT_EPS)).astype(np.float32)


def instantaneous_frequency(x: np.ndarray) -> np.ndarray:
    phase = np.unwrap(np.angle(x))
    inst_freq = np.diff(phase, prepend=phase[0]) / np.pi
    return np.nan_to_num(inst_freq).astype(np.float32)


def frequency_channels(x: np.ndarray, channel_format: str) -> np.ndarray:
    spectrum = np.fft.fftshift(np.fft.fft(x))
    spectrum = spectrum / max(np.sqrt(np.mean(np.abs(spectrum) ** 2)), FLOAT_EPS)
    return complex_channels(spectrum, channel_format)


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
    zxx = _spectrogram_zxx(x, scipy_window, freq_bins, nperseg, noverlap)
    if channel_format == "real_imag":
        return _spectrogram_real_imag(zxx, freq_bins, time_bins)
    if channel_format == "mag_phase":
        return _spectrogram_mag_phase(zxx, freq_bins, time_bins)
    if channel_format == "complex_powers":
        return _spectrogram_complex_powers(x, scipy_window, freq_bins, time_bins, nperseg, noverlap)
    if channel_format == "scf":
        return scf_channels(x, n_alpha=time_bins, n_freq=freq_bins, nperseg=nperseg)
    raise ValueError(f"Unsupported channel format: {channel_format}")


def _parse_window(window: str) -> str | tuple[str, float]:
    if ":" in window:
        name, beta = window.split(":", 1)
        return (name, float(beta))
    return window


def _spectrogram_zxx(
    x: np.ndarray,
    window: str | tuple[str, float],
    nfft: int,
    nperseg: int,
    noverlap: int,
    shift: bool = True,
) -> np.ndarray:
    _, _, zxx = signal.spectrogram(
        x,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=False,
        return_onesided=False,
        scaling="spectrum",
        mode="complex",
    )
    if shift:
        return np.fft.fftshift(zxx, axes=0)
    return zxx


def _spectrogram_real_imag(zxx: np.ndarray, freq_bins: int, time_bins: int) -> np.ndarray:
    real = resize_2d(np.real(zxx), freq_bins, time_bins)
    imag = resize_2d(np.imag(zxx), freq_bins, time_bins)
    scale = max(np.sqrt(np.mean(real**2 + imag**2)), FLOAT_EPS)
    return np.stack((real / scale, imag / scale)).astype(np.float32)


def _spectrogram_mag_phase(zxx: np.ndarray, freq_bins: int, time_bins: int) -> np.ndarray:
    mag = resize_2d(np.log1p(np.abs(zxx)), freq_bins, time_bins)
    phase = resize_2d(np.angle(zxx), freq_bins, time_bins) / np.pi
    mag = (mag - np.mean(mag)) / max(np.std(mag), FLOAT_EPS)
    return np.stack((mag, phase)).astype(np.float32)


def _spectrogram_complex_powers(
    x: np.ndarray,
    window: str | tuple[str, float],
    freq_bins: int,
    time_bins: int,
    nperseg: int,
    noverlap: int,
) -> np.ndarray:
    channels = []
    for power in (1, 2, 4):
        zxx = _spectrogram_zxx(x ** power, window, freq_bins, nperseg, noverlap)
        for part in (np.real(zxx), np.imag(zxx)):
            ch = resize_2d(part, freq_bins, time_bins).astype(np.float32)
            channels.append(ch / max(np.sqrt(np.mean(ch**2)), FLOAT_EPS))
    return np.stack(channels)


def resize_2d(x: np.ndarray, rows: int, cols: int) -> np.ndarray:
    if x.shape == (rows, cols):
        return x
    return scipy.ndimage.zoom(x, (rows / x.shape[0], cols / x.shape[1]), order=1)


def scf_channels(x: np.ndarray, n_alpha: int = 64, n_freq: int = 64, nperseg: int = 64) -> np.ndarray:
    zxx = _spectrogram_zxx(x, "hann", n_freq, nperseg, nperseg * 3 // 4, shift=False)
    scf = np.zeros((n_alpha, zxx.shape[0]), dtype=np.float32)
    for dk in range(n_alpha):
        shifted = np.roll(zxx, -dk, axis=0)
        scf[dk] = np.abs(np.mean(zxx * np.conj(shifted), axis=1))
    scf = (scf - np.mean(scf)) / max(np.std(scf), FLOAT_EPS)
    return scf[np.newaxis]
