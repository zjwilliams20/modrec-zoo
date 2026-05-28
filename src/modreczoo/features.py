import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis

from modreczoo.transforms import remove_empirical_cfo


N_CSP_CANONICAL_FEATURES = 13
N_CSP_EXPERT_FEATURES = 107
EPS = 1e-10


def iq_features(x: np.ndarray) -> np.ndarray:
    amp = np.abs(x)
    phase = np.unwrap(np.angle(x))
    inst_freq = np.diff(phase, prepend=phase[0])
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(x)))
    spectrum = spectrum / max(np.sum(spectrum), np.finfo(np.float32).eps)
    freqs = np.linspace(-0.5, 0.5, len(spectrum), endpoint=False)
    spectral_centroid = np.sum(freqs * spectrum)
    spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * spectrum))
    return np.nan_to_num(
        np.array(
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
    )


def csp_canonical_features(x: np.ndarray) -> np.ndarray:
    """Canonical 13-feature CSP vector from cumulants, phase, IF, and x^2 spectrum."""
    x = _csp_signal(x)
    m20, c40, c42, c41, m42 = _cumulants(x)
    amp = np.abs(x)
    d = _unit_phase(x[1:] * x[:-1].conj())
    dphi = np.diff(np.unwrap(np.angle(x)))
    x2_spec = np.abs(np.fft.fft(x**2))
    half = x2_spec[1 : len(x2_spec) // 2]

    return np.nan_to_num(
        np.array(
            [
                float(abs(m20)),
                float(abs(c40)),
                c42,
                float(abs(c41)),
                m42,
                float(amp.std() / (amp.mean() + EPS)),
                float(np.mean(amp**6)),
                float(np.mean(amp**8)),
                float(abs(np.mean(d**2))),
                float(abs(np.mean(d**4))),
                float(abs(np.mean(d**8))),
                float(dphi.std() / (np.abs(dphi).mean() + EPS)),
                float(half.max() / (half.mean() + EPS)),
            ],
            dtype=np.float32,
        )
    )


def csp_expert_features(x: np.ndarray) -> np.ndarray:
    """Extended 107-feature learned CSP/statistical vector."""
    x = _csp_signal(x)
    m20, c40, c42, c41, m42 = _cumulants(x)
    amp = np.abs(x)
    dphi = np.diff(np.unwrap(np.angle(x)))
    dphi_c = dphi - dphi.mean()
    dphi_var = float(np.mean(dphi_c**2)) + EPS

    feats = [
        float(abs(m20)),
        float(abs(c40)),
        c42,
        float(abs(c41)),
    ]
    feats.extend(_amplitude_features(amp, m42))
    feats.extend(_phase_features(x, amp))
    feats.extend(_if_features(dphi, dphi_c, dphi_var))
    feats.extend(_symbol_rate_if_features(x, dphi, dphi_c, dphi_var))
    feats.extend(_spectral_features(x, amp))
    feats.extend(_phase_profile_features(x))

    return np.nan_to_num(np.array(feats, dtype=np.float32))


def _csp_signal(x: np.ndarray) -> np.ndarray:
    x = remove_empirical_cfo(x - np.mean(x))
    return x / np.sqrt(max(float(np.mean(np.abs(x) ** 2)), EPS))


def _cumulants(x: np.ndarray) -> tuple[complex, complex, float, complex, float]:
    m20 = np.mean(x**2)
    m40 = np.mean(x**4)
    m41 = np.mean(x**3 * x.conj())
    m42 = float(np.mean(np.abs(x) ** 4))
    c40 = m40 - 3 * m20**2
    c41 = m41 - 3 * m20
    c42 = float((m42 - abs(m20) ** 2 - 2.0).real)
    return m20, c40, c42, c41, m42


def _unit_phase(x: np.ndarray) -> np.ndarray:
    return x / (np.abs(x) + EPS)


def _amplitude_features(amp: np.ndarray, m42: float) -> list[float]:
    amp_c = amp - amp.mean()
    amp_var = float(np.mean(amp_c**2)) + EPS
    feats = [
        m42,
        float(amp.std() / (amp.mean() + EPS)),
        float(np.mean(amp**6)),
        float(np.mean(amp**8)),
    ]
    feats.extend(np.quantile(amp, [0.10, 0.25, 0.50, 0.75, 0.90, 0.95]).astype(float).tolist())
    feats.extend(float(np.mean(amp < thr)) for thr in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70))
    feats.extend((float(np.mean(amp > 1.4)), float(np.mean(amp > 1.8))))
    feats.extend(float(np.mean(amp_c[lag:] * amp_c[:-lag]) / amp_var) for lag in (1, 2, 4))
    return feats


def _phase_features(x: np.ndarray, amp: np.ndarray) -> list[float]:
    ph = x / (amp + EPS)
    d = _unit_phase(x[1:] * x[:-1].conj())
    feats = [float(abs(np.mean(ph**2))), float(abs(np.mean(ph**4)))]
    for k in (2, 4, 8):
        dk = d**k
        feats.extend((float(abs(np.mean(dk))), float(np.mean(dk).real)))
    return feats


def _if_features(dphi: np.ndarray, dphi_c: np.ndarray, dphi_var: float) -> list[float]:
    hist_raw, _ = np.histogram(np.abs(dphi), bins=8, range=(0, np.pi), density=True)
    return [
        float(dphi.std() / (np.abs(dphi).mean() + EPS)),
        float(kurtosis(dphi)),
        float(np.mean(dphi_c[1:] * dphi_c[:-1]) / dphi_var),
        *hist_raw.astype(float).tolist(),
    ]


def _symbol_rate_if_features(
    x: np.ndarray,
    dphi: np.ndarray,
    dphi_c: np.ndarray,
    dphi_var: float,
) -> list[float]:
    t_rough, bw_90 = _psd_bandwidth_t_rough(x)
    dphi_norm = dphi * t_rough
    hist_norm, _ = np.histogram(np.abs(dphi_norm), bins=8, range=(0, np.pi), density=True)
    lag_t = max(1, min(int(round(t_rough)), len(dphi_c) - 1))
    return [
        t_rough,
        bw_90,
        float(np.std(dphi_norm) / (np.abs(dphi_norm).mean() + EPS)),
        float(kurtosis(dphi_norm)),
        *hist_norm.astype(float).tolist(),
        float(np.mean(dphi_c[lag_t:] * dphi_c[:-lag_t]) / dphi_var),
    ]


def _spectral_features(x: np.ndarray, amp: np.ndarray) -> list[float]:
    _, psd = welch(x, fs=1.0, nperseg=256, return_onesided=False)
    _, amp_psd = welch(amp, fs=1.0, nperseg=256, return_onesided=True)
    amp_psd_a = np.abs(amp_psd) + EPS
    x2_spec = np.abs(np.fft.fft(x**2))
    half = x2_spec[1 : len(x2_spec) // 2]
    return [
        _wiener_entropy(psd),
        float(amp_psd_a[1:].max() / amp_psd_a[1:].mean()),
        _wiener_entropy(amp_psd),
        float(half.max() / (half.mean() + EPS)),
    ]


def _phase_profile_features(x: np.ndarray) -> list[float]:
    x_phase = _unit_phase(x)
    n_fft = len(x) * 2
    pc4_profile, re4_real = _phase_acf_profile(x_phase, 4, n_fft)
    pc2_profile, _ = _phase_acf_profile(x_phase, 2, n_fft)
    pc8_profile, _ = _phase_acf_profile(x_phase, 8, n_fft)

    pc4_early = float(pc4_profile[0:4].mean())
    pc4_mid = float(pc4_profile[8:14].mean())
    pc4_late = float(pc4_profile[18:29].mean())
    pc4_decay = float(pc4_late / (pc4_early + 1e-6))
    peak_lag_idx = int(pc4_profile.argmax())
    re4_min = float(re4_real.min())
    re4_asym = float(re4_real.max() + re4_real.min())
    re4_at_peak = float(re4_real[peak_lag_idx])

    return [
        pc4_early,
        pc4_mid,
        pc4_late,
        pc4_decay,
        float(pc4_profile.max()),
        float(pc4_profile.min()),
        float(peak_lag_idx + 2) / 30.0,
        float(pc2_profile[18:29].mean()),
        float(pc8_profile[18:29].mean()),
        re4_min,
        re4_asym,
        re4_at_peak,
        float(abs(np.mean(x**6))),
        float(abs(np.mean(x**8))),
        *re4_real.tolist(),
    ]


def _phase_acf_profile(x_phase: np.ndarray, power: int, n_fft: int) -> tuple[np.ndarray, np.ndarray]:
    xp = x_phase**power
    acf = np.fft.ifft(np.abs(np.fft.fft(xp, n=n_fft)) ** 2)
    return (
        (np.abs(acf[2:31]) / len(x_phase)).astype(np.float32),
        (acf[2:31].real / len(x_phase)).astype(np.float32),
    )


def _psd_bandwidth_t_rough(x: np.ndarray) -> tuple[float, float]:
    _, psd = welch(x, fs=1.0, nperseg=512, return_onesided=False)
    psd = np.abs(psd)
    half_n = len(psd) // 2
    center_psd = np.zeros(half_n + 1)
    center_psd[0] = psd[0]
    for k in range(1, half_n + 1):
        center_psd[k] = psd[k] + (psd[-k] if k < len(psd) - k else psd[k])
    cumsum = np.cumsum(center_psd)
    idx_90 = int(np.searchsorted(cumsum, 0.90 * cumsum[-1]))
    bw_90 = max(idx_90, 1) / len(psd)
    rs_rough = 2.0 * bw_90 / 1.25
    t_rough = float(np.clip(1.0 / max(rs_rough, 1.0 / 128), 1.0, 128.0))
    return t_rough, float(bw_90)


def _wiener_entropy(power: np.ndarray) -> float:
    power = np.abs(power) + EPS
    return float(np.exp(np.mean(np.log(power))) / np.mean(power))
