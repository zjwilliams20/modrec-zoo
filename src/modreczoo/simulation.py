import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import polars as pl
import scipy.special as special
import scipy.signal as signal
from scipy.stats import qmc
from tqdm import tqdm


MODULATIONS = ("2PSK", "4PSK", "8PSK", "pi/4-DQPSK", "16QAM", "64QAM", "256QAM", "MSK")
SUPPORTED_MODULATIONS = ("2ASK", *MODULATIONS, "DQPSK")


DEFAULT_PARAMS = {
    "n_samples": 4096,
    "snr_range": (0.0, 20.0),  # in-band SNR, dB
    "cfo_range": (-1 / 1000, 1 / 1000),  # cycles per sample
    "cpo_range": (0.0, 1.0),  # cycles
    "sto_range": (-1 / 2, 1 / 2),  # symbols
    "upsample_factor_range": (2, 11),  # high endpoint is exclusive
    "downsample_factor_range": (1, 10),  # high endpoint is exclusive; clipped so up/down > 1
    "ebw_range": (0.1, 1.0),  # SRRC excess bandwidth
    "channel": "awgn",
    "rician_k_range": (3.0, 12.0),  # dB
    "n_taps_range": (2, 7),  # high endpoint is exclusive
    "delay_spread_symbols_range": (0.25, 4.0),
    "delay_decay_symbols_range": (0.5, 3.0),
    "sampler": "sobol",
    "seed": None,
}


@dataclass(frozen=True)
class Modem:
    name: str
    points: np.ndarray
    bits: np.ndarray

    @property
    def order(self) -> int:
        return len(self.points)

    @property
    def bits_per_symbol(self) -> int:
        return self.bits.shape[1]


def normalize_power(x: np.ndarray) -> np.ndarray:
    return x / np.sqrt(np.mean(np.abs(x) ** 2))


def gray_code(n: np.ndarray) -> np.ndarray:
    return n ^ (n >> 1)


def bits_to_int(bits: np.ndarray) -> np.ndarray:
    weights = 1 << np.arange(bits.shape[1] - 1, -1, -1)
    return bits @ weights


def ints_to_bits(values: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint16)
    shifts = np.arange(width - 1, -1, -1, dtype=np.uint16)
    return ((values[:, np.newaxis] >> shifts) & 1).astype(np.uint8)


def make_psk_modem(name: str, order: int) -> Modem:
    width = int(math.log2(order))
    binary = np.arange(order, dtype=np.uint16)
    gray = gray_code(binary)
    phases = 2 * np.pi * gray / order
    if order == 2:
        phases += np.pi
    elif order == 4:
        phases += np.pi / 4
    points = np.exp(1j * phases)
    bits = ints_to_bits(binary, width)
    return Modem(name, normalize_power(points), bits)


def make_ask_modem() -> Modem:
    return Modem("2ASK", np.array([-1.0, 1.0], dtype=np.complex128), ints_to_bits(np.arange(2), 1))


def make_qam_modem(name: str, order: int) -> Modem:
    side = int(np.sqrt(order))
    width = int(math.log2(order))
    axis_width = width // 2
    binary = np.arange(order, dtype=np.uint16)
    bits = ints_to_bits(binary, width)
    i_bin = bits_to_int(bits[:, :axis_width])
    q_bin = bits_to_int(bits[:, axis_width:])
    i_gray = gray_code(i_bin)
    q_gray = gray_code(q_bin)
    levels = 2 * np.arange(side) - (side - 1)
    points = levels[i_gray] + 1j * levels[q_gray]
    return Modem(name, normalize_power(points.astype(np.complex128)), bits)


MODEMS: Dict[str, Modem] = {
    "2ASK": make_ask_modem(),
    "2PSK": make_psk_modem("2PSK", 2),
    "4PSK": make_psk_modem("4PSK", 4),
    "8PSK": make_psk_modem("8PSK", 8),
    "16QAM": make_qam_modem("16QAM", 16),
    "64QAM": make_qam_modem("64QAM", 64),
    "256QAM": make_qam_modem("256QAM", 256),
}


def rng_from_seed(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


def signal_seed(dataset_seed: Optional[int], signal_id: int) -> Optional[int]:
    if dataset_seed is None:
        return None
    seed_sequence = np.random.SeedSequence([int(dataset_seed), int(signal_id)])
    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])


def random_bits(rng: np.random.Generator, n_bits: int) -> np.ndarray:
    return rng.integers(0, 2, size=n_bits, dtype=np.uint8)


def modulate_bits(modulation: str, bits: np.ndarray) -> np.ndarray:
    if modulation == "MSK":
        return modulate_msk(bits)
    if modulation in ("DQPSK", "pi/4-DQPSK"):
        return modulate_pi4_dqpsk(bits)

    modem = MODEMS[modulation]
    n_symbols = len(bits) // modem.bits_per_symbol
    bit_groups = bits[: n_symbols * modem.bits_per_symbol].reshape(n_symbols, modem.bits_per_symbol)
    indices = bits_to_int(bit_groups)
    return modem.points[indices]


def demodulate_symbols(modulation: str, symbols: np.ndarray) -> np.ndarray:
    if modulation in ("DQPSK", "pi/4-DQPSK"):
        return demodulate_pi4_dqpsk(symbols)
    if modulation == "MSK":
        raise ValueError("MSK BER sanity check is not implemented in the coherent symbol modem.")

    modem = MODEMS[modulation]
    distances = np.abs(symbols[:, np.newaxis] - modem.points[np.newaxis, :]) ** 2
    nearest = np.argmin(distances, axis=1)
    return modem.bits[nearest].reshape(-1)


def generate_symbols(modulation: str, n_symbols: int, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    rng = rng if rng is not None else rng_from_seed(None)
    if modulation == "MSK":
        bits = random_bits(rng, n_symbols)
    elif modulation in ("DQPSK", "pi/4-DQPSK"):
        bits = random_bits(rng, 2 * n_symbols)
    else:
        bits = random_bits(rng, MODEMS[modulation].bits_per_symbol * n_symbols)
    return modulate_bits(modulation, bits), bits


def modulate_pi4_dqpsk(bits: np.ndarray) -> np.ndarray:
    n_symbols = len(bits) // 2
    pairs = bits[: 2 * n_symbols].reshape(n_symbols, 2)
    dibits = bits_to_int(pairs)
    phase_steps = np.array([np.pi / 4, 3 * np.pi / 4, -np.pi / 4, -3 * np.pi / 4])
    phases = np.cumsum(phase_steps[dibits])
    return np.exp(1j * phases)


def demodulate_pi4_dqpsk(symbols: np.ndarray) -> np.ndarray:
    if len(symbols) < 2:
        return np.empty(0, dtype=np.uint8)
    phase_diff = np.angle(symbols[1:] * np.conj(symbols[:-1]))
    reference = np.array([np.pi / 4, 3 * np.pi / 4, -np.pi / 4, -3 * np.pi / 4])
    wrapped = np.angle(np.exp(1j * (phase_diff[:, np.newaxis] - reference[np.newaxis, :])))
    dibits = np.argmin(np.abs(wrapped), axis=1)
    return ints_to_bits(dibits, 2).reshape(-1)


def modulate_msk(bits: np.ndarray) -> np.ndarray:
    nrz = 2 * bits.astype(float) - 1
    phase = np.cumsum((np.pi / 2) * nrz)
    return np.exp(1j * phase)


def srrc_filter(samples_per_symbol: int, beta: float, span_symbols: int = 8) -> np.ndarray:
    if samples_per_symbol < 1:
        raise ValueError("samples_per_symbol must be at least 1.")
    if beta <= 0 or beta > 1:
        raise ValueError("SRRC excess bandwidth beta must be in (0, 1].")

    n_taps = span_symbols * samples_per_symbol + 1
    t = (np.arange(n_taps) - (n_taps - 1) / 2) / samples_per_symbol
    h = np.empty_like(t, dtype=float)

    zero = np.isclose(t, 0.0)
    singular = np.isclose(np.abs(t), 1 / (4 * beta))
    normal = ~(zero | singular)

    h[zero] = 1 + beta * (4 / np.pi - 1)
    h[singular] = (beta / np.sqrt(2)) * (
        (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
        + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
    )
    h[normal] = (
        np.sin(np.pi * t[normal] * (1 - beta))
        + 4 * beta * t[normal] * np.cos(np.pi * t[normal] * (1 + beta))
    ) / (np.pi * t[normal] * (1 - (4 * beta * t[normal]) ** 2))
    return h / np.sqrt(np.sum(h**2))


def trim_filter_delay(x: np.ndarray, n_taps: int, downsample_factor: int) -> np.ndarray:
    delay_samples = (n_taps - 1) / 2
    start = int(round(delay_samples / downsample_factor))
    return x[start:] if start > 0 else x


def apply_pulse_shape(
    symbols: np.ndarray,
    modulation: str,
    upsample_factor: int,
    downsample_factor: int,
    ebw: float,
) -> np.ndarray:
    if upsample_factor < 2:
        raise ValueError("upsample_factor must be at least 2.")
    if downsample_factor < 1:
        raise ValueError("downsample_factor must be at least 1.")
    if upsample_factor / downsample_factor <= 1:
        raise ValueError("upsample_factor / downsample_factor must be greater than 1.")

    if modulation == "MSK":
        taps = np.ones(upsample_factor, dtype=float)
    else:
        taps = srrc_filter(upsample_factor, ebw)
    shaped = signal.upfirdn(taps, symbols, up=upsample_factor, down=downsample_factor)
    return normalize_power(trim_filter_delay(shaped, len(taps), downsample_factor))


def apply_cfo(x: np.ndarray, cfo: float, cpo: float = 0.0) -> np.ndarray:
    n = np.arange(len(x))
    return x * np.exp(1j * 2 * np.pi * (cfo * n + cpo))


def apply_sto(x: np.ndarray, sto_symbols: float, osr: float) -> np.ndarray:
    if abs(sto_symbols) < 1e-12:
        return x
    n = np.arange(len(x))
    shifted = n + sto_symbols * osr
    real = np.interp(shifted, n, np.real(x), left=0.0, right=0.0)
    imag = np.interp(shifted, n, np.imag(x), left=0.0, right=0.0)
    return real + 1j * imag


def in_band_noise_fraction(osr: float, ebw: float) -> float:
    return float(min(1.0, (1.0 + ebw) / max(float(osr), 1e-12)))


def add_awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator, osr: float = 1.0, ebw: float = 1.0) -> np.ndarray:
    """Add complex AWGN for an in-band SNR target.

    SRRC one-sided bandwidth is (1 + beta) / (2T), so the complex two-sided
    in-band fraction of the sampled spectrum is roughly (1 + beta) / osr.
    """
    power = np.mean(np.abs(x) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = power / (snr_linear * in_band_noise_fraction(osr, ebw))
    noise = np.sqrt(noise_power / 2) * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
    return x + noise


def apply_channel(
    x: np.ndarray,
    channel: str,
    rng: np.random.Generator,
    osr: float,
    rician_k_db: Optional[float] = None,
    n_taps: int = 4,
    delay_spread_symbols: float = 2.0,
    delay_decay_symbols: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    if channel == "awgn":
        return x, channel_metadata([], [], rician_k_db=np.nan)

    if channel == "rayleigh":
        delays, taps = fading_taps(
            rng=rng,
            osr=osr,
            n_taps=n_taps,
            delay_spread_symbols=delay_spread_symbols,
            delay_decay_symbols=delay_decay_symbols,
            rician_k_db=None,
        )
        return apply_tapped_delay_line(x, delays, taps), channel_metadata(delays, taps, rician_k_db=np.nan)

    if channel == "rician":
        k_db = 6.0 if rician_k_db is None else rician_k_db
        delays, taps = fading_taps(
            rng=rng,
            osr=osr,
            n_taps=n_taps,
            delay_spread_symbols=delay_spread_symbols,
            delay_decay_symbols=delay_decay_symbols,
            rician_k_db=k_db,
        )
        return apply_tapped_delay_line(x, delays, taps), channel_metadata(delays, taps, rician_k_db=float(k_db))

    if channel == "soft_limiter":
        magnitude = np.abs(x)
        limited = np.tanh(magnitude) * np.exp(1j * np.angle(x))
        return limited, channel_metadata([], [], rician_k_db=np.nan)

    raise ValueError(f"Unsupported channel: {channel}")


def fading_taps(
    rng: np.random.Generator,
    osr: float,
    n_taps: int,
    delay_spread_symbols: float,
    delay_decay_symbols: float,
    rician_k_db: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    max_delay = max(1, int(round(delay_spread_symbols * osr)))
    n_taps = max(2, min(n_taps, max_delay + 1))
    if n_taps == max_delay + 1:
        delays = np.arange(max_delay + 1, dtype=int)
    else:
        extra_delays = rng.choice(np.arange(1, max_delay + 1), size=n_taps - 1, replace=False)
        delays = np.sort(np.concatenate(([0], extra_delays))).astype(int)

    delay_symbols = delays / osr
    decay = max(delay_decay_symbols, 1e-6)
    pdp = np.exp(-delay_symbols / decay)
    pdp = pdp / np.sum(pdp)

    scatter = np.sqrt(pdp / 2) * (rng.standard_normal(n_taps) + 1j * rng.standard_normal(n_taps))
    if rician_k_db is None:
        taps = scatter
    else:
        k = 10 ** (rician_k_db / 10)
        los = np.zeros(n_taps, dtype=np.complex128)
        los[0] = 1.0
        taps = np.sqrt(k / (k + 1)) * los + np.sqrt(1 / (k + 1)) * scatter

    return delays, taps / np.sqrt(np.sum(np.abs(taps) ** 2))


def apply_tapped_delay_line(x: np.ndarray, delays: np.ndarray, taps: np.ndarray) -> np.ndarray:
    h = np.zeros(int(np.max(delays)) + 1, dtype=np.complex128)
    h[delays] = taps
    return signal.lfilter(h, [1.0], x)


def channel_metadata(delays: Iterable[int], taps: Iterable[complex], rician_k_db: float) -> Dict[str, float]:
    delays = np.asarray(list(delays), dtype=int)
    taps = np.asarray(list(taps), dtype=np.complex128)
    powers = np.abs(taps) ** 2
    if len(delays) and np.sum(powers) > 0:
        mean_delay = float(np.sum(delays * powers) / np.sum(powers))
        rms_delay = float(np.sqrt(np.sum(((delays - mean_delay) ** 2) * powers) / np.sum(powers)))
        max_delay = int(np.max(delays))
    else:
        rms_delay = 0.0
        max_delay = 0

    return {
        "rician_k_db": float(rician_k_db),
        "channel_n_taps": int(len(delays)),
        "channel_max_delay_samples": max_delay,
        "channel_rms_delay_samples": rms_delay,
        "channel_tap_delays": json.dumps(delays.astype(int).tolist()),
        "channel_tap_real": json.dumps(np.real(taps).round(8).tolist()),
        "channel_tap_imag": json.dumps(np.imag(taps).round(8).tolist()),
    }


def pad_or_trim(x: np.ndarray, n_samples: int) -> np.ndarray:
    if len(x) >= n_samples:
        return x[:n_samples]
    return np.pad(x, (0, n_samples - len(x)))


def generate_signal(
    modulation: str,
    snr_db: float,
    cfo: float,
    sto: float,
    upsample_factor: int,
    downsample_factor: int,
    ebw: float,
    cpo: float = 0.0,
    n_samples: int = 32768,
    signal_id: int = 0,
    channel: str = "awgn",
    rician_k_db: Optional[float] = None,
    n_taps: int = 4,
    delay_spread_symbols: float = 2.0,
    delay_decay_symbols: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    debug: bool = False,
) -> Dict:
    rng = rng if rng is not None else rng_from_seed(None)
    osr = upsample_factor / downsample_factor
    n_symbols = int(math.ceil(n_samples / osr)) + 16
    symbols, bits = generate_symbols(modulation, n_symbols, rng)

    shaped = pad_or_trim(apply_pulse_shape(symbols, modulation, upsample_factor, downsample_factor, ebw), n_samples)
    shifted = apply_cfo(shaped, cfo, cpo=cpo)
    shifted = apply_sto(shifted, sto, osr)
    faded, channel_metadata = apply_channel(
        shifted,
        channel,
        rng,
        osr=osr,
        rician_k_db=rician_k_db,
        n_taps=n_taps,
        delay_spread_symbols=delay_spread_symbols,
        delay_decay_symbols=delay_decay_symbols,
    )
    received = add_awgn(faded, snr_db, rng, osr=osr, ebw=ebw)

    metadata = {
        "signal_id": signal_id,
        "modulation": modulation,
        "snr_db": float(snr_db),
        "cfo": float(cfo),
        "cpo": float(cpo),
        "sto": float(sto),
        "upsample_factor": int(upsample_factor),
        "downsample_factor": int(downsample_factor),
        "osr": float(osr),
        "ebw": float(ebw),
        "n_samples": int(n_samples),
        "n_symbols": int(n_symbols),
        "channel": channel,
        "snr_definition": "in_band_post_channel",
        "channel_delay_spread_symbols": float(delay_spread_symbols),
        "channel_delay_decay_symbols": float(delay_decay_symbols),
        **channel_metadata,
    }

    result = {"signal": received.astype(np.complex64), "metadata": metadata}
    if debug:
        result.update(
            {
                "bits": bits,
                "symbols": symbols,
                "signal_preshift": shaped,
                "signal_clean": shifted,
            }
        )
    return result


def generate_dataset(
    modulations: Iterable[str],
    n_signals: int,
    params: dict,
    debug: bool = False,
    num_workers: int = 1,
) -> Tuple[np.ndarray, pl.DataFrame, Optional[Dict[str, np.ndarray]]]:
    rng = rng_from_seed(params.get("seed"))
    modulations = tuple(modulations)
    dataset = np.empty((n_signals, params["n_samples"]), dtype=np.complex64)
    metadata_rows: List[Dict] = []
    debug_arrays: Dict[str, List[np.ndarray]] = {"signal_clean": [], "signal_preshift": []}
    design = sample_parameter_design(n_signals, modulations, params, rng)
    num_workers = max(1, int(num_workers))

    if num_workers == 1:
        results = (_generate_dataset_signal(i, design[i], params, debug, rng) for i in range(n_signals))
    else:
        tasks = ((i, design[i], params, debug) for i in range(n_signals))
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(_generate_dataset_signal_worker, tasks)

    for i, signal_data in tqdm(results, total=n_signals, desc="Generating signals"):
        dataset[i] = signal_data["signal"]
        metadata_rows.append(signal_data["metadata"])
        if debug:
            debug_arrays["signal_clean"].append(pad_or_trim(signal_data["signal_clean"], params["n_samples"]).astype(np.complex64))
            debug_arrays["signal_preshift"].append(pad_or_trim(signal_data["signal_preshift"], params["n_samples"]).astype(np.complex64))

    metadata_rows.sort(key=lambda row: row["signal_id"])
    metadata = pl.DataFrame(metadata_rows)
    extras = {key: np.stack(value) for key, value in debug_arrays.items()} if debug else None
    return dataset, metadata, extras


def _generate_dataset_signal(
    signal_id: int,
    row: Dict,
    params: dict,
    debug: bool,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[int, Dict]:
    seed = params.get("seed")
    signal_rng = rng if seed is None and rng is not None else rng_from_seed(signal_seed(seed, signal_id))
    try:
        return signal_id, generate_signal(
            modulation=row["modulation"],
            snr_db=row["snr_db"],
            cfo=row["cfo"],
            sto=row["sto"],
            upsample_factor=row["upsample_factor"],
            downsample_factor=row["downsample_factor"],
            ebw=row["ebw"],
            cpo=row["cpo"],
            n_samples=int(params["n_samples"]),
            signal_id=signal_id,
            channel=params["channel"],
            rician_k_db=row["rician_k_db"],
            n_taps=row["n_taps"],
            delay_spread_symbols=row["delay_spread_symbols"],
            delay_decay_symbols=row["delay_decay_symbols"],
            rng=signal_rng,
            debug=debug,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to generate signal_id={signal_id}") from exc


def _generate_dataset_signal_worker(args: Tuple[int, Dict, dict, bool]) -> Tuple[int, Dict]:
    signal_id, row, params, debug = args
    return _generate_dataset_signal(signal_id, row, params, debug)


def sample_parameter_design(
    n_signals: int,
    modulations: Tuple[str, ...],
    params: dict,
    rng: np.random.Generator,
) -> List[Dict]:
    sampler = params.get("sampler", "sobol")
    if sampler == "sobol":
        seed = params.get("seed")
        sobol = qmc.Sobol(d=11, scramble=True, seed=seed)
        unit = sobol.random(n_signals)
    elif sampler == "random":
        unit = rng.random((n_signals, 11))
    else:
        raise ValueError(f"Unsupported sampler: {sampler}")

    labels = balanced_modulation_labels(n_signals, modulations, rng)
    rows = []
    for i in range(n_signals):
        u = unit[i]
        upsample_factor = max(2, scale_int(u[4], params["upsample_factor_range"]))
        downsample_factor = max(1, scale_int(u[5], params["downsample_factor_range"]))
        downsample_factor = min(downsample_factor, upsample_factor - 1)
        rows.append(
            {
                "modulation": labels[i],
                "snr_db": scale_float(u[0], params["snr_range"]),
                "cfo": scale_float(u[1], params["cfo_range"]),
                "cpo": scale_float(u[2], params["cpo_range"]),
                "sto": scale_float(u[3], params["sto_range"]),
                "upsample_factor": upsample_factor,
                "downsample_factor": downsample_factor,
                "osr": float(upsample_factor / downsample_factor),
                "ebw": scale_float(u[6], params["ebw_range"]),
                "rician_k_db": scale_float(u[7], params["rician_k_range"]),
                "n_taps": scale_int(u[8], params["n_taps_range"]),
                "delay_spread_symbols": scale_float(u[9], params["delay_spread_symbols_range"]),
                "delay_decay_symbols": scale_float(u[10], params["delay_decay_symbols_range"]),
            }
        )
    return rows


def balanced_modulation_labels(n_signals: int, modulations: Tuple[str, ...], rng: np.random.Generator) -> np.ndarray:
    labels = np.resize(np.asarray(modulations), n_signals)
    rng.shuffle(labels)
    return labels


def scale_float(u: float, bounds: Tuple[float, float]) -> float:
    low, high = bounds
    return float(low + u * (high - low))


def scale_int(u: float, bounds: Tuple[int, int]) -> int:
    low, high = bounds
    return int(min(high - 1, low + math.floor(u * (high - low))))


def awgn_symbol_channel(symbols: np.ndarray, ebn0_db: float, bits_per_symbol: int, rng: np.random.Generator) -> np.ndarray:
    ebn0 = 10 ** (ebn0_db / 10)
    noise_variance = 1 / (bits_per_symbol * ebn0)
    noise = np.sqrt(noise_variance / 2) * (rng.standard_normal(len(symbols)) + 1j * rng.standard_normal(len(symbols)))
    return symbols + noise


def theoretical_ber(modulation: str, ebn0_db: float) -> float:
    ebn0 = 10 ** (ebn0_db / 10)
    if modulation in ("2ASK", "2PSK", "4PSK"):
        return 0.5 * special.erfc(np.sqrt(ebn0))
    if modulation == "8PSK":
        m = 8
        k = math.log2(m)
        return (2 / k) * 0.5 * special.erfc(np.sqrt(k * ebn0) * np.sin(np.pi / m))
    if modulation.endswith("QAM"):
        m = int(modulation.replace("QAM", ""))
        k = math.log2(m)
        return (4 / k) * (1 - 1 / np.sqrt(m)) * 0.5 * special.erfc(np.sqrt(3 * k * ebn0 / (2 * (m - 1))))
    if modulation in ("DQPSK", "pi/4-DQPSK"):
        return 0.5 * np.exp(-ebn0)
    raise ValueError(f"No theoretical BER model is implemented for {modulation}.")


def estimate_ber(
    modulation: str,
    ebn0_db: float,
    n_bits: int = 200_000,
    seed: Optional[int] = None,
) -> Tuple[float, float, int]:
    if modulation == "MSK":
        raise ValueError("MSK BER sanity check is not implemented.")
    rng = rng_from_seed(seed)
    bits_per_symbol = 2 if modulation in ("DQPSK", "pi/4-DQPSK") else MODEMS[modulation].bits_per_symbol
    n_symbols = max(2, math.ceil(n_bits / bits_per_symbol))
    tx_bits = random_bits(rng, n_symbols * bits_per_symbol)
    tx_symbols = modulate_bits(modulation, tx_bits)
    rx_symbols = awgn_symbol_channel(tx_symbols, ebn0_db, bits_per_symbol, rng)
    rx_bits = demodulate_symbols(modulation, rx_symbols)
    if modulation in ("DQPSK", "pi/4-DQPSK"):
        tx_bits = tx_bits[2 : 2 + len(rx_bits)]
    else:
        tx_bits = tx_bits[: len(rx_bits)]
    errors = int(np.count_nonzero(tx_bits != rx_bits))
    return errors / len(rx_bits), theoretical_ber(modulation, ebn0_db), errors


def ber_sweep(
    modulations: Iterable[str],
    ebn0_dbs: Iterable[float],
    n_bits: int,
    seed: Optional[int],
) -> pl.DataFrame:
    rows = []
    for modulation in modulations:
        for ebn0_db in ebn0_dbs:
            empirical, theory, errors = estimate_ber(modulation, float(ebn0_db), n_bits=n_bits, seed=seed)
            rows.append(
                {
                    "modulation": modulation,
                    "ebn0_db": float(ebn0_db),
                    "empirical_ber": float(empirical),
                    "theory_ber": float(theory),
                    "errors": errors,
                    "n_bits": n_bits,
                }
            )
    return pl.DataFrame(rows)
