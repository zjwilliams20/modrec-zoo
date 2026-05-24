import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl
import scipy.signal as signal
from scipy.special import logsumexp
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

from modreczoo.simulation import MODEMS, MODULATIONS, in_band_noise_fraction, srrc_filter


ORACLE_CACHE_VERSION = 2
ORACLE_CACHE_PARQUET = "oracle_predictions.parquet"
ORACLE_CACHE_JSON = "oracle_predictions.json"


@dataclass(frozen=True)
class OracleConfig:
    edge_symbols: int = 10
    min_noise_variance: float = 1e-6


def oracle_predict(
    x: np.ndarray,
    metadata: dict,
    modulations: Iterable[str] = MODULATIONS,
    config: OracleConfig = OracleConfig(),
) -> tuple[str, dict[str, float]]:
    scores = oracle_scores(x, metadata, modulations, config)
    return max(scores, key=scores.get), scores


def oracle_scores(
    x: np.ndarray,
    metadata: dict,
    modulations: Iterable[str] = MODULATIONS,
    config: OracleConfig = OracleConfig(),
) -> dict[str, float]:
    samples = _metadata_compensated_samples(np.asarray(x, dtype=np.complex128), metadata, config)
    noise_variance = _symbol_noise_variance(metadata, config)
    return {
        modulation: _modulation_log_likelihood(samples, modulation, noise_variance)
        for modulation in modulations
    }


def _oracle_predict_one(args: tuple[np.ndarray, dict[str, object], list[str], OracleConfig]) -> tuple[str, dict[str, float]]:
    x, row, labels, config = args
    return oracle_predict(x, row, labels, config)


def evaluate_oracle(
    signals: np.ndarray,
    metadata: pl.DataFrame,
    indices: np.ndarray | None = None,
    modulations: Iterable[str] = MODULATIONS,
    config: OracleConfig = OracleConfig(),
    num_workers: int = 1,
) -> dict:
    labels = list(modulations)
    label_to_id = {label: i for i, label in enumerate(labels)}
    if indices is None:
        indices = np.arange(len(metadata), dtype=np.int64)

    def _task_iter():
        for idx in indices:
            yield signals[int(idx)], metadata.row(int(idx), named=True), labels, config

    y_true, y_pred, scores = [], [], []
    if num_workers > 1:
        chunksize = max(1, len(indices) // (num_workers * 4))
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            results = tqdm(
                ex.map(_oracle_predict_one, _task_iter(), chunksize=chunksize),
                total=len(indices),
                desc="Oracle predictions",
                unit="sig",
            )
            rows_meta = [metadata.row(int(idx), named=True) for idx in indices]
            for (pred, row_scores), row in zip(results, rows_meta):
                y_true.append(label_to_id[row["modulation"]])
                y_pred.append(label_to_id[pred])
                scores.append(row_scores)
    else:
        for idx in tqdm(indices, desc="Oracle predictions", unit="sig"):
            row = metadata.row(int(idx), named=True)
            pred, row_scores = oracle_predict(signals[int(idx)], row, labels, config)
            y_true.append(label_to_id[row["modulation"]])
            y_pred.append(label_to_id[pred])
            scores.append(row_scores)

    y_true_np = np.asarray(y_true, dtype=int)
    y_pred_np = np.asarray(y_pred, dtype=int)
    nll_bits = oracle_nll_bits(scores, labels, y_true_np)
    return {
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)) if len(y_true_np) else 0.0,
        "confusion": confusion_matrix(y_true_np, y_pred_np, labels=np.arange(len(labels))),
        "y_true": y_true_np,
        "y_pred": y_pred_np,
        "nll_bits": nll_bits,
        "labels": labels,
        "scores": scores,
    }


def oracle_nll_bits(scores: list[dict[str, float]], labels: list[str], y_true: np.ndarray) -> np.ndarray:
    nll = np.empty(len(scores), dtype=np.float32)
    for i, row_scores in enumerate(scores):
        logits = np.array([row_scores[label] for label in labels], dtype=np.float64)
        log_p_true = logits[int(y_true[i])] - logsumexp(logits)
        nll[i] = float(-log_p_true / np.log(2.0))
    return nll


def build_oracle_cache(
    dataset_dir: str | Path,
    signals: np.ndarray,
    metadata: pl.DataFrame,
    modulations: Iterable[str],
    config: OracleConfig = OracleConfig(),
    force: bool = False,
    num_workers: int = 1,
) -> dict:
    dataset_dir = Path(dataset_dir)
    parquet_path = dataset_dir / ORACLE_CACHE_PARQUET
    json_path = dataset_dir / ORACLE_CACHE_JSON
    if not force and (parquet_path.exists() or json_path.exists()):
        raise FileExistsError(f"Oracle cache already exists in {dataset_dir}; use --force-oracle-cache.")

    labels = list(modulations)
    result = evaluate_oracle(signals, metadata, modulations=labels, config=config, num_workers=num_workers)
    signal_ids = _signal_ids(metadata)
    rows = pl.DataFrame(
        {
            "signal_id": signal_ids,
            "modulation": metadata["modulation"].to_list(),
            "oracle_pred": [labels[int(i)] for i in result["y_pred"]],
            "oracle_nll_bits": result["nll_bits"],
        }
    )
    cache_meta = {
        "oracle_cache_version": ORACLE_CACHE_VERSION,
        "labels": labels,
        "row_count": len(metadata),
        "oracle_config": asdict(config),
    }
    rows.write_parquet(parquet_path)
    json_path.write_text(json.dumps(cache_meta, indent=2) + "\n", encoding="utf-8")
    return {
        "parquet_path": str(parquet_path),
        "json_path": str(json_path),
        "row_count": len(metadata),
        "accuracy": result["accuracy"],
    }


def load_oracle_cache(
    dataset_dir: str | Path,
    metadata: pl.DataFrame,
    indices: np.ndarray,
    modulations: Iterable[str],
) -> dict | None:
    dataset_dir = Path(dataset_dir)
    parquet_path = dataset_dir / ORACLE_CACHE_PARQUET
    json_path = dataset_dir / ORACLE_CACHE_JSON
    labels = list(modulations)
    invalid = _oracle_cache_invalid_reason(parquet_path, json_path, metadata, labels)
    if invalid is not None:
        return None

    cache = pl.read_parquet(parquet_path)
    requested = pl.DataFrame({"signal_id": _signal_ids(metadata)[indices]})
    rows = requested.join(cache, on="signal_id", how="left")
    if rows["oracle_pred"].null_count() or rows["oracle_nll_bits"].null_count():
        return None

    label_to_id = {label: i for i, label in enumerate(labels)}
    y_true = np.asarray([label_to_id[str(v)] for v in metadata[indices]["modulation"].to_list()], dtype=int)
    y_pred = np.asarray([label_to_id[str(v)] for v in rows["oracle_pred"].to_list()], dtype=int)
    nll_bits = rows["oracle_nll_bits"].to_numpy().astype(np.float32, copy=False)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "confusion": confusion_matrix(y_true, y_pred, labels=np.arange(len(labels))),
        "y_true": y_true,
        "y_pred": y_pred,
        "nll_bits": nll_bits,
        "labels": labels,
    }


def oracle_cache_status(
    dataset_dir: str | Path,
    metadata: pl.DataFrame,
    modulations: Iterable[str],
) -> str:
    dataset_dir = Path(dataset_dir)
    invalid = _oracle_cache_invalid_reason(
        dataset_dir / ORACLE_CACHE_PARQUET,
        dataset_dir / ORACLE_CACHE_JSON,
        metadata,
        list(modulations),
    )
    return "valid" if invalid is None else invalid


def _oracle_cache_invalid_reason(
    parquet_path: Path,
    json_path: Path,
    metadata: pl.DataFrame,
    labels: list[str],
) -> str | None:
    if not parquet_path.exists() or not json_path.exists():
        return "missing oracle cache"
    try:
        cache_meta = json.loads(json_path.read_text(encoding="utf-8"))
        cache = pl.read_parquet(parquet_path)
    except Exception as exc:
        return f"unreadable oracle cache: {exc}"

    if cache_meta.get("oracle_cache_version") != ORACLE_CACHE_VERSION:
        return "oracle cache version mismatch"
    if cache_meta.get("labels") != labels:
        return "oracle cache label order mismatch"
    if int(cache_meta.get("row_count", -1)) != len(metadata):
        return "oracle cache row count mismatch"

    required = {"signal_id", "modulation", "oracle_pred", "oracle_nll_bits"}
    missing = sorted(required - set(cache.columns))
    if missing:
        return f"oracle cache missing columns: {', '.join(missing)}"
    if len(cache) != len(metadata):
        return "oracle cache prediction row count mismatch"

    metadata_signal_ids = _signal_ids(metadata)
    cache_signal_ids = cache["signal_id"].to_numpy()
    if len(set(metadata_signal_ids.tolist())) != len(metadata_signal_ids):
        return "metadata signal_id values are not unique"
    if len(set(cache_signal_ids.tolist())) != len(cache_signal_ids):
        return "oracle cache signal_id values are not unique"
    if set(cache_signal_ids.tolist()) != set(metadata_signal_ids.tolist()):
        return "oracle cache signal_id coverage mismatch"
    if set(cache["oracle_pred"].unique().to_list()) - set(labels):
        return "oracle cache contains unknown predicted labels"
    return None


def _signal_ids(metadata: pl.DataFrame) -> np.ndarray:
    if "signal_id" in metadata.columns:
        return metadata["signal_id"].to_numpy().astype(np.int64, copy=False)
    return np.arange(len(metadata), dtype=np.int64)


def _metadata_compensated_samples(
    x: np.ndarray,
    metadata: dict,
    config: OracleConfig,
) -> list[np.ndarray]:
    symbol_period = int(metadata.get("symbol_period", 1))
    upsample_factor = int(metadata.get("upsample_factor", round(float(metadata["osr"]))))
    downsample_factor = int(metadata.get("downsample_factor", 1))
    osr = float(metadata.get("osr", symbol_period * upsample_factor / downsample_factor))
    x = _undo_channel(x, metadata)
    x = _undo_sto(x, float(metadata.get("sto", 0.0)), osr)
    x = _undo_cfo_cpo(x, float(metadata["cfo"]), float(metadata.get("cpo", 0.0)))

    if symbol_period > 1:
        # Invert stage 2: undo waveform resample back to symbol_period samp/sym.
        if upsample_factor != 1 or downsample_factor != 1:
            x = signal.resample_poly(x, downsample_factor, upsample_factor)
        # Invert stage 1: matched filter designed at symbol_period samp/sym.
        x = _matched_filter(x, symbol_period, 1, float(metadata["ebw"]), str(metadata["modulation"]))
        sample_stride = symbol_period
    else:
        # Classic single-stage path: SRRC at upsample_factor samp/sym.
        x = _matched_filter(x, upsample_factor, downsample_factor, float(metadata["ebw"]), str(metadata["modulation"]))
        sample_stride = upsample_factor

    sampled = []
    for offset in range(sample_stride):
        symbols = x[offset::sample_stride]
        if len(symbols) > 2 * config.edge_symbols:
            symbols = symbols[config.edge_symbols : -config.edge_symbols]
        if len(symbols):
            sampled.append(_unit_power(symbols))
    return sampled


def _undo_channel(x: np.ndarray, metadata: dict) -> np.ndarray:
    if str(metadata.get("channel", "awgn")) == "awgn" or int(metadata.get("channel_n_taps", 0)) == 0:
        return x

    delays = np.asarray(_json_list(metadata.get("channel_tap_delays", "[]")), dtype=int)
    real = np.asarray(_json_list(metadata.get("channel_tap_real", "[]")), dtype=float)
    imag = np.asarray(_json_list(metadata.get("channel_tap_imag", "[]")), dtype=float)
    if len(delays) == 0:
        return x
    taps = real + 1j * imag
    h = np.zeros(int(np.max(delays)) + 1, dtype=np.complex128)
    h[delays] = taps
    return signal.lfilter([1.0], h, x)


def _json_list(value: object) -> list:
    import json

    if isinstance(value, str):
        return json.loads(value)
    return list(value)


def _undo_cfo_cpo(x: np.ndarray, cfo: float, cpo: float) -> np.ndarray:
    n = np.arange(len(x))
    return x * np.exp(-1j * 2 * np.pi * (cfo * n + cpo))


def _undo_sto(x: np.ndarray, sto_symbols: float, osr: float) -> np.ndarray:
    if abs(sto_symbols) < 1e-12:
        return x
    n = np.arange(len(x))
    shifted = n - sto_symbols * osr
    real = np.interp(shifted, n, np.real(x), left=0.0, right=0.0)
    imag = np.interp(shifted, n, np.imag(x), left=0.0, right=0.0)
    return real + 1j * imag


def _matched_filter(
    x: np.ndarray,
    upsample_factor: int,
    downsample_factor: int,
    ebw: float,
    modulation: str,
) -> np.ndarray:
    if downsample_factor > 1:
        x = signal.resample_poly(x, downsample_factor, 1)
    if modulation == "MSK":
        return x
    taps = srrc_filter(upsample_factor, ebw)
    return signal.convolve(x, taps[::-1], mode="same")


def _unit_power(x: np.ndarray) -> np.ndarray:
    return x / np.sqrt(max(float(np.mean(np.abs(x) ** 2)), np.finfo(float).eps))


def _symbol_noise_variance(metadata: dict, config: OracleConfig) -> float:
    snr_linear = 10 ** (float(metadata["snr_db"]) / 10)
    osr = float(metadata.get("osr", float(metadata.get("upsample_factor", 1)) / float(metadata.get("downsample_factor", 1))))
    ebw = float(metadata.get("ebw", 1.0))
    return max(1.0 / (snr_linear * in_band_noise_fraction(osr, ebw)), config.min_noise_variance)


def _modulation_log_likelihood(samples_by_offset: list[np.ndarray], modulation: str, noise_variance: float) -> float:
    if modulation in ("pi/4-DQPSK", "DQPSK"):
        references = np.exp(1j * np.array([np.pi / 4, 3 * np.pi / 4, -np.pi / 4, -3 * np.pi / 4]))
        return _best_offset_likelihood(samples_by_offset, references, 2.0 * noise_variance, differential=True)
    if modulation == "MSK":
        references = np.exp(1j * np.array([np.pi / 2, -np.pi / 2]))
        return _best_offset_likelihood(samples_by_offset, references, 2.0 * noise_variance, differential=True)
    if modulation not in MODEMS:
        raise ValueError(f"Unsupported oracle modulation: {modulation}")
    return _best_offset_likelihood(samples_by_offset, MODEMS[modulation].points, noise_variance)


def _best_offset_likelihood(
    samples_by_offset: list[np.ndarray],
    constellation: np.ndarray,
    noise_variance: float,
    differential: bool = False,
) -> float:
    scores = []
    for samples in samples_by_offset:
        if differential:
            samples = samples[1:] * np.conj(samples[:-1])
            samples = samples / np.maximum(np.abs(samples), np.finfo(float).eps)
        if len(samples) == 0:
            continue
        distances = np.abs(samples[:, None] - constellation[None, :]) ** 2
        scores.append(float(np.sum(logsumexp(-distances / noise_variance, axis=1) - np.log(len(constellation)))))
    return max(scores) if scores else float("-inf")
