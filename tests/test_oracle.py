import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl

from modreczoo.oracle import (
    ORACLE_CACHE_JSON,
    build_oracle_cache,
    evaluate_oracle,
    load_oracle_cache,
    oracle_predict,
    oracle_scores,
)
from modreczoo.simulation import MODULATIONS, generate_signal


def _metadata(modulation: str, snr_db: float = 30.0) -> dict:
    return {
        "modulation": modulation,
        "snr_db": snr_db,
        "cfo": 0.0004,
        "cpo": 0.37,
        "sto": 0.2,
        "upsample_factor": 8,
        "downsample_factor": 1,
        "osr": 8.0,
        "ebw": 0.35,
        "n_samples": 2048,
        "channel": "awgn",
        "channel_n_taps": 0,
    }


def _signal(modulation: str, snr_db: float = 30.0, seed: int = 1) -> tuple[np.ndarray, dict]:
    metadata = _metadata(modulation, snr_db)
    result = generate_signal(
        modulation=modulation,
        snr_db=metadata["snr_db"],
        cfo=metadata["cfo"],
        cpo=metadata["cpo"],
        sto=metadata["sto"],
        upsample_factor=metadata["upsample_factor"],
        downsample_factor=metadata["downsample_factor"],
        ebw=metadata["ebw"],
        n_samples=metadata["n_samples"],
        channel=metadata["channel"],
        rng=np.random.default_rng(seed),
        debug=False,
    )
    return result["signal"], metadata


def test_oracle_predicts_high_snr_awgn_modulations_without_bits() -> None:
    for seed, modulation in enumerate(MODULATIONS, start=1):
        x, metadata = _signal(modulation, seed=seed)
        pred, scores = oracle_predict(x, metadata)
        assert pred == modulation
        assert set(scores) == set(MODULATIONS)


def test_oracle_uses_nuisance_metadata() -> None:
    x, metadata = _signal("16QAM")
    correct_scores = oracle_scores(x, metadata)

    wrong_metadata = dict(metadata)
    wrong_metadata.update({"cfo": 0.0, "cpo": 0.0, "sto": 0.0})
    wrong_scores = oracle_scores(x, wrong_metadata)

    assert correct_scores["16QAM"] > wrong_scores["16QAM"]


def test_oracle_accepts_dqpsk_alias() -> None:
    x, metadata = _signal("DQPSK")
    pred, _ = oracle_predict(x, metadata, modulations=("DQPSK", "4PSK", "8PSK"))
    assert pred == "DQPSK"


def test_evaluate_oracle_returns_confusion_and_accuracy() -> None:
    signals, rows = [], []
    for seed, modulation in enumerate(("2PSK", "8PSK", "16QAM", "MSK"), start=1):
        x, metadata = _signal(modulation, seed=seed)
        signals.append(x)
        rows.append(metadata)

    result = evaluate_oracle(
        signals=np.stack(signals),
        metadata=pl.DataFrame(rows),
        modulations=("2PSK", "8PSK", "16QAM", "MSK"),
    )
    assert np.isclose(result["accuracy"], 1.0)
    assert result["confusion"].shape == (4, 4)
    assert result["y_true"].tolist() == result["y_pred"].tolist()


def _cache_dataset() -> tuple[np.ndarray, pl.DataFrame, list[str]]:
    labels = ["2PSK", "4PSK", "16QAM"]
    signals, rows = [], []
    for signal_id, modulation in enumerate(labels):
        x, metadata = _signal(modulation, seed=signal_id + 10)
        metadata["signal_id"] = signal_id
        signals.append(x)
        rows.append(metadata)
    return np.stack(signals), pl.DataFrame(rows), labels


def test_oracle_cache_round_trip_loads_index_subset(tmp_path: Path) -> None:
    signals, metadata, labels = _cache_dataset()
    build_oracle_cache(tmp_path, signals, metadata, labels)

    result = load_oracle_cache(tmp_path, metadata, np.asarray([2, 0], dtype=np.int64), labels)

    assert result is not None
    assert result["labels"] == labels
    assert result["y_true"].tolist() == [2, 0]
    assert result["y_pred"].shape == (2,)
    assert result["nll_bits"].shape == (2,)
    assert np.all(np.isfinite(result["nll_bits"]))


def test_missing_oracle_cache_returns_none(tmp_path: Path) -> None:
    _, metadata, labels = _cache_dataset()

    result = load_oracle_cache(tmp_path, metadata, np.asarray([0, 1], dtype=np.int64), labels)

    assert result is None


def test_oracle_cache_label_mismatch_is_invalid(tmp_path: Path) -> None:
    signals, metadata, labels = _cache_dataset()
    build_oracle_cache(tmp_path, signals, metadata, labels)
    meta_path = tmp_path / ORACLE_CACHE_JSON
    cache_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cache_meta["labels"] = list(reversed(cache_meta["labels"]))
    meta_path.write_text(json.dumps(cache_meta), encoding="utf-8")

    result = load_oracle_cache(tmp_path, metadata, np.asarray([0, 1], dtype=np.int64), labels)

    assert result is None


if __name__ == "__main__":
    test_oracle_predicts_high_snr_awgn_modulations_without_bits()
    test_oracle_uses_nuisance_metadata()
    test_oracle_accepts_dqpsk_alias()
    test_evaluate_oracle_returns_confusion_and_accuracy()
    with TemporaryDirectory() as tmp:
        test_oracle_cache_round_trip_loads_index_subset(Path(tmp))
    with TemporaryDirectory() as tmp:
        test_missing_oracle_cache_returns_none(Path(tmp))
    with TemporaryDirectory() as tmp:
        test_oracle_cache_label_mismatch_is_invalid(Path(tmp))
