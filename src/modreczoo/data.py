from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from modreczoo.auxiliary import MetadataTargetEncoder
from modreczoo.features import csp_canonical_features, csp_expert_features, iq_features
from modreczoo.models import representation_for_model
from modreczoo.transforms import (
    complex_channels,
    frequency_channels,
    normalize_signal,
    remove_empirical_cfo,
    spectrogram_channels,
)


SIGNALS_FILE = "signals.npy"
EXTRAS_FILE = "extras.npz"
METADATA_FILE = "metadata.parquet"
README_MODULATION_ORDER = ("2PSK", "4PSK", "8PSK", "pi/4-DQPSK", "16QAM", "64QAM", "256QAM", "MSK")


def save_dataset(
    output_dir: str,
    signals: np.ndarray,
    metadata: pl.DataFrame,
    extras: dict[str, np.ndarray] | None = None,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target = output / SIGNALS_FILE
    already_memmapped = isinstance(signals, np.memmap) and Path(signals.filename).resolve() == target.resolve()
    if not already_memmapped:
        np.save(target, signals)
    if extras:
        np.savez(output / EXTRAS_FILE, **extras)
    metadata.write_parquet(output / METADATA_FILE)


def load_dataset(output_dir: str) -> tuple[np.ndarray, pl.DataFrame]:
    output = Path(output_dir)
    signals = np.load(output / SIGNALS_FILE, mmap_mode="r")
    metadata = ensure_symbol_metadata(pl.read_parquet(output / METADATA_FILE))
    return signals, metadata


def ensure_symbol_metadata(metadata: pl.DataFrame) -> pl.DataFrame:
    columns = set(metadata.columns)
    if "symbol_period" not in columns:
        metadata = metadata.with_columns(pl.lit(1).alias("symbol_period"))
        columns.add("symbol_period")
    if "symbol_rate" in columns:
        return metadata
    if {"osr", "symbol_period"} <= columns:
        return metadata.with_columns(
            (1.0 / (pl.col("symbol_period").cast(pl.Float64) * pl.col("osr").cast(pl.Float64))).alias("symbol_rate")
        )
    if {"upsample_factor", "downsample_factor", "symbol_period"} <= columns:
        return metadata.with_columns(
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
        label_to_id: dict[str, int],
        representation: str,
        channel_format: str,
        remove_cfo: bool,
        cfo_estimator: str,
        spectrogram_freq_bins: int = 64,
        spectrogram_time_bins: int = 64,
        spectrogram_nperseg: int = 64,
        spectrogram_noverlap: int = 48,
        spectrogram_window: str = "hann",
        n_samples: int | None = None,
        auxiliary_encoders: tuple[MetadataTargetEncoder, ...] = (),
    ) -> None:
        self.signals = signals
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
        self.auxiliary_targets = {
            encoder.column: encoder.encode(metadata)
            for encoder in auxiliary_encoders
        }

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple:
        idx = int(self.indices[item])
        x = self._prepare_signal(self.signals[idx])
        y = self.label_to_id[str(self.labels[idx])]
        features = torch.from_numpy(self._features(x)).float()
        label = torch.tensor(y, dtype=torch.long)
        if not self.auxiliary_targets:
            return features, label
        auxiliary = {
            name: torch.tensor(targets[idx], dtype=torch.long)
            for name, targets in self.auxiliary_targets.items()
        }
        return features, label, auxiliary

    def _prepare_signal(self, x: np.ndarray) -> np.ndarray:
        x = normalize_signal(x)
        if self.n_samples is not None:
            x = _resize_1d(x, self.n_samples)
        if self.remove_cfo:
            x = remove_empirical_cfo(x, estimator=self.cfo_estimator)
        return x

    def _features(self, x: np.ndarray) -> np.ndarray:
        if self.representation == "time":
            return complex_channels(x, self.channel_format)
        if self.representation == "frequency":
            return frequency_channels(x, self.channel_format)
        if self.representation == "spectrogram":
            return spectrogram_channels(
                x,
                channel_format=self.channel_format,
                freq_bins=self.spectrogram_freq_bins,
                time_bins=self.spectrogram_time_bins,
                nperseg=self.spectrogram_nperseg,
                noverlap=self.spectrogram_noverlap,
                window=self.spectrogram_window,
            )
        if self.representation == "iq_features":
            return iq_features(x)
        if self.representation == "csp_features":
            return csp_expert_features(x)
        if self.representation == "csp_canonical":
            return csp_canonical_features(x)
        if self.representation == "joint_csp":
            return _joint_csp_channels(x, self.channel_format)
        raise ValueError(f"Unsupported representation: {self.representation}")


def _joint_csp_channels(x: np.ndarray, channel_format: str) -> np.ndarray:
    """Return (6 + N_CSP_EXPERT_FEATURES, N) array for the JointCSPCNN.

    Channels 0–5: complex_powers of the signal (Re/Im of z, z², z⁴).
    Channels 6–112: 107 CSP expert features broadcast across the time axis so
    that ``JointCSPCNN.forward()`` can recover them via ``x[:, 6:, 0]``.

    The complex_powers are NOT per-power RMS-normalized (ri_norm removed).
    z is already unit-power from ``normalize_signal()``, and z², z⁴ retain
    their amplitude kurtosis, which is class-discriminative information.
    """
    from modreczoo.transforms import complex_powers_channels
    from modreczoo.features import csp_expert_features

    sig = complex_powers_channels(x)                        # (6, N)
    csp = csp_expert_features(x)                            # (107,)
    csp_bc = np.broadcast_to(csp[:, None], (csp.shape[0], sig.shape[-1]))  # (107, N)
    return np.concatenate([sig, csp_bc], axis=0).astype(np.float32)        # (113, N)


def _resize_1d(x: np.ndarray, n_samples: int) -> np.ndarray:
    if x.shape[0] > n_samples:
        return x[:n_samples]
    if x.shape[0] < n_samples:
        return np.pad(x, (0, n_samples - x.shape[0]), mode="constant")
    return x


def get_data_loader(
    signals: np.ndarray,
    metadata: pl.DataFrame,
    indices: np.ndarray,
    label_to_id: dict[str, int],
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
    n_samples: int | None = None,
    auxiliary_encoders: tuple[MetadataTargetEncoder, ...] = (),
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
        auxiliary_encoders=auxiliary_encoders,
    )
    if num_workers > 0:
        loader_kwargs.setdefault("persistent_workers", persistent_workers)
    loader_kwargs.setdefault("pin_memory", pin_memory)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **loader_kwargs)


def ordered_modulation_labels(observed_labels: list[str]) -> list[str]:
    labels = [label for label in README_MODULATION_ORDER if label in observed_labels]
    labels.extend(sorted(label for label in observed_labels if label not in labels))
    return labels


# ── Online dataset ─────────────────────────────────────────────────────────────

_ONLINE_DEFAULT_PARAMS: dict = {
    "n_samples": 32768,
    "snr_range": (0.0, 30.0),
    "cfo_range": (-1 / 1000, 1 / 1000),
    "cpo_range": (0.0, 1.0),
    "sto_range": (-0.5, 0.5),
    "symbol_period_range": (2, 16),
    "osr_range": (1.0, 2.0),
    "ebw_range": (0.1, 1.0),
    "channel": "awgn",
}

_ONLINE_MODULATIONS: tuple[str, ...] = (
    "2PSK", "4PSK", "8PSK", "pi/4-DQPSK", "16QAM", "64QAM", "256QAM", "MSK"
)


class OnlineModrecDataset(IterableDataset):
    """Generates modulation signals on-the-fly using the simulation distribution.

    Each call to ``__iter__`` yields a fresh, newly-simulated signal. No two
    epochs see the same realizations, providing effective data augmentation over
    the joint parameter space (SNR, CFO, CPO, STO, pulse shape, OSR).

    The parameter distribution matches ``baseline_32768_200k`` by default
    (SNR ∈ [0,30] dB, AWGN only, random uniform sampling). Pass custom
    ``params`` to change any parameter range.

    Use with ``num_workers ≥ 4`` for throughput: each DataLoader worker gets
    its own seeded RNG derived from ``(worker_id * 2_654_435_761 + seed) % 2³²``,
    preventing correlated samples across workers.  At K=32768, typical throughput
    is ~200–400 signals/s per worker (CSP features: ~3 ms; signal gen: ~2 ms).

    Args:
        label_to_id: mapping from modulation string to integer class label.
        representation: feature representation (e.g. ``"time"``, ``"joint_csp"``).
        channel_format: channel layout for time/frequency/spectrogram reps.
        steps_per_epoch: number of samples to yield per ``__iter__`` call.
            Use with ``DataLoader(drop_last=True)`` so every epoch has the same
            number of batches. Default: 140_000 (matches baseline_32768_200k
            train-split size).
        n_samples: signal length in samples. Default: 32768.
        params: override any parameter from ``_ONLINE_DEFAULT_PARAMS``.
        seed: global RNG seed (worker RNGs derived from this).
    """

    def __init__(
        self,
        label_to_id: dict[str, int],
        representation: str,
        channel_format: str,
        steps_per_epoch: int = 140_000,
        n_samples: int = 32768,
        params: dict | None = None,
        seed: int | None = None,
    ) -> None:
        self.label_to_id = label_to_id
        self.representation = representation
        self.channel_format = channel_format
        self.steps_per_epoch = steps_per_epoch
        self.n_samples = n_samples
        self.params = {**_ONLINE_DEFAULT_PARAMS, **(params or {})}
        self.seed = seed
        self.modulations = sorted(label_to_id.keys())

    def _sample_signal(self, rng: np.random.Generator) -> tuple[np.ndarray, int]:
        """Sample one (signal, label) pair using the provided RNG."""
        from modreczoo.simulation import generate_signal, rational_resample_factors

        p = self.params
        modulation = self.modulations[rng.integers(len(self.modulations))]

        snr_lo, snr_hi = p["snr_range"]
        cfo_lo, cfo_hi = p["cfo_range"]
        cpo_lo, cpo_hi = p["cpo_range"]
        sto_lo, sto_hi = p["sto_range"]
        sp_lo, sp_hi = p["symbol_period_range"]
        osr_lo, osr_hi = p["osr_range"]
        ebw_lo, ebw_hi = p["ebw_range"]

        snr_db = float(rng.uniform(snr_lo, snr_hi))
        cfo    = float(rng.uniform(cfo_lo, cfo_hi))
        cpo    = float(rng.uniform(cpo_lo, cpo_hi))
        sto    = float(rng.uniform(sto_lo, sto_hi))
        sp     = int(rng.integers(sp_lo, sp_hi))
        osr    = float(rng.uniform(osr_lo, osr_hi))
        ebw    = float(rng.uniform(ebw_lo, ebw_hi))

        up, down = rational_resample_factors(osr, min_ratio=2.0 / sp)
        result = generate_signal(
            modulation=modulation, snr_db=snr_db, cfo=cfo, sto=sto, cpo=cpo,
            upsample_factor=up, downsample_factor=down, ebw=ebw,
            symbol_period=sp, n_samples=self.n_samples, channel=p["channel"], rng=rng,
        )
        sig = result["signal"].astype(np.complex64)
        label = self.label_to_id[modulation]
        return sig, label

    def _features(self, x: np.ndarray) -> np.ndarray:
        """Apply representation transform to a unit-power signal."""
        x = normalize_signal(x)
        if self.representation == "time":
            return complex_channels(x, self.channel_format)
        if self.representation == "frequency":
            return frequency_channels(x, self.channel_format)
        if self.representation == "csp_features":
            return csp_expert_features(x)
        if self.representation == "joint_csp":
            return _joint_csp_channels(x, self.channel_format)
        raise ValueError(f"OnlineModrecDataset: unsupported representation '{self.representation}'")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            n_workers = worker_info.num_workers
            # Each worker generates its share of steps; derive independent seed.
            seed = ((self.seed or 0) + worker_id * 2_654_435_761) % (2 ** 32)
            rng = np.random.default_rng(seed)
            steps = self._steps_for_worker(worker_id, n_workers)
        else:
            rng = np.random.default_rng(self.seed)
            steps = self.steps_per_epoch

        for _ in range(steps):
            sig, label = self._sample_signal(rng)
            feat = self._features(sig)
            yield torch.from_numpy(feat).float(), torch.tensor(label, dtype=torch.long)

    def _steps_for_worker(self, worker_id: int, n_workers: int) -> int:
        """Distribute steps_per_epoch evenly across workers (last gets remainder)."""
        base = self.steps_per_epoch // n_workers
        return base + (self.steps_per_epoch % n_workers if worker_id == n_workers - 1 else 0)


def get_online_data_loader(
    label_to_id: dict[str, int],
    model_name: str,
    channel_format: str = "real_imag",
    batch_size: int = 64,
    num_workers: int = 4,
    steps_per_epoch: int = 140_000,
    n_samples: int = 32768,
    params: dict | None = None,
    seed: int | None = None,
    **loader_kwargs,
) -> DataLoader:
    """DataLoader wrapping :class:`OnlineModrecDataset`.

    Example::

        dl = get_online_data_loader(
            label_to_id={"2PSK": 0, ...},
            model_name="joint_csp_cnn",
            batch_size=32,
            num_workers=8,
        )
        for feats, labels in dl:  # feats: (32, 113, 32768)
            ...
    """
    dataset = OnlineModrecDataset(
        label_to_id=label_to_id,
        representation=representation_for_model(model_name),
        channel_format=channel_format,
        steps_per_epoch=steps_per_epoch,
        n_samples=n_samples,
        params=params,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        **loader_kwargs,
    )
