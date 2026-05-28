from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

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
    np.save(output / SIGNALS_FILE, signals)
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

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = int(self.indices[item])
        x = self._prepare_signal(self.signals[idx])
        y = self.label_to_id[str(self.labels[idx])]
        return torch.from_numpy(self._features(x)).float(), torch.tensor(y, dtype=torch.long)

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
        raise ValueError(f"Unsupported representation: {self.representation}")


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


def ordered_modulation_labels(observed_labels: list[str]) -> list[str]:
    labels = [label for label in README_MODULATION_ORDER if label in observed_labels]
    labels.extend(sorted(label for label in observed_labels if label not in labels))
    return labels
