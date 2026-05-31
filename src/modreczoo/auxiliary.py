from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
import torch


MISSING_CLASS = "<missing>"
UNKNOWN_CLASS = "<unknown>"


@dataclass(frozen=True)
class MetadataTargetEncoder:
    column: str
    classes: tuple[str, ...]
    bin_edges: tuple[float, ...] | None = None

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def encode(self, metadata: pl.DataFrame) -> np.ndarray:
        values = metadata[self.column].to_numpy()
        if self.bin_edges is not None:
            return self._encode_binned(values)
        return self._encode_categorical(values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "classes": list(self.classes),
            "bin_edges": list(self.bin_edges) if self.bin_edges is not None else None,
            "n_classes": self.n_classes,
        }

    def _encode_binned(self, values: np.ndarray) -> np.ndarray:
        missing_id = self.classes.index(MISSING_CLASS)
        encoded = np.full(len(values), missing_id, dtype=np.int64)
        numbers, finite = _finite_float_values(values)
        encoded[finite] = np.digitize(numbers[finite], self.bin_edges, right=True)
        return encoded

    def _encode_categorical(self, values: np.ndarray) -> np.ndarray:
        class_to_id = {label: i for i, label in enumerate(self.classes)}
        missing_id = class_to_id[MISSING_CLASS]
        unknown_id = class_to_id[UNKNOWN_CLASS]
        encoded = np.empty(len(values), dtype=np.int64)
        for i, value in enumerate(values):
            if _is_missing(value):
                encoded[i] = missing_id
            else:
                encoded[i] = class_to_id.get(_format_value(value), unknown_id)
        return encoded


def build_metadata_target_encoders(
    metadata: pl.DataFrame,
    train_indices: np.ndarray,
    columns: list[str] | None,
    n_bins: int = 8,
) -> tuple[MetadataTargetEncoder, ...]:
    if not columns:
        return ()
    if n_bins < 2:
        raise ValueError("Expected at least two bins for numeric auxiliary metadata targets.")

    encoders = []
    seen = set()
    for column in columns:
        if column in seen:
            raise ValueError(f"Duplicate auxiliary metadata target: {column}.")
        if column == "modulation":
            raise ValueError("Use the primary classifier for modulation, not --aux-targets modulation.")
        if column not in metadata.columns:
            raise ValueError(f"Auxiliary metadata target {column!r} is not present in metadata.")
        seen.add(column)
        train_values = metadata[column].to_numpy()[train_indices]
        encoders.append(_build_encoder(column, train_values, n_bins))
    return tuple(encoders)


def auxiliary_class_counts(encoders: tuple[MetadataTargetEncoder, ...]) -> dict[str, int]:
    return {encoder.column: encoder.n_classes for encoder in encoders}


def auxiliary_target_info(encoders: tuple[MetadataTargetEncoder, ...]) -> list[dict[str, Any]]:
    return [encoder.to_dict() for encoder in encoders]


def unpack_batch(
    batch: tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None, dict[str, torch.Tensor] | None]:
    if len(batch) == 2:
        xb, yb = batch
        return xb, yb, None, None
    if len(batch) == 3:
        xb, yb, auxiliary = batch
        return xb, yb, auxiliary, None
    if len(batch) == 4:
        xb, yb, auxiliary, raw_meta = batch
        return xb, yb, auxiliary, raw_meta
    raise ValueError(f"Expected DataLoader batch with 2, 3, or 4 fields, got {len(batch)}.")


def _build_encoder(column: str, values: np.ndarray, n_bins: int) -> MetadataTargetEncoder:
    numbers, finite = _finite_float_values(values)
    if finite.any() and _looks_numeric(values):
        unique = np.unique(numbers[finite])
        if len(unique) > n_bins:
            edges = _quantile_edges(numbers[finite], n_bins)
            classes = (*_bin_labels(edges), MISSING_CLASS)
            return MetadataTargetEncoder(column, classes, tuple(float(x) for x in edges))
        classes = [_format_float(value) for value in unique]
        classes.extend([MISSING_CLASS, UNKNOWN_CLASS])
        return MetadataTargetEncoder(column, tuple(classes))

    classes = sorted({_format_value(v) for v in values if not _is_missing(v)})
    if not classes:
        raise ValueError(f"Auxiliary metadata target {column!r} has no observed training values.")
    classes.extend([MISSING_CLASS, UNKNOWN_CLASS])
    return MetadataTargetEncoder(column, tuple(classes))


def _looks_numeric(values: np.ndarray) -> bool:
    for value in values:
        if _is_missing(value):
            continue
        try:
            float(value)
        except (TypeError, ValueError):
            return False
    return True


def _finite_float_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    numbers = np.empty(len(values), dtype=np.float64)
    finite = np.zeros(len(values), dtype=bool)
    for i, value in enumerate(values):
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = np.nan
        numbers[i] = number
        finite[i] = np.isfinite(number)
    return numbers, finite


def _quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    return np.unique(np.quantile(values, quantiles))


def _bin_labels(edges: np.ndarray) -> tuple[str, ...]:
    if len(edges) == 0:
        return ("all",)
    labels = [f"<= {_format_float(edges[0])}"]
    labels.extend(
        f"({_format_float(left)}, {_format_float(right)}]"
        for left, right in zip(edges[:-1], edges[1:])
    )
    labels.append(f"> {_format_float(edges[-1])}")
    return tuple(labels)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False


def _format_value(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return _format_float(value)
    return str(value)


def _format_float(value: float) -> str:
    return f"{value:.6g}"
