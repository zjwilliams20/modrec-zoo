import numpy as np
import polars as pl
import torch

from modreczoo.auxiliary import build_metadata_target_encoders
from modreczoo.data import get_data_loader
from modreczoo.evaluation import evaluate
from modreczoo.models.cnn import CNN1D
from modreczoo.models.wrappers import ModelWithPreprocessor, MultiTaskModel
from modreczoo.preprocessing import make_preprocessor


def _toy_dataset() -> tuple[np.ndarray, pl.DataFrame, dict[str, int]]:
    labels = np.array(["A", "B", "A", "B"])
    rng_i = np.random.default_rng(0)
    rng_q = np.random.default_rng(1)
    signals = (rng_i.normal(size=(4, 32)) + 1j * rng_q.normal(size=(4, 32))).astype(np.complex64)
    metadata = pl.DataFrame(
        {
            "modulation": labels,
            "snr_db": [0.0, 4.0, 8.0, 12.0],
            "channel": ["awgn", "rayleigh", "awgn", "rayleigh"],
        }
    )
    return signals, metadata, {"A": 0, "B": 1}


def test_default_loader_keeps_two_field_batches() -> None:
    signals, metadata, label_to_id = _toy_dataset()

    batch = next(
        iter(
            get_data_loader(
                signals,
                metadata,
                np.arange(4),
                label_to_id,
                "time_cnn",
                batch_size=2,
                pin_memory=False,
            )
        )
    )

    assert len(batch) == 2


def test_auxiliary_loader_adds_encoded_metadata_targets() -> None:
    signals, metadata, label_to_id = _toy_dataset()
    encoders = build_metadata_target_encoders(metadata, np.arange(4), ["snr_db", "channel"], n_bins=2)

    xb, yb, auxiliary = next(
        iter(
            get_data_loader(
                signals,
                metadata,
                np.arange(4),
                label_to_id,
                "time_cnn",
                batch_size=2,
                pin_memory=False,
                auxiliary_encoders=encoders,
            )
        )
    )

    assert xb.shape == (2, 2, 32)
    assert yb.shape == (2,)
    assert set(auxiliary) == {"snr_db", "channel"}
    assert auxiliary["snr_db"].dtype == torch.long


def test_learned_fir_starts_as_identity() -> None:
    preprocessor, out_channels = make_preprocessor("learned_fir", in_channels=2, kernel_size=3)
    x = torch.randn(3, 2, 16)

    assert out_channels == 2
    assert torch.allclose(preprocessor(x), x)


def test_preprocessed_multitask_model_keeps_primary_forward() -> None:
    preprocessor, _ = make_preprocessor("radio_transform", in_channels=2)
    model = MultiTaskModel(
        ModelWithPreprocessor(CNN1D(n_classes=2, in_channels=2), preprocessor),
        {"snr_db": 3},
    )
    x = torch.randn(4, 2, 64)

    primary = model(x)
    outputs = model.forward_all(x)
    loss = outputs["modulation"].sum() + outputs["snr_db"].sum()
    loss.backward()

    assert primary.shape == (4, 2)
    assert outputs["modulation"].shape == (4, 2)
    assert outputs["snr_db"].shape == (4, 3)
    assert model.aux_heads["snr_db"].weight.grad is not None


def test_evaluate_reports_auxiliary_accuracy() -> None:
    signals, metadata, label_to_id = _toy_dataset()
    encoders = build_metadata_target_encoders(metadata, np.arange(4), ["snr_db"], n_bins=2)
    loader = get_data_loader(
        signals,
        metadata,
        np.arange(4),
        label_to_id,
        "time_cnn",
        batch_size=2,
        pin_memory=False,
        auxiliary_encoders=encoders,
    )
    model = MultiTaskModel(CNN1D(n_classes=2, in_channels=2), {"snr_db": encoders[0].n_classes})

    metrics = evaluate(
        model,
        loader,
        torch.device("cpu"),
        n_classes=2,
        auxiliary_tasks={"snr_db": encoders[0].n_classes},
    )

    assert metrics["auxiliary"]["snr_db"]["n"] == 4
    assert 0.0 <= metrics["auxiliary"]["snr_db"]["accuracy"] <= 1.0
