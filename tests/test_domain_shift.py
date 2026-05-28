from pathlib import Path

import numpy as np
import polars as pl
import torch

from modreczoo.domain_shift import DomainSpec, RunInfo, _last_linear, parse_domain_specs, reconstruct_split_indices
from modreczoo.data import save_dataset
from modreczoo.training import stratified_split


def test_parse_domain_specs_accepts_auto_and_paths() -> None:
    specs = parse_domain_specs(["train:auto", "channels=data/baseline_4096_channels"])

    assert specs == [
        DomainSpec("train", None, "train"),
        DomainSpec("channels", "data/baseline_4096_channels", None),
    ]


def test_last_linear_hook_captures_prelogit_embedding() -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.ReLU(),
        torch.nn.Linear(4, 2),
    )
    _name, layer = _last_linear(model)
    captured = []
    handle = layer.register_forward_pre_hook(lambda _m, inputs: captured.append(inputs[0].detach().clone()))
    try:
        _ = model(torch.ones(5, 3))
    finally:
        handle.remove()

    assert captured[0].shape == (5, 4)


def test_reconstruct_heldout_split_indices_matches_training_helper(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    labels = np.array(["A", "B"] * 20)
    metadata = pl.DataFrame({"signal_id": np.arange(len(labels)), "modulation": labels})
    signals = np.ones((len(labels), 16), dtype=np.complex64)
    save_dataset(str(dataset_dir), signals, metadata)
    run = RunInfo(
        run_id="r",
        run_name="run",
        artifact_dir=tmp_path,
        model_uri="",
        params={
            "dataset_dir": str(dataset_dir),
            "test_dataset_source": "heldout_split",
            "sample_frac": "1.0",
            "max_examples": "none",
            "seed": "7",
        },
        config={"train_frac": 0.7, "val_frac": 0.15},
        model_info={},
    )

    expected = stratified_split(labels, 0.7, 0.15, 7)

    assert reconstruct_split_indices(run, "train", labels)[1].tolist() == expected[0].tolist()
    assert reconstruct_split_indices(run, "val", labels)[1].tolist() == expected[1].tolist()
    assert reconstruct_split_indices(run, "test", labels)[1].tolist() == expected[2].tolist()
