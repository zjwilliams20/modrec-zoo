# CLAUDE.md

## Behavior Rules

* Prioritize simplicity and less code over backwards compatibility.
* Always prefer less code to more flexibility unless specifically requested.
* Prefer terminal-friendly fixed-width text tables over Markdown pipe tables.
* Use short lists when table rows contain long explanations.
* Liberally validate code syntax with `python -m py_compile path/to/file.py` after edits.
* This is a research repository. Code reliability and robust unit-testing is not as important as simplicity.
* A partial test suite exists in `tests/` — run with `uv run pytest tests/`.

## Project Overview

ModRec Zoo is a deep learning research playground for **modulation recognition** (classifying RF signals by modulation type). Compares generalization techniques — including meta-learning — across neural architectures and signal representations. Experiments tracked in MLflow.

8 modulation classes: BPSK, QPSK, 8-PSK, π/4-DQPSK, 16-QAM, 64-QAM, 256-QAM, MSK.

## Key Commands

```bash
uv sync
uv run modreczoo-simulate generate --output-dir data/awgn_sobol --n-signals 1000 --channel awgn --sampler sobol --seed 0
uv run modreczoo-train --dataset-dir data/awgn_sobol
uv run modreczoo-train --command sweep --dataset-dir data/awgn_snr0_30 --models resnet_1d dilated_cnn_1d --sweep-channel-formats real_imag mag
uv run mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --host 127.0.0.1 --port 5000
uv run basedpyright src/
bash scripts/csp-sweep.sh
```

## File Map

```
src/modreczoo/
  simulation.py   — synthetic I/Q generation; writes signals.npy + extras.npz + metadata.parquet
  data.py         — ModrecDataset, load_dataset(), channel format transforms in __getitem__
  training.py     — train/val/test loop, MLflow logging, CHANNEL_FORMATS, input_channels_for()
  evaluation.py   — per-class metrics, accuracy-by-SNR, calibration, bootstrap CI
  reporting.py    — HTML performance explorer, prediction/error tables
  plotting.py     — confusion matrix, SNR curves, calibration diagrams
  models/
    registry.py   — model catalog, make_model() factory, representations, required channel formats
    cnn.py        — CNN1D, CNN2D
    resnet.py     — ResNet1D, ResNet2D and residual blocks
    mlp.py        — FeatureMLP
    complex.py    — ComplexCNN1D
    dilated.py    — DilatedCNN1D
    transformer.py — PatchTransformer1D
    multiscale.py — MultiScalePyramidNet
    streams.py    — APFNet, MultiStreamNet
  cli/
    simulate.py   — modreczoo-simulate
    train.py      — modreczoo-train (accepts --config YAML via jsonargparse)
    plot_spectrograms.py — modreczoo-plot-spectrograms
```

## Dataset Metadata Columns

Key signal-parameter columns written to `metadata.parquet` by `simulation.py`:

- `symbol_period` — samples/symbol at the pulse-shaping stage (integer ≥ 2)
- `upsample_factor`, `downsample_factor` — rational resampling applied after pulse shaping; approximate a target drawn from `osr_range`
- `osr` — realized resampling ratio: `upsample_factor / downsample_factor` (not samples/symbol)
- `symbol_rate` — normalized symbol rate at the output sample rate: `1 / (symbol_period * osr)`
- Effective samples/symbol at output: `symbol_period * osr`

`osr` and `symbol_rate` are decoupled: `osr` captures only the resampling stage; `symbol_period` captures the pulse-shaping resolution.

## Channel Formats

Defined in `training.py`: `real_imag`, `mag`, `mag_phase`, `mag_inst_freq`, `differential_complex`, `apf`, `complex_powers`, `scf`. Input channel count comes from `input_channels_for(representation, channel_format)`. Models needing a fixed format (e.g. `scf_resnet` → `scf`) declare it in `MODEL_REQUIRED_CHANNEL_FORMATS` and it auto-overrides `--channel-format`.

## MLflow

SQLite at `mlflow/mlflow.db`, artifacts at `mlflow/artifacts/`, staging at `mlflow/staging/<run_id>/`.

## Adding a Model

1. Implement `nn.Module` under `models/`.
2. Register in `MODEL_REPRESENTATIONS` (and `MODEL_REQUIRED_CHANNEL_FORMATS` if format is fixed) in `registry.py`.
3. Add branch in `make_model()`.
4. Add name to `MODEL_NAMES` in `training.py`.
