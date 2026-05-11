# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ModRec Zoo is a deep learning research playground for **modulation recognition** (classifying RF communications signals by modulation type). The goal is to compare generalization techniques — including meta-learning — across various neural architectures and signal representations. Experiments track results in MLflow.

8 modulation classes: BPSK, QPSK, 8-PSK, π/4-DQPSK, 16-QAM, 64-QAM, 256-QAM, MSK.

## Setup & Common Commands

```bash
uv sync                         # install deps
uv run modreczoo-simulate generate --output-dir data/awgn_sobol --n-signals 1000 --channel awgn --sampler sobol --seed 0
uv run modreczoo-train --dataset-dir data/awgn_sobol
uv run modreczoo-train --command sweep --dataset-dir data/awgn_snr0_30 --models resnet_1d dilated_cnn_1d --sweep-channel-formats real_imag mag
uv run mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --host 127.0.0.1 --port 5000
bash scripts/csp-sweep.sh       # CSP-inspired architecture sweep (override env vars to configure)
```

Type checking:
```bash
uv run basedpyright src/
```

No test suite exists yet.

## Architecture

### Data flow

1. **`simulation.py`** — generates I/Q signals with controlled impairments (CFO, STO, OSR, EBW, channel). Writes `signals.npz` + `metadata.parquet` via `data.save_dataset`.
2. **`data.py`** — loads datasets, transforms raw I/Q into model inputs via `ModrecDataset`. The `channel_format` arg controls representation (real/imag, magnitude, phase, etc.) applied in `__getitem__`.
3. **`training.py`** — orchestrates train/val/test splits, the training loop, MLflow logging, evaluation, and artifact generation.
4. **`evaluation.py`** — per-class metrics, accuracy-by-SNR, calibration stats, bootstrap CI.
5. **`plotting.py`** — confusion matrix, SNR accuracy curve, reliability/calibration diagrams, input examples.

### Models (`src/modreczoo/models/`)

- **`registry.py`** — `make_model(name, ...)` factory and `MODEL_REPRESENTATIONS` / `MODEL_REQUIRED_CHANNEL_FORMATS` dicts. Models that require a specific channel format (e.g. `scf_resnet` → `scf`, `apf_net_1d` → `apf`) are declared here and automatically override `--channel-format`.
- **`baselines.py`** — `TimeCNN`, `ResNet1D`, `SpectrogramCNN`, `SpectrogramResNet`, `FeatureMLP`.
- **`complex.py`** — `ComplexCNN1D`.
- **`dilated.py`** — `DilatedCNN1D`.
- **`advanced.py`** — `PatchTransformer1D`, `MultiScalePyramidNet`, `APFNet`, `MultiLagNet`, `CyclicCAFNet`.

### Channel formats

`CHANNEL_FORMATS` in `training.py`: `real_imag`, `mag`, `mag_phase`, `mag_inst_freq`, `differential_complex`, `apf`, `complex_powers`, `scf`. The number of input channels is determined by `input_channels_for(representation, channel_format)`.

### MLflow

All runs go to `mlflow/mlflow.db` (SQLite) with artifacts under `mlflow/artifacts/`. Staging files land in `mlflow/staging/<run_id>/` before being uploaded. Run names are auto-generated from model + hyperparams; sweep runs include only the swept parameters in the name.

### CLI entrypoints

| Command | Module |
|---|---|
| `modreczoo-simulate` | `cli/simulate.py` |
| `modreczoo-train` | `cli/train.py` |
| `modreczoo-plot-spectrograms` | `cli/plot_spectrograms.py` |

`modreczoo-train` accepts a `--config` YAML file (via jsonargparse) for reproducible experiment configs.

### Adding a new model

1. Implement the `nn.Module` in an appropriate file under `models/`.
2. Add an entry to `MODEL_REPRESENTATIONS` in `registry.py` (and `MODEL_REQUIRED_CHANNEL_FORMATS` if it needs a fixed format).
3. Add a branch in `make_model()` in `registry.py`.
4. Add the name string to `MODEL_NAMES` in `training.py`.
