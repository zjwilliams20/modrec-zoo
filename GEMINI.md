# Modulation Recognition (ModRec) Project Context

This project is a research playground for deep learning algorithms applied to **Modulation Recognition (ModRec)**. The primary goal is to compare various deep-learning-based generalization techniques (including meta-learning) and evaluate their performance across different environmental conditions (domains), specifically varying channel characteristics.

## Project Structure

- `src/modreczoo/`: Core package containing all logic.
    - `cli/`: Command-line interfaces for simulation, training, and plotting.
    - `models/`: Implementation of various neural network architectures.
        - `registry.py`: Central hub for model creation and configuration.
        - `baselines.py`: Simple models like 1D-CNNs, ResNets, and MLPs.
        - `advanced.py`: More complex models like Transformers and Pyramid Nets.
        - `complex.py`, `dilated.py`: Specialized convolutional layers.
    - `simulation.py`: Logic for generating synthetic RF signals with impairments (CFO, STO, OSR, etc.) and channel effects (AWGN, Rayleigh, Rician).
    - `data.py`: Data loading, preprocessing (normalization, CFO estimation/removal), and PyTorch `Dataset`/`DataLoader` implementations.
    - `training.py`: Orchestration of the training loop, MLflow logging, and evaluation.
    - `evaluation.py`: Detailed performance metrics (Accuracy by SNR, Calibration stats ECE/MCE, Bootstrap CI).
    - `plotting.py`: Visualization tools for signals, confusion matrices, and metrics.
- `scripts/`: Shell scripts for running sweeps and environment setup.
- `mlflow/`: Local directory for MLflow tracking (SQLite DB and artifacts).

## Main Technologies

- **Language:** Python 3.12+
- **Deep Learning:** PyTorch
- **Signal Processing:** NumPy, SciPy
- **Data Management:** Polars (for metadata), Parquet, NPZ (for signals)
- **Experiment Tracking:** MLflow
- **Package Management:** `uv`

## Building and Running

### Environment Setup
The project uses `uv` for dependency management.
```bash
uv sync
```

### Signal Simulation
Generate synthetic datasets for training/testing:
```bash
uv run modreczoo-simulate generate \
  --output-dir data/my_dataset \
  --n-signals 1000 \
  --channel awgn \
  --sampler sobol
```

### Model Training
Train models against a generated dataset:
```bash
uv run modreczoo-train --dataset-dir data/my_dataset --models resnet_1d time_cnn --epochs 10
```

### Experiment Tracking
Visualize results in the MLflow UI:
```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --host 127.0.0.1 --port 5000
```

## Development Conventions

- **Model Registration:** All new models should be added to the registry in `src/modreczoo/models/registry.py` to be accessible via the CLI.
- **Signal Representations:** The project supports multiple representations: `time`, `frequency`, `spectrogram`, `features`, and specialized ones like `scf` (Spectral Correlation Function).
- **Metadata Handling:** Metadata is managed using Polars and stored as Parquet files for efficiency.
- **Logging:** MLflow is the primary tool for logging parameters, metrics, and artifacts (plots, model checkpoints, summary tables).
- **CFO Handling:** The codebase includes several CFO estimation techniques (`lag_correlation`, `phase_slope`, `spectral_centroid`) which can be applied during training/inference.
- **Testing:** Scripts for basic tests and sweeps are located in the `scripts/` directory.

## Key Files for Reference

- `src/modreczoo/models/registry.py`: Defines available models and their required input formats.
- `src/modreczoo/simulation.py`: Contains the `DEFAULT_PARAMS` and logic for signal synthesis.
- `src/modreczoo/data.py`: Defines `ModrecDataset` and preprocessing steps.
- `src/modreczoo/training.py`: The main training workflow and MLflow configuration.
