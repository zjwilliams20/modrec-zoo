# AGENTS.md

## Project Preferences

* Prioritize simplicity and less code over backwards compatibility.
* Always prefer less code to more flexibility unless specifically requested.
* Prefer terminal-friendly fixed-width text tables over Markdown pipe tables.
* Use short lists when table rows contain long explanations.
* Liberally validate code syntax with `python -m py_compile`.
* This is a research repository. Code reliability and robust unit-testing is not as important as simplicity.

## Common Commands

* Install/update dependencies with `uv sync`.
* Run Python entrypoints through `uv run`, for example:
  * `uv run modreczoo-simulate generate --output-dir data/awgn_sobol --n-signals 1000 --channel awgn --sampler sobol --seed 0`
  * `uv run modreczoo-train --dataset-dir data/awgn_sobol`
  * `uv run modreczoo-train --command sweep --dataset-dir data/awgn_snr0_30 --models resnet_1d dilated_cnn_1d --sweep-channel-formats real_imag mag`
* Open MLflow with `uv run mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --host 127.0.0.1 --port 5000`.
* Type checking, when useful: `uv run basedpyright src/`.
* Prefer `python -m py_compile path/to/file.py` for quick syntax validation after edits.

## Repository Map

* `src/modreczoo/simulation.py`: synthetic I/Q generation and metadata.
* `src/modreczoo/data.py`: dataset loading, normalization, and model input transforms.
* `src/modreczoo/training.py`: train/val/test orchestration, MLflow logging, evaluation, and artifacts.
* `src/modreczoo/evaluation.py`: metrics, calibration, bootstrap summaries, and per-slice evaluation helpers.
* `src/modreczoo/reporting.py`: HTML performance explorer and prediction/error tables.
* `src/modreczoo/plotting.py`: figures and interactive signal inspection utilities.
* `src/modreczoo/models/`: model implementations and registry.
* `src/modreczoo/cli/`: console script entrypoints.

## MLflow

* Runs use local SQLite tracking at `mlflow/mlflow.db`.
* Artifacts live under `mlflow/artifacts/`.
* Temporary staging files live under `mlflow/staging/<run_id>/`.
* Run names are generated from model and hyperparameters; sweep run names include only swept parameters.

## Adding Models

* Add the `torch.nn.Module` under `src/modreczoo/models/`.
* Register representation and required channel format, if any, in `src/modreczoo/models/registry.py`.
* Add a branch in `make_model()`.
* Add the model name to `MODEL_NAMES` in `src/modreczoo/training.py`.

## Data Notes

* Datasets are `signals.npy`, optional `extras.npz`, and `metadata.parquet`.
* `load_dataset()` is the standard entrypoint; avoid ad hoc metadata reads unless specifically needed.
* Channel format handling happens in `ModrecDataset.__getitem__`; model-specific forced formats are defined in the registry.
* Simulation sampling metadata terms:
  * `symbol_period`: samples per symbol used for the initial pulse-shaping filter.
  * `osr`: post-pulse-shaping resampling ratio, equal to `upsample_factor / downsample_factor`.
  * `upsample_factor`, `downsample_factor`: coarse rational approximation of the requested `osr_range`.
  * Effective samples per symbol: `symbol_period * osr`.
  * `symbol_rate`: normalized post-resampling symbol rate, equal to `1 / (symbol_period * osr)`.
  * `osr_range`: generator input range for desired OSR; metadata records only the realized rational approximation as `osr`.
