# Modulation Recognition (ModRec)

Author: Zach Williams

Date: 4/26/2026

Modulation recognition is a problem in RFML in which we'd like to classify communications signals based on their time-frequency characteristics using linear/non-linear processing. The ultimate goal in this particular setup is to be invariant to certain environmental characteristics.

Our specific objective here is not to optimally solve this modrec problem, but to compare various deep-learning based generalization techniques including meta-learning. Where by generalization here, we mean relative to some definition of domain. Channel is the best contender for generalization since this is agnostic to most of the other dimensions and is something we truly would like to be able to handle.

## Dataset

The dimensions of the data mostly follow [CSPB.ML.2018R2: Correcting an RNG Flaw in CSPB.ML.2018](https://cyclostationary.blog/2023/09/25/cspb-ml-2018r2-correcting-an-rng-flaw-in-cspb-ml-2018/) and include:
* Modulations/classes: {BPSK, QPSK, 8-PSK, pi/4-DQPSK, 16-QAM, 64-QAM, 256-QAM, MSK}
* In-band SNR: [0, 16] dB
* Carrier frequency offset (CFO): +-1/1000 (fractional frequency)
* Symbol timing offset (STO): +-1/2 (symbols)
* Oversample ratio (OSR): [1, 20] (samples per symbol)
* Square-root raise cosine (SRRC) excess bandwidth (EBW): [0.1, 1.0]
* Number samples: 32768
* Channel: AWGN, Rayleigh, Rician, non-linear, etc.

## Implementation

The code utilizes numpy/scipy to generate the I/Q with specific impairments, then trains a shallow real-valued 1D-CNN over the I/Q vector, as a simple baseline. Note that much of the codebase leverages AI tools to generate code. I take personal responsibility for any errors made by the tools and will do my best to be rigorous about ensuring functionality.

## Baselines

Initial supervised baselines include simple PyTorch models for time-domain I/Q,
frequency-domain I/Q, spectrogram-like representations, and hand-engineered
signal features. Planned literature baselines include:

* RiftNet: https://ieeexplore.ieee.org/document/9369455
* SVM-feature network: https://ieeexplore.ieee.org/document/8610499
* Capsule network approach: https://digitalcommons.odu.edu/cgi/viewcontent.cgi?article=1425&context=ece_fac_pubs

Potential research directions include:
* OOD pushes
* Multi-task learning: predict SNR, OSR, CFO, STO, does this help?
* Differentiable preprocessing: channel formats, continuously valued powers, fractional delays
* Architectures: generative stuff, KAN?
* Better algorithms for uncertainty awareness

## Differentiable Frontends and Auxiliary Tasks

The default training path still uses the dataset-level NumPy/SciPy preprocessing
and single modulation classifier. Opt-in differentiable frontends run inside the
PyTorch model with `--preprocessor`:

```bash
uv run modreczoo-train \
  --dataset-dir data/baseline_4096 \
  --models resnet_1d \
  --preprocessor radio_transform \
  --aux-targets snr_db osr ebw channel
```

Available frontends:
* `none`: existing behavior.
* `normalize`: differentiable per-example centering/RMS normalization.
* `learned_fir`: identity-initialized learnable FIR filterbank.
* `radio_transform`: identity-initialized learned time/frequency/phase correction
  for time-domain `real_imag` I/Q.

Auxiliary metadata heads are hard-parameter-sharing classifiers attached to the
backbone pre-logit feature. Numeric metadata targets are quantile-binned from
the training split (`--aux-bins`, default 8); categorical columns use observed
training values plus missing/unknown classes. Loss weighting defaults to a fixed
auxiliary weight (`--aux-loss-weight`, default 0.2), with optional homoscedastic
uncertainty weighting via `--aux-loss-mode uncertainty`.

The interface follows:
* [Radio Transformer Networks](https://arxiv.org/abs/1605.00716): learned
  synchronization/normalization for modulation recognition.
* [Spatial Transformer Networks](https://arxiv.org/abs/1506.02025):
  differentiable sampling for learned input transforms.
* [SincNet](https://arxiv.org/abs/1808.00158) and
  [LEAF](https://arxiv.org/abs/2101.08596): trainable frontends replacing fixed
  handcrafted filterbanks.
* [Caruana's multi-task learning formulation](https://www.cs.cornell.edu/~caruana/mlj97.pdf):
  shared representations trained from related labels.
* [Kendall, Gal, and Cipolla](https://arxiv.org/abs/1705.07115): optional
  homoscedastic uncertainty weighting for multi-task losses.

## Usage

Install the project in editable mode with uv:

```bash
uv sync
```

Generate a synthetic dataset with the simulator:

```bash
uv run modreczoo-simulate generate \
  --output-dir data/awgn_sobol \
  --n-signals 1000 \
  --n-samples 32768 \
  --channel awgn \
  --sampler sobol \
  --seed 0
```

Train the baseline models against that dataset:

```bash
uv run modreczoo-train --dataset-dir data/awgn_sobol
```

Training defaults to a local `mlflow/` directory containing the SQLite store,
artifacts, and staging files. Open the local MLflow UI with:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --host 127.0.0.1 --port 5000
```
