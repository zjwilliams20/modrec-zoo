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

## Usage

Create and activate a Python environment, then install the project dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Generate a synthetic dataset with the simulator:

```bash
python simulator.py generate \
  --output-dir data/awgn_sobol \
  --n-signals 1000 \
  --n-samples 32768 \
  --channel awgn \
  --sampler sobol \
  --seed 0
```

Train the baseline models against that dataset:

```bash
python train.py --dataset-dir data/awgn_sobol
```

To profile the training bottleneck, sample the first 50 training batches:

```bash
python train.py --dataset-dir data/awgn_sobol --num-workers 4 --profile-batches 50
```

This prints and logs MLflow metrics for data wait time, host-to-device transfer
time, GPU/CPU compute time, MLflow metric logging time, and artifact logging
time. Compare `--num-workers 0`, `4`, and `8` to see whether DataLoader work is
starving the GPU.

Training defaults to a local `mlflow/` directory containing the SQLite store,
artifacts, and staging files. Open the local MLflow UI with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --host 127.0.0.1 --port 5000
```

On RunPod, attach a network volume at `/workspace`, clone the repo under
`/workspace`, and expose port `5000` on the pod. Start the MLflow tracking
server first so it owns artifact serving through the RunPod proxy:

```bash
mlflow server \
  --backend-store-uri sqlite:////workspace/mlflow/mlflow.db \
  --artifacts-destination /workspace/mlflow/artifacts \
  --host 0.0.0.0 \
  --port 5000 \
  --allowed-hosts "*" \
  --cors-allowed-origins "*"
```

In another shell on the same pod, train through that local HTTP server:

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
python train.py --mlflow-profile runpod --dataset-dir /workspace/data/awgn_sobol
```
