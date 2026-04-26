# Modulation Recognition (ModRec)

Author: Zach Williams

Date: 9/2/2025

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
