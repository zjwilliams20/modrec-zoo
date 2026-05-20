# Model Zoo

**Task:** Automatic Modulation Recognition — 8 classes (2PSK, 4PSK, 8PSK, π/4-DQPSK, 16QAM, 64QAM, 256QAM, MSK)
**Input:** 2048 complex samples · SNR 20–40 dB · OSR uniform 1–20 samples/symbol

---

## Results at a Glance

| model               | best format          | test acc | notes                                  |
|---------------------|----------------------|----------|----------------------------------------|
| `apf_net_1d`        | apf (required)       | ~84%     | 3-stream encoder + attention fusion    |
| `resnet_1d`         | differential_complex | 82.2%    | standard backbone, best representation |
| `multiscale_pyramid_1d` | differential_complex | 81.6% | slight val→test overfit             |
| `patch_transformer_1d`  | apf              | 66.7%    | underperforms; likely needs more data  |
| `multi_stream_1d`   | —                    | —        | per-channel streams + attention fusion |
| `multilag_net_1d`   | multilag (required)  | —        | multi-lag CSP features → ResNet1D      |
| `cyclic_caf_1d`     | cyclic_caf (required)| —        | CAF magnitude spectra → ResNet1D       |
| `scf_resnet`        | scf (required)       | —        | spectral correlation fn → ResNet2D     |

---

## Architecture Overview

```
─────────────────── BASELINES ──────────────────────────────────────────────────────

 CNN1D (time/freq)      CNN2D                   ResNet2D
 ──────────────────     ───────────────         ──────────────────
 (B, C, 2048)           (B, C, F, T)            (B, C, F, T)
       │                      │                       │
 Conv1d k=9 → 32        Conv2d k=3 → 24c        Stem: Conv2d(k=fk×tk, s=2)
 BN+ReLU+MaxPool2              │                 MaxPool2d(s=2)
       │                Conv2d k=3 → 48c               │
 Conv1d k=7 → 64        BN+ReLU+MaxPool2         Stage1: 2× ResBlock2D(c→c)
 BN+ReLU+MaxPool2              │                 Stage2: 2× ResBlock2D(c→2c, s=2)
       │                Conv2d k=3 → 96c         Stage3: 2× ResBlock2D(2c→4c, s=2)
 Conv1d k=5 → 128       BN+ReLU                  Stage4: 2× ResBlock2D(4c→8c, s=2)
 BN+ReLU+AvgPool→1      AdaptiveAvgPool2d(1,1)         │
       │                      │                 GlobalAvgPool2d
 Linear → C             Linear → C              Linear → C
                                                 (anisotropic kernel: freq_k × time_k)


─────────────────── 1D BACKBONES ───────────────────────────────────────────────────

 ResNet1D               DilatedCNN1D            ComplexCNN1D
 ──────────────────     ───────────────         ──────────────────
 (B, C, 2048)           (B, C, 2048)            (B, 2, 2048) → xr, xi
       │                      │                       │
 Stem: Conv1d k=7 s=2   Stem: Conv1d k=7 s=2   ComplexBlock k=9 → 32
 BN+ReLU+MaxPool2       BN+ReLU+MaxPool2        (yr = Wr·xr − Wi·xi)
       │                      │                 (yi = Wr·xi + Wi·xr)
 Stage1: 2× Res(32→32)  Cell d=1  ──┐ AvgPool        │
 Stage2: 2× Res(32→64, s=2)         ├→ cat    ComplexBlock k=7 → 64
 Stage3: 2× Res(64→128, s=2) d=2  ──┤ MaxPool        │
 Stage4: 2× Res(128→256, s=2)       │         ComplexBlock k=5 → 128
       │                 Cell d=4  ──┤               │
 GlobalAvgPool          Cell d=8  ──┤         AvgPool → cat(xr,xi)
 Linear → C             Cell d=16 ──┤         Linear(256) → C
                        Cell d=32 ──┘
                              │
                        cat(6×2×32=384)
                        Linear 384→256
                        ReLU+Dropout
                        Linear → C

 Each dilation cell doubles the receptive field; avg+max pooled after
 every cell so the classifier sees features at all temporal scales.


─────────────────── REPRESENTATION-DRIVEN ──────────────────────────────────────────

 APFNet                             MultiStreamNet
 ─────────────────────────────────  ──────────────────────────────────────────
 Preprocessing: 4-ch APF           Arbitrary C-channel input. One independent
   ch0: log|x|      (amplitude)    StreamEncoder per channel; outputs fused
   ch1: cos∠x, ch2: sin∠x (phase) via self-attention across C channel-tokens.
   ch3: Δ∠x/π      (inst. freq.)
   PSK  → phase stream                       (B, C, 2048)
   MSK  → freq stream                              │
   QAM  → amplitude + phase              per-channel split
                                    ┌──────────────┼──────────────┐
 (B, 4, 2048)                       ▼              ▼              ▼
       │                       Enc[0]          Enc[1]      … Enc[C-1]
 ┌─────┼────────┐               (64)            (64)           (64)
 ▼     ▼        ▼               └──────────────┬──────────────┘
 AmpEnc PhaseEnc FreqEnc                   stack (B, C, 64)
 ch[0] ch[1:3] ch[3]                            │
  (64)   (64)   (64)                 MultiheadAttn(heads=C)
    └─────┼─────┘                   + residual + LayerNorm
       stack                                    │
   (B, 3, 64)                            mean(dim=1)
        │                                Linear → C
 MultiheadAttn(heads=4)
 + residual + LayerNorm
        │
   mean(dim=1)
   Linear → C

 StreamEncoder (shared by APFNet and MultiStreamNet):
   Conv1d(k=7,s=2) → BN+ReLU
   ResBlock1D(32→64, s=2)
   ResBlock1D(64→128, s=2)
   ResBlock1D(128→64, s=2)
   GlobalMeanPool → 64-d feat


─────────────────── CSP-INSPIRED ───────────────────────────────────────────────────

 multilag_net_1d             cyclic_caf_1d               scf_resnet
 ──────────────────────      ─────────────────────        ──────────────────────
 Preprocessing: multilag     Preprocessing: cyclic_caf   Preprocessing: scf
   For τ ∈ {1, 4, 16}:        For τ ∈ {1, 4, 16}:        SCF: cross-spectral
   z[n]·z*[n−τ] → Re/Im       |FFT(z[n]·z*[n−τ])|         density at each
   3 lags × 2 ch = 6 ch        max-norm per lag            cyclic freq α
   RMS-norm per lag            3 ch total                  (B, 1, α, F)

 (B, 6, 2048)               (B, 3, 2048)
       │                          │
  ResNet1D                   ResNet1D                    ResNet2D
       │                          │                          │
  Linear → C                Linear → C                 Linear → C


─────────────────── ATTENTION / MULTI-SCALE ────────────────────────────────────────

 PatchTransformer1D             MultiScalePyramidNet
 ──────────────────────         ─────────────────────────────────────
 (B, 2, 2048)                   (B, C, 2048)
       │                              │
 split → (B, 64, 64)           ┌─────┼─────┬──────┬──────┐
         [64 patches, P=32]     s=1   s=2   s=4   s=8  s=16
       │                        │     │     │      │     │
 PatchEmbed: Linear(64→128)  AvgPool per scale (explicit downsampling)
       │                        │     │     │      │     │
 prepend [CLS] token          ScEnc ScEnc ScEnc ScEnc ScEnc
 + learnable pos_embed          │     │     │      │     │
       │                      (32)  (32)  (32)   (32)  (32)
 4× TransformerEncoderLayer     └─────┴──────┴─────┴─────┘
   (d=128, heads=4, ff=512,              │
    pre-norm, dropout=0.1)           cat(5×32=160)
       │                                 │
 [CLS] token → LayerNorm        Linear(160→128)
 Linear(128 → C)                ReLU+Dropout(0.2)
                                Linear → C

 At scale s, symbols are ~s samples apart. At the                  ScaleEncoder:
 correct OSR the k× branch sees ~1 sample/symbol —                 Conv1d(k=7,s=2)
 the most information-dense representation.                         BN+ReLU
                                                                    2×ResBlock1D
                                                                    AdaptiveAvgPool→1
```

---

## Channel Formats

All models except `spectrogram_*` consume 1D time-domain tensors; spectrograms use 2D.
The format is an input preprocessing step, not part of the model.

| format               | channels | description                                                              |
|----------------------|----------|--------------------------------------------------------------------------|
| `real_imag`          | 2        | Raw I and Q — information-complete, no structure imposed                 |
| `mag`                | 1        | `log(1+|x|)` normalized — amplitude only                                |
| `mag_phase`          | 2        | Log-magnitude + unwrapped phase/π                                        |
| `mag_inst_freq`      | 2        | Log-magnitude + Δphase/π (instantaneous frequency)                      |
| `differential_complex` | 2      | `Re(d), Im(d)` where `d[n]=x[n]·x*[n-1]`, RMS-normalized               |
| `apf`                | 4        | `[log\|x\|, cos∠x, sin∠x, Δ∠x/π]` — required by APFNet                |
| `multilag`           | 6        | Re/Im of `x[n]·x*[n-τ]` for τ∈{1,4,16}, RMS-norm — required by `multilag_net_1d` |
| `cyclic_caf`         | 3        | `\|FFT(x[n]·x*[n-τ])\|` for τ∈{1,4,16}, max-norm — required by `cyclic_caf_1d` |
| `scf`                | 1 (2D)   | Spectral correlation function image — required by `scf_resnet`           |

`differential_complex` removes the unknown carrier phase entirely; `apf` additionally
decouples the three information streams that are orthogonal to the modulation taxonomy.
`multilag` and `cyclic_caf` extend differential features to multiple lags, inspired by
cyclostationary signal processing (CSP).

---

## Per-Model Reference

### `time_cnn` / `frequency_cnn`

Three-layer plain 1D CNN. `frequency_cnn` is the same architecture applied to the FFT
magnitude spectrum instead of raw I/Q (handled in the data loader, not the model).
No residual connections; fast to train, reasonable ceiling.

- **File:** `models/baselines.py` · `CNN1D`
- **Formats:** any 1-D format
- **Params:** ~170 K

---

### `spectrogram_cnn`

Three-layer 2D CNN on STFT spectrograms. Isotropic 3×3 kernels. Simple enough to
overfit on small datasets; useful as a spectrogram baseline.

- **File:** `models/baselines.py` · `CNN2D`
- **Formats:** any 2-D spectrogram format
- **Key args:** `--spectrogram-base-channels` (default 24)

---

### `spectrogram_resnet`

Full 4-stage 2D ResNet on STFT spectrograms with **anisotropic** kernels
(`freq_kernel × time_kernel`). The asymmetry matters: frequency structure
(narrowband vs. wideband) has different scale than temporal structure (symbol patterns).

- **File:** `models/baselines.py` · `ResNet2D`
- **Formats:** any 2-D spectrogram format
- **Key args:** `--spectrogram-freq-kernel` (default 5), `--spectrogram-time-kernel` (default 3), `--spectrogram-base-channels`

---

### `feature_mlp`

Shallow MLP over 10 hand-computed features. Included as a lower bound; not competitive
with any learned representation.

- **File:** `models/baselines.py` · `FeatureMLP`
- **Formats:** `features`

---

### `resnet_1d`

Standard 4-stage 1D ResNet adapted from ResNet-18. Stem halves the sequence length
twice (stride 2 conv + MaxPool), then four residual stages double the channel count
while halving temporal resolution. Global average pool collapses the remaining 64
time steps to a single feature vector.

This is the go-to backbone for 1D signals in this zoo. Pairs exceptionally well with
`differential_complex`.

- **File:** `models/baselines.py` · `ResNet1D`
- **Formats:** any 1-D format
- **Params:** ~340 K

---

### `dilated_cnn_1d`

Shared-channel dilated conv stack (d = 1, 2, 4, 8, 16, 32). After each cell the
current activations are both avg- and max-pooled to a scalar and concatenated into a
growing feature vector. The final vector therefore captures temporal structure at six
exponentially-spaced receptive field sizes simultaneously.

Receptive field at d=32: ≈ 200 samples (plus the stem's stride 4 → ~800 original
samples). Unlike ResNet, no information is discarded to extend receptive field — every
cell reads from the same spatial resolution.

- **File:** `models/dilated.py` · `DilatedCNN1D`
- **Formats:** any 1-D format
- **Params:** ~230 K

---

### `complex_cnn_1d`

Applies **complex-valued convolutions**: weight matrices are complex, so convolution
computes `yr = Wr·xr − Wi·xi` and `yi = Wr·xi + Wi·xr`, preserving the geometric
structure of the complex plane. In principle this gives the network explicit awareness
of phase relationships. In practice the gain over `resnet_1d + real_imag` is modest.

Requires `real_imag` input (splits I and Q channels internally).

- **File:** `models/complex.py` · `ComplexCNN1D`
- **Formats:** `real_imag` only
- **Params:** ~320 K

---

### `patch_transformer_1d`

ViT-style transformer. The 2048-sample signal is divided into 64 non-overlapping
patches of 32 samples each. A linear projection embeds each patch into a 128-d token;
a learnable CLS token is prepended; learnable positional encodings are added.
Four pre-norm TransformerEncoder layers with 4-head attention and FF width 512.
Classification reads from the CLS token.

Currently the weakest architecture in the zoo. The fixed positional encoding is
misaligned with variable OSR (symbol boundaries fall at different positions per sample),
which is the structural advantage CNNs have via translation equivariance.

- **File:** `models/advanced.py` · `PatchTransformer1D`
- **Formats:** any 1-D format
- **Key args:** `--transformer-patch-size`, `--transformer-d-model`, `--transformer-n-heads`, `--transformer-n-layers`
- **Params:** ~660 K

---

### `multiscale_pyramid_1d`

Five parallel branches at AvgPool downsampling factors 1×, 2×, 4×, 8×, 16×. Each
branch runs an independent `_ScaleEncoder` (stem + 2 ResBlock1D + AdaptiveAvgPool)
and produces a 32-d feature vector. The five vectors are concatenated (160-d) and
classified by a small MLP.

Motivation: at the correct OSR `k`, the `k×` downsampled branch operates at
approximately 1 sample per symbol — the highest information density for that OSR.
The explicit multi-scale construction is more direct than DilatedCNN's receptive field
trick, but currently shows slightly more val→test overfit.

- **File:** `models/advanced.py` · `MultiScalePyramidNet`
- **Formats:** any 1-D format
- **Params:** ~440 K

---

### `apf_net_1d`

Three independent stream encoders read the **amplitude**, **phase**, and **frequency**
channels of the APF representation. Each encoder is a small ResNet (stem + 3 residual
blocks with progressive channel doubling, global mean pool → 64-d). The three 64-d
feature vectors are treated as tokens and fused by a single 4-head self-attention
layer with a residual connection and LayerNorm. The mean-pooled output is classified
by a linear layer.

The architecture is designed around the modulation taxonomy: PSK concentrates
information in the phase stream, MSK in the frequency stream, QAM in both amplitude
and phase. The attention fusion weights streams per-example without requiring the
network to discover the decomposition from raw I/Q.

cos/sin encoding of phase (`cos∠x`, `sin∠x`) is used rather than the raw phase angle
to avoid the ±π discontinuity at the branch cut.

- **File:** `models/advanced.py` · `APFNet`
- **Formats:** `apf` (forced — 4-channel input required)
- **Params:** ~380 K

---

### `multi_stream_1d`

Generalises APFNet's stream-encoder + attention-fusion pattern to arbitrary
channel counts. One independent `_StreamEncoder` (identical to APFNet's) processes
each input channel, producing a 64-d token. The tokens are fused by a single
multi-head self-attention layer where `num_heads = in_channels`; `stream_dim` is
rounded up to the nearest multiple of `in_channels` to satisfy PyTorch's attention
constraint. Classification reads from the mean-pooled output.

Unlike APFNet, no physics-motivated channel decomposition is assumed — the model
discovers inter-channel interactions purely from data. Works with any channel format.

- **File:** `models/advanced.py` · `MultiStreamNet`
- **Formats:** any 1-D format
- **Params:** varies with `in_channels`

---

### `multilag_net_1d`

`ResNet1D` applied to the **multi-lag** representation. For each of three lags
τ ∈ {1, 4, 16}, computes the conjugate lag product `z[n]·z*[n-τ]` and extracts
real and imaginary parts, giving 6 channels total (each lag RMS-normalised). This
extends the differential-complex idea to multiple time scales simultaneously,
capturing CSP-like structure without an explicit spectral analysis.

- **File:** `models/baselines.py` · `ResNet1D` (reused)
- **Formats:** `multilag` (forced — 6-channel input)

---

### `cyclic_caf_1d`

`ResNet1D` applied to **cyclic autocorrelation function** magnitude spectra. For
each of three lags τ ∈ {1, 4, 16}, computes `|FFT(z[n]·z*[n-τ])|` and
max-normalises it, yielding 3 channels of length 2048. The resulting spectra are
cyclostationary features sensitive to the modulation's characteristic symbol rate.

- **File:** `models/baselines.py` · `ResNet1D` (reused)
- **Formats:** `cyclic_caf` (forced — 3-channel input)

---

### `scf_resnet`

`ResNet2D` applied to the **spectral correlation function** (SCF). The SCF is a
2D image (cyclic frequency α × spectral frequency f) estimating the cross-spectral
density between the signal and a frequency-shifted copy of itself. The result is
a single-channel 2D representation that highlights the cyclostationary structure
unique to each modulation class.

- **File:** `models/baselines.py` · `ResNet2D` (reused)
- **Formats:** `scf` (forced — 1-channel 2D input)
