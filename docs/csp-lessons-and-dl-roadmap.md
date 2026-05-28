# CSP Expert Feature Lessons & Deep-Learning Roadmap

*Last updated: 2026-05-28 — 6-model ensemble=81.92% (new best); JointV4-APF running*

---

## 1. What the CSP Iterations Taught Us

### 1.1 Architecture is Not the Bottleneck

An architecture sweep (tiny MLP → deep ResMLP-256 with 4 residual blocks, 552k
params; plus XGBoost) showed no difference at 70-feat → same accuracy ~68.5%.
Every accuracy gain came from NEW FEATURES, not bigger models.  Saturation of
feature-MLP capacity happened very early; after ~64 hidden units the curve is flat.

Implication for DL: a raw-IQ convolutional model at the same dataset size
probably also has "enough" capacity.  The bottleneck is whether the architecture
can discover the relevant statistics, not whether it has enough weights.

### 1.2 The Decisive Information: Signed Cyclic Autocorrelation

The key discriminant for the PSK sub-classes is `Re(E[d_T^4])` — the SIGNED
real part of the 4th-power phase differential autocorrelation, as a function of
lag T:

```
Re(E[d_T^4]) = Re( E[ exp(j·4·(φ[n] − φ[n−T])) ] )
```

At T = T_s (symbol period), the theoretical value per modulation type:

| Class       | d^4 at T_s | Re(E[d^4])  |
|-------------|------------|-------------|
| 4PSK        | +1 always  | ≈ +1        |
| π/4-DQPSK  | −1 always  | ≈ −1        |
| 8PSK        | ±1 equally | ≈  0        |
| QAM, MSK    | random     | ≈  0        |

Taking |E[d^4]| (the standard magnitude profile) throws away the crucial sign
that separates 4PSK from π/4-DQPSK.  Once we added `re4_min` (v18) and the
full 29-point signed profile (v19), 4PSK recall jumped from 69.9% → 76.4%.

**The full 29-point profile shape is more informative than any set of summary
statistics:**

```
              T=2   T=4   T=6   T=8   T=12  T=16  T=20  T=26  T=30
4PSK:         +.17  +.13  +.12  +.12  +.12  +.12  +.12  +.11  +.09  ← sustained plateau
8PSK:         +.16  +.08  +.04  +.02  +.00  +.00  +.00  +.00  +.00  ← rapid decay to 0
π/4-DQPSK:   +.15  +.03  −.01  −.05  −.08  −.04  −.01  +.01  +.02  ← unique negative dip
```

No single scalar summary (min, max, mean) captures this three-way distinction as
cleanly as the raw shape.

### 1.3 Two Hard Information-Theoretic Ceilings

**64QAM vs 256QAM — cannot be separated without symbol timing:**
- Theoretical cumulant difference (C42/M21^2): only 0.0143 = 2.3% of the 64QAM value
- SRRC filtering dilutes this by ~10× → effective separation ~0.0014, noise-floor level
- All statistics tried: C40, C42, M63, M84, amplitude entropy, cross-stats → max 0.10σ
- This is a measurement bottleneck, not an ML bottleneck.  Deep learning cannot
  recover information that is not present in the signal statistics.

**16QAM vs 64QAM — separable but limited by low SNR:**
- M84 (8th amplitude moment): 9.5% difference → ~5σ at moderate SNR — well separated
- Confusions are almost entirely at SNR < 5 dB (16.8% of the dataset)
- Robust trimmed moments would help; deeper models would not

### 1.4 The Low-SNR Problem

The dataset has 16.8% of signals below 5 dB SNR.  At that SNR the noise
smears the phase differences that drive all higher-order statistics.  Even
`|E[d^4]|` at the correct T_s drops from ~1.0 (high SNR) toward ~0 (≈ noise
floor 1/√N).  This causes cascading failures across the PSK features.

No amount of feature engineering or architecture search can recover information
that the noise has destroyed.  The path forward here is data augmentation or
auxiliary SNR estimation, not more feature kinds.

### 1.5 Blindly-Computed Features We Found Are Near-Useless

- `|E[x^6]|` and `|E[x^8]|` (Group 9 conjugate moments): near-zero for all
  non-BPSK modulations due to M-fold constellation symmetry.  After SRRC filtering
  small values remain (0.05–0.32) but carry low discrimination.  These should be
  removed and replaced with amplitude-based robust statistics.

- Blind symbol timing estimators (T_ph4, T_amp, T_bw): 22–183% error — too
  inaccurate to enable constellation-sampling features.  The scan-over-all-lags
  strategy (profile approach) is the correct alternative.

---

## 2. The Feature → Model Progression

Accuracy history on the baseline_4096 dataset (200k signals, 8 classes):

```
  Model                   Features  Test acc    Key addition
  csp_canonical_mlp          13      ~0.63      Swami & Sadler 2000 theory
  csp_expert_mlp (v15)       70      0.6844     multi-scale profile Groups 8-9
  csp_expert_mlp (v16)       74      0.7096     Group 8 FFT autocorrelation
  csp_expert_mlp (v18)       77      0.7207*    signed re4_min, re4_asym, pc4_min
  csp_expert_mlp (v19)      107      0.7262*    re4_at_peak + full 29-pt profile
  HybridCSPNet (v19)         107      0.7288*    1D-conv branch on re4 profile

  JointCSPCNN v1 ens.      107+raw   0.8092*    CSP MLP + mini-ResNet1D (base=16, 480k)
  JointCSPCNN v2 ens.      107+raw   0.8125*    base_ch=32 signal branch (900k)
  JointCSPCNN v3 ens.      107+raw   0.8106*    ZOO ResNet1D (base=32, 2b/stage, 1.29M)
  4-model cross-ens.       107+raw   0.8165     v1×2 + v2×2 arch+seed diversity
  JointCSPCNN v2b ens.     107+raw   0.8130*    base_ch=32, LR=2e-3 (complementary to v2)
  LR-diversity 4-model     107+raw   0.8170     v2×2 + v2b×2 (same arch, 2 LRs)
  6-model ensemble         107+raw   0.8192     v1×2+v2×2+v2b×2  ← CURRENT BEST
  JointV4-APF (running)    107+raw   TBD        APF signal branch (4ch: log-mag,ph,IF)

  DL baseline (APF/ResNet1D)          0.76       (target: 76% — exceeded by +5.92pp)
  Theoretical ceiling                ~0.85       (64/256-QAM + low-SNR floor)

  *ensemble of 2 random seeds
```

**Key results (2026-05-28):**
- JointCSPCNN v1 ensemble: **80.92%** — +4.92pp above DL baseline, +8.04pp above CSP-only
- JointCSPCNN v2 ensemble: **81.25%** — +0.33pp improvement, 256QAM partially fixed
- 6-model ensemble: **81.92%** — new best, +5.92pp above DL baseline

The expert CSP features contribute ~4% on top of the standalone DL model; the signal CNN
contributes ~8% on top of the standalone CSP model. Together they reach 81%+ without any
attention mechanisms or transformers — pure two-branch fusion.

V2 per-class delta vs v1 (ensemble):
  - 256QAM: +5.9% (larger signal branch learns amplitude envelope better)
  - PSK triangle: 827 → 979 errors (+152, worse — LR=1e-3 limits phase discrimination)
  - Net: +33 basis points via better QAM discrimination despite PSK regression

V3 (0.8106): class-weighted loss hurt cross-class calibration; 256QAM didn't improve vs v2.
MC Dropout: zero benefit at p=0.20–0.25 (too light for implicit ensemble gain).

4-model cross-ensemble (0.8165): architecture diversity (base=16 vs base=32) is the strongest
lever found so far. Even weaker individual models combine to beat any 2-model same-arch ensemble.
256QAM recall hit 61.0% — the ensemble finds a better 64/256-QAM decision boundary.

V2b (0.8130): confirmed LR=1e-3 bottleneck. LR=2e-3 gives complementary per-class profile:
256QAM: 60.6% (vs v2's 55.5%), 16QAM: 75.0% (vs v2's 81.6%). Creates LR-diversity ensemble.

LR-diversity 4-model (0.8170): combining v2+v2b LR variants slightly exceeds arch-diversity.

6-model ensemble (0.8192): all three diversity axes (arch, LR, seed) combined. Per-class:
  16QAM=84.6%, 256QAM=60.0%, 4PSK=94.2%, 64QAM=41.0%, 8PSK=86.9%, π/4=88.7%
  Diminishing returns: only +0.22pp over best 4-model. The 256QAM/64QAM ceiling (~61%/41%)
  persists across all ensemble compositions — confirming the SRRC information-theoretic limit.

JointV4-APF (running): APF signal branch [log-mag, cos(ph), sin(ph), instantaneous_freq].
  If per-class profile differs from complex_powers, can create new diversity source for 8-model ens.

---

## 3. Deep Learning Roadmap

### 3.1 Representations Already in the Codebase

The channel-format representations in `data.py` span a gradient from raw IQ to
CSP-motivated:

```
  raw I/Q        → real_imag, mag_phase
  1st-order diff → differential_complex      (lag-1 phase diff, Re+Im)
  amplitude+phase→ apf                       (log_mag, cos_ph, sin_ph, IF)
  power series   → complex_powers            (x, x², x⁴, Re+Im each)
  multi-lag diff → multilag                  (x·x*[n-τ] Re+Im, several τ)
  cyclic spectrum→ cyclic_caf                (|FFT(x·x*[n-τ])| for several τ)
  full SCF       → scf                       (spectral correlation function 2D)
```

The `cyclic_caf` and `complex_powers` formats are the closest to what the CSP
expert features compute.  The step from there to explicit 4th-power signed
profiles is small.

### 3.2 Suggestions Per Existing Model

#### `resnet_1d` (and `dilated_cnn_1d`)
**Gap:** kernel size 3 is far too short for cyclostationary patterns spanning T_s
≈ 6–20 samples.  The network has to compose many small convolutions to "see"
one symbol period.

**Suggestions:**
- Use `complex_powers` channel format (6ch I/Q + squared + 4th power) instead of
  `real_imag`.  This pre-computes the power operations the network needs to discover.
- Add a second input head on the unit-phasor signal `(x/|x|)^4` so the 4th-power
  cyclostationary structure is visible in channel 1.
- For `dilated_cnn_1d`: the exponential dilation schedule already spans multi-symbol
  ranges — use this model with `cyclic_caf` channel format for best alignment.

#### `patch_transformer_1d`
**Gap:** Absolute positional encodings treat all patch positions as independent;
cyclostationary signals have PERIODIC structure at lag T_s.

**Suggestions:**
- Replace absolute positional encodings with **relative positional encodings**
  (RoPE or ALiBi).  This lets attention naturally represent "lag" rather than
  "absolute position".  A pair of patches separated by T_s patches will always
  look like the same lag regardless of where in the signal they are.
- Experiment with `patch_size=T_s_prior` (e.g., 8–16 samples) to align patches
  with symbol boundaries.
- Multi-resolution patchification: run 3 parallel patch embeddings at sizes 8,
  16, 32 and concatenate the CLS tokens before the final classifier.

#### `multiscale_pyramid_1d`
**Gap:** The pyramid addresses OSR variability (good!) but each scale encoder
still uses short convolutional kernels that learn local features.

**Suggestions:**
- This is the right high-level design for cyclostationary features.  Extend it:
  at each scale, use `complex_powers` preprocessing (4th-power) before the
  per-scale encoder.  Now each branch is explicitly looking at the 4th-power
  cyclostationary structure at that scale.
- Replace concat→MLP with cross-scale attention: let scales vote on which is
  closest to the true T_s.  Soft weighting of scale branches is more robust than
  hard concatenation.

#### `apf_net_1d`
**Gap:** The phase stream (cos/sin instantaneous phase) is only lag-0 phase —
it does not capture inter-symbol phase differences that are the core of PSK
discrimination.

**Suggestions:**
- Replace or augment the phase stream with **3 phase-power streams**:
  `(x/|x|)^2`, `(x/|x|)^4`, `(x/|x|)^8` as separate 1-channel signals.  Each
  reveals a different modulation order's symmetry (2PSK, 4PSK/π4-DQPSK, 8PSK).
- Extend the IF stream to multi-lag: pass not just lag-1 instantaneous frequency
  but the 4th-power differential at lags 2, 4, 8, 16 as a multi-channel 1D signal.
  This is the key information the CSP expert features exploit.

#### `complex_cnn_1d`
**Gap:** The input is standard complex I/Q; complex convolutions learn
cross-I/Q correlations but not higher-order phase statistics.

**Suggestion:**
- Pre-process: stack the complex unit phasor `u = x/|x|` alongside raw x.
  A complex convolution on `u^4` computes exactly the cyclic autocorrelation
  `E[u(n)^4 · conj(u(n−T))^4]` when the kernel has support spanning T samples —
  this IS the signed re4 profile that drives our best features.

#### `multilag_net_1d`
**Gap:** Uses `multilag` format (several lag products as channels) but with a
ResNet1D backbone that may not best exploit the lag dimension.

**Suggestion:**
- The lag products `x[n]·x*[n-τ]` already contain the per-lag phase information.
  Raising each lag product to the 4th power (i.e., `(x[n]·x*[n-τ])^4 / |...|^4`)
  before passing to the ResNet would give the signed cyclic autocorrelation as input.
- Better backbone choice: since the lag dimension is the meaningful axis, use a
  small 1D MLP or Transformer on the per-lag aggregated features rather than a
  convolutional model that processes lags as spatial positions.

#### `cyclic_caf_1d`
**Gap:** Computes `|FFT(x·x*[n-τ])|` — the MAGNITUDE cyclic spectrum.  Like the
magnitude-only `|E[d^4]|` profile in our CSP features, this loses the sign.

**Suggestions:**
- Use SIGNED cyclic spectrum: `Re(FFT(u^4[n]))` at each cycle frequency, not the
  magnitude.  This is the frequency-domain dual of the re4 profile and preserves
  the 4PSK/π4-DQPSK sign discrimination.
- Use a 2D ResNet (cycle-freq × lag as a 2D image) instead of ResNet1D — the
  SCF is a 2D function and the ResNet1D flattens its structure.

#### `scf_resnet`
**Current state:** Full 2D spectral correlation function with a 2D ResNet backbone.
This is theoretically the richest representation.

**Suggestions:**
- The relevant structure in the SCF is the LOCATION of cyclic spectral lines at
  α = k/T_s, not local texture.  A Transformer operating on (α, f) tokens would
  be more appropriate than a convolutional ResNet that treats both axes as spatial.
- Consider separating the conjugate and non-conjugate SCF (two 2D images) to
  make the BPSK conjugate cyclic line explicitly visible.

### 3.3 New Model Ideas

#### A. `CyclicProfileNet` — differentiable CSP in the forward pass

Compute the FFT-based cyclic autocorrelation profiles inside the model as a
differentiable preprocessing layer:

```python
class CyclicProfileLayer(nn.Module):
    """Differentiable computation of |E[d_T^k]| and Re(E[d_T^k]) profiles."""
    def forward(self, x):           # x: (B, 2, N) complex as real+imag
        z = torch.view_as_complex(x.permute(0, 2, 1).contiguous())
        u = z / (z.abs() + 1e-10)   # unit phasor
        profiles = []
        for k in [2, 4, 8]:
            uk = u ** k
            # FFT autocorrelation: O(N log N) for all lags simultaneously
            fft_uk = torch.fft.fft(uk, n=2*uk.shape[-1])
            acf = torch.fft.ifft(fft_uk.abs() ** 2)   # circular autocorr
            mag_profile  = acf[:, 2:31].abs() / uk.shape[-1]    # (B, 29)
            if k == 4:
                re4_profile = acf[:, 2:31].real / uk.shape[-1]  # (B, 29) signed
            profiles += [mag_profile]
        # Stack: (B, 4, 29) — 3 magnitude profiles + 1 signed re4 profile
        return torch.stack([*profiles, re4_profile], dim=1)
```

Then pass the (B, 4, 29) tensor (or similarly shaped) to a lightweight 1D
Transformer or MLP.  This is the "no-hand-coded-stats" version of the CSP expert
features — the network sees the raw profiles and learns what statistics to extract.

#### B. SNR-Adaptive Weighting

Since ~17% of signals are below 5 dB and almost all errors occur there:

```python
class SNRGatedMLP(nn.Module):
    def forward(self, feats):
        # Estimate SNR from amplitude variation coefficient (feature index 5)
        snr_proxy = feats[:, 5:6]  # amp_std/amp_mean: low for high-SNR PSK
        gate = torch.sigmoid(self.snr_gate(snr_proxy))  # (B, n_features)
        return self.classifier(feats * gate)
```

Or use multi-task learning: add an SNR regression head and let the shared
encoder learn a representation that is SNR-aware.

#### C. Phase-Power Multi-Stream (`PhaseStreamNet`)

Build on `APFNet` but with streams aligned to PSK order:

```
Stream 1: amplitude envelope    → constant for PSK, variable for QAM
Stream 2: (x/|x|)^2 profile    → detects 2-fold symmetry (BPSK)
Stream 3: (x/|x|)^4 profile    → detects 4-fold (4PSK, π/4-DQPSK) + sign
Stream 4: (x/|x|)^8 profile    → detects 8-fold (8PSK)
```

Process each as a 1D signal with a small ResNet, fuse with attention.  This is a
deep learning model that explicitly mirrors the cumulant-order hierarchy from the
AMC literature (Swami & Sadler 2000).

### 3.4 Preprocessing Improvements

1. **Signed cyclic channel format** (new `data.py` transform): Extend `multilag`
   to include Re(x^4[n]·conj(x^4[n-τ])) / |...| alongside the magnitude.  This
   exposes the sign information that drives the biggest accuracy gains.

2. **Unit-phasor powers** (new channel format): 6-channel signal
   `[Re(u^2), Im(u^2), Re(u^4), Im(u^4), Re(u^8), Im(u^8)]` where `u = x/|x|`.
   The 4th-power channel is exactly the signal whose autocorrelation gives the
   signed re4 profile.  Any 1D CNN on this representation implicitly computes
   that profile.

3. **Trimmed amplitude moments**: Replace raw M84 with 5%-trimmed mean of |x|^8
   to reduce noise sensitivity at low SNR.  Practically: `np.sort(amp**8)[:int(0.95*N)].mean()`.

---

## 4. Actionable Priorities

**Immediate** (JointCSPCNN variants, being run now):
- v2: base_ch=32, 60 epochs — more signal branch capacity
- v3: ZOO ResNet1D backbone (base=32, 2 blocks/stage) — max signal capacity
- Cross-ensemble: v1-s0 + v1-s42 + v2-s0 + v2-s42 (4-model ensemble)

**Short-term** (CSP feature improvements, v20):
- Remove useless conjugate moments |E[x^6]|, |E[x^8]| from Group 9
- Add trimmed M84 (more robust at low SNR for 16/64-QAM separation)
- Add `re4_late = mean(re4_real[18:29])` (explicit sustained-plateau indicator for 4PSK)

**Medium-term** (new DL channel format):
- Implement `signed_cyclic_profile` channel format: (B, 4, 29) tensor from
  CyclicProfileLayer above, used with a Transformer backbone
- Try this with `patch_transformer_1d` backbone treating lag-dimension as "tokens"

**Longer-term** (architecture):
- SNR-adaptive training: curriculum from easy (high-SNR) to hard (low-SNR)
- Relative positional encodings in `patch_transformer_1d`
- PhaseStreamNet built on APFNet's stream encoder pattern

---

## 5. What Deep Learning Can and Cannot Improve

Two benchmarks: `CSP-only` (v19 HybridCSPNet ensemble) and `JointCSPCNN v1` (seed 0):

| Error source            | CSP-only acc | JointCSPCNN v1 | Theoretical | DL can help? |
|-------------------------|-------------|-----------------|-------------|--------------|
| 64QAM ↔ 256QAM          | 56% / 38%   | TBD (pending)   | ~50% each*  | Partially    |
| 16QAM ↔ 64QAM (low SNR) | ~71%        | TBD             | ~95%+       | Yes          |
| 4PSK ↔ π/4-DQPSK        | 74% / 82%   | TBD             | ~90%+       | Yes          |
| 8PSK ↔ 4PSK/π4          | ~73%        | TBD             | ~90%+       | Yes          |
| 2PSK, MSK               | 93-96%      | TBD             | 98%+        | Marginally   |

*(JointCSPCNN per-class breakdown pending seed-42 completion; expected large PSK improvement)

*Without symbol timing, 64/256-QAM are fundamentally indistinguishable given
SRRC filtering.  However, at 4096 samples the amplitude histogram provides additional
separation — the JointCSPCNN signal branch appears to exploit this more than predicted.

The deep learning advantage over hand-crafted CSP features is:
1. **Learning the summary statistics** — instead of min/max of the re4 profile,
   the network finds the optimal functional of the profile for each pair of classes
2. **Joint optimization** — all features are optimized together for the 8-class
   problem, not independently designed per pair
3. **Low-SNR robustness** — with enough training data at low SNR, the network can
   learn to interpolate between "high-SNR use phase features" and "low-SNR use
   amplitude features" more smoothly than explicit thresholds

The expert features set the floor; the DL target ceiling is limited by the
information-theoretic bounds identified above.

---

## 6. What JointCSPCNN Taught Us (2026-05-28)

After the CSP feature engineering hit a ceiling at ~73%, combining expert features with
a raw-signal mini-ResNet yielded 80.33% — well above the 76% DL baseline. Key lessons:

### 6.1 Expert Features + Raw Signal > Either Alone

```
  CSP expert features only:       72.88%   (fundamental SRRC dilution limit)
  DL signal CNN only (ZOO):       76.00%   (baseline on baseline_4096)
  Joint (CSP + signal mini-CNN):  80.33%   (+7.4% over CSP, +4.3% over DL-only)
```

The two branches capture genuinely complementary information:
- **CSP branch**: Re(E[d^4]) profile discriminates PSK types even at high SNR via closed-form theory
- **Signal branch**: Raw amplitude at 4096 samples × temporal pattern captures QAM order and low-SNR patterns

### 6.2 The Signal CNN Does What SRRC Prevents Expert Features From Doing

Expert features computed on SRRC-filtered signals lose inter-symbol amplitude information.
The CNN operating on raw IQ can still extract the amplitude histogram shape that distinguishes
64-QAM from 256-QAM — not perfectly, but enough to reduce errors substantially.

This is a key lesson: **hand-crafted features applied to SRRC signals face a mathematical ceiling
that the same raw data does not have.** The signal CNN can learn SRRC-aware representations.

### 6.3 OneCycleLR Peak Tells You a Lot

Val accuracy peaked at epoch 20/80 (80.33%) and plateaued at ~79% for the remaining 60 epochs.
This means:
- The correct learning rate hits a good basin quickly in phase 1
- phase 2 (LR decay) can't improve beyond the epoch-20 basin
- **Action**: use fewer epochs or cosine-annealing with restarts (SGDR) to escape this basin

### 6.4 What Expert Features Still Add Over DL-Only

Even though the signal CNN alone gets 76%, adding 107 CSP features pushes to 80.33%:
- The re4 profile directly encodes the PSK-class decision in a noise-robust way
- The amplitude moments (Group 2) help at high SNR where constellation shape is clear
- The combined model has access to both the theory-derived statistics AND the raw signal

This supports using expert features as **additional input channels** in future architectures
rather than replacing DL with hand-crafted features or vice versa.
