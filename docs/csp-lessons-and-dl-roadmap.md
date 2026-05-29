# CSP Expert Feature Lessons & Deep-Learning Roadmap

*Last updated: 2026-05-29 — 6-model ensemble=81.92% (baseline_4096); JointCSPCNN on baseline_32768 4-model ensemble=81.22%; Phase transition sweep on 200k DONE: CSP-only K=32768 = 81.43%; JointV2-s0 on 200k DONE: test=85.57% (+4.14pp over CSP-only); 5 configs remaining*

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

---

## 7. Phase Transition Study: Feature Quality vs. Observation Window K (COMPLETE)

### 7.1 Motivation

The baseline_4096 study operated on signals with ~321 symbols (mean T_s ≈ 12.75 samples).
CSP features are time-averages — their variance decreases with more symbols, so their
discriminative power improves with K. The open question: **where does the transition occur
from "too few symbols to reliably distinguish" to "CSP-converged"?**

### 7.2 Literature Basis

- **Gardner (1991)** "Exploitation of Spectral Redundancy in Cyclostationary Signals,"
  *IEEE Signal Processing Magazine* 8(2):14–36, Sec. IV.
  Establishes coherent averaging gain: Var(cyclic stat) ∝ 1/K → SNR ∝ √K.

- **Swami & Sadler (2000)** "Hierarchical Digital Modulation Classification Using Cumulants,"
  *IEEE Transactions on Communications* 48(3):416–429, Eq. (9)–(11).
  Asymptotic variance of normalized cumulant estimates; 3σ threshold: K >> (σ₀/Δfeature)².

- **Dobre, Abdi, Bar-Ness, Su (2007)** "Survey of Automatic Modulation Classification
  Techniques: Classical Approaches and New Trends," *IET Communications* 1(2):137–156,
  Secs. III–IV. Effect of pulse shaping (SRRC) on cumulant feature separability.

### 7.3 Empirical Setup

Dataset: `baseline_32768` — 40k signals × 32,768 samples, 8 balanced classes, SNR 0–30 dB.
Sweep: K ∈ {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768} = ~5 to ~2,570 symbols.
Classifier: ResMLP-256-4b (same as baseline_4096 CSP-only experiments).
Split: fixed 70/15/15 train/val/test across all K for direct comparability.

### 7.4 Key Empirical Estimate (NOT a literature claim)

Our measured Δ(C₄₂) ≈ 0.0014 for 64 vs 256-QAM after SRRC filtering.
Using the Swami & Sadler variance formula as a back-of-envelope, this suggests
thousands of symbols may be required for reliable CSP-based QAM-order discrimination.
**This is an empirical estimate from our data + literature formula — validate from the sweep.**

### 7.5 Results — Phase Transition Sweep (COMPLETE 2026-05-28)

ResMLP-256-4b, single seed, 30 epochs, 40k signals (5k/class), SNR 0-30 dB.
Gain/doubling = accuracy increase per 2× increase in K (log-linear view).

```
K        ~Symbols  Test Acc  Gain/2×K   16QAM  256QAM  4PSK   8PSK   π/4-DQ  MSK
────────────────────────────────────────────────────────────────────────────────────
    64          5    0.314      —        —      0.215  0.152  0.241  0.224   —
   128         10    0.387    +7.3pp     —      0.279  0.287  0.320  0.329   —
   256         20    0.474    +8.8pp     —      0.288  0.479  0.476  0.476   —
   512         40    0.558    +8.4pp     —      0.239  0.611  0.568  0.617   —
  1024         80    0.618    +6.0pp     —      0.339  0.651  0.715  0.652   —
  2048        161    0.664    +4.6pp     —      0.381  0.704  0.771  0.660   —
  4096        321    0.704    +4.0pp    0.669   0.479  0.739  0.732  0.725  0.955
  8192        642    0.737    +3.3pp    —       0.480  0.765  0.767  0.756   —
 16384       1285    0.764    +2.6pp    —       0.571  0.803  0.741  0.772   —
 32768       2570    0.788    +2.5pp    0.835   0.595  0.815  0.783  0.780  0.996
────────────────────────────────────────────────────────────────────────────────────
  baseline_4096 CSP-only ensemble (200k signals): 0.729  (for comparison)
  baseline_4096 JointCSPCNN ensemble (best):      0.819
```

**Key findings:**

1. **Not a sharp transition — log-linear growth across all K.**
   Accuracy grows ~√K (each doubling adds 2.5–8pp), with no clear inflection.
   The "fast phase" (K<512) yields 8pp/doubling; "slow phase" (K>4096) yields 3pp/doubling.

2. **16QAM benefits most from longer windows (+16.5pp, K=4096→32768).**
   The amplitude histogram converges slowly because SRRC blends symbol amplitudes
   across neighboring time steps; more symbols reveal the 4-ring amplitude structure.

3. **MSK converges earliest — 99.6% at K=32768.**
   Constant instantaneous frequency is an extremely stable feature; reliable
   within ~100 symbols. Already ~95.5% at K=4096.

4. **QAM (64+256) still improving at K=32768 (+13.9% / +11.6%) — not yet converged.**
   Consistent with the theoretical prediction that Δ(C₄₂)≈0.0014 requires enormous K.
   The empirical slope of +2.5pp/doubling at K=32768 implies we'd need K≈500k+ samples
   to fully converge — beyond what the dataset provides.
   *(Note: this extrapolation uses the empirical slope, not a literature formula.)*

5. **K=4096 CSP-only result (70.4%) is 2/3 of the way to K=32768 (78.8%).**
   Adding the JointCSPCNN signal branch at K=4096 jumps to 81.9%, showing that
   deep learning recovers much of the missing integration gain implicitly.

6. **PSK triangle (4PSK, 8PSK, π/4-DQPSK) converges by ~K=2048 (~160 symbols).**
   The re4 profile shape is already clear at that point; further K adds little for PSK.

### 7.6 Goal A: JointCSPCNN on baseline_32768 (COMPLETE 2026-05-28)

Architecture adapted for 32768-sample inputs:
- Stem: Conv1d(6ch, base_ch, 15, stride=8) + MaxPool1d(stride=4) = 32× reduction
  (vs. 4× for baseline_4096 stem) → same 1024 time steps entering the stages
- Stages: identical to v2/v2b (4 stages, 1+2+2+2 strides = 8×)
- Total: 256× downsampling → 128 final steps (same as baseline_4096)
- Script: `/tmp/train_joint_32768.py`

**Results (4 configs: JointV2×2 seeds + JointV2b×2 seeds):**

```
Config          Test     Best val   Notes
----------------|--------|----------|----------------------------------
JointV2-s0      0.7872   0.7962    LR=1e-3, 60ep
JointV2-s42     0.7880   0.7867    LR=1e-3, 60ep
JointV2b-s0     0.7950   0.7955    LR=2e-3, 80ep  ← V2b clearly better
JointV2b-s42    0.7955   0.7953    LR=2e-3, 80ep
v2 ensemble     0.7932             seed diversity only
v2b ensemble    0.8082             big jump from LR diversity
4-model ens.    0.8122             BEST on baseline_32768

Reference: baseline_4096 6-model ensemble = 0.8192
```

**Per-class breakdown (32768 4-model vs 4096 6-model):**

```
Class         32768   4096    Δ       Interpretation
-------------|--------|--------|-------|---------------------------------------
16QAM         87.6%   84.6%  +3.0pp  More amplitude samples → 4-ring structure
256QAM        53.9%   60.0%  -6.1pp  Needs more training examples
2PSK         100.0%  100.0%  +0.0pp
4PSK          90.7%   94.2%  -3.5pp  Small-dataset regression (temporal patterns)
64QAM         54.0%   41.0% +13.0pp  Biggest winner: SRRC amplitude convergence
8PSK          88.4%   86.9%  +1.5pp
MSK          100.0%   99.8%  +0.2pp
π/4-DQPSK    75.2%   88.7% -13.5pp  Biggest loser: needs many training examples
```

**Key findings from Goal A:**

1. **Individual model ceiling: ~79.6%** — below baseline_4096's ~80.4%.
   Despite 8× longer signals, only 28k training examples (vs 140k) limits the joint model.

2. **Signal branch adds ~0pp over CSP-only at K=32768** (CSP-only: 78.8%, JointV2-s0: 78.7%).
   On baseline_4096, the signal branch added +8pp. The dataset-size asymmetry explains this:
   with 5× fewer examples, the model can't learn complementary patterns beyond what the
   CSP features already encode.

3. **V2b (LR=2e-3) outperforms V2 (LR=1e-3) by +0.7pp** on the small dataset.
   On baseline_4096 the two were tied (~80.2–80.4%). Higher LR acts as implicit
   regularization when the gradient estimate is noisier (small N).

4. **4-model ensemble (81.22%) vs 6-model baseline_4096 (81.92%) = −0.7pp gap.**
   8× longer signals partially compensate for 5× fewer training signals, but at the
   ensemble level the comparison is nearly fair given the architectural disadvantage.

5. **Class-specific signal-length effects** are confounded with dataset-size effects:
   64QAM's +13pp likely reflects SRRC convergence (signal-length effect);
   π/4-DQPSK's −13.5pp reflects the small training set (dataset-size effect).
   A controlled experiment (same N signals, varying K) would isolate the two.

---

## 8. Controlled Experiment: baseline_32768_200k (phase transition DONE; arch search running)

### 8.1 Motivation

The 40k sweep (Section 7) confounded two variables:
- **Signal-length effect**: longer K → better CSP integration → higher accuracy
- **Dataset-size effect**: baseline_32768_40k has 5× fewer training examples than baseline_4096

`baseline_32768_200k` (200k signals × 32768 samples) holds dataset size fixed at 200k
while using 8× longer signals than baseline_4096. This isolates signal-length effects.

### 8.2 Experiments Running

**A) Phase Transition Sweep** (`/tmp/phase_transition_200k.py`):
- K ∈ {64, 128, ..., 32768} on 200k signals (140k/30k/30k split)
- Same ResMLP-256-4b classifier as 40k sweep
- Δ vs 40k at each K isolates dataset-size contribution
- Feature cache: `/tmp/phase_trans_200k_feats/feats_K{K}.npy`
- Timing: ~183s/K (IPC-overhead dominated; all 10 K values take ~30 min)

**B) Architecture Search** (`/tmp/train_joint_200k.py`):
- 6 configs: JointCSPCNN (V2×2 + V2b×2 seeds) + JointCSPAttn (V2b×2 seeds)
- `JointCSPAttn` replaces GAP with learned temporal attention pooling (+33k params)
  Motivated by: pi/4-DQPSK phase transitions are temporally structured; 32768-sample
  signals have sufficient temporal context for attention to discriminate positions.
- Auto-launched when K=32768 CSP features ready (cache reused, no duplicate computation)
- CSP cache: `/tmp/csp_200k_features.npy` (copied from K=32768 phase transition cache)

### 8.3 Hotpath Integrations (2026-05-29)

New models and utilities added to `src/modreczoo/models/` and `src/modreczoo/`:

**`joint_csp_attn`** (`models/joint.py`, `models/registry.py`):
- Same as `joint_csp_cnn` but signal branch uses `_AttentionPool1D` instead of GAP
- Additive attention: score = Tanh(Linear(C→C//2)) → Linear(C//2→1) → softmax → weighted sum
- Works for any input length (handles K=4096 and K=32768 equally)
- 932,872 params (vs 899,848 for GAP variant; +33k for attention scorer)
- Registered for use via `modreczoo-train --models joint_csp_attn`

**`joint_csp_dual`** (`models/joint.py`, `models/registry.py`):
- Signal branch with 12-channel input: `complex_powers` (6ch) + `unit_phasor_powers` (6ch)
- `unit_phasor_powers` is computed from the first 2 complex_powers channels IN THE FORWARD PASS
  (u = z/(|z|+ε), then u², u⁴, u⁸ by iterative squaring). No data pipeline changes needed.
- Theoretical motivation: complementary representations. complex_powers retains amplitude
  kurtosis (QAM/PSK separator); unit_phasor_powers removes amplitude (PSK order discriminant).
  The ResNet can allocate early filters to amplitude variation, later filters to phase structure.
- Only +1,344 params vs JointCSPCNN (901,192 vs 899,848): overhead is the first Conv1d only
- Registered for use via `modreczoo-train --models joint_csp_dual`

**`unit_phasor_powers`** channel format (`transforms.py`, `training.py`):
- 6-channel format: [Re(u²), Im(u²), Re(u⁴), Im(u⁴), Re(u⁸), Im(u⁸)] where u = x/(|x|+ε)
- Complements `complex_powers`: pure phase discrimination, amplitude-invariant
- u⁴ channel enables CNNs to implicitly learn the signed re4 cyclostationary profile
  (4PSK: Re(u⁴)=1; π/4-DQPSK: Re(u⁴)=-1; 8PSK: Re(u⁴)≈0 — Swami & Sadler 2000)
- Registered as channel format: `modreczoo-train --sweep-channel-formats unit_phasor_powers`

**`OnlineModrecDataset`** (`data.py`):
- `IterableDataset` that generates signals on-the-fly using `simulation.generate_signal`
- Matches `baseline_32768_200k` parameter distribution (SNR ∈ [0,30] dB, AWGN, uniform sampling)
- Worker-safe: each DataLoader worker has an independent seeded RNG
- Use case: unlimited training data; effective data augmentation for large models or overfitting scenarios
- Per-signal: ~2ms signal gen + ~3ms CSP features (K=32768) → 8 workers yield ~400 signals/s
- API: `get_online_data_loader(label_to_id, model_name, batch_size, num_workers, steps_per_epoch)`

### 8.4 Phase Transition Results (200k, 2026-05-29)

**Finding: Dataset-size and signal-length effects are separable and additive.**

Phase transition sweep complete (9/10 K values; K=32768 pending):

```
  K       ~Sym   200k acc  40k acc   Δ(data)   Notes
  ─────────────────────────────────────────────────
     64      5    0.3230    0.3140   +0.009    outlier: noise floor
    128     10    0.4130    0.3870   +0.026
    256     20    0.5031    0.4740   +0.029
    512     40    0.5850    0.5580   +0.027
   1024     80    0.6452    0.6180   +0.027
   2048    161    0.6931    0.6640   +0.029
   4096    321    0.7309    0.7040   +0.027
   8192    643    0.7599    0.7370   +0.023
  16384   1285    0.7880    0.7640   +0.024
  32768   2570    0.8143    0.7880   +0.026    DONE

  Mean Δ(200k − 40k) for K ≥ 128: +0.027 pp
```

**Per-class gain from K=4096 → K=32768 (200k CSP-only):**

```
  Class       K=4096   K=32768    Δ       Note
  ────────────────────────────────────────────────
  64-QAM       0.429    0.614   +0.185   QAM needs many symbols
  16-QAM       0.722    0.862   +0.140   QAM amplitude converges slowly
  256-QAM      0.507    0.639   +0.132   highest-order QAM, hardest
  4-PSK        0.787    0.856   +0.069
  8-PSK        0.722    0.765   +0.043
  2-PSK        0.932    0.976   +0.044   simple BPSK; converges fast
  pi/4-DQPSK   0.783    0.810   +0.027   unique signed d^4 profile
  MSK          0.965    0.993   +0.028   constant envelope; trivial
```

CSP-only ceiling at K=32768, 200k: **81.43%** (vs 40k: 78.8%, vs JointCSP ensemble 40k: 81.22%)

**Key findings:**

1. **Feature-noise floor at K=64**: Only +0.9pp improvement from 5× more data. At ~5 symbols,
   estimation noise of CSP features dominates — more training samples can't compensate for
   per-signal noise. This is distinct from the integration-gain phase boundary.

2. **Separability**: For K ≥ 128, the dataset-size gain (+2.7pp) is constant and independent
   of K. The phase transition curve shifts up uniformly; 5× more data does not change the
   slope of integration gain, only the intercept.

3. **Diminishing data returns**: The 40k → 200k gain (+2.7pp) is smaller than the K=4096 →
   K=32768 gain (+5.6pp for 40k). Signal-length (more CSP integration) is more valuable than
   more training examples at this feature regime.

4. **Class-heterogeneous integration**: QAM classes (64-QAM: +18.5pp, 16-QAM: +14.0pp) gain
   far more from longer K than PSK/MSK. QAM needs both amplitude AND phase statistics to
   converge; PSK relies primarily on phase, which converges faster.

5. **CSP-only at K=32768, 200k nearly matches JointCSPCNN ensemble at 40k (81.22%)**: The
   signal branch may add less on 200k than expected, since CSP features themselves are stronger.

### 8.5 Architecture Search Results (200k, IN PROGRESS)

Architecture search (`train_joint_200k.py`) running on GPU — 6 configs, ~3 min/epoch:
- JointV2-s0, JointV2-s42   — JointCSPCNN, 899k params, LR=1e-3, 60 epochs
- JointV2b-s0, JointV2b-s42 — JointCSPCNN, 899k params, LR=2e-3, 80 epochs
- AttnV2b-s0, AttnV2b-s42   — JointCSPAttn (attention pooling), 932k params, LR=2e-3, 80 epochs

**Interim results (JointV2-s0):**

```
  Epoch   val     best    Note
  ──────────────────────────────────────────────────────
    10    0.7533  0.7981  peak LR (destabilized); true best ~ep8
    20    0.7994  0.8108  converging; within 0.35pp of CSP-only
    30    0.8172  0.8492  ★ ABOVE CSP-ONLY by +3.49pp; best peaked ~ep26
    40    0.8444  0.8492  val recovered; best unchanged — plateau forming
    50    0.8403  0.8572  best ticked up +0.80pp — still improving in LR tail
  CSP-only (200k, K=32768):  0.8143   ← ceiling beaten by +4.29pp at ep50
```

    60    0.8462  0.8572  training complete — best unchanged from ep50
  CSP-only (200k, K=32768):  0.8143   ← beaten by +4.14pp

  [JointV2-s0] Test: 0.8557  (best val: 0.8572)

Signal branch contribution at 200k scale:
  Test accuracy:   85.57%
  vs CSP-only:     +4.14pp (81.43%)
  vs 40k ensemble: +4.35pp (81.22%)
  Val/test gap:    0.15pp (very healthy generalization)

JointV2-s42 started — 5 configs remaining.

Expected arch search completion: ~21.8h from 02:21 (≈ 23:00 today).
Reference: CSP-only (200k, K=32768) = 81.43%; beaten at epoch 30 (JointV2-s0).
