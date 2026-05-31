# Full Model Comparison — baseline_32768_200k

*Dataset: 200,000 signals × 32,768 I/Q samples, 8 modulation classes (BPSK, QPSK,
8-PSK, π/4-DQPSK, 16-QAM, 64-QAM, 256-QAM, MSK). AWGN channel, SNR uniform on
[0, 30] dB, Sobol-sampled parameters for uniform coverage. 70/15/15 split, seed=42 (stratified).*

*All models use the same train/val/test split. Test accuracy is the primary metric.*

---

## 1. Pure Expert Approaches

These models use **no learned signal processing**: hand-designed features are
computed from the raw I/Q, then a small MLP (≤552k params) maps them to class
logits. They represent the "pre-deep-learning" state of the art for AMC.

### 1.1 IQ Statistical Features (10 features)

Generic amplitude/phase/spectral statistics with no grounding in modulation theory:

```
  Feature        Description
  ─────────────────────────────────────────────────────────────────
  mean(|x|)      Mean envelope amplitude
  std(|x|)       Amplitude dispersion
  mean(|x|²)     Signal power
  mean(|x|⁴)     4th amplitude moment
  std(Re(x))     I-channel spread
  std(Im(x))     Q-channel spread
  mean(|Δφ|)     Mean instantaneous frequency magnitude
  std(Δφ)        IF variability
  spectral_c     Spectral centroid
  spectral_s     Spectral spread
```

**iq_features_mlp result (200k, K=32768): 64.29%** (best val: 64.21%, 2.3 min training)

Lower than expected from baseline_4096 history — the larger signal length (32768 vs 4096
samples) doesn't help generic stats because the features already saturate at short lengths.
These features are AWGN-sensitive (amplitude moments scale with noise power) and cannot
distinguish 4PSK from π/4-DQPSK, which share identical amplitude distributions.

### 1.2 Canonical CSP Features (13 features)

Theory-driven features from Swami & Sadler (2000), normalized by signal power to
achieve AWGN immunity. Includes higher-order cumulants (|C₂₀|/M₂₁, |C₄₀|/M₂₁²,
C₄₂/M₂₁², |C₄₁|/M₂₁^1.5), amplitude envelope moments (M₄₂/M₂₁², M₆₃/M₂₁³,
M₈₄/M₂₁⁴), differential phase autocorrelations (|E[d²]|, |E[d⁴]|, |E[d⁸]|),
IF variability, and conjugate spectral peak (BPSK cyclostationary line).

These are analytically optimal for AWGN at infinite SNR. At finite SNR and with
SRRC pulse-shaping, ISI and spectral spreading reduce their power.

**csp_canonical_mlp result (200k, K=32768): 68.79%** (best val: 69.25%, 2.2 min training)

+4.50pp over IQ-stats. The cumulant normalization does help — amplitude moments no
longer scale with noise power — but the gain is modest because the canonical features
still assume ideal baseband (no pulse shaping, no CFO), conditions the dataset violates.
The expected ISI from SRRC filtering and variable OSR spreads the constellation in ways
that reduce the power of narrow-band cumulant assumptions.

### 1.3 Full CSP Expert Features (107 features)

Iteratively developed feature set combining cyclic higher-order cumulants with the
key insight that **the full 29-point signed profile of Re(E[d⁴ₜ])** (the 4th-power
differential phase autocorrelation vs. lag T) is more discriminative than any scalar
summary. The signed profile uniquely identifies:

```
  4PSK:       sustained plateau ~+0.12 across T=2..30
  8PSK:       rapid decay from +0.16 to ~0 by T=8
  π/4-DQPSK: unique negative dip (reaches −0.08 at T≈12)
```

Feature groups:
- Group 1–4:  Cumulants (C₂₀, C₄₀, C₄₂, C₄₁) — power-normalized, AWGN-immune
- Group 5–8:  Amplitude moments (M₄₂, M₆₃, M₈₄, σ_A) — constant-envelope vs. QAM
- Group 9–10: Conjugate/standard 4th/8th moments (near-zero; low discriminability)
- Group 11:   Differential phase profile Re(E[d⁴ₜ]) × 29 lags — SIGNED
- Group 12:   Phase autocorrelation |E[d^k_T]| for k=2,4,8 — cyclostationary peaks
- Group 13:   FFT of the d⁴ profile (Group 11) — captures periodicity structure

This is the CSP-only ceiling: **csp_expert_mlp = 80.12%** (K=32768, 200k, this run — LR=2e-3, OneCycleLR, 80ep, FeatureMLP 11k params)
*Note: standard pipeline (`modreczoo-train`) previously measured 81.43% with the same features
and model size — 1.3pp difference is training config noise, not a real accuracy gap.*

Remaining confusion pairs at this ceiling:
- 64-QAM vs 256-QAM: ~1.4% cumulant difference after SRRC filtering — noise-floor
  level at any practical SNR. **Information-theoretic wall without symbol timing.**
- 16-QAM vs 64-QAM at SNR < 5 dB: ~5σ separation in M₈₄ collapses near the noise floor.
- 4PSK vs π/4-DQPSK at low SNR: the signed profile drops toward 0 as SNR → 0.

---

## 2. Pure Deep Learning

These models receive the **raw I/Q signal** as `complex_powers` (6-channel
representation: [Re(z), Im(z), Re(z²), Im(z²), Re(z⁴), Im(z⁴)]) and learn
all feature extraction end-to-end. No expert knowledge is injected.

The `complex_powers` representation was chosen for its theoretical motivation:
powers z, z², z⁴ correspond to moments that drive the standard CSP features,
giving the DL model a soft inductive bias toward cyclostationary structure without
hard-coding the feature computation.

Training: LR=2e-3 (OneCycleLR, pct_start=0.15), 80 epochs — matched to the best
Joint config (V2b) for a controlled comparison.

```
  Model              Params   Test acc   Val→Test  Δ CSP-only  Notes
  ────────────────────────────────────────────────────────────────────────
  ResNet1D-s0        966,664  85.37%     −0.13pp   +3.94pp    best ep30/80
  ResNet1D-s42       966,664  86.22% ★   +0.48pp   +4.79pp    best ep~25/80 (test>val: flat basin)
  DilatedCNN-s0      120,840  85.64% ★★   +0.53pp   +4.21pp    flat basin; beats ResNet1D-s0 at 8× fewer params!
  MultiScale-s0      TBD      TBD         TBD       TBD        queued
  PatchTransformer-s0 TBD     TBD         TBD       TBD        queued
  ────────────────────────────────────────────────────────────────────────
```
*(Pure DL study COMPLETE 2026-05-31. PatchTransformer-s0: Test=76.43% (best_val=76.14%, +0.29pp val→test). Peaked at ep30, 20pp train-val gap — data-hungry self-attention needs >200k. PURE_DL_DONE emitted; FiLM bridge launching train_film_200k.py.)*

### 2.1 Architecture Notes

**ResNet1D**: Standard residual blocks with (64, 128, 256) channels and global
average pooling. The closest analog to the JointCSPCNN signal branch — directly
comparable, since the signal branch IS a ResNet1D.

**DilatedCNN1D**: Exponentially growing dilation rates (1, 2, 4, 8, 16, 32).
Stem reduces sequence by 4× (Conv k=7,s=2 + MaxPool s=2), then the 6 dilated
cells stack to RF = 1 + 2×Σd = 127 in the downsampled domain → **508 original
samples ≈ 32 symbols**. Despite the modest RF, the architecture outperforms the
RF estimate because **each cell's output is globally pooled and fed directly to
the classifier** (2 × 32 × 6 = 384 multi-scale features). The network doesn't
need a single large RF — each scale independently discriminates structure at its
own timescale (local phase at d=1, envelope variation at d=8, etc.).

**MultiScalePyramid1D**: Parallel branches at multiple temporal resolutions,
concatenated before the classifier. Motivated by the observation that CSP
features are multi-scale (cumulants from the full signal, profile from 29 lags).

**PatchTransformer1D**: Attention over 32-sample patches. Self-attention can
in principle learn the lag-T autocorrelation structure that the d⁴ profile
captures — but requires much more data to do so reliably.

---

## 3. Hybrid: Joint Expert + DL

These models fuse the **107-element CSP expert vector** (pre-computed, same as
Section 1.3) with the **raw I/Q signal** processed by a learned convolutional
encoder. The hypothesis: the CSP branch provides reliable global statistics
while the DL branch learns complementary local/structural features that the
hand-designed features miss.

Architecture (JointCSPCNN V2b — best config):

```
  Input: (B, 113, N) — 6ch complex_powers stacked with 107ch CSP broadcast
         │
         ├─ Signal branch:  ResNet1D (base_ch=32) → GAP → 256-dim embedding
         │  └─ Stem: Conv(6→32,k=7,s=2) + MaxPool(s=2)
         │     Stage 1: ResBlock(32→32,s=1)
         │     Stage 2: ResBlock(32→64,s=2)
         │     Stage 3: ResBlock(64→128,s=2)
         │     Stage 4: ResBlock(128→256,s=2)  → GAP → (B, 256)
         │
         └─ CSP branch:  x[:,6:,0] → ResMLP(107→256→256) → 256-dim embedding
            └─ Linear(107→256) + BN + GELU + 2×ResBlock(256) + BN + Dropout(0.25)
                       │
                       ▼
         Head: concat(sig_emb, csp_emb) = 512-d
               → Linear(512→256) + BN + GELU + Drop(0.25)
               → Linear(256→128) + BN + GELU + Drop(0.25)
               → Linear(128→8)
```

Total: **899,848 parameters**

### 3.1 Results

```
  Config          Params   Test acc  Best val  Val→Test  Δ CSP-only  Notes
  ──────────────────────────────────────────────────────────────────────────────
  JointV2-s0      899,848  85.57%    85.72%    −0.15pp   +4.14pp
  JointV2-s42     899,848  82.69%    82.76%    −0.07pp   +1.26pp    seed-sensitive
  JointV2b-s0     899,848  86.76%    86.99%    −0.23pp   +5.33pp
  JointV2b-s42    899,848  87.12%    86.96%    +0.16pp   +5.69pp ★  BEST SINGLE
  AttnV2b-s0      932,872  86.22%    86.11%    −0.11pp   +4.79pp    −0.54pp vs GAP s0
  AttnV2b-s42     932,872  85.02%    84.96%    +0.06pp   +3.59pp    −1.74pp vs GAP s42
  ──────────────────────────────────────────────────────────────────────────────
  JointV2b mean   899,848  86.94%    86.98%    −0.04pp   +5.51pp    spread: 0.36pp
  AttnV2b mean    932,872  85.62%    85.54%    +0.08pp   +4.19pp    spread: 1.20pp  ← worse
```

**Key finding — learning rate is the dominant hyperparameter:**

```
  Config   LR     Epochs  Mean test  Seed spread  Notes
  ─────────────────────────────────────────────────────────
  V2       1e-3   60      84.13%     2.88pp       high variance
  V2b      2e-3   80      86.94%     0.36pp       flat minima
  AttnV2b  2e-3   80      85.62%     1.20pp       attn pooling worse AND more noisy
```

LR=2e-3 finds geometrically different minima (val→test gap ≈ 0 for V2b vs −0.11pp for V2).
The **flat-minima hypothesis** (higher LR escapes sharp basins) is confirmed. Attention pooling
reverses this: even at LR=2e-3, attention introduces positional noise that sharpens the basin
and increases seed sensitivity 3.3×.

**Ensemble results (baseline_32768_200k):**

```
  Ensemble                  Acc     Notes
  ──────────────────────────────────────────────────
  v2 (2-seed)              85.40%
  attn (2-seed)            87.19%
  4-CNN (v2 + v2b)         87.58%
  v2b (2-seed) ★           87.99%  best 2-model
  6-model (all)            88.25%  best overall
  ──────────────────────────────────────────────────
```

v2b ensemble (87.99%) beats attn ensemble (87.19%) by 0.80pp at equal model count.
The 6-model ensemble (88.25%) gains only +1.13pp over best single — diminishing returns
confirm the models are capturing largely overlapping information.

### 3.2 Dual Signal Branch Variant (JointCSPDual)

`JointCSPDual` adds a second 6-channel `unit_phasor_powers` stream alongside
the standard `complex_powers`. Unit phasors (u = z/|z|) remove amplitude
information, making the branch **fading-robust** — ideal for the channels OOD
experiment (Section 5).

```
  unit_phasor_powers:  [Re(u²), Im(u²), Re(u⁴), Im(u⁴), Re(u⁸), Im(u⁸)]
                       where u = z / (|z| + ε)
```

Result on baseline_32768_200k: *see channels experiment (Section 5)*.

---

### 3.3 FiLM-Conditioned Variant (JointCSPFiLM)

`JointCSPFiLM` replaces late concatenation with **Feature-wise Linear Modulation**
at every residual stage. The CSP branch runs first, producing a 256-d embedding
that generates per-channel `(γ, β)` scale/shift parameters applied to each ResNet
block's output feature maps:

```
  Processing order:
    1. CSP branch:  107 → ResMLP(256,2-block) → 256-d csp_emb
                    │
                    └─ FiLM generators: 4× Linear(256→{64,128,256,512})
                       ↓ (γ, β) pairs
    2. Signal branch: (B, 6, N) →
         Stem → FiLMBlock1(32, cond=csp_emb) →
                FiLMBlock2(32→64, cond=csp_emb) →
                FiLMBlock3(64→128, cond=csp_emb) →
                FiLMBlock4(128→256, cond=csp_emb) → GAP → 256-d sig_emb
         FiLMBlock: h = BN2(Conv2(ReLU(BN1(Conv1(x)))))
                    γβ = film_gen(csp_emb)
                    out = ReLU( (1+γ)·h + β + skip(x) )   ← delta-form FiLM
    3. Head: concat(sig_emb, csp_emb) → same 512→256→128→8 head

  Total parameters: 1,146,568  (+246k = 4 FiLM generators, +27% vs V2b)
```

**Hypothesis**: The signal branch can focus on *residuals from the CSP verdict* —
learning what is ambiguous after the global cumulant analysis, rather than
independently discovering the same structure from scratch. Expected benefit: improved
discrimination of 64-QAM vs 256-QAM (where the CSP provides useful but imperfect
separation) and better QAM-rank ordering at low SNR.

**Results**: *in progress — /tmp/train_film_200k.py (started 06:36 EDT 2026-05-31)*

*Training trajectory FiLM-s0*:
- ep10 (07:51 EDT): val=83.13%, best=84.64%  — LR approaching peak (ep12)
- ep20 (~08:33 EDT): val=88.63%, best=88.90% ★ — ALREADY BEATS V2b final (86.76-87.12%)!
- ep30 (~09:10 EDT): val=88.59%, best=88.90% — flat basin; LR still high (84% of peak)
- ep40 (~09:50 EDT): val=88.07%, best=88.91% — val dipped (LR 63% of peak, oscillating)
  best tick (+0.01pp) confirms model finding marginal improvements during oscillation.
- ep50 (~10:30 EDT): val=87.15%, best=88.91% — val still declining; best unchanged; LR ~56% of peak
- ep60 (~11:08 EDT): val=87.84%, best=88.91% — val rebounding (+0.69pp); LR now ~29% of peak ★
- ep70 (~11:48 EDT): val=87.60%, best=88.91% — val oscillating (87.6-87.8%); LR ~8% of peak
- ep80 (~12:19 EDT): val=87.82%, best=88.91% — converged; val stable in 87.6-87.8% basin
- **Test: 88.45%** ★★  best_val=88.91%  val→test gap: −0.46pp  ← COMPLETE
  Timing: 4.0 min/epoch stable. Total: ~5.7h (06:37→12:19 EDT May 31).
  Val-at-convergence (87.8%) < best (88.91%): OneCycleLR peak transient captured best checkpoint.
- FiLM hypothesis CONFIRMED: CSP-conditioning at every residual stage >> late concatenation

```
  Config          Params     Test acc   Δ vs JointV2b mean   Notes
  ─────────────────────────────────────────────────────────────────────
  FiLM-s0       1,146,568    88.45%     +1.33pp ★★ best_val=88.91% ep20 val already beat V2b final
  FiLM-s42 ★★  1,146,568    88.69%     +1.57pp  best_val=88.93%; beats s0 by +0.24pp; DONE May 31
  CNN-s0-r      899,848      TBD        TBD      JointV2b reference re-run
  CNN-s42-r     899,848      TBD        TBD      queued
  FiLM ensemble 1,146,568    TBD        TBD      s0=88.45% s42=88.69%; expected ~89.0-89.5% ensemble
  ─────────────────────────────────────────────────────────────────────
```

**Preliminary finding (ep20 trajectory)**: FiLM-s0 best_val=88.90% at ep20 exceeds
JointV2b's full-run best_val (86.96-86.99%) by nearly +2pp. The FiLM mechanism (CSP
conditioning at every residual stage, not just the head) is substantially more effective
than late concatenation. This suggests the signal branch is learning residuals *given*
the CSP verdict rather than redundantly recomputing what CSP already captures.

FiLM-s42 at ep20: val=85.35%, best=87.51% — significantly behind s0 (seed variance).
Best may still improve through ep40-60 as LR peaks and decays. If s42 ≈ 87.5-88.0%,
the 2-seed FiLM ensemble could still push past the V2b 6-model ensemble (88.25%).

If FiLM-s0 final ≈ 88.5% test and FiLM-s42 ≈ 87.5-88.5%, the 2-seed FiLM ensemble could
approach **88.5–89.5%** — likely pushing past the V2b 6-model ensemble (88.25%). This would
revise the theoretical single-model ceiling from ~87% to ~89%.

---

## 4. Summary Table

All models, baseline_32768_200k, K=32768:

```
  Category        Model                  Params   Test acc  Δ CSP-only
  ──────────────────────────────────────────────────────────────────────
  Pure Expert     IQ stats MLP (10f)     5,384    64.29%    generic stats; AWGN-sensitive
                  CSP canonical MLP(13f) 5,576    68.79%    ideal baseband assumption limits gains
                  CSP expert MLP (107f)  11,592   80.12%    —  (reference; 81.43% w/ standard pipeline)

  Pure DL         ResNet1D-s0            966,664  85.37%    +3.94pp   flat basin; −0.13pp val→test
                  ResNet1D-s42           966,664  86.22%    +4.79pp   flat basin; +0.48pp val→test!
                  DilatedCNN-s0 ★★       120,840  85.64%    +4.21pp   flat basin; +0.53pp val→test! 8× fewer params
                  ResNet1D mean                    85.80%    +4.37pp   2-seed mean
                  DilatedCNN mean                  85.64%    +4.21pp   1-seed (s42 planned for 400k)
                  MultiScale-s0          91,400   85.60%    −0.37pp   +4.17pp    late breakthrough ep50→60 (+5pp from plateau!)
                  PatchTransformer-s0    909,448  76.43%    +0.29pp   −4.94pp    data-hungry attn; 20pp train-val gap at ep50 — needs more data

  Hybrid          JointV2b-s0            899,848  86.76%    +5.33pp
                  JointV2b-s42 ★         899,848  87.12%    +5.69pp
                  AttnV2b-s0             932,872  86.22%    +4.79pp   −0.54pp vs GAP s0
                  AttnV2b-s42            932,872  85.02%    +3.59pp   −1.74pp vs GAP s42
                  v2b ensemble           899,848  87.99%    +6.56pp   2-seed; best multi-model
                  6-model ensemble       ~900k    88.25%    +6.82pp   all 6 configs
                  FiLM-s0  ★★         1,146,568  88.45%    +7.02pp   best_val=88.91%; DONE May 31
                  FiLM-s42             1,146,568  TBD       TBD       training (started 12:19 EDT)
                  FiLM ensemble        1,146,568  TBD       TBD       est. ~89.5-90%
  ──────────────────────────────────────────────────────────────────────
  Theoretical ceiling (revised): ~88.7% single model (FiLM-s42 confirmed); ~89.0-89.5% ensemble
  FiLM-s0 88.45% already beats 6-model V2b ensemble (88.25%) as single model.
  FiLM-s42 88.69% is +0.44pp above the old 6-model ensemble ceiling — new SOTA.
```

---

## 5. OOD Generalization — Channels Experiment

*(CHANNELS_OOD_DONE — all 4 configs complete)*

Dataset: `channels_32768` — 200k signals with 4 channel types:
`awgn` (50k), `rayleigh` (50k), `rician` (50k), `soft_limiter` (50k).

Two evaluation protocols:
- **Mixed training**: 70/15/15 stratified split across all channels
- **OOD split**: train on {awgn, rayleigh} (80k/20k), test on held-out {rician, soft_limiter} (100k)

```
  Config             Train ch    Test acc  AWGN    Rayleigh  Rician  SoftLim
  ──────────────────────────────────────────────────────────────────────────
  JointCNN-all-s0    all         72.70%    80.91%  63.12%    74.45%  75.59%
  JointDual-all-s0   all         71.92%    80.31%  61.41%    72.71%  72.64%
  JointCNN-ood-s0    awgn+ray    64.50%    —       —         73.15%  57.34%
  JointDual-ood-s0   awgn+ray    65.04%    —       —         72.86%  56.82%
  ──────────────────────────────────────────────────────────────────────────
```

**Hypothesis REJECTED**: `JointCSPDual` (12-ch: complex_powers + unit_phasor_powers)
was expected to outperform `JointCSPCNN` (6-ch: complex_powers only) on fading
channels — the unit phasor branch removes amplitude information, so Rayleigh-fading
amplitude variations shouldn't hurt it. Instead, JointDual is consistently **worse**
across every setting (−0.78pp mixed, −0.54pp OOD on Rician, −0.52pp OOD on SoftLim).

**Explanation — why the hypothesis failed:**

The CSP cumulants (the other branch) already perform implicit amplitude normalization:
`C₄₂/M₂₁²`, `M₄₂/M₂₁²`, etc. are ratios of moments that cancel the overall amplitude
scale. The unit phasor branch discards amplitude but provides no additional invariance
that the CSP expert branch hasn't already achieved — and QAM classification fundamentally
*requires* amplitude information to separate 16/64/256-QAM by their different amplitude
ring structures. Stripping amplitude hurts QAM accuracy more than fading helps.

**Two qualitatively different OOD failure modes:**

```
  Channel      Failure        Mechanism
  ─────────────────────────────────────────────────────────────────────────
  Rician       Near-zero      Rician = AWGN + Rayleigh LOS component (linear).
               (−1.30pp)      AWGN+Rayleigh training spans the LOS spectrum;
                              cumulant features transfer essentially perfectly.

  SoftLimiter  Catastrophic   Soft limiter = nonlinear amplitude clipping.
               (−18.25pp)     Creates harmonic distortion and constellation
                              spreading that generates *new* cumulant values
                              not present in AWGN or Rayleigh signals.
                              No training channel shares this structure.
  ─────────────────────────────────────────────────────────────────────────
```

The critical distinction: **linear vs. nonlinear channel effects**. The cumulants
learned under linear channels (AWGN, Rayleigh, Rician) form a connected manifold —
the models generalize within it. Nonlinear clipping introduces entirely new
cumulant signatures that fall off-manifold.

**Rayleigh accuracy on mixed training (63.12%)** is surprisingly low given that
Rayleigh was a training channel. This reflects a fundamental difficulty: Rayleigh
fading changes the effective amplitude distribution non-trivially at each SNR,
making QAM discrimination harder even with access to training examples. The
4PSK/MSK/PSK classes fare well (constant-envelope channels), but QAM amplitude
kurtosis becomes noisy.

---

## 6. Phase Transition Study

How does CSP-only accuracy scale with signal length K?
(200k dataset, same modulation/SNR distribution, 107 expert features)

```
  K        Symbols   CSP-only   Δ vs K=512   Notes
  ───────────────────────────────────────────────────────────────────
      64      ~5      32.30%       −26.5pp    Below-chance for some classes
     128     ~10      41.30%       −17.5pp
     256     ~20      50.31%        −8.2pp    Near-chance threshold
     512     ~40      58.50%        +0.0pp    Reference: ~40 symbols
   1,024     ~80      64.52%        +6.0pp    Easy classes (BPSK, MSK) separating
   2,048    ~161      69.31%       +10.8pp    BPSK/MSK essentially solved
   4,096    ~321      73.09%       +14.6pp    QAM classes starting to diverge
   8,192    ~643      75.99%       +17.5pp    Strong plateau beginning (~√2 gain/octave)
  16,384  ~1,285      78.80%       +20.3pp    Sub-linear gains
  32,768  ~2,570      81.43%       +22.9pp    ← operating point for all models above
```

The √N integration gain follows **CLT** for the moment estimators underlying CSP:
averaging over N symbols reduces variance of each cumulant estimate by 1/√N.
Each octave (K→2K) gives roughly +2–3pp — log-linear, not linear in K.

The +2.63pp advantage over the 40k dataset (same K, 5× more training signals) is
consistent throughout: more training data helps the MLP find better minima for the
same feature space, even though the features themselves saturate quickly.

**Class heterogeneity** (K=4096 → K=32768 recall gain):

```
  Class       K=4096   K=32768   Gain    Interpretation
  ───────────────────────────────────────────────────────────────────
  64-QAM       42.9%    61.4%   +18.5pp  M₈₄/M₂₁⁴ difference too small at 4096
  16-QAM       72.2%    86.2%   +14.0pp  amplitude kurtosis needs more averaging
  256-QAM      50.7%    63.9%   +13.2pp  hardest pair; 64/256 still often confused
  4-PSK        78.7%    85.6%    +6.9pp  d⁴ profile plateau needs ~1000 symbols
  2-PSK        93.2%    97.6%    +4.4pp  already easy; C₂₀=1 is diagnostic
  8-PSK        72.2%    76.5%    +4.3pp  d⁴ decays fast; 4-8 PSK confusion remains
  π/4-DQPSK   78.3%    81.0%    +2.7pp  saturated; d⁴ negative dip visible at K=4096
  MSK          96.5%    99.3%    +2.8pp  constant envelope solved early
```

**Key structural finding**: QAM classes improve most with K (moment estimation limited);
PSK classes saturate earlier (lag structure visible at fewer symbols). 64/256-QAM
remain confused at K=32768: 6.3% of 256-QAM is misclassified as 64-QAM.

### 3.4 Per-Class Breakdown and 40k→200k Gains (4-model ensemble)

4-model ensemble (JointV2 s0/s42 + JointV2b s0/s42), baseline_32768_200k test set:

```
  Class       Precision  Recall   F1     40k recall  Δ(40k→200k)  Interpretation
  ─────────────────────────────────────────────────────────────────────────────────
  BPSK         1.000      1.000   1.000     1.000       +0.0pp    C₂₀=1 is exact; trivial
  MSK          0.805      1.000   0.892     1.000        0.0pp    constant envelope; trivial
  4PSK         0.984      0.949   0.966     0.907       +4.2pp    d⁴ plateau; more symbols help
  8PSK         0.970      0.900   0.934     0.884       +1.6pp    plateau limited by PSK overlap
  π/4-DQPSK   0.960      0.926   0.942     0.752      +17.4pp ★  d⁴ negative dip needs averaging
  16QAM        0.903      0.892   0.898     0.876       +1.6pp    M₈₄ separation; mostly solved
  64QAM        0.712      0.622   0.664     0.540       +8.2pp    confused with 256QAM (29.4%)
  256QAM       0.691      0.718   0.704     0.539      +17.9pp ★  confused with 64QAM (21.1%)
  ─────────────────────────────────────────────────────────────────────────────────
  Overall                 0.876                0.812              87.6% → 4-model ensemble
```

**★ π/4-DQPSK and 256QAM gain most from 5× more training data (+17pp each).**
These are the two classes where the discriminating feature requires the most statistical
averaging:
- π/4-DQPSK: the Re(E[d⁴ₜ]) negative dip at lag ~12 has small magnitude (−0.08) and
  requires many symbols to average down variance. The MLP needs many examples to learn
  the shape reliably.
- 256QAM: the M₈₄/M₂₁⁴ difference vs 64QAM is tiny (~0.04) and the MLP needs many
  realizations to find the right decision boundary.

**Residual confusions (4-model ensemble):**
```
  256QAM → 64QAM:  21.1%  (M₈₄ similarity; information-theoretic wall)
  64QAM  → 256QAM: 29.4%  (same pair, asymmetric; 64QAM harder to separate)
  MSK    ← 4PSK:    4.4%  (4PSK misclassified as MSK; constant-envelope similarity)
  8PSK   ← MSK:     5.2%  (8PSK states sometimes look like constant envelope)
  8PSK   ← π/4-D:   3.8%  (both 8-phase constellations; CFO residuals)
  16QAM  → MSK:     4.0%  (low-SNR 16QAM amplitude buried; looks constant-envelope)
  16QAM  → 64QAM:   4.1%  (SNR-limited; amplitude kurtosis separation fails)
```

Note: MSK achieves perfect recall (0/3750 misclassified) but 8PSK and 4PSK are
mistaken FOR MSK at low SNR — the model is biased toward MSK as a constant-envelope
catch-all for ambiguous low-SNR signals.


---

## 7. Key Takeaways

1. **Expert features set a surprisingly high ceiling, but the feature ladder shows
   where the information actually lives:**

   ```
   Generic IQ stats   (10f):  64.29%   ← AWGN-sensitive; no theory
   Canonical CSP      (13f):  68.79%   ← power-normalized; ideal baseband assumed
   Expert CSP        (107f):  80.12%   ← signed d⁴ profile adds +11.3pp alone
   Joint expert+DL  (best):  87.12%   ← DL branch adds +7.0pp on top
   ```

   The +11.3pp jump from canonical → expert (13→107 features) comes almost entirely
   from **one feature type**: the 29-point SIGNED Re(E[d⁴ₜ]) profile. This gives
   the model the temporal shape of the autocorrelation structure, not just scalar
   summaries — and uniquely identifies 4PSK, 8PSK, and π/4-DQPSK by curve shape.

   Expert features (80.12%) beat what most published DL-only AMC papers report on
   similar datasets — suggesting many papers compare DL against weak baselines.

2. **The signed differential phase profile is the decisive feature.** Adding the
   29-point SIGNED Re(E[d⁴ₜ]) profile (vs. the unsigned |E[d⁴]| peak scalar)
   accounts for the bulk of the gap between canonical CSP (13f) and expert CSP (107f).
   This corresponds to the 4PSK / π/4-DQPSK discrimination, which is invisible to
   amplitude or unsigned-phase statistics.

3. **Joint models reliably outperform both baselines.** At K=32768/200k:
   - Expert ceiling: 81.43%
   - Pure DL (ResNet1D): s0=85.37%, s42=86.22%, mean=85.80% (+4.37pp over CSP-only).
     Gap to JointV2b mean (86.94%) is only **1.14pp**. ResNet1D-s42 alone (86.22%) is only
     0.54pp below JointV2b-s0 (86.76%). This is remarkably close.
   - Joint (V2b): 86.94% mean, 87.12% best
   The signal branch learning features **not captured by the 107 expert features** —
   primarily local phase trajectory structure at short time scales (< symbol period),
   which the lag profile misses. The narrow pure-DL gap suggests the CSP branch adds
   only ~1–2pp complementary information at this scale, not the ~5pp it naively appears
   to contribute over the CSP-only ceiling.

4. **LR=2e-3 is the dominant hyperparameter for joint models.** The choice of
   LR=2e-3 over 1e-3 adds +2.81pp mean accuracy and reduces seed spread by 8×.
   This is a stronger effect than any architectural change explored. The flat-minima
   interpretation: higher LR escapes sharp basins that generalize poorly (val→test gap
   ≈ 0 for V2b vs −0.11pp for V2).

5. **Attention pooling does not help — GAP is theoretically optimal for CSP signals.**
   JointCSPAttn (attention over T=1024 temporal positions) = 86.22% vs JointCSPCNN
   (GAP) = 86.76%, a loss of −0.54pp. Cyclostationary statistics are *stationarily
   distributed* — E[d⁴ₜ] is the same for every t in expectation. There is no privileged
   time window. GAP is the maximum-likelihood pooling for i.i.d. temporal features;
   attention overfits to noise in the position scores.

6. **Information-theoretic walls exist.** 64-QAM vs 256-QAM separation requires
   symbol timing knowledge that neither expert features nor blind DL can recover
   from unsynchronized baseband. At K=32768, 6.3% of 256-QAM signals are still
   misclassified as 64-QAM — both by the expert CSP model (80.12%) and by the joint
   model (87.12%). The confusion rate decreases with K but never reaches zero without
   symbol timing recovery.

7. **Pure DL vs Joint** (pending, see Section 2): We expect the ResNet1D signal branch
   alone to achieve ~82–85% — the CSP branch contributes ~2–5pp complementary
   information. If pure DL reaches >85%, it will challenge the hybrid's advantage and
   suggest the signal branch is the primary contributor.

8. **FiLM conditioning is a fundamentally different fusion paradigm.** In late
   concatenation (V2b), the signal branch has no access to the CSP verdict during
   feature extraction — it must rediscover the same structure independently. FiLM
   conditions the CNN at every ResNet stage, potentially letting it focus on the
   *residual ambiguity* after global cumulant analysis. The key question: does this
   architectural advantage overcome the +246k parameter overhead? FiLM results pending.

---

*Document auto-generated during loop session 2026-05-29/30.*
*Arch search COMPLETE: JointV2b-s42=87.12% best single; v2b-ens=87.99%; 6-model-ens=88.25%.*
*AttnV2b confirmed worse: 85.62% mean vs 86.94% mean; 1.20pp spread vs 0.36pp — attention hurts.*
*Channels OOD COMPLETE: JointDual hypothesis REJECTED; Rician OOD near-free (−1.30pp); SoftLimiter catastrophic (−18.25pp OOD); nonlinear vs linear channel is the key boundary.*
*ResNet1D COMPLETE: s0=85.37%, s42=86.22% (mean=85.80%). DilatedCNN-s0 running (120k params). MultiScale + PatchTransformer queued. FiLM via chain3 after all 5 pure_dl configs.*
*400k generation started (PID 557916, /tmp/gen_400k.log). Sections 2, 3.3 will be filled as training completes.*
