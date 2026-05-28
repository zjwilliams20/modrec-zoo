# CSP Expert + Hybrid DL: baseline_4096 Progress

**Dataset:** `baseline_4096` — 200k signals, 8 classes, 4096 complex samples, SNR 0–30 dB (uniform), SRRC filtered
**Task:** 8-class AMC (2PSK, 4PSK, 8PSK, π/4-DQPSK, 16QAM, 64QAM, 256QAM, MSK), 25k signals/class
**Split:** 70% train / 15% val / 15% test (stratified, seed=42) → Train=139,999 Val=30,001 Test=30,000
**DL baseline:** ~76% (APF, resnet_1d + complex_powers, multiscale + real_imag)

*Last updated: 2026-05-28*

---

## Accuracy Timeline

```
Version   Features   Architecture          Test acc   Delta    Key addition
────────────────────────────────────────────────────────────────────────────────
v13          13      tiny MLP              ~0.630      —       Swami & Sadler cumulants
v15          70      flat ResMLP-256-4b    0.6844    +5.4%    multi-scale pc4 profile
v16          74      flat ResMLP-256-4b    0.7096    +2.5%    FFT-based autocorr, Groups 8-9
v18          77      flat ResMLP ensemble  0.7207    +1.1%    re4_min, re4_asym, pc4_min
v19         107      flat ResMLP ensemble  0.7262    +0.6%    re4_at_peak + 29-pt raw re4 profile
v19-hybrid  107      HybridCSPNet ens.     0.7288    +0.3%    1D-conv branch on re4 profile
v19-joint   107+raw  JointCSPCNN ens.      TBD       —       CSP features + mini-ResNet on signal
```

**DL target: 76%** — achievable; the gap from 72.9% → 76% is ~1,000 fewer errors (of 8,200 total).

---

## Error Budget (v19 ensemble, 30k test samples)

```
Source                          Errors    % of total   Addressable?
─────────────────────────────────────────────────────────────────────
64QAM ↔ 256QAM confusion        ~2,638      32.1%      No  — info-theoretic limit
PSK triangle (4/8/π4 cross)      1,678      20.4%      Yes — main remaining target
16QAM ↔ 64QAM / 16QAM ↔ 256QAM ~1,931      23.5%      Partially — low-SNR limited
Other cross-class                ~1,967      23.9%      Partially
Total                            8,214     100.0%
```

To reach 76% (7,200 errors): need to eliminate **~1,014 errors**.
- Fixing 60% of PSK triangle (1,007 errors) → 76.0% ✓

---

## PSK Triangle Detail (v19 ensemble)

The 4PSK / 8PSK / π/4-DQPSK three-way confusion drives 20% of all errors.
Theoretical discriminants that work:

```
Feature           4PSK         8PSK         π/4-DQPSK   Key?
────────────────────────────────────────────────────────────
Re(E[d^4]) at Ts  ≈ +1.0       ≈  0          ≈ -1.0     YES - fundamental sign
re4_min (T=2..30)  +0.06        -0.03         -0.19      YES (but noisy)
pc4_late (|.|)     ≈ 0.23       ≈ 0.00        ≈ 0.18     Partial (4PSK≈π4-DQPSK)
pc8_late (|.|)     ≈ 0.38       ≈ 0.30        ≈ 0.38     PSK vs QAM only

re4 PROFILE SHAPE (T=2..30):
  4PSK:      +.17 +.13 +.12 +.12 +.12 ... sustained positive plateau
  8PSK:      +.16 +.08 +.04 +.02 +.00 ... rapid decay to near-zero
  π/4-DQPSK: +.15 +.03 −.01 −.05 −.08 −.04 +.01 ... single negative DIP then recovery
```

The raw 29-point re4 profile (Group 10 in v19) gives the model the complete shape.

---

## Why the Expert-Feature Ceiling is ~73%

1. **64/256-QAM**: M84 differs by only 0.3% → noise-floor level after SRRC dilution. 
   No feature can separate them without symbol timing. (~32% of all errors are this pair.)

2. **Low-SNR regime** (0–5 dB, 16.8% of dataset): cyclic autocorrelation features lose
   discriminative power when noise amplitude exceeds the cyclostationary signal component.
   The feature values converge toward noise floor for all modulations.

3. **PSK at low SNR**: 4th-power phase noise = 4× phase noise → wraps at 5 dB SNR,
   destroying the Re(E[d^4]) signal. This causes the residual PSK triangle errors.

---

## Key Feature Engineering Discoveries

### Group 8: Multi-scale phase concentration profile (v15)
FFT-based autocorrelation: O(N log N) for all 29 lags simultaneously.
`pc4_profile[T] = |E[(x/|x|)^4 · conj((x/|x|)^4[n−T])]|` for T=2..30.
Summary stats: early/mid/late/decay/max/min/argmax → +2.5% over canonical CSP.

### Signed Re(E[d^4]) = the decisive sign (v18)
```
4PSK:       d^4 = +1 always  → Re(E[d^4]) = +1 at Ts
π/4-DQPSK: d^4 = -1 always  → Re(E[d^4]) = -1 at Ts   ← NEGATIVE!
8PSK:       d^4 = ±1 equally → Re(E[d^4]) ≈  0 at Ts
```
`|E[d^4]|` throws away this sign — once re4_min was added, 4PSK recall jumped 6.5%.

### Full signed re4 profile (v19)
Passing all 29 signed values lets the model see the complete shape:
- Single negative dip → π/4-DQPSK
- Sustained plateau   → 4PSK
- Fast decay to zero  → 8PSK
This is more informative than min/max summary statistics alone.

### What doesn't help for QAM discrimination
- `|E[x^6]|`, `|E[x^8]|` (Group 9): near-zero for all non-BPSK due to M-fold
  constellation symmetry. Useless for QAM-order discrimination.
- Entropy, histogram features, cross-statistics: all give ≤0.18σ separation for
  64 vs 256-QAM due to SRRC dilution of amplitude differences.
- Blind symbol timing: best estimator has 22-183% error → can't resample at symbol instants.

---

## Architecture Experiments

### Flat ResMLP (v18, v19)
4 residual blocks, hidden=256, OneCycleLR, BatchNorm, GELU, label_smoothing=0.05.
Architecture is NOT the bottleneck — proved by sweeping tiny→large MLP, XGBoost,
feat-dropout ensemble: all plateau at the same accuracy.

### HybridCSPNet (v19)
MLP branch on scalar features (78-d) + 1D-conv branch on re4 profile (29-d).
The conv branch learns profile shapes without hand-coded summary statistics.
Result: +0.26% over flat MLP → **72.88% ensemble**. Modest improvement.
8PSK→π4-DQPSK confusion slightly worsened (model over-indexes on negative dip).

### JointCSPCNN (current)
MLP branch on 107 CSP features → 256-d embedding.
Mini-ResNet1D on complex_powers(x) format [6ch × 4096] → 128-d embedding.
Fusion: concat(384-d) → MLP → 8 classes. 480k total params.
**Result: TBD** — expected to significantly close the gap to 76%.

---

## Next Steps (ordered by expected impact)

1. **JointCSPCNN results** (running now):
   - If ≥75%: try larger signal branch (base_ch=32) and deeper fusion
   - If 73-74%: ensemble with flat v19 MLP for marginal gain

2. **Larger signal branch**: base_channels=32 (4× params), possibly reaching closer to 
   the standalone APF/ResNet1D performance on this dataset

3. **Fix useless features**: Remove `|E[x^6]|`, `|E[x^8]|` from Group 9, replace with:
   - Trimmed M84 = `np.sort(amp**8)[:int(0.95*N)].mean()` (robust to low-SNR noise)
   - `re4_late = mean(re4_real[18:29])` (explicit sustained-plateau indicator for 4PSK)

4. **SNR-adaptive training**: Oversample signals from 0-5 dB range (16.8% of dataset)
   to balance the per-SNR class distribution during training.

5. **Full DL approach**: Use established zoo models (APFNet, ResNet1D + complex_powers)
   directly on baseline_4096. The CSP hybrid adds expert priors but the raw-signal
   DL models may already incorporate the equivalent information.

---

## Theoretical Upper Bound

Given:
- 64 ↔ 256-QAM: ~50% confusion ceiling (equal probability guessing) = ~2,638 unavoidable errors
- Low-SNR (≤5 dB) signals: ~5,040 signals × best-case 30% error = ~1,512 unavoidable errors
- Practical ceiling without symbol timing or known SNR: ~83-87% on this dataset
  (consistent with ZOO.md results on higher-SNR dataset where APF achieves ~84%)

The 76% DL target already achieved by APF/ResNet1D suggests the raw-signal models
are capturing ~60% of what's theoretically available. Expert CSP features capture
~50% (72.9% current). The joint model should capture ~65-70% → 74-76%.
