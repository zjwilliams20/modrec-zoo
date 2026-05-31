# Full Model Comparison — baseline_32768_400k

*Dataset: 400,000 signals × 32,768 I/Q samples, 8 modulation classes (BPSK, QPSK,
8-PSK, π/4-DQPSK, 16-QAM, 64-QAM, 256-QAM, MSK). AWGN channel, SNR uniform on
[0, 30] dB, Sobol-sampled parameters for uniform coverage. 70/15/15 split, seed=42.*

*Companion to `full-comparison-200k.md` — directly comparable, 2× the training data.*

---

## Motivation: Why 400k?

The 200k study revealed two classes with >17pp gain over the 40k baseline:
- **π/4-DQPSK**: needs many examples to average the Re(E[d⁴ₜ]) negative dip (small magnitude −0.08)
- **256QAM**: M₈₄/M₂₁⁴ separation from 64QAM is tiny (~0.04); decision boundary needs many realizations

The 40k→200k trend suggests another +2–4pp overall at 400k, with concentrated gains
in these two classes. The joint model ceiling may also shift up: if the information-theoretic
wall for 64/256-QAM is primarily a *data* limit (MLP variance, not feature informativeness),
then 400k might crack it.

---

## Expected vs. Observed Gains (to be filled)

```
  Model                200k result  400k result  Δ     Notes
  ────────────────────────────────────────────────────────────
  CSP-only (K=32768)   81.43%        TBD         TBD   40k→200k was +2.63pp
  Pure DL ResNet1D     ~85-86%       TBD         TBD
  JointV2b single      86.94% mean   TBD         TBD
  JointV2b ensemble    87.99%        TBD         TBD
  FiLM single          88.45%★★      TBD         TBD   200k DONE; beats 6-model V2b ensemble (88.25%)
  ────────────────────────────────────────────────────────────
```

---

## 1. Phase Transition Study (CSP-only, K varied)

*Script: `/tmp/train_phase_transition_400k.py` — COMPLETE.*

All K values use `csp_expert_features` (107 features) + `FeatureMLP`, LR=2e-3, 80 epochs.
Split: stratified 70/15/15, seed=42.

```
  K        ~Symbols  400k result  200k ref   Δ       N_sigs  Compute  Regime
  ─────────────────────────────────────────────────────────────────────────────
      64      ~4       33.92%      32.30%    +1.62pp  400k    505s    data-limited
     128      ~8       42.26%      41.30%    +0.96pp  400k    474s    data-limited
     256     ~16       51.19%      50.31%    +0.88pp  400k    501s    data-limited
     512     ~32       59.07%      58.50%    +0.57pp  400k    530s    data-limited
   1,024     ~64       64.80%      64.52%    +0.28pp  400k    599s    data-limited
   2,048    ~128       69.33%      69.31%    +0.02pp  400k    730s    ★ CROSSOVER
   4,096    ~256       70.90%      73.09%    −2.19pp  50k*    130s    feature-limited†
   8,192    ~512       74.39%      75.99%    −1.60pp  50k*    195s    feature-limited†
  16,384  ~1,024       76.17%      78.80%    −2.63pp  50k*    328s    feature-limited†
  32,768  ~2,048       78.29%      81.43%    −3.14pp  50k*    597s    feature-limited†
  ─────────────────────────────────────────────────────────────────────────────
  * 50k subsample: K≥4096 compute cost scales as O(K×N); capped to avoid 70h runtime.
  † Negative Δ is a subsample artifact: 50k gives ~35k train vs 200k's ~140k train.
    In the feature-limited regime, training set size directly governs decision boundary
    quality. A fair 400k comparison would require 400k×K=32768 = 72h (infeasible).
```

**Complete findings**:
- K=64:  +1.62pp (data-limited: noisy features, many examples help most)
- K=128: +0.96pp (improvement shrinking as features gain statistical clarity)
- K=256: +0.88pp (IPC-dominated compute; all signals sent to worker regardless of K)
- K=512: +0.57pp (each K-doubling adds ~30–70s compute, not 2×)
- K=1024: +0.28pp (features gaining clarity; MLP benefit shrinking)
- K=2048: +0.02pp ★ **DATA-LIMITED REGIME ENDS.** At ≥128 symbols, 200k examples
          fully exploit CSP feature quality — 400k adds nothing for fixed features.
- K≥4096: negative Δ due to 50k subsample; not a true 400k-vs-200k comparison.
  Estimated true Δ at K=32768 with full 400k: ≈ +0pp (saturated, same as K=2048).

**Core finding**: CSP expert features saturate with ~140k training examples (200k dataset).
The gain from 400k is concentrated in K≤2048 (noisy-feature regime), peaking at
K=64 (+1.62pp) and reaching noise level by K=2048 (+0.02pp).

**Subsample deltas** at K≥4096 are non-monotone (−2.19, −1.60, −2.63, −3.14pp).
Two competing effects: (1) feature quality improves with K (pushes 400k-sub up), (2) fixed
~35k-vs-140k training penalty (pulls accuracy down). The temporary recovery at K=8192
(−1.60) suggests feature quality improvement briefly offsets the training deficit at 512
symbols before diminishing integration gains cede control back to the training deficit.

**Implication for DL**: DL models are more data-hungry than fixed-feature MLPs (learned
features need more examples to generalize). Expect 400k to give +2–4pp for DL at K=32768
where CSP features give essentially +0pp.

**Timing**: IPC-dominated at small K (K=64: 505s ≈ K=128: 474s). Compute overtakes IPC
around K≈2048 (full 400k) or K≈512 (50k subsample).

---

## 2. Architecture Search

*Script: `/tmp/train_arch_400k.py` — starts after 200k GPU experiments complete.*

### 2.1 Pure DL (complex_powers, no CSP)

*Pruned for 400k compute budget (400k epochs ≈ 2-3× longer than 200k). Dropped:*
*ResNet1D-s0 (s42 strictly better), MultiScale-s0 (≈DilatedCNN acc, more params), PatchTransformer (76.43% → 20pp train-val gap, needs >>400k signals for attn to converge).*

```
  Model              Params   400k acc  200k acc  Δ     Notes
  ──────────────────────────────────────────────────────────────────
  ResNet1D-s42       966,664   TBD       86.22%   TBD   200k best pure DL; primary reference
  DilatedCNN-s0      120,840   TBD       85.64%   TBD   200k: +0.53pp val→test; 8× fewer params
  DilatedCNN-s42     120,840   TBD       N/A      TBD   2nd seed; not run at 200k
  ──────────────────────────────────────────────────────────────────
```

### 2.2 Hybrid Joint CSP+DL

```
  Config           Params     400k acc  200k acc  Δ     Notes
  ────────────────────────────────────────────────────────────────────
  JointV2b-s0      899,848    TBD       86.76%   TBD
  JointV2b-s42     899,848    TBD       87.12%   TBD   200k best single ★
  FiLM-s0        1,146,568    TBD       88.45%★★ TBD   200k test=88.45%, best_val=88.91%; +1.33pp vs V2b-s42
  FiLM-s42       1,146,568    TBD       88.69%★★ TBD   200k DONE; beats s0 by +0.24pp; new single-model SOTA
  JointDilated-s0   572,328   TBD       N/A      TBD   NEW: DilatedCNN backbone
  JointDilated-s42  572,328   TBD       N/A      TBD   NEW: DilatedCNN backbone
  V2b ensemble     899,848    TBD       87.99%   TBD
  FiLM ensemble  1,146,568    TBD       TBD      TBD
  6-model grand             TBD        N/A       TBD   DilCNN-s0+ResNet-s42+V2b×2+FiLM×2
  9-model grand             TBD        N/A       TBD   all 9 configs
  ────────────────────────────────────────────────────────────────────
```

**JointCSPDilated hypothesis** (new architecture, no 200k baseline):
- Signal branch: DilatedCNN backbone (d=1,2,4,8,16,32) → 384-d multi-scale embedding
  (avg+max pool per cell) rather than ResNet1D GAP → 256-d.
- Motivation: CSP features are global moment summaries; DilatedCNN's per-cell pooling
  is scale-specific. Less redundant with CSP than ResNet1D's GAP (another global average).
- Parameters: 572k (37% fewer than V2b). Signal branch is 20k vs V2b's ~440k.
- Expected: ~86-88% single — similar to V2b if CSP-dilated complementarity holds.
  If confirmed, JointDilated+V2b cross-arch ensemble could outperform FiLM+V2b.

---

## 3. Key Questions This Study Answers

1. **Data scaling vs. architecture**: Does 2× data close the pure-DL → joint gap,
   or does expert CSP remain complementary regardless of scale?

2. **64/256-QAM wall**: Is the M₈₄ confusion reduced with more training examples?
   The 200k 4-model ensemble still shows 29.4% of 64-QAM misclassified as 256-QAM.
   Expected: 400k reduces this by 5–10pp if the limit is statistical, not theoretical.

3. **FiLM vs. late concatenation at scale**: FiLM's hypothesis (conditioning signal
   branch on CSP residuals) may become more valuable with more data — more examples
   let the network learn sharper residual boundaries.

4. **Integration gain ceiling**: Does the √N improvement in cumulant estimation
   continue above K=32768? (Only relevant if we later test K=65536.)

5. **JointCSPDilated (new)**: Does the multi-scale DilatedCNN backbone (384-d, 20k
   params) produce less redundant features with CSP than ResNet1D's GAP (256-d, 440k
   params)? Expected test if complementarity holds: JointDilated ≥ JointV2b at 37%
   fewer parameters.

---

## 4. Per-Class Analysis

*(To be filled when arch search completes)*

```
  Class       200k recall  400k recall  Δ     Interpretation
  ────────────────────────────────────────────────────────────────
  BPSK         100%          TBD         TBD
  MSK          100%          TBD         TBD
  4PSK          94.9%         TBD         TBD
  8PSK          90.0%         TBD         TBD
  π/4-DQPSK    92.6%         TBD         TBD   expect +5pp (dip averaging)
  16QAM         89.2%         TBD         TBD
  64QAM         62.2%         TBD         TBD   expect +5-10pp (M₈₄ wall)
  256QAM        71.8%         TBD         TBD   expect +5-10pp (M₈₄ wall)
  ────────────────────────────────────────────────────────────────
```

---

*Document created 2026-05-30.*
*Generation COMPLETE 13:03 (98GB, 400k signals). Phase transition COMPLETE 14:47 EDT.*
*CSP feature cache precomputed: /tmp/csp_400k_features.npy (164MB, 400k×107, 100% finite).*
*pure_dl 200k COMPLETE 2026-05-31: ResNet1D-s0=85.37%, s42=86.22%, DilatedCNN=85.64%, MultiScale=85.60%, PatchTransformer=76.43%.*
*PURE_DL_DONE emitted 06:36 EDT. train_film_200k.py (FiLM-s0 first) AND train_arch_400k.py both launched 06:37 EDT (concurrent).*
*07:58 EDT: 400k arch search killed — GPU starvation from page-cache competition (FiLM's 200k warm cache monopolized GPU; 400k 0 epochs in 80 min). Sequential restart bridge /tmp/launch_400k_after_film.sh armed — will restart 400k once FILM_200K_DONE. FiLM now has sole GPU: 78% util, 3621 MiB used.*
*FiLM-s0 trajectory: ep10 val=83.13%, ep20 val=88.63% best=88.90%, ep30 val=88.59% best=88.90% (flat basin).*
*Timing: ep1-10=7.4min/ep (cold cache), ep11-20=4.5min/ep (warming), ep21-30=2.3min/ep (sole GPU + hot cache).*
*FiLM-s0 est. done ~11:05 EDT May 31. All 4 FiLM configs done ~20:20 EDT May 31.*
*400k arch search (9 configs: 3 pure DL + 6 joint) restarts after FILM_200K_DONE, est. ~20:20 EDT May 31.*
*At 3.5-4 min/epoch × 80 ep × 9 configs ≈ 42-48h → results by ~Jun 2-3.*
