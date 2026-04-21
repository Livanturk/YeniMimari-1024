# 8-bit BI-RADS Ablation Study — Master Task Tracker

> Last updated: 2026-04-15

---

## A-Series: Initial 8-bit Ablation (COMPLETED)

- [x] A1: ConvNeXtV2-L + Focal (buggy norms) — val 0.7108, test 0.6270
- [x] A1-CE: ConvNeXtV2-L + CE (buggy norms) — val 0.7141, test 0.6370
- [x] A3: DINOv2 ViT-L + Focal (buggy norms) — val 0.6940, test 0.6325
- [x] Finding: CE > Focal for ConvNeXtV2 (+1.0pp), but DINOv2 needs focal (Lesson #14)

## B-Series: Bug Fixes & Individual Techniques (COMPLETED)

- [x] B1: ConvNeXtV2-L + CE + bug fixes — val 0.7334, test 0.6387 (+0.17pp vs A1-CE)
- [x] B2: DINOv2 + CE + bug fixes — val 0.6940, test 0.6136 (REGRESSED -1.9pp, CE hurts DINOv2)
- [x] B3: B1 + Mixup/CutMix — val 0.7193, test 0.6459 (best regularizer, gap 7.3pp)
- [x] B4: B1 + CORAL Ordinal — val 0.4449, FAILED (permanently abandoned, Lesson #16)
- [x] B5: B1 + SWA — val 0.7286, **test 0.6615** (best 8-bit result, gap 6.7pp)
- [x] Update tasks/lessons.md with B-series findings (Lessons #13-#20)
- [x] Update EXPERIMENTS.md with B-series results

### B-Series Key Lessons
- Bug fixes increased overfitting (gap 7.7pp→9.5pp) — Lesson #13
- DINOv2 needs focal loss, not CE — Lesson #14
- Mixup/CutMix best gap regularizer — Lesson #15
- CORAL ordinal dead for non-contiguous BI-RADS — Lesson #16
- SWA best absolute test F1 but BR1 regressed -7.3pp — Lesson #17


## C-Series: Combinations & Targeted Regularization (PLANNED)

### Tier 1: Combination Ablations (from Lesson #20)

#### C1: ConvNeXt + SWA + Mixup/CutMix (~17h)
- [x] Create config: `convnextv2_large_8bit_ablation_c1.yaml` (baseline: B5)
- [x] Create report stub: `experiments/convnextv2_large_8bit_ablation_c1/report.md`
- [x] Run experiment — val 0.7158, test **0.6431** (gap 7.27pp)
- [x] Analyze: SWA+Mixup antagonistic, WORSE than both alone (Lesson #23)

#### C2: DINOv2 + Focal + Bug Fixes (~8h)
- [x] Create config: `dinov2_vitl_8bit_ablation_c2.yaml` (baseline: A3)
- [x] Create report stub: `experiments/dinov2_vitl_8bit_ablation_c2/report.md`
- [x] Run experiment — val 0.6905, test **0.6240** (gap 6.65pp)
- [x] Analyze: bug fixes hurt DINOv2 too, -0.85pp vs A3 (Lesson #24)

#### C3: ConvNeXt + SWA + BR1 Weight 1.80 (~17h)
- [x] Create config: `convnextv2_large_8bit_ablation_c3.yaml` (baseline: B5)
- [x] Create report stub: `experiments/convnextv2_large_8bit_ablation_c3/report.md`
- [x] Run experiment — val 0.7105, test **0.6346** (gap 7.59pp)
- [x] Analyze: BR1 +1.2pp but BR2 -9.8pp, zero-sum (Lesson #27)

### Tier 2: Regularization Angle Ablations (all baseline: B5)

#### C4: Capacity Reduction — ConvNeXtV2-Base + SWA (~12h)
- [x] Create config: `convnextv2_large_8bit_ablation_c4.yaml` (backbone: Base ~89M)
- [x] Create report stub: `experiments/convnextv2_large_8bit_ablation_c4/report.md`
- [x] Run experiment — val 0.7178, test **0.6269** (gap 9.09pp)
- [x] Analyze: capacity reduction INCREASES gap, larger model better (Lesson #25)

#### C5: Feature Preservation — Backbone LR 0.05 + SWA (~17h)
- [x] Create config: `convnextv2_large_8bit_ablation_c5.yaml` (lr_scale: 0.2→0.05)
- [x] Create report stub: `experiments/convnextv2_large_8bit_ablation_c5/report.md`
- [x] Run experiment — val 0.7114, test **0.6284** (gap 8.30pp)
- [x] Analyze: backbone under-adapted, BR2 -10pp (Lesson #26)

#### C6: Asymmetry Loss Ablation — Weight 0.0 + SWA (~17h) — NEW 8-BIT CHAMPION
- [x] Create config: `convnextv2_large_8bit_ablation_c6.yaml` (asym_weight: 0.10→0.0)
- [x] Create report stub: `experiments/convnextv2_large_8bit_ablation_c6/report.md`
- [x] Run experiment — val 0.7183, test **0.6762** (gap 4.21pp) — BEST 8-BIT RESULT
- [x] Analyze: asymmetry loss was HURTING generalization, removal = +1.47pp (Lesson #22)

#### C7: Focal Loss on ConvNeXtV2 + SWA (~17h)
- [x] Create config: `convnextv2_large_8bit_ablation_c7.yaml` (loss: CE→Focal gamma=2.0)
- [x] Create report stub: `experiments/convnextv2_large_8bit_ablation_c7/report.md`
- [x] Run experiment — val 0.7261, test **0.6468** (gap 7.93pp)
- [x] Analyze: focal still harmful for ConvNeXtV2 even with SWA (Lesson #28)

#### C8: Extreme Dropout + SWA (~17h)
- [x] Create config: `convnextv2_large_8bit_ablation_c8.yaml` (clf_drop: 0.5→0.7, proj_drop: 0.2→0.4)
- [x] Create report stub: `experiments/convnextv2_large_8bit_ablation_c8/report.md`
- [ ] Run experiment: `python train.py --config configs/experiment_v2_birads/convnextv2_large_8bit_ablation_c8.yaml`
- [ ] Analyze results — given 6/7 "add regularization" failed, low expectations

### C-Series Post-Analysis (C1-C7 Complete)
- [x] Update EXPERIMENTS.md with C1-C7 results
- [x] Update tasks/lessons.md with C-series findings (Lessons #22-#29)
- [x] Decision: C1 = 0.6431 < 0.68 → SWA+Mixup antagonistic, D-series uses C6 as base instead
- [x] Decision: C1 BR1 = 0.458 < 0.50 → C3 ran but also failed (BR2 collapse)
- [x] Decision: C2 = 0.6240 < 0.65 → DINOv2 track stalled, bugs hurt DINOv2 too
- [x] Decision: **C6 is the clear winner** (0.6762, gap 4.21pp) → D-series baseline
- [ ] Run C8 (Extreme Dropout) for completeness — low expectations
- [x] Plan D-series → designed and configs created (D1-D7)

---

## D-Series: Loss Simplification & Component Isolation (COMPLETED)

> Baseline: **C6** (CE + SWA + no asymmetry, test F1=0.6762, gap=4.21pp)
> Principle: Lesson #29 — Simplification > Regularization
> **Result: ALL 7 experiments below C6. C6 confirmed as Goldilocks configuration.**

### Tier 1: Auxiliary Head Ablation Cascade (baseline: C6)

#### D1: No Subgroup Heads (~17h)
- [x] Create config: `convnextv2_large_8bit_ablation_d1.yaml`
- [x] Create report stub: `experiments/convnextv2_large_8bit_ablation_d1/report.md`
- [x] Run experiment — val 0.7138, test **0.6563** (gap 5.75pp)
- [x] Analyze: subgroup head IS helpful, -1.99pp vs C6. SWA LOST. (Lesson #30)

#### D2: No Binary Head (~17h)
- [x] Create config: `convnextv2_large_8bit_ablation_d2.yaml`
- [x] Create report stub: `experiments/convnextv2_large_8bit_ablation_d2/report.md`
- [x] Run experiment — val 0.7147, test **0.6453** (gap 6.94pp)
- [x] Analyze: binary head CRITICAL, -3.09pp vs C6 (3x above weight). SWA LOST. (Lesson #31)

#### D3: Pure L_full Only (~17h)
- [x] Create config: `convnextv2_large_8bit_ablation_d3.yaml`
- [x] Create report stub: `experiments/convnextv2_large_8bit_ablation_d3/report.md`
- [x] Run experiment — val 0.7237, test **0.6476** (gap 7.61pp)
- [x] Analyze: multi-task learning essential, -2.86pp vs C6. SWA WON. (Lesson #30)

### Tier 2: Component Isolation (baseline: C6)

#### D4: No SWA — SWA Isolation Test (~17h)
- [x] Create config: `convnextv2_large_8bit_ablation_d4.yaml`
- [x] Create report stub: `experiments/convnextv2_large_8bit_ablation_d4/report.md`
- [x] Run experiment — val 0.7109, test **0.6615** (gap 4.94pp)
- [x] Analyze: SWA worth +1.47pp. D4=B5=0.6615 exactly — clean loss = dirty+SWA. (Lesson #32)

### Tier 3: DINOv2 Rescue (baseline: C2)

#### D5: DINOv2 + Focal + SWA + No Asymmetry (~8h)
- [x] Create config: `dinov2_vitl_8bit_ablation_d5.yaml`
- [x] Create report stub: `experiments/dinov2_vitl_8bit_ablation_d5/report.md`
- [x] Run experiment — val 0.6925, test **0.6383** (gap 5.42pp)
- [x] Analyze: +1.43pp vs C2 but below 0.65 target. DINOv2 track abandoned. (Lesson #33)

### Tier 4: SWA + Mixup Disambiguation (baseline: C6)

#### D6: Mixup Only + No SWA + No Asymmetry (~17h)
- [x] Create config: `convnextv2_large_8bit_ablation_d6.yaml`
- [x] Create report stub: `experiments/convnextv2_large_8bit_ablation_d6/report.md`
- [x] Run experiment — val 0.7249, test **0.6353** (gap 8.96pp) — worst gap in D-series
- [x] Analyze: Mixup -2.62pp vs D4 (no Mixup). Mixup HARMFUL on clean loss. (Lesson #34)

#### D7: SWA + Mixup + No Asymmetry — C1 Retest (~17h)
- [x] Create config: `convnextv2_large_8bit_ablation_d7.yaml`
- [x] Create report stub: `experiments/convnextv2_large_8bit_ablation_d7/report.md`
- [x] Run experiment — val 0.7159, test **0.6563** (gap 5.96pp)
- [x] Analyze: SWA+Mixup antagonism intrinsic, NOT from asymmetry noise. (Lesson #35)

### D-Series Post-Analysis (COMPLETED)
- [x] Update EXPERIMENTS.md with all D-series results
- [x] Update tasks/lessons.md with D-series findings (Lessons #30-#37)
- [x] Decision: D1 (0.6563) & D3 (0.6476) < C6 → ❌ no further loss simplification
- [x] Decision: D4 (0.6615) >> B1 (0.6387) → ✅ clean loss alone is valuable (+2.28pp)
- [x] Decision: D4 (0.6615) < C6 (0.6762) → ✅ SWA is essential (+1.47pp on top)
- [x] Decision: D5 (0.6383) < 0.65 → ❌ ensemble path abandoned
- [x] Decision: D7 (0.6563) < C6 → ❌ SWA+Mixup antagonism NOT confounded
- [x] Decision: No D-series winner beats C6 → **C6 remains E-series baseline**

---

## E-Series: SWA Optimization & Pipeline Transfer (PLANNED)

> Baseline: **C6** (CE + SWA + no asymmetry, test F1=0.6762, gap=4.21pp)
> Principle: Lesson #37 — C6 is Goldilocks; now optimize SWA trajectory and fine-tune hyperparams
> Key insight: OneCycleLR is suboptimal for SWA (LR still climbing during averaging phase)

### Tier 1: LR Schedule Optimization (baseline: C6)

#### E1: Cosine Warmup LR (~17h)
- [x] Create config: `convnextv2_large_8bit_ablation_e1.yaml`
- [x] Create report stub: `experiments/convnextv2_large_8bit_ablation_e1/report.md`
- [ ] Run experiment: `python train.py --config configs/experiment_v2_birads/convnextv2_large_8bit_ablation_e1.yaml`
- [ ] Analyze results — smooth LR decay during SWA vs OneCycleLR's climbing LR

#### E2: Cosine Warm Restarts LR (~17h)
- [x] Create config: `convnextv2_large_8bit_ablation_e2.yaml`
- [x] Create report stub: `experiments/convnextv2_large_8bit_ablation_e2/report.md`
- [ ] Run experiment: `python train.py --config configs/experiment_v2_birads/convnextv2_large_8bit_ablation_e2.yaml`
- [ ] Analyze results — SWA literature's optimal schedule (diverse weight snapshots)

### Tier 2: SWA Parameter Tuning (baseline: C6)

#### E3: Later SWA + Longer Training (~17h+)
- [x] Create config: `convnextv2_large_8bit_ablation_e3.yaml`
- [x] Create report stub: `experiments/convnextv2_large_8bit_ablation_e3/report.md`
- [ ] Run experiment: `python train.py --config configs/experiment_v2_birads/convnextv2_large_8bit_ablation_e3.yaml`
- [ ] Analyze results — swa_start=10 + patience=30 → more converged SWA averaging

### Tier 3: Regularization Fine-Tuning (baseline: C6)

#### E4: Higher Label Smoothing (~17h)
- [x] Create config: `convnextv2_large_8bit_ablation_e4.yaml`
- [x] Create report stub: `experiments/convnextv2_large_8bit_ablation_e4/report.md`
- [ ] Run experiment: `python train.py --config configs/experiment_v2_birads/convnextv2_large_8bit_ablation_e4.yaml`
- [ ] Analyze results — label_smoothing 0.05→0.10 closes remaining 4.21pp gap?

#### E5: Boosted Binary Head Weight (~17h)
- [x] Create config: `convnextv2_large_8bit_ablation_e5.yaml`
- [x] Create report stub: `experiments/convnextv2_large_8bit_ablation_e5/report.md`
- [ ] Run experiment: `python train.py --config configs/experiment_v2_birads/convnextv2_large_8bit_ablation_e5.yaml`
- [ ] Analyze results — binary=0.20, subgroup=0.35 (Lesson #31 motivated)

### Tier 4: Spatial Augmentation (baseline: C6)

#### E6: Stronger Spatial Augmentation (~17h)
- [x] Create config: `convnextv2_large_8bit_ablation_e6.yaml`
- [x] Create report stub: `experiments/convnextv2_large_8bit_ablation_e6/report.md`
- [ ] Run experiment: `python train.py --config configs/experiment_v2_birads/convnextv2_large_8bit_ablation_e6.yaml`
- [ ] Analyze results — rotation=20, erasing=0.2 (spatial diversity without label corruption)

### Tier 5: Cross-Pipeline Transfer (baseline: 16-bit v1)

#### E7: 16-bit + SWA + No Asymmetry (~10h)
- [x] Create config: `convnextv2_large_16bit_ablation_e7.yaml`
- [x] Create report stub: `experiments/convnextv2_large_16bit_ablation_e7/report.md`
- [ ] Run experiment: `python train.py --config configs/experiment_v2_birads/convnextv2_large_16bit_ablation_e7.yaml`
- [ ] Analyze results — C6 insights (no asym + SWA) on 16-bit. Target: >= 0.74

### E-Series Post-Analysis
- [ ] Update EXPERIMENTS.md with all E-series results
- [ ] Update tasks/lessons.md with E-series findings
- [ ] Decision: If E1 or E2 > C6 → best LR schedule for F-series
- [ ] Decision: If E5 > C6 → binary head weight matters, tune further
- [ ] Decision: If E7 > 0.74 → 16-bit pipeline is the production path
- [ ] Decision: Best E-series winner → F-series baseline (or combine best changes)

### Recommended Run Order
1. **E7** (16-bit, ~10h) — fastest, independent pipeline, highest potential impact
2. **E1** (cosine_warmup, ~17h) — highest priority LR schedule test
3. **E2** (warm_restarts, ~17h) — SWA-optimal schedule test
4. **E5** (binary weight, ~17h) — lesson-motivated tuning
5. **E3** (later SWA, ~17h+) — SWA timing test (may run longer due to patience=30)
6. **E4** (label smoothing, ~17h) — regularization tuning
7. **E6** (augmentation, ~17h) — spatial diversity test

**Parallel strategy:** Run E7 first (~10h). Then E1+E2 as a pair (~17h). Then E5+E3.

### Bonus: TTA on Existing C6 Model (FREE — no training) — SUPERSEDED by Tier 1 pipeline below
- [x] Fix normalization bug in `ensemble_evaluate.py` (cosmetic — train.py eval was bug-free, see Lesson #44)
- [~] Old `ensemble_evaluate.py --tta` route abandoned: MODELS list is outdated and TTA lacks view-swap. Replaced by Tier 1 pipeline.

---

## Tier 1: Inference-Time Pipeline on C6 (IN PROGRESS, 2026-04-18)

Baseline (post-fix): test F1 = **0.6762**, val F1 = 0.7218, gap = 4.55pp, ECE test = 0.214.
Cache: `artifacts/c6_{val,test}_*.npy` (all 4 head logits + labels). Tier 1 tasks read from this cache — no further forward passes.

- [x] Task 0.1 bug fix + logit extraction (Lesson #44)
- [x] Task 1.1 TTA — tta8=0.6808 (+0.45pp), peak tta6=0.6847 (+0.84pp). hflip_swap hurts; rotations win. Lesson #45.
- [x] Task 1.2 Temperature scaling — T_opt ≈ 0.73 (underconfident, not overconfident). ECE floor ~0.13 (scalar-T limit). Lesson #46.
- [x] Task 1.3 Threshold offsets — NEGATIVE RESULT. Val +1.43pp but test −0.53pp (nonTTA) / −0.18pp (tta8). Zero-sum BR2↔BR4. Lesson #47. Excluded from default pipeline; ablation only.
- [x] Task 1.4 Gating — NEUTRAL. Hard gate +0.03pp (nonTTA) / 0.00pp (tta8). Soft α-CV bimodal (tta8 std=0.49 guardrail broken). Pure hier < pure full. Hier duplicates full-head info. Lesson #48.
- [ ] Task 1.5 Cumulative pipeline eval + ablation table + decision point — script: `tools/cumulative_eval_c6.py`

### Tier 2 — Routing (decision depends on Task 1.5)

Expected Task 1.5 cumulative: nonTTA ≈ 0.6765, tta8 ≈ 0.6807 → both < 0.70 → root-cause path.
Root cause (Lessons #47, #48 combined): val→test prior shift + full-head-already-sub-head-informed.

- [x] Task 2.2 F2 — Logit-adjusted training. ALL τ REGRESSED (−2.9 to −3.7pp vs C6). BR2 sacrificed on BR2↔BR4 zero-sum axis (third manifestation after Lessons #27, #47). Val-test gap WIDENED (+3pp). Lesson #49. **F2 PERMANENTLY ABANDONED.**
- [ ] Task 2.0 Multi-seed ensemble — seed=123, seed=2024 configs written. 2×17h GPU (paralel). 3-seed tta8 blend for final F1.
- [~] Task 2.1 F1 16-bit — defer (300GB preprocessing; only if multi-seed fails too).

---

## F-Series: 16-bit Pipeline Transfer (PLANNED)

> Baseline: **C6** (8-bit champion, test F1=0.6762, gap=4.21pp)
> Dataset: **Dataset_1024_16bit** (7,557 hasta) / **Dataset_Test_1024_16bit** (1,655 hasta)
> Normalizasyon: DATASET_STATS_16BIT — mean=0.1220, std=0.2044 (all-pixel, transforms.py)
> Principle: 8-bit'in en iyi 4 config'ini 16-bit'e taşıyarak bit derinliğinin etkisini izole et.
>
> **Deney Tasarımı:**
> F1 = C6 champion transfer (baseline). F1 vs F2 = asymmetry etkisi. F1 vs F3 = SWA etkisi. F1 vs F4 = SWA+Mixup antagonizmi.
> Cross-pipeline karşılaştırma: her F deneyi ↔ 8-bit karşılığı → 16-bit kazancını ölç.

### F1: C6 Champion Transfer (CE + SWA + No Asymmetry) (~17h)
- [x] Create config: `convnextv2_large_16bit_ablation_f1.yaml`
- [ ] Run experiment: `python train.py --config configs/experiment_v2_birads/convnextv2_large_16bit_ablation_f1.yaml`
- [ ] Analyze results — F1 vs C6 (0.6762): 16-bit kazancı ne kadar? Target: >= 0.69

### F2: Asymmetry Retest (CE + SWA + Asymmetry=0.10) (~17h)
- [x] Create config: `convnextv2_large_16bit_ablation_f2.yaml`
- [ ] Run experiment: `python train.py --config configs/experiment_v2_birads/convnextv2_large_16bit_ablation_f2.yaml`
- [ ] Analyze results — F1 vs F2: asymmetry loss 16-bit'te de zararlı mı?

### F3: SWA Isolation (CE + No SWA + No Asymmetry) (~17h)
- [x] Create config: `convnextv2_large_16bit_ablation_f3.yaml`
- [ ] Run experiment: `python train.py --config configs/experiment_v2_birads/convnextv2_large_16bit_ablation_f3.yaml`
- [ ] Analyze results — F1 vs F3: SWA 16-bit'te ne kadar katkı sağlıyor?

### F4: SWA+Mixup Antagonism Retest (CE + SWA + Mixup/CutMix) (~17h)
- [x] Create config: `convnextv2_large_16bit_ablation_f4.yaml`
- [ ] Run experiment: `python train.py --config configs/experiment_v2_birads/convnextv2_large_16bit_ablation_f4.yaml`
- [ ] Analyze results — F1 vs F4: SWA+Mixup antagonizmi 16-bit'te devam ediyor mu?

### F-Series Post-Analysis
- [ ] Update EXPERIMENTS.md with all F-series results
- [ ] Update tasks/lessons.md with F-series findings
- [ ] Decision: F1 > C6 (0.6762)? → 16-bit pipeline 8-bit'ten üstün
- [ ] Decision: F1 > F2? → asymmetry bit-depth bağımsız zararlı (Lesson #22 evrensel)
- [ ] Decision: F1 > F3? → SWA 16-bit'te de essential
- [ ] Decision: F1 > F4? → SWA+Mixup antagonizmi evrensel (Lesson #35 evrensel)
- [ ] Decision: Best F-series winner → production pipeline seçimi (8-bit vs 16-bit)

### Recommended Run Order
1. **F1** (~17h) — Champion transfer, ana sonuç: 16-bit baseline
2. **F3** (~17h) — SWA isolation, F1 ile birlikte 2×2 faktöriyel tamamlanır
3. **F2** (~17h) — Asymmetry retest
4. **F4** (~17h) — SWA+Mixup retest (en düşük öncelik, negatif sonuç bekleniyor)

**Parallel strategy:** F1+F3 birlikte başlat (bağımsız, 2×2 faktöriyelin iki ayağı). Sonra F2+F4.

### F-Series Experiment Design Overview

| Exp | 8-bit Source | 8-bit Test F1 | Variable | 16-bit Question |
|-----|-------------|:-------------:|----------|-----------------|
| F1  | C6          | 0.6762        | Baseline transfer | 16-bit dynamic range ne kadar kazandırır? |
| F2  | B5          | 0.6615        | +asymmetry=0.10 | Asymmetry loss 16-bit'te de zararlı mı? |
| F3  | D4          | 0.6615        | -SWA | SWA 16-bit'te de essential mi? |
| F4  | D7          | 0.6563        | +Mixup/CutMix | SWA+Mixup antagonizmi 16-bit'te devam mı? |

### Expected 2×2 Factorial (F1/F3 + 8-bit C6/D4)

```
|             | 8-bit          | 16-bit        | Δ (16bit-8bit) |
|-------------|:--------------:|:-------------:|:--------------:|
| No SWA      | D4 = 0.6615    | F3 = ???      | ???            |
| SWA         | C6 = 0.6762    | F1 = ???      | ???            |
| Δ SWA       | +1.47pp        | ???           |                |
```

---

## Reference: Key Metrics

| Experiment | Best Val F1 | Test F1 | Gap | BR1 | BR2 | BR4 | BR5 |
|-----------|:-----------:|:-------:|:---:|:---:|:---:|:---:|:---:|
| **C6 (best 8-bit)** | **0.7183** | **0.6762** | **4.21pp** | **0.531** | **0.798** | 0.518 | 0.857 |
| D4 (clean, no SWA) | 0.7109 | 0.6615 | 4.94pp | 0.500 | 0.757 | 0.542 | 0.847 |
| B5 (prev best) | 0.7286 | 0.6615 | 6.71pp | 0.453 | 0.798 | 0.547 | 0.848 |
| B1 (baseline) | 0.7334 | 0.6387 | 9.47pp | 0.526 | 0.675 | 0.498 | 0.856 |

**Progress:** C6 (test F1=0.6762, gap=4.21pp) confirmed as optimal 8-bit configuration after D-series. All 7 D-series experiments scored below C6. Key finding: D4=B5=0.6615 exactly — clean loss alone matches dirty loss + SWA.

## E-Series Experiment Design Overview

| Experiment | Strategy | Variable vs C6 | Hypothesis |
|---|---|---|---|
| E1 | LR schedule | cosine_warmup (warmup=5, min_lr=1e-6) | Smooth LR decay during SWA -> better averaging |
| E2 | LR schedule | cosine_warm_restarts (T_0=10, T_mult=2) | Diverse SWA snapshots from LR restarts |
| E3 | SWA timing | swa_start=10, patience=30 | More converged weights before averaging |
| E4 | Regularization | label_smoothing=0.10 | Stronger label regularization closes gap |
| E5 | Loss weights | binary=0.20, subgroup=0.35 | Boost 7x-efficient binary gradient anchor |
| E6 | Augmentation | rotation=20, erasing=0.2 | Spatial diversity without label corruption |
| E7 | Pipeline transfer | 16-bit + SWA + no asymmetry | C6 insights on 16-bit (baseline 0.7233) |

## D-Series Results Overview

| Experiment | Strategy | Variable vs C6 | Test F1 | Δ vs C6 | Gap | Verdict |
|---|---|---|:---:|:---:|:---:|---|
| D4 | SWA isolation | use_swa=false | 0.6615 | -1.47pp | 4.94pp | SWA essential (+1.47pp) |
| D1 | Loss simplification | -subgroup_head | 0.6563 | -1.99pp | 5.75pp | Subgroup head helpful |
| D7 | C1 disambiguation | +Mixup (SWA kept) | 0.6563 | -1.99pp | 5.96pp | Antagonism intrinsic |
| D3 | Loss simplification | -subgroup -binary | 0.6476 | -2.86pp | 7.61pp | Multi-task essential |
| D2 | Loss simplification | -binary_head | 0.6453 | -3.09pp | 6.94pp | Binary head critical |
| D5 | DINOv2 rescue | +SWA, -asymmetry | 0.6383 | -3.79pp | 5.42pp | Architecture gap persists |
| D6 | Mixup alternative | +Mixup, -SWA | 0.6353 | -4.09pp | 8.96pp | Mixup harmful on clean loss |

## C-Series Results Overview

| Experiment | Angle | Variable | Test F1 | Gap | Verdict |
|---|---|---|:---:|:---:|---|
| **C6** | Loss pruning | asymmetry_weight=0.0 | **0.6762** | **4.21pp** | **NEW BEST — simplification wins** |
| C7 | Loss landscape | Focal + SWA | 0.6468 | 7.93pp | Focal still harmful for ConvNeXtV2 |
| C1 | Combination | SWA + Mixup/CutMix | 0.6431 | 7.27pp | Antagonistic, worse than both alone |
| C3 | Class balance | BR1 weight 1.80 | 0.6346 | 7.59pp | Zero-sum: BR1 +1pp, BR2 -10pp |
| C5 | Feature preservation | backbone_lr=0.05 | 0.6284 | 8.30pp | Under-adapted, BR2 collapsed |
| C4 | Capacity | Base backbone (~89M) | 0.6269 | 9.09pp | Larger model better in low-data |
| C2 | Backbone rescue | DINOv2 + Focal + fixes | 0.6240 | 6.65pp | Bug fixes hurt DINOv2 too |
| C8 | Explicit dropout | clf=0.7, proj=0.4 | *pending* | — | Low expectations given C/D-series pattern |
