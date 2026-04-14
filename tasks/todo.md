# 8-bit BI-RADS Ablation Study — Master Task Tracker

> Last updated: 2026-04-14

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

## Reference: Key Metrics

| Experiment | Best Val F1 | Test F1 | Gap | BR1 | BR2 | BR4 | BR5 |
|-----------|:-----------:|:-------:|:---:|:---:|:---:|:---:|:---:|
| **C6 (best 8-bit)** | **0.7183** | **0.6762** | **4.21pp** | **0.531** | **0.798** | 0.518 | 0.857 |
| D4 (clean, no SWA) | 0.7109 | 0.6615 | 4.94pp | 0.500 | 0.757 | 0.542 | 0.847 |
| B5 (prev best) | 0.7286 | 0.6615 | 6.71pp | 0.453 | 0.798 | 0.547 | 0.848 |
| B1 (baseline) | 0.7334 | 0.6387 | 9.47pp | 0.526 | 0.675 | 0.498 | 0.856 |

**Progress:** C6 (test F1=0.6762, gap=4.21pp) confirmed as optimal 8-bit configuration after D-series. All 7 D-series experiments scored below C6. Key finding: D4=B5=0.6615 exactly — clean loss alone matches dirty loss + SWA.

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
