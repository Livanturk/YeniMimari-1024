# 8-bit BI-RADS Ablation Study — Master Task Tracker

> Last updated: 2026-04-12

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
- 8-bit vs 16-bit gap remains 6.2pp (0.6615 vs 0.7233) — Lesson #19

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

## D-Series: Loss Simplification & Component Isolation (PLANNED)

> Baseline: **C6** (CE + SWA + no asymmetry, test F1=0.6762, gap=4.21pp)
> Principle: Lesson #29 — Simplification > Regularization

### Tier 1: Auxiliary Head Ablation Cascade (baseline: C6)

#### D1: No Subgroup Heads (~17h)
- [x] Create config: `convnextv2_large_8bit_ablation_d1.yaml`
- [x] Create report stub: `experiments/convnextv2_large_8bit_ablation_d1/report.md`
- [ ] Run experiment: `python train.py --config configs/experiment_v2_birads/convnextv2_large_8bit_ablation_d1.yaml`
- [ ] Analyze results — subgroup loss (0.45 wt) is noise or useful signal?

#### D2: No Binary Head (~17h)
- [x] Create config: `convnextv2_large_8bit_ablation_d2.yaml`
- [x] Create report stub: `experiments/convnextv2_large_8bit_ablation_d2/report.md`
- [ ] Run experiment: `python train.py --config configs/experiment_v2_birads/convnextv2_large_8bit_ablation_d2.yaml`
- [ ] Analyze results — backbone gradient shortcut helpful or harmful?

#### D3: Pure L_full Only (~17h)
- [x] Create config: `convnextv2_large_8bit_ablation_d3.yaml`
- [x] Create report stub: `experiments/convnextv2_large_8bit_ablation_d3/report.md`
- [ ] Run experiment: `python train.py --config configs/experiment_v2_birads/convnextv2_large_8bit_ablation_d3.yaml`
- [ ] Analyze results — maximum Occam's razor: single objective optimal?

### Tier 2: Component Isolation (baseline: C6)

#### D4: No SWA — SWA Isolation Test (~17h)
- [x] Create config: `convnextv2_large_8bit_ablation_d4.yaml`
- [x] Create report stub: `experiments/convnextv2_large_8bit_ablation_d4/report.md`
- [ ] Run experiment: `python train.py --config configs/experiment_v2_birads/convnextv2_large_8bit_ablation_d4.yaml`
- [ ] Analyze results — does clean loss alone beat B1 (0.6387)?

### Tier 3: DINOv2 Rescue (baseline: C2)

#### D5: DINOv2 + Focal + SWA + No Asymmetry (~8h)
- [x] Create config: `dinov2_vitl_8bit_ablation_d5.yaml`
- [x] Create report stub: `experiments/dinov2_vitl_8bit_ablation_d5/report.md`
- [ ] Run experiment: `python train.py --config configs/experiment_v2_birads/dinov2_vitl_8bit_ablation_d5.yaml`
- [ ] Analyze results — does SWA + clean loss rescue DINOv2? Target: >= 0.65

### Tier 4: SWA + Mixup Disambiguation (baseline: C6)

#### D6: Mixup Only + No SWA + No Asymmetry (~17h)
- [x] Create config: `convnextv2_large_8bit_ablation_d6.yaml`
- [x] Create report stub: `experiments/convnextv2_large_8bit_ablation_d6/report.md`
- [ ] Run experiment: `python train.py --config configs/experiment_v2_birads/convnextv2_large_8bit_ablation_d6.yaml`
- [ ] Analyze results — Mixup on clean loss vs B3 (0.6459, dirty loss)

#### D7: SWA + Mixup + No Asymmetry — C1 Retest (~17h)
- [x] Create config: `convnextv2_large_8bit_ablation_d7.yaml`
- [x] Create report stub: `experiments/convnextv2_large_8bit_ablation_d7/report.md`
- [ ] Run experiment: `python train.py --config configs/experiment_v2_birads/convnextv2_large_8bit_ablation_d7.yaml`
- [ ] Analyze results — was C1's SWA+Mixup antagonism caused by asymmetry noise?

### D-Series Post-Analysis
- [ ] Update EXPERIMENTS.md with all D-series results
- [ ] Update tasks/lessons.md with D-series findings
- [ ] Decision: If D1 or D3 > C6 → further loss simplification for E-series
- [ ] Decision: If D4 ~ B1 → SWA is essential; if D4 >> B1 → clean loss alone valuable
- [ ] Decision: If D5 >= 0.65 → ensemble C6/D-winner + D5 for E-series
- [ ] Decision: If D7 > C6 → SWA+Mixup work on clean loss (Lesson #23 was confounded)
- [ ] Decision: Best D-series winner → E-series baseline

### Recommended Run Order
1. **D5** (DINOv2, ~8h) — fastest, independent backbone
2. **D1** (No subgroup, ~17h) — highest-weight auxiliary head test
3. **D3** (Pure L_full, ~17h) — maximum simplification
4. **D4** (No SWA, ~17h) — critical isolation test
5. **D7** (SWA+Mixup retest, ~17h) — C1 disambiguation
6. **D2** (No binary, ~17h) — small-weight head, likely mild effect
7. **D6** (Mixup only, ~17h) — alternative regularizer path

**Parallel strategy:** Run D5 on GPU-0 first (~8h). Then D1+D3 or D4+D7 as pairs.

---

## Reference: Key Metrics

| Experiment | Best Val F1 | Test F1 | Gap | BR1 | BR2 | BR4 | BR5 |
|-----------|:-----------:|:-------:|:---:|:---:|:---:|:---:|:---:|
| 16-bit ref | 0.6867 | **0.7233** | -3.7pp | — | — | — | — |
| **C6 (best 8-bit)** | **0.7183** | **0.6762** | **4.21pp** | **0.531** | **0.798** | 0.518 | 0.857 |
| B5 (prev best) | 0.7286 | 0.6615 | 6.71pp | 0.453 | 0.798 | 0.547 | 0.848 |
| B3 (Mixup) | 0.7193 | 0.6459 | 7.34pp | 0.513 | 0.732 | 0.489 | 0.850 |
| B1 (baseline) | 0.7334 | 0.6387 | 9.47pp | 0.526 | 0.675 | 0.498 | 0.856 |

**Progress:** 16-bit gap narrowed from 6.18pp (B5) to **4.71pp** (C6). Key insight: removing asymmetry loss was more effective than any added regularization.

## D-Series Experiment Design Overview

| Experiment | Strategy | Variable vs C6 | Hypothesis |
|---|---|---|---|
| D1 | Loss simplification | use_subgroup_head=false | Subgroup loss (0.45 wt) may be noise like asymmetry |
| D2 | Loss simplification | use_binary_head=false | Binary gradient shortcut — helpful or harmful? |
| D3 | Loss simplification | No subgroup + no binary | Maximum Occam's razor: single L_full objective |
| D4 | SWA isolation | use_swa=false | Is C6's gain from clean loss alone, or SWA synergy? |
| D5 | DINOv2 rescue | +SWA, -asymmetry (from C2) | Apply winning insights to DINOv2 → ensemble path |
| D6 | Mixup alternative | +Mixup, -SWA | Mixup on clean loss vs SWA on clean loss |
| D7 | C1 disambiguation | +Mixup (SWA kept) | Was SWA+Mixup antagonism from asymmetry noise? |

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
| C8 | Explicit dropout | clf=0.7, proj=0.4 | *pending* | — | Low expectations given C-series pattern |
