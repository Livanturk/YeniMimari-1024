# 8-bit BI-RADS Ablation Study — Master Task Tracker

> Last updated: 2026-04-11

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

## C-Series: Combinations & Targeted Fixes (PLANNED — ~42h total)

### C2: DINOv2 + Focal + Bug Fixes (~8h, run FIRST)
- [x] Create config: `dinov2_vitl_8bit_ablation_c2.yaml` (baseline: A3)
- [x] Create report stub: `experiments/dinov2_vitl_8bit_ablation_c2/report.md`
- [ ] Run experiment
- [ ] Analyze results — does DINOv2 + focal + fixes beat A3 (0.6325)?

### C1: ConvNeXt + SWA + Mixup/CutMix (~17h, run SECOND)
- [x] Create config: `convnextv2_large_8bit_ablation_c1.yaml` (baseline: B5)
- [x] Create report stub: `experiments/convnextv2_large_8bit_ablation_c1/report.md`
- [ ] Run experiment
- [ ] Analyze results — do Mixup and SWA stack? Target: test F1 >= 0.68

### C3: ConvNeXt + SWA + BR1 Weight 1.80 (~17h, run THIRD)
- [x] Create config: `convnextv2_large_8bit_ablation_c3.yaml` (baseline: B5)
- [x] Create report stub: `experiments/convnextv2_large_8bit_ablation_c3/report.md`
- [ ] Run experiment
- [ ] Analyze results — does BR1 recover (target >= 0.50) without macro collapse?

### C-Series Post-Analysis
- [ ] Update EXPERIMENTS.md with C-series results
- [ ] Update tasks/lessons.md with C-series findings
- [ ] Decision: If C1 >= 0.68 → plan D-series ensemble (C1 + C2)
- [ ] Decision: If C1 BR1 >= 0.50 → C3 may be unnecessary
- [ ] Decision: If C2 >= 0.65 → DINOv2 track alive, consider C2 + SWA combo

---

## Reference: Key Metrics

| Experiment | Best Val F1 | Test F1 | Gap | BR1 | BR2 | BR4 | BR5 |
|-----------|:-----------:|:-------:|:---:|:---:|:---:|:---:|:---:|
| 16-bit ref | 0.6867 | **0.7233** | -3.7pp | — | — | — | — |
| B5 (best 8-bit) | 0.7286 | **0.6615** | 6.7pp | 0.453 | 0.798 | 0.547 | 0.848 |
| B3 (best gap) | 0.7193 | 0.6459 | 7.3pp | 0.513 | 0.732 | 0.489 | 0.850 |
| B1 (baseline) | 0.7334 | 0.6387 | 9.5pp | 0.526 | 0.675 | 0.498 | 0.856 |

**Target:** Close the 6.2pp gap to 16-bit (0.7233). C1 expected ~0.68-0.69.
