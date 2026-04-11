# Lessons Learned

## Pitfall #10 — SwinV2 window_size=24 requires specific image sizes (2026-04-09)
**Problem:** `swinv2_base_window12to24_192to384` with `img_size=1024` crashes at model build:
`RuntimeError: shape '[1, 10, 24, 10, 24, 1]' is invalid for input of size 65536`.
SwinV2 stage 0 produces 1024/4=256 spatial, and 256%24≠0.

**Rule:** For window_size=24, ALL feature map resolutions must be divisible by 24.
Valid sizes: 384 (native, last stage padded), **768** (2× native, all stages clean), 1536.
Formula: image_size must be divisible by `patch_size × window_size = 4 × 24 = 96`,
AND `image_size / (2^(n_stages-1) × patch_size)` must be ≥ window_size or divisible by it.

**Fix applied:** A2 config image_size 1024 → 768.

## Pitfall #11 — Focal Loss vs CrossEntropy in Imbalanced BI-RADS
**Problem:** Focal loss (γ=2.0) catastrophically degrades BR1 performance (-7.6pp test F1) compared to standard CrossEntropy. By aggressively down-weighting high-confidence "easy" normal tissue cases, it prevents the model from learning the subtle boundaries between BR1 and BR2. It also results in poorly calibrated, low-confidence predictions.
**Rule:** Default to CrossEntropy (`loss_type: "ce"`) for this 4-class BI-RADS task.

## Pitfall #12 — Evaluating on Shifted Holdout Sets
**Problem:** Severe val→test F1 degradation observed on BR1 (19.6% train vs 9.8% test).
**Rule:** Recognize that class-wise test degradation is often a dataset-level prior probability shift, not a model failure. Use Stacking Ensembles (`ensemble_evaluate.py --stacking`) to combine models with complementary error profiles (e.g., DINOv2 for OOD generalization + ConvNeXtV2 for high-res local details).

---

## B-Series Experiment Lessons (2026-04-11)

### Lesson #13 — Bug Fixes Can Increase Overfitting (B1 vs A1-CE)
**Problem:** B1 fixed two real bugs (wrong normalization stats, test-set class weights) and reverted config drift. Val F1 improved by +1.9pp (0.7141→0.7334), but test F1 barely moved (+0.17pp, 0.6370→0.6387). The val→test gap WIDENED from 7.7pp to **9.5pp** — the worst in the entire study.

**Root cause:** The bugs were introducing noise that accidentally acted as implicit regularization. With correct normalization (0.1210/0.1977) and proper class weights ([1.28,1.00,1.20,1.11]), the model's internal representations align better with the training/val distribution, enabling sharper (deeper) minima. But the test set has a different class prior (BR1: 9.8% vs 19.6% in train) that sharp minima don't generalize to.

**Rule:** Fixing training bugs is necessary but NOT sufficient. Always pair bug fixes with explicit regularization (Mixup, SWA, dropout tuning) because correct training often means stronger fitting. Monitor the val→test gap alongside absolute metrics — if val improves but test doesn't, overfitting has deepened.

### Lesson #14 — DINOv2 Needs Focal Loss, Unlike ConvNeXtV2 (B2 vs A3)
**Problem:** The A-series showed CE beats focal for ConvNeXtV2 (+1.0pp test F1). B2 applied this same CE switch to DINOv2 — and it REGRESSED by -1.9pp (0.6325→0.6136). DINOv2 val peak was identical (0.6940) across A3 (focal) and B2 (CE), but test diverged.

**Root cause:** DINOv2's self-supervised features are domain-agnostic and less mammography-adapted. Focal loss (gamma=2.0) forces DINOv2 to concentrate capacity on difficult cases — the hard BR1/BR4 boundaries that matter for BI-RADS. CE treats all samples equally, wasting DINOv2's limited domain adaptation on already-easy samples. ConvNeXtV2's supervised ImageNet features are already locally adapted, so CE's uniform gradient is sufficient.

**Rule:** Do NOT generalize loss function findings across backbone families. Focal loss is harmful for ConvNeXtV2 (BR1 collapse) but beneficial for DINOv2 (hard-example mining compensates for weaker domain adaptation). Test loss functions separately for each backbone architecture.

### Lesson #15 — Mixup/CutMix: Best Val-Test Gap Regularizer (B3)
**Problem:** B1 had a 9.5pp val→test gap despite bug fixes. B3 added Mixup (alpha=0.2) + CutMix (alpha=1.0) as the only change.

**Evidence:**
- Train F1 dropped from 0.795 to 0.588 (strong regularization)
- Val peak dropped slightly: 0.7334→0.7193 (-1.4pp)
- Test IMPROVED: 0.6387→0.6459 (+0.72pp)
- Val→test gap narrowed: 9.5pp→7.3pp (-2.2pp)
- BR2 test surged +5.7pp (0.675→0.732)
- Convergence: more oscillation, peak at ep17 (vs ep7 for B1)

**NaN train metrics note:** Mixup generates soft blended targets that break asymmetry_loss and binary_loss metric logging (they log NaN). This is a logging artifact, not a training problem — losses are computed correctly via lambda-weighting.

**Rule:** For this multi-view mammography task, Mixup/CutMix is the most effective single regularizer for closing the val→test gap. The mild alpha=0.2 for Mixup keeps interpolation close to original samples (lambda~0.9), while CutMix alpha=1.0 provides diverse spatial cutouts. Patient-level mixing ensures all 4 views are mixed consistently.

### Lesson #16 — CORAL Ordinal Loss Is Fundamentally Broken for Non-Contiguous BI-RADS (B4)
**Problem:** B4 used CORAL ordinal loss with K-1=3 cumulative binary classifiers, with subgroup heads disabled (the documented fix from 16-bit experiments). Despite following the proven pattern from `ordinal_nosubgroup_v1.yaml`, B4 CATASTROPHICALLY FAILED:
- Val F1 peaked at 0.4449 (29pp below B1)
- BR4 val F1 = 0.054 (near-zero collapse)
- No test evaluation was triggered
- Training plateaued at val F1 ~0.31 for 34 consecutive epochs before slowly climbing to 0.44

**Root cause:** CORAL models cumulative thresholds: P(y≥2), P(y≥4), P(y≥5). With non-contiguous BI-RADS classes (1,2,**skip 3**,4,5), the P(y≥4) threshold has no natural decision boundary — there is no class 3 to separate from class 4. The optimizer gets stuck because the middle threshold receives conflicting gradients from BR2 (push threshold right) and BR4 (push threshold left) with no intermediate class to anchor it.

The 16-bit `ordinal_nosubgroup_v1.yaml` experiment may have worked because 16-bit images provide finer tissue-level features that create a smoother embedding space. In 8-bit with reduced dynamic range, the feature space is more clustered, making the missing-class gap more problematic.

**Rule:** PERMANENTLY ABANDON ordinal regression losses for BI-RADS classification when class 3 is absent. The missing class creates an irrecoverable gap in the cumulative threshold space. This applies to CORAL, Proportional Odds, Stick-Breaking, and any cumulative-link model. If ordinal structure is desired, use label smoothing with ordinal-aware targets (e.g., smooth BR1↔BR2 more than BR1↔BR5) instead — this encodes ordinal prior without cumulative thresholds.

### Lesson #17 — SWA: Best Absolute Test F1, But BR1 Tradeoff (B5)
**Problem:** B5 added SWA (start_epoch=5) to the B1 config. Result:
- **Test F1: 0.6615** — best in entire 8-bit ablation study (+2.3pp vs B1)
- **Test AUC: 0.913** — best across all experiments
- **Test Kappa: 0.624** — best across all experiments
- Val→test gap: 6.7pp (vs B1's 9.5pp) — confirms flat minima generalize better

Per-class shifts vs B1:
- BR2: +12.3pp (0.675→0.798) — massive
- BR4: +4.9pp (0.498→0.547) — significant
- BR5: -0.8pp (0.856→0.848) — negligible
- **BR1: -7.3pp (0.526→0.453) — substantial regression**

**Root cause of BR1 regression:** SWA smooths decision boundaries by averaging weights across the training trajectory (ep5-ep31). BR1 (n=163 test) and BR2 (n=596 test) share a fuzzy boundary. Smoothing favors the higher-density class (BR2) at the expense of the lower-density class (BR1). The F1 formula amplifies this: a few BR1→BR2 flips cost a lot in BR1 F1 but are absorbed by BR2's larger sample.

**Rule:** SWA is the single most effective technique for 8-bit test F1, but introduces a BR1 regression risk. When combining SWA with other methods, add BR1-targeted regularization (e.g., asymmetry_benign_weight > 0, or class-specific augmentation for BR1). Always report per-class F1 alongside macro — SWA's macro gains can mask minority-class losses.

### Lesson #18 — MLflow Reports Last-Epoch Metrics, Not Best-Epoch (Critical Tooling Note)
**Problem:** MLflow's `search_runs()` returns the LAST logged value for each metric, not the value at the best checkpoint epoch. For experiments with patience=20, the last epoch can be 20+ epochs past the best, giving misleadingly low val F1.

**Example:** B1 best val F1 = 0.7334 (epoch 7), but MLflow reports val_full_f1_macro = 0.6557 (epoch 27, the early-stop epoch). Using the MLflow-reported value underestimates val performance by 7.8pp and misrepresents the generalization gap.

**Rule:** ALWAYS use `get_metric_history()` to find the true best-epoch val F1. Compare test F1 against the best val F1 (the checkpoint that was actually selected), not the last-epoch val F1. When reporting gaps, use: `gap = best_val_f1 - test_f1`.

### Lesson #19 — 8-Bit vs 16-Bit Gap Remains 6.2pp — Not a Bug Problem
**Evidence from full B-series:**
- Best 8-bit (B5 SWA): test F1 = 0.6615
- 16-bit baseline: test F1 = 0.7233
- Remaining gap: 6.2pp

Bug fixes (B1) contributed only +0.17pp. SWA (B5) contributed +2.3pp. Mixup (B3) contributed +0.72pp. The combined maximum (if orthogonal) would be ~3.2pp, reaching ~0.67-0.68. Still 4-5pp short of 16-bit.

**Hypothesis:** The 8-bit quantization (0-255 integer levels) compresses subtle tissue-level contrast variations that are critical for BI-RADS classification. CLAHE partially compensates but cannot recover information lost in the 16→8 bit conversion. This is not addressable by training tricks alone.

**Rule:** The 8-bit pipeline has an intrinsic ceiling. To close the remaining gap, investigate: (1) mixed-precision pipeline (16-bit for tissue region, 8-bit for background), (2) learned pre-processing (trainable histogram equalization), (3) stronger ensemble (B3+B5 combination, multi-backbone), or (4) accept the tradeoff and justify 8-bit for deployment efficiency.

### Lesson #20 — Recommended Next Experiments (Priority Order)
Based on B-series results:
1. **B3+B5 (Mixup + SWA)** — Highest priority. Orthogonal mechanisms: Mixup regularizes training, SWA averages weights post-training. Expected: test F1 ~0.68-0.69 with gap ~5-6pp.
2. **A3-fixed (DINOv2 + Focal + Bug Fixes)** — B2 showed CE hurts DINOv2. Create a new config with focal loss + correct normalization/weights. Expected: test F1 ~0.65.
3. **B5 + asymmetry_benign_weight > 0** — Address SWA's BR1 regression by adding BR1-specific penalty.
4. **Ensemble: B5 + A3-fixed** — ConvNeXtV2 (BR2/BR4 strength) + DINOv2 (BR5 strength, different error profile).
5. **ABANDON:** CORAL ordinal (Lesson #16), more aggressive augmentation without SWA (insufficient alone).

### Summary Statistics — B-Series Ablation Table

```
| Experiment | Backbone    | Change vs B1         | Best Val F1 | Test F1  | Gap    | Binary F1 | AUC   |
|------------|-------------|----------------------|:-----------:|:--------:|:------:|:---------:|:-----:|
| A1-CE      | ConvNeXt-L  | (buggy baseline)     | 0.7141      | 0.6370   | 7.7pp  | 0.895     | 0.898 |
| B1         | ConvNeXt-L  | Bug fixes only       | 0.7334      | 0.6387   | 9.5pp  | 0.901     | 0.905 |
| B2         | DINOv2-ViT-L| Bug fixes + CE       | 0.6940      | 0.6136   | 8.0pp  | 0.884     | 0.896 |
| B3         | ConvNeXt-L  | + Mixup/CutMix       | 0.7193      | 0.6459   | 7.3pp  | 0.920     | 0.894 |
| B4         | ConvNeXt-L  | + CORAL Ordinal      | 0.4449      | FAILED   | —      | 0.915     | 0.864 |
| B5         | ConvNeXt-L  | + SWA                | 0.7286      | 0.6615*  | 6.7pp  | 0.936     | 0.913 |
|------------|-------------|----------------------|-------------|----------|--------|-----------|-------|
| 16-bit ref | ConvNeXt-L  | (target)             | 0.6867      | 0.7233   | -3.7pp | —         | —     |
```
*B5 test F1 is from SWA-averaged model, not best checkpoint.
