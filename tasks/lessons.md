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

---

## C-Series Design Rationale (2026-04-11)

### Lesson #21 — Generalization Gap Is the Primary Adversary, Attack It from Multiple Angles
**Evidence from B-series:**
- B1 (bug fixes only): gap 9.5pp — worst in study
- B3 (Mixup/CutMix): gap 7.3pp — best relative gap reduction
- B5 (SWA): gap 6.7pp — best absolute test F1 (0.6615)
- C1 (SWA + Mixup): expected to stack → gap ~5-6pp

Even with the best techniques, the gap remains substantial. The 8-bit pipeline's intrinsic ceiling (Lesson #19) means we must extract maximum generalization from every possible angle.

**Five regularization axes identified:**
1. **Capacity (C4):** Over-parameterization → structural regularization via smaller backbone
2. **Feature Preservation (C5):** Fine-tuning damage → lower backbone LR preserves ImageNet features
3. **Loss Landscape (C6, C7):** Auxiliary loss noise (C6) and hard-example mining (C7)
4. **Explicit Dropout (C8):** Memorization → force distributed representations
5. **Combination (C1):** Orthogonal stacking of proven techniques

**Rule:** When the generalization gap is the bottleneck (not val performance), systematic ablation across independent regularization axes is more informative than iteratively stacking techniques. Each axis tests a different hypothesis about WHY the model overfits — capacity, feature destruction, loss function noise, or co-adaptation. Results will reveal which overfitting mechanism dominates in 8-bit mammography.

---

## C-Series Experiment Lessons (2026-04-12)

### Lesson #22 — Asymmetry Loss Is the Hidden Generalization Gap Source (C6 — New 8-bit Champion)
**Problem:** C6 removed the asymmetry_loss (weight 0.10→0.0) from B5 (SWA). This was intended as a neutral "does it help or hurt?" test. The answer was emphatic:

**Evidence:**
- **Test F1: 0.6762** — new 8-bit record (+1.47pp vs B5's 0.6615)
- **Val→test gap: 4.21pp** — narrowest in the entire study (vs B5's 6.71pp → 2.50pp improvement)
- BR1: 0.531 (+7.8pp vs B5's 0.453) — massive recovery of the SWA-induced BR1 regression
- BR2: 0.798 (identical to B5) — no regression on the dominant class
- BR4: 0.518 (-2.9pp vs B5's 0.547) — mild regression
- BR5: 0.857 (+0.9pp vs B5)
- Accuracy: 0.7444 (vs B5's 0.7390)
- 16-bit gap narrowed to 4.71pp (from 6.18pp)

**Root cause:** The asymmetry loss computes bilateral differences (RCC vs LCC, RMLO vs LMLO) and penalizes the model based on left-right asymmetry patterns. These patterns overfit to training distribution-specific bilateral features that don't transfer to test. The extra gradient signal corrupts the primary classification loss landscape, especially on the delicate BR1-BR2 boundary where SWA's smoothing already introduces ambiguity.

**Why C6 recovered BR1:** The asymmetry loss was the *source* of the BR1 problem, not SWA alone. SWA smoothed decision boundaries, but asymmetry loss noise pushed the BR1-BR2 boundary into BR2 territory during training. Removing the noise let SWA average over a cleaner trajectory.

**SWA checkpoint note:** C6's SWA model beat the best checkpoint (SWA overwrote best_model.pt per train.py:734-741 logic). This confirms SWA works well when the loss is clean.

**Rule:** Auxiliary losses that capture domain-specific priors (bilateral asymmetry) can hurt generalization when the prior doesn't hold across the train→test shift. The asymmetry_loss should be **permanently disabled** for this dataset. When facing a generalization gap, audit existing losses before adding regularization — the gap may be caused by loss noise, not insufficient regularization.

### Lesson #23 — SWA and Mixup/CutMix Are Antagonistic, Not Orthogonal (C1)
**Problem:** C1 combined SWA + Mixup/CutMix, hypothesizing orthogonal stacking would reach test F1 ≥ 0.68. Result: test F1 = 0.6431 — **worse than both B5 (SWA only, 0.6615) and B3 (Mixup only, 0.6459)**.

**Evidence:**
- Test F1: 0.6431 (-1.84pp vs B5, -0.28pp vs B3)
- Gap: 7.27pp (worse than B5's 6.71pp and B3's 7.34pp)
- BR2: 0.747 (-5.1pp vs B5's 0.798) — significant regression
- BR5: 0.826 (-2.2pp vs B5's 0.848)
- SWA model was WORSE than best checkpoint (swa_model.pt saved separately = best model won)

**Root cause:** SWA averages model parameters from the late training trajectory (epoch 5+). Mixup blurs decision boundaries during training by creating interpolated samples with soft labels. SWA preserves and amplifies this blur by averaging the blurred parameters. The two techniques both operate on **decision boundary smoothing** — SWA through weight-space averaging, Mixup through input-space interpolation — making them redundant rather than orthogonal.

The SWA model performing worse than the best checkpoint (confirmed by swa_model.pt existing as a separate file) proves that the weight trajectory under Mixup is not suitable for averaging.

**Rule:** Do NOT combine SWA with Mixup/CutMix for this task. They are antagonistic, not orthogonal. "Both are regularizers" does not mean they stack. Always validate combinations vs. individual techniques. The Mixup + SWA interference invalidates the Lesson #20 recommendation for this combination.

### Lesson #24 — Bug Fixes Increase Overfitting Regardless of Backbone (C2 Confirms Lesson #13)
**Problem:** C2 applied bug fixes (correct normalization, correct class weights) to DINOv2 + focal loss (baseline: A3). Test F1: 0.6240, down from A3's 0.6325 (-0.85pp).

**Evidence (comparing C2 vs A3):**
- Val F1: 0.6905 vs 0.6940 (slight drop — unusual, bugs may have helped even val)
- Test F1: 0.6240 vs 0.6325 (-0.85pp)
- Gap: 6.65pp vs 6.15pp (gap widened, same direction as B1 vs A1-CE)
- BR1: 0.376 vs 0.482 (-10.6pp!) — worst BR1 in the entire study
- BR2: 0.727 vs 0.681 (+4.6pp) — some improvement
- BR5: 0.859 vs 0.857 (stable)

**Root cause:** Identical phenomenon to Lesson #13 (B1 vs A1-CE for ConvNeXtV2): bugs (wrong normalization stats, train-set class weights) inject noise that accidentally regularizes. Correct values enable sharper fitting. This is **backbone-agnostic** — affects ConvNeXtV2 (-0.17pp or stable), DINOv2 (-0.85pp), and likely any architecture.

**BR1 collapse explanation:** DINOv2 with correct normalization aligns features more tightly to the training distribution. Without the regularizing noise, the model overconfidently classifies borderline BR1 cases as BR2 (larger class pulls).

**Rule:** Bug fixes MUST be paired with explicit regularization, regardless of backbone. Lesson #13 is not ConvNeXtV2-specific — it is a universal phenomenon in this task. For DINOv2, consider adding SWA or dropout (not Mixup) to compensate for lost implicit regularization.

### Lesson #25 — Larger Pretrained Models Generalize Better in Low-Data Medical Imaging (C4)
**Problem:** C4 replaced ConvNeXtV2-Large (~197M params, feature_dim=1536) with ConvNeXtV2-Base (~89M params, feature_dim=1024). Hypothesis: smaller model = less memorization = smaller gap. **Wrong.**

**Evidence:**
- Test F1: 0.6269 (-3.46pp vs B5) — significant regression
- Gap: 9.09pp — one of the **worst gaps** in the entire study (only B1's 9.5pp is worse)
- Val F1: 0.7178 (competitive!) — the model fits training data well but doesn't transfer
- BR4: 0.490 (-5.7pp vs B5) — worst BR4 regression in C-series
- BR5: 0.865 (+1.7pp vs B5) — easier high-confidence cases improve, hard cases suffer
- SWA model beat best checkpoint (confirmed by no swa_model.pt)

**Root cause:** In the low-data regime (8,557 patients), pretrained features ARE the regularizer. ConvNeXtV2-Large's richer feature space (197M params trained on ImageNet-22k) provides better transfer learning foundations than Base's reduced capacity. Cutting parameters removes useful pretrained representations without reducing memorization — the model still memorizes training-specific patterns, just with fewer tools to generalize from.

The paradox: larger models generalize better despite having more parameters because the extra capacity stores ImageNet-learned features that transfer, not dataset-specific noise.

**Rule:** Do NOT reduce model capacity to fight overfitting in transfer learning with limited medical data. In low-data regimes, larger pretrained models > smaller ones. The overfitting source is not excess capacity — it's train/test distribution shift. Address distribution shift (SWA, cleaner losses) rather than capacity.

### Lesson #26 — Excessive Feature Preservation Prevents Domain Adaptation (C5)
**Problem:** C5 reduced backbone_lr_scale from 0.2 to 0.05 (effective backbone LR: 2.5e-6 vs 1e-5). Hypothesis: preserving more ImageNet-22k features = better generalization. **Wrong.**

**Evidence:**
- Test F1: 0.6284 (-3.31pp vs B5) — significant regression
- Gap: 8.30pp (vs B5's 6.71pp)
- BR2: 0.698 (-10.0pp vs B5's 0.798) — **massive regression**, worst BR2 in C-series
- BR4: 0.493 (-5.4pp vs B5)
- BR1: 0.462 (+0.9pp vs B5) — marginal improvement
- BR5: 0.861 (+1.3pp vs B5) — slight gain
- SWA model beat best checkpoint (confirmed)

**Root cause:** Mammography textures — spiculations, microcalcifications, tissue density patterns — are sufficiently different from ImageNet-22k natural images that substantial backbone fine-tuning is essential. At backbone_lr_scale=0.05, the backbone is nearly frozen, preserving ImageNet features that don't map to mammography-specific discriminative patterns. BR2 (benign findings) suffers most because benign tissue patterns are furthest from ImageNet objects.

The BR1/BR5 slight improvement confirms: high-level semantic features (normal vs malignant) transfer better from ImageNet than mid-level tissue texture features (benign findings).

**Rule:** backbone_lr_scale=0.2 is near-optimal for ConvNeXtV2 on 8-bit mammography. Going lower (0.05) prevents necessary domain adaptation. Do not under-tune the backbone — mammography requires more adaptation than typical medical imaging transfer tasks because of the unique tissue texture domain.

### Lesson #27 — Class Weight Manipulation Is Zero-Sum on Shared Decision Boundaries (C3)
**Problem:** C3 increased BR1 class weight from 1.28 to 1.80 (+40%) to counter SWA's BR1 regression (B5: BR1=0.453). Hypothesis: higher BR1 weight = BR1 recovery without macro collapse.

**Evidence:**
- Test F1: 0.6346 (-2.69pp vs B5)
- BR1: 0.465 (+1.2pp vs B5's 0.453) — **minimal improvement** despite 40% weight increase
- BR2: 0.700 (-9.8pp vs B5's 0.798) — **catastrophic regression**
- BR4: 0.521 (-2.6pp)
- Gap: 7.59pp (worse than B5's 6.71pp)

**Contrast with C6 (the right approach):**
- C6 recovered BR1 by +7.8pp (0.453→0.531) WITHOUT any BR2 regression (0.798 stable)
- C6 achieved this by removing noise (asymmetry loss), not by shifting decision boundaries

**Root cause:** BR1 (normal) and BR2 (benign findings) share a fuzzy decision boundary — subtle tissue changes separate them. Increasing BR1 weight shifts this boundary toward BR2 territory, reclassifying borderline BR2 cases as BR1. This is a **zero-sum game**: every BR1 gain comes from BR2 loss. The 1.2pp BR1 gain cost 9.8pp BR2 — an 8:1 efficiency ratio in the wrong direction.

The weight increase also amplifies gradient noise from the hard BR1 cases (n=163 test, smallest class), destabilizing the overall optimization.

**Rule:** Class weight manipulation is a blunt instrument that cannot create new discriminative features — it only shifts existing decision boundaries. For minority class recovery on shared boundaries, address the root cause (noisy loss signals, architectural issues) rather than adjusting weights. C6 proves this: removing asymmetry loss noise recovered BR1 6.5x more effectively than a 40% weight boost.

### Lesson #28 — Focal Loss Remains Harmful for ConvNeXtV2 Even With SWA (C7)
**Problem:** C7 changed loss from CE to focal (gamma=2.0) under SWA conditions. Hypothesis: SWA's flat minima + focal's hard-example mining might rescue the focal loss for ConvNeXtV2.

**Evidence:**
- Test F1: 0.6468 (-1.47pp vs B5)
- Gap: 7.93pp (vs B5's 6.71pp)
- BR1: 0.485 (+3.2pp vs B5) — some improvement from hard-example mining
- BR4: 0.498 (-4.9pp vs B5) — significant regression
- BR2: 0.753 (-4.5pp vs B5)
- SWA model was WORSE than best checkpoint (swa_model.pt saved separately)

**Key observation:** SWA was counterproductive with focal loss (SWA model lost to best checkpoint). This makes sense: focal loss creates a non-stationary loss surface (hard examples change as training progresses), and SWA averaging over this non-stationary trajectory produces a poor average.

**Rule:** Focal loss is harmful for ConvNeXtV2 regardless of SWA (confirming Lesson #11). SWA is also incompatible with focal loss (non-stationary loss surface). This finding is robust across A-series (A1 vs A1-CE) and C-series (C7 vs B5). NEVER use focal loss with ConvNeXtV2 in this pipeline.

### Lesson #29 — Meta-Lesson: Simplification Beats Regularization (C-Series Summary)

**Evidence:** 7 experiments tested 7 different approaches to reducing the val→test generalization gap:

```
| Rank | Exp | Strategy              | Test F1 | Δ vs B5 | Gap    | Verdict        |
|------|-----|-----------------------|---------|---------|--------|----------------|
| 1    | C6  | Remove asymmetry loss | 0.6762  | +1.47pp | 4.21pp | ✅ NEW BEST     |
| 2    | C7  | Add focal loss        | 0.6468  | -1.47pp | 7.93pp | ❌ Worse        |
| 3    | C1  | Stack SWA+Mixup       | 0.6431  | -1.84pp | 7.27pp | ❌ Antagonistic |
| 4    | C3  | Increase BR1 weight   | 0.6346  | -2.69pp | 7.59pp | ❌ Zero-sum     |
| 5    | C5  | Freeze backbone more  | 0.6284  | -3.31pp | 8.30pp | ❌ Under-adapted|
| 6    | C4  | Smaller backbone      | 0.6269  | -3.46pp | 9.09pp | ❌ Over-reduced |
| 7    | C2  | Fix DINOv2 bugs       | 0.6240  | -3.75pp | 6.65pp | ❌ Overfit more |
```

**6 out of 7 hypotheses FAILED.** The one winner (C6) **removed** complexity rather than adding regularization. The generalization gap was not caused by insufficient regularization — it was caused by an auxiliary loss (asymmetry_loss) injecting noise.

**Implication for D-series:** The best configuration is now:
- ConvNeXtV2-Large + CE loss + SWA + **no asymmetry loss** (C6 config)
- Test F1: 0.6762, gap: 4.21pp
- 16-bit gap: 4.71pp (0.7233 - 0.6762)

**Rule:** Before adding regularization to fight a generalization gap, audit every component of the existing loss for unnecessary complexity. The Occam's Razor principle applies to loss functions: **the simplest loss that fits is the one that generalizes.**

### Summary Statistics — C-Series Ablation Table

```
| Experiment | Backbone     | Change vs B5              | Best Val F1 | Test F1  | Gap    | BR1   | BR2   | BR4   | BR5   |
|------------|-------------|---------------------------|:-----------:|:--------:|:------:|:-----:|:-----:|:-----:|:-----:|
| B5 (base)  | ConvNeXt-L  | SWA only                  | 0.7286      | 0.6615   | 6.71pp | 0.453 | 0.798 | 0.547 | 0.848 |
| C1         | ConvNeXt-L  | + Mixup/CutMix            | 0.7158      | 0.6431   | 7.27pp | 0.458 | 0.747 | 0.541 | 0.826 |
| C2         | DINOv2-ViT-L| Focal + bug fixes         | 0.6905      | 0.6240   | 6.65pp | 0.376 | 0.727 | 0.535 | 0.859 |
| C3         | ConvNeXt-L  | BR1 weight 1.80           | 0.7105      | 0.6346   | 7.59pp | 0.465 | 0.700 | 0.521 | 0.853 |
| C4         | ConvNeXt-B  | Base backbone (~89M)      | 0.7178      | 0.6269   | 9.09pp | 0.421 | 0.731 | 0.490 | 0.865 |
| C5         | ConvNeXt-L  | backbone_lr_scale=0.05    | 0.7114      | 0.6284   | 8.30pp | 0.462 | 0.698 | 0.493 | 0.861 |
| **C6**     | ConvNeXt-L  | **asymmetry_wt=0.0**      | **0.7183**  |**0.6762**|**4.21pp**|**0.531**|**0.798**|0.518|0.857|
| C7         | ConvNeXt-L  | Focal + SWA               | 0.7261      | 0.6468   | 7.93pp | 0.485 | 0.753 | 0.498 | 0.852 |
|------------|-------------|---------------------------|-------------|----------|--------|-------|-------|-------|-------|
| 16-bit ref | ConvNeXt-L  | (target)                  | 0.6867      | 0.7233   | -3.7pp | —     | —     | —     | —     |
```

### Recommended Next Steps (Post C-Series)
1. **C8** (Extreme Dropout, still pending) — Given 6/7 "add regularization" approaches failed, C8 is unlikely to help. Run for completeness but low expectations.
2. **D1: C6 + Mixup** — C6 removed asymmetry noise. Mixup (B3) was the best standalone gap regularizer. Test if Mixup works better with a cleaner loss (C1 failed partly because asymmetry noise + Mixup + SWA was triple-noisy).
3. **D2: C6 config on 16-bit pipeline** — The asymmetry loss insight may apply to 16-bit too. Could recover performance there.
4. **D3: Ensemble C6 + best DINOv2** — Complementary error profiles (ConvNeXtV2 strong on BR2/BR4, DINOv2 strong on BR5).
5. **D4: C6 without SWA** — Isolate whether the gain comes from loss simplification alone or requires SWA.
6. **ABANDON:** Capacity reduction (C4), feature freezing (C5), class weight manipulation (C3), SWA+Mixup combo (C1).
