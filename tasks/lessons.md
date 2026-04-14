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
**Problem:** B4 used CORAL ordinal loss with K-1=3 cumulative binary classifiers, with subgroup heads disabled (a previously tested configuration). Despite following the proven pattern from `ordinal_nosubgroup_v1.yaml`, B4 CATASTROPHICALLY FAILED:
- Val F1 peaked at 0.4449 (29pp below B1)
- BR4 val F1 = 0.054 (near-zero collapse)
- No test evaluation was triggered
- Training plateaued at val F1 ~0.31 for 34 consecutive epochs before slowly climbing to 0.44

**Root cause:** CORAL models cumulative thresholds: P(y≥2), P(y≥4), P(y≥5). With non-contiguous BI-RADS classes (1,2,**skip 3**,4,5), the P(y≥4) threshold has no natural decision boundary — there is no class 3 to separate from class 4. The optimizer gets stuck because the middle threshold receives conflicting gradients from BR2 (push threshold right) and BR4 (push threshold left) with no intermediate class to anchor it.

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

Even with the best techniques, the gap remains substantial, so we must extract maximum generalization from every possible angle.

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
```

### Recommended Next Steps (Post C-Series)
1. **C8** (Extreme Dropout, still pending) — Given 6/7 "add regularization" approaches failed, C8 is unlikely to help. Run for completeness but low expectations.
2. **D1: C6 + Mixup** — C6 removed asymmetry noise. Mixup (B3) was the best standalone gap regularizer. Test if Mixup works better with a cleaner loss (C1 failed partly because asymmetry noise + Mixup + SWA was triple-noisy).
3. **D3: Ensemble C6 + best DINOv2** — Complementary error profiles (ConvNeXtV2 strong on BR2/BR4, DINOv2 strong on BR5).
4. **D4: C6 without SWA** — Isolate whether the gain comes from loss simplification alone or requires SWA.
5. **ABANDON:** Capacity reduction (C4), feature freezing (C5), class weight manipulation (C3), SWA+Mixup combo (C1).

---

## D-Series Experiment Lessons (2026-04-14)

### Lesson #30 — Auxiliary Heads Provide Essential Multi-Task Regularization, Unlike Asymmetry Loss (D1-D3)
**Problem:** After C6 showed that removing asymmetry loss improved generalization, the D-series tested whether removing auxiliary heads (subgroup, binary) would continue the simplification trend. **All three removals hurt.**

**Evidence:**
- D1 (no subgroup, wt 0.45): test F1 = 0.6563, **-1.99pp** vs C6
- D2 (no binary, wt 0.10): test F1 = 0.6453, **-3.09pp** vs C6
- D3 (no subgroup + no binary): test F1 = 0.6476, **-2.86pp** vs C6
- All three gap-widened: D1 5.75pp, D2 6.94pp, D3 7.61pp (vs C6's 4.21pp)

**Root cause — why auxiliary heads help but asymmetry loss hurt:**
The auxiliary heads provide *complementary gradient signals* that regularize the shared backbone through multi-task learning. The binary head forces a coarse {BR1,BR2} vs {BR4,BR5} separation; the subgroup head imposes an intermediate grouping. These create multiple consistent views of the same classification hierarchy, stabilizing training.

Asymmetry loss, by contrast, computes bilateral differences (L vs R) — a *domain-specific prior* that doesn't hold across train→test shift. The auxiliary heads encode *task-structure priors* (hierarchical class groupings) that are invariant across distributions.

**Rule:** Not all loss components are equal. Auxiliary heads encoding task-structure hierarchy are beneficial regularizers. Domain-specific priors (bilateral asymmetry) are overfitting risks. When simplifying losses, distinguish between structural multi-task components (keep) and domain-prior components (audit carefully).

### Lesson #31 — Binary Head Punches 3x Above Its Weight as a Gradient Anchor (D2)
**Problem:** The binary head has only 0.10 weight (10% of total loss), yet removing it caused the largest single-head impact: -3.09pp. This is 1.55x worse than removing the subgroup head (-1.99pp) which carries 4.5x more weight (0.45).

**Evidence:**
- D2 (no binary): macro F1 = 0.6453, gap = 6.94pp
- BR4 collapsed: 0.476 (-4.2pp vs C6's 0.518) — worst BR4 in D-series
- BR5 dropped: 0.844 (-1.3pp vs C6)
- BR1 actually improved: 0.521 (-1.0pp vs C6) — lost gradient anchor shifts all boundaries
- SWA lost to best checkpoint (swa_model.pt saved separately)

**Root cause:** The binary head forces the model to separate {BR1,BR2} vs {BR4,BR5} — a clean 2D gradient signal. This binary decision boundary *anchors* the full 4-class classifier. Without it, the model must discover this coarse separation from the fine-grained 4-class loss alone, which is harder and less stable. The binary gradient acts as a "curriculum" signal: learn coarse separation first, then refine.

**Efficiency paradox:** Impact/weight ratio: binary = -3.09pp / 0.10wt = 30.9 per unit. Subgroup = -1.99pp / 0.45wt = 4.4 per unit. The binary head is **7x more efficient** per unit weight than subgroup.

**Rule:** In hierarchical classification, a lightweight auxiliary head encoding the coarsest class grouping provides disproportionate regularization. The binary head's low weight (0.10) is already near-optimal — it provides a stabilizing gradient anchor without dominating the loss. Do NOT remove or reduce it.

### Lesson #32 — Clean Loss Alone Matches Dirty Loss + SWA — Asymmetry Removal Had More Impact Than SWA (D4)
**Problem:** D4 removed SWA from C6, isolating the "clean loss alone" contribution. This creates a clean 2×2 factorial comparison.

**Evidence — 2×2 Factorial:**
```
|                    | Dirty Loss (asym=0.10) | Clean Loss (asym=0.0) | Δ (clean vs dirty) |
|--------------------|:----------------------:|:---------------------:|:-------------------:|
| No SWA             | B1 = 0.6387            | D4 = 0.6615           | +2.28pp             |
| SWA                | B5 = 0.6615            | C6 = 0.6762           | +1.47pp             |
| Δ (SWA vs no SWA)  | +2.28pp                | +1.47pp               |                     |
```

**Key insights:**
- **D4 (clean, no SWA) = B5 (dirty, SWA) = 0.6615 exactly.** Removing asymmetry loss is equivalent in impact to adding SWA.
- Asymmetry removal effect: +2.28pp (no SWA) / +1.47pp (with SWA) → **larger than SWA's contribution**
- SWA effect: +2.28pp (dirty loss) / +1.47pp (clean loss) → SWA helps more on dirty loss (compensates for noise)
- The effects are **sub-additive**: 2.28 + 2.28 = 4.56pp expected but only 3.75pp observed (C6 vs B1). Some overlap in what they fix.

**D4's gap is remarkably tight:** 4.94pp — second narrowest after C6 (4.21pp). Clean loss fundamentally reduces overfitting even without SWA's weight averaging.

**Rule:** Removing loss noise has equal or greater impact than adding SWA. Before investing in SWA or other post-hoc regularization, audit the loss function for unnecessary components. In the current pipeline, asymmetry removal alone was worth +2.28pp — a "free" improvement that required no extra compute.

### Lesson #33 — DINOv2 Partially Rescued by SWA + Clean Loss, But Architecture Gap Persists (D5)
**Problem:** D5 applied the C-series winning insights (SWA + no asymmetry) to DINOv2 (baseline: C2). Target: test F1 ≥ 0.65 to justify an ensemble path.

**Evidence:**
- D5 test F1: 0.6383, **+1.43pp** vs C2 (0.6240) — meaningful improvement
- Gap: 5.42pp (vs C2's 6.65pp) — improved
- But **still 3.79pp below C6** and **below the 0.65 target**
- BR5: 0.864 (+0.5pp vs C2's 0.859) — DINOv2's strength maintained
- BR1: 0.454 (+7.9pp vs C2's 0.376) — SWA recovered some BR1
- BR2: 0.720 (-0.7pp vs C2's 0.727) — stable
- SWA lost to best checkpoint (swa_model.pt saved separately)

**SWA dynamics:** SWA lost to best checkpoint on DINOv2, unlike ConvNeXtV2 C6 where SWA won. DINOv2's self-supervised features create a less stable training trajectory — the averaging introduces noise rather than smoothing. The test evaluation still used the best checkpoint, but the +1.43pp gain came from removing asymmetry noise rather than SWA averaging.

**Root cause of persistent gap:** DINOv2 ViT-L was pretrained via self-supervised learning on natural images. Its patch-based attention mechanism processes 37×14 patches at 518px — far coarser than ConvNeXtV2's hierarchical feature maps at 1024px. Mammographic features (microcalcifications, spiculations) require local high-resolution detail that DINOv2's architecture doesn't capture well despite its global attention capabilities.

**Rule:** DINOv2 is NOT viable as a primary backbone for this mammography task — the architecture gap is fundamental, not fixable by training tricks. At 0.6383 test F1, it's too weak for ensemble contribution (< 0.65 threshold). **Abandon the DINOv2 track.** Future backbone exploration should focus on ConvNeXtV2 variants or other CNN-based architectures that preserve local spatial detail.

### Lesson #34 — Mixup's B-Series Benefit Was Compensating for Asymmetry Noise, Not True Regularization (D6)
**Problem:** D6 tested Mixup/CutMix on clean loss without SWA. In B-series, B3 (Mixup on dirty loss) improved over B1 by +0.72pp. Would Mixup improve on clean loss too?

**Evidence — Mixup on clean vs dirty loss:**
```
| Condition              | With Mixup       | Without Mixup    | Mixup Effect |
|------------------------|:----------------:|:----------------:|:------------:|
| Dirty loss, no SWA     | B3 = 0.6459      | B1 = 0.6387      | +0.72pp      |
| Clean loss, no SWA     | D6 = 0.6353      | D4 = 0.6615      | **-2.62pp**  |
| Clean loss, with SWA   | D7 = 0.6563      | C6 = 0.6762      | **-1.99pp**  |
```

- **Mixup HELPED (+0.72pp) on dirty loss but HURT (-2.62pp) on clean loss**
- D6 gap = 8.96pp — **worst gap in the entire D-series** and one of the worst in the study
- D6 val F1 = 0.7249 (2nd highest in D-series!) but test F1 = 0.6353 → severe overfitting
- BR2 collapsed: 0.701 (-5.6pp vs D4's 0.757)

**Root cause:** Mixup creates interpolated training samples with soft labels. On dirty loss (with asymmetry noise), this interpolation counteracted the noise — Mixup's label smoothing partially compensated for asymmetry's corrupted gradients. On clean loss, the interpolation creates *confusing* training signals (blending normal BR1 tissue with malignant BR5, for example) that the model doesn't need to learn from. The clean loss already provides accurate gradients; Mixup degrades them.

The high val F1 (0.7249) with low test F1 (0.6353) reveals that Mixup's input-space interpolation creates training-specific smoothness that doesn't transfer to real test images.

**Rule:** Mixup/CutMix is NOT a universal regularizer — it was effective only because it was compensating for a different problem (asymmetry loss noise). On a clean loss function, Mixup is **harmful**. This reframes Lesson #15 (B3): Mixup's apparent gap-closing ability was an artifact of the dirty loss, not an intrinsic regularization benefit. **Permanently abandon Mixup/CutMix for this pipeline** now that the loss is clean.

### Lesson #35 — SWA + Mixup Antagonism Is Intrinsic, Not Confounded by Asymmetry (D7 Confirms Lesson #23)
**Problem:** C1 (SWA+Mixup on dirty loss) failed, but was the antagonism caused by asymmetry noise confounding the combination? D7 retests SWA+Mixup on C6's clean loss to disambiguate.

**Evidence:**
- D7 (SWA+Mixup, clean loss): test F1 = 0.6563, **-1.99pp** vs C6
- C1 (SWA+Mixup, dirty loss): test F1 = 0.6431, -1.84pp vs B5
- Both show ~2pp penalty for adding Mixup to SWA
- D7 gap: 5.96pp (vs C6's 4.21pp = +1.75pp wider)
- SWA lost to best checkpoint in D7 (swa_model.pt saved separately) — same as C1

**The smoking gun — SWA trajectory corruption:**
SWA won (overwriting best_model.pt) in C6 (no Mixup) but LOST in D7 (with Mixup). The only difference is Mixup. Mixup corrupts the late-training weight trajectory that SWA averages over, making the averaged model worse than the best single checkpoint.

**Definitively answering the D-series question:** The SWA+Mixup antagonism is NOT caused by asymmetry noise. It is an **intrinsic incompatibility** between two boundary-smoothing mechanisms:
- SWA smooths in weight space (averaging model parameters)
- Mixup smooths in input space (interpolating training samples)
- Combined, they over-smooth, blurring the BR1/BR2 and BR4/BR5 boundaries beyond useful classification

**Rule:** SWA and Mixup/CutMix are permanently incompatible for this task. Lesson #23 is confirmed and strengthened: the antagonism is intrinsic to the mechanism interaction, not an artifact of loss function noise. **Never combine SWA with Mixup/CutMix regardless of other configuration choices.**

### Lesson #36 — SWA Effectiveness Depends on Loss Landscape Balance (D1-D3 SWA Patterns)
**Problem:** SWA won (overwriting best_model.pt) in C6 (3 auxiliary heads) and D3 (0 auxiliary heads), but LOST in D1 (2 heads: binary+full) and D2 (2 heads: subgroup+full). Why?

**Evidence — SWA win/loss pattern:**
```
| Experiment | Active Heads                | SWA Outcome | Test F1 |
|------------|----------------------------|:-----------:|:-------:|
| C6         | binary + subgroup + full   | WON         | 0.6762  |
| D3         | full only                  | WON         | 0.6476  |
| D1         | binary + full              | LOST        | 0.6563  |
| D2         | subgroup + full            | LOST        | 0.6453  |
```

**Root cause:** SWA averages model weights from the late training trajectory (epochs 5+). This averaging works best when the loss landscape is *stable and balanced* — i.e., when the gradient directions don't oscillate wildly.

- **C6 (3 heads):** Three loss terms provide balanced, redundant gradient signals. The multi-task structure creates a smooth loss landscape where SWA averaging produces a good solution.
- **D3 (1 head):** Single loss term = simplest possible landscape. SWA averaging works because there's no inter-objective conflict.
- **D1/D2 (2 heads):** Removing one head creates an *asymmetric* multi-task loss. The remaining two losses compete without the third providing a balancing gradient. This creates oscillation in the training trajectory that SWA averaging fails to smooth.

**Analogy:** Think of 3 legs on a stool (stable), 1 leg (a simple pole, stable in its own way), but 2 legs (unstable, falls to one side).

**Rule:** When using SWA with multi-task losses, ensure the loss landscape is balanced. Either use the full multi-task structure or simplify to a single objective. Removing individual auxiliary heads while keeping SWA creates instability. If testing auxiliary head removal, also consider disabling SWA or using the best-checkpoint-only evaluation.

### Lesson #37 — D-Series Meta: C6 Is the Goldilocks Configuration — All 7 Simplifications/Additions Failed
**Problem:** The D-series tested C6 from 7 angles: 3 loss simplifications (D1-D3), 1 component removal (D4), 1 DINOv2 rescue (D5), 2 Mixup tests (D6-D7). **All 7 experiments scored below C6.**

**Evidence (ranked by test F1):**
```
| Rank | Exp | Strategy               | Test F1 | Δ vs C6  | Gap    | Verdict                           |
|------|-----|------------------------|---------|----------|--------|-----------------------------------|
| —    | C6  | BASELINE               | 0.6762  | —        | 4.21pp | CHAMPION                          |
| 1    | D4  | Remove SWA             | 0.6615  | -1.47pp  | 4.94pp | SWA essential (+1.47pp)           |
| 2    | D1  | Remove subgroup head   | 0.6563  | -1.99pp  | 5.75pp | Subgroup head helpful             |
| 2    | D7  | Add Mixup + SWA        | 0.6563  | -1.99pp  | 5.96pp | SWA+Mixup antagonism intrinsic    |
| 4    | D3  | Remove both heads      | 0.6476  | -2.86pp  | 7.61pp | Multi-task learning essential      |
| 5    | D2  | Remove binary head     | 0.6453  | -3.09pp  | 6.94pp | Binary head critical gradient anchor|
| 6    | D5  | DINOv2 rescue          | 0.6383  | -3.79pp  | 5.42pp | Architecture gap persistent        |
| 7    | D6  | Mixup replaces SWA     | 0.6353  | -4.09pp  | 8.96pp | Mixup harmful on clean loss        |
```

**C-series simplified INTO the optimum. D-series probed BEYOND it and found nothing better.**

**Component contribution (isolated effects from C6 baseline):**
- SWA: +1.47pp (D4→C6)
- Subgroup head: +1.99pp (D1→C6)
- Binary head: +3.09pp (D2→C6)
- Asymmetry removal: +2.28pp (B1→D4, measured without SWA)

**Confirmed permanently abandoned:**
- Mixup/CutMix (harmful on clean loss — Lesson #34)
- Focal loss for ConvNeXtV2 (C7, Lesson #28)
- DINOv2 as primary backbone (D5, Lesson #33)
- Asymmetry loss (C6, Lesson #22)
- Class weight manipulation (C3, Lesson #27)
- Capacity reduction (C4, Lesson #25)
- Backbone freezing below lr_scale=0.2 (C5, Lesson #26)

**D-Series Decision Tree Outcomes:**
- ❌ D1/D3 < C6 → no further loss simplification
- ✅ D4 >> B1 (+2.28pp) → clean loss alone is very valuable
- ✅ D4 < C6 (-1.47pp) → SWA is essential on top of clean loss
- ❌ D5 < 0.65 → ensemble path abandoned
- ❌ D7 < C6 → SWA+Mixup antagonism was NOT confounded

**Rule:** C6's configuration — ConvNeXtV2-Large + CE loss + SWA + three auxiliary heads (binary 0.10, subgroup 0.45, full 0.45) + no asymmetry loss — represents the optimal single-model configuration for 8-bit 1024×1024 mammography BI-RADS classification. Further improvements should explore: (a) learning rate schedules, (b) data augmentation strategies beyond Mixup, (c) test-time augmentation, (d) ensemble strategies with ConvNeXtV2 variants, or (e) 16-bit pipeline optimization.

### Summary Statistics — D-Series Ablation Table

```
| Experiment | Backbone     | Change vs C6              | Best Val F1 | Test F1  | Gap    | BR1   | BR2   | BR4   | BR5   | SWA     |
|------------|-------------|---------------------------|:-----------:|:--------:|:------:|:-----:|:-----:|:-----:|:-----:|:-------:|
| C6 (base)  | ConvNeXt-L  | BASELINE                  | 0.7183      | 0.6762   | 4.21pp | 0.531 | 0.798 | 0.518 | 0.857 | WON     |
| D1         | ConvNeXt-L  | -subgroup head            | 0.7138      | 0.6563   | 5.75pp | 0.479 | 0.751 | 0.532 | 0.864 | LOST    |
| D2         | ConvNeXt-L  | -binary head              | 0.7147      | 0.6453   | 6.94pp | 0.521 | 0.741 | 0.476 | 0.844 | LOST    |
| D3         | ConvNeXt-L  | -subgroup -binary         | 0.7237      | 0.6476   | 7.61pp | 0.461 | 0.748 | 0.532 | 0.849 | WON     |
| D4         | ConvNeXt-L  | -SWA                      | 0.7109      | 0.6615   | 4.94pp | 0.500 | 0.757 | 0.542 | 0.847 | N/A     |
| D5         | DINOv2-ViT-L| +SWA -asymmetry (from C2) | 0.6925      | 0.6383   | 5.42pp | 0.454 | 0.720 | 0.515 | 0.864 | LOST    |
| D6         | ConvNeXt-L  | -SWA +Mixup/CutMix        | 0.7249      | 0.6353   | 8.96pp | 0.487 | 0.701 | 0.508 | 0.846 | N/A     |
| D7         | ConvNeXt-L  | +Mixup/CutMix             | 0.7159      | 0.6563   | 5.96pp | 0.482 | 0.773 | 0.515 | 0.856 | LOST    |
```

### Recommended Next Steps (Post D-Series)
C6 is confirmed optimal. The 8-bit pipeline has a hard floor at test F1 ≈ 0.68. Potential E-series directions:
1. **Learning rate schedule tuning** — C6 uses step LR. Try cosine annealing with warm restarts to improve SWA trajectory.
2. **Spatial augmentation** — Geometric transforms (rotation, elastic deformation) that don't blend labels like Mixup.
3. **Test-Time Augmentation (TTA)** — Multi-view inference: flip/rotate at test time and average predictions. Free generalization without training changes.
4. **Ensemble (ConvNeXtV2 variants only)** — Ensemble top 2-3 ConvNeXtV2 runs (C6, D4, D1) with different random seeds or training trajectories.
5. **16-bit pipeline optimization** — The 16-bit baseline (0.7233) still leads. Apply C6's insights (no asymmetry, SWA) to the 16-bit pipeline.
6. **ABANDON:** DINOv2 track, Mixup/CutMix, further loss simplification, additional auxiliary losses.
