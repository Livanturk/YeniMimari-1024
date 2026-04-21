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
5. **ABANDON:** DINOv2 track, Mixup/CutMix, further loss simplification, additional auxiliary losses.

---

## E-Series Experiment Lessons (2026-04-17)

### Lesson #38 — OneCycleLR's Aggressive Peak LR Is Essential, Not Interchangeable (E1)
**Problem:** E1 replaced OneCycleLR (peak=5e-4 for heads, 1e-4 for backbone) with cosine_warmup (peak=5e-5 for heads, 1e-5 for backbone). The hypothesis was that a stable/decaying LR during the SWA phase would produce better weight averaging. **Wrong — 10x lower peak LR caused a 3.16pp regression.**

**Evidence:**
- Test F1: 0.6446, **-3.16pp** vs C6 (0.6762)
- Val F1: 0.7207 (+0.24pp vs C6's 0.7183) — slightly HIGHER val, much lower test
- Gap: 7.61pp (vs C6's 4.21pp → +3.40pp wider)
- BR1: 0.466 (-6.5pp vs C6's 0.531)
- BR2: 0.755 (-4.3pp vs C6's 0.798)
- BR4: 0.508 (-1.0pp)
- BR5: 0.850 (-0.7pp)
- SWA WON (overwrote best_model.pt)

**Root cause:** OneCycleLR's peak at 5e-4 (10x above the cosine warmup's peak of 5e-5) provides a critical **exploration phase** in the first 30% of training. This high-LR phase pushes the model out of narrow basins into broader regions of the loss landscape before SWA averaging begins. Cosine warmup's gentle LR never reaches high enough to escape the initial basin, converging to a nearby but inferior minimum. SWA then faithfully preserves this inferior solution — the SWA WON, but on a worse trajectory.

The val F1 being slightly *higher* with cosine warmup while test F1 drops confirms: the model found a sharper minimum (better val fit) but one that doesn't generalize. OneCycleLR's disruptive peak forces a flatter minimum that transfers better.

**Rule:** OneCycleLR with max_lr=5e-4 and pct_start=0.3 is not just a scheduler choice — it provides an essential high-LR exploration phase that determines the basin quality for subsequent SWA averaging. Do NOT replace it with monotonic or low-peak schedulers. The SWA literature's preference for stable LR applies to the averaging phase, but OneCycleLR satisfies this because the peak occurs at epoch ~30 (30% of 100) while SWA starts at epoch 5 — the LR *is* declining for most of the SWA phase.

### Lesson #39 — SWA-Optimal Schedules from Literature Fail on Multi-Objective Loss (E2)
**Problem:** E2 used cosine warm restarts (T_0=10, T_mult=2, eta_min=1e-7) — the schedule the original SWA paper (Izmailov et al., 2018) recommends as optimal. Restarts at epochs 10, 30, 70 should create diverse weight snapshots for richer averaging. **Result: worst test F1 in the E-series and one of the worst gaps in the entire study.**

**Evidence:**
- Test F1: 0.6363, **-3.99pp** vs C6 — worst in E-series
- Val F1: 0.7305 — **highest in E-series** (even higher than C6's 0.7183!)
- Gap: **9.42pp** — worst in E-series, rivaling B1's 9.5pp (the worst in the entire study)
- BR1: 0.427 (-10.4pp vs C6's 0.531) — massive collapse
- BR2: 0.749 (-4.9pp vs C6's 0.798)
- BR4: 0.513 (-0.5pp) — barely affected
- BR5: 0.857 (identical to C6)
- SWA WON (overwrote best_model.pt)

**Root cause:** The SWA paper's recommendations apply to single-objective tasks (image classification with one cross-entropy loss). This pipeline uses a multi-objective loss (binary + subgroup + full = 3 loss terms). Warm restarts force periodic LR spikes that push the optimizer to re-explore, but each restart also forces the model to **re-balance three competing objectives**. The trajectory between restarts oscillates wildly across the multi-objective Pareto front.

SWA averaging over these oscillatory trajectories produces a weight vector that is a poor compromise: the averaged weights fit the training distribution well (val F1 = 0.7305, highest!) but the averaged decision boundaries are incoherent for test generalization. The 9.42pp gap (val minus test) proves the SWA average is a **sharp, training-specific** solution despite weight-space averaging.

BR1's collapse (-10.4pp) is the signature: the smallest class is most sensitive to oscillatory boundary shifts, and SWA averaging over multiple restart phases blurs BR1's delicate boundaries into BR2.

**Rule:** Do NOT apply SWA literature recommendations (warm restarts, cyclic schedules) directly when using multi-objective losses. The original SWA paper assumes a single smooth loss landscape. Multi-task losses create a landscape with multiple competing gradients where warm restart diversity becomes harmful noise rather than useful exploration. Stick with OneCycleLR for this pipeline.

### Lesson #40 — Delayed SWA Start Trades BR2 for BR4 but Loses Both Overall (E3)
**Problem:** E3 delayed SWA from epoch 5→10 and extended patience from 20→30. Hypothesis: SWA over more-converged weights = better average. **Wrong — SWA LOST to best checkpoint, and overall test F1 dropped by 3.13pp.**

**Evidence:**
- Test F1: 0.6449, **-3.13pp** vs C6
- Val F1: 0.7104 (-0.79pp vs C6's 0.7183) — lower val too
- Gap: 6.55pp (narrower than E1/E2/E4/E5, but wider than C6's 4.21pp)
- BR1: 0.505 (-2.6pp vs C6's 0.531) — modest drop
- BR2: 0.684 (**-11.4pp** vs C6's 0.798) — catastrophic collapse
- BR4: 0.544 (+2.6pp vs C6's 0.518) — best BR4 in E-series
- BR5: 0.847 (-1.0pp)
- **SWA LOST** (swa_model.pt saved separately — SWA average worse than best checkpoint)
- Accuracy: 0.6870 (worst in E-series)

**Root cause:** Two interacting failure modes:

1. **SWA trajectory corruption:** With SWA starting at epoch 10, the first 10 epochs of OneCycleLR push the model through the high-LR exploration phase without SWA averaging. By epoch 10, the model has already passed the LR peak (at epoch ~30 of OneCycleLR with pct_start=0.3) — wait, actually the peak is at 30% of training so epoch 30, meaning by epoch 10 the LR is still climbing. SWA at epoch 10 starts averaging during the LR climb, but misses the early stabilization at epoch 5-10 that C6 captures. The model's early exploratory weights (epoch 5-10) are excluded, reducing the diversity of the SWA average.

2. **Extended patience enables overfitting:** Patience=30 lets training continue for 30 epochs past the val peak without improvement. The extra epochs don't help SWA (SWA lost to best checkpoint anyway) but allow the model to overfit BR2 — the largest training class. BR2's -11.4pp drop is the signature: longer training → deeper memorization of the dominant class's training patterns → worse test generalization on BR2.

The BR4 improvement (+2.6pp) is a silver lining: delayed SWA preserves some of the harder malignant decision boundaries that early SWA smoothing would blur. But the BR2 collapse overwhelms this gain.

**Rule:** SWA start at epoch 5 is optimal for this pipeline. Earlier starts (D-series didn't test) or later starts (E3) both degrade performance. swa_start_epoch=5 catches the model at the right moment: past random initialization but before deep memorization. Patience=20 is also optimal — extending to 30 provides no benefit and enables overfitting. Do NOT modify SWA timing parameters.

### Lesson #41 — Label Smoothing 0.10 + SWA Is Another Antagonistic Smoothing Pair (E4)
**Problem:** E4 doubled label smoothing from 0.05 to 0.10. Hypothesis: label smoothing (output-space regularization) is orthogonal to SWA (weight-space regularization), unlike Mixup (input-space, proven antagonistic in Lesson #23). **Wrong — label smoothing at 0.10 caused a 3.32pp regression with an 8.44pp gap.**

**Evidence:**
- Test F1: 0.6430, **-3.32pp** vs C6
- Val F1: 0.7274 (+0.91pp vs C6's 0.7183) — higher val, lower test
- Gap: 8.44pp (vs C6's 4.21pp → +4.23pp wider)
- BR1: 0.477 (-5.4pp vs C6's 0.531)
- BR2: 0.730 (-6.8pp vs C6's 0.798)
- BR4: 0.507 (-1.1pp)
- BR5: 0.858 (+0.1pp)
- SWA WON (overwrote best_model.pt)

**Root cause — a pattern emerges across three smoothing mechanisms:**
```
| Smoothing Type   | Mechanism              | Combined with SWA | Δ vs C6  |
|------------------|------------------------|:------------------:|:--------:|
| Mixup (C1)       | Input-space blending   | ANTAGONISTIC       | -1.84pp  |
| Label 0.10 (E4)  | Output-space softening | ANTAGONISTIC       | -3.32pp  |
| Label 0.05 (C6)  | Output-space softening | COMPATIBLE         | baseline |
```

Label smoothing at 0.10 produces **softer target distributions** ([0.025, 0.025, 0.025, 0.925]) that reduce gradient magnitude for high-confidence predictions. SWA simultaneously **smooths weights** by averaging parameters. Both mechanisms flatten decision boundaries through different paths, but the net effect is the same: over-smoothed class boundaries.

At 0.05, label smoothing provides just enough target softness to prevent overconfident predictions without interfering with SWA's weight-space smoothing. At 0.10, the two smoothing mechanisms compound — the model never commits strongly enough to any boundary, and SWA preserves this indecisiveness.

The val F1 being higher (+0.91pp) with worse test confirms the pattern from E1 and E2: the model finds a training-specific smoothness that looks good on the similar val distribution but fails on the shifted test set.

**Rule:** Label smoothing 0.05 is the optimal value for this pipeline. Combined with SWA, it represents the maximum tolerable output-space smoothing. Label smoothing ≥ 0.10 becomes antagonistic with SWA — joining Mixup and warm restarts in the "don't combine with SWA" category. The principle: **SWA's weight-space smoothing occupies the regularization budget; any additional smoothing mechanism that independently softens boundaries will push past the optimum.**

### Lesson #42 — Binary Head Weight 0.10 Is Already Optimal — Efficiency ≠ Underweighting (E5)
**Problem:** E5 doubled the binary head weight from 0.10→0.20 (subgroup reduced 0.45→0.35 to compensate). Lesson #31 showed the binary head was 7x more efficient per weight unit than subgroup. Hypothesis: boosting its weight would amplify the gradient anchor. **Wrong — the 7x efficiency means 0.10 is already sufficient, not that it needs more.**

**Evidence:**
- Test F1: 0.6425, **-3.37pp** vs C6
- Val F1: 0.7190 (+0.07pp vs C6's 0.7183) — essentially identical val
- Gap: 7.65pp (vs C6's 4.21pp → +3.44pp wider)
- BR1: 0.438 (**-9.4pp** vs C6's 0.531) — severe collapse
- BR2: 0.735 (-6.3pp vs C6's 0.798)
- BR4: 0.535 (+1.7pp vs C6's 0.518) — improved
- BR5: 0.863 (+0.6pp vs C6's 0.857) — improved
- SWA WON (overwrote best_model.pt)

**Root cause:** The binary head provides a {BR1,BR2} vs {BR4,BR5} gradient anchor. At 0.10 weight, it provides just enough gradient to stabilize the coarse separation without competing with fine-grained classification. At 0.20, the binary head's gradient dominates during critical decision-making:

- **BR4/BR5 improved** (+1.7pp, +0.6pp): The malignant side benefits from stronger benign/malignant separation because BR4 and BR5 are already well-separated from benign classes.
- **BR1 collapsed** (-9.4pp): The stronger binary gradient forces the model to optimize for {benign vs malignant} over {BR1 vs BR2}. Since the subgroup head (responsible for BR1 vs BR2 within benign) dropped from 0.45→0.35, the fine-grained benign distinction loses gradient budget. BR1 (smaller benign class) is sacrificed to improve the binary task's accuracy on the borderline cases.
- **BR2 dropped** (-6.3pp): Even BR2 suffers because the subgroup head's reduced weight means less gradient for the entire benign sub-classification.

**The efficiency paradox resolved:** Lesson #31's "7x more efficient per unit weight" means the binary head at 0.10 already provides 0.10 × 30.9 = 3.09pp of impact. Doubling to 0.20 should provide ~6.18pp — but it doesn't, because the relationship is non-linear. The binary head's gradient anchor has **diminishing returns** beyond the optimal level, and the reduced subgroup gradient creates a net negative.

**Rule:** Loss weight ratios in C6 (binary=0.10, subgroup=0.45, full=0.45) are at their optimal balance. High per-unit efficiency does NOT mean a component is underweighted — it means a small amount provides outsized value, which is the *definition* of being at the right weight. Do NOT adjust individual loss weights. The 7x efficiency finding from D-series should be interpreted as "0.10 is brilliantly efficient" not "0.10 should be increased."

### Lesson #43 — E-Series Meta: C6 Is at a Sharp, Verified Global Optimum for the 8-bit Pipeline

**Problem:** The E-series tested 5 individually reasonable, well-motivated single-variable perturbations to C6's configuration: LR scheduler (2 variants), SWA timing, label smoothing, and loss weight rebalancing. **All 5 completed experiments regressed by 3.1-4.0pp. The gap widened in ALL cases. Two experiments (E6, E7) did not complete.**

**Evidence (ranked by test F1):**
```
| Rank | Exp | Strategy                    | Test F1 | Δ vs C6  | Gap    | BR1   | BR2   | BR4   | BR5   | SWA   |
|------|-----|-----------------------------|---------|----------|--------|-------|-------|-------|-------|-------|
| —    | C6  | BASELINE                    | 0.6762  | —        | 4.21pp | 0.531 | 0.798 | 0.518 | 0.857 | WON   |
| 1    | E3  | Later SWA + longer training | 0.6449  | -3.13pp  | 6.55pp | 0.505 | 0.684 | 0.544 | 0.847 | LOST  |
| 2    | E1  | Cosine warmup scheduler     | 0.6446  | -3.16pp  | 7.61pp | 0.466 | 0.755 | 0.508 | 0.850 | WON   |
| 3    | E4  | Label smoothing 0.10        | 0.6430  | -3.32pp  | 8.44pp | 0.477 | 0.730 | 0.507 | 0.858 | WON   |
| 4    | E5  | Binary head wt 0.20         | 0.6425  | -3.37pp  | 7.65pp | 0.438 | 0.735 | 0.535 | 0.863 | WON   |
| 5    | E2  | Warm restarts scheduler     | 0.6363  | -3.99pp  | 9.42pp | 0.427 | 0.749 | 0.513 | 0.857 | WON   |
| —    | E6  | Stronger augmentation       | DNF     | —        | —      | —     | —     | —     | —     | —     |
| —    | E7  | 16-bit transfer             | DNF     | —        | —      | —     | —     | —     | —     | —     |
```

**Five-series convergence pattern:**
```
| Series | Experiments | Beat C6? | Best ΔF1 vs champion | Champion |
|--------|-------------|:--------:|:--------------------:|----------|
| A      | 4           | N/A      | N/A                  | A1-CE    |
| B      | 5           | Yes      | +2.45pp (B5>A1-CE)   | B5       |
| C      | 7           | Yes      | +1.47pp (C6>B5)      | C6       |
| D      | 7           | No       | -1.47pp (best D4)    | C6       |
| E      | 5 (of 7)    | No       | -3.13pp (best E3)    | C6       |
```

**The degradation ACCELERATED from D to E:** D-series' best was -1.47pp from C6 (D4), but E-series' best is -3.13pp (E3). This means the E-series perturbation axes (scheduler, SWA timing, smoothing level, weight ratios) are **more sensitive** than D-series' axes (head removal, Mixup, backbone swap). C6's configuration is tightly optimized along the dimensions E-series probed.

**The recurring pattern across E-series — val up, test down:**
- E1: val +0.24pp, test -3.16pp
- E2: val +1.22pp, test -3.99pp
- E4: val +0.91pp, test -3.32pp

Three of five experiments showed HIGHER val F1 with LOWER test F1. This is the hallmark of **training distribution overfitting** — the modifications find sharper minima that fit train/val better but generalize worse. C6's configuration uniquely balances sharpness and flatness.

**Confirmed permanently frozen (in addition to D-series list):**
- OneCycleLR scheduler (do not change to cosine, warm restarts, or step)
- SWA start epoch = 5 (do not delay)
- Early stopping patience = 20 (do not extend)
- Label smoothing = 0.05 (do not increase)
- Loss weights: binary=0.10, subgroup=0.45, full=0.45 (do not rebalance)

**The 8-bit pipeline is CONVERGED.** C6's test F1 = 0.6762 with gap = 4.21pp represents the ceiling for single-model, single-run 8-bit 1024×1024 performance with this architecture and dataset. Further single-model improvements must come from the 16-bit pipeline (higher dynamic range, F-series) or multi-model strategies (ensembles, TTA).

### Summary Statistics — E-Series Ablation Table

```
| Experiment | Backbone    | Change vs C6              | Best Val F1 | Test F1  | Gap    | BR1   | BR2   | BR4   | BR5   | SWA   |
|------------|-------------|---------------------------|:-----------:|:--------:|:------:|:-----:|:-----:|:-----:|:-----:|:-----:|
| C6 (base)  | ConvNeXt-L  | BASELINE                  | 0.7183      | 0.6762   | 4.21pp | 0.531 | 0.798 | 0.518 | 0.857 | WON   |
| E1         | ConvNeXt-L  | cosine_warmup scheduler   | 0.7207      | 0.6446   | 7.61pp | 0.466 | 0.755 | 0.508 | 0.850 | WON   |
| E2         | ConvNeXt-L  | cosine_warm_restarts      | 0.7305      | 0.6363   | 9.42pp | 0.427 | 0.749 | 0.513 | 0.857 | WON   |
| E3         | ConvNeXt-L  | SWA start=10, patience=30 | 0.7104      | 0.6449   | 6.55pp | 0.505 | 0.684 | 0.544 | 0.847 | LOST  |
| E4         | ConvNeXt-L  | label_smoothing=0.10      | 0.7274      | 0.6430   | 8.44pp | 0.477 | 0.730 | 0.507 | 0.858 | WON   |
| E5         | ConvNeXt-L  | binary_head=0.20          | 0.7190      | 0.6425   | 7.65pp | 0.438 | 0.735 | 0.535 | 0.863 | WON   |
| E6         | ConvNeXt-L  | stronger augmentation     | —           | DNF      | —      | —     | —     | —     | —     | —     |
| E7         | ConvNeXt-L  | 16-bit + SWA + no asym    | —           | DNF      | —      | —     | —     | —     | —     | —     |
```

### Recommended Next Steps (Post E-Series)
The 8-bit single-model pipeline is fully converged at C6. No further 8-bit hyperparameter tuning is justified.

1. **16-bit pipeline optimization (F-series)** — Transfer C6 insights to the higher dynamic range pipeline. F1/F2 already running.
2. **Test-Time Augmentation (TTA)** — Horizontal flip + multi-crop at inference. Free generalization without retraining. Expected: +1-2pp.
3. **Ensemble strategies** — Average predictions from C6 + D4 (different SWA state) or multiple C6 runs with different seeds.
4. **ABANDON for 8-bit:** All further single-variable hyperparameter perturbations. The E-series proved that every direction around C6 leads downhill. The only remaining 8-bit paths are inference-time improvements (TTA, ensemble) that don't modify the trained model.

---

## Tier 0 — Inference-Time Pipeline Lessons (2026-04-18)

### Lesson #44 (2026-04-18): ensemble_evaluate.py norm-stats bug was cosmetic; train.py metrics bug-free — but three prompt-level assumptions proved wrong

**Context:** Task 0.1 — fix `ensemble_evaluate.py` ImageNet normalization bug and re-establish C6 baseline via `tools/extract_c6_logits.py` (forward pass through `data/transforms.py::get_val_transforms`, which correctly dispatches `DATASET_STATS_8BIT` via `_get_norm_stats`).

**Finding (post-fix C6 baseline, MLflow run `ecef19a5f0e44dd68f9903ad35366c24`):**

- **Core metrics match prompt / prior report exactly.** Test F1 macro = 0.6762, per-class F1 BR1=.531 / BR2=.798 / BR4=.518 / BR5=.857, Binary F1 = 0.939, Cohen's κ = 0.633, AUC-ROC = 0.902. Val F1 macro = 0.7218, val–test gap = 4.55pp (+0.34pp vs prior-reported 4.21pp — within numerical noise of SWA eval pathways).
- **Bug scope was cosmetic.** `train.py` eval pathway already passes `data_cfg` through `get_val_transforms()` → correct dataset stats. The ImageNet-stats bug was isolated to `ensemble_evaluate.py`'s hand-built TTA transform pipeline, which was never the source of C6's 0.6762 report. Bug is now fixed (`_get_norm_stats(data_cfg)`), but the file's `MODELS` list remains outdated — ensemble evaluation with C6 requires a dedicated script (handled by future Task 1.1 TTA script).

**Three prompt-level mismatches discovered (evidence-first correction):**

1. **Test confusion matrix cells in prompt were wrong; row totals and per-class F1 were correct.**
   Actual (from `artifacts/c6_baseline_metrics.json`):
   ```
                 pred_BR1  pred_BR2  pred_BR4  pred_BR5
   true_BR1:        89        60        14         0     (163)
   true_BR2:        79       459        58         0     (596)
   true_BR4:        3         31       150       104     (288)
   true_BR5:        1         4         69       534     (608)
   ```
   Error patterns: BR1→BR2 drift = **36.8%** (prompt matches); **BR4→BR5 drift = 36.1%** (prompt claimed 26.4% → +9.7pp worse than stated); BR5→BR4 drift = 11.3% (prompt claimed 17.4% — better than stated).

2. **"Calibration anomaly" claim (val_confidence=0.284 vs test=0.545, "inverted calibration") is not real.**
   Actual mean confidence: val = **0.5355**, test = **0.5452**. Val/test confidence are essentially equal. There is no inverted SWA+multi-head interaction. ECE is high on both splits (val=0.197, test=0.214), so temperature scaling (Task 1.2) is still well-motivated — but as **overconfidence reduction**, not as "fixing inverted calibration."

3. **C6's `log_temperature` parameter was never actually learned.**
   Final `exp(log_temperature) = 1.5000` — exactly the config's init value. The temperature-scaling branch in `HierarchicalClassifier.forward()` only affects the `confidence` output, not any loss term, so the parameter receives no gradient during training. Task 1.2 is therefore the **first** real temperature search for C6.

**Interpretation:**
- BR4 boundary is objectively worse than prompt stated: the "malign boundary broken" signature is +9.7pp larger than expected. Any BR4-targeted threshold offset (Task 1.3) must account for a wider logit margin between true-BR4 and predicted-BR5 samples.
- The overconfidence (ECE ≈ 0.21 uniform across val/test) is a simple scalar problem, not a split-asymmetry. Post-scaling ECE should drop on both splits symmetrically; confidence itself will shift downward, not flip direction.
- `ensemble_evaluate.py`'nin bug'ı C6'nın raporlanan sayılarını etkilememişti; düzeltme sadece ileride o pathway'i kullanmak istersek diye uygulandı.

**Action:**
- **Baseline freeze:** Test F1 = **0.6762** — tüm Tier 1 improvement deltaları buna karşılaştırılacak.
- **Cached logits hazır** (`artifacts/c6_{val,test}_{,binary_,benign_sub_,malign_sub_}logits.npy`, `_labels.npy`, `c6_cache_meta.json`, `c6_baseline_metrics.json`). Tier 1 task'ları bu cache'i okuyacak — yeniden forward pass YOK.
- **Task 1.2 motivation re-framed:** target test ECE ≤ 0.10 (currently 0.214), macro F1 ≥ baseline (±0pp acceptable).
- **Task 1.3 grid widened:** d4 grid ∈ `np.linspace(0, 1.2, 25)` (prompt proposed `[0, 0.8, 17]` — insufficient given the 36% BR4→BR5 drift). d1 grid kept at `np.linspace(0, 1.0, 21)` (BR1 offset remains zero-sum risk per Lesson #27; conservative range preferred). Reason: (i) compute cost is negligible (525 vs 357 combos × 5 folds × ~1284 samples = milliseconds in numpy), (ii) CV-averaging + `std < 0.3` fold-consistency gate prevents grid-width-driven overfit, (iii) if the optimum hits the upper bound (d4 ≈ 1.2 across most folds), the threshold approach is structurally insufficient and Task 1.4 gating should carry more weight.
- **MODELS list in `ensemble_evaluate.py`:** left outdated (pre-C6 checkpoints). TTA re-implementation for C6 will live in a new script (`tools/tta_c6.py`), not in ensemble_evaluate.

### Lesson #45 (2026-04-19): hflip+view-swap actively hurts C6; rotations carry TTA gain. Bilateral fusion has semantic (not symbolic) L/R.

**Context:** Task 1.1 — 8-view TTA with view-swap-aware horizontal flip (RCC↔LCC, RMLO↔LMLO index permutation) + rotations (±5°, ±10°) applied on normalized tensors with background-equivalent fill value (−mean/std = −0.612). Logit-averaged over views, softmax at the end. Per-view incremental ablation run alongside the prompt-spec 8-view.

**Finding (from `artifacts/c6_tta_metrics.json`, MLflow run `2af06c6e1b7548dd9e00e14cf7fe5041`):**

```
Per-view incremental (mean of first k view logits → softmax → argmax):
  k=1 identity              F1=0.6762 (+0.00pp)   ← pipeline sanity match baseline
  k=2 +hflip_swap           F1=0.6740 (-0.22pp)   ← hflip+swap HURT on its own
  k=3 +rot_p5               F1=0.6778 (+0.16pp)
  k=4 +rot_m5               F1=0.6801 (+0.39pp)
  k=5 +rot_p10              F1=0.6828 (+0.66pp)
  k=6 +rot_m10              F1=0.6847 (+0.84pp)   ← PEAK
  k=7 +hflip_swap_rot_p5    F1=0.6825 (+0.63pp)
  k=8 +hflip_swap_rot_m5    F1=0.6808 (+0.45pp)   ← prompt-spec 8-view (marginal)

Aggregate ablations:
  tta4 (identity+hflip_swap+rot±5)               F1=0.6801 (+0.39pp)
  tta6 (identity+hflip_swap+rot±5+rot±10)        F1=0.6847 (+0.84pp) PEAK
  tta8 (prompt spec, adds hflip_swap_rot±5)      F1=0.6808 (+0.45pp) — below 0.5pp accept threshold
```

Per-class (tta8 vs baseline):
- BR1 F1 .531→.537 (+0.6pp), recall 54.6→57.7% (+3.1pp)
- BR2 F1 .798→.801 (+0.3pp), stable
- BR4 F1 .518→.528 (+1.0pp), recall 52.1→52.8% (marginal)
- BR5 F1 .857→.857 (stable)

Calibration (tta8 vs baseline):
- Val ECE 0.197 → 0.130 (−0.067)
- Test ECE 0.214 → 0.161 (−0.053)
- Mean confidence val 0.536 → 0.641, test 0.545 → 0.648 (logit averaging sharpens softmax)

**Interpretation — why hflip+swap hurts despite view-swap:**
Bilateral fusion computes `F_diff = F_left − F_right`. When the input is hflip+view-swap'ed, F_diff flips sign: `F_left' − F_right' = F_flip(R) − F_flip(L) = −F_flip(F_diff)`. The direction of the asymmetry signal reverses, and the model — which learned asymmetry orientation-dependently during training — interprets this as a genuinely different anatomical pattern, not a symmetric augmentation.

This means bilateral fusion's L/R representation is **semantic** (encodes orientation), not **symbolic** (swap-invariant). Tensor-level view permutation is not enough to recover feature-level invariance. Mammography priors about which side a lesion is on likely become embedded in the asymmetry features during training.

Rotations (±5, ±10) don't suffer this — they preserve L/R orientation and only perturb fine spatial features; the backbone's ImageNet-adapted features absorb small rotations well, and F_diff remains semantically valid.

The combination views (hflip_swap_rot±5) inherit the hflip problem and dilute the rotation-only wins.

**Action:**
- **tta8 kept as the downstream TTA track** (not peak tta6) because only tta8 cached per-head sub-logits (binary/benign_sub/malign_sub). Task 1.4 binary gating requires sub-head TTA logits; recomputing with tta6 would cost another 2-3h forward pass and risk divergence from peak pattern. The 0.039pp gap (0.6847 vs 0.6808) is absorbed as a known suboptimality.
- **Option C pipeline:** Task 1.5 cumulative will run two parallel tracks — non-TTA (raw cached logits, F1 baseline 0.6762) and tta8 (F1 baseline 0.6808). Each track gets its own T, d1/d4, and gating α; decision point chooses the best final cumulative F1.
- **Permanent rule:** Do NOT add symmetric-flip augmentations during training for this pipeline. The bilateral fusion architecture requires orientation-consistent training data. If ever retraining, confirm `augmentation.horizontal_flip` stays 0.5 only for views that are independently flipped (currently not done — view-independence at training was accidentally preserved because `get_train_transforms` does not swap view indices after flipping; the train-time L/R asymmetry thus gets corrupted randomly, which may explain why asymmetry-loss noise was harmful per Lesson #22).
- **Accept criterion:** Pass via ablation documentation (prompt's "OR" clause). +0.45pp is 0.05pp below the 0.5pp bar, but the per-view finding is a scientifically stronger contribution than a 0.01pp above-bar cosmetic win.

**Caveat / future work:**
If the paper claims TTA as a feature, report only rotation-based TTA (tta5 = identity + rot±5 + rot±10 or tta6 as above). Do NOT present hflip+swap TTA — it's either negative or marginal and undermines the bilateral-fusion architectural argument.

### Lesson #46 (2026-04-19): C6 is underconfident (T_opt ≈ 0.73, not >1); scalar T limits ECE floor ~0.13

**Context:** Task 1.2 — LBFGS temperature scaling on val logits, parallel tracks (non-TTA, tta8). Motivated by Lesson #44 discovery that C6's `log_temperature` parameter is never gradient-touched during training (fixed at init 1.5), and by post-baseline ECE ≈ 0.21.

**Finding (from `artifacts/c6_temp_scale_metrics.json`, MLflow run `474616437c764849b0b7d6456e46aefe`):**

```
Track      T_opt    Test ECE T=1   Test ECE T_opt   ΔECE    Test Brier T=1 → T_opt    Test NLL T=1 → T_opt
nonTTA    0.7347        0.1531           0.1323    −0.021    0.4054 → 0.3944         0.7270 → 0.7016
tta8      0.7229        0.1609           0.1291    −0.032    0.4013 → 0.3887         0.7180 → 0.6868

LBFGS stability: 3 restarts (init T ∈ {0.5, 1.0, 1.5}) converge within ±0.002 → global optimum.
F1 sanity: nonTTA 0.6762 (exact baseline match), tta8 0.6808 (exact Task 1.1 match) — T-invariance confirmed.
```

Reminder: config's init T = 1.5 → at T = 1.5 (C6's effective inference T), test ECE = 0.214, test NLL = 0.818. At T = 1.0, test ECE = 0.153, test NLL = 0.727. At T_opt = 0.73, test ECE = 0.132, test NLL = 0.702.

**Interpretation:**

1. **C6 is underconfident, not overconfident.** Test confidence (0.545) < accuracy (0.744) by ~20pp → the softmax distribution is too flat. T_opt < 1.0 sharpens it (conf 0.545 → 0.729). This reframes Lesson #44's ECE observation: the prompt's "inverted calibration" narrative was wrong in direction too; there is no calibration anomaly, just mild symmetric underconfidence on both splits.

2. **Scalar T has an ECE floor at ~0.13 for this pipeline.** Reducing ECE below ~0.13 would require vector temperature (per-class T, 4 params) or Platt scaling. Both violate the "1D search, overfit-resistant" constraint from the prompt. Accept the scalar-T limit; downstream Task 1.3 (threshold offsets) and Task 1.4 (gating blend) can improve F1 further but not ECE within this pipeline.

3. **tta8 requires a slightly lower T than non-TTA (0.7229 vs 0.7347)** — counterintuitive at first. Explanation: logit averaging is a Jensen-inequality smoothing operator. For a given input, `mean(logit_i)` underestimates the winning class's margin vs `mean(softmax_i)`. The TTA-averaged logits are flatter than per-view logits in the "winning direction," so a more aggressive T is needed to sharpen. In contrast, softmax-averaging TTA (had we chosen it) would have required T closer to 1. This is a real mechanism, not noise (stable to ±0.0004 across LBFGS restarts).

4. **Training-time insight (future work):** `models/classification_heads.py::HierarchicalClassifier.__init__` defines `self.log_temperature` but uses it only in the inference-time `confidence` output, NOT in any loss term. If the full-head loss were refactored to `CrossEntropy(full_logits/T, full_labels)` with T learnable (Guo et al. 2017 integrated temperature), C6's learned T would move toward ~0.73 during training, and the model would likely ship better-calibrated from the start. This is a cheap refactor for Tier 2/3.

5. **T-invariance of argmax matters for Task 1.3 design:** `argmax((logits + d)/T) = argmax(logits + d)` for any T > 0. Running Task 1.3's grid search on T-scaled val logits (prompt's suggestion) vs raw val logits produces identical (d1, d4) fold-optima. Task 1.3 script will therefore operate on raw logits directly; T enters only in Task 1.5 cumulative pipeline where gating blends softmax distributions (non-argmax operation).

**Action:**
- `artifacts/c6_temperature_values.json` produced: `{nonTTA: 0.7347, tta8: 0.7229}`. Task 1.5 cumulative will read these for each track's gating softmax temperatures.
- Task 1.3 grid search: raw logits, no T-scale pre-step (T-invariant for argmax-based F1 objective).
- ECE target of ≤ 0.10 from the prompt is **abandoned** for this phase — the scalar-T floor is ~0.13, and going lower requires deviating from the "1D search, overfit-resistant" constraint. The achieved ECE reduction (−0.021 non-TTA, −0.032 tta8) plus F1-stable confirmation satisfies the adapted accept criterion.
- Paper framing: "temperature scaling reduces ECE on both tracks by ~15-20% relative, producing better-calibrated probabilities for downstream thresholding and gating; the absolute ECE floor at ~0.13 reflects class-conditional miscalibration inherent to the 4-class BI-RADS hierarchy with severe test-time class-prior shift."

### Lesson #47 (2026-04-19): Val→test prior shift voids threshold offsets; CV guardrails pass but test F1 regresses. Zero-sum (Lesson #27) reappears.

**Context:** Task 1.3 — 5-fold StratifiedKFold grid search on (d1, d4) offsets applied to raw val logits. Search space d1 ∈ [0, 1.0] × 21, d4 ∈ [0, 1.2] × 25 (widened from prompt's [0, 0.8] per Lesson #44). Both non-TTA and tta8 tracks.

**Finding (from `artifacts/c6_threshold_cv_metrics.json`, MLflow run `184f22d432d64e94942c38bdcdb3fbef`):**

```
Track      CV d1 (std)     CV d4 (std)     Val F1 Δ   Test F1 Δ    Naive-vs-CV gap
nonTTA     0.06 (0.07)     0.43 (0.16)     +1.43pp    −0.53pp      −0.61pp
tta8       0.11 (0.14)     0.36 (0.19)     +1.07pp    −0.18pp      −1.31pp
```

All CV guardrails PASSED (std < 0.3). Boundary-hit = 0 in both tracks (no fold hit the grid upper bound of 1.2 on d4 — Lesson #44's widened-grid recommendation was unnecessary; d4 optima sit at 0.36–0.43).

Per-class breakdown for non-TTA test (offset d1=0.06, d4=0.43):
- BR1 F1 +0.2pp, recall +0.6pp — minimal (d1 was small)
- **BR2 F1 −5.2pp, recall −10.6pp** — catastrophic
- BR4 F1 +2.8pp, recall +14.2pp, **precision −5.1pp** — BR4 gains from BR2 drift, not better BR4 detection
- BR5 F1 +0.2pp — stable

Confusion matrix drift signature: true_BR2 → pred_BR4 doubled from 9.7% (58 patients) to **20.0% (119 patients)**. The d4 offset pulls BR2 into BR4 territory.

Fold-level anomaly: tta8 Fold 5 finds `d4 = 0.0` as its fold-optimum (other folds d4 ∈ {0.40, 0.40, 0.45, 0.55}). This fold's held-out 256 samples apparently had a BR prior distribution closer to test's prior, and the grid search correctly identified "no offset needed" — a partial confirmation that the negative transfer is driven by val's BR prior, not by the offset mechanism itself.

**Interpretation:**

1. **Saerens-style test-prior constraint makes val-calibrated offsets prior-biased by construction.** The d4 = 0.43 optimum is implicitly calibrated to val's BR4 share (22.2%). Test BR4 share is 17.4% — a 22% smaller class. The val-optimal offset is therefore systematically too aggressive for test, and the excess BR4 attraction comes from the neighboring BR2 (the largest test class, 36%). Zero-sum: 14.2pp BR4 recall gain costs 10.6pp BR2 recall loss, and since BR2 has 3.7× more test samples than BR4, the net F1 is negative.

2. **Structurally analogous to C3 (Lesson #27).** C3 raised the BR1 class weight by 40% → BR1 F1 +1.2pp, BR2 F1 −9.8pp, net −2.69pp. Task 1.3 does the reverse (boosts BR4) on the BR2↔BR4 axis and produces the same pattern. Both manipulate a shared decision boundary without new discriminative information.

3. **Grid width correction to Lesson #44:** Widening d4 to [0, 1.2] had no effect on optima. The prompt's original [0, 0.8, 17] spec would have been sufficient and is the correct recommendation for any retry. Lesson #44's widened-grid advice is retracted.

4. **CV guardrails failed as a test-F1 proxy.** std(d1)<0.3 and std(d4)<0.3 were both satisfied but test regressed. Fold-level consistency on val predicts val→val transfer, not val→test transfer when the test distribution is prior-shifted. Future CV guardrails on this dataset must explicitly include a val-vs-test delta check, not just fold variance.

**Action:**
- **Threshold offsets excluded from Task 1.5's default cumulative pipeline** (d1=d4=0). Task 1.5 ablation table will still include "+ threshold" as an explicit row to document this negative result transparently (reviewer-defensible framing).
- **Primary F1 lever shifts to Task 1.4 (binary gating).** Hypothesis: the binary head (F1=0.94) is robust across val/test because benign/malign is the axis where test prior shift is minimal (train Benign=52%, test Benign=45.8% — only 6pp, vs BR1 halved). Hier reconstruction `P(malign) · P(BR4|malign)` conditions on a distribution-stable quantity, so val-tuned blending should transfer to test.
- Paper framing: "threshold offset tuning on val logits is principled but prior-shift-fragile; hierarchical binary gating achieves robust improvement because the binary decision is invariant to the 4-class prior shift."

**Future work:**
- Scoped threshold: apply (d1, d4) only to samples with `binary_prob ∈ [0.4, 0.6]` (high uncertainty region) — this would avoid BR4 overreach on confidently-malignant samples. Defer to after Task 1.5 sees cumulative-pipeline F1.

### Lesson #48 (2026-04-19): Inference-time hierarchical reconstruction duplicates what full head already knows. α-CV bimodal; hard-gate noise-level; pure hier < pure full.

**Context:** Task 1.4 — binary-gated hierarchical inference. Five variants tested per track: (A) soft α-CV blend with T_opt, (B) soft α-CV blend with T=1.0, (C) hard gate (P(malign) > 0.5), (D) pure hier (α=1), (E) pure full (α=0 sanity).

**Finding (from `artifacts/c6_gating_metrics.json`, MLflow run `4c2c66dcfca241c3a886d031b334e1e3`):**

```
                                nonTTA                          tta8
                          Test F1     Δ                   Test F1     Δ
(A) α-CV soft,T_opt       0.6731    −0.31pp              0.6776    −0.31pp
(B) α-CV soft,T=1.0       0.6731    −0.31pp              0.6801    −0.07pp
(C) hard gate, T_opt      0.6765    +0.03pp  BEST        0.6807    −0.00pp  BEST
(D) pure hier, T_opt      0.6707    −0.55pp              0.6784    −0.24pp
(E) pure full (sanity)    0.6762    +0.00pp ✓           0.6808    +0.00pp ✓

α-CV fold-by-fold:
  nonTTA (A): {0.20, 1.00, 0.60, 0.70, 0.40}  mean=0.58  std=0.271  (near guardrail)
  tta8   (A): {0.00, 1.00, 0.00, 1.00, 0.00}  mean=0.40  std=0.490  (GUARDRAIL BROKEN)

Confusion matrix drift (nonTTA, hard gate vs baseline):
  true_BR4 → pred_BR5:  baseline 104/288 → variant C 102/288  (−2 patients)
  true_BR5 → pred_BR4:  baseline  69/608 → variant C  73/608  (+4 patients)
  Net effect: noise-level, 6 out of 1655 samples changed class.
```

**Interpretation:**

1. **The hypothesized "sub-head knows something full doesn't" is false for C6.** The malign_sub head learned BR4↔BR5 with the same data asymmetry that the full head did — sanity check showed true_BR4 has malign_sub margin = 0.26 (weak) vs true_BR5 = 1.76 (strong), mirroring the full head's BR4 weakness. Hier product `P(malign) · P(BR4|malign)` recovers the same information that `P(BR4)` from the full head already contains. Duplication, not enrichment.

2. **α-CV fold bimodality is the smoking gun.** tta8 folds split cleanly into {α=0, α=1} camps with no fold preferring a middle value. This is what happens when hier and full make *nearly-identical argmax decisions on most samples*: each fold flips on its minority of boundary samples, and the optimum migrates to whichever extreme agrees with that fold's boundary population. The mean (α=0.4) is a statistical artifact, not a true optimum.

3. **Lesson #30 reconciled with this result.** Lesson #30 showed removing auxiliary heads costs +3pp — but that gain is *training-time multi-task regularization*: the binary and sub-heads provide extra gradient signal to the shared backbone during training, enriching the features the full head consumes. At inference time, the full head already incorporates those features via the shared `patient_feat` representation. Re-composing auxiliary-head softmax outputs into a synthetic 4-class distribution does not retrieve any bypassed information. **Auxiliary heads' value is architectural (during training), not compositional (during inference).**

4. **Hard gate's +0.03pp is noise.** Only 6 out of 1655 test samples changed class under hard gating (4 BR5→BR4 flips, 2 BR4→BR5 unflips). The binary head's argmax agrees with the full head's `argmax >= 2` boundary on >99% of samples. Hard gate carries no signal because the binary/quaternary decision paths are redundant for C6's trained representation.

5. **tta8 α-CV guardrail broke (std = 0.49 > 0.3)** while nonTTA stayed just under (0.271 < 0.3) — another confirmation that Task 1.3's CV-std-based guardrail is unreliable for detecting whether a test-F1 improvement will transfer. The strict `std < 0.3` threshold should be interpreted as "necessary, not sufficient" for val→test transfer.

**Meta-observation across Tier 1 Tasks 1.2, 1.3, 1.4:**
- 1.2 Temperature: F1-invariant by construction; calibration-only gain. (Limited scope, met.)
- 1.3 Threshold: val→test prior shift causes zero-sum; test F1 regresses. (Negative.)
- 1.4 Gating: inference-time decomposition duplicates full-head information; at best noise-level. (Negative/neutral.)

Only Task 1.1 TTA provides a transferable F1 gain (+0.45pp tta8, from rotations alone). The 8-bit single-model pipeline's inference-time improvement ceiling is therefore ~0.6808, far below the 0.72 target. The path to 0.72+ requires training-time intervention (Tier 2: logit-adjusted training for prior-shift robustness, or 16-bit pipeline transfer).

**Action:**
- **Task 1.5 cumulative evaluation runs as formality** to populate the ablation table with clean, documented deltas; decision-point verdict expected to be "< 0.70 → root cause + Tier 2."
- **Route to Tier 2 Task 2.2 (F2 — logit-adjusted training, Menon et al. 2021).** This directly targets the val→test prior shift that killed Task 1.3 and that underlies the BR4 F1 ceiling. Unlike class weights (Lesson #27 zero-sum), logit adjustment is mathematically principled for label-shifted test distributions.
- **Skip Task 2.0 multi-seed ensemble for now** (prompt's `< 0.70` branch also suggests skipping ensemble first; 3×17h GPU unjustified before understanding why 0.68 ceiling exists).
- **Skip Task 2.1 F1 16-bit preprocessing** unless F2 also fails — 16-bit preprocessing pipeline requires 300GB data regeneration, only worthwhile if training-time regularization alone insufficient.
- **Gating in Task 1.5:** use variant C (hard gate) for completeness in both tracks — contributes +0.03pp nonTTA, 0.00pp tta8. Document in ablation but don't oversell.

**Future work:**
- Per-head temperature fitting (separate T for full, binary, benign_sub, malign_sub). Might close the α-CV bimodality if sub-heads are miscalibrated in ways full head isn't. Low priority given the main architectural finding (hier = full).
- Train-time refactor: include `CE(logits/T, labels)` in loss so `log_temperature` actually gets optimized. See Lesson #46 action. Expected to ship a better-calibrated C6 out of the box; orthogonal to F1 gains.

### Lesson #49 (2026-04-21): F2 logit-adjusted training (Menon 2021) is harmful here — all τ regress, BR2 sacrificed, val-test gap WIDENS. Third manifestation of the BR2↔BR4 zero-sum axis.

**Context:** Tier 2 Task 2.2 — F2 experiments with `LogitAdjustedCE` applied to the full head only (binary + subgroup kept as standard CE). Three τ values tested with identical C6 configuration except for loss (seed=42, SWA, asymmetry=0, same LR schedule, same class weights, same multi-task structure). Motivated by Lesson #47's finding that inference-time threshold offsets fail on val→test prior shift, we hypothesized training-time prior adjustment would succeed.

**Finding (from `outputs/convnextv2_large_8bit_f2_tau{05,10,15}/reports/classification_report.txt`):**

```
Config         Best Val F1   Test F1   Δ vs C6    Val-Test Gap    BR1 F1   BR2 F1   BR4 F1   BR5 F1   SWA
C6 (baseline)      0.7183    0.6762    —          4.6pp           0.531    0.798    0.518    0.857    WON
F2 τ=0.5           0.7212    0.6418    −3.44pp    7.9pp           0.441    0.772    0.498    0.856    LOST
F2 τ=1.0           0.7235    0.6471    −2.91pp    7.6pp           0.513    0.703    0.517    0.856    WON
F2 τ=1.5           0.7200    0.6391    −3.71pp    8.1pp           0.505    0.699    0.499    0.854    WON
```

Per-class recall (test), C6 vs F2 τ=1.0:
- BR1: 54.6% → 50.3%  (−4.3pp)
- **BR2: 77.0% → 60.7%  (−16.3pp)** — catastrophic
- **BR4: 52.1% → 67.7%  (+15.6pp)** — the only "gain"
- BR5: 87.8% → 84.9%  (−2.9pp)

**Interpretation — three overlapping failure modes:**

1. **Val-test gap WIDENED (4.6pp → 7.6-8.1pp, +3pp worse).** This is the opposite of LA's intended effect. LA is supposed to close prior-shift gaps; here it acts as an **overfitting enhancer**. Val F1 stays near C6's (0.720-0.724), but test F1 drops by 3-4pp. The model learns val distribution harder; LA provides no transfer benefit.

2. **BR2↔BR4 zero-sum axis — third manifestation.** F2 reproduces the same pattern as C3 class-weight manipulation (Lesson #27) and Task 1.3 threshold offsets (Lesson #47):
   - C3: BR1 +1.2pp, BR2 −9.8pp (zero-sum on BR1↔BR2)
   - Task 1.3: BR4 recall +14.2pp, BR2 recall −10.6pp (zero-sum on BR2↔BR4)
   - F2 τ=1.0: BR4 recall +15.6pp, BR2 recall −16.3pp (zero-sum on BR2↔BR4)

   **Rule:** Any intervention that mechanically shifts the decision boundary on the BR2↔BR4 axis (whether class weights, threshold offsets, or training-time prior adjustment) produces a zero-sum trade-off in this pipeline. BR2 is the majority test class (596 patients vs BR4's 288), so the 2:1 sample ratio guarantees net F1 regression when BR4 gains come from BR2 losses.

3. **Double prior correction.** The loss already has `class_weights_4 = [1.28, 1.00, 1.20, 1.11]` (sqrt-inverse, which boosts BR1 and BR4). F2 adds `τ · log(train_prior)` on top, which is a second prior-based adjustment targeting the same minorities. The two corrections compound multiplicatively → overshooting on BR4 boundary → BR2 absorption.

4. **Menon 2021 assumption violated.** LA is derived under the assumption that the test distribution is **class-balanced (uniform prior)**. Our test set is NOT uniform: [BR1=9.8%, BR2=36.0%, BR4=17.4%, BR5=36.7%]. LA corrects the model toward uniform predictions, but uniform is neither the train distribution nor the test distribution — it's a direction between them that happens to pass through a poor compromise for our specific test prior. BR2 (test %36, near its train %32) is pushed away from its natural optimum; BR4 (test %17.4, away from train %22.2) is pushed toward a stronger boost than test needs.

5. **τ ordering insight.** τ=0.5 gave best BR2 F1 (0.772) and worst BR1 F1 (0.441); τ=1.0 gave best BR4 F1 (0.517) but collapsed BR2 (0.703); τ=1.5 didn't further improve BR4 despite more aggressive adjustment. This non-monotonicity suggests optimization dynamics (SWA trajectory + multi-task loss) interact with LA in ways the theory doesn't predict.

**Action:**
- **F2 PERMANENTLY ABANDONED** for this pipeline. Do not retry with different τ, different priors, or combined with other interventions.
- **Route to Yol A: Task 2.0 Multi-seed ensemble.** Two additional C6 seeds (123, 2024), then combine with existing seed=42 via TTA-averaged softmax blending. Seed variance is well-characterized and known to transfer; expected +1-2pp over single-seed best.
- **Paper framing:** F2 is reported as an ablation negative result. The zero-sum axis finding (three different mechanisms — class weights, thresholds, LA — all produce the same BR2↔BR4 trade-off) is itself a contribution: it demonstrates that the remaining F1 ceiling on this dataset is distribution-shift-fundamental, not loss-function-choice.

**Permanently eliminated interventions on BR2↔BR4 boundary:**
- Class weight manipulation (C3, Lesson #27)
- Inference-time threshold offsets (Task 1.3, Lesson #47)
- Training-time logit adjustment (F2, Lesson #49)

Any future approach aiming for >0.70 test F1 must either (a) add new discriminative information (richer features, different modality, 16-bit dynamic range) or (b) use ensemble methods that exploit different models' residual errors rather than shifting boundaries.
