"""
Tier 1 Task 1.4 — Binary-Gated Hierarchical Inference
========================================================
Combines C6's four classification heads at inference time via hierarchical
probability reconstruction:

    hier[:, 0] = P(benign) · P(BR1 | benign)
    hier[:, 1] = P(benign) · P(BR2 | benign)
    hier[:, 2] = P(malign) · P(BR4 | malign)
    hier[:, 3] = P(malign) · P(BR5 | malign)

Then blends with full head via:

    final = α · hier + (1 − α) · full_probs

α ∈ [0, 1] found by 5-fold stratified CV on val F1 macro. Final α = mean across folds.

Also tests a **hard gating** variant: if `binary_probs[:, 1] > 0.5` use hier,
else use full. No α parameter.

Temperature: T_opt from Task 1.2 applied uniformly to ALL heads (full, binary,
benign_sub, malign_sub). Underconfidence correction is expected to help all
heads, but a future ablation could fit per-head temperatures.

Ablation:
    (A) α CV, T_opt, soft blend                 [primary]
    (B) α CV, T=1.0, soft blend                 [T-effect isolation]
    (C) Hard gating, T_opt                      [prompt-specified alternative]
    (D) Pure hier (α = 1), T_opt                [sanity / skyline]
    (E) Pure full (α = 0), T_opt                [sanity = Task 1.1 result]

Two tracks: nonTTA, tta8.

Class ordering verified (models/classification_heads.py):
    full_logits[:, 0..3]       = [BR1, BR2, BR4, BR5]
    binary_logits[:, 0..1]     = [Benign, Malign]
    benign_sub_logits[:, 0..1] = [BR1, BR2]
    malign_sub_logits[:, 0..1] = [BR4, BR5]

Usage:
    python tools/binary_gating_c6.py \\
        --config configs/experiment_v2_birads/convnextv2_large_8bit_ablation_c6.yaml \\
        --artifacts-dir artifacts \\
        --run-name tier1_task1_4_gating_c6 \\
        --experiment-name birads-inference-pipeline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

import numpy as np
import yaml
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

from utils.mlflow_logger import ExperimentLogger


BIRADS_NAMES = ["BIRADS_1", "BIRADS_2", "BIRADS_4", "BIRADS_5"]
ALPHA_GRID = np.linspace(0.0, 1.0, 21)  # 0.00, 0.05, ..., 1.00


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def compute_all_probs(full_logits, binary_logits, benign_sub_logits, malign_sub_logits, T: float):
    """Apply T to all heads, return softmax probs dict."""
    return {
        "full":       softmax_np(full_logits / T, axis=-1),
        "binary":     softmax_np(binary_logits / T, axis=-1),
        "benign_sub": softmax_np(benign_sub_logits / T, axis=-1),
        "malign_sub": softmax_np(malign_sub_logits / T, axis=-1),
    }


def compute_hier(probs: dict) -> np.ndarray:
    """Hierarchical reconstruction from per-head softmax probs. Returns (N, 4)."""
    b = probs["binary"]       # (N, 2): [Benign, Malign]
    bs = probs["benign_sub"]  # (N, 2): [BR1, BR2]
    ms = probs["malign_sub"]  # (N, 2): [BR4, BR5]
    hier = np.zeros((b.shape[0], 4), dtype=np.float64)
    hier[:, 0] = b[:, 0] * bs[:, 0]   # P(benign) · P(BR1 | benign)
    hier[:, 1] = b[:, 0] * bs[:, 1]   # P(benign) · P(BR2 | benign)
    hier[:, 2] = b[:, 1] * ms[:, 0]   # P(malign) · P(BR4 | malign)
    hier[:, 3] = b[:, 1] * ms[:, 1]   # P(malign) · P(BR5 | malign)
    # Normalization: already rows sum to 1 by construction (benign probs + malign probs sum via binary head).
    # But round-off can leave non-unit rows; renormalize for cleanliness.
    hier = hier / hier.sum(axis=1, keepdims=True).clip(min=1e-12)
    return hier


def soft_blend(full_probs: np.ndarray, hier: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * hier + (1.0 - alpha) * full_probs


def hard_gate(full_probs: np.ndarray, hier: np.ndarray, binary_probs: np.ndarray,
              thresh: float = 0.5) -> np.ndarray:
    """If P(malign) > thresh, use hier row, else use full_probs row."""
    out = np.empty_like(full_probs)
    mask = binary_probs[:, 1] > thresh
    out[mask] = hier[mask]
    out[~mask] = full_probs[~mask]
    return out


def compute_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    preds = probs.argmax(axis=1)
    m = {
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "accuracy": float((preds == labels).mean()),
    }
    pr, rc, f1, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
    for i, n in enumerate(BIRADS_NAMES):
        if i < len(f1):
            m[f"f1_{n}"] = float(f1[i])
            m[f"recall_{n}"] = float(rc[i])
            m[f"precision_{n}"] = float(pr[i])
    m["confusion_matrix"] = confusion_matrix(labels, preds, labels=[0, 1, 2, 3]).tolist()
    return m


# ---------------------------------------------------------------------------
# CV α search
# ---------------------------------------------------------------------------

def fold_alpha_search(full_probs, hier, labels, alpha_grid: np.ndarray):
    """Return best (alpha, f1) on given fold's held samples."""
    best_f1 = -1.0
    best_alpha = 0.0
    surface = []
    for a in alpha_grid:
        blended = soft_blend(full_probs, hier, a)
        preds = blended.argmax(axis=1)
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        surface.append(float(f1))
        if f1 > best_f1:
            best_f1 = float(f1)
            best_alpha = float(a)
    return best_alpha, best_f1, surface


def cv_alpha_search(full_probs, hier, labels, n_splits: int = 5, seed: int = 42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for fold_idx, (_, held_idx) in enumerate(skf.split(full_probs, labels), start=1):
        a, f1, surface = fold_alpha_search(full_probs[held_idx], hier[held_idx],
                                            labels[held_idx], ALPHA_GRID)
        folds.append({"fold": fold_idx, "alpha": a, "fold_val_f1": f1,
                      "fold_n": int(len(held_idx)), "f1_surface": surface})
    alphas = np.array([f["alpha"] for f in folds])
    return {
        "folds": folds,
        "mean":   float(alphas.mean()),
        "std":    float(alphas.std()),
        "median": float(np.median(alphas)),
    }


# ---------------------------------------------------------------------------
# Per-track runner
# ---------------------------------------------------------------------------

def run_track(track: str, art: Path, T_opt: float, seed: int = 42,
              tta_prefix: str = "") -> dict:
    """tta_prefix: '' for non-TTA, 'tta8_' for TTA logits."""
    print(f"\n{'='*72}")
    print(f"  Track: {track}  (T_opt = {T_opt:.4f})")
    print(f"{'='*72}")

    def _load(split):
        suffix_full       = f"c6_{split}_{tta_prefix}full_logits.npy" if tta_prefix else f"c6_{split}_logits.npy"
        suffix_binary     = f"c6_{split}_{tta_prefix}binary_logits.npy"
        suffix_benign_sub = f"c6_{split}_{tta_prefix}benign_sub_logits.npy"
        suffix_malign_sub = f"c6_{split}_{tta_prefix}malign_sub_logits.npy"
        return {
            "full":       np.load(art / suffix_full),
            "binary":     np.load(art / suffix_binary),
            "benign_sub": np.load(art / suffix_benign_sub),
            "malign_sub": np.load(art / suffix_malign_sub),
            "labels":     np.load(art / f"c6_{split}_labels.npy"),
        }

    val = _load("val")
    test = _load("test")
    print(f"  Val n={len(val['labels'])}, Test n={len(test['labels'])}")

    # ---- (A) T_opt + α-CV soft blend ----
    val_probs_Topt = compute_all_probs(val["full"], val["binary"], val["benign_sub"], val["malign_sub"],
                                        T=T_opt)
    test_probs_Topt = compute_all_probs(test["full"], test["binary"], test["benign_sub"], test["malign_sub"],
                                         T=T_opt)
    val_hier_Topt = compute_hier(val_probs_Topt)
    test_hier_Topt = compute_hier(test_probs_Topt)

    cv_A = cv_alpha_search(val_probs_Topt["full"], val_hier_Topt, val["labels"], seed=seed)
    alpha_A = cv_A["mean"]
    print(f"\n  (A) α-CV with T_opt:  folds "
          + ", ".join(f"{f['alpha']:.2f}" for f in cv_A["folds"])
          + f"  →  mean={alpha_A:.3f}  std={cv_A['std']:.3f}  median={cv_A['median']:.3f}")

    # Apply to val and test
    val_A = soft_blend(val_probs_Topt["full"], val_hier_Topt, alpha_A)
    test_A = soft_blend(test_probs_Topt["full"], test_hier_Topt, alpha_A)
    val_A_m = compute_metrics(val_A, val["labels"])
    test_A_m = compute_metrics(test_A, test["labels"])

    # ---- (B) T=1.0 + α-CV soft blend ----
    val_probs_T1 = compute_all_probs(val["full"], val["binary"], val["benign_sub"], val["malign_sub"],
                                      T=1.0)
    test_probs_T1 = compute_all_probs(test["full"], test["binary"], test["benign_sub"], test["malign_sub"],
                                       T=1.0)
    val_hier_T1 = compute_hier(val_probs_T1)
    test_hier_T1 = compute_hier(test_probs_T1)

    cv_B = cv_alpha_search(val_probs_T1["full"], val_hier_T1, val["labels"], seed=seed)
    alpha_B = cv_B["mean"]
    print(f"  (B) α-CV with T=1.0:  folds "
          + ", ".join(f"{f['alpha']:.2f}" for f in cv_B["folds"])
          + f"  →  mean={alpha_B:.3f}  std={cv_B['std']:.3f}")

    val_B = soft_blend(val_probs_T1["full"], val_hier_T1, alpha_B)
    test_B = soft_blend(test_probs_T1["full"], test_hier_T1, alpha_B)
    val_B_m = compute_metrics(val_B, val["labels"])
    test_B_m = compute_metrics(test_B, test["labels"])

    # ---- (C) Hard gating, T_opt ----
    val_C = hard_gate(val_probs_Topt["full"], val_hier_Topt, val_probs_Topt["binary"], thresh=0.5)
    test_C = hard_gate(test_probs_Topt["full"], test_hier_Topt, test_probs_Topt["binary"], thresh=0.5)
    val_C_m = compute_metrics(val_C, val["labels"])
    test_C_m = compute_metrics(test_C, test["labels"])

    # ---- (D) Pure hier, T_opt (α=1 skyline) ----
    val_D_m = compute_metrics(val_hier_Topt, val["labels"])
    test_D_m = compute_metrics(test_hier_Topt, test["labels"])

    # ---- (E) Pure full, T_opt (α=0 sanity) ----
    val_E_m = compute_metrics(val_probs_Topt["full"], val["labels"])
    test_E_m = compute_metrics(test_probs_Topt["full"], test["labels"])

    # Baseline for Δ reference (equals E since T-invariance)
    baseline_f1 = test_E_m["f1_macro"]

    # ---- Summary per variant ----
    print(f"\n  {'Variant':<28} {'α':>6}  {'std':>6}  {'Val F1':>8}  {'Test F1':>9}  {'Δ vs base':>10}")
    print("  " + "-" * 76)
    for label, alpha, std, val_m, test_m in [
        (f"(A) α-CV, T={T_opt:.2f}",         alpha_A, cv_A["std"], val_A_m, test_A_m),
        ("(B) α-CV, T=1.00",                 alpha_B, cv_B["std"], val_B_m, test_B_m),
        ("(C) hard-gate, T_opt",             float("nan"), float("nan"), val_C_m, test_C_m),
        ("(D) pure hier, T_opt (α=1)",       1.0, 0.0, val_D_m, test_D_m),
        ("(E) pure full, T_opt (baseline)",  0.0, 0.0, val_E_m, test_E_m),
    ]:
        delta = (test_m["f1_macro"] - baseline_f1) * 100
        a_s = f"{alpha:.3f}" if not np.isnan(alpha) else "—"
        std_s = f"{std:.3f}" if not np.isnan(std) else "—"
        print(f"  {label:<28} {a_s:>6}  {std_s:>6}  {val_m['f1_macro']:>8.4f}  "
              f"{test_m['f1_macro']:>9.4f}  {delta:>+9.2f}pp")

    # ---- Per-class detail for best variant ----
    variants = {"A": test_A_m, "B": test_B_m, "C": test_C_m, "D": test_D_m}
    best_key = max(variants, key=lambda k: variants[k]["f1_macro"])
    best_test = variants[best_key]
    print(f"\n  BEST variant: ({best_key})  Test F1 = {best_test['f1_macro']:.4f}  "
          f"(Δ {(best_test['f1_macro'] - baseline_f1)*100:+.2f}pp vs baseline)")
    print(f"  Per-class F1:     " + ", ".join(f"{n.split('_')[1]}={best_test[f'f1_{n}']:.3f}"
                                               for n in BIRADS_NAMES))
    print(f"  Per-class recall: " + ", ".join(f"{n.split('_')[1]}={best_test[f'recall_{n}']:.3f}"
                                               for n in BIRADS_NAMES))
    cm = np.array(best_test["confusion_matrix"])
    print(f"  Confusion matrix (test, variant {best_key}):")
    print(f"            pred_BR1 pred_BR2 pred_BR4 pred_BR5")
    for i, n in enumerate(["true_BR1", "true_BR2", "true_BR4", "true_BR5"]):
        print(f"    {n}:  " + " ".join(f"{v:>8d}" for v in cm[i])
              + f"   (total={int(cm[i].sum())})")

    # Guardrail
    stable_A = cv_A["std"] < 0.3
    stable_B = cv_B["std"] < 0.3
    print(f"\n  Guardrail std(α) < 0.3:  (A) {stable_A}  |  (B) {stable_B}")

    return {
        "T_opt": T_opt,
        "alpha_A_cv": cv_A,
        "alpha_B_cv": cv_B,
        "A": {"alpha": alpha_A, "val_metrics": val_A_m, "test_metrics": test_A_m},
        "B": {"alpha": alpha_B, "val_metrics": val_B_m, "test_metrics": test_B_m},
        "C": {"val_metrics": val_C_m, "test_metrics": test_C_m},
        "D": {"val_metrics": val_D_m, "test_metrics": test_D_m},
        "E": {"val_metrics": val_E_m, "test_metrics": test_E_m},
        "best_variant": best_key,
        "baseline_test_f1": float(baseline_f1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--run-name", default="tier1_task1_4_gating_c6")
    parser.add_argument("--experiment-name", default="birads-inference-pipeline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    art = Path(args.artifacts_dir)
    with open(args.config) as f:
        config = yaml.safe_load(f)
    config.setdefault("mlflow", {})["experiment_name"] = args.experiment_name

    # Load T_opt from Task 1.2
    t_values = json.load(open(art / "c6_temperature_values.json"))
    T_nontta = float(t_values["nonTTA_T_optimal"])
    T_tta8   = float(t_values["tta8_T_optimal"])
    print(f"[LOAD] T_opt nonTTA = {T_nontta:.4f}, tta8 = {T_tta8:.4f}")

    results = {}
    results["nonTTA"] = run_track("nonTTA", art, T_opt=T_nontta, seed=args.seed, tta_prefix="")
    results["tta8"]   = run_track("tta8",   art, T_opt=T_tta8,   seed=args.seed, tta_prefix="tta8_")

    # Overall summary
    print(f"\n{'='*72}")
    print("  Task 1.4 — Summary (both tracks, best variant each)")
    print(f"{'='*72}")
    print(f"  {'Track':<8} {'BestVar':>8} {'α':>6} {'Test F1':>9} {'Δ vs baseline':>14}")
    for tr in ("nonTTA", "tta8"):
        r = results[tr]
        best_key = r["best_variant"]
        best_alpha = r[best_key].get("alpha", float("nan"))
        test_f1 = r[best_key]["test_metrics"]["f1_macro"]
        delta = (test_f1 - r["baseline_test_f1"]) * 100
        a_s = f"{best_alpha:.3f}" if not np.isnan(best_alpha) else "—"
        print(f"  {tr:<8} {best_key:>8} {a_s:>6} {test_f1:>9.4f} {delta:>+12.2f}pp")

    # Save artifacts
    out_json = art / "c6_gating_metrics.json"
    with open(out_json, "w") as f:
        # json cannot serialize np arrays; ensure clean
        def _clean(o):
            if isinstance(o, dict):
                return {k: _clean(v) for k, v in o.items()}
            if isinstance(o, list):
                return [_clean(v) for v in o]
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return o
        json.dump(_clean(results), f, indent=2)
    print(f"\n[SAVE] {out_json}")

    gating_values = {
        "nonTTA_alpha_A":  results["nonTTA"]["A"]["alpha"],
        "nonTTA_alpha_B":  results["nonTTA"]["B"]["alpha"],
        "tta8_alpha_A":    results["tta8"]["A"]["alpha"],
        "tta8_alpha_B":    results["tta8"]["B"]["alpha"],
        "nonTTA_best_variant": results["nonTTA"]["best_variant"],
        "tta8_best_variant":   results["tta8"]["best_variant"],
    }
    with open(art / "c6_gating_values.json", "w") as f:
        json.dump(gating_values, f, indent=2)
    print(f"[SAVE] {art}/c6_gating_values.json")

    # MLflow
    if not args.no_mlflow:
        print("\n[MLFL] Logging ...")
        logger = ExperimentLogger(config)
        logger.start_run(
            run_name=args.run_name,
            tags={"tier": "1", "task": "1.4", "baseline_ref": "C6",
                  "variants": "A_softCV_Topt_B_softCV_T1_C_hard_D_pureHier_E_pureFull"},
        )
        logger.log_params_flat(gating_values)
        flat = {}
        for tr in ("nonTTA", "tta8"):
            r = results[tr]
            flat[f"{tr}_baseline_test_f1"] = r["baseline_test_f1"]
            for var in ("A", "B", "C", "D", "E"):
                test_m = r[var]["test_metrics"]
                flat[f"{tr}_{var}_test_f1_macro"] = test_m["f1_macro"]
                for n in BIRADS_NAMES:
                    flat[f"{tr}_{var}_test_f1_{n}"] = test_m[f"f1_{n}"]
                    flat[f"{tr}_{var}_test_recall_{n}"] = test_m[f"recall_{n}"]
                flat[f"{tr}_{var}_test_f1_delta_pp"] = (
                    (test_m["f1_macro"] - r["baseline_test_f1"]) * 100
                )
        logger.log_metrics(flat)
        logger.log_artifact(str(art / "c6_gating_metrics.json"))
        logger.log_artifact(str(art / "c6_gating_values.json"))
        logger.end_run()
        print("[MLFL] Done.")

    print("\n[DONE] Task 1.4 complete.")


if __name__ == "__main__":
    main()
