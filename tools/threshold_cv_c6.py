"""
Tier 1 Task 1.3 — 5-fold CV threshold offsets (d1, d4) on raw C6 logits
==========================================================================
Grid search on val logits to find BR1 and BR4 logit offsets that maximize
test F1 macro while remaining robust to fold variance.

Key design points:
  - Runs on raw logits (no T-scale pre-step) because argmax is T-invariant:
    argmax((logits + d)/T) = argmax(logits + d).
    Using T-scaled logits here would give identical fold optima.
  - Only d1 (BR1) and d4 (BR4) are free; d2 = d5 = 0 fixed, so the minority
    classes are the only ones boosted. This is a 2D grid search.
  - 5-fold StratifiedKFold on val set (n=1284, ~257/fold). Per-fold optimum
    recorded; final offset = mean over folds (and median reported).
  - Fold-std guardrail: if std(d1) > 0.3 or std(d4) > 0.3, threshold
    approach is flagged as unstable and downstream weight should be reduced.
  - Additional "naive" single-pass optimum (whole-val grid search) computed
    for comparison — if naive vs CV-averaged gap > 1pp, overfit is evident.

Runs both tracks in parallel:
  - nonTTA: c6_{val,test}_logits.npy
  - tta8:   c6_{val,test}_tta8_full_logits.npy

Outputs:
  - artifacts/c6_threshold_cv_metrics.json  (per-track full results)
  - artifacts/c6_threshold_values.json      (selected (d1, d4) per track)
  - MLflow run tier1_task1_3_thresh_c6

CPU-only, expected ~1-2 minutes total.

Usage:
    python tools/threshold_cv_c6.py \\
        --config configs/experiment_v2_birads/convnextv2_large_8bit_ablation_c6.yaml \\
        --artifacts-dir artifacts \\
        --run-name tier1_task1_3_thresh_c6 \\
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
D1_GRID = np.linspace(0.0, 1.0, 21)   # 0.00, 0.05, ..., 1.00
D4_GRID = np.linspace(0.0, 1.2, 25)   # 0.00, 0.05, ..., 1.20


# ---------------------------------------------------------------------------
# Grid search core
# ---------------------------------------------------------------------------

def apply_offsets(logits: np.ndarray, d1: float, d4: float) -> np.ndarray:
    """Add offset vector [d1, 0, d4, 0] to each row."""
    return logits + np.array([d1, 0.0, d4, 0.0], dtype=logits.dtype)


def grid_search_f1(
    logits: np.ndarray,
    labels: np.ndarray,
    d1_grid: np.ndarray = D1_GRID,
    d4_grid: np.ndarray = D4_GRID,
) -> tuple[float, float, float, np.ndarray]:
    """
    Exhaustive 2D grid. Returns (best_d1, best_d4, best_f1, f1_surface).
    f1_surface has shape (len(d1_grid), len(d4_grid)).
    """
    best_f1 = -1.0
    best_d1 = 0.0
    best_d4 = 0.0
    surface = np.zeros((len(d1_grid), len(d4_grid)), dtype=np.float64)
    for i, d1 in enumerate(d1_grid):
        for j, d4 in enumerate(d4_grid):
            preds = apply_offsets(logits, d1, d4).argmax(axis=1)
            f1 = f1_score(labels, preds, average="macro", zero_division=0)
            surface[i, j] = f1
            if f1 > best_f1:
                best_f1 = f1
                best_d1 = float(d1)
                best_d4 = float(d4)
    return best_d1, best_d4, float(best_f1), surface


def compute_test_metrics(logits, labels, d1, d4) -> dict:
    preds = apply_offsets(logits, d1, d4).argmax(axis=1)
    m = {"f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0))}
    pr, rc, f1, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
    for i, n in enumerate(BIRADS_NAMES):
        if i < len(f1):
            m[f"f1_{n}"] = float(f1[i])
            m[f"recall_{n}"] = float(rc[i])
            m[f"precision_{n}"] = float(pr[i])
    m["accuracy"] = float((preds == labels).mean())
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2, 3]).tolist()
    m["confusion_matrix"] = cm
    return m


# ---------------------------------------------------------------------------
# Per-track runner
# ---------------------------------------------------------------------------

def run_track(track: str, val_logits, val_labels, test_logits, test_labels,
              seed: int = 42) -> dict:
    print(f"\n{'='*72}")
    print(f"  Track: {track}")
    print(f"{'='*72}")
    print(f"  Val n={len(val_labels)}, Test n={len(test_labels)}")
    print(f"  d1 grid: {len(D1_GRID)} points in [0, 1.0]")
    print(f"  d4 grid: {len(D4_GRID)} points in [0, 1.2]")

    # --- Naive single-pass search on all val ---
    naive_d1, naive_d4, naive_f1, naive_surface = grid_search_f1(val_logits, val_labels)
    print(f"\n  [NAIVE] whole-val optimum:  d1={naive_d1:.3f}  d4={naive_d4:.3f}  "
          f"val F1={naive_f1:.4f}")
    naive_test = compute_test_metrics(test_logits, test_labels, naive_d1, naive_d4)

    # --- 5-fold CV ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_records = []
    for fold_idx, (_, held_idx) in enumerate(skf.split(val_logits, val_labels), start=1):
        h_logits = val_logits[held_idx]
        h_labels = val_labels[held_idx]
        d1, d4, f1, _ = grid_search_f1(h_logits, h_labels)
        fold_records.append({"fold": fold_idx, "d1": d1, "d4": d4,
                             "fold_val_f1": f1, "fold_n": int(len(held_idx))})
        print(f"  [FOLD{fold_idx}] n={len(held_idx):>4d}  d1={d1:.3f}  d4={d4:.3f}  "
              f"fold_val F1={f1:.4f}")

    d1_arr = np.array([r["d1"] for r in fold_records])
    d4_arr = np.array([r["d4"] for r in fold_records])
    cv_d1_mean = float(d1_arr.mean())
    cv_d4_mean = float(d4_arr.mean())
    cv_d1_median = float(np.median(d1_arr))
    cv_d4_median = float(np.median(d4_arr))
    cv_d1_std = float(d1_arr.std())
    cv_d4_std = float(d4_arr.std())

    print(f"\n  [CV]  fold optima: d1 mean={cv_d1_mean:.3f} std={cv_d1_std:.3f} median={cv_d1_median:.3f}")
    print(f"  [CV]  fold optima: d4 mean={cv_d4_mean:.3f} std={cv_d4_std:.3f} median={cv_d4_median:.3f}")

    # Guardrail
    stable_d1 = cv_d1_std < 0.3
    stable_d4 = cv_d4_std < 0.3
    print(f"  [GUARDRAIL] std(d1) < 0.3: {stable_d1}  |  std(d4) < 0.3: {stable_d4}")

    # Boundary-hit check: if many folds hit grid upper bound
    d1_boundary_hits = int((d1_arr >= D1_GRID[-1] - 1e-9).sum())
    d4_boundary_hits = int((d4_arr >= D4_GRID[-1] - 1e-9).sum())
    if d4_boundary_hits >= 3:
        print(f"  [BOUNDARY] d4 hits upper bound ({D4_GRID[-1]:.2f}) in {d4_boundary_hits}/5 folds "
              f"→ threshold approach may be structurally insufficient; lean on Task 1.4 gating.")
    if d1_boundary_hits >= 3:
        print(f"  [BOUNDARY] d1 hits upper bound ({D1_GRID[-1]:.2f}) in {d1_boundary_hits}/5 folds "
              f"→ review BR1 strategy.")

    # --- Apply CV-averaged offsets to val (sanity) and test ---
    final_d1 = cv_d1_mean
    final_d4 = cv_d4_mean
    val_final = compute_test_metrics(val_logits, val_labels, final_d1, final_d4)
    test_final = compute_test_metrics(test_logits, test_labels, final_d1, final_d4)

    # Baseline (d1=d4=0) metrics for delta reporting
    val_baseline = compute_test_metrics(val_logits, val_labels, 0.0, 0.0)
    test_baseline = compute_test_metrics(test_logits, test_labels, 0.0, 0.0)

    # Naive vs CV gap on test (overfit indicator)
    naive_vs_cv_test_f1_gap_pp = (naive_test["f1_macro"] - test_final["f1_macro"]) * 100
    print(f"\n  [APPLY] final (CV-mean) d1={final_d1:.3f}  d4={final_d4:.3f}")
    print(f"  [APPLY] val   F1 : {val_baseline['f1_macro']:.4f} → {val_final['f1_macro']:.4f}  "
          f"(Δ {(val_final['f1_macro']-val_baseline['f1_macro'])*100:+.2f}pp)")
    print(f"  [APPLY] test  F1 : {test_baseline['f1_macro']:.4f} → {test_final['f1_macro']:.4f}  "
          f"(Δ {(test_final['f1_macro']-test_baseline['f1_macro'])*100:+.2f}pp)")
    print(f"  [APPLY] test  per-class F1: "
          + ", ".join(f"{n.split('_')[1]}={test_final[f'f1_{n}']:.3f}" for n in BIRADS_NAMES))
    print(f"  [APPLY] test  per-class recall: "
          + ", ".join(f"{n.split('_')[1]}={test_final[f'recall_{n}']:.3f}" for n in BIRADS_NAMES))
    print(f"  [OVERFIT] naive-vs-CV test F1 gap: {naive_vs_cv_test_f1_gap_pp:+.3f}pp "
          f"(|gap| > 1pp = overfit warning)")

    return {
        "naive": {"d1": naive_d1, "d4": naive_d4, "val_f1": naive_f1,
                  "test_metrics": naive_test},
        "fold_records": fold_records,
        "cv_d1_mean": cv_d1_mean, "cv_d1_std": cv_d1_std, "cv_d1_median": cv_d1_median,
        "cv_d4_mean": cv_d4_mean, "cv_d4_std": cv_d4_std, "cv_d4_median": cv_d4_median,
        "stable_d1": stable_d1, "stable_d4": stable_d4,
        "d1_boundary_hits": d1_boundary_hits,
        "d4_boundary_hits": d4_boundary_hits,
        "final_d1": final_d1, "final_d4": final_d4,
        "val_baseline":   val_baseline,
        "val_with_offsets":   val_final,
        "test_baseline":  test_baseline,
        "test_with_offsets":  test_final,
        "naive_vs_cv_test_f1_gap_pp": float(naive_vs_cv_test_f1_gap_pp),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--run-name", default="tier1_task1_3_thresh_c6")
    parser.add_argument("--experiment-name", default="birads-inference-pipeline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    art = Path(args.artifacts_dir)
    with open(args.config) as f:
        config = yaml.safe_load(f)
    config.setdefault("mlflow", {})["experiment_name"] = args.experiment_name

    val_labels = np.load(art / "c6_val_labels.npy")
    test_labels = np.load(art / "c6_test_labels.npy")

    nontta_val = np.load(art / "c6_val_logits.npy")
    nontta_test = np.load(art / "c6_test_logits.npy")
    tta8_val = np.load(art / "c6_val_tta8_full_logits.npy")
    tta8_test = np.load(art / "c6_test_tta8_full_logits.npy")

    print(f"[LOAD] non-TTA val={nontta_val.shape}, test={nontta_test.shape}")
    print(f"[LOAD] tta8     val={tta8_val.shape}, test={tta8_test.shape}")

    results = {}
    results["nonTTA"] = run_track("nonTTA", nontta_val, val_labels, nontta_test, test_labels,
                                  seed=args.seed)
    results["tta8"]   = run_track("tta8",   tta8_val,   val_labels, tta8_test,   test_labels,
                                  seed=args.seed)

    # --- Overall summary ---
    print(f"\n{'='*72}")
    print("  Task 1.3 — Summary (both tracks)")
    print(f"{'='*72}")
    print(f"  {'Track':<8} {'d1':>6} {'d4':>6} {'Test F1 base':>13} {'Test F1 shifted':>16} "
          f"{'Δpp':>7} {'std(d1)':>8} {'std(d4)':>8}")
    for tr in ("nonTTA", "tta8"):
        r = results[tr]
        f1b = r["test_baseline"]["f1_macro"]
        f1s = r["test_with_offsets"]["f1_macro"]
        print(f"  {tr:<8} {r['final_d1']:>6.3f} {r['final_d4']:>6.3f} "
              f"{f1b:>13.4f} {f1s:>16.4f} {(f1s-f1b)*100:>+7.2f} "
              f"{r['cv_d1_std']:>8.3f} {r['cv_d4_std']:>8.3f}")

    # Save artifacts
    with open(art / "c6_threshold_cv_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVE] {art}/c6_threshold_cv_metrics.json")

    thr_values = {
        "nonTTA_d1": results["nonTTA"]["final_d1"],
        "nonTTA_d4": results["nonTTA"]["final_d4"],
        "tta8_d1":   results["tta8"]["final_d1"],
        "tta8_d4":   results["tta8"]["final_d4"],
    }
    with open(art / "c6_threshold_values.json", "w") as f:
        json.dump(thr_values, f, indent=2)
    print(f"[SAVE] {art}/c6_threshold_values.json")

    # MLflow
    if not args.no_mlflow:
        print("\n[MLFL] Logging ...")
        logger = ExperimentLogger(config)
        logger.start_run(
            run_name=args.run_name,
            tags={"tier": "1", "task": "1.3", "baseline_ref": "C6",
                  "search_space": "d1[0,1.0]x25 d4[0,1.2]x25",
                  "cv": "5fold_stratified_seed42"},
        )
        logger.log_params_flat({
            **thr_values,
            "d1_grid_max": float(D1_GRID[-1]),
            "d4_grid_max": float(D4_GRID[-1]),
            "cv_folds": 5,
            "seed": args.seed,
        })
        flat = {}
        for tr in ("nonTTA", "tta8"):
            r = results[tr]
            flat[f"{tr}_cv_d1_mean"] = r["cv_d1_mean"]
            flat[f"{tr}_cv_d1_std"]  = r["cv_d1_std"]
            flat[f"{tr}_cv_d4_mean"] = r["cv_d4_mean"]
            flat[f"{tr}_cv_d4_std"]  = r["cv_d4_std"]
            flat[f"{tr}_test_f1_baseline"]    = r["test_baseline"]["f1_macro"]
            flat[f"{tr}_test_f1_with_offset"] = r["test_with_offsets"]["f1_macro"]
            flat[f"{tr}_test_f1_delta_pp"] = (
                (r["test_with_offsets"]["f1_macro"] - r["test_baseline"]["f1_macro"]) * 100
            )
            flat[f"{tr}_naive_vs_cv_gap_pp"] = r["naive_vs_cv_test_f1_gap_pp"]
            for n in BIRADS_NAMES:
                flat[f"{tr}_test_f1_{n}"]     = r["test_with_offsets"][f"f1_{n}"]
                flat[f"{tr}_test_recall_{n}"] = r["test_with_offsets"][f"recall_{n}"]
        logger.log_metrics(flat)
        logger.log_artifact(str(art / "c6_threshold_cv_metrics.json"))
        logger.log_artifact(str(art / "c6_threshold_values.json"))
        logger.end_run()
        print("[MLFL] Done.")

    print("\n[DONE] Task 1.3 complete. Downstream offsets:")
    for k, v in thr_values.items():
        print(f"  {k} = {v:.3f}")


if __name__ == "__main__":
    main()
