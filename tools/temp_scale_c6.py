"""
Tier 1 Task 1.2 — Temperature Scaling on C6 Val Logits
========================================================
LBFGS optimization of a single scalar T on the **val** set, applied to
**test** afterwards. Runs two parallel tracks per Option C:
    1. non-TTA  (cached raw C6 val/test logits)
    2. tta8     (TTA-8-view logit-averaged val/test cache)

Important invariance:
    argmax(logits / T) = argmax(logits) for any T > 0
    → classification metrics (F1, accuracy, confusion matrix) are T-invariant.
    Only calibration metrics (NLL, ECE, Brier, mean confidence) change.
    Downstream Task 1.3 (threshold offsets) and Task 1.4 (gating blend)
    get indirect F1 effects through distribution shape changes.

Usage:
    python tools/temp_scale_c6.py \
        --config configs/experiment_v2_birads/convnextv2_large_8bit_ablation_c6.yaml \
        --artifacts-dir artifacts \
        --run-name tier1_task1_2_temp_c6 \
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
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score

from utils.mlflow_logger import ExperimentLogger


BIRADS_NAMES = ["BIRADS_1", "BIRADS_2", "BIRADS_4", "BIRADS_5"]


# ---------------------------------------------------------------------------
# Temperature fit
# ---------------------------------------------------------------------------

def fit_temperature_lbfgs(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    init_T: float = 1.5,
    max_iter: int = 200,
    n_restarts: int = 3,
) -> dict:
    """
    LBFGS on log_T. Returns T_opt, fit NLL trajectory (val NLL before/after).

    Multi-restart from {0.5, 1.0, 1.5} to avoid local optima. Picks lowest val NLL.
    """
    logits_t = torch.from_numpy(val_logits).float()
    labels_t = torch.from_numpy(val_labels).long()

    def _nll_at_T(T: float) -> float:
        with torch.no_grad():
            return float(F.cross_entropy(logits_t / T, labels_t).item())

    nll_before = _nll_at_T(1.0)
    nll_at_init = _nll_at_T(init_T)

    best_T = init_T
    best_nll = float("inf")
    all_restarts = []

    for start_T in [0.5, 1.0, 1.5]:
        log_T = torch.nn.Parameter(
            torch.tensor(float(np.log(start_T)), dtype=torch.float32)
        )
        optimizer = torch.optim.LBFGS(
            [log_T], lr=0.05, max_iter=max_iter,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-7, tolerance_change=1e-9,
        )

        def closure():
            optimizer.zero_grad()
            T = torch.exp(log_T)
            loss = F.cross_entropy(logits_t / T, labels_t)
            loss.backward()
            return loss

        optimizer.step(closure)

        T_val = float(torch.exp(log_T).detach().item())
        nll_val = _nll_at_T(T_val)
        all_restarts.append({"start_T": start_T, "converged_T": T_val, "val_nll": nll_val})
        if nll_val < best_nll:
            best_nll = nll_val
            best_T = T_val

    return {
        "T_optimal": best_T,
        "val_nll_T1.0": nll_before,
        "val_nll_Tinit": nll_at_init,
        "val_nll_Topt": best_nll,
        "restarts": all_restarts,
    }


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def nll_np(logits: np.ndarray, labels: np.ndarray, T: float = 1.0) -> float:
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    log_norm = np.log(np.exp(z).sum(axis=1, keepdims=True))
    log_probs = z - log_norm
    return float(-log_probs[np.arange(len(labels)), labels].mean())


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    e_val = 0.0
    n = len(labels)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        e_val += (mask.sum() / n) * abs(correct[mask].mean() - conf[mask].mean())
    return float(e_val)


def brier(probs: np.ndarray, labels: np.ndarray, K: int = 4) -> float:
    oh = np.eye(K)[labels]
    return float(np.mean(np.sum((probs - oh) ** 2, axis=1)))


def eval_metrics(logits: np.ndarray, labels: np.ndarray, T: float) -> dict:
    probs = softmax_np(logits / T)
    preds = probs.argmax(axis=1)
    return {
        "T": float(T),
        "nll": nll_np(logits, labels, T=T),
        "ece_15bins": ece(probs, labels, 15),
        "brier": brier(probs, labels),
        "mean_confidence": float(probs.max(axis=1).mean()),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "accuracy": float((preds == labels).mean()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_track(track_name: str, val_logits, val_labels, test_logits, test_labels) -> dict:
    print(f"\n{'='*68}")
    print(f"  Track: {track_name}")
    print(f"{'='*68}")

    fit = fit_temperature_lbfgs(val_logits, val_labels, init_T=1.5)
    T_opt = fit["T_optimal"]
    print(f"  LBFGS restarts:")
    for r in fit["restarts"]:
        print(f"    start T={r['start_T']:.2f}  → converged T={r['converged_T']:.4f}  (val NLL={r['val_nll']:.4f})")
    print(f"  Selected T_optimal = {T_opt:.4f}")

    results = {"T_optimal": T_opt, "restarts": fit["restarts"]}
    for split_name, logits, labels in [
        ("val", val_logits, val_labels),
        ("test", test_logits, test_labels),
    ]:
        for tag, T in [("T1.0", 1.0), ("T1.5", 1.5), ("Topt", T_opt)]:
            m = eval_metrics(logits, labels, T=T)
            results[f"{split_name}_{tag}"] = m
            print(f"  [{track_name}/{split_name:<4}/{tag:<4}]  "
                  f"T={T:.4f}  NLL={m['nll']:.4f}  ECE={m['ece_15bins']:.4f}  "
                  f"Brier={m['brier']:.4f}  Conf={m['mean_confidence']:.4f}  "
                  f"F1={m['f1_macro']:.4f}")

    # Delta summary
    test_ece_delta = results["test_Topt"]["ece_15bins"] - results["test_T1.0"]["ece_15bins"]
    test_nll_delta = results["test_Topt"]["nll"] - results["test_T1.0"]["nll"]
    results["delta_test_ece_Topt_vs_T1"] = test_ece_delta
    results["delta_test_nll_Topt_vs_T1"] = test_nll_delta
    print(f"  ΔTest ECE (T_opt − T=1):  {test_ece_delta:+.4f}")
    print(f"  ΔTest NLL (T_opt − T=1):  {test_nll_delta:+.4f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--run-name", default="tier1_task1_2_temp_c6")
    parser.add_argument("--experiment-name", default="birads-inference-pipeline")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    art = Path(args.artifacts_dir)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config.setdefault("mlflow", {})["experiment_name"] = args.experiment_name

    # Load shared labels
    val_labels = np.load(art / "c6_val_labels.npy")
    test_labels = np.load(art / "c6_test_labels.npy")

    # Track 1: non-TTA
    nontta_val = np.load(art / "c6_val_logits.npy")
    nontta_test = np.load(art / "c6_test_logits.npy")
    # Track 2: tta8
    tta8_val = np.load(art / "c6_val_tta8_full_logits.npy")
    tta8_test = np.load(art / "c6_test_tta8_full_logits.npy")

    print(f"[LOAD] non-TTA val={nontta_val.shape}, test={nontta_test.shape}")
    print(f"[LOAD] tta8     val={tta8_val.shape}, test={tta8_test.shape}")

    results = {}
    results["nonTTA"] = run_track("nonTTA", nontta_val, val_labels, nontta_test, test_labels)
    results["tta8"]   = run_track("tta8",   tta8_val,   val_labels, tta8_test,   test_labels)

    # Summary table
    print(f"\n{'='*68}")
    print("  Task 1.2 — Summary")
    print(f"{'='*68}")
    print(f"  {'Track':<8} {'T_opt':>8} {'Test ECE T=1':>14} {'Test ECE T_opt':>16} "
          f"{'ΔECE':>8} {'F1 (T-inv)':>12}")
    for tr in ("nonTTA", "tta8"):
        r = results[tr]
        print(f"  {tr:<8} {r['T_optimal']:>8.4f} "
              f"{r['test_T1.0']['ece_15bins']:>14.4f} "
              f"{r['test_Topt']['ece_15bins']:>16.4f} "
              f"{r['delta_test_ece_Topt_vs_T1']:>+8.4f} "
              f"{r['test_Topt']['f1_macro']:>12.4f}")
    print("\n  (F1 is T-invariant by construction; shown only as sanity — should match Task 1.1 / baseline.)")

    # Accept criterion check
    print(f"\n  Accept criterion: test ECE lower after T-scaling, F1 unchanged.")
    for tr in ("nonTTA", "tta8"):
        r = results[tr]
        passed = r["test_Topt"]["ece_15bins"] <= r["test_T1.0"]["ece_15bins"]
        f1_stable = abs(r["test_Topt"]["f1_macro"] - r["test_T1.0"]["f1_macro"]) < 1e-6
        print(f"    [{tr}]  ECE reduced: {passed}  |  F1 stable: {f1_stable}")

    # Save artifacts
    with open(art / "c6_temp_scale_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVE] {art}/c6_temp_scale_metrics.json")

    t_values = {
        "nonTTA_T_optimal": results["nonTTA"]["T_optimal"],
        "tta8_T_optimal":   results["tta8"]["T_optimal"],
    }
    with open(art / "c6_temperature_values.json", "w") as f:
        json.dump(t_values, f, indent=2)
    print(f"[SAVE] {art}/c6_temperature_values.json")

    # MLflow
    if not args.no_mlflow:
        print("\n[MLFL] Logging ...")
        logger = ExperimentLogger(config)
        logger.start_run(
            run_name=args.run_name,
            tags={"tier": "1", "task": "1.2", "baseline_ref": "C6",
                  "note": "T-scaling does not change argmax; F1 metrics are T-invariant"},
        )
        logger.log_params_flat(t_values)
        flat = {}
        for tr in ("nonTTA", "tta8"):
            r = results[tr]
            flat[f"{tr}_T_optimal"] = r["T_optimal"]
            flat[f"{tr}_delta_test_ece"] = r["delta_test_ece_Topt_vs_T1"]
            flat[f"{tr}_delta_test_nll"] = r["delta_test_nll_Topt_vs_T1"]
            for split in ("val", "test"):
                for tag in ("T1.0", "T1.5", "Topt"):
                    for k, v in r[f"{split}_{tag}"].items():
                        if isinstance(v, (int, float)):
                            flat[f"{tr}_{split}_{tag}_{k}"] = float(v)
        logger.log_metrics(flat)
        logger.log_artifact(str(art / "c6_temp_scale_metrics.json"))
        logger.log_artifact(str(art / "c6_temperature_values.json"))
        logger.end_run()
        print("[MLFL] Done.")

    print("\n[DONE] Task 1.2 complete. Downstream T values:")
    print(f"  nonTTA T = {t_values['nonTTA_T_optimal']:.4f}")
    print(f"  tta8   T = {t_values['tta8_T_optimal']:.4f}")


if __name__ == "__main__":
    main()
