"""
Tier 1 Task 1.5 — Cumulative Pipeline Evaluation + Decision Point
===================================================================
Reads all Tier 1 task cached artifacts and composes a single ablation table
across the complete inference pipeline, per Option C parallel tracks:

  Track 1 — nonTTA: raw C6 cached logits
  Track 2 — tta8:   TTA-8-view logit-averaged cached logits

Pipeline steps (incremental):
  C0  baseline                          (raw full head, T=1, no gate, no thresh)
  C1  + T-scale                         (F1-invariant; changes ECE / NLL only)
  C2  + T + hard gating                 (binary_prob > 0.5 → hier, else full)
  C3  + T + threshold (ablation row)    (d1/d4 from Task 1.3 — negative result)
  Cfull  + T + gating + threshold       (all components, full pipeline)
  Crec   + T + gating (no threshold)    (our recommended cumulative, post Task 1.4)

Reads:
  - artifacts/c6_{val,test}_{,tta8_}{full,binary,benign_sub,malign_sub}_logits.npy
  - artifacts/c6_{val,test}_labels.npy
  - artifacts/c6_temperature_values.json   (T_opt per track)
  - artifacts/c6_threshold_values.json     (d1, d4 per track)

Produces:
  - artifacts/c6_cumulative_metrics.json   (all rows × both tracks)
  - tasks/tier1_results.md                 (markdown ablation table, per prompt)
  - MLflow run tier1_cumulative_c6

Decision point verdict logic (from prompt):
  max(cumulative test F1) ≥ 0.72  → Tier 2 F1/F2, defer ensemble
  max(cumulative test F1) ∈ [0.70, 0.72)  → Multi-seed ensemble priority
  max(cumulative test F1) < 0.70         → Root cause + Tier 2 (F2 recommended)

Usage:
    python tools/cumulative_eval_c6.py \\
        --config configs/experiment_v2_birads/convnextv2_large_8bit_ablation_c6.yaml \\
        --artifacts-dir artifacts \\
        --run-name tier1_cumulative_c6 \\
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

from utils.mlflow_logger import ExperimentLogger


BIRADS_NAMES = ["BIRADS_1", "BIRADS_2", "BIRADS_4", "BIRADS_5"]
BIRADS_SHORT = ["BR1", "BR2", "BR4", "BR5"]


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def softmax_np(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def ece(probs, labels, n_bins=15):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        e += (mask.sum() / len(labels)) * abs(correct[mask].mean() - conf[mask].mean())
    return float(e)


def compute_metrics(probs, labels):
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
    m["binary_f1"] = float(f1_score((labels >= 2).astype(int), (preds >= 2).astype(int),
                                     zero_division=0))
    m["ece_15bins"] = ece(probs, labels)
    m["mean_confidence"] = float(probs.max(axis=1).mean())
    m["confusion_matrix"] = confusion_matrix(labels, preds, labels=[0, 1, 2, 3]).tolist()
    return m


def hier_and_binary(b_logits, bs_logits, ms_logits, T=1.0):
    b = softmax_np(b_logits / T)
    bs = softmax_np(bs_logits / T)
    ms = softmax_np(ms_logits / T)
    N = b.shape[0]
    hier = np.zeros((N, 4))
    hier[:, 0] = b[:, 0] * bs[:, 0]
    hier[:, 1] = b[:, 0] * bs[:, 1]
    hier[:, 2] = b[:, 1] * ms[:, 0]
    hier[:, 3] = b[:, 1] * ms[:, 1]
    hier = hier / hier.sum(axis=1, keepdims=True).clip(min=1e-12)
    return hier, b


def apply_pipeline(full_logits, b_logits, bs_logits, ms_logits,
                    T=1.0, d1=0.0, d4=0.0, use_gating=False):
    shifted = full_logits + np.array([d1, 0.0, d4, 0.0], dtype=full_logits.dtype)
    full_probs = softmax_np(shifted / T)
    if not use_gating:
        return full_probs
    hier, b_probs = hier_and_binary(b_logits, bs_logits, ms_logits, T=T)
    mask = b_probs[:, 1] > 0.5
    return np.where(mask[:, None], hier, full_probs)


# ---------------------------------------------------------------------------
# Per-track pipeline
# ---------------------------------------------------------------------------

def load_split(art: Path, split: str, tta_prefix: str = "") -> dict:
    prefix = "tta8_" if tta_prefix else ""
    full_name = f"c6_{split}_{prefix}full_logits.npy" if tta_prefix else f"c6_{split}_logits.npy"
    return {
        "full":       np.load(art / full_name),
        "binary":     np.load(art / f"c6_{split}_{prefix}binary_logits.npy"),
        "benign_sub": np.load(art / f"c6_{split}_{prefix}benign_sub_logits.npy"),
        "malign_sub": np.load(art / f"c6_{split}_{prefix}malign_sub_logits.npy"),
        "labels":     np.load(art / f"c6_{split}_labels.npy"),
    }


def run_cumulative_track(track: str, art: Path, T_opt: float, d1: float, d4: float,
                          tta_prefix: str = "") -> dict:
    print(f"\n{'='*78}")
    print(f"  Track: {track}  (T_opt={T_opt:.4f}, d1={d1:.3f}, d4={d4:.3f})")
    print(f"{'='*78}")

    val = load_split(art, "val", tta_prefix)
    test = load_split(art, "test", tta_prefix)

    # Rows, on TEST split:
    configs = [
        # (label, T, d1, d4, gate)
        ("C0_baseline",         1.0,    0.0, 0.0, False),
        ("C1_plus_Tscale",      T_opt,  0.0, 0.0, False),
        ("C2_plus_T_gate",      T_opt,  0.0, 0.0, True),
        ("C3_plus_T_thresh",    T_opt,  d1,  d4,  False),
        ("Cfull_T_gate_thresh", T_opt,  d1,  d4,  True),
        ("Crec_T_gate",         T_opt,  0.0, 0.0, True),  # recommended
    ]

    rows = {}
    print(f"\n  {'Config':<24} {'Test F1':>8} {'BR1':>6} {'BR2':>6} {'BR4':>6} {'BR5':>6} "
          f"{'BinF1':>7} {'ECE':>6} {'MeanConf':>9}")
    print("  " + "-" * 86)

    for label, T, d1_r, d4_r, gate in configs:
        probs = apply_pipeline(test["full"], test["binary"], test["benign_sub"], test["malign_sub"],
                                T=T, d1=d1_r, d4=d4_r, use_gating=gate)
        m = compute_metrics(probs, test["labels"])
        rows[label] = m
        print(f"  {label:<24} {m['f1_macro']:>8.4f} "
              f"{m['f1_BIRADS_1']:>6.3f} {m['f1_BIRADS_2']:>6.3f} "
              f"{m['f1_BIRADS_4']:>6.3f} {m['f1_BIRADS_5']:>6.3f} "
              f"{m['binary_f1']:>7.3f} {m['ece_15bins']:>6.3f} {m['mean_confidence']:>9.3f}")

    # Δ vs raw baseline (C0)
    base_f1 = rows["C0_baseline"]["f1_macro"]
    print(f"\n  Δ vs C0_baseline:")
    for label in rows:
        if label == "C0_baseline":
            continue
        d = (rows[label]["f1_macro"] - base_f1) * 100
        print(f"    {label:<24}  Δ = {d:+.2f}pp")

    return {"rows": rows, "baseline_f1": float(base_f1), "T_opt": T_opt, "d1": d1, "d4": d4}


# ---------------------------------------------------------------------------
# Markdown table writer
# ---------------------------------------------------------------------------

def write_markdown_table(out_path: Path, results: dict) -> None:
    lines = []
    lines.append("# Tier 1 Cumulative Ablation — C6 Inference Pipeline\n")
    lines.append("Generated by `tools/cumulative_eval_c6.py`. Each row applies the listed "
                 "components in addition to the raw C6 baseline forward pass. Test set "
                 "(1,655 patients, 4-class).\n")

    for tr in ("nonTTA", "tta8"):
        r = results[tr]
        base_f1 = r["baseline_f1"]
        lines.append(f"\n## Track: {tr}  "
                     f"(T_opt={r['T_opt']:.4f}, d1={r['d1']:.3f}, d4={r['d4']:.3f})\n")
        lines.append("| Config | Test F1 macro | Δ vs baseline | BR1 F1 | BR2 F1 | BR4 F1 | BR5 F1 | Binary F1 | ECE | Mean Conf |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for label, row in r["rows"].items():
            d = (row["f1_macro"] - base_f1) * 100
            lines.append(
                f"| {label} | {row['f1_macro']:.4f} | {d:+.2f}pp "
                f"| {row['f1_BIRADS_1']:.3f} | {row['f1_BIRADS_2']:.3f} "
                f"| {row['f1_BIRADS_4']:.3f} | {row['f1_BIRADS_5']:.3f} "
                f"| {row['binary_f1']:.3f} | {row['ece_15bins']:.3f} "
                f"| {row['mean_confidence']:.3f} |"
            )
    lines.append("\n## Row Legend\n")
    lines.append("- **C0_baseline**: raw full-head argmax, T=1, no gate/threshold.")
    lines.append("- **C1_plus_Tscale**: apply T_opt from Task 1.2 (F1-invariant; only ECE changes).")
    lines.append("- **C2_plus_T_gate**: T_opt + hard gating (binary prob > 0.5 → hier, else full).")
    lines.append("- **C3_plus_T_thresh**: T_opt + threshold offsets (Task 1.3; negative result, "
                 "ablation only).")
    lines.append("- **Cfull_T_gate_thresh**: T + gating + threshold (full pipeline including "
                 "the negative component).")
    lines.append("- **Crec_T_gate**: **recommended** — T + gating, no threshold "
                 "(best-defensible cumulative).")
    lines.append("\n## Cross-Track Summary\n")
    best_overall = max(
        ((tr, label, r["rows"][label]["f1_macro"])
         for tr, r in results.items() for label in r["rows"]),
        key=lambda x: x[2],
    )
    lines.append(f"- Best cumulative: **{best_overall[0]} / {best_overall[1]} = {best_overall[2]:.4f}**")
    lines.append(f"- nonTTA baseline (C0): {results['nonTTA']['baseline_f1']:.4f}")
    lines.append(f"- tta8 baseline (C0):   {results['tta8']['baseline_f1']:.4f}")
    lines.append(f"- Best Crec_T_gate:     nonTTA {results['nonTTA']['rows']['Crec_T_gate']['f1_macro']:.4f} "
                 f"/ tta8 {results['tta8']['rows']['Crec_T_gate']['f1_macro']:.4f}")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"\n[SAVE] Markdown ablation table → {out_path}")


# ---------------------------------------------------------------------------
# Decision point
# ---------------------------------------------------------------------------

def decision_verdict(best_f1: float) -> str:
    if best_f1 >= 0.72:
        return ("ROUTE_TIER2_F1_F2",
                f"Max cumulative F1 = {best_f1:.4f} ≥ 0.72 → proceed to Tier 2 F1/F2; "
                "defer multi-seed ensemble.")
    if best_f1 >= 0.70:
        return ("ROUTE_ENSEMBLE",
                f"Max cumulative F1 = {best_f1:.4f} ∈ [0.70, 0.72) → multi-seed ensemble "
                "(Tier 2 Task 2.0) takes priority.")
    return ("ROUTE_ROOT_CAUSE_THEN_TIER2",
            f"Max cumulative F1 = {best_f1:.4f} < 0.70 → root-cause analysis before Tier 2. "
            "Sanity (E variants in Task 1.4 matched baseline) indicates no pipeline bug; "
            "root cause is architectural+data: inference-time tricks duplicate full-head info "
            "(Lesson #48) and val→test prior shift kills threshold (Lesson #47). "
            "Recommend Tier 2 Task 2.2 (F2, logit-adjusted training, Menon 2021) as "
            "directly-targeting intervention.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--run-name", default="tier1_cumulative_c6")
    parser.add_argument("--experiment-name", default="birads-inference-pipeline")
    parser.add_argument("--output-md", default="tasks/tier1_results.md")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    art = Path(args.artifacts_dir)
    with open(args.config) as f:
        config = yaml.safe_load(f)
    config.setdefault("mlflow", {})["experiment_name"] = args.experiment_name

    # Load per-track values from prior tasks
    t_vals  = json.load(open(art / "c6_temperature_values.json"))
    th_vals = json.load(open(art / "c6_threshold_values.json"))
    print(f"[LOAD] T_opt: nonTTA={t_vals['nonTTA_T_optimal']:.4f}, "
          f"tta8={t_vals['tta8_T_optimal']:.4f}")
    print(f"[LOAD] thresh: nonTTA d1={th_vals['nonTTA_d1']:.3f} d4={th_vals['nonTTA_d4']:.3f}, "
          f"tta8 d1={th_vals['tta8_d1']:.3f} d4={th_vals['tta8_d4']:.3f}")

    results = {}
    results["nonTTA"] = run_cumulative_track(
        "nonTTA", art,
        T_opt=float(t_vals["nonTTA_T_optimal"]),
        d1=float(th_vals["nonTTA_d1"]),
        d4=float(th_vals["nonTTA_d4"]),
        tta_prefix="",
    )
    results["tta8"] = run_cumulative_track(
        "tta8", art,
        T_opt=float(t_vals["tta8_T_optimal"]),
        d1=float(th_vals["tta8_d1"]),
        d4=float(th_vals["tta8_d4"]),
        tta_prefix="tta8_",
    )

    # Decision point
    all_f1 = [results[tr]["rows"][row]["f1_macro"]
              for tr in results for row in results[tr]["rows"]]
    best_f1 = max(all_f1)
    best_where = [(tr, row) for tr in results for row in results[tr]["rows"]
                  if results[tr]["rows"][row]["f1_macro"] == best_f1][0]

    verdict_code, verdict_msg = decision_verdict(best_f1)

    print(f"\n{'='*78}")
    print("  DECISION POINT")
    print(f"{'='*78}")
    print(f"  Best cumulative test F1: {best_f1:.4f}  (at {best_where[0]}/{best_where[1]})")
    print(f"  Verdict: {verdict_code}")
    print(f"  {verdict_msg}")

    # Save JSON
    out_dict = {"results": results, "best_test_f1": float(best_f1),
                "best_where": best_where,
                "decision_verdict": {"code": verdict_code, "message": verdict_msg}}
    out_path = art / "c6_cumulative_metrics.json"

    def _clean(o):
        if isinstance(o, dict):  return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):  return [_clean(v) for v in o]
        if isinstance(o, tuple): return [_clean(v) for v in o]
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)):  return int(o)
        if isinstance(o, np.ndarray):     return o.tolist()
        return o
    with open(out_path, "w") as f:
        json.dump(_clean(out_dict), f, indent=2)
    print(f"\n[SAVE] {out_path}")

    # Markdown table
    md_path = Path(args.output_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    write_markdown_table(md_path, results)

    # MLflow
    if not args.no_mlflow:
        print("\n[MLFL] Logging ...")
        logger = ExperimentLogger(config)
        logger.start_run(
            run_name=args.run_name,
            tags={"tier": "1", "task": "1.5", "baseline_ref": "C6",
                  "verdict": verdict_code},
        )
        logger.log_params_flat({
            "nonTTA_T_opt": t_vals["nonTTA_T_optimal"],
            "tta8_T_opt":   t_vals["tta8_T_optimal"],
            "nonTTA_d1":    th_vals["nonTTA_d1"],
            "nonTTA_d4":    th_vals["nonTTA_d4"],
            "tta8_d1":      th_vals["tta8_d1"],
            "tta8_d4":      th_vals["tta8_d4"],
        })
        flat = {"best_cumulative_test_f1": float(best_f1)}
        for tr in ("nonTTA", "tta8"):
            r = results[tr]
            flat[f"{tr}_baseline_f1"] = float(r["baseline_f1"])
            for label, row in r["rows"].items():
                flat[f"{tr}_{label}_test_f1"] = float(row["f1_macro"])
                flat[f"{tr}_{label}_test_ece"] = float(row["ece_15bins"])
                flat[f"{tr}_{label}_delta_pp"] = float(
                    (row["f1_macro"] - r["baseline_f1"]) * 100
                )
                for n in BIRADS_NAMES:
                    flat[f"{tr}_{label}_test_f1_{n}"] = float(row[f"f1_{n}"])
        logger.log_metrics(flat)
        logger.log_artifact(str(out_path))
        logger.log_artifact(str(md_path))
        logger.end_run()
        print("[MLFL] Done.")

    print("\n[DONE] Task 1.5 complete.")
    print(f"\n  >>> Decision: {verdict_code}")
    print(f"  >>> {verdict_msg}")


if __name__ == "__main__":
    main()
