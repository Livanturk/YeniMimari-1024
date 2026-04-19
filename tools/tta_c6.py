"""
Tier 1 Task 1.1 — Test-Time Augmentation for C6
================================================
8-view TTA with view-swap-aware horizontal flip.

TTA configurations:
    0. identity
    1. hflip + view-swap (RCC↔LCC, RMLO↔LMLO)
    2. rotate +5°
    3. rotate -5°
    4. rotate +10°
    5. rotate -10°
    6. hflip+swap + rotate +5°
    7. hflip+swap + rotate -5°

Rotation fill value = -mean/std (background-equivalent after normalization).
Letterbox zero pixels map to this value in the normalized tensor.

Outputs 4-view and 8-view ablation metrics; caches TTA-averaged per-head
logits for downstream Tier 1 tasks (1.4 binary gating may use them).

Usage:
    python tools/tta_c6.py \
        --config configs/experiment_v2_birads/convnextv2_large_8bit_ablation_c6.yaml \
        --checkpoint outputs/convnextv2_large_8bit_ablation_c6/checkpoints/best_model.pt \
        --output-dir artifacts \
        --run-name tier1_task1_1_tta_c6 \
        --experiment-name birads-inference-pipeline \
        --device 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Proje root
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF
import yaml
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from data.dataset import create_dataloaders
from data.transforms import _get_norm_stats
from models.full_model import build_model
from utils.mlflow_logger import ExperimentLogger


BIRADS_NAMES = ["BIRADS_1", "BIRADS_2", "BIRADS_4", "BIRADS_5"]


# ---------------------------------------------------------------------------
# TTA primitives
# ---------------------------------------------------------------------------

VIEW_SWAP_IDX = [1, 0, 3, 2]  # RCC↔LCC, RMLO↔LMLO


def tta_identity(images: torch.Tensor, fill: float) -> torch.Tensor:
    return images


def tta_hflip_swap(images: torch.Tensor, fill: float) -> torch.Tensor:
    """Horizontal flip each image, then swap view order to preserve L/R semantics."""
    flipped = images.flip(dims=[-1])          # (B, 4, 3, H, W)
    return flipped[:, VIEW_SWAP_IDX]


def _rotate_batch(images: torch.Tensor, degrees: float, fill: float) -> torch.Tensor:
    """Rotate every (3, H, W) image by `degrees` with background fill."""
    B, V, C, H, W = images.shape
    # Tek batch dim'inde dön: (B*V, C, H, W)
    flat = images.reshape(B * V, C, H, W)
    rot = TF.rotate(
        flat,
        angle=float(degrees),
        interpolation=TF.InterpolationMode.BILINEAR,
        expand=False,
        fill=[float(fill)] * C,
    )
    return rot.reshape(B, V, C, H, W)


def tta_rot(degrees: float):
    def _f(images: torch.Tensor, fill: float) -> torch.Tensor:
        return _rotate_batch(images, degrees, fill)
    return _f


def tta_hflip_swap_then_rot(degrees: float):
    def _f(images: torch.Tensor, fill: float) -> torch.Tensor:
        return _rotate_batch(tta_hflip_swap(images, fill), degrees, fill)
    return _f


TTA_CONFIGS_8 = [
    ("identity",            tta_identity),
    ("hflip_swap",          tta_hflip_swap),
    ("rot_p5",              tta_rot(+5)),
    ("rot_m5",              tta_rot(-5)),
    ("rot_p10",             tta_rot(+10)),
    ("rot_m10",             tta_rot(-10)),
    ("hflip_swap_rot_p5",   tta_hflip_swap_then_rot(+5)),
    ("hflip_swap_rot_m5",   tta_hflip_swap_then_rot(-5)),
]

TTA_CONFIGS_4 = [
    ("identity",      tta_identity),
    ("hflip_swap",    tta_hflip_swap),
    ("rot_p5",        tta_rot(+5)),
    ("rot_m5",        tta_rot(-5)),
]


# ---------------------------------------------------------------------------
# Metrics (aligned with extract_c6_logits.py)
# ---------------------------------------------------------------------------

def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def expected_calibration_error(probs, labels, n_bins=15):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def brier(probs, labels, K=4):
    oh = np.eye(K)[labels]
    return float(np.mean(np.sum((probs - oh) ** 2, axis=1)))


def compute_metrics_from_probs(probs: np.ndarray, labels: np.ndarray) -> dict:
    preds = probs.argmax(axis=1)
    m = {}
    m["accuracy"] = float(accuracy_score(labels, preds))
    m["f1_macro"] = float(f1_score(labels, preds, average="macro", zero_division=0))
    pr, rc, f1, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
    for i, name in enumerate(BIRADS_NAMES):
        if i < len(f1):
            m[f"f1_{name}"] = float(f1[i])
            m[f"recall_{name}"] = float(rc[i])
    bin_preds = (preds >= 2).astype(int)
    bin_labels = (labels >= 2).astype(int)
    m["binary_f1"] = float(f1_score(bin_labels, bin_preds, zero_division=0))
    m["cohen_kappa"] = float(cohen_kappa_score(labels, preds))
    try:
        m["auc_roc_macro"] = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
    except ValueError:
        m["auc_roc_macro"] = float("nan")
    m["ece_15bins"] = expected_calibration_error(probs, labels, n_bins=15)
    m["brier"] = brier(probs, labels, K=4)
    m["mean_confidence"] = float(probs.max(axis=1).mean())
    return m


# ---------------------------------------------------------------------------
# Forward with TTA
# ---------------------------------------------------------------------------

@torch.no_grad()
def forward_tta(model, loader, device, tta_configs, fill_value, split_name):
    """
    Her batch için tüm TTA view'larını uygular, LOGIT düzeyinde ortalar.

    Logit averaging (softmax değil) seçimi:
      - Task 1.5 cumulative pipeline: TTA_avg → T-scale → gating → threshold.
        T-scale sadece logit düzeyinde matematiksel anlamlı.
      - Per-view logit cache'i, Task 1.2 sonucu (T_optimal) ile downstream
        task'ların GPU pass tekrarı yapmadan offline çalışmasına olanak verir.
      - Numerik olarak softmax averaging'e çok yakın (small Jensen gap),
        4-class problem için pratik fark < 0.1pp.

    Returns:
        dict:
            - "full_logits":       (N, 4) TTA-averaged
            - "binary_logits":     (N, 2)
            - "benign_sub_logits": (N, 2)
            - "malign_sub_logits": (N, 2)
            - "labels":            (N,)
            - "per_view_full_logits": (V, N, 4)   # offline retry için
    """
    all_full, all_bin, all_ben, all_mal, all_lbl = [], [], [], [], []
    all_per_view = [[] for _ in tta_configs]
    n_batches = len(loader)

    for i, batch in enumerate(loader):
        imgs = batch["images"].to(device, non_blocking=True)
        lbl = batch["label"]

        full_sum = None
        bin_sum = None
        ben_sum = None
        mal_sum = None

        for v_idx, (v_name, v_fn) in enumerate(tta_configs):
            aug = v_fn(imgs, fill_value)
            out = model(aug)
            fl = out["full_logits"].float()
            bl = out["binary_logits"].float()
            bnl = out["benign_sub_logits"].float()
            mll = out["malign_sub_logits"].float()

            all_per_view[v_idx].append(fl.cpu().numpy())

            if full_sum is None:
                full_sum = fl.clone()
                bin_sum = bl.clone()
                ben_sum = bnl.clone()
                mal_sum = mll.clone()
            else:
                full_sum += fl
                bin_sum += bl
                ben_sum += bnl
                mal_sum += mll

        V = len(tta_configs)
        all_full.append((full_sum / V).cpu().numpy())
        all_bin.append((bin_sum / V).cpu().numpy())
        all_ben.append((ben_sum / V).cpu().numpy())
        all_mal.append((mal_sum / V).cpu().numpy())
        all_lbl.append(lbl.numpy())

        if (i + 1) % 50 == 0 or (i + 1) == n_batches:
            print(f"  [{split_name}] batch {i+1}/{n_batches}  (×{V} views)")

    return {
        "full_logits":       np.concatenate(all_full, axis=0),
        "binary_logits":     np.concatenate(all_bin, axis=0),
        "benign_sub_logits": np.concatenate(all_ben, axis=0),
        "malign_sub_logits": np.concatenate(all_mal, axis=0),
        "labels":            np.concatenate(all_lbl, axis=0),
        "per_view_full_logits": np.stack(
            [np.concatenate(x, axis=0) for x in all_per_view], axis=0
        ),  # (V, N, 4)
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--run-name", default="tier1_task1_1_tta_c6")
    parser.add_argument("--experiment-name", default="birads-inference-pipeline")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--skip-val", action="store_true", help="Val TTA'yı atla (baseline match için)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f"cuda:{args.device}")
        print(f"[DEV ] GPU: {torch.cuda.get_device_name(args.device)}")
    else:
        device = torch.device("cpu")
        print("[DEV ] CPU")

    # Config & model
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config.setdefault("mlflow", {})["experiment_name"] = args.experiment_name

    mean, std = _get_norm_stats(config["data"])
    # Background-equivalent fill value after normalization (scalar; all 3 channels identical)
    fill_value = float(-mean[0] / std[0])
    print(f"[NORM] mean={mean[0]:.4f}, std={std[0]:.4f}, fill_value (bg-eq)={fill_value:.4f}")

    loaders = create_dataloaders(config)
    model = build_model(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=False)
    model.eval()
    T_trained = float(torch.exp(model.classifier.log_temperature).detach().cpu().item())
    print(f"[MODEL] Learned temperature: {T_trained:.4f} (will use T=1.0 here; temp scaling is Task 1.2)")

    # Baseline (from cache) for sanity comparison
    baseline_path = out_dir / "c6_baseline_metrics.json"
    baseline_metrics = None
    if baseline_path.exists():
        baseline_metrics = json.load(open(baseline_path))
        print(f"[BASE] Loaded baseline metrics from {baseline_path}")

    # OPTIMIZATION: tta4 = first 4 views of tta8. Tek forward pass yap, tta4 offline türet.
    all_results = {}
    splits_to_run = ["test"] if args.skip_val else ["val", "test"]

    # ---- tta8 forward passes (val + test) ----
    print(f"\n{'='*64}\n  Running tta8 ({len(TTA_CONFIGS_8)} views): "
          f"{[n for n,_ in TTA_CONFIGS_8]}\n{'='*64}")

    tta8_results = {}
    out_by_split = {}

    for split in splits_to_run:
        print(f"\n[FWD ] tta8 / {split} ...")
        out = forward_tta(
            model, loaders[split], device, TTA_CONFIGS_8, fill_value, split,
        )
        out_by_split[split] = out

        probs = softmax_np(out["full_logits"], axis=-1)
        metrics = compute_metrics_from_probs(probs, out["labels"])
        tta8_results[split] = metrics

        np.save(out_dir / f"c6_{split}_tta8_full_logits.npy",       out["full_logits"])
        np.save(out_dir / f"c6_{split}_tta8_binary_logits.npy",     out["binary_logits"])
        np.save(out_dir / f"c6_{split}_tta8_benign_sub_logits.npy", out["benign_sub_logits"])
        np.save(out_dir / f"c6_{split}_tta8_malign_sub_logits.npy", out["malign_sub_logits"])
        np.save(out_dir / f"c6_{split}_tta8_per_view_full_logits.npy",
                out["per_view_full_logits"])

        print(f"  [tta8/{split}] F1 macro = {metrics['f1_macro']:.4f}, "
              "per-class F1 = "
              + ", ".join(f"{n.split('_')[1]}={metrics[f'f1_{n}']:.3f}" for n in BIRADS_NAMES))

    # Confusion matrix tta8 test
    if "test" in tta8_results:
        test_probs = softmax_np(out_by_split["test"]["full_logits"], axis=-1)
        tta8_results["confusion_matrix_test"] = confusion_matrix(
            out_by_split["test"]["labels"], test_probs.argmax(axis=1), labels=[0, 1, 2, 3]
        ).tolist()
    all_results["tta8"] = tta8_results

    # ---- tta4 derived offline from tta8 per-view logits (first 4 views are identical) ----
    print(f"\n{'='*64}\n  Deriving tta4 offline (first 4 of tta8): "
          f"{[n for n,_ in TTA_CONFIGS_4]}\n{'='*64}")
    tta4_results = {}
    for split in splits_to_run:
        per_view = out_by_split[split]["per_view_full_logits"][:4]  # (4, N, 4)
        avg_logits = per_view.mean(axis=0)
        probs = softmax_np(avg_logits, axis=-1)
        metrics = compute_metrics_from_probs(probs, out_by_split[split]["labels"])
        tta4_results[split] = metrics
        print(f"  [tta4/{split}] F1 macro = {metrics['f1_macro']:.4f}, "
              "per-class F1 = "
              + ", ".join(f"{n.split('_')[1]}={metrics[f'f1_{n}']:.3f}" for n in BIRADS_NAMES))
    if "test" in tta4_results:
        per_view = out_by_split["test"]["per_view_full_logits"][:4]
        probs = softmax_np(per_view.mean(axis=0), axis=-1)
        tta4_results["confusion_matrix_test"] = confusion_matrix(
            out_by_split["test"]["labels"], probs.argmax(axis=1), labels=[0, 1, 2, 3]
        ).tolist()
    all_results["tta4"] = tta4_results

    # ---------- Summary ----------
    print("\n" + "=" * 64)
    print("  Task 1.1 TTA — Summary vs C6 baseline")
    print("=" * 64)
    base_test_f1 = baseline_metrics["test"]["f1_macro"] if baseline_metrics else None
    for tag, r in all_results.items():
        if "test" in r:
            t = r["test"]
            delta = (t["f1_macro"] - base_test_f1) * 100 if base_test_f1 is not None else None
            delta_str = f"({delta:+.2f}pp)" if delta is not None else ""
            print(f"  [{tag}] Test F1={t['f1_macro']:.4f} {delta_str}  "
                  f"BR1={t['f1_BIRADS_1']:.3f} BR2={t['f1_BIRADS_2']:.3f} "
                  f"BR4={t['f1_BIRADS_4']:.3f} BR5={t['f1_BIRADS_5']:.3f}  "
                  f"ECE={t['ece_15bins']:.3f}")

    # Per-view incremental ablation (8-view only, test)
    per_view_path = out_dir / "c6_test_tta8_per_view_full_logits.npy"
    if per_view_path.exists():
        per_view = np.load(per_view_path)  # (V=8, N, 4) — LOGITS
        labels = np.load(out_dir / "c6_test_labels.npy")
        print("\n  Per-view incremental (mean of first k view LOGITS, then softmax):")
        running_sum = None
        inc_table = []
        for k, (name, _) in enumerate(TTA_CONFIGS_8, start=1):
            running_sum = per_view[k - 1] if running_sum is None else running_sum + per_view[k - 1]
            probs_k = softmax_np(running_sum / k, axis=-1)
            f1 = f1_score(labels, probs_k.argmax(axis=1), average="macro", zero_division=0)
            delta = (f1 - base_test_f1) * 100 if base_test_f1 is not None else None
            inc_table.append({"k": k, "added": name, "test_f1_macro": float(f1),
                              "delta_pp": float(delta) if delta is not None else None})
            delta_s = f"({delta:+.2f}pp)" if delta is not None else ""
            print(f"    k={k}  +{name:<22s}  F1={f1:.4f} {delta_s}")
        all_results["incremental_test"] = inc_table

    # Dump summary JSON
    out_json = {
        "baseline_test_f1_macro": base_test_f1,
        "results": all_results,
        "tta_configs_8": [n for n, _ in TTA_CONFIGS_8],
        "tta_configs_4": [n for n, _ in TTA_CONFIGS_4],
        "fill_value": fill_value,
        "temperature_used": 1.0,
    }
    with open(out_dir / "c6_tta_metrics.json", "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"\n[SAVE] {out_dir}/c6_tta_metrics.json")

    # MLflow
    if not args.no_mlflow:
        print("\n[MLFL] Logging to MLflow ...")
        logger = ExperimentLogger(config)
        logger.start_run(
            run_name=args.run_name,
            tags={"tier": "1", "task": "1.1", "baseline_ref": "C6",
                  "tta_views": "8", "view_swap_hflip": "true"},
        )
        logger.log_params_flat({
            "config_path": args.config, "checkpoint_path": args.checkpoint,
            "fill_value_bgeq": fill_value, "temperature_used": 1.0,
        })
        flat = {}
        if base_test_f1 is not None:
            flat["baseline_test_f1_macro"] = float(base_test_f1)
        for tag, r in all_results.items():
            if not isinstance(r, dict):
                continue
            for split in ("val", "test"):
                if split in r and isinstance(r[split], dict):
                    for k, v in r[split].items():
                        if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
                            flat[f"{tag}_{split}_{k}"] = float(v)
        if base_test_f1 is not None and "tta8" in all_results:
            flat["tta8_test_delta_pp"] = float(
                (all_results["tta8"]["test"]["f1_macro"] - base_test_f1) * 100
            )
            flat["tta4_test_delta_pp"] = float(
                (all_results["tta4"]["test"]["f1_macro"] - base_test_f1) * 100
            )
        logger.log_metrics(flat)
        logger.log_artifact(str(out_dir / "c6_tta_metrics.json"))
        logger.end_run()
        print("[MLFL] Done.")

    print("\n[DONE] Task 1.1 TTA complete.")


if __name__ == "__main__":
    main()
