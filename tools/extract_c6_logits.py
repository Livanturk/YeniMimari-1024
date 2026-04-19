"""
Tier 0 Task 0.1 + Tier 1 Common Pre-req
========================================
C6 checkpoint ile val + test set'te forward pass. Raw per-head logit'leri
disk'e cacheliyor (sonraki Tier 1 task'ları bu cache'i kullanacak) ve
bug-fix sonrası baseline test F1'i MLflow'a loglar.

Kullanım:
    python tools/extract_c6_logits.py \
        --config configs/experiment_v2_birads/convnextv2_large_8bit_ablation_c6.yaml \
        --checkpoint outputs/convnextv2_large_8bit_ablation_c6/checkpoints/best_model.pt \
        --output-dir artifacts \
        --run-name tier0_c6_baseline_postfix

Bug-fix notu:
    ensemble_evaluate.py ImageNet stats kullanıyordu (yanlış). Bu script
    data/transforms.py'deki get_val_transforms() helper'ını kullanıyor;
    bu helper bit_depth=8 için DATASET_STATS_8BIT (mean=0.1210/std=0.1977)
    döndürür.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Proje root'unu sys.path'e ekle (script tools/ altından çalıştırılıyor)
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
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
from models.full_model import build_model
from utils.mlflow_logger import ExperimentLogger


BIRADS_NAMES = ["BIRADS_1", "BIRADS_2", "BIRADS_4", "BIRADS_5"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    meta = {
        "epoch": ckpt.get("epoch"),
        "metrics": ckpt.get("metrics", {}),
    }
    print(f"[CKPT] Loaded {ckpt_path} (epoch={meta['epoch']})")
    return meta


@torch.no_grad()
def forward_pass(model: torch.nn.Module, loader, device: torch.device, split: str) -> dict:
    """Returns dict of per-head logits and labels as numpy arrays."""
    full_list, bin_list, ben_list, mal_list, labels_list = [], [], [], [], []
    n_batches = len(loader)
    for i, batch in enumerate(loader):
        imgs = batch["images"].to(device, non_blocking=True)
        lbl = batch["label"]
        out = model(imgs)
        full_list.append(out["full_logits"].float().cpu().numpy())
        bin_list.append(out["binary_logits"].float().cpu().numpy())
        ben_list.append(out["benign_sub_logits"].float().cpu().numpy())
        mal_list.append(out["malign_sub_logits"].float().cpu().numpy())
        labels_list.append(lbl.numpy())
        if (i + 1) % 50 == 0 or (i + 1) == n_batches:
            print(f"  [{split}] batch {i+1}/{n_batches}")
    return {
        "full": np.concatenate(full_list, axis=0),
        "binary": np.concatenate(bin_list, axis=0),
        "benign_sub": np.concatenate(ben_list, axis=0),
        "malign_sub": np.concatenate(mal_list, axis=0),
        "labels": np.concatenate(labels_list, axis=0),
    }


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Standart ECE, 15 bin, argmax-based confidence."""
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
        acc_bin = correct[mask].mean()
        conf_bin = conf[mask].mean()
        ece += (mask.sum() / n) * abs(acc_bin - conf_bin)
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray, num_classes: int = 4) -> float:
    one_hot = np.eye(num_classes)[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def compute_metrics(full_logits: np.ndarray, labels: np.ndarray, temperature: float) -> dict:
    """Apply model's learned temperature to full_logits, compute full metric suite."""
    scaled = full_logits / temperature
    probs = softmax_np(scaled, axis=-1)
    preds = probs.argmax(axis=1)

    m = {}
    m["accuracy"] = float(accuracy_score(labels, preds))
    m["f1_macro"] = float(f1_score(labels, preds, average="macro", zero_division=0))
    pr, rc, f1, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
    for i, name in enumerate(BIRADS_NAMES):
        if i < len(f1):
            m[f"f1_{name}"] = float(f1[i])
            m[f"precision_{name}"] = float(pr[i])
            m[f"recall_{name}"] = float(rc[i])

    # Binary (derived from argmax: class_idx >= 2 → malign)
    bin_preds = (preds >= 2).astype(int)
    bin_labels = (labels >= 2).astype(int)
    m["binary_f1"] = float(f1_score(bin_labels, bin_preds, zero_division=0))
    m["binary_accuracy"] = float(accuracy_score(bin_labels, bin_preds))

    m["cohen_kappa"] = float(cohen_kappa_score(labels, preds))
    try:
        m["auc_roc_macro"] = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
    except ValueError:
        m["auc_roc_macro"] = float("nan")

    # Calibration
    m["ece_15bins"] = expected_calibration_error(probs, labels, n_bins=15)
    m["brier"] = brier_score(probs, labels, num_classes=4)
    m["mean_confidence"] = float(probs.max(axis=1).mean())

    return m, preds


def save_cache(split_data: dict, split_name: str, out_dir: Path) -> None:
    prefix = f"c6_{split_name}"
    np.save(out_dir / f"{prefix}_logits.npy", split_data["full"])
    np.save(out_dir / f"{prefix}_labels.npy", split_data["labels"])
    np.save(out_dir / f"{prefix}_binary_logits.npy", split_data["binary"])
    np.save(out_dir / f"{prefix}_benign_sub_logits.npy", split_data["benign_sub"])
    np.save(out_dir / f"{prefix}_malign_sub_logits.npy", split_data["malign_sub"])
    print(f"[SAVE] {split_name}: {split_data['full'].shape[0]} samples → {out_dir}/{prefix}_*.npy")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--run-name", default="tier0_c6_baseline_postfix")
    parser.add_argument("--experiment-name", default="birads-inference-pipeline")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--no-mlflow", action="store_true", help="MLflow logging'i atla (debug için)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f"cuda:{args.device}")
        print(f"[DEV ] GPU: {torch.cuda.get_device_name(args.device)}")
    else:
        device = torch.device("cpu")
        print(f"[DEV ] CPU")

    # Config
    config = load_config(args.config)
    config.setdefault("mlflow", {})["experiment_name"] = args.experiment_name

    # Data
    print(f"[DATA] Creating dataloaders via data.dataset.create_dataloaders ...")
    loaders = create_dataloaders(config)
    n_val = len(loaders["val"].dataset)
    n_test = len(loaders["test"].dataset)
    print(f"[DATA] val={n_val} patients, test={n_test} patients")

    # Model
    print(f"[MODEL] Building model from config ...")
    model = build_model(config).to(device)
    meta = load_checkpoint(model, args.checkpoint, device)
    model.eval()

    # Read trained temperature from model
    T_trained = float(torch.exp(model.classifier.log_temperature).detach().cpu().item())
    print(f"[MODEL] Learned temperature (C6 final): {T_trained:.4f}")

    # Forward passes
    print("[FWD ] Val set ...")
    val_data = forward_pass(model, loaders["val"], device, "val")
    print("[FWD ] Test set ...")
    test_data = forward_pass(model, loaders["test"], device, "test")

    # Save cache
    save_cache(val_data, "val", out_dir)
    save_cache(test_data, "test", out_dir)

    # Also save metadata
    meta_json = {
        "config_path": args.config,
        "checkpoint_path": args.checkpoint,
        "ckpt_epoch": meta.get("epoch"),
        "learned_temperature": T_trained,
        "val_n": int(n_val),
        "test_n": int(n_test),
        "class_ordering_full": ["BIRADS_1", "BIRADS_2", "BIRADS_4", "BIRADS_5"],
        "class_ordering_binary": ["Benign", "Malign"],
        "class_ordering_benign_sub": ["BIRADS_1", "BIRADS_2"],
        "class_ordering_malign_sub": ["BIRADS_4", "BIRADS_5"],
    }
    with open(out_dir / "c6_cache_meta.json", "w") as f:
        json.dump(meta_json, f, indent=2)

    # Compute metrics
    print("\n[METR] Computing val / test metrics with learned temperature ...")
    val_metrics, _ = compute_metrics(val_data["full"], val_data["labels"], T_trained)
    test_metrics, test_preds = compute_metrics(test_data["full"], test_data["labels"], T_trained)

    val_test_gap = val_metrics["f1_macro"] - test_metrics["f1_macro"]

    # Print summary
    print("\n" + "=" * 64)
    print("  Bug-Fix Post C6 Baseline")
    print("=" * 64)
    print(f"  Learned T            : {T_trained:.4f}")
    print(f"  Val  F1 macro        : {val_metrics['f1_macro']:.4f}")
    print(f"  Test F1 macro        : {test_metrics['f1_macro']:.4f}")
    print(f"  Val–Test gap         : {val_test_gap*100:+.2f}pp")
    print(f"  Test per-class F1    : "
          + ", ".join(f"{n}={test_metrics[f'f1_{n}']:.3f}" for n in BIRADS_NAMES))
    print(f"  Test Binary F1       : {test_metrics['binary_f1']:.4f}")
    print(f"  Test Cohen's kappa   : {test_metrics['cohen_kappa']:.4f}")
    print(f"  Test AUC-ROC (macro) : {test_metrics['auc_roc_macro']:.4f}")
    print(f"  Test ECE (15 bins)   : {test_metrics['ece_15bins']:.4f}")
    print(f"  Test Brier           : {test_metrics['brier']:.4f}")
    print(f"  Test mean confidence : {test_metrics['mean_confidence']:.4f}")
    print(f"  Val  ECE (15 bins)   : {val_metrics['ece_15bins']:.4f}")
    print(f"  Val  mean confidence : {val_metrics['mean_confidence']:.4f}")

    # Confusion matrix
    cm = confusion_matrix(test_data["labels"], test_preds, labels=[0, 1, 2, 3])
    print("\n  Confusion matrix (test):")
    print("             pred_BR1 pred_BR2 pred_BR4 pred_BR5")
    for i, name in enumerate(["true_BR1", "true_BR2", "true_BR4", "true_BR5"]):
        row = " ".join(f"{v:>8d}" for v in cm[i])
        print(f"   {name}:  {row}   (total={cm[i].sum()})")

    # Save metrics JSON
    metrics_json = {
        "val": val_metrics,
        "test": test_metrics,
        "val_test_gap_f1_macro_pp": val_test_gap * 100.0,
        "confusion_matrix_test": cm.tolist(),
        "learned_temperature": T_trained,
    }
    with open(out_dir / "c6_baseline_metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"\n[SAVE] Metrics → {out_dir}/c6_baseline_metrics.json")

    # MLflow logging
    if not args.no_mlflow:
        print("\n[MLFL] Logging to MLflow ...")
        logger = ExperimentLogger(config)
        logger.start_run(
            run_name=args.run_name,
            tags={
                "tier": "0",
                "task": "0.1",
                "baseline_ref": "C6",
                "checkpoint": args.checkpoint,
                "bug_fix": "normalization_imagenet_to_dataset_stats_8bit",
            },
        )
        logger.log_params_flat({
            "config_path": args.config,
            "checkpoint_path": args.checkpoint,
            "learned_temperature": T_trained,
            "ckpt_epoch": meta.get("epoch"),
        })
        flat_metrics = {}
        for split, m in (("val", val_metrics), ("test", test_metrics)):
            for k, v in m.items():
                if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
                    flat_metrics[f"{split}_{k}"] = float(v)
        flat_metrics["val_test_gap_f1_macro_pp"] = float(val_test_gap * 100.0)
        logger.log_metrics(flat_metrics)

        logger.log_artifact(str(out_dir / "c6_baseline_metrics.json"))
        logger.log_artifact(str(out_dir / "c6_cache_meta.json"))
        logger.end_run()
        print("[MLFL] Done.")

    print("\n[DONE] Tier 0 extraction complete. Cache files:")
    for p in sorted(out_dir.glob("c6_*.npy")):
        print(f"  - {p}")


if __name__ == "__main__":
    main()
