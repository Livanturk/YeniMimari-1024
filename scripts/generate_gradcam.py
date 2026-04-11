"""
Saved checkpoint'tan Grad-CAM görselleştirmeleri üretir.
Eğitim sırasında GradCAM üretilmemişse bu script ile sonradan üretilebilir.

Kullanım:
    python scripts/generate_gradcam.py --config <config.yaml> --device <gpu_id>
"""

import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import create_dataloaders
from models.full_model import build_model
from models.gradcam import generate_gradcam_for_patient, save_gradcam_visualization
from train import load_config, apply_output_dirs


def main():
    parser = argparse.ArgumentParser(description="Checkpoint'tan Grad-CAM üret")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=20)
    args = parser.parse_args()

    config = load_config(args.config)
    experiment_name = apply_output_dirs(config, args.config)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # Checkpoint yolu
    checkpoint_dir = config["checkpoint"]["save_dir"]
    ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"[HATA] Checkpoint bulunamadı: {ckpt_path}")
        sys.exit(1)

    print(f"[BİLGİ] Config: {args.config}")
    print(f"[BİLGİ] Checkpoint: {ckpt_path}")
    print(f"[BİLGİ] Cihaz: {device}")

    # Model oluştur ve checkpoint yükle
    model = build_model(config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[BİLGİ] Model yüklendi (epoch {ckpt.get('epoch', '?')})")

    # Test dataloader
    dataloaders = create_dataloaders(config)
    test_loader = dataloaders["test"]

    # GradCAM üret
    gradcam_dir = config.get("visualization", {}).get("gradcam", {}).get(
        "save_dir", f"outputs/{experiment_name}/gradcam"
    )
    os.makedirs(gradcam_dir, exist_ok=True)

    num_classes = config["model"]["classification"]["num_classes"]
    samples_per_class = args.num_samples // num_classes
    class_counts = {c: 0 for c in range(num_classes)}
    samples_done = 0

    print(f"[BİLGİ] Grad-CAM üretiliyor ({args.num_samples} örnek, {gradcam_dir})...")

    for batch in test_loader:
        if samples_done >= args.num_samples:
            break

        images = batch["images"].to(device)
        labels = batch["label"]
        patient_ids = batch["patient_id"]

        with torch.no_grad():
            outputs = model(images)
            preds = outputs["full_logits"].argmax(dim=-1)
            confidences = outputs.get("confidence", torch.zeros(len(labels)))

        for i in range(len(labels)):
            if samples_done >= args.num_samples:
                break

            label_class = labels[i].item()
            if class_counts[label_class] >= samples_per_class:
                continue

            try:
                heatmaps = generate_gradcam_for_patient(
                    model=model,
                    images=images[i:i+1],
                    target_class=preds[i].item(),
                )

                original_views = {
                    name: images[i, j]
                    for j, name in enumerate(["RCC", "LCC", "RMLO", "LMLO"])
                }

                save_gradcam_visualization(
                    original_images=original_views,
                    heatmaps=heatmaps,
                    patient_id=patient_ids[i],
                    true_label=labels[i].item(),
                    pred_label=preds[i].item(),
                    confidence=confidences[i].item(),
                    save_dir=gradcam_dir,
                )
                class_counts[label_class] += 1
                samples_done += 1
                print(f"  [{samples_done}/{args.num_samples}] {patient_ids[i]} (class {label_class})")
            except Exception as e:
                print(f"  [UYARI] Grad-CAM hatası ({patient_ids[i]}): {e}")

    print(f"[BİLGİ] Tamamlandı: {samples_done} Grad-CAM görselleştirmesi → {gradcam_dir}")


if __name__ == "__main__":
    main()
