"""
Mammography Image Transforms
=============================
Eğitim ve doğrulama/test için görüntü dönüşüm pipeline'ları.

16-bit PNG desteği:
- Görüntüler dataset.py'de numpy→tensor olarak [0,1] aralığına normalize edilir.
- Bu modül SADECE spatial augmentation ve channel normalization yapar.
- Dataset varyantına (seg/noseg) göre farklı normalizasyon istatistikleri kullanılır.

Mamografi görüntüleri için özel dikkat edilmesi gerekenler:
- Dikey çevirme genelde yapılmaz (anatomik yönelimi bozar).
- Aşırı renk/kontrast değişikliği diagnostik bilgiyi bozabilir.
- Grayscale görüntüler 3 kanala kopyalandığı için, 3 kanala da aynı
  normalizasyon değeri uygulanmalıdır.
"""

import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF


# ─── Eski 512×512 16-bit dataset istatistikleri (geriye uyumluluk) ────────────
# Dataset_512 / Dataset_512_Test için. "seg" = segmentasyon maskeli, "noseg" = maskesiz.
DATASET_STATS_512 = {
    "seg": {
        "mean": [0.0921, 0.0921, 0.0921],
        "std":  [0.1520, 0.1520, 0.1520],
    },
    "noseg": {
        "mean": [0.0718, 0.0718, 0.0718],
        "std":  [0.1499, 0.1499, 0.1499],
    },
}

# ─── 1024×1024 8-bit dataset istatistikleri (all-pixel, 0-1 scale) ───────────
# Dataset_1024_8bit (8,557 hasta). CLAHE + letterbox 1024×1024 + 8-bit PNG.
# Train all-pixel: mean=0.1210, std=0.1977 | tissue: mean=0.3512, std=0.1804
DATASET_STATS_8BIT = {
    "noseg": {
        "mean": [0.1210, 0.1210, 0.1210],
        "std":  [0.1977, 0.1977, 0.1977],
    },
}

# ─── 1024×1024 16-bit dataset istatistikleri (all-pixel, 0-1 scale) ──────────
# Dataset_1024_16bit (7,557 hasta). Aynı pipeline, 16-bit PNG çıktı.
# Train all-pixel: mean=0.1220, std=0.2044 | tissue: mean=0.3540, std=0.1978
# Test  all-pixel: mean=0.1247, std=0.2051 | tissue: mean=0.3554, std=0.1945
# Train-Test farkı ihmal edilebilir — train istatistikleri her ikisinde kullanılır.
DATASET_STATS_16BIT = {
    "noseg": {
        "mean": [0.1220, 0.1220, 0.1220],
        "std":  [0.2044, 0.2044, 0.2044],
    },
}

# Fallback: hiçbir eşleşme yoksa (genişletilebilirlik için korunur)
IMAGENET_MEAN = [0.449, 0.449, 0.449]
IMAGENET_STD = [0.226, 0.226, 0.226]


def _get_norm_stats(data_cfg: dict) -> tuple:
    """
    Config'den uygun normalizasyon istatistiklerini döndürür.

    Seçim mantığı:
        bit_depth=8  → DATASET_STATS_8BIT (1024×1024 8-bit pipeline)
        bit_depth=16, image_size≥1024 → DATASET_STATS_16BIT (1024×1024 16-bit pipeline)
        bit_depth=16, image_size<1024  → DATASET_STATS_512  (eski 512×512 16-bit pipeline)
        Hiçbiri eşleşmezse → ImageNet istatistikleri (fallback)
    """
    bit_depth = data_cfg.get("bit_depth", 8)
    variant = data_cfg.get("dataset_variant", "noseg")
    image_size = data_cfg.get("image_size", 1024)

    if bit_depth == 8 and variant in DATASET_STATS_8BIT:
        stats = DATASET_STATS_8BIT[variant]
    elif bit_depth == 16 and image_size >= 1024 and variant in DATASET_STATS_16BIT:
        stats = DATASET_STATS_16BIT[variant]
    elif bit_depth == 16 and variant in DATASET_STATS_512:
        stats = DATASET_STATS_512[variant]
    else:
        return IMAGENET_MEAN, IMAGENET_STD

    return stats["mean"], stats["std"]


def get_train_transforms(data_cfg: dict) -> torch.nn.Module:
    """
    Eğitim seti için augmentation pipeline'ı oluşturur.

    16-bit modda girdi zaten (3, H, W) float32 tensor olarak gelir.
    Transform'lar torchvision.transforms.v2 ile tensor üzerinde çalışır.
    """
    aug = data_cfg.get("augmentation", {})
    img_size = data_cfg["image_size"]
    mean, std = _get_norm_stats(data_cfg)

    transform_list = [
        T.Resize((img_size, img_size), antialias=True),
    ]

    if aug.get("enabled", True):
        if aug.get("horizontal_flip", 0) > 0:
            transform_list.append(
                T.RandomHorizontalFlip(p=aug["horizontal_flip"])
            )

        if aug.get("rotation_degrees", 0) > 0:
            transform_list.append(
                T.RandomRotation(degrees=aug["rotation_degrees"])
            )

        brightness = aug.get("brightness", 0)
        contrast = aug.get("contrast", 0)
        if brightness > 0 or contrast > 0:
            transform_list.append(
                T.ColorJitter(brightness=brightness, contrast=contrast)
            )

        transform_list.append(
            T.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
            )
        )

    # Normalizasyon (tensor üzerinde çalışır)
    transform_list.append(T.Normalize(mean=mean, std=std))

    # Random erasing
    if aug.get("enabled", True) and aug.get("random_erasing", 0) > 0:
        transform_list.append(
            T.RandomErasing(p=aug["random_erasing"])
        )

    return T.Compose(transform_list)


def get_val_transforms(data_cfg: dict) -> torch.nn.Module:
    """
    Doğrulama ve test seti için transform pipeline'ı.
    Augmentation uygulanmaz, sadece resize ve normalize yapılır.
    """
    img_size = data_cfg["image_size"]
    mean, std = _get_norm_stats(data_cfg)

    return T.Compose([
        T.Resize((img_size, img_size), antialias=True),
        T.Normalize(mean=mean, std=std),
    ])


def get_inverse_normalize(data_cfg: dict = None) -> T.Normalize:
    """
    Normalize işlemini tersine çevirir (Grad-CAM görselleştirmesi için).
    """
    if data_cfg is not None:
        mean, std = _get_norm_stats(data_cfg)
    else:
        mean, std = IMAGENET_MEAN, IMAGENET_STD

    inv_mean = [-m / s for m, s in zip(mean, std)]
    inv_std = [1.0 / s for s in std]
    return T.Normalize(mean=inv_mean, std=inv_std)
