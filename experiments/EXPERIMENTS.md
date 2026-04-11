# Deney İndeksi

> Otomatik oluşturulmuştur. `python scripts/report.py index` ile güncellenir.

Son güncelleme: 2026-04-11

| Durum | Deney | Tarih | Val F1 | Test F1 | Backbone | Baseline |
|:---:|---|---|:---:|:---:|---|---|
| ✅ | [convnextv2_large_8bit_ablation_a1](convnextv2_large_8bit_ablation_a1/report.md) | 2026-04-09 | 0.7108 | 0.6270 | `convnextv2_large` | `convnextv2_large_seg_deformable_asymmetry_malignonly_v1` |
| ✅ | [convnextv2_large_8bit_ablation_a1_ce](convnextv2_large_8bit_ablation_a1_ce/report.md) | 2026-04-09 | 0.7141 | 0.6370 | `convnextv2_large` | `convnextv2_large_8bit_ablation_a1` |
| ✅ | [convnextv2_large_seg_deformable_asymmetry_malignonly_v1](convnextv2_large_seg_deformable_asymmetry_malignonly_v1/report.md) | 2026-03-16 | 0.6867 | 0.7233 | `convnextv2_large` | `convnextv2_large_seg_deformable_v1` |
| ✅ | [convnextv2_large_seg_deformable_asymmetry_malignonly_v2](convnextv2_large_seg_deformable_asymmetry_malignonly_v2/report.md) | 2026-03-17 | 0.6959 | 0.7018 | `convnextv2_large` | `convnextv2_large_seg_deformable_asymmetry_malignonly_v1` |
| ✅ | [convnextv2_large_seg_deformable_v1](convnextv2_large_seg_deformable_v1/report.md) | 2026-03-15 | 0.6863 | 0.7029 | `convnextv2_large` | `convnextv2_base_original` |
| ✅ | [dinov2_vitl_8bit_ablation_a3](dinov2_vitl_8bit_ablation_a3/report.md) | 2026-04-09 | 0.6940 | 0.6325 | `vit_large_patch14_dinov2` | `convnextv2_large_8bit_ablation_a1` |
| ✅ | [ensemble_3model_convnextv2_large](ensemble_3model_convnextv2_large/report.md) | 2026-03-18 | — | 0.7357 | `convnextv2_large (×3)` | `convnextv2_large_seg_deformable_asymmetry_malignonly_v1` |
| ✅ | [convnextv2_large_8bit_ablation_b1](convnextv2_large_8bit_ablation_b1/report.md) | 2026-04-09 | 0.7334 | 0.6387 | `convnextv2_large` | `convnextv2_large_8bit_ablation_a1_ce` |
| ✅ | [dinov2_vitl_8bit_ablation_b2](dinov2_vitl_8bit_ablation_b2/report.md) | 2026-04-09 | 0.6940 | 0.6136 | `vit_large_patch14_dinov2` | `dinov2_vitl_8bit_ablation_a3` |
| ✅ | [convnextv2_large_8bit_ablation_b3](convnextv2_large_8bit_ablation_b3/report.md) | 2026-04-09 | 0.7193 | 0.6459 | `convnextv2_large` | `convnextv2_large_8bit_ablation_b1` |
| ❌ | [convnextv2_large_8bit_ablation_b4](convnextv2_large_8bit_ablation_b4/report.md) | 2026-04-09 | 0.4449 | FAILED | `convnextv2_large` | `convnextv2_large_8bit_ablation_b1` |
| ✅ | [convnextv2_large_8bit_ablation_b5](convnextv2_large_8bit_ablation_b5/report.md) | 2026-04-09 | 0.7286 | 0.6615 | `convnextv2_large` | `convnextv2_large_8bit_ablation_b1` |
| 📋 | [convnextv2_large_8bit_ablation_c1](convnextv2_large_8bit_ablation_c1/report.md) | 2026-04-11 | — | — | `convnextv2_large` | `convnextv2_large_8bit_ablation_b5` |
| 📋 | [dinov2_vitl_8bit_ablation_c2](dinov2_vitl_8bit_ablation_c2/report.md) | 2026-04-11 | — | — | `vit_large_patch14_dinov2` | `dinov2_vitl_8bit_ablation_a3` |
| 📋 | [convnextv2_large_8bit_ablation_c3](convnextv2_large_8bit_ablation_c3/report.md) | 2026-04-11 | — | — | `convnextv2_large` | `convnextv2_large_8bit_ablation_b5` |
| 🔄 | [swinv2_base_8bit_ablation_a2](swinv2_base_8bit_ablation_a2/report.md) | 2026-04-09 | 0.6013* | — | `swinv2_base_window12to24_192to384` | `convnextv2_large_8bit_ablation_a1` |
| 📋 | [swinv2_base_cosine_v1](swinv2_base_cosine_v1/report.md) | 2026-04-05 | — | — | `swinv2_base_window12to24_192to384` | `swinv2_base_v4` |
| 📋 | [swinv2_base_focal_asymmetry_v1](swinv2_base_focal_asymmetry_v1/report.md) | 2026-04-05 | — | — | `swinv2_base_window12to24_192to384` | `swinv2_base_focal_v1` |
| 📋 | [swinv2_base_focal_v1](swinv2_base_focal_v1/report.md) | 2026-04-05 | — | — | `swinv2_base_window12to24_192to384` | `swinv2_base_cosine_v1` |
| ✅ | [swinv2_base_v4](swinv2_base_v4/report.md) | 2026-03-20 | 0.6465 | — | `swinv2_base_window12to24_192to384` | `swinv2_base_v1` |
