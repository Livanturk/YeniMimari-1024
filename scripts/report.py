#!/usr/bin/env python3
"""
Deney Raporu Yönetim Aracı
===========================
Her eğitim deneyi için yapılandırılmış rapor oluşturur, doldurur ve indexler.

Kullanım:
    # Yeni deney raporu oluştur (config diff otomatik hesaplanır)
    python scripts/report.py new swinv2_base_focal_v1
    python scripts/report.py new swinv2_base_focal_v1 --baseline convnextv2_focal_v1

    # Eğitim tamamlandıktan sonra metrikleri checkpoint'ten doldur
    python scripts/report.py fill swinv2_base_focal_v1

    # experiments/EXPERIMENTS.md master indexini yeniden oluştur
    python scripts/report.py index
"""

import argparse
import glob
import os
import re
import sys
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = ROOT / "experiments"
CONFIGS_DIR = ROOT / "configs"
OUTPUTS_DIR = ROOT / "outputs"

IGNORED_DIFF_KEYS = {
    "project.name", "project.description",
    "mlflow.experiment_name",
    "wandb.project", "wandb.entity",
    "visualization.gradcam.save_dir",
    "visualization.confusion_matrix.save_dir",
    "visualization.classification_report.save_dir",
    "checkpoint.save_dir",
}

# ─────────────────────────────────────────────────────────────────────────────
# YAML yardımcıları
# ─────────────────────────────────────────────────────────────────────────────

def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def flatten(d: dict, prefix: str = "") -> dict:
    """Nested dict → dot-separated flat dict."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        elif isinstance(v, list):
            out[key] = str(v)
        else:
            out[key] = v
    return out


def config_diff(base: dict, target: dict) -> list:
    """(key, base_val, target_val) listesi — sadece farklı olanlar."""
    fb = flatten(base)
    ft = flatten(target)
    diffs = []
    for k in sorted(set(fb) | set(ft)):
        if k in IGNORED_DIFF_KEYS:
            continue
        bv = fb.get(k, "—")
        tv = ft.get(k, "—")
        if str(bv) != str(tv):
            diffs.append((k, bv, tv))
    return diffs


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint'ten metrik okuma
# ─────────────────────────────────────────────────────────────────────────────

def read_checkpoint_metrics(exp_name: str) -> dict | None:
    """Best checkpoint'ten val metriklerini okur."""
    ckpt_dir = OUTPUTS_DIR / exp_name / "checkpoints"
    candidates = (
        list(ckpt_dir.glob("best*.pt"))
        + list(ckpt_dir.glob("best*.pth"))
        + sorted(ckpt_dir.glob("*.pt"))
    )
    if not candidates:
        return None

    try:
        import torch
        ckpt = torch.load(candidates[0], map_location="cpu", weights_only=False)
        m = ckpt.get("metrics", {})
        epoch = ckpt.get("epoch", "?")

        def _get(keys):
            for k in keys:
                if k in m:
                    v = m[k]
                    return f"{v:.4f}" if isinstance(v, float) else str(v)
            return "?"

        return {
            "best_epoch": epoch,
            "val_f1_macro":  _get(["val_full_f1_macro", "full_f1_macro"]),
            "val_f1_br1":    _get(["val_full_f1_BIRADS_1", "full_f1_BIRADS_1"]),
            "val_f1_br2":    _get(["val_full_f1_BIRADS_2", "full_f1_BIRADS_2"]),
            "val_f1_br4":    _get(["val_full_f1_BIRADS_4", "full_f1_BIRADS_4"]),
            "val_f1_br5":    _get(["val_full_f1_BIRADS_5", "full_f1_BIRADS_5"]),
            "val_auc":       _get(["val_full_auc_roc", "full_auc_roc"]),
            "val_kappa":     _get(["val_full_cohens_kappa", "full_cohens_kappa"]),
            "val_binary_f1": _get(["val_binary_f1", "binary_f1"]),
        }
    except Exception as e:
        print(f"[UYARI] Checkpoint okunamadı: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Frontmatter okuma/yazma
# ─────────────────────────────────────────────────────────────────────────────

FM_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def read_frontmatter(content: str) -> tuple[dict, str]:
    """(frontmatter_dict, body_without_fm) döndürür."""
    m = FM_PATTERN.match(content)
    if not m:
        return {}, content
    fm = yaml.safe_load(m.group(1)) or {}
    body = content[m.end():]
    return fm, body


def write_frontmatter(fm: dict, body: str) -> str:
    lines = []
    for k, v in fm.items():
        if v is None:
            lines.append(f"{k}: ~")
        elif isinstance(v, str) and (" " in v or ":" in v or v == ""):
            lines.append(f'{k}: "{v}"')
        else:
            lines.append(f"{k}: {v}")
    fm_str = "\n".join(lines)
    return f"---\n{fm_str}\n---\n{body}"


# ─────────────────────────────────────────────────────────────────────────────
# NEW komutu
# ─────────────────────────────────────────────────────────────────────────────

def cmd_new(exp_name: str, baseline_name: str | None):
    # configs/ altında düz veya alt klasörlerde ara
    candidates = list(CONFIGS_DIR.rglob(f"{exp_name}.yaml"))
    if not candidates:
        print(f"[HATA] Config bulunamadı: {exp_name}.yaml (configs/ ve alt klasörleri tarandı)")
        sys.exit(1)
    config_path = candidates[0]

    exp_dir = EXPERIMENTS_DIR / exp_name
    exp_dir.mkdir(exist_ok=True)
    report_path = exp_dir / "report.md"

    if report_path.exists():
        print(f"[UYARI] Rapor zaten var: {report_path}")
        print("Üzerine yazmak için raporu manuel silin.")
        return

    cfg = load_yaml(config_path)

    # Config diff
    diff_rows = []
    baseline_label = "—"
    if baseline_name:
        base_candidates = list(CONFIGS_DIR.rglob(f"{baseline_name}.yaml"))
        if base_candidates:
            base_cfg = load_yaml(base_candidates[0])
            diff_rows = config_diff(base_cfg, cfg)
            baseline_label = baseline_name
        else:
            print(f"[UYARI] Baseline config bulunamadı: {baseline_name}.yaml")

    # Diff tablosu
    if diff_rows:
        diff_table = (
            "| Alan | Baseline (`" + baseline_label + "`) | Bu Deney | Gerekçe |\n"
            "|---|---|---|---|\n"
        )
        for key, bv, tv in diff_rows:
            diff_table += f"| `{key}` | `{bv}` | `{tv}` | *(açıkla)* |\n"
    elif baseline_name:
        diff_table = "_Baseline ile aynı parametre değerleri._\n"
    else:
        diff_table = "_Baseline belirtilmedi. `--baseline <config_adı>` ile yeniden oluşturun._\n"

    backbone = cfg.get("model", {}).get("backbone", {}).get("name", "—")
    scheduler = cfg.get("training", {}).get("scheduler", {}).get("name", "—")
    lr = cfg.get("training", {}).get("optimizer", {}).get("lr", "—")

    fm = {
        "experiment": exp_name,
        "date": str(date.today()),
        "config": f"configs/{exp_name}.yaml",
        "baseline": baseline_name or "—",
        "backbone": backbone,
        "status": "planned",
        "best_epoch": None,
        "val_f1_macro": None,
        "test_f1_macro": None,
    }

    body = f"""
# {exp_name}

## Motivasyon / Hipotez
*(Bu deneyin amacını ve beklentini buraya yaz.)*

## Config Özeti
- **Backbone:** `{backbone}`
- **Scheduler:** `{scheduler}`
- **LR:** `{lr}`

## Baseline'dan Değişiklikler
{diff_table}
## Sonuçlar

### Val — En İyi Checkpoint (Epoch: ?)
| Metrik | Değer |
|---|---|
| **F1 Macro** | ? |
| F1 BR1 | ? |
| F1 BR2 | ? |
| F1 BR4 | ? |
| F1 BR5 | ? |
| AUC-ROC | ? |
| Cohen's Kappa | ? |
| Binary F1 | ? |

### Test
| Metrik | Değer |
|---|---|
| **F1 Macro** | ? |
| F1 BR1 | ? |
| F1 BR2 | ? |
| F1 BR4 | ? |
| F1 BR5 | ? |
| AUC-ROC | ? |

## Analiz
*(Eğitim tamamlandığında doldur: ne çalıştı, ne çalışmadı, dikkat çeken bulgular.)*

## Sonraki Adım
*(Bu deneyin sonucuna göre bir sonraki adım ne olmalı?)*
"""

    content = write_frontmatter(fm, body)
    report_path.write_text(content)
    print(f"[OK] Rapor oluşturuldu: {report_path}")
    print(f"     Config diff: {len(diff_rows)} parametre değişikliği")


# ─────────────────────────────────────────────────────────────────────────────
# FILL komutu
# ─────────────────────────────────────────────────────────────────────────────

def cmd_fill(exp_name: str):
    report_path = EXPERIMENTS_DIR / exp_name / "report.md"
    if not report_path.exists():
        print(f"[HATA] Rapor bulunamadı: {report_path}")
        print(f"       Önce: python scripts/report.py new {exp_name}")
        sys.exit(1)

    metrics = read_checkpoint_metrics(exp_name)
    if not metrics:
        print(f"[HATA] Checkpoint bulunamadı: outputs/{exp_name}/checkpoints/")
        sys.exit(1)

    content = report_path.read_text()
    fm, body = read_frontmatter(content)

    # Frontmatter güncelle
    fm["best_epoch"] = metrics["best_epoch"]
    fm["val_f1_macro"] = metrics["val_f1_macro"]
    fm["status"] = "completed"

    # Val tablosunu doldur
    replacements = {
        "| **F1 Macro** | ? |":      f"| **F1 Macro** | {metrics['val_f1_macro']} |",
        "| F1 BR1 | ? |":            f"| F1 BR1 | {metrics['val_f1_br1']} |",
        "| F1 BR2 | ? |":            f"| F1 BR2 | {metrics['val_f1_br2']} |",
        "| F1 BR4 | ? |":            f"| F1 BR4 | {metrics['val_f1_br4']} |",
        "| F1 BR5 | ? |":            f"| F1 BR5 | {metrics['val_f1_br5']} |",
        "| AUC-ROC | ? |":           f"| AUC-ROC | {metrics['val_auc']} |",
        "| Cohen's Kappa | ? |":     f"| Cohen's Kappa | {metrics['val_kappa']} |",
        "| Binary F1 | ? |":         f"| Binary F1 | {metrics['val_binary_f1']} |",
        "(Epoch: ?)":                f"(Epoch: {metrics['best_epoch']})",
    }

    for old, new in replacements.items():
        body = body.replace(old, new, 1)

    content = write_frontmatter(fm, body)
    report_path.write_text(content)
    print(f"[OK] Val metrikleri dolduruldu: {report_path}")
    print(f"     Epoch: {metrics['best_epoch']}, Val F1 Macro: {metrics['val_f1_macro']}")
    print("     Test metrikleri için raporu manuel olarak düzenleyin.")


# ─────────────────────────────────────────────────────────────────────────────
# INDEX komutu
# ─────────────────────────────────────────────────────────────────────────────

STATUS_ICON = {
    "completed": "✅",
    "running":   "🔄",
    "planned":   "📋",
    "failed":    "❌",
}


def cmd_index():
    reports = sorted(
        p for p in EXPERIMENTS_DIR.glob("*/report.md")
        if not p.parent.name.startswith("_")
    )
    if not reports:
        print("[UYARI] Hiç rapor bulunamadı.")
        return

    rows = []
    for rp in reports:
        content = rp.read_text()
        fm, _ = read_frontmatter(content)
        if not fm:
            continue
        status = fm.get("status", "?")
        icon = STATUS_ICON.get(status, "?")
        exp = fm.get("experiment", rp.parent.name)
        d = fm.get("date", "—")
        val_f1 = fm.get("val_f1_macro") or "—"
        test_f1 = fm.get("test_f1_macro") or "—"
        baseline = fm.get("baseline", "—")
        backbone = fm.get("backbone", "—")
        rows.append((icon, exp, status, str(d), str(val_f1), str(test_f1), str(backbone), str(baseline)))

    header = (
        "# Deney İndeksi\n\n"
        "> Otomatik oluşturulmuştur. `python scripts/report.py index` ile güncellenir.\n\n"
        f"Son güncelleme: {date.today()}\n\n"
        "| Durum | Deney | Tarih | Val F1 | Test F1 | Backbone | Baseline |\n"
        "|:---:|---|---|:---:|:---:|---|---|\n"
    )
    table_rows = ""
    for icon, exp, status, d, val, test, bb, bl in rows:
        exp_link = f"[{exp}]({exp}/report.md)"
        bb_short = bb.split(".")[0] if bb != "—" else "—"
        table_rows += f"| {icon} | {exp_link} | {d} | {val} | {test} | `{bb_short}` | `{bl}` |\n"

    index_content = header + table_rows
    index_path = EXPERIMENTS_DIR / "EXPERIMENTS.md"
    index_path.write_text(index_content)
    print(f"[OK] Index güncellendi: {index_path} ({len(rows)} deney)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Deney raporu oluşturma ve yönetim aracı",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_new = sub.add_parser("new", help="Yeni deney raporu oluştur")
    p_new.add_argument("exp_name", help="Deney adı (= config dosya adı, .yaml olmadan)")
    p_new.add_argument("--baseline", help="Baseline config adı (diff için)", default=None)

    p_fill = sub.add_parser("fill", help="Checkpoint'ten val metriklerini doldur")
    p_fill.add_argument("exp_name", help="Deney adı")

    sub.add_parser("index", help="EXPERIMENTS.md master indexini yeniden oluştur")

    args = parser.parse_args()

    if args.cmd == "new":
        cmd_new(args.exp_name, args.baseline)
    elif args.cmd == "fill":
        cmd_fill(args.exp_name)
    elif args.cmd == "index":
        cmd_index()


if __name__ == "__main__":
    main()
