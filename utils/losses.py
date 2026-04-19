"""
Multi-Head Loss Function
=========================
Hiyerarşik çoklu-kafa sınıflandırıcı için bileşik kayıp fonksiyonu.

Kayıp Bileşenleri:
    L_total = w1 * L_binary + w2 * L_subgroup + w3 * L_full

    L_binary:   CrossEntropy(binary_logits, binary_labels)
                Benign vs Malign ayrımı için.

    L_subgroup: CrossEntropy(benign_sub_logits, benign_labels) +
                CrossEntropy(malign_sub_logits, malign_labels)
                Alt grup ayrımları için. Sadece ilgili örneklere uygulanır.

    L_full:     CrossEntropy(full_logits, full_labels)
                4-sınıf direkt tahmin için.

Class Weights:
    Sınıf dengesizliğini telafi etmek için ağırlıklar kullanılır.
    Az temsil edilen sınıflar daha yüksek ağırlık alır.

Loss Türleri:
    - "ce": Standard CrossEntropyLoss (varsayılan)
    - "focal": Focal Loss — zor örneklere daha fazla odaklanır.
      Kolay örneklerin (yüksek olasılıklı doğru tahminler) kaybı
      (1-p)^gamma ile azaltılır. gamma=2.0 önerilir.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.classification_heads import HierarchicalClassifier


class OrdinalLoss(nn.Module):
    """
    CORAL (Consistent Rank Logits) tabanlı Ordinal Regression Kaybı.

    BI-RADS sınıfları arasındaki doğal sıralamayı (1 < 2 < 4 < 5) yakalamak için
    K-1 bağımsız binary sınıflandırıcı kullanır.

    Her sınıflandırıcı P(rank >= k) kümülatif olasılığını öğrenir:
        - label=0 (BIRADS 1): P(≥1)=0, P(≥2)=0, P(≥3)=0
        - label=1 (BIRADS 2): P(≥1)=1, P(≥2)=0, P(≥3)=0
        - label=2 (BIRADS 4): P(≥1)=1, P(≥2)=1, P(≥3)=0
        - label=3 (BIRADS 5): P(≥1)=1, P(≥2)=1, P(≥3)=1

    Standard CE'den farkı: Komşu sınıf hataları uzak sınıf hatalarından
    daha az cezalandırılır — bu BI-RADS klinik yapısına uygundur.

    Args:
        num_classes: Toplam sınıf sayısı (K). Default: 4
        weight: (K,) sınıf ağırlıkları. None ise eşit ağırlık.
    """

    def __init__(
        self,
        num_classes: int = 4,
        weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_ranks = num_classes - 1  # K-1 binary threshold
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, K-1) ordinal logit değerleri.
            targets: (B,) sınıf indeksleri (0 to K-1).

        Returns:
            Skaler ordinal kayıp.
        """
        B = logits.size(0)

        # Kümülatif binary hedefler: I(label >= k) her k=1,...,K-1 için
        rank_targets = torch.zeros(B, self.num_ranks, device=logits.device)
        for k in range(1, self.num_ranks + 1):
            rank_targets[:, k - 1] = (targets >= k).float()

        if self.weight is not None:
            # Her örnek için sınıf ağırlığı uygula
            sample_weights = self.weight[targets]   # (B,)
            loss = F.binary_cross_entropy_with_logits(
                logits, rank_targets, reduction="none"
            ).mean(dim=-1)                          # (B,)
            loss = (loss * sample_weights).mean()
        else:
            loss = F.binary_cross_entropy_with_logits(logits, rank_targets)

        return loss

    @staticmethod
    def decode(logits: torch.Tensor) -> torch.Tensor:
        """
        Ordinal logitlerden sınıf tahminleri üret.

        Tahmin = kaç rank eşiğinin sigmoid > 0.5 olduğu sayısı.

        Args:
            logits: (B, K-1) ordinal logitler.

        Returns:
            (B,) sınıf indeksleri (0 to K-1).
        """
        return (torch.sigmoid(logits) > 0.5).sum(dim=-1).long()

    @staticmethod
    def to_class_probs(logits: torch.Tensor) -> torch.Tensor:
        """
        Ordinal logitlerden sınıf olasılıkları hesapla (metrik ve confidence için).

        P(class=0) = 1 - P(rank>=1)
        P(class=k) = P(rank>=k) - P(rank>=k+1)  k=1,...,K-2
        P(class=K-1) = P(rank>=K-1)

        Args:
            logits: (B, K-1) ordinal logitler.

        Returns:
            (B, K) normalize sınıf olasılıkları.
        """
        B = logits.size(0)
        probs_cumul = torch.sigmoid(logits)         # (B, K-1)

        ones  = torch.ones(B, 1, device=logits.device)
        zeros = torch.zeros(B, 1, device=logits.device)
        probs_ext = torch.cat([ones, probs_cumul, zeros], dim=-1)  # (B, K+1)

        class_probs = (probs_ext[:, :-1] - probs_ext[:, 1:]).clamp(min=0.0)  # (B, K)
        class_probs = class_probs / (class_probs.sum(dim=-1, keepdim=True) + 1e-8)
        return class_probs


class AsymmetryContrastiveLoss(nn.Module):
    """
    Bilateral Asimetri Tutarlılık Kaybı.

    Sol-sağ meme fark vektörü (F_diff = F_left - F_right) üzerinde
    sınıfa bağımlı kısıtlar uygular:

        Benign: Asimetri KÜÇÜK olmalı (simetrik memeler normal)
            L_benign = mean( (||F_diff|| / margin)^2 )

        Malign: Asimetri BÜYÜK olmalı (kitleli taraf öne çıkmalı)
            L_malign = mean( ReLU(1 - ||F_diff|| / margin)^2 )

    Args:
        margin: Hedef F_diff norm büyüklüğü (malign için referans). Default: 1.0
        benign_weight: Benign kaybı ağırlığı.
        malign_weight: Malign kaybı ağırlığı.
    """

    def __init__(
        self,
        margin: float = 1.0,
        benign_weight: float = 1.0,
        malign_weight: float = 1.0,
    ):
        super().__init__()
        self.margin = margin
        self.benign_weight = benign_weight
        self.malign_weight = malign_weight

    def forward(
        self, f_diff: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            f_diff: (B, dim) sol-sağ fark vektörü (F_left - F_right).
            labels: (B,) 4-sınıf etiketleri (0=BIRADS1, 1=BIRADS2, 2=BIRADS4, 3=BIRADS5).

        Returns:
            Skaler asimetri kaybı.
        """
        binary = (labels >= 2).float()              # 0=benign, 1=malign
        diff_norm = torch.norm(f_diff, p=2, dim=-1) # (B,)

        loss = torch.tensor(0.0, device=f_diff.device)
        n_terms = 0

        benign_mask = binary < 0.5
        if benign_mask.any():
            # Benign: küçük asimetriyi ödüllendir → büyük diff'i cezalandır
            benign_loss = (diff_norm[benign_mask] / self.margin).pow(2).mean()
            loss = loss + self.benign_weight * benign_loss
            n_terms += 1

        malign_mask = binary > 0.5
        if malign_mask.any():
            # Malign: büyük asimetriyi ödüllendir → küçük diff'i cezalandır
            malign_loss = F.relu(1.0 - diff_norm[malign_mask] / self.margin).pow(2).mean()
            loss = loss + self.malign_weight * malign_loss
            n_terms += 1

        return loss / max(n_terms, 1)


class FocalLoss(nn.Module):
    """
    Focal Loss: Zor örneklere odaklanan kayıp fonksiyonu.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        weight: Sınıf ağırlıkları (alpha). None ise eşit ağırlık.
        gamma: Focusing parametresi. gamma=0 → standard CE.
               gamma arttıkça kolay örneklerin etkisi azalır.
        label_smoothing: Etiket yumuşatma faktörü.
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C) ham logit değerleri.
            targets: (B,) sınıf indeksleri.

        Returns:
            Skaler focal loss değeri.
        """
        num_classes = logits.size(-1)

        # Label smoothing uygula
        with torch.no_grad():
            smooth_targets = torch.zeros_like(logits)
            smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        # Log-softmax ve softmax hesapla
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - probs) ** self.gamma

        # Class weight uygula
        if self.weight is not None:
            class_weight = self.weight.unsqueeze(0)  # (1, C)
            focal_weight = focal_weight * class_weight

        # Focal loss = -alpha * (1-p)^gamma * log(p) * smooth_target
        loss = -focal_weight * log_probs * smooth_targets
        loss = loss.sum(dim=-1).mean()

        return loss


class LogitAdjustedCE(nn.Module):
    """
    Logit-Adjusted Cross Entropy (Menon et al., ICLR 2021:
    "Long-tail learning via logit adjustment").

    Training-time transform:
        adjusted_logits = logits + tau * log(train_prior)
        loss = CE(adjusted_logits, targets, weight, label_smoothing)

    Inference-time: use RAW logits (no adjustment).

    The additive log-prior term pushes the model to produce a tempered
    Bayes-optimal classifier so that its raw logits approximate the
    class-conditional likelihoods. Under test-time prior shift, this is
    more robust than plain CE + class weights because the correction is
    additive in logit space (argmax-shifting) rather than multiplicative
    on the gradient (which is zero-sum on shared decision boundaries —
    see Lessons #27, #47).

    Degenerate cases (unit-tested):
      - Uniform prior (any tau): constant shift across classes → exactly
        equivalent to standard CE (softmax is shift-invariant).
      - tau = 0: exactly equivalent to standard CE.

    Args:
        train_prior: (K,) list/tensor of class frequencies (normalized or raw).
        tau: strength of adjustment. 1.0 matches the Menon et al. default.
        label_smoothing: forwarded to F.cross_entropy.
        class_weights: optional (K,) tensor of per-class loss weights.
    """

    def __init__(
        self,
        train_prior,
        tau: float = 1.0,
        label_smoothing: float = 0.05,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        prior = torch.tensor(train_prior, dtype=torch.float32)
        prior = prior / prior.sum().clamp(min=1e-12)     # defensive normalize
        self.register_buffer("log_prior", torch.log(prior.clamp(min=1e-12)))
        self.tau = float(tau)
        self.label_smoothing = float(label_smoothing)
        if class_weights is not None:
            self.register_buffer("class_weights_t", class_weights)
        else:
            self.register_buffer("class_weights_t", torch.empty(0))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Defensive: ensure buffers are on the same device as the incoming logits.
        # build_loss_function also calls .to(device) on the outer MultiHeadLoss
        # so this is usually a no-op, but it guards against callers that bypass
        # the builder.
        log_prior = self.log_prior.to(logits.device, non_blocking=True)
        adjusted = logits + self.tau * log_prior
        if self.class_weights_t.numel() > 0:
            weight = self.class_weights_t.to(logits.device, non_blocking=True)
        else:
            weight = None
        return F.cross_entropy(
            adjusted,
            targets,
            weight=weight,
            label_smoothing=self.label_smoothing,
        )


class MultiHeadLoss(nn.Module):
    """
    Hiyerarşik çoklu-kafa kayıp fonksiyonu.

    Args:
        loss_weights: Her head'in toplam kayba katkı ağırlığı.
            {"binary_head": 0.3, "subgroup_head": 0.3, "full_head": 0.4}
        class_weights_4: 4-sınıf için sınıf ağırlıkları (tensor).
        class_weights_binary: Binary için sınıf ağırlıkları (tensor).
        class_weights_benign_sub: Benign subgroup (BIRADS 1 vs 2) ağırlıkları.
        class_weights_malign_sub: Malign subgroup (BIRADS 4 vs 5) ağırlıkları.
        use_binary: Binary head kaybını hesapla.
        use_subgroup: Subgroup head kaybını hesapla.
        label_smoothing: Etiket yumuşatma (overfitting'i azaltır).
    """

    def __init__(
        self,
        loss_weights: dict,
        class_weights_4: Optional[torch.Tensor] = None,
        class_weights_binary: Optional[torch.Tensor] = None,
        class_weights_benign_sub: Optional[torch.Tensor] = None,
        class_weights_malign_sub: Optional[torch.Tensor] = None,
        use_binary: bool = True,
        use_subgroup: bool = True,
        label_smoothing: float = 0.05,
        loss_type: str = "ce",
        focal_gamma: float = 2.0,
        use_ordinal: bool = False,
        asymmetry_loss_weight: float = 0.0,
        asymmetry_margin: float = 1.0,
        asymmetry_benign_weight: float = 1.0,
        asymmetry_malign_weight: float = 1.0,
        train_prior: Optional[list] = None,
        logit_adjustment_tau: float = 1.0,
    ):
        super().__init__()

        self.w_binary = loss_weights.get("binary_head", 0.3)
        self.w_subgroup = loss_weights.get("subgroup_head", 0.3)
        self.w_full = loss_weights.get("full_head", 0.4)
        self.use_binary = use_binary
        self.use_subgroup = use_subgroup
        self.loss_type = loss_type
        self.use_ordinal = use_ordinal
        self.w_asymmetry = asymmetry_loss_weight

        # Ordinal loss (full head yerine kullanılır)
        if use_ordinal:
            print(f"[LOSS] Ordinal Loss (CORAL) aktif — full_head CE yerine")
            self.ordinal_criterion = OrdinalLoss(
                num_classes=4,
                weight=class_weights_4,
            )

        # Asymmetry contrastive loss
        if asymmetry_loss_weight > 0:
            print(f"[LOSS] Asymmetry Contrastive Loss aktif "
                  f"(w={asymmetry_loss_weight}, margin={asymmetry_margin}, "
                  f"benign_w={asymmetry_benign_weight}, malign_w={asymmetry_malign_weight})")
            self.asymmetry_criterion = AsymmetryContrastiveLoss(
                margin=asymmetry_margin,
                benign_weight=asymmetry_benign_weight,
                malign_weight=asymmetry_malign_weight,
            )

        if loss_type == "focal":
            print(f"[LOSS] Focal Loss aktif (gamma={focal_gamma}, label_smoothing={label_smoothing})")

            self.full_criterion = FocalLoss(
                weight=class_weights_4,
                gamma=focal_gamma,
                label_smoothing=label_smoothing,
            )
            self.binary_criterion = FocalLoss(
                weight=class_weights_binary,
                gamma=focal_gamma,
                label_smoothing=label_smoothing,
            )
            self.benign_sub_criterion = FocalLoss(
                weight=class_weights_benign_sub,
                gamma=focal_gamma,
                label_smoothing=label_smoothing,
            )
            self.malign_sub_criterion = FocalLoss(
                weight=class_weights_malign_sub,
                gamma=focal_gamma,
                label_smoothing=label_smoothing,
            )
        elif loss_type == "logit_adjusted":
            if train_prior is None:
                raise ValueError(
                    "loss_type='logit_adjusted' requires training.train_prior "
                    "in the config (list of K class frequencies)."
                )
            print(f"[LOSS] Logit-Adjusted CE aktif (Menon 2021): "
                  f"tau={logit_adjustment_tau}, prior={train_prior}, "
                  f"label_smoothing={label_smoothing}")
            print(f"[LOSS] NOT: Logit adjustment SADECE full_head'e uygulanır. "
                  f"Binary ve subgroup head'ler standart CE (prompt Task 2.2 kuralı).")
            self.full_criterion = LogitAdjustedCE(
                train_prior=train_prior,
                tau=logit_adjustment_tau,
                label_smoothing=label_smoothing,
                class_weights=class_weights_4,
            )
            self.binary_criterion = nn.CrossEntropyLoss(
                weight=class_weights_binary,
                label_smoothing=label_smoothing,
            )
            self.benign_sub_criterion = nn.CrossEntropyLoss(
                weight=class_weights_benign_sub,
                label_smoothing=label_smoothing,
            )
            self.malign_sub_criterion = nn.CrossEntropyLoss(
                weight=class_weights_malign_sub,
                label_smoothing=label_smoothing,
            )
        else:
            # Standard CrossEntropy (varsayılan)
            self.full_criterion = nn.CrossEntropyLoss(
                weight=class_weights_4,
                label_smoothing=label_smoothing,
            )
            self.binary_criterion = nn.CrossEntropyLoss(
                weight=class_weights_binary,
                label_smoothing=label_smoothing,
            )
            self.benign_sub_criterion = nn.CrossEntropyLoss(
                weight=class_weights_benign_sub,
                label_smoothing=label_smoothing,
            )
            self.malign_sub_criterion = nn.CrossEntropyLoss(
                weight=class_weights_malign_sub,
                label_smoothing=label_smoothing,
            )

    def forward(
        self, outputs: dict, labels: torch.Tensor
    ) -> dict:
        """
        Toplam kaybı hesaplar.

        Args:
            outputs: Model çıkışları (binary_logits, subgroup logits, full_logits).
            labels: (B,) 4-sınıf etiketleri.

        Returns:
            dict:
                - "total_loss": Toplam ağırlıklı kayıp.
                - "binary_loss": Binary head kaybı.
                - "benign_sub_loss": Benign subgroup kaybı.
                - "malign_sub_loss": Malign subgroup kaybı.
                - "full_loss": Full head kaybı.
        """
        # Etiketleri her head için dönüştür
        label_dict = HierarchicalClassifier.convert_labels(labels)

        losses = {}
        total_loss = torch.tensor(0.0, device=labels.device)

        # --- Full Head Loss (her zaman aktif) ---
        if self.use_ordinal and "ordinal_logits" in outputs:
            # CORAL Ordinal Loss: full CE yerine
            full_loss = self.ordinal_criterion(outputs["ordinal_logits"], label_dict["full"])
        else:
            full_loss = self.full_criterion(outputs["full_logits"], label_dict["full"])
        losses["full_loss"] = full_loss
        total_loss = total_loss + self.w_full * full_loss

        # --- Binary Head Loss ---
        if self.use_binary:
            binary_loss = self.binary_criterion(
                outputs["binary_logits"], label_dict["binary"]
            )
            losses["binary_loss"] = binary_loss
            total_loss = total_loss + self.w_binary * binary_loss

        # --- Subgroup Head Loss ---
        # Sadece ilgili örneklere uygulanır (benign → benign head, malign → malign head)
        if self.use_subgroup:
            benign_sub_loss = torch.tensor(0.0, device=labels.device)
            malign_sub_loss = torch.tensor(0.0, device=labels.device)

            # Benign alt grubu (BIRADS 1 ve 2 olan örnekler)
            benign_mask = label_dict["benign_mask"]
            if benign_mask.any():
                benign_logits = outputs["benign_sub_logits"][benign_mask]
                benign_labels = label_dict["benign_sub"]
                benign_sub_loss = self.benign_sub_criterion(benign_logits, benign_labels)

            # Malign alt grubu (BIRADS 4 ve 5 olan örnekler)
            malign_mask = label_dict["malign_mask"]
            if malign_mask.any():
                malign_logits = outputs["malign_sub_logits"][malign_mask]
                malign_labels = label_dict["malign_sub"]
                malign_sub_loss = self.malign_sub_criterion(malign_logits, malign_labels)

            subgroup_loss = (benign_sub_loss + malign_sub_loss) / 2.0
            losses["benign_sub_loss"] = benign_sub_loss
            losses["malign_sub_loss"] = malign_sub_loss
            total_loss = total_loss + self.w_subgroup * subgroup_loss

        # --- Asymmetry Contrastive Loss (bilateral f_diff üzerinde) ---
        if self.w_asymmetry > 0 and "f_diff" in outputs and outputs["f_diff"] is not None:
            asym_loss = self.asymmetry_criterion(outputs["f_diff"], labels)
            losses["asymmetry_loss"] = asym_loss
            total_loss = total_loss + self.w_asymmetry * asym_loss

        losses["total_loss"] = total_loss
        return losses


def build_loss_function(config: dict, device: torch.device) -> MultiHeadLoss:
    """
    Config'den loss fonksiyonu oluşturur.

    Args:
        config: YAML konfigürasyonu.
        device: CUDA/CPU cihazı.

    Returns:
        MultiHeadLoss instance.
    """
    train_cfg = config["training"]
    ablation_cfg = config.get("ablation", {})

    # 4-sınıf ağırlıkları
    cw = train_cfg.get("class_weights", [1.0, 1.0, 1.0, 1.0])
    class_weights_4 = torch.tensor(cw, dtype=torch.float32).to(device)

    # Binary ağırlıklar (sqrt-inverse frequency)
    # BIRADS-Full-Train-8Bit-Processed: Benign(BR1+BR2)=4432, Malign(BR4+BR5)=4125
    # sqrt(4432/4125)=1.037 → Benign:1.00, Malign:1.04
    class_weights_binary = torch.tensor([1.00, 1.04], dtype=torch.float32).to(device)

    # Subgroup ağırlıkları (sqrt-inverse frequency)
    # Benign: BIRADS 1 (1678) vs BIRADS 2 (2754) → sqrt(2754/1678)=1.281 → [1.28, 1.00]
    class_weights_benign_sub = torch.tensor([1.28, 1.00], dtype=torch.float32).to(device)
    # Malign: BIRADS 4 (1898) vs BIRADS 5 (2227) → sqrt(2227/1898)=1.084 → [1.08, 1.00]
    class_weights_malign_sub = torch.tensor([1.08, 1.00], dtype=torch.float32).to(device)

    criterion = MultiHeadLoss(
        loss_weights=train_cfg["loss_weights"],
        class_weights_4=class_weights_4,
        class_weights_binary=class_weights_binary,
        class_weights_benign_sub=class_weights_benign_sub,
        class_weights_malign_sub=class_weights_malign_sub,
        use_binary=ablation_cfg.get("use_binary_head", True),
        use_subgroup=ablation_cfg.get("use_subgroup_head", True),
        label_smoothing=train_cfg.get("label_smoothing", 0.05),
        loss_type=train_cfg.get("loss_type", "ce"),
        focal_gamma=train_cfg.get("focal_gamma", 2.0),
        use_ordinal=ablation_cfg.get("use_ordinal_head", False),
        asymmetry_loss_weight=train_cfg.get("asymmetry_loss_weight", 0.0),
        asymmetry_margin=train_cfg.get("asymmetry_margin", 1.0),
        asymmetry_benign_weight=train_cfg.get("asymmetry_benign_weight", 1.0),
        asymmetry_malign_weight=train_cfg.get("asymmetry_malign_weight", 1.0),
        train_prior=train_cfg.get("train_prior", None),
        logit_adjustment_tau=train_cfg.get("logit_adjustment_tau", 1.0),
    )
    # Move registered buffers (e.g. LogitAdjustedCE.log_prior) onto the target device.
    # nn.CrossEntropyLoss/FocalLoss weights were already pre-placed, but buffers
    # attached inside sub-modules are only moved via nn.Module.to().
    return criterion.to(device)
