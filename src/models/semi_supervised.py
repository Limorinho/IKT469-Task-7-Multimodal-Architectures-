"""
Semi-supervised and self-supervised multimodal learning.

Techniques:
  - PseudoLabelWrapper: wraps any fusion model with pseudo-labeling
  - ConsistencyModel: self-supervised consistency training via augmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.dataset import NUM_GENRES
from src.models.encoders import IMAGE_EMBED_DIM, TEXT_EMBED_DIM, ImageEncoder, TextEncoder

PROJ_DIM = 512


class PseudoLabelWrapper(nn.Module):
    """
    Pseudo-labeling wrapper for any fusion model.

    Usage:
      1. Train base model on labeled data.
      2. Call generate_pseudo_labels() on unlabeled data.
      3. Combine pseudo-labeled data with labeled data and retrain.
    """

    def __init__(self, base_model: nn.Module, confidence_threshold: float = 0.8):
        super().__init__()
        self.model = base_model
        self.threshold = confidence_threshold

    def forward(self, images: torch.Tensor, texts: list[str]) -> torch.Tensor:
        return self.model(images, texts)

    @torch.no_grad()
    def generate_pseudo_labels(
        self, images: torch.Tensor, texts: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (pseudo_labels, mask) where mask=True means the sample's
        max sigmoid probability exceeds the confidence threshold.
        """
        self.eval()
        logits = self.model(images, texts)
        probs = torch.sigmoid(logits)                        # (B, C)
        max_probs = probs.max(dim=1).values                  # (B,)
        mask = max_probs >= self.threshold                   # (B,) bool
        pseudo_labels = (probs >= 0.5).float()               # (B, C)
        return pseudo_labels, mask


class ConsistencyModel(nn.Module):
    """
    Self-supervised consistency training.

    The model is trained with two objectives:
      1. Supervised cross-entropy on labeled samples.
      2. Consistency loss: predictions under two augmented views of the same
         input should agree (MSE between sigmoid outputs).

    Architecture: shared encoders + projection heads for contrastive repr,
    plus a classification head.
    """

    def __init__(self, num_classes: int = NUM_GENRES, freeze_backbones: bool = False,
                 consistency_weight: float = 1.0):
        super().__init__()
        self.image_encoder = ImageEncoder(freeze_backbone=freeze_backbones)
        self.text_encoder = TextEncoder(freeze_backbone=freeze_backbones)
        self.consistency_weight = consistency_weight

        self.image_proj = nn.Sequential(
            nn.Linear(IMAGE_EMBED_DIM, PROJ_DIM),
            nn.LayerNorm(PROJ_DIM),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(TEXT_EMBED_DIM, PROJ_DIM),
            nn.LayerNorm(PROJ_DIM),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(PROJ_DIM * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        # Projection head for contrastive/consistency representation
        self.proj_head = nn.Sequential(
            nn.Linear(PROJ_DIM * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    def encode(self, images: torch.Tensor, texts: list[str]) -> torch.Tensor:
        device = images.device
        img_feats = self.image_proj(self.image_encoder(images))
        txt_feats = self.text_proj(self.text_encoder(texts, device))
        return torch.cat([img_feats, txt_feats], dim=1)  # (B, PROJ_DIM*2)

    def forward(self, images: torch.Tensor, texts: list[str]) -> torch.Tensor:
        return self.classifier(self.encode(images, texts))

    def consistency_loss(
        self,
        images_aug1: torch.Tensor,
        texts: list[str],
        images_aug2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Consistency loss: the model's predictions on two augmented views
        of the same image (with same text) should agree.
        """
        logits1 = self.forward(images_aug1, texts)
        logits2 = self.forward(images_aug2, texts)
        probs1 = torch.sigmoid(logits1)
        probs2 = torch.sigmoid(logits2)
        return F.mse_loss(probs1, probs2)

    def contrastive_loss(
        self,
        images_aug1: torch.Tensor,
        texts: list[str],
        images_aug2: torch.Tensor,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """
        NT-Xent contrastive loss over projected representations.
        Positive pairs: two augmented views of the same sample.
        """
        z1 = F.normalize(self.proj_head(self.encode(images_aug1, texts)), dim=1)
        z2 = F.normalize(self.proj_head(self.encode(images_aug2, texts)), dim=1)

        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)                          # (2B, d)
        sim = torch.mm(z, z.T) / temperature                    # (2B, 2B)

        # Mask out self-similarity
        mask = torch.eye(2 * B, device=z.device).bool()
        sim.masked_fill_(mask, float("-inf"))

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0, B, device=z.device),
        ])
        return F.cross_entropy(sim, labels)
