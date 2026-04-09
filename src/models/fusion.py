"""Early and late fusion multimodal classifiers."""

import torch
import torch.nn as nn

from src.data.dataset import NUM_GENRES
from src.models.encoders import IMAGE_EMBED_DIM, TEXT_EMBED_DIM, ImageEncoder, TextEncoder

PROJ_DIM = 512


class EarlyFusionModel(nn.Module):
    """
    Early fusion: project image and text to shared dim, concatenate, then classify.
    Fusion happens before any task-specific layers.
    """

    def __init__(self, num_classes: int = NUM_GENRES, freeze_backbones: bool = False):
        super().__init__()
        self.image_encoder = ImageEncoder(freeze_backbone=freeze_backbones)
        self.text_encoder = TextEncoder(freeze_backbone=freeze_backbones)

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

    def forward(self, images: torch.Tensor, texts: list[str]) -> torch.Tensor:
        device = images.device
        img_feats = self.image_proj(self.image_encoder(images))     # (B, PROJ_DIM)
        txt_feats = self.text_proj(self.text_encoder(texts, device)) # (B, PROJ_DIM)
        fused = torch.cat([img_feats, txt_feats], dim=1)            # (B, PROJ_DIM*2)
        return self.classifier(fused)                               # (B, num_classes)

    def get_embeddings(self, images: torch.Tensor, texts: list[str]) -> torch.Tensor:
        """Return the fused representation before the classifier head."""
        device = images.device
        img_feats = self.image_proj(self.image_encoder(images))
        txt_feats = self.text_proj(self.text_encoder(texts, device))
        return torch.cat([img_feats, txt_feats], dim=1)


class LateFusionModel(nn.Module):
    """
    Late fusion: independent classifiers per modality, combine logits.
    Each stream learns a full classification, then outputs are averaged.
    """

    def __init__(self, num_classes: int = NUM_GENRES, freeze_backbones: bool = False,
                 learned_weights: bool = True):
        super().__init__()
        self.image_encoder = ImageEncoder(freeze_backbone=freeze_backbones)
        self.text_encoder = TextEncoder(freeze_backbone=freeze_backbones)

        self.image_head = nn.Sequential(
            nn.Linear(IMAGE_EMBED_DIM, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
        self.text_head = nn.Sequential(
            nn.Linear(TEXT_EMBED_DIM, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

        if learned_weights:
            # Learnable scalar weights for combining logits
            self.fusion_weights = nn.Parameter(torch.ones(2) / 2)
        else:
            self.register_buffer("fusion_weights", torch.ones(2) / 2)
        self.learned_weights = learned_weights

    def forward(self, images: torch.Tensor, texts: list[str]) -> torch.Tensor:
        device = images.device
        img_logits = self.image_head(self.image_encoder(images))       # (B, C)
        txt_logits = self.text_head(self.text_encoder(texts, device))  # (B, C)

        if self.learned_weights:
            w = torch.softmax(self.fusion_weights, dim=0)
        else:
            w = self.fusion_weights
        return w[0] * img_logits + w[1] * txt_logits

    def get_modality_logits(self, images: torch.Tensor, texts: list[str]):
        """Return per-modality logits for analysis."""
        device = images.device
        img_logits = self.image_head(self.image_encoder(images))
        txt_logits = self.text_head(self.text_encoder(texts, device))
        return img_logits, txt_logits
