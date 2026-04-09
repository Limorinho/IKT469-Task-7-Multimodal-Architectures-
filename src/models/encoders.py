"""Image and text encoders for multimodal learning."""

import torch
import torch.nn as nn
from torchvision import models
from transformers import DistilBertModel, DistilBertTokenizer

IMAGE_EMBED_DIM = 2048
TEXT_EMBED_DIM = 768


class ImageEncoder(nn.Module):
    """ResNet-50 backbone, outputs 2048-dim pooled features."""

    def __init__(self, freeze_backbone: bool = False):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])  # remove FC
        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 224, 224)
        feats = self.encoder(x)          # (B, 2048, 1, 1)
        return feats.flatten(1)          # (B, 2048)


class TextEncoder(nn.Module):
    """DistilBERT encoder, outputs 768-dim CLS token embedding."""

    def __init__(self, model_name: str = "distilbert-base-uncased",
                 freeze_backbone: bool = False):
        super().__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.bert = DistilBertModel.from_pretrained(model_name)
        if freeze_backbone:
            for p in self.bert.parameters():
                p.requires_grad = False

    def tokenize(self, texts: list[str], device: torch.device) -> dict:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in encoded.items()}

    def forward(self, texts: list[str], device: torch.device) -> torch.Tensor:
        tokens = self.tokenize(texts, device)
        out = self.bert(**tokens)
        return out.last_hidden_state[:, 0, :]  # (B, 768) — CLS token
