import torch
import torch.nn as nn
from torchvision import models


class ImageEncoder(nn.Module):
    """Pretrained ResNet50 image encoder. Outputs 2048-d pooled features."""

    output_dim: int = 2048

    def __init__(self, pretrained: bool = True, freeze: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)
