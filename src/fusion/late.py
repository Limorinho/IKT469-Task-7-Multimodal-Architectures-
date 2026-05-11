import torch
import torch.nn as nn


class LateFusion(nn.Module):
    """Late fusion: independent classifiers per modality; average their logits.

    Unlike the other fusion modules, this outputs logits directly (shape [B, num_classes])
    because the fusion *is* the prediction average. MultimodalClassifier detects this and
    skips the final linear layer.
    """

    is_logit_fusion = True

    def __init__(
        self,
        text_dim: int = 300,
        image_dim: int = 4096,
        hidden_dim: int = 128,
        num_classes: int = 23,
    ):
        super().__init__()
        self.text_head = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes),
        )
        self.image_head = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        text_logits = self.text_head(text_emb)
        image_logits = self.image_head(image_emb)
        return 0.5 * (text_logits + image_logits)
