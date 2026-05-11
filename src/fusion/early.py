import torch
import torch.nn as nn


class ConcatMLP(nn.Module):
    """Early fusion: concatenate text and image features, pass through an MLP."""

    def __init__(self, text_dim: int = 300, image_dim: int = 4096, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim + image_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([text_emb, image_emb], dim=1)
        return self.net(x)
