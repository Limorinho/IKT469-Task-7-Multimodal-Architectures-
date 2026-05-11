import torch
import torch.nn as nn


class TextOnly(nn.Module):
    """Text-only baseline: discards image, projects text to hidden_dim."""

    def __init__(self, text_dim: int = 300, hidden_dim: int = 128, image_dim: int = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        return self.net(text_emb)


class ImageOnly(nn.Module):
    """Image-only baseline: discards text, projects image to hidden_dim."""

    def __init__(self, image_dim: int = 4096, hidden_dim: int = 128, text_dim: int = 300):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        return self.net(image_emb)
