import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Two-layer MLP projection head for contrastive learning (SimCLR style)."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def info_nce(
    text_z: torch.Tensor, image_z: torch.Tensor, temperature: float = 0.07
) -> torch.Tensor:
    """Symmetric cross-modal InfoNCE loss.

    Given paired projections (text_z[i], image_z[i]) for i in [0, B), treat each pair as
    a positive and all other (text_z[i], image_z[j!=i]) as negatives. Compute both the
    text->image and image->text cross-entropy directions and average them.
    """
    text_z = F.normalize(text_z, dim=-1)
    image_z = F.normalize(image_z, dim=-1)
    logits = text_z @ image_z.t() / temperature
    targets = torch.arange(text_z.shape[0], device=text_z.device)
    loss_t2i = F.cross_entropy(logits, targets)
    loss_i2t = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_t2i + loss_i2t)
