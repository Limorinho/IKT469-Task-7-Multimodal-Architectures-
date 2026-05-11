import torch
from src.ssl.contrastive import info_nce, ProjectionHead


def test_info_nce_is_finite():
    text = torch.randn(8, 64)
    image = torch.randn(8, 64)
    loss = info_nce(text, image, temperature=0.07)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_info_nce_lower_when_aligned():
    rng = torch.Generator().manual_seed(0)
    base = torch.randn(8, 64, generator=rng)
    text_aligned = base + 0.01 * torch.randn(8, 64, generator=rng)
    image_aligned = base + 0.01 * torch.randn(8, 64, generator=rng)
    aligned_loss = info_nce(text_aligned, image_aligned).item()

    text_random = torch.randn(8, 64, generator=rng)
    image_random = torch.randn(8, 64, generator=rng)
    random_loss = info_nce(text_random, image_random).item()

    assert aligned_loss < random_loss


def test_projection_head_output_shape():
    head = ProjectionHead(input_dim=300, hidden_dim=128, output_dim=64)
    out = head(torch.randn(4, 300))
    assert out.shape == (4, 64)
