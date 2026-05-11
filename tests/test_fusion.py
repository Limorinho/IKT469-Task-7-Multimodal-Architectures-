import torch
from src.fusion.unimodal import TextOnly, ImageOnly
from src.fusion.early import ConcatMLP
from src.fusion.late import LateFusion
from src.fusion.gmu import Gmu

B, T, I, H, C = 4, 300, 4096, 128, 23
text = torch.randn(B, T)
image = torch.randn(B, I)


def test_textonly_output_shape():
    m = TextOnly(text_dim=T, hidden_dim=H)
    out = m(text, image)
    assert out.shape == (B, H)


def test_imageonly_output_shape():
    m = ImageOnly(image_dim=I, hidden_dim=H)
    out = m(text, image)
    assert out.shape == (B, H)


def test_early_output_shape():
    m = ConcatMLP(text_dim=T, image_dim=I, hidden_dim=H)
    out = m(text, image)
    assert out.shape == (B, H)


def test_late_returns_logits():
    m = LateFusion(text_dim=T, image_dim=I, hidden_dim=H, num_classes=C)
    out = m(text, image)
    assert out.shape == (B, C)


def test_gmu_still_works():
    m = Gmu(text_dim=T, img_dim=I, output_dim=H)
    out = m(image, text)
    assert out.shape == (B, H)
