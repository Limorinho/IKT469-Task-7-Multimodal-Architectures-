import torch
from src.combiner.classifier import MultimodalClassifier
from src.fusion.early import ConcatMLP
from src.fusion.late import LateFusion
from src.fusion.gmu import Gmu

B, T, I, H, C = 4, 300, 4096, 128, 23
text = torch.randn(B, T)
image = torch.randn(B, I)


def test_classifier_with_early_fusion():
    fusion = ConcatMLP(text_dim=T, image_dim=I, hidden_dim=H)
    clf = MultimodalClassifier(fusion=fusion, num_classes=C, hidden_dim=H)
    logits = clf(text, image)
    assert logits.shape == (B, C)


def test_classifier_with_late_fusion_skips_final_linear():
    fusion = LateFusion(text_dim=T, image_dim=I, hidden_dim=H, num_classes=C)
    clf = MultimodalClassifier(fusion=fusion, num_classes=C, hidden_dim=H)
    logits = clf(text, image)
    assert logits.shape == (B, C)


def test_classifier_with_gmu():
    fusion = Gmu(text_dim=T, img_dim=I, output_dim=H)

    class GmuWrapper(torch.nn.Module):
        def __init__(self, gmu):
            super().__init__()
            self.gmu = gmu

        def forward(self, text_emb, image_emb):
            return self.gmu(image_emb, text_emb)

    clf = MultimodalClassifier(fusion=GmuWrapper(fusion), num_classes=C, hidden_dim=H)
    logits = clf(text, image)
    assert logits.shape == (B, C)
