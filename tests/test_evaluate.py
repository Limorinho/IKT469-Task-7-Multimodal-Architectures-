import torch
import numpy as np
from src.training.evaluate import compute_f1_scores, evaluate_metrics


def test_compute_f1_scores_keys():
    y_true = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
    y_pred = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    scores = compute_f1_scores(y_true, y_pred)
    assert set(scores.keys()) == {"micro", "macro", "weighted", "samples"}
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_compute_f1_scores_perfect():
    y_true = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
    scores = compute_f1_scores(y_true, y_true)
    assert scores["micro"] == 1.0
    assert scores["macro"] == 1.0


def test_evaluate_metrics_returns_dict():
    class DummyModel(torch.nn.Module):
        def forward(self, text, image):
            return torch.zeros(text.shape[0], 3)

    loader = [{
        "text_emb": torch.randn(2, 5),
        "image_emb": torch.randn(2, 5),
        "label": torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
    }]
    result = evaluate_metrics(DummyModel(), loader, device="cpu")
    assert "f1" in result and "loss" in result
    assert set(result["f1"].keys()) == {"micro", "macro", "weighted", "samples"}
