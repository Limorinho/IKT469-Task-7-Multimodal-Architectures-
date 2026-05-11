import torch
from src.ssl.pseudo_label import pseudo_labels


def test_pseudo_labels_high_confidence_passes():
    logits = torch.tensor([[10.0, -10.0, 10.0], [-10.0, 10.0, -10.0]])
    mask, labels = pseudo_labels(logits, threshold=0.9)
    assert mask.tolist() == [True, True]
    assert labels.tolist() == [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]


def test_pseudo_labels_unconfident_rejected():
    logits = torch.tensor([[0.5, -0.5, 0.0]])
    mask, labels = pseudo_labels(logits, threshold=0.9)
    assert mask.tolist() == [False]


def test_pseudo_labels_partial_confidence_rejects_row():
    logits = torch.tensor([[10.0, 0.1, -10.0]])
    mask, _ = pseudo_labels(logits, threshold=0.9)
    assert mask.tolist() == [False]
