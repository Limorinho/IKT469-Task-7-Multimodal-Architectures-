import torch


def pseudo_labels(
    logits: torch.Tensor, threshold: float = 0.9
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate confidence-thresholded pseudo-labels for multi-label classification.

    A sample is kept (mask=True) only if all classes are confidently above the threshold
    or confidently below 1-threshold. This conservative criterion avoids reinforcing
    uncertain predictions.

    Returns
    -------
    mask : Tensor[B] of bool
    labels : Tensor[B, C] of float
    """
    probs = torch.sigmoid(logits)
    confident = (probs > threshold) | (probs < 1.0 - threshold)
    mask = confident.all(dim=-1)
    labels = (probs > threshold).float()
    return mask, labels
