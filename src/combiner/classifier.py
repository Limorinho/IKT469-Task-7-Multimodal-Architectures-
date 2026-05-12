import torch
import torch.nn as nn


class MultimodalClassifier(nn.Module):
    """Generic multimodal classifier.

    The fusion module takes (text_emb, image_emb) and returns either:
    - a hidden representation [B, hidden_dim] (default), which is then mapped to logits
      by an internal Linear layer; or
    - logits directly [B, num_classes] if fusion has attribute `is_logit_fusion = True`
      (used by late fusion, which averages per-modality logits).
    """

    def __init__(self, fusion: nn.Module, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.fusion = fusion
        self.is_logit_fusion = getattr(fusion, "is_logit_fusion", False)
        if self.is_logit_fusion:
            self.head = nn.Identity()
        else:
            self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        fused = self.fusion(text_emb, image_emb)
        return self.head(fused)


class EncodedMultimodalClassifier(nn.Module):
    """Multimodal classifier with built-in image and text encoders.

    Takes raw inputs (image tensor + tokenized text) and runs them through
    frozen or trainable encoders before passing to the fusion module.
    """

    def __init__(
        self,
        text_encoder: nn.Module,
        image_encoder: nn.Module,
        fusion: nn.Module,
        num_classes: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.fusion = fusion
        self.is_logit_fusion = getattr(fusion, "is_logit_fusion", False)
        if self.is_logit_fusion:
            self.head = nn.Identity()
        else:
            self.head = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        text_emb = self.text_encoder(input_ids, attention_mask)
        image_emb = self.image_encoder(image)
        fused = self.fusion(text_emb, image_emb)
        return self.head(fused)
