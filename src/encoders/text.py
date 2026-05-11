import torch
import torch.nn as nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    """Pretrained BERT text encoder. Outputs the [CLS] pooled embedding (768-d)."""

    output_dim: int = 768

    def __init__(self, model_name: str = "bert-base-uncased", freeze: bool = True):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return out.pooler_output
