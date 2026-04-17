import torch
import torch.nn as nn
from src.fusion.gmu import Gmu

class MultimodalClassifier(nn.Module):
    def __init__(self, text_embedding,  image_embedding, text_dim, image_dim, num_classes):
        super(MultimodalClassifier, self).__init__()
        self.text_model = text_embedding 
        self.image_model = image_embedding 
        self.fusion = Gmu(text_dim, image_dim, 128)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, attention_mask, text_input, image_input):
        text_features = self.text_model(text_input, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        image_features = self.image_model(image_input)

        fused_features = self.fusion(image_features, text_features)

        output = self.classifier(fused_features)
        return output
