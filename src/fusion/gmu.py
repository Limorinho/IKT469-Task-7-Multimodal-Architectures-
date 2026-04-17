import torch
import torch.nn as nn

class Gmu(nn.Module):
    def __init__(self, text_dim, img_dim, output_dim):
        super(Gmu, self).__init__()
        # dimensions
        self.text_dim = text_dim
        self.img_dim = img_dim
        self.output_dim = output_dim

        # neural network layers for projection and gating
        self.fc_text = nn.Linear(text_dim, output_dim)
        self.fc_img = nn.Linear(img_dim, output_dim)
        self.gate = nn.Linear(text_dim + img_dim, output_dim)

        # activation functions
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()



    def forward(self, image_emb, text_emb):
        # project the vectors into the expected output dimension to blend them together
        text_proj = self.tanh(self.fc_text(text_emb))
        img_proj = self.tanh(self.fc_img(image_emb))

        combined = torch.cat((text_emb, image_emb), dim=1)
        gate_values = self.sigmoid(self.gate(combined))

        output = gate_values * text_proj + (1 - gate_values) * img_proj
        return output
