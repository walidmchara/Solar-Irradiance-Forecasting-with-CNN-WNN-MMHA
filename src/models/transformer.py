import torch
from torch import nn

class TransformerRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_heads=4, dropout=0.2, max_length=512):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.proj = nn.Linear(input_size, hidden_size)
        self.pos = nn.Parameter(torch.zeros(1, max_length, hidden_size))
        layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads,
            dim_feedforward=hidden_size*4, dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, 1)
    def forward(self, x):
        n = x.size(1)
        z = self.proj(x) + self.pos[:, :n]
        z = self.encoder(z)
        return self.head(self.norm(z[:, -1])).squeeze(-1)
