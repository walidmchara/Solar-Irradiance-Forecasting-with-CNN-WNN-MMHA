from __future__ import annotations

import torch
from torch import nn


class WaveletFeatureBlock(nn.Module):
    def __init__(self, channels: int, hidden_size: int):
        super().__init__()
        self.low = nn.Sequential(
            nn.Linear(channels, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.high = nn.Sequential(
            nn.Linear(channels, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x):
        # x: (batch, time, channels)
        low = torch.mean(x, dim=1, keepdim=True)
        high = x - low
        return self.low(low).squeeze(1), self.high(high).mean(dim=1)


class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        batch_size, seq_len, _ = x.size()
        q = q.view(batch_size, seq_len, self.num_heads, self.d_model // self.num_heads).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_model // self.num_heads).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_model // self.num_heads).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_model // self.num_heads) ** 0.5
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out(context)


class CNNWNNMMHARegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
                 num_heads: int = 4, dropout: float = 0.2, max_length: int = 512):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.conv1 = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(hidden_size),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(hidden_size),
        )
        self.wavelet = WaveletFeatureBlock(hidden_size, hidden_size)

        self.pos_embedding = nn.Parameter(torch.zeros(1, max_length, hidden_size))
        self.attn_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                MaskedMultiHeadAttention(hidden_size, num_heads=num_heads, dropout=dropout),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.transpose(1, 2)

        low, high = self.wavelet(x)
        x = x + low.unsqueeze(1) + high.unsqueeze(1)
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]

        for layer in self.attn_layers:
            residual = x
            x = layer[0](x)
            x = layer[1](x)
            x = residual + x
            x = x + self.mlp(layer[0](x))

        return self.head(x[:, -1]).squeeze(-1)
