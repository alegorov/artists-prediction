import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, emb_dim, mult=4, p=0.0):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * mult),
            nn.Dropout(p),
            nn.GELU(),
            nn.Linear(emb_dim * mult, emb_dim)
        )

    def forward(self, x):
        return self.fc(x)


class MHSA_branch(nn.Module):
    def __init__(
            self, embedding_size, n_heads=4, dropout=0.0,
            num_layers=1, dim_feedforward=2048,
    ):
        super().__init__()
        self.norm_layers = nn.ParameterList([
            nn.LayerNorm(embedding_size) for _ in range(num_layers)
        ])
        self.encoder_layers = nn.ParameterList([
            nn.TransformerEncoderLayer(
                embedding_size,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                activation="gelu",
                batch_first=True,
                dropout=dropout,
                norm_first=False) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer, norm in zip(self.encoder_layers, self.norm_layers):
            x = norm(x)
            x = layer(x)
        return x


class AttentionPooling(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.GELU(),
            nn.Linear(embedding_size, 1)
        )

    def forward(self, x, mask=None):
        attn_logits = self.attn(x)
        if mask is not None:
            attn_logits[mask] = -float('inf')
        attn_weights = torch.softmax(attn_logits, dim=1)
        x = x * attn_weights
        # x = self.dropout(x)
        x = x.sum(dim=1)
        return x
