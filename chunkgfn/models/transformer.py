import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)


class Transformer(nn.Module):
    def __init__(
        self,
        state_dim,
        action_embedding_dim,
        hidden_dim,
        num_layers,
        num_head,
        max_len=60,
        dropout=0,
    ):
        super().__init__()
        self.pos = PositionalEncoding(hidden_dim, dropout=dropout, max_len=max_len + 2)
        self.embedding = nn.Linear(state_dim, hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(
            hidden_dim, num_head, hidden_dim, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.action_embedding_layer = nn.Linear(hidden_dim, action_embedding_dim)
        self.action_embedding_layer.weight.data.fill_(0.0)
        self.action_embedding_layer.bias.data.fill_(0.0)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos(x)

        x = self.encoder(x, src_key_padding_mask=mask)
        pooled_x = x[:, 0]

        y = self.action_embedding_layer(pooled_x)
        return y
