import math

import torch
from einops import repeat
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
        output_dim,
        hidden_dim,
        num_layers,
        num_head,
        max_len=60,
        dropout=0,
        use_bos_token=False,
    ):
        super().__init__()
        if use_bos_token:
            self.bos_token = nn.Parameter(torch.randn(state_dim))
            self.pos = PositionalEncoding(
                hidden_dim, dropout=dropout, max_len=max_len + 2
            )
        else:
            self.bos_token = None
            self.pos = PositionalEncoding(
                hidden_dim, dropout=dropout, max_len=max_len + 1
            )
        self.embedding = nn.Linear(state_dim, hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(
            hidden_dim, num_head, hidden_dim, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output = nn.Linear(hidden_dim, output_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, mask=None):
        bs, max_len, dim = x.shape
        if self.bos_token is not None:
            bos_token_ = repeat(self.bos_token, "d -> b t d", b=bs, t=1)
            x = torch.cat([bos_token_, x], dim=1)
            if mask is not None:
                mask = torch.cat([torch.zeros(bs, 1, device=mask.device), mask], dim=1)

        x = self.embedding(x)
        x = self.pos(x)

        x = self.encoder(x, src_key_padding_mask=mask)
        pooled_x = x[:, 0]

        y = self.output(pooled_x)
        return y
