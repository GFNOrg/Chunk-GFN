from torch import nn

from .transformer import PositionalEncoding


class ActionModel(nn.Module):
    def __init__(
        self, n_primitive_actions=3, hidden_dim=256, action_embedding_dimension=128
    ):
        super(ActionModel, self).__init__()
        self.primitive_embedding = nn.Embedding(n_primitive_actions, hidden_dim)
        self.rnn_encoder = nn.GRU(
            hidden_dim, hidden_dim, num_layers=2, batch_first=True
        )
        self.out_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, action_embedding_dimension)
        )

    def forward(self, x):
        emb = self.primitive_embedding(x)
        s, _ = self.rnn_encoder(emb)
        out = s[:, -1]
        out = self.out_layer(out)
        return out


class ActionEncoder(nn.Module):
    def __init__(
        self,
        n_primitive_actions,
        action_embedding_dimension,
        hidden_dim,
        num_layers,
        num_head,
        max_len=60,
        dropout=0,
    ):
        super().__init__()
        self.pos = PositionalEncoding(hidden_dim, dropout=dropout, max_len=max_len + 1)
        self.embedding = nn.Embedding(n_primitive_actions, hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(
            hidden_dim, num_head, hidden_dim, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.action_embedding_layer = nn.Linear(hidden_dim, action_embedding_dimension)
        self.action_embedding_layer.weight.data.fill_(0.0)
        self.action_embedding_layer.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos(x)

        x = self.encoder(x)
        pooled_x = x[:, 0]

        y = self.action_embedding_layer(pooled_x)
        return y
