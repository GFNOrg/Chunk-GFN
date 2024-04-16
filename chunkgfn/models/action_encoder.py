from torch import nn


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
