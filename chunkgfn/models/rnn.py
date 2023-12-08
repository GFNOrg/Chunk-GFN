from torch import nn


class RNN(nn.Module):
    def __init__(self, num_layers, hidden_dim, vocab_size, act):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding = nn.Linear(vocab_size, hidden_dim)

        self.rnn_encoder = nn.GRU(
            hidden_dim, hidden_dim, num_layers=2, batch_first=True
        )

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(act)
        for _ in range(self.num_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(act)
        self.layers.append(nn.LayerNorm(hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, vocab_size))

    def forward(self, x, state):
        """
        Args:
            x [B, N, dim]
            state [B, M, dim]
        """

        _, x_ = self.rnn_encoder(self.embedding(x))
        s, _ = self.rnn_encoder(self.embedding(state), x_)
        out = s[:, -1]
        for layer in self.layers:
            out = layer(out)

        return out


class logZ(nn.Module):
    def __init__(self, num_layers, hidden_dim, vocab_size, act):
        super(logZ, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding = nn.Linear(vocab_size, hidden_dim)

        self.rnn_encoder = nn.GRU(
            hidden_dim, hidden_dim, num_layers=2, batch_first=True
        )

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(act)
        for _ in range(self.num_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(act)
        self.layers.append(nn.LayerNorm(hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, 1))

    def forward(self, x):
        """
        Args:
            x [B, N, dim]
        """
        x__, _ = self.rnn_encoder(self.embedding(x))
        logZ = x__[:, -1]
        for layer in self.layers:
            logZ = layer(logZ)
        return logZ
