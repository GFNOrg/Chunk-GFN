from torch import nn

from .base_policy import BasePolicy


class UnconditionalRNN(BasePolicy):
    def __init__(self, num_layers, hidden_dim, state_dim, act, action_embedding_dim):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.action_embedding_dim = action_embedding_dim

        self.state_embedding = nn.Linear(state_dim, hidden_dim)

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
        # This layer generates the action embedding that will be used for picking the next action
        self.action_embedding_layer = nn.Linear(hidden_dim, action_embedding_dim)
        self.action_embedding_layer.weight.data.fill_(0.0)
        self.action_embedding_layer.bias.data.fill_(0.0)

    def forward(self, state):
        """
        Args:
            state [batch_size, max_len, state_dim]
        """
        s, _ = self.rnn_encoder(self.state_embedding(state))
        out = s[:, -1]
        for layer in self.layers:
            out = layer(out)
        out = self.action_embedding_layer(out)

        return out


class logZ(nn.Module):
    def __init__(self, num_layers, hidden_dim, input_vocab_size, act):
        super(logZ, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.input_vocab_size = input_vocab_size
        self.embedding = nn.Linear(input_vocab_size, hidden_dim)

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
            x [B, N, input_dim]
        """
        x__, _ = self.rnn_encoder(self.embedding(x))
        logZ = x__[:, -1]
        for layer in self.layers:
            logZ = layer(logZ)
        return logZ


class Critic(nn.Module):
    def __init__(self, num_layers, hidden_dim, state_dim, act):
        super(Critic, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim

        self.state_embedding = nn.Linear(state_dim, hidden_dim)

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

        self.action_embedding_layer = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        """
        Args:
            state [batch_size, max_len, state_dim]
        """
        s, _ = self.rnn_encoder(self.state_embedding(state))
        out = s[:, -1]
        for layer in self.layers:
            out = layer(out)
        out = self.action_embedding_layer(out)

        return out
