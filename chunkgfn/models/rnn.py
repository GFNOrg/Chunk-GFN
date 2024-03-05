import torch
from torch import nn


class RNN(nn.Module):
    def __init__(self, num_layers, hidden_dim, input_dim, state_dim, act, n_actions):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.n_actions = n_actions

        self.input_embedding = nn.Linear(input_dim, hidden_dim)
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
        self.logits_layer = nn.Linear(hidden_dim, n_actions)
        self.logits_layer.weight.data.fill_(0.0)
        self.logits_layer.bias.data.fill_(0.0)

    def forward(self, x, state):
        """
        Args:
            x [batch_size, max_len, input_dim]
            state [batch_size, max_len, state_dim]
        """

        _, x_ = self.rnn_encoder(self.input_embedding(x))
        s, _ = self.rnn_encoder(self.state_embedding(state), x_)
        out = s[:, -1]
        for layer in self.layers:
            out = layer(out)
        out = self.logits_layer(out)

        return out


class UnconditionalRNN(nn.Module):
    def __init__(self, num_layers, hidden_dim, state_dim, act, n_actions):
        super(UnconditionalRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.n_actions = n_actions

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
        self.logits_layer = nn.Linear(hidden_dim, n_actions)
        self.logits_layer.weight.data.fill_(0.0)
        self.logits_layer.bias.data.fill_(0.0)

    def forward(self, state):
        """
        Args:
            state [batch_size, max_len, state_dim]
        """
        s, _ = self.rnn_encoder(self.state_embedding(state))
        out = s[:, -1]
        for layer in self.layers:
            out = layer(out)
        out = self.logits_layer(out)

        return out


class RNN_Embedding(nn.Module):
    def __init__(
        self,
        num_layers,
        hidden_dim,
        input_dim,
        embedding_dim,
        state_dim,
        act,
        n_actions,
    ):
        super(RNN_Embedding, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.embedding_dim = embedding_dim

        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        self.state_embedding = nn.Linear(state_dim, hidden_dim)
        self.action_embeddings = nn.Parameter(torch.randn(n_actions, embedding_dim))

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
        self.emmbedding_layer = nn.Linear(hidden_dim, embedding_dim)
        self.emmbedding_layer.weight.data.fill_(0.0)
        self.emmbedding_layer.bias.data.fill_(0.0)

    def get_logits(self, embedding):
        logits = torch.matmul(embedding, self.action_embeddings.T)
        logits = logits / (self.embedding_dim**0.5)  # Same scaling as in self attention
        return logits

    def forward(self, x, state):
        """
        Args:
            x [batch_size, max_len, input_dim]
            state [batch_size, max_len, state_dim]
        """

        _, x_ = self.rnn_encoder(self.input_embedding(x))
        s, _ = self.rnn_encoder(self.state_embedding(state), x_)
        out = s[:, -1]
        for layer in self.layers:
            out = layer(out)
        embedding = self.emmbedding_layer(out)
        logits = self.get_logits(embedding)

        return logits


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
