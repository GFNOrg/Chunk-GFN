from torch import nn

from .base_policy import BasePolicy


class MLP(BasePolicy):
    def __init__(
        self, num_layers, hidden_dim, in_dim, activation, action_embedding_dim
    ):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.action_embedding_dim = action_embedding_dim

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hidden_dim))
        self.layers.append(activation)
        for _ in range(self.num_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(activation)
        self.layers.append(nn.LayerNorm(hidden_dim))
        # This layer generates the action embedding that will be used for picking the next action
        self.action_embedding_layer = nn.Linear(hidden_dim, action_embedding_dim)
        self.action_embedding_layer.weight.data.fill_(0.0)
        self.action_embedding_layer.bias.data.fill_(0.0)

    def forward(self, state):
        """
        Generate the action embedding given a state
        Args:
            state [batch_size, dim]: Input state.
        Returns:
            action_embedding [batch_size, action_embedding]: Action embedding.
        """
        out = state
        for layer in self.layers:
            out = layer(out)
        action_embedding = self.action_embedding_layer(out)

        return action_embedding


class Critic(BasePolicy):
    def __init__(self, num_layers, hidden_dim, in_dim, activation):
        super(Critic, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hidden_dim))
        self.layers.append(activation)
        for _ in range(self.num_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(activation)
        self.layers.append(nn.LayerNorm(hidden_dim))
        # This layer generates the action embedding that will be used for picking the next action
        self.action_embedding_layer = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        """
        Generate the action embedding given a state
        Args:
            state [batch_size, dim]: Input state.
        Returns:
            action_embedding [batch_size, action_embedding]: Action embedding.
        """
        out = state
        for layer in self.layers:
            out = layer(out)
        action_embedding = self.action_embedding_layer(out)

        return action_embedding
