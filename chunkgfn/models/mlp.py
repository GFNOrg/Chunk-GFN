from torch import nn


class MLP(nn.Module):
    def __init__(self, num_layers, hidden_dim, in_dim, activation, n_actions):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.n_actions = n_actions

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hidden_dim))
        self.layers.append(activation)
        for _ in range(self.num_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(activation)
        self.layers.append(nn.LayerNorm(hidden_dim))
        self.logits_layer = nn.Linear(hidden_dim, n_actions)
        self.logits_layer.weight.data.fill_(0.0)
        self.logits_layer.bias.data.fill_(0.0)

    def forward(self, state):
        """
        Args:
            state [batch_size, dim]
        """
        out = state
        for layer in self.layers:
            out = layer(out)
        out = self.logits_layer(out)

        return out
