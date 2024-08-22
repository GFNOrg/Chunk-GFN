import torch
from torch import nn


class ThermEncoder(nn.Module):
    def __init__(self, num_layers, hidden_dim, embedding_size, activation):
        super(ThermEncoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_size = embedding_size

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(embedding_size, hidden_dim))
        self.layers.append(activation)
        for _ in range(self.num_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(activation)
        self.layers.append(nn.LayerNorm(hidden_dim))
        self.logit_layer = nn.Linear(hidden_dim, 1)

    def forward(self, temperature):
        """
        Generate the action embedding given a state
        Args:
            state [batch_size, dim]: Input state.
        Returns:
            action_embedding [batch_size, action_embedding]: Action embedding.
        """
        assert (temperature > 0).all() and (
            temperature <= 1
        ).all(), "The temperature needs to be in (0,1]"
        levels = torch.linspace(
            0, 1, self.embedding_size, device=temperature.device
        ).float()
        encoded = (temperature.unsqueeze(-1) >= levels.unsqueeze(0)).float()
        out = encoded
        for layer in self.layers:
            out = layer(out)
        logit_scale = self.logit_layer(out)

        return logit_scale
