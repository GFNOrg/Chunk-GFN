import torch
from torch import nn


class RNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, out_dim, act=nn.ReLU()):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Linear(in_dim, hidden_dim)

        self.rnn_encoder = nn.GRU(
            hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(act)
        for _ in range(self.num_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(act)
        self.layers.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, xstate):
        """
        Args:
            xstate [batch_shape, 2*seq_length, dim]: Tensor that contains the conditioning vector which takes up seq_length of space as well as the state that takes the remaining seq_length.
        """
        x, state = torch.chunk(xstate, 2, 1)
        _, x_ = self.rnn_encoder(self.embedding(x))
        s, _ = self.rnn_encoder(self.embedding(state), x_)
        out = s[:, -1]
        for layer in self.layers:
            out = layer(out)
        return out
