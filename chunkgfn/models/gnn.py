import torch
from torch import nn
import torch_geometric.nn as pyg_nn

from .base_policy import BasePolicy

class GAT(BasePolicy):
    def __init__(self, num_layers, hidden_dim, node_dim, act, action_embedding_dim, global_feature_dim):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.node_dim = node_dim
        self.action_embedding_dim = action_embedding_dim
        self.global_feature_dim = global_feature_dim

        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        self.global_embedding = nn.Linear(global_feature_dim, hidden_dim)
        
        self.gnn_layers = nn.ModuleList([
            pyg_nn.GATv2Conv(hidden_dim, hidden_dim) for _ in range(2)
        ])

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(hidden_dim * 2, hidden_dim))  # *2 for concatenated global feature
        self.layers.append(act)
        for _ in range(self.num_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(act)
        self.layers.append(nn.LayerNorm(hidden_dim))
        
        self.action_embedding_layer = nn.Linear(hidden_dim, action_embedding_dim)
        self.action_embedding_layer.weight.data.fill_(0.0)
        self.action_embedding_layer.bias.data.fill_(0.0)

    def forward(self, node_features, edge_index, batch_id, is_final):
        """
        Forward pass through the UnconditionalGNN model.

        Args:
            node_features (torch.Tensor): Node features of shape [num_nodes, dim].
            edge_index (torch.Tensor): Edge index of shape [2, num_edges].
            batch_id (torch.Tensor): Batch ID for each node of shape [num_nodes].
            is_final (torch.Tensor): One-hot tensor indicating if the graph is in its final state, of shape [batch_size,2].
        """

        # Apply GNN layers
        x_graph = self.node_embedding(node_features)
        for gnn_layer in self.gnn_layers:
            x_graph = gnn_layer(x_graph, edge_index)
            x_graph = torch.relu(x_graph)

        graph_embedding = pyg_nn.global_mean_pool(x_graph, batch_id)

        # Process global feature
        global_out = self.global_embedding(is_final)

        # Concatenate graph embedding and global feature
        out = torch.cat([graph_embedding, global_out], dim=1)

        # Apply remaining layers
        for layer in self.layers:
            out = layer(out)
        out = self.action_embedding_layer(out)

        return out

class Critic(nn.Module):
    def __init__(self, num_layers, hidden_dim, node_dim, act, global_feature_dim):
        super(Critic, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.node_dim = node_dim
        self.global_feature_dim = global_feature_dim

        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        self.global_embedding = nn.Linear(global_feature_dim, hidden_dim)
        
        self.gnn_layers = nn.ModuleList([
            pyg_nn.GATv2Conv(hidden_dim, hidden_dim) for _ in range(2)
        ])

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(hidden_dim * 2, hidden_dim))  # *2 for concatenated global feature
        self.layers.append(act)
        for _ in range(self.num_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(act)
        self.layers.append(nn.LayerNorm(hidden_dim))

        self.q_value = nn.Linear(hidden_dim, 1)

    def forward(self, node_features, edge_index, batch_id, is_final):
        """
        Forward pass through the UnconditionalGNN model.

        Args:
            node_features (torch.Tensor): Node features of shape [num_nodes, dim].
            edge_index (torch.Tensor): Edge index of shape [2, num_edges].
            batch_id (torch.Tensor): Batch ID for each node of shape [num_nodes].
            is_final (torch.Tensor): One-hot tensor indicating if the graph is in its final state, of shape [batch_size,2].
        """

        # Apply GNN layers
        x_graph = self.node_embedding(node_features)
        for gnn_layer in self.gnn_layers:
            x_graph = gnn_layer(x_graph, edge_index)
            x_graph = torch.relu(x_graph)

        graph_embedding = pyg_nn.global_mean_pool(x_graph, batch_id)

        # Process global feature
        global_out = self.global_embedding(is_final)

        # Concatenate graph embedding and global feature
        out = torch.cat([graph_embedding, global_out], dim=1)

        # Apply remaining layers
        for layer in self.layers:
            out = layer(out)
        out = self.q_value(out)

        return out
