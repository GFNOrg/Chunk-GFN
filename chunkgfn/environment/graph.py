import torch
import networkx as nx
from typing import List

from .base_module import BaseUnConditionalEnvironmentModule
from chunkgfn.utils.cache import cached_property_with_invalidation
from ..constants import EPS
from einops import repeat
from torch_geometric.utils import dense_to_sparse
import torch.nn.functional as F

class GraphGenerationModule(BaseUnConditionalEnvironmentModule):
    def __init__(
        self,
        num_train_iterations: int,
        max_nodes: int = 10,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ) -> None:
        self.save_hyperparameters(logger=False)
        super().__init__(
            num_train_iterations,
            batch_size,
            num_workers,
            pin_memory,
            **kwargs,
        )
        # Environment variables
        self.exit_action = "<EOG>"
        self.discovered_modes = set()  # Tracks the number of modes we discovered
        self.visited = set()  # Tracks the number of states we visited
        self.max_nodes = max_nodes
        
        self.atomic_tokens = [self.exit_action] + ["A"] + [chr(i + ord("B")-1) for i in range(1, max_nodes)]
        self.actions = self.atomic_tokens.copy()
        
        self.s0 = -torch.ones(self.max_nodes + 1, self.max_nodes)

    @cached_property_with_invalidation("actions")
    def n_actions(self):
        return len(self.actions)
    
    @cached_property_with_invalidation("actions")
    def action_indices(self) -> dict[str, int]:
        """Get the action indices. For each action, if it's a primitive one then keep
        its a list of one element which is its original index, otherwise, keep a list of
        indices of the primitive actions that make up the action.
        Returns:
            action_indices (dict[str, list[int]]): Dictionary of action indices.
        """
        action_indices = {}
        for action in self.actions:
            if action != self.exit_action:
                action_indices[action] = [self.atomic_tokens.index(a) for a in action]
            else:
                action_indices[action] = [0]

        return action_indices
    
    def _directed_adjancecy_to_undirected(self, adj: torch.Tensor) -> torch.Tensor:
        """
        Converts a directed adjacency matrix to an undirected one by adding the transpose of the matrix to itself.

        Args:
            adj (torch.Tensor): The directed adjacency matrix.

        Returns:
            torch.Tensor: The undirected adjacency matrix.
        """
        adj_T = adj.transpose(2,1).clone()
        adj_T[adj_T == -1] = 0
        undirected_adj = adj+adj_T
        return undirected_adj

    def _add_bog_node(self, adj: torch.Tensor) -> torch.Tensor:
        """
        Adds a virtual BOG (Beginning of Graph) node to the adjacency matrix.

        This method prepends a row and column to the adjacency matrix to simulate a BOG node. The BOG node is connected to all other nodes.

        Args:
            adj (torch.Tensor): The original adjacency matrix.

        Returns:
            torch.Tensor: The modified adjacency matrix with the added BOG node.
        """
        bs, max_nodes, _ = adj.shape
        adj_with_bog_node = -torch.ones(bs, max_nodes+1, max_nodes+1).to(adj)
        adj_with_bog_node[:,1:,1:] = adj
        adj_with_bog_node[:,0] = 0 # Add BOG node
        
        mask = ~(adj == -1).all(dim=-1) # Fetch indices of existant nodes
        adj_with_bog_node[:,0,1:][mask] = 1 # Connect BOG to all valid nodes
        adj_with_bog_node[:,1:,0][mask] = 1 # Connect all valid nodes to BOG
        return adj_with_bog_node
    
    def preprocess_states(self, state: torch.Tensor):
        """
        Preprocesses the given state tensor for graph generation tasks.

        This method extracts adjacency matrix, node features, batch ids, and checks if the state is final.
        It masks out non-existent nodes (those containing -1) and prepares the data for further processing.

        Args:
            state (torch.Tensor): The state tensor to preprocess. Expected shape is [batch_size, max_nodes+1, max_nodes].

        Returns:
            tuple: A tuple containing:
                - node_features (torch.Tensor): A tensor of shape [masked_batch_size, 1] containing node features.
                - edge_index (torch.Tensor): A tensor of shape [2, num_edges] containing edge indices.
                - batch_ids (torch.Tensor): A tensor of shape [masked_batch_size] containing batch ids.
                - is_final (torch.Tensor): A one-hot tensor of shape [batch_size, 2] indicating if the state is final.
        """
        bs, _, max_nodes = state.shape
        adj = self._directed_adjancecy_to_undirected(state[:,:-1])
        adj = self._add_bog_node(adj)
        is_final = F.one_hot(self.is_final_state(state).long(), num_classes=2).float()
        mask = ~(adj == -1).all(dim=-1) # Masks non-existant nodes (the ones containing -1)
        
        # Create node features. There are no node attributes, so they all have the same value
        node_features = torch.ones(bs, max_nodes+1, 1, dtype=torch.float).to(state.device)
        node_features[:,0] = 0 # Feature for BOG node
        
        # Batch ids (just like torch geometric)
        batch_ids = repeat(torch.arange(bs), "... -> ... n", n=max_nodes+1).to(state.device)
        
        # Use mask
        node_features = node_features[mask]
        batch_ids = batch_ids[mask]
        edge_index, _ = dense_to_sparse(adj, mask)
        return node_features, edge_index, batch_ids, is_final
        

    def setup_val_test_datasets(self):
        # For this environment, we don't need separate val/test datasets
        pass

    def is_initial_state(self, states: torch.Tensor) -> torch.Tensor:
        """Check if the states are the initial state.
        Args:
            states (torch.Tensor[batch_size, max_nodes+1, max_nodes]): Batch of states.
        Returns:
            is_initial (torch.Tensor[batch_size]): Whether the states are the initial state or not.
        """
        is_initial = (states == self.s0.to(states.device)).all(dim=-1).all(dim=-1)
        return is_initial
    
    def is_final_state(self, states: torch.Tensor) -> torch.Tensor:
        """Check if the states are the final state.
        Args:
            states (torch.Tensor[batch_size, max_nodes+1, max_nodes]): Batch of states.
        Returns:
            is_final (torch.Tensor[batch_size]): Whether the states are the final state or not.
        """
        is_final = (states[:,-1] == 1).all(dim=-1)
        return is_final

    def get_forward_mask(self, state: torch.Tensor) -> torch.Tensor:
        batch_size = state.shape[0]
        mask = torch.zeros(batch_size, len(self.actions), dtype=torch.bool).to(state.device)
        
        last_node_idx = self._get_last_node_index(state)
        is_empty = (last_node_idx == -1)
        is_single_node = (last_node_idx == 0)
        is_multi_node = (last_node_idx > 0)
        
        # Empty graph: only allow adding a node
        mask[is_empty, 1] = True
        
        # Single node: allow stopping or adding a node
        mask[is_single_node, :2] = True
        
        # Multi-node graphs
        if is_multi_node.any():
            multi_node_indices = torch.where(is_multi_node)[0]
            for i in multi_node_indices:
                last_node_connected = (state[i, last_node_idx[i], :last_node_idx[i]] == 1).any()
                
                if not last_node_connected:
                    # If last node is not connected, force edge creation
                    mask[i, 2:2+last_node_idx[i]] = True
                else:
                    # If last node is connected, allow stopping, adding node, or adding edge to farther nodes
                    mask[i, :2] = True  # Allow stopping and adding node
                    
                    # Find the farthest connected node
                    farthest_connected = (state[i, last_node_idx[i], :last_node_idx[i]] == 1).nonzero(as_tuple=True)[0].min()
                    relative_to_last_node = last_node_idx[i] - farthest_connected # This computes the E-i
                    
                    # Allow adding edges only to nodes farther than the farthest connected node
                    if farthest_connected > 0:
                        mask[i, 2+relative_to_last_node:2+last_node_idx[i]] = True
        
        # Ensure we can only add a node if we haven't reached MAX_NODES
        mask[:, 1] &= (last_node_idx < self.max_nodes - 1)
        
        return mask
    
    def get_backward_mask(self, state: torch.Tensor) -> torch.Tensor:
        batch_size = state.shape[0]
        mask = torch.zeros(batch_size, len(self.actions), dtype=torch.bool).to(state.device)
        
        last_node_idx = self._get_last_node_index(state)
        is_complete = self.is_final_state(state)
        is_single_node = (last_node_idx == 0) & ~is_complete
        is_multi_node = (last_node_idx > 0) & ~is_complete
        
        # Complete graph: only allow undoing <EOG>
        mask[is_complete, 0] = True
        
        # Single node: only allow removing that node
        mask[is_single_node, 1] = True
        
        # Multi-node graphs
        if is_multi_node.any():
            multi_node_indices = torch.where(is_multi_node)[0]
            for i in multi_node_indices:
                last_node_connected = (state[i, last_node_idx[i], :last_node_idx[i]] == 1).any()
                
                if not last_node_connected:
                    # If last node is not connected, only allow removing the node
                    mask[i, 1] = True
                else:
                    # If last node is connected, allow undoing the most recently added edge
                    farthest_connected = (state[i, last_node_idx[i], :last_node_idx[i]] == 1).nonzero(as_tuple=True)[0].min()
                    relative_to_last_node = last_node_idx[i] - farthest_connected
                    mask[i, 1+relative_to_last_node] = True
        
        return mask

    def forward_step(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform a forward step in the graph environment.

        This function takes the current state and an action, and returns the new state
        after applying the action, along with a boolean indicating whether the episode is done.

        Args:
            state (torch.Tensor): The current state of the graph environment.
                Shape: [batch_size, max_nodes + 1, max_nodes]
            action (torch.Tensor): The action to be applied.
                Shape: [batch_size]
                Values: 0 (stop), 1 (add node), 2+ (add edge)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - new_state (torch.Tensor): The updated state after applying the action.
                    Shape: [batch_size, max_nodes + 1, max_nodes]
                - done (torch.Tensor): A boolean tensor indicating whether each episode in the batch is done.
                    Shape: [batch_size]

        Note:
            - The state tensor represents the adjacency matrix of the graph, with an additional row
              for the stop action.
            - The 'done' tensor is computed based on the current state, not the new state.
        """
        new_state = state.clone()
        # Compute done based on the current state not the new one.
        done = self.is_final_state(state)
        
        last_node_idx = self._get_last_node_index(new_state)
        
        # Add node
        add_node = (action == 1) & ~done
        new_state[add_node, last_node_idx[add_node] + 1] = 0
        
        # Add edge
        add_edge = (action >= 2) & ~done
        edge_targets = last_node_idx[add_edge] - (action[add_edge] - 1)
        new_state[add_edge, last_node_idx[add_edge], edge_targets] = 1
        
        # Stop generation
        stop = (action == 0) & ~done
        new_state[stop, -1] = 1
        
        
        return new_state, done
    
    def backward_step(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        new_state = state.clone()
        # Compute done based on the current state not the new one.
        done = self.is_initial_state(state)
        
        last_node_idx = self._get_last_node_index(new_state)
        
        # Undo stop generation
        undo_stop = (action == 0) & ~done
        new_state[undo_stop, -1] = -1
        
        # Undo add node
        undo_add_node = (action == 1) & ~done
        new_state[undo_add_node, last_node_idx[undo_add_node]] = -1
        
        # Undo add edge
        undo_add_edge = (action >= 2) & ~done
        edge_targets = last_node_idx[undo_add_edge] - (action[undo_add_edge] - 1)
        new_state[undo_add_edge, last_node_idx[undo_add_edge], edge_targets] = 0
        
        return new_state, done

    def _get_last_node_index(self, state: torch.Tensor) -> torch.Tensor:
        return (state[:, :-1] != -1).any(dim=-1).sum(dim=-1) - 1

    def state_to_networkx(self, state_batch: torch.Tensor) -> List[nx.Graph]:
        graphs = []
        for state in state_batch:
            G = nx.Graph()
            num_nodes = (state[:-1] != -1).any(dim=-1).sum().item()
            G.add_nodes_from(range(num_nodes))
            
            for i in range(num_nodes):
                for j in range(i):
                    if state[i, j] == 1:
                        G.add_edge(i, j)
            
            graphs.append(G)
        return graphs

    def compute_logreward(self, state: torch.Tensor) -> torch.Tensor:
        graphs = self.state_to_networkx(state)
        cycle_basis_lengths = torch.tensor([len(nx.cycle_basis(G)) for G in graphs])
        return torch.log(cycle_basis_lengths+EPS)
    
    def compute_metrics(
        self, states: torch.Tensor, logreward: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute metrics for the given state.
        Args:
            state (torch.Tensor[batch_size, ndim+1]): Batch of states.
        Returns:
            metrics (dict[str, torch.Tensor]): Dictionary of metrics.
        """
        metrics = {}
        metrics = {
            "num_modes": float(len(self.discovered_modes)),
            "num_visited": float(len(self.visited)),
        }
        return metrics
    
    def state_dict(self):
        state = {
            "discovered_modes": self.discovered_modes,
            "visited": self.visited,
            "actions": self.actions,
            "action_len": self.action_len,
            "action_frequency": self.action_frequency,
        }
        return state

    def load_state_dict(self, state_dict):
        self.discovered_modes = state_dict["discovered_modes"]
        self.visited = state_dict["visited"]
        self.actions = state_dict["actions"]
        self.action_len = state_dict["action_len"]
        self.action_frequency = state_dict["action_frequency"]
