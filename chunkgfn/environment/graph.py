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
        threshold: int = 25,
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
        self.discovered_modes = []  # Tracks the number of modes we discovered
        self.visited = set()  # Tracks the number of states we visited
        self.max_nodes = max_nodes
        self.threshold = threshold
        
        self.atomic_tokens = [self.exit_action] + ["A"] + [chr(i+ord("B")-1) for i in range(1, max_nodes)]
        self.alpha2token = {self.exit_action: self.exit_action, "A": "A"} | {chr(i+ord("B")-1): f"E-{i}" for i in range(1, max_nodes)}
        self.actions = self.atomic_tokens.copy()
        self.action_len = torch.ones(
                len(self.actions)
            ).long()
        self.action_frequency = torch.zeros(
            len(self.actions)
        )  # Tracks the frequency of each action. Can change during training.
        
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
        _actions = ["<EOG>" if k == "<EOG>" else "".join([self.alpha2token[a] for a in k]) for k in self.actions]
        _atomic_tokens = [self.alpha2token[k] for k in self.atomic_tokens]
        for action in _actions:
            split_action = self._split_action_sequence(action)
            action_indices[action] = [_atomic_tokens.index(a) for a in split_action]
        return action_indices
    
    @cached_property_with_invalidation("actions")
    def one_hot_action_tensor(self):
        """One-hot encoding tensor for self.actions. Actions that are composed of more
        than an atomic token, will have a one-hot encoding that spans multiple timesteps.
        """
        one_hot_action_tensor = []
        num_action_nodes = []
        max_action_size = -1
        _actions = ["<EOG>" if k == "<EOG>" else "".join([self.alpha2token[a] for a in k]) for k in self.actions]
        for action in _actions:
            split_action = self._split_action_sequence(action)
            action_tensor = []
            for act in split_action:
                if act == "<EOG>":
                    action_tensor.append(torch.ones(1, self.max_nodes))
                elif act == "A":
                    act_tensor = torch.zeros(1, self.max_nodes)
                    act_tensor[:,0] = 1
                    action_tensor.append(act_tensor)
                else:
                    
                    edge_idx = int(act[2:])
                    
                    if len(action_tensor) == 0:
                        act_tensor = torch.zeros(1, self.max_nodes)
                        act_tensor[:,edge_idx] = 1
                        action_tensor.append(act_tensor)
                    else:
                        action_tensor[-1][:,edge_idx] = 1
            
            action_tensor = torch.cat(action_tensor, dim=0)
            max_action_size = max(max_action_size, action_tensor.shape[0])
            num_action_nodes.append(len(action_tensor))
            one_hot_action_tensor.append(action_tensor)

        for i, action_tensor in enumerate(one_hot_action_tensor):
            if action_tensor.shape[0] < max_action_size:
                # Pad with -1
                action_tensor = torch.cat([action_tensor, -torch.ones(max_action_size - action_tensor.shape[0], self.max_nodes)], dim=0)
            one_hot_action_tensor[i] = action_tensor
        one_hot_action_tensor = torch.stack(one_hot_action_tensor, dim=0)
        self.num_action_nodes = torch.tensor(num_action_nodes)
        return one_hot_action_tensor
    
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
    
    def _split_action_sequence(self, action_sequence: str) -> list:
        actions = []
        i = 0
        while i < len(action_sequence):
            if action_sequence[i:i+5] == '<EOG>':
                actions.append('<EOG>')
                i += 5
            elif action_sequence[i] == 'A':
                actions.append('A')
                i += 1
            elif action_sequence[i:i+2] == 'E-':
                j = i + 2
                while j < len(action_sequence) and action_sequence[j].isdigit():
                    j += 1
                actions.append(action_sequence[i:j])
                i = j
            else:
                raise ValueError(f"Invalid action in sequence: {action_sequence[i:]}")
        return actions

    def get_forward_mask(self, state: torch.Tensor) -> torch.Tensor:
        batch_size, _, max_nodes = state.shape
        mask = torch.zeros(batch_size, len(self.actions), dtype=torch.bool, device=state.device)
        _arange = torch.arange(batch_size, device=state.device)

        _actions = ["<EOG>" if k == "<EOG>" else "".join([self.alpha2token[a] for a in k]) for k in self.actions]
        for i, action_sequence in enumerate(_actions):
            individual_actions = self._split_action_sequence(action_sequence)    
            mask[:, i] = torch.ones(batch_size, dtype=torch.bool, device=state.device)
            
            temp_state = state.clone()
            
            
            for action in individual_actions:
                temp_last_node_idx = self._get_last_node_index(temp_state)
                if action == '<EOG>':
                    # Allow stopping if there's exactly one node or if the last node is connected
                    is_single_node = (temp_last_node_idx == 0)
                    is_last_node_connected = (temp_last_node_idx > 0) & (temp_state[_arange, temp_last_node_idx].sum(dim=1) > 0)
                    mask[:, i] &= (is_single_node | is_last_node_connected)
                elif action == 'A':
                    # Allow adding a node if we haven't reached MAX_NODES and either:
                    # 1. We have one or less nodes
                    # 2. The last node is connected
                    can_add_node = (temp_last_node_idx < max_nodes - 1)
                    is_one_or_less_nodes = (temp_last_node_idx <= 0)
                    is_last_node_connected = (temp_last_node_idx > 0) & (temp_state[_arange, temp_last_node_idx].sum(dim=1) > 0)

                    mask[:, i] &= can_add_node & (is_one_or_less_nodes | is_last_node_connected)
                    
                    # Simulate adding a node
                    if mask[:, i].any():
                        temp_last_node_idx += 1
                        # Fill the new row with zeros
                        temp_state[_arange[mask[:, i]], temp_last_node_idx[mask[:, i]]] = 0
                elif action.startswith('E-'):
                    edge_idx = int(action[2:])  # Extract the number after 'E-'
                    

                    # Allow adding an edge for multi-node graphs
                    mask[:, i] &= (temp_last_node_idx > 0)
                    
                    # Ensure the edge connection is valid
                    farthest_connected = self._get_farthest_connected_node_index(temp_state)
                    relative_to_last_node = temp_last_node_idx - farthest_connected
                    mask[:, i] &= (edge_idx > relative_to_last_node)
                    mask[:, i] &= (edge_idx <= temp_last_node_idx)
                    
                    
                    # Simulate adding the edge
                    if mask[:, i].any():
                        temp_state[_arange[mask[:, i]], temp_last_node_idx[mask[:, i]], temp_last_node_idx[mask[:, i]]-edge_idx] = 1
                
                # If at any point the action is not valid, break the sequence
                if not mask[:, i].any():
                    break
        return mask
    
    def get_backward_mask(self, state: torch.Tensor) -> torch.Tensor:
        batch_size = state.shape[0]
        mask = torch.zeros(batch_size, len(self.actions), dtype=torch.bool, device=state.device)
        _arange = torch.arange(batch_size, device=state.device)

        _actions = ["<EOG>" if k == "<EOG>" else "".join([self.alpha2token[a] for a in k]) for k in self.actions]
        for i, action_sequence in enumerate(_actions):
            individual_actions = self._split_action_sequence(action_sequence)[::-1] # We check in reverse   
            mask[:, i] = torch.ones(batch_size, dtype=torch.bool, device=state.device)
            
            temp_state = state.clone()
            
            
            for action in individual_actions:
                is_final = self.is_final_state(temp_state)
                temp_last_node_idx = self._get_last_node_index(temp_state)
                if action == '<EOG>':
                    # Allow stopping if the state is final
                    mask[:, i] &= is_final
                elif action == 'A':
                    # Allow removing a node if the last is not connected (which applies also to a graph with a single node)
                    is_last_node_not_connected = ~(temp_state[_arange, temp_last_node_idx].sum(dim=1) > 0)
                    mask[:, i] &= is_last_node_not_connected & (~is_final)
                    
                    # Simulate removing a node
                    if mask[:, i].any():
                        temp_state[_arange[mask[:, i]], temp_last_node_idx[mask[:, i]]] = -1
                        temp_last_node_idx -= 1                
                
                elif action.startswith('E-'):
                    edge_idx = int(action[2:])  # Extract the number after 'E-'
                    
                    is_last_node_connected = (temp_state[_arange, temp_last_node_idx].sum(dim=1) > 0)
                    farthest_connected = self._get_farthest_connected_node_index(temp_state)
                    relative_to_last_node = temp_last_node_idx - farthest_connected
                    # Allow adding an edge for multi-node graphs
                    mask[:, i] &= is_last_node_connected  & (~is_final)
                    mask[:, i] &= (relative_to_last_node == edge_idx)  & (~is_final)
                
                    # Simulate removing the edge
                    if mask[:, i].any():
                        temp_state[_arange[mask[:, i]], temp_last_node_idx[mask[:, i]], temp_last_node_idx[mask[:, i]]-edge_idx] = 0
                
                # If at any point the action is not valid, break the sequence
                if not mask[:, i].any():
                    break
        return mask

    def forward_step(self, state: torch.Tensor, action: torch.Tensor):
        max_nodes = state.shape[-1]
        new_state = torch.cat(
                [
                    state.clone(),
                    -torch.ones_like(state).to(state)
                ],
                dim=1,
            )
        one_hot_action_tensor = self.one_hot_action_tensor.to(state.device)
        last_node_idx = self._get_last_node_index(state)
        done = self.is_final_state(state)
        
        to_exit = (action == 0) & (~done) # Exit action
        
        action_to_be_applied = one_hot_action_tensor[action]
        bs, n_rows, n_edges = action_to_be_applied.shape
        to_append = (action_to_be_applied[:,0,0] == 1).clone()
        
        rel2abs_idx = last_node_idx.unsqueeze(1)+torch.arange(n_rows, device=state.device).unsqueeze(0)
        rel2abs_idx[to_append] += 1
        action_to_be_applied = self._rel2abs(action_to_be_applied[...,1:], rel2abs_idx)
        
        # Deal with actions that first add a new node
        nodefirst_mask = to_append & (~done) & (~to_exit)
        new_state[nodefirst_mask] = torch.scatter(
                    new_state[nodefirst_mask],
                    1,
                    (rel2abs_idx[nodefirst_mask].unsqueeze(2)).repeat(1,1,max_nodes),
                    action_to_be_applied[nodefirst_mask],
                )

        # Deal with actions that first add edges to the last node
        edgefirst_mask = ~to_append & (~done) & (~to_exit)
        new_state[edgefirst_mask, last_node_idx[edgefirst_mask]] += action_to_be_applied[edgefirst_mask, 0]
        new_state[edgefirst_mask] = torch.scatter(
                        new_state[edgefirst_mask],
                        1,
                        (rel2abs_idx[edgefirst_mask][:,1:].unsqueeze(2)).repeat(1,1,max_nodes),
                        action_to_be_applied[edgefirst_mask,1:],
                    )
        
        new_state = new_state[:,:max_nodes+1]
        # Deal with exit actions
        new_state[to_exit,-1] = 1
        
        used_actions = action[
            ~done
        ]  # Only picks the actions that actually are used for updating the state.
        self.action_frequency += torch.bincount(
            used_actions.to(self.action_frequency.device), minlength=len(self.actions)
        )
        
        return new_state, done
    
    def backward_step(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        new_state = state.clone()
        last_node_idx = self._get_last_node_index(state)
        one_hot_action_tensor = self.one_hot_action_tensor.to(state.device)
        num_action_nodes = self.num_action_nodes.to(state.device)
        done = self.is_initial_state(state)

        action_to_be_applied = one_hot_action_tensor[action]
        bs, n_rows, n_edges = action_to_be_applied.shape
        rel2abs_idx = last_node_idx.unsqueeze(1)+torch.arange(n_rows, device=state.device).unsqueeze(0)
        
        abs_action = self._rel2abs(action_to_be_applied[...,1:], rel2abs_idx)

        undo_stop = (action == 0) & ~done

        # Node first mask
        nodefirst_mask = (action_to_be_applied[:,0,0] == 1) & (~done) & (~undo_stop)
        where2remove = torch.arange(new_state.shape[1], device=state.device).unsqueeze(0) >= (
                    last_node_idx + 1 - num_action_nodes[action]
                ).unsqueeze(1)
        new_state[nodefirst_mask] = torch.where(where2remove[nodefirst_mask].unsqueeze(-1), -1, new_state[nodefirst_mask])

        # Edge first mask
        edgefirst_mask = (action_to_be_applied[:,0,0] != 1) & (~done) & (~undo_stop)
        action_with_node = (action_to_be_applied[...,0] == 1).any(dim=-1) & (~done) & (~undo_stop)
        # First, undo action starting with node addition
        where2remove = torch.arange(new_state.shape[1], device=state.device).unsqueeze(0) >= (
                    last_node_idx + 2 - num_action_nodes[action]
                ).unsqueeze(1)
        new_state[edgefirst_mask & action_with_node] = torch.where(
            where2remove[edgefirst_mask & action_with_node].unsqueeze(-1),
            -1,
            new_state[edgefirst_mask & action_with_node])
        # Now, whatever is left is only edge removal
        last_node_idx = self._get_last_node_index(new_state)
        edge_positions = abs_action[:,0].bool()
        new_state[edgefirst_mask, last_node_idx[edgefirst_mask]] = torch.where(edge_positions[edgefirst_mask], 0, new_state[edgefirst_mask, last_node_idx[edgefirst_mask]])

        # Last, we deal with undoing exit action
        new_state[undo_stop, -1] = -1
        
        return new_state, done
    
    def _rel2abs(self, action_tensor, last_node_idx):
        batch_size, n_rows, n_edges = action_tensor.shape


        # Initialize c with False values
        c = torch.zeros((batch_size, n_rows, n_edges + 1)).to(action_tensor)

        # Create a tensor representing the indices of the edges
        j_indices = torch.arange(n_edges).unsqueeze(0).unsqueeze(0).expand(batch_size, n_rows, -1).to(action_tensor.device)

        # Calculate the target indices for each True value in `a`
        target_indices = last_node_idx.unsqueeze(-1) - j_indices - 1

        # Mask to keep only valid indices within the range [0, num_edges]
        valid_mask = (target_indices >= 0) & (target_indices <= n_edges)

        # Apply the mask to keep only valid target indices
        target_indices = target_indices[valid_mask]

        # Batch indices and n indices
        batch_indices = torch.arange(batch_size, device=action_tensor.device).unsqueeze(1).unsqueeze(2).expand(batch_size, n_rows, n_edges)[valid_mask]        
        n_indices = torch.arange(n_rows, device=action_tensor.device).unsqueeze(0).unsqueeze(-1).expand(batch_size, n_rows, n_edges)[valid_mask]

        # Set corresponding positions in c to True
        c[batch_indices, n_indices, target_indices] = action_tensor[valid_mask]
        c[(c == -1).any(dim=-1)] = -1
        return c

    def _get_last_node_index(self, state: torch.Tensor) -> torch.Tensor:
        return (state[:, :-1] != -1).any(dim=-1).sum(dim=-1) - 1
    
    def _get_farthest_connected_node_index(self, state: torch.Tensor) -> torch.Tensor:
        bs, _, max_nodes = state.shape
        last_node_idx = self._get_last_node_index(state)
        last_node = state[torch.arange(bs), last_node_idx]
        
        connected = (last_node == 1).any(dim=-1)
        farthest_connected = torch.argmax((state[torch.arange(bs), last_node_idx] == 1).float(), dim=-1)
        farthest_connected[~connected] = last_node_idx[~connected]
        return farthest_connected

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
        # num_nodes = torch.tensor([len(graph.nodes) for graph in graphs])
        max_cycles = self.max_nodes * (self.max_nodes - 1) // 2 - self.max_nodes + 1
        reward = cycle_basis_lengths / max_cycles
        # reward[max_cycles <= 0] = 0
        reward = torch.clip(reward, min=EPS)
        logreward = reward.log()
        return logreward
    
    def compute_metrics(
        self, states: torch.Tensor, logreward: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute metrics for the given state.
        Args:
            state (torch.Tensor[batch_size, ndim+1]): Batch of states.
        Returns:
            metrics (dict[str, torch.Tensor]): Dictionary of metrics.
        """
        graphs = self.state_to_networkx(states)
        cycle_basis_lengths = [len(nx.cycle_basis(G)) for G in graphs]
        adj_strings = self.state_to_adj_string(states)
        # Filter unique modes
        
  
        
        self.visited.update(set(adj_strings))
        metrics = {
            "num_modes": float(len(self.discovered_modes)),
            "num_visited": float(len(self.visited)),
        }
        return metrics
    
    def state_to_adj_string(self, states):
        graphs = self.state_to_networkx(states)
        strings = []
        for graph in graphs:
            adj_string = []
            for node in graph.nodes:
                neighbors = ";".join(list(map(str, graph.neighbors(node))))
                if len(neighbors) == 0:
                    adj_string.append(f"{node}")
                else:
                    adj_string.append(f"{node}:{neighbors}")
            strings.append("|".join(adj_string))
        return strings
    
    def state_dict(self):
        state = {
            "discovered_modes": self.discovered_modes,
            "visited": self.visited,
            "actions": self.actions,
            "action_len": self.action_len,
            "action_frequency": self.action_frequency,
            "data_val_samples": self.data_val.samples,
            "data_val_logrewards": self.data_val.logrewards,
            "data_test_samples": self.data_test.samples,
            "data_test_logrewards": self.data_test.logrewards,
        }
        return state

    def load_state_dict(self, state_dict):
        self.discovered_modes = state_dict["discovered_modes"]
        self.visited = state_dict["visited"]
        self.actions = state_dict["actions"]
        self.action_len = state_dict["action_len"]
        self.action_frequency = state_dict["action_frequency"]
        self.data_val = GraphDataset(
            state_dict["data_val_samples"], state_dict["data_val_logrewards"]
        )
        self.data_test = GraphDataset(
            state_dict["data_test_samples"], state_dict["data_test_logrewards"]
        )
        
    def _create_dataset(self):
        max_nodes = self.max_nodes
        _modes = torch.ones(1, max_nodes + 1, max_nodes)
        _modes[0,:max_nodes, :max_nodes] = torch.tril(torch.ones((max_nodes, max_nodes)), diagonal=-1)
        _modes = _modes.repeat(16, 1, 1)
        _logrewards = self.compute_logreward(_modes)
        return _modes, _logrewards

    def setup_val_test_datasets(self):
        val_samples, val_logrewards = self._create_dataset()
        test_samples, test_logrewards = self._create_dataset()
        self.data_val = GraphDataset(val_samples, val_logrewards)
        self.data_test = GraphDataset(test_samples, test_logrewards)

class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, samples, logrewards):
        self.samples = samples
        self.logrewards = logrewards

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """Get the sample and logreward at the given index.
        Args:
            index (int): The index.
        Returns:
            sample (torch.Tensor[max_len, dim]): The sample.
            logr (torch.Tensor): The logreward.
        """
        sample, logr = self.samples[index], self.logrewards[index]
        return sample, logr