import torch
from einops import rearrange
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

from ..constants import EPS
from .base_module import BaseUnconditionalEnvironmentModule

ALPHABET = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]


def _safe_log(x):
    return torch.log(x + EPS)


class HyperGridDataset(torch.utils.data.Dataset):
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


class HyperGridModule(BaseUnconditionalEnvironmentModule):
    """A `HyperGrid` for defining the bit-sequence task in (Malkin, et. al. 2022).
    Based on: https://github.com/GFNOrg/torchgfn/blob/master/src/gfn/gym/hypergrid.py

    We represent the state as a (ndim+1) tensor where the last "bit" is the "exit" bit.

    Args:
        ndim (int): Number of dimensions of the hypergrid.
        side_length (int): Side length of the hypergrid.
        num_modes (int): Number of modes to test against.
        R0 (float): Base reward.
        R1 (float): Reward for the first condition.
        R2 (float): Reward for the second condition.
        num_train_iterations (int): Number of training iterations.
        batch_size (int, optional): Batch size. Defaults to 64.
        num_workers (int, optional): Number of workers. Defaults to 0.
        pin_memory (bool, optional): Whether to pin memory. Defaults to False.
    """

    def __init__(
        self,
        ndim: int,
        side_length: int,
        num_modes: int,
        R0: float,
        R1: float,
        R2: float,
        num_train_iterations: int,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            num_train_iterations,
            batch_size,
            num_workers,
            pin_memory,
            **kwargs,
        )
        assert (
            num_modes <= (side_length - 1 / 4) ** ndim
        ), f"Number of modes is too large. Keep it lower than (side_length - 1 / 4) ** ndim = {(side_length - 1 / 4) ** ndim}."
        assert side_length > 1, "Side length should be greater than 1."
        self.ndim = ndim
        self.side_length = side_length
        self.num_modes = num_modes
        self.R0 = R0
        self.R1 = R1
        self.R2 = R2

        # Environment variables
        self.discovered_modes = set()  # Tracks the number of modes we discovered
        self.visited = set()  # Tracks the number of states we visited

        self.s0 = torch.zeros(self.ndim + 1).long()  # Initial state
        self.sf = torch.cat(
            [torch.ones(self.ndim) * (self.side_length - 1), torch.tensor([1])]
        ).long()  # Final state
        self.actions = [ALPHABET[i] for i in range(self.ndim)] + [
            "<EXIT>"
        ]  # Actions can change during training
        self.action_len = torch.Tensor(
            [1] * len(self.actions)
        ).long()  # Length of each action. Can change during training.
        self.action_frequency = torch.zeros(
            len(self.actions)
        )  # Tracks the frequency of each action. Can change during training.

    @property
    def acting_tensor(self) -> torch.Tensor:
        """A tensor that summarizes the exact operation to be performed on the state.
        For example, If we have a chunk of actions [A, B, C] then the resulting tensor
        for that action will be: [1,1,1]
        Returns:
            acting_tensor (torch.Tensor[n_actions, ndim]): A tensor that summarizes the exact operation to be performed on the state.
        """
        acting_tensor = torch.zeros(len(self.actions), self.ndim + 1).long()
        for i, action in enumerate(self.actions):
            # The <EXIT> action is like adding a zero vector to the state.
            if action != "<EXIT>":
                for a in action:
                    acting_tensor[i, ALPHABET.index(a)] += 1
            else:
                acting_tensor[i, -1] = 1  # The "exit" bit is on.
        return acting_tensor

    @property
    def action_indices(self) -> dict[str, int]:
        """Get the action indices. For each action, if it's a primitive one then keep
        its a list of one element which is its original index, otherwise, keep a list of
        indices of the primitive actions that make up the action.
        Returns:
            action_indices (dict[str, list[int]]): Dictionary of action indices.
        """
        action_indices = {}
        for action in self.actions:
            if action != "<EXIT>":
                action_indices[action] = [ALPHABET.index(a) for a in action]
            else:
                action_indices[action] = [self.ndim]

        return action_indices

    def preprocess_state(self, states: torch.Tensor) -> torch.Tensor:
        """Preprocess the states so that it can be input to the policy model.
        Args:
            states (torch.Tensor[batch_size, ndim+1]): The states.
        Returns:
            processed_states (torch.Tensor[batch_size, (ndim+1)*side_length]): The preprocessed states.
        """
        bs, dim = states.shape
        processed_state = torch.zeros(bs, dim, self.side_length).to(states.device)
        processed_state = torch.scatter(processed_state, 2, states.unsqueeze(-1), 1)
        processed_state = rearrange(processed_state, "b d h -> b (d h)")
        return processed_state

    def is_initial_state(self, states: torch.Tensor) -> torch.Tensor:
        """Check if the state is the initial state.
        Args:
            states (torch.Tensor[batch_size, ndim+1]): Batch of states.
        Returns:
            is_initial (torch.Tensor[batch_size]): Whether the state is the initial state or not.
        """
        is_initial = (states == self.s0.to(states.device)).all(dim=-1)
        return is_initial

    def is_terminal_state(self, states: torch.Tensor) -> torch.Tensor:
        """Check if the state is terminal. The state is terminal if the "exit" bit is ON.
        Args:
            states (torch.Tensor[batch_size, ndim+1]): Batch of states.
        Returns:
            is_terminal (torch.Tensor[batch_size]): Whether the state is terminal or not.
        """
        is_terminal = states[:, -1] == 1
        return is_terminal

    def compute_logreward(self, states: torch.Tensor) -> torch.Tensor:
        """Compute the logreward for the given state. We use the logreward formula from (Malkin, et. al. 2022).
        :math:
        ```
        R(s^T) = R0 + R1\prod_{i=1}^D \mathbb{1}\left[\lvert \frac{s^d}{H-1}
        -0.5\rvert\in (0.25,0.5]\right] + R2\prod_{i=1}^D \mathbb{1}\left[\lvert \frac{s^d}{H-1}-0.5\rvert\in (0.3,0.4]\right]
        ```
        Args:
            state (torch.Tensor[batch_size, ndim]): Batch of states.
        Returns:
            logreward (torch.Tensor[batch_size]): Batch of logreward.
        """
        if len(states.shape) == self.ndim + 1:
            states = states[:, :-1]  # If the state contains the "exit" bit, remove it.

        # This is safer since floating point errors can occur.
        normalized_coords = (20 * states - 10 * (self.side_length - 1)).abs()
        reward = (
            self.R0
            + self.R1
            * torch.prod(
                (normalized_coords > 5 * (self.side_length - 1))
                & (normalized_coords <= 10 * (self.side_length - 1)),
                dim=-1,
            )
            + self.R2
            * torch.prod(
                (normalized_coords > 6 * (self.side_length - 1))
                & (normalized_coords <= 8 * (self.side_length - 1)),
                dim=-1,
            )
        )
        log_reward = _safe_log(reward)

        return log_reward

    def compute_metrics(self, states: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute metrics for the given state.
        Args:
            state (torch.Tensor[batch_size, ndim+1]): Batch of states.
        Returns:
            metrics (dict[str, torch.Tensor]): Dictionary of metrics.
        """
        metrics = {}
        states = states[:, :-1]  # The state contains the "exit" bit, remove it.
        # This is safer since floating point errors can occur.
        normalized_coords = (10 * states - 5 * (self.side_length - 1)).abs()
        modes = torch.all(
            (normalized_coords > 3 * (self.side_length - 1))
            & (normalized_coords <= 4 * (self.side_length - 1)),
            dim=-1,
        )
        modes_found = set([tuple(s.tolist()) for s in states[modes.bool()]])
        self.discovered_modes.update(modes_found)
        self.visited.update([tuple(s.tolist()) for s in states])
        metrics = {
            "num_modes": float(len(self.discovered_modes)),
            "num_visited": float(len(self.visited)),
        }
        return metrics

    def forward_step(self, states: torch.Tensor, forward_action: torch.Tensor):
        """Change the states after you apply the forward action only if
        it's not a terminal state.
        Args:
            states (torch.Tensor[batch_size, ndim+1]): Batch of states.
            forward_action (torch.Tensor[batch_size]): Batch of forward actions.
                Each element corresponds to the index of the action.
        Returns:
            new_states (torch.Tensor[batch_size, ndim+1]): Batch of new states.
            done (torch.Tensor[batch_size]): Whether state (and not new_state) is done or not.
        """
        done = self.is_terminal_state(states)
        acting_tensor = self.acting_tensor.to(states.device)
        new_states = states.clone()
        new_states[~done] += acting_tensor[forward_action[~done]]

        used_actions = forward_action[
            ~done
        ]  # Only picks the actions that actually are used for updating the state.
        self.action_frequency += torch.bincount(
            used_actions.to(self.action_frequency.device), minlength=len(self.actions)
        )

        return new_states, done

    def backward_step(self, states: torch.Tensor, backward_action: torch.Tensor):
        """Change the state after you apply the backward action only we're not
        at the initial state.
        Args:
            states (torch.Tensor[batch_size, ndim+1]): Batch of states.
            backward_action (torch.Tensor[batch_size]): Batch of backward actions.
                Each element corresponds to the index of the action.
        Returns:
            new_states (torch.Tensor[batch_size, ndim+1]): Batch of new states.
            done (torch.Tensor[batch_size]): Whether state (and not new_state) is done or not.
        """
        done = self.is_initial_state(states)
        acting_tensor = self.acting_tensor.to(states.device)
        new_states = states.clone()
        new_states[~done] -= acting_tensor[backward_action[~done]]
        return new_states, done

    def get_invalid_actions_mask(self, states: torch.Tensor):
        """Get the invalid actions mask for a batch of states.
        Args:
            states (torch.Tensor[batch_size, ndim+1]): Batch of states.
        Returns:
            actions_mask (torch.Tensor[batch_size, n_actions]): Invalid actions mask.
        """
        # Check the difference between the current state and the final one
        diff = self.sf.to(states.device) - states
        acting_tensor = self.acting_tensor.unsqueeze(0).to(states.device)
        actions_mask = (diff.unsqueeze(1) >= acting_tensor).all(dim=-1)
        actions_mask[self.is_terminal_state(states)] = False
        actions_mask[self.is_terminal_state(states), self.actions.index("<EXIT>")] = (
            True
        )

        return actions_mask

    def get_parent_actions(self, states: torch.Tensor):
        """Get the parent actions of a batch of states.
        Args:
            states (torch.Tensor[batch_size, ndim+1]): Batch of states.
        """
        diff = states - self.s0.to(states.device)
        acting_tensor = self.acting_tensor.unsqueeze(0).to(states.device)
        parent_actions = (diff.unsqueeze(1) >= acting_tensor).all(dim=-1)

        # When it's an exit state, only the <EXIT> backward action is allowed.
        parent_actions[self.is_terminal_state(states)] = False
        parent_actions[self.is_terminal_state(states), self.actions.index("<EXIT>")] = (
            True
        )
        return parent_actions

    def chunk(self, actions: torch.Tensor, dones: torch.Tensor):
        """Find the most valuable subsequence of actions from the corpus.
        Args:
            actions (torch.Tensor[batch_size, traj_length]): Batch of sequence of actions.
            dones (torch.Tensor[batch_size, traj_length]): Batch of sequence of terminations.
        """
        # Consider actions for which the trajectory is not yet done
        dones = dones[:, :-1]  # The last step is always True
        action_strings = [
            "".join([self.actions[j] for j in action if not dones[i, j]]).replace(
                "<EXIT>", ""
            )
            for i, action in enumerate(actions)
        ]

        # Apply BPE algorithm to the state_strings and get the most frequent token
        vocab_dict = {k: i for i, k in enumerate(self.actions)}
        tokenizer = Tokenizer(BPE(vocab_dict, [], unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=len(self.actions))
        tokenizer.train_from_iterator(action_strings, trainer=trainer)
        new_token = list(
            set(tokenizer.get_vocab().keys()).difference(set(self.actions))
        )[0]
        self.add_to_vocab(new_token)

    def add_to_vocab(self, token):
        if token not in self.actions:
            self.actions.append(token)
            self.action_len = torch.cat(
                [self.action_len, torch.tensor([len(token)])], dim=0
            )
            # Reset the action frequency
            self.action_frequency = torch.cat(
                [self.action_frequency * 0, torch.tensor([0.0])], dim=0
            )

    def state_dict(self):
        state = {
            "discovered_modes": self.discovered_modes,
            "visited": self.visited,
            "actions": self.actions,
            "action_len": self.action_len,
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
        self.data_val = HyperGridDataset(
            state_dict["data_val_samples"], state_dict["data_val_logrewards"]
        )
        self.data_test = HyperGridDataset(
            state_dict["data_test_samples"], state_dict["data_test_logrewards"]
        )

    def _create_dataset(self):
        """Create the a dataset of states near the modes. We sample from all states
        that are in the plateaux of height 0.5+R0 (see (Malkin, et. al. 2022)).
        Returns:
            _modes (list[torch.Tensor[ndim]]): Sampled states.
            _logrewards (list[torch.Tensor[ndim]]): Samples logrewards.
        """
        _modes = []
        _logrewards = []
        while len(_modes) < self.num_modes:
            range_one = torch.randint(
                0, int(0.25 * (self.side_length - 1)), (self.ndim,)
            )
            range_two = torch.randint(
                int(0.75 * (self.side_length - 1)), self.side_length - 1, (self.ndim,)
            )
            mask = (torch.rand(self.ndim) > 0.5).int()
            mode = range_one * (1 - mask) + range_two * mask
            mode = torch.cat([mode, torch.tensor([1])])
            if mode.tolist() not in _modes:
                _modes.append(mode)
                _logrewards.append(self.compute_logreward(mode))
        return _modes, _logrewards

    def setup_val_test_datasets(self):
        val_samples, val_logrewards = self._create_dataset()
        test_samples, test_logrewards = self._create_dataset()
        self.data_val = HyperGridDataset(val_samples, val_logrewards)
        self.data_test = HyperGridDataset(test_samples, test_logrewards)
