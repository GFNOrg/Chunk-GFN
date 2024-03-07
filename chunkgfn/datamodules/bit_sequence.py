import random

import numpy as np
import torch
from polyleven import levenshtein
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from torch.utils.data import Dataset

from .base_module import BaseUnconditionalEnvironmentModule


class BitSequenceModule(BaseUnconditionalEnvironmentModule):
    """A `BitSequenceModule` for defining the bit-sequence task in (Malkin, et. al. 2022).
    Based on: https://gist.github.com/MJ10/59bfcc8bce4b5fce9c1c38a81b1105ae
    """

    def __init__(
        self,
        max_len: int,
        num_modes: int,
        num_train_iterations: int,
        num_val_iterations: int,
        num_test_iterations: int,
        threshold: float,
        oracle_difficulty: str = "medium",
        batch_size: int = 64,
        sample_exact_length: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__(
            num_train_iterations,
            num_val_iterations,
            num_test_iterations,
            batch_size,
            num_workers,
            pin_memory,
        )
        self.max_len = max_len
        self.oracle_difficulty = oracle_difficulty
        self.num_modes = num_modes
        self.threshold = threshold
        self.sample_exact_length = sample_exact_length

        # Environment variables
        self.discovered_modes = set()  # Tracks the number of modes we discovered
        self.visited = set()  # Tracks the number of states we visited
        self.atomic_tokens = [
            "<EOS>",
            "0",
            "1",
        ]  # Atomic tokens for representing the states. Stays fixed during training.
        self.s0 = -torch.ones(
            1 + self.max_len, len(self.atomic_tokens)
        )  # Initial state
        self.padding_token = -torch.ones(len(self.atomic_tokens))
        self.eos_token = torch.tensor([1, 0, 0])
        self.actions = [
            "<EOS>",
            "0",
            "1",
        ]  # Actions can change during training. Not to be confused with atomic_tokens.
        self.action_len = torch.Tensor(
            [1, 1, 1]
        ).long()  # Length of each action. Can change during training.

        self.create_modes()

    @property
    def one_hot_action_tensor(self):
        """One-hot encoding tensor for self.actions. Actions that are composed of more than an atomic token,
        will have a one-hot encoding that spans multiple timesteps.
        """
        one_hot_action_tensor = -torch.ones(
            len(self.actions), self.action_len.max().item(), len(self.atomic_tokens)
        )
        for action in self.actions:
            idx = self.actions.index(action)
            if action == "<EOS>":
                one_hot_action_tensor[idx, :1] = torch.eye(len(self.atomic_tokens))[
                    self.atomic_tokens.index(action)
                ]
            else:
                one_hot_action_tensor[idx, : len(action)] = torch.eye(
                    len(self.atomic_tokens)
                )[[self.atomic_tokens.index(x) for x in action]]
        return one_hot_action_tensor

    def create_modes(self):
        """Create the modes for the bit-sequence task."""

        if self.oracle_difficulty == "medium":
            vocab = ["00000000", "11111111", "11110000", "00001111", "00111100"]
            self.modes = [
                "".join(random.choices(vocab, k=self.max_len // len(vocab[0])))
                for _ in range(self.num_modes)
            ]

        elif self.oracle_difficulty == "hard":
            vocab = ["00000000", "11111111", "11110000", "00001111", "00111100"]
            self.modes = [
                "".join(
                    random.choices(
                        vocab,
                        k=random.randint(
                            (self.max_len // len(vocab[0])) * 0.5,
                            (self.max_len // len(vocab[0])),
                        ),
                    )
                )
                for _ in range(self.num_modes)
            ]
        self.len_modes = torch.tensor([len(m) for m in self.modes])

    def is_initial_state(self, states: torch.Tensor) -> torch.Tensor:
        """Check if the state is the initial state.
        Args:
            states (torch.Tensor[batch_size, max_len, dim]): Batch of states.
        Returns:
            is_initial (bool): Whether the state is the initial state or not.
        """
        is_initial = (
            (states == self.s0.to(states.device)).all(dim=-1).all(dim=-1)
        )  # shape: [batch_size]
        return is_initial

    def to_raw(self, states: torch.Tensor) -> list[str]:
        """Convert the states to raw data.
        Args:
            states (torch.Tensor[batch_size, max_len, dim]): Batch of states.
        Returns:
            raw (list[str]): List of states in their string representation.
        """
        strings = []
        for state in states.cpu():
            # Cut the state before it arrives at [-1,-1,...]
            nonzero = (state == self.padding_token).nonzero()
            if len(nonzero) > 0:
                state = state[: nonzero[0][0]]

            indices = state.argmax(dim=-1)
            strings.append("".join([self.atomic_tokens[i] for i in indices]))
        return strings

    def compute_logreward(self, states: torch.Tensor) -> torch.Tensor:
        """Compute the reward for the given state and action.
        Args:
            state (torch.Tensor[batch_size, max_len, dim]): Batch of states.
        Returns:
            reward (torch.Tensor[batch_size]): Batch of rewards.
        """
        strings = [
            s.replace("<EOS>", "") for s in self.to_raw(states)
        ]  # remove <EOS> tokens for computing reward

        dists = torch.tensor([[levenshtein(s, i) for i in self.modes] for s in strings])
        values, indices = torch.min(dists, dim=-1)
        reward = 1 - values / self.len_modes[indices]
        return reward

    def compute_logreward_from_strings(self, strings: list[str]) -> torch.Tensor:
        """Compute the reward for a list of strings directly.
        Args:
            string (list[str]): List of strings.
        Returns:
            reward (torch.Tensor): Batch of rewards.
        """
        dists = torch.tensor([[levenshtein(s, i) for i in self.modes] for s in strings])
        values, indices = torch.min(dists, dim=-1)
        reward = 1 - values / self.len_modes[indices]
        return reward

    def compute_metrics(self, states: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute metrics for the given state.
        Args:
            state (torch.Tensor[batch_size, max_len, dim]): Batch of states.
        Returns:
            metrics (dict[str, torch.Tensor]): Dictionary of metrics.
        """
        strings = [
            s.replace("<EOS>", "") for s in self.to_raw(states)
        ]  # remove <EOS> tokens
        self.visited.update(set(strings))
        dists = torch.tensor([[levenshtein(s, i) for i in self.modes] for s in strings])
        values, indices = torch.min(dists, dim=-1)
        reward = 1 - values / self.len_modes[indices]
        mode_indices = indices[
            reward > self.threshold
        ].tolist()  # Find the indices of the modes that are close to the samples
        modes_found = set([self.modes[i] for i in mode_indices])
        self.discovered_modes.update(modes_found)
        metrics = {
            "num_modes": len(self.discovered_modes),
            "num_visited": len(self.visited),
        }

        return metrics

    def forward_step(self, state: torch.Tensor, forward_action: torch.Tensor):
        """Change the state after you apply the forward action.
        Args:
            state (torch.Tensor[batch_size, max_len, dim]): Batch of states.
            forward_action (torch.Tensor[batch_size]): Batch of forward actions. Each element corresponds to the index of the action.
        Returns:
            new_state (torch.Tensor[batch_size, max_len, dim]): Batch of new states.
            done (torch.Tensor[batch_size]): Whether the trajectory is done or not.
        """
        bs, max_len, dim = state.shape
        eos_token = self.eos_token.to(state)
        padding_token = self.padding_token.to(state)
        one_hot_action_tensor = self.one_hot_action_tensor.to(state.device)
        max_action_len = one_hot_action_tensor.shape[1]
        # Update the state by filling the current timestep with the sampled action only if it doesn't contain EOS token
        new_state = torch.cat(
            [
                state.clone(),
                padding_token.unsqueeze(0).unsqueeze(1).repeat(bs, max_action_len, 1),
            ],
            dim=1,
        )
        start_indices = torch.argmax(
            ((state == padding_token).all(dim=-1) + 0), dim=-1
        )  # Where to start inserting action
        index = start_indices.unsqueeze(1) + torch.arange(max_action_len).unsqueeze(
            0
        ).to(state.device)
        done = torch.where((state == eos_token).all(dim=-1).any(dim=-1), True, False)
        new_state = torch.where(
            done.unsqueeze(1).unsqueeze(2),
            new_state,
            torch.scatter(
                new_state,
                1,
                index.unsqueeze(2).repeat(1, 1, dim),
                one_hot_action_tensor[forward_action],
            ),
        )
        new_state = new_state[:, :max_len, :]

        return new_state, done

    def backward_step(self, state: torch.Tensor, backward_action: torch.Tensor):
        """Change the state after you apply the backward action.
        Args:
            state (torch.Tensor[batch_size, max_len, dim]): Batch of states.
            backward_action (torch.Tensor[batch_size]): Batch of backward actions. Each element corresponds to the index of the action.
        Returns:
            new_state (torch.Tensor[batch_size, max_len, dim]): Batch of new states.
        """
        bs, max_len, dim = state.shape
        new_state = state.clone()
        action_len = self.action_len.to(state.device)

        where_padding = (state == self.padding_token.to(state)).all(dim=-1)

        start_indices = torch.where(
            where_padding.any(dim=-1), torch.argmax(where_padding + 0, dim=-1), max_len
        )
        done = start_indices == 0
        mask = torch.arange(max_len).unsqueeze(0).to(state.device) >= (
            start_indices - action_len[backward_action]
        ).unsqueeze(1)
        mask &= torch.arange(max_len).unsqueeze(0).to(state.device) < (
            start_indices
        ).unsqueeze(1)
        mask &= (~done).unsqueeze(-1)
        new_state[mask] = self.padding_token.to(state.device)

        return new_state, done

    def get_invalid_actions_mask(self, states: torch.Tensor):
        """Get the invalid actions mask for a batch of states.
        Args:
            states (torch.Tensor[batch_size, max_len, dim]): Batch of states.
        Returns:
            actions_mask (torch.Tensor[batch_size, n_actions]): Invalid actions mask.
        """
        # Count how many tokens we can still insert
        len_tokens_to_go = (
            (states == self.padding_token.to(states.device)).all(dim=-1).sum(dim=1)
        )
        actions_mask = len_tokens_to_go.unsqueeze(1) > self.action_len.to(
            states.device
        ).unsqueeze(0)  # Only use actions that can fit in the state
        eos_token_idx = self.atomic_tokens.index("<EOS>")
        if self.sample_exact_length:
            # Don't allow the EOS token to be sampled if the state is not full
            actions_mask[len_tokens_to_go > 1, eos_token_idx] = 0

        actions_mask[len_tokens_to_go <= 1, eos_token_idx] = (
            1  # We make sure that the EOS token is always available at the last step
        )
        actions_mask = actions_mask.to(states.device)
        return actions_mask

    def chunk(self, final_states: torch.Tensor):
        """Find the most valuable token from the corpus.
        Args:
            final_states (torch.Tensor[batch_size, max_len, state_vocab_dim]): Batch of final states.
        """
        # Convert token indices to strings
        state_strings = [s.replace("<EOS>", "") for s in self.to_raw(final_states)]
        # Apply BPE algorithm to the state_strings and get the most frequent token
        vocab_dict = {k: i for i, k in enumerate(self.actions)}
        tokenizer = Tokenizer(BPE(vocab_dict, [], unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=len(self.actions))
        tokenizer.train_from_iterator(state_strings, trainer=trainer)
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

    def state_dict(self):
        state = {
            "discovered_modes": self.discovered_modes,
            "visited": self.visited,
            "actions": self.actions,
            "action_len": self.action_len,
            "modes": self.modes,
            "len_modes": self.len_modes,
            "data_val": self.data_val,
            "data_test": self.data_test,
        }
        return state

    def load_state_dict(self, state_dict):
        self.discovered_modes = state_dict["discovered_modes"]
        self.visited = state_dict["visited"]
        self.actions = state_dict["actions"]
        self.action_len = state_dict["action_len"]
        self.modes = state_dict["modes"]
        self.len_modes = state_dict["len_modes"]
        self.data_val = state_dict["data_val"]
        self.data_test = state_dict["data_test"]

    def get_parent_actions(self, states: torch.Tensor):
        """Get the parent actions of a batch of states.
        Args:
            states (torch.Tensor[batch_size, max_len, state_vocab_dim]): Batch of states.
        """
        # Convert token indices to strings
        state_strings = self.to_raw(states)
        # Get the parent actions
        parent_actions = torch.zeros(
            states.shape[0], len(self.actions), dtype=torch.int64
        )
        max_action_len = self.action_len.long().max().item()
        for i, state in enumerate(state_strings):
            if len(state) > 0:
                if "<EOS>" in state:
                    parent_actions_ = ["<EOS>"]
                else:
                    parent_actions_ = set()
                    for j in range(max_action_len):
                        parent_actions_.add("".join(state[-j - 1 :]))
                    parent_actions_ = list(
                        parent_actions_.intersection(set(self.actions))
                    )
                parent_actions_ = [self.actions.index(a) for a in parent_actions_]
                parent_actions[i, parent_actions_] = 1
            else:
                parent_actions[i] = 1
        return parent_actions

    def build_test(self):
        """Build the test points around the modes.
        returns:
            test_seq (list[str]): List of test sequences.
            test_rs (list[float]): List of test logrewards.
        """
        test_seq = []
        vocab = ["0", "1"]

        def noise_seq(x, n):
            x = list(x)
            idces = list(range(len(x)))
            for i in range(n):
                j = idces.pop(np.random.randint(len(idces)))
                r = x[j]
                while r == x[j]:
                    r = vocab[np.random.randint(len(vocab))]
                x[j] = r
            return "".join(x)

        for m in self.modes:
            for n in range(1, len(m) + 1):
                s = noise_seq(m, n)
                s_idx = torch.tensor(
                    [self.atomic_tokens.index(char) for char in s]
                    + [self.atomic_tokens.index("<EOS>")]
                )
                s_tensor = torch.zeros(s_idx.shape[0], len(self.atomic_tokens))
                s_tensor[torch.arange(s_idx.shape[0]), s_idx] = 1
                test_seq.append(s_tensor)
        test_seq = torch.stack(test_seq, dim=0)
        test_rs = self.compute_logreward(test_seq)
        return test_seq, test_rs

    def setup_val_test_datasets(self):
        val_seq, val_rs = self.build_test()
        test_seq, test_rs = self.build_test()

        class BitSequenceDataset(Dataset):
            def __init__(self, sequences, logrewards):
                self.sequences = sequences
                self.logrewards = logrewards

            def __len__(self):
                return len(self.sequences)

            def __getitem__(self, index):
                """Get the sequence and logreward at the given index.
                Args:
                    index (int): The index.
                Returns:
                    seq (torch.Tensor[max_len, dim]): The sequence.
                    logr (torch.Tensor): The logreward.
                """
                seq, logr = self.sequences[index], self.logrewards[index]
                return seq, logr

        self.data_val = BitSequenceDataset(val_seq, val_rs)
        self.data_test = BitSequenceDataset(test_seq, test_rs)
