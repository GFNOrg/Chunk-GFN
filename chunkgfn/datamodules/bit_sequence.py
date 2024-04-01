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


class BitSequenceModule(BaseUnconditionalEnvironmentModule):
    """A `BitSequenceModule` for defining the bit-sequence task in (Malkin, et. al. 2022).
    Based on: https://gist.github.com/MJ10/59bfcc8bce4b5fce9c1c38a81b1105ae
    """

    def __init__(
        self,
        max_len: int,
        num_modes: int,
        num_train_iterations: int,
        threshold: float,
        oracle_difficulty: str = "medium",
        batch_size: int = 64,
        sample_exact_length: bool = False,
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
        self.action_frequency = torch.zeros(
            len(self.actions)
        )  # Tracks the frequency of each action. Can change during training.

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
            if action != "<EOS>":
                action_indices[action] = [self.atomic_tokens.index(a) for a in action]
            else:
                action_indices[action] = [0]

        return action_indices

    def preprocess_state(self, states: torch.Tensor) -> torch.Tensor:
        """Preprocess states so that it can be input to the policy model.
        Args:
            states (torch.Tensor[batch_size, max_len, dim]): The states.
        Returns:
            processed_states (torch.Tensor[batch_size, max_len, dim]): The preprocessed states.
        """
        return states

    def create_modes(self):
        """Create the modes for the bit-sequence task depending on the oracle difficulty.
        If the difficulty is "medium" then the modes all have the same length `max_len`
        and are unique.
        If the difficulty is "hard" then the modes have different lengths with
        maximum length of `max_len` and minimum length of `max_len//2` and are unique.
        """
        vocab = ["00000000", "11111111", "11110000", "00001111", "00111100"]
        self.modes = set()
        if self.oracle_difficulty == "medium":
            while len(self.modes) < self.num_modes:
                self.modes.add(
                    "".join(random.choices(vocab, k=self.max_len // len(vocab[0])))
                )

        elif self.oracle_difficulty == "hard":
            while len(self.modes) < self.num_modes:
                self.modes.add(
                    "".join(
                        random.choices(
                            vocab,
                            k=random.randint(
                                (self.max_len // len(vocab[0])) * 0.5,
                                (self.max_len // len(vocab[0])),
                            ),
                        )
                    )
                )

        self.modes = list(self.modes)
        self.len_modes = torch.tensor([len(m) for m in self.modes])

    def is_initial_state(self, states: torch.Tensor) -> torch.Tensor:
        """Check if the states are the initial state.
        Args:
            states (torch.Tensor[batch_size, max_len, dim]): Batch of states.
        Returns:
            is_initial (torch.Tensor[batch_size]): Whether the states are the initial state or not.
        """
        is_initial = (states == self.s0.to(states.device)).all(dim=-1).all(dim=-1)
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
        """Compute the reward for the given states and action.
        Args:
            states (torch.Tensor[batch_size, max_len, dim]): Batch of states.
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
        """Compute metrics for the given states.
        Args:
            states (torch.Tensor[batch_size, max_len, dim]): Batch of states.
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
            "num_modes": float(len(self.discovered_modes)),
            "num_visited": float(len(self.visited)),
        }

        return metrics

    def forward_step(self, states: torch.Tensor, forward_action: torch.Tensor):
        """Change the states after you apply the forward action.
        Args:
            states (torch.Tensor[batch_size, max_len, dim]): Batch of states.
            forward_action (torch.Tensor[batch_size]): Batch of forward actions. Each element corresponds to the index of the action.
        Returns:
            new_states (torch.Tensor[batch_size, max_len, dim]): Batch of new states.
            done (torch.Tensor[batch_size]): Whether the trajectory is done or not.
        """
        bs, max_len, dim = states.shape
        eos_token = self.eos_token.to(states)
        padding_token = self.padding_token.to(states)
        one_hot_action_tensor = self.one_hot_action_tensor.to(states.device)
        max_action_len = one_hot_action_tensor.shape[1]
        # Update the state by filling the current timestep with the sampled action only if it doesn't contain EOS token
        new_states = torch.cat(
            [
                states.clone(),
                padding_token.unsqueeze(0).unsqueeze(1).repeat(bs, max_action_len, 1),
            ],
            dim=1,
        )
        start_indices = torch.argmax(
            ((states == padding_token).all(dim=-1) + 0), dim=-1
        )  # Where to start inserting action
        index = start_indices.unsqueeze(1) + torch.arange(max_action_len).unsqueeze(
            0
        ).to(states.device)
        done = torch.where((states == eos_token).all(dim=-1).any(dim=-1), True, False)
        new_states = torch.where(
            done.unsqueeze(1).unsqueeze(2),
            new_states,
            torch.scatter(
                new_states,
                1,
                index.unsqueeze(2).repeat(1, 1, dim),
                one_hot_action_tensor[forward_action],
            ),
        )
        new_states = new_states[:, :max_len, :]

        used_actions = forward_action[
            ~done
        ]  # Only picks the actions that actually are used for updating the state.
        self.action_frequency += torch.bincount(
            used_actions.to(self.action_frequency.device), minlength=len(self.actions)
        )

        return new_states, done

    def backward_step(self, states: torch.Tensor, backward_action: torch.Tensor):
        """Change the states after you apply the backward action.
        Args:
            states (torch.Tensor[batch_size, max_len, dim]): Batch of states.
            backward_action (torch.Tensor[batch_size]): Batch of backward actions. Each element corresponds to the index of the action.
        Returns:
            new_states (torch.Tensor[batch_size, max_len, dim]): Batch of new states.
        """
        bs, max_len, dim = states.shape
        new_states = states.clone()
        action_len = self.action_len.to(states.device)

        where_padding = (states == self.padding_token.to(states)).all(dim=-1)

        start_indices = torch.where(
            where_padding.any(dim=-1), torch.argmax(where_padding + 0, dim=-1), max_len
        )
        done = start_indices == 0
        mask = torch.arange(max_len).unsqueeze(0).to(states.device) >= (
            start_indices - action_len[backward_action]
        ).unsqueeze(1)
        mask &= torch.arange(max_len).unsqueeze(0).to(states.device) < (
            start_indices
        ).unsqueeze(1)
        mask &= (~done).unsqueeze(-1)
        new_states[mask] = self.padding_token.to(states.device)

        return new_states, done

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

    def chunk(self, actions: torch.Tensor, dones: torch.Tensor):
        """Find the most valuable subsequence of actions from the corpus.
        Args:
            actions (torch.Tensor[batch_size, traj_length]): Batch of sequence of actions.
            dones (torch.Tensor[batch_size, traj_length]): Batch of sequence of terminations.
        """
        # Convert token indices to strings
        dones = dones[:, :-1]  # The last step is always True
        action_strings = [
            "".join([self.actions[j] for j in action if not dones[i, j]]).replace(
                "<EOS>", ""
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
            "modes": self.modes,
            "len_modes": self.len_modes,
            "data_val_sequences": self.data_val.sequences,
            "data_val_logrewards": self.data_val.logrewards,
            "data_test_sequences": self.data_test.sequences,
            "data_test_logrewards": self.data_test.logrewards,
        }
        return state

    def load_state_dict(self, state_dict):
        self.discovered_modes = state_dict["discovered_modes"]
        self.visited = state_dict["visited"]
        self.actions = state_dict["actions"]
        self.action_len = state_dict["action_len"]
        self.modes = state_dict["modes"]
        self.len_modes = state_dict["len_modes"]
        self.data_val = BitSequenceDataset(
            state_dict["data_val_sequences"], state_dict["data_val_logrewards"]
        )
        self.data_test = BitSequenceDataset(
            state_dict["data_test_sequences"], state_dict["data_test_logrewards"]
        )

    def get_parent_actions(self, states: torch.Tensor):
        """Get the parent actions of a batch of states.
        Args:
            states (torch.Tensor[batch_size, max_len, state_vocab_dim]): Batch of states.
        """
        bs, max_len, dim = states.shape
        padding_token = self.padding_token.to(states.device)
        one_hot_action_tensor = self.one_hot_action_tensor.to(states.device)
        final_state = torch.cat(
            [
                states,
                padding_token.unsqueeze(0)
                .unsqueeze(1)
                .repeat(bs, 1 + self.action_len.max().item(), 1),
            ],
            dim=1,
        )
        n_actions = one_hot_action_tensor.shape[0]
        one_hot_action_tensor = torch.cat(
            [
                one_hot_action_tensor,
                padding_token.unsqueeze(0).unsqueeze(1).repeat(n_actions, 1, 1),
            ],
            dim=1,
        )
        simplified = torch.argmax(final_state, dim=-1)
        simplified = torch.where(
            (final_state == padding_token).all(dim=-1),
            torch.tensor(len(self.atomic_tokens)),
            simplified,
        )  # dummy value to map padding_token to
        simplified_one_hot_action_tensor = torch.argmax(one_hot_action_tensor, dim=-1)
        simplified_one_hot_action_tensor = torch.where(
            (one_hot_action_tensor == padding_token).all(dim=-1),
            torch.tensor(len(self.atomic_tokens)),
            simplified_one_hot_action_tensor,
        )
        unfolded = simplified.unfold(
            1, 1 + self.action_len.max().item(), 1
        )  # This generates a sliding windows view of the tensor that we can compare against.
        parent_actions = (
            (unfolded.unsqueeze(-2) == simplified_one_hot_action_tensor)
            .all(dim=-1)
            .any(dim=1)
            .long()
        )
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

        self.data_val = BitSequenceDataset(val_seq, val_rs)
        self.data_test = BitSequenceDataset(test_seq, test_rs)
