from abc import ABC
from typing import List

import torch
from torch.utils.data import Dataset

from .base_module import BaseUnconditionalEnvironmentModule


class SequenceDataset(Dataset):
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


class BaseSequenceModule(BaseUnconditionalEnvironmentModule, ABC):
    """Base Sequence module that can be inherited from for specific tasks."""

    def __init__(
        self,
        atomic_tokens: List[str],
        max_len: int,
        num_train_iterations: int,
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
        self.sample_exact_length = sample_exact_length

        # Environment variables
        self.discovered_modes = set()  # Tracks the number of modes we discovered
        self.visited = set()  # Tracks the number of states we visited
        self.atomic_tokens = (
            [self.exit_action] + atomic_tokens
        )  # Atomic tokens for representing the states. Stays fixed during training.
        self.s0 = -torch.ones(
            1 + self.max_len, len(self.atomic_tokens)
        )  # Initial state
        self.padding_token = -torch.ones(len(self.atomic_tokens))
        self.eos_token = torch.tensor([1] + [0] * (len(self.atomic_tokens) - 1))
        # Actions can change during training. Not to be confused with atomic_tokens.
        self.actions = self.atomic_tokens.copy()
        self.action_len = torch.ones(
            len(self.actions)
        ).long()  # Length of each action. Can change during training.
        self.action_frequency = torch.zeros(
            len(self.actions)
        )  # Tracks the frequency of each action. Can change during training.

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
            if action == self.exit_action:
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
            if action != self.exit_action:
                action_indices[action] = [self.atomic_tokens.index(a) for a in action]
            else:
                action_indices[action] = [0]

        return action_indices

    def preprocess_states(self, states: torch.Tensor) -> torch.Tensor:
        """Preprocess states so that it can be input to the policy model.
        Args:
            states (torch.Tensor[batch_size, max_len, dim]): The states.
        Returns:
            processed_states (torch.Tensor[batch_size, max_len, dim]): The preprocessed states.
        """
        return states

    def is_initial_state(self, states: torch.Tensor) -> torch.Tensor:
        """Check if the states are the initial state.
        Args:
            states (torch.Tensor[batch_size, max_len, dim]): Batch of states.
        Returns:
            is_initial (torch.Tensor[batch_size]): Whether the states are the initial state or not.
        """
        is_initial = (states == self.s0.to(states.device)).all(dim=-1).all(dim=-1)
        return is_initial

    def to_strings(self, states: torch.Tensor) -> list[str]:
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

    def get_forward_mask(self, states: torch.Tensor):
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
        eos_token_idx = self.atomic_tokens.index(self.exit_action)
        eos_token_idx = self.atomic_tokens.index(self.exit_action)
        if self.sample_exact_length:
            # Don't allow the EOS token to be sampled if the state is not full
            actions_mask[len_tokens_to_go > 1, eos_token_idx] = 0

        actions_mask[len_tokens_to_go <= 1, eos_token_idx] = (
            1  # We make sure that the EOS token is always available at the last step
        )
        actions_mask = actions_mask.to(states.device)
        return actions_mask

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
        self.data_val = SequenceDataset(
            state_dict["data_val_sequences"], state_dict["data_val_logrewards"]
        )
        self.data_test = SequenceDataset(
            state_dict["data_test_sequences"], state_dict["data_test_logrewards"]
        )

    def build_test(self):
        raise NotImplementedError  # TODO: add random sampling by default

    def setup_val_test_datasets(self):
        val_seq, val_rs = self.build_test()
        test_seq, test_rs = self.build_test()

        self.data_val = SequenceDataset(val_seq, val_rs)
        self.data_test = SequenceDataset(test_seq, test_rs)
