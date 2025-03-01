import random
from abc import ABC, abstractmethod
from copy import copy
from typing import Any, Optional

import torch
from lightning import LightningDataModule
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from torch.utils.data import DataLoader, Dataset


class BaseUnConditionalEnvironmentModule(LightningDataModule, ABC):
    """A `BaseUnConditionalEnvironmentModule` for defining unconditional
    environment datamodules.
    """

    def __init__(
        self,
        num_train_iterations: int,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = True,
    ) -> None:
        """Initialize the `BaseUnConditionalEnvironmentModule`.
        Args:
            batch_size (int): The batch size. Defaults to 64.
            num_workers (int): The number of workers for the dataloaders. Defaults to 0.
            pin_memory (bool): Whether to pin memory for the dataloaders. Defaults to False.

        """
        super().__init__()

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.persistent_workers = persistent_workers
        self.exit_action = "<EOS>"

        self.num_train_iterations = num_train_iterations

    @abstractmethod
    def preprocess_states(self, state: torch.Tensor) -> Any:
        """Preprocess the state so that it can be input to the policy model.
        Args:
            state (torch.Tensor[batch_size, *state_shape]): The state.
        Returns:
            *output (Any): A possible a tuple of tensors/metadata to be fed to the model.
        """
        NotImplementedError

    def setup(self, stage: Optional[str] = None) -> None:
        """Instantiate dummy datasets for the trainer to use."""

        class DummyDataset(Dataset):
            def __init__(self, num_elements):
                self.num_elements = num_elements

            def __len__(self):
                return self.num_elements

            def __getitem__(self, index):
                return index

        self.data_train = DummyDataset(
            self.num_train_iterations * self.hparams.batch_size
        )
        self.setup_val_test_datasets()

    @abstractmethod
    def setup_val_test_datasets(self):
        """Instantiate datasets for the val and test dataloaders to use."""
        NotImplementedError

    @abstractmethod
    def is_initial_state(self, state: torch.Tensor) -> bool:
        """Check if the state is the initial state.
        Args:
            state (torch.Tensor[batch_size, *state_shape]): Batch of states.
        Returns:
            is_initial (bool): Whether the state is the initial state or not.
        """
        NotImplementedError

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the val dataloader.

        :return: The val dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=self.persistent_workers,
        )

    @abstractmethod
    def forward_step(self, state: torch.Tensor, forward_action: torch.Tensor):
        """Change the state after you apply the forward action.
        Args:
            state (torch.Tensor[batch_size, *state_shape]): Batch of states.
            forward_action (torch.Tensor[batch_size]): Batch of forward actions. Each element corresponds to the index of the action.
        Returns:
            new_state (torch.Tensor[batch_size, *state_shape]): Batch of new states.
            done (torch.Tensor[batch_size]): Whether the trajectory is done or not.
        """
        NotImplementedError

    @abstractmethod
    def backward_step(self, state: torch.Tensor, backward_action: torch.Tensor):
        """Change the state after you apply the backward action.
        Args:
            state (torch.Tensor[batch_size, *state_shape]): Batch of states.
            backward_action (torch.Tensor[batch_size]): Batch of backward actions. Each element corresponds to the index of the action.
        Returns:
            new_state (torch.Tensor[batch_size, *state_shape]): Batch of new states.
            done (torch.Tensor[batch_size]): Whether the trajectory is done or not.
        """
        NotImplementedError

    @abstractmethod
    def get_forward_mask(self, states: torch.Tensor):
        """Get the forward actions mask for a batch of states.
        Args:
            states (torch.Tensor[batch_size, *state_shape]): Batch of states.
        Returns:
            actions_mask (torch.Tensor[batch_size, n_actions]): Forward actions mask.
        """
        NotImplementedError

    @abstractmethod
    def get_backward_mask(self, states: torch.Tensor):
        """Get the backward actions mask of a batch of states.
        Args:
            states (torch.Tensor[batch_size, *state_shape]): Batch of states.
        Returns:
            actions_mask (torch.Tensor[batch_size, n_actions]): Backward actions mask.
        """
        NotImplementedError

    @abstractmethod
    def compute_logreward(self, states: torch.Tensor):
        """Compute the logreward for a batch of states.
        Args:
            states (torch.Tensor[batch_size, *state_shape]): Batch of states.
        """
        NotImplementedError

    def _make_action_strings(self, actions, dones):
        """First removes the exit action and then converts actions indices to strings.
        Args:
            actions: Tensor of action indices
            dones: Tensor of whether the trajectories are done or not.
        """
        # Convert token indices to strings
        dones = dones[:, :-1]  # The last step is always True

        action_strings = [
            "".join(
                [
                    self.actions[act_idx]
                    for idx, act_idx in enumerate(action)
                    if not dones[i, idx]
                ]
            ).replace(self.exit_action, "")
            for i, action in enumerate(actions)
        ]

        return action_strings

    def chunk_uniform(self, n_tokens_to_add: int, remove_old: bool = False):
        """Adds random bigrams to the action space, using current actions.
        If we're required to remove old tokens, we initiate the new library with atomic tokens
        instead of building from previous ones.
        """
        if remove_old: # This means that we're in chunking replacement mode
            non_exit_actions = copy(self.atomic_tokens)
        else:
            non_exit_actions = copy(self.actions)
        non_exit_actions.remove(self.exit_action)

        novel_tokens = set()
        while len(novel_tokens) < n_tokens_to_add:
            # Get a bigram.
            candidate_token = "".join(random.choices(non_exit_actions, k=2))
            if candidate_token not in non_exit_actions:
                novel_tokens.add(candidate_token)  # Removes duplicates.
                non_exit_actions.append(candidate_token)

        old_tokens = set(self.actions) - set(self.atomic_tokens)
        if remove_old:
            self.remove_from_vocab(list(old_tokens))
        self.add_to_vocab(list(novel_tokens))

    def chunk_bpe(
        self,
        actions: torch.Tensor,
        dones: torch.Tensor,
        n_tokens_to_add: int,
        remove_old: bool = False,
    ):
        """Find the most valuable subsequence of actions from the corpus.
        Args:
            actions (torch.Tensor[batch_size, traj_length]): Batch of sequence of actions.
            dones (torch.Tensor[batch_size, traj_length]): Batch of sequence of terminations.
            remove_old (bool): Removes older new tokens from the library.
        """
        action_strings = self._make_action_strings(actions, dones)

        # Apply BPE algorithm to the state_strings and get the most frequent token
        vocab_dict = {k: i for i, k in enumerate(self.actions)}
        tokenizer = Tokenizer(BPE(vocab_dict, [], unk_token="[UNK]"))
        # tokenizer.pre_tokenizer = Whitespace()
        # vocab size is number of current actions (removing exit), plus n.
        vocab_size = len(self.actions) - 1 + n_tokens_to_add
        trainer = BpeTrainer(vocab_size=vocab_size)
        tokenizer.train_from_iterator(action_strings, trainer=trainer)

        # Sorts the BPE vocab dict by occurance ascending, finds the most useful
        # novel token.
        old_tokens = set(self.actions) - set(self.atomic_tokens)
        novel_tokens = [
            action
            for action in tokenizer.get_vocab().keys()
            if action not in self.actions
        ]

        novel_tokens = set(
            novel_tokens[-n_tokens_to_add:]
        )  # This makes sure we don't more tokens that intended
        self.add_to_vocab(list(novel_tokens))
        if remove_old:
            self.remove_from_vocab(list(old_tokens))

    def chunk_wordpiece(
        self,
        actions: torch.Tensor,
        dones: torch.Tensor,
        n_tokens_to_add: int,
        remove_old: bool = False,
    ):
        """Find the most valuable subsequence of actions from the corpus.
        Args:
            actions (torch.Tensor[batch_size, traj_length]): Batch of sequence of actions.
            dones (torch.Tensor[batch_size, traj_length]): Batch of sequence of terminations.
        """
        action_strings = self._make_action_strings(actions, dones)

        # Apply WP algorithm to the state_strings and get the most frequent token
        vocab_dict = {k: i for i, k in enumerate(self.actions)}
        tokenizer = Tokenizer(WordPiece(vocab=vocab_dict, unk_token="[UNK]"))

        # vocab size is number of current actions (removing exit), plus n.
        vocab_size = len(self.actions) - 1 + n_tokens_to_add
        trainer = WordPieceTrainer(
            vocab_size=vocab_size,
            continuing_subword_prefix="",  # No prefix allowed.
        )
        tokenizer.train_from_iterator(action_strings, trainer=trainer)

        old_tokens = set(self.actions) - set(self.atomic_tokens)
        novel_tokens = [
            action
            for action in tokenizer.get_vocab().keys()
            if action not in self.actions
        ]

        novel_tokens = set(
            novel_tokens[-n_tokens_to_add:]
        )  # This makes sure we don't more tokens that intended
        self.add_to_vocab(list(novel_tokens))
        if remove_old:
            self.remove_from_vocab(list(old_tokens))

    def add_to_vocab(self, tokens: list[str]):
        assert all([x not in self.actions for x in tokens])
        _actions = copy(self.actions)
        self.actions = _actions + tokens
        self.action_len = torch.cat(
            [self.action_len, torch.tensor([len(x) for x in tokens])], dim=0
        ).long()
        # Reset the action frequency.
        self.action_frequency = torch.cat(
            [self.action_frequency * 0, torch.zeros(len(tokens))], dim=0
        )

    def remove_from_vocab(self, tokens: list[str]):
        assert all([x in self.actions for x in tokens])
        for token in tokens:
            idx = self.actions.index(token)
            _actions = copy(self.actions)
            self.actions = _actions[:idx] + _actions[idx + 1 :]
            self.action_len = torch.cat(
                [self.action_len[:idx], self.action_len[idx + 1 :]], dim=0
            ).long()
            self.action_frequency = torch.cat(
                [self.action_frequency[:idx], self.action_frequency[idx + 1 :]], dim=0
            )
