from abc import ABC, abstractmethod
from typing import Any, Optional
from copy import copy
import random

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.models import BPE, WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import Tokenizer


class BaseEnvironmentModule(LightningDataModule, ABC):
    """A `BaseEnvironmentModule` for defining environment datamodules."""

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = True,
    ) -> None:
        """Initialize the `BaseEnvironmentModule`.
        Args:
            batch_size (int): The batch size. Defaults to 64.
            num_workers (int): The number of workers for the dataloaders. Defaults to 0.
            pin_memory (bool): Whether to pin memory for the dataloaders. Defaults to False.

        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.persistent_workers = persistent_workers
        self.exit_action = "<EOS>"

    @abstractmethod
    def preprocess_states(self, state: torch.Tensor) -> torch.Tensor:
        """Preprocess the state so that it can be input to the policy model.
        Args:
            state (torch.Tensor[batch_size, *state_shape]): The state.
        Returns:
            processed_state (torch.Tensor[batch_size, *processed_state_shape]): The preprocessed state.
        """
        NotImplementedError

    @abstractmethod
    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        NotImplementedError

    @abstractmethod
    def is_initial_state(self, state: torch.Tensor) -> bool:
        """Check if the state is the initial state.
        Args:
            state (torch.Tensor[batch_size, max_len, dim]): Batch of states.
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
            state (torch.Tensor[batch_size, max_len, dim]): Batch of states.
            forward_action (torch.Tensor[batch_size]): Batch of forward actions. Each element corresponds to the index of the action.
        Returns:
            new_state (torch.Tensor[batch_size, max_len, dim]): Batch of new states.
            done (torch.Tensor[batch_size]): Whether the trajectory is done or not.
        """
        NotImplementedError

    @abstractmethod
    def backward_step(self, state: torch.Tensor, backward_action: torch.Tensor):
        """Change the state after you apply the backward action.
        Args:
            state (torch.Tensor[batch_size, max_len, dim]): Batch of states.
            backward_action (torch.Tensor[batch_size]): Batch of backward actions. Each element corresponds to the index of the action.
        Returns:
            new_state (torch.Tensor[batch_size, max_len, dim]): Batch of new states.
        """
        NotImplementedError

    @abstractmethod
    def get_forward_mask(self, states: torch.Tensor):
        """Get the invalid actions mask for a batch of states.
        Args:
            states (torch.Tensor[batch_size, max_len, dim]): Batch of states.
        Returns:
            actions_mask (torch.Tensor[batch_size, n_actions]): Invalid actions mask.
        """
        NotImplementedError

    @abstractmethod
    def compute_logreward(self, inputs: torch.Tensor, states: torch.Tensor):
        """Compute the logreward for a batch of sentences.
        Args:
            inputs (torch.Tensor[batch_size, max_len, input_vocab_dim]): Batch of inputs.
            states (torch.Tensor[batch_size, max_len, state_vocab_dim]): Batch of states.
        """
        NotImplementedError

    def _make_action_strings(self, actions, dones):
        """First removes the exit action and then converts actions indicrs to strings.
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

    def chunk_uniform(self, n_tokens_to_add: int):
        """Adds random bigrams to the action space, using current actions."""
        non_exit_actions = copy(self.actions)
        non_exit_actions.remove(self.exit_action)

        novel_tokens = set()
        while len(novel_tokens) < n_tokens_to_add:
            # Get a bigram.
            candidate_token = "".join(random.choices(non_exit_actions, k=2))
            if candidate_token not in non_exit_actions:
                novel_tokens.add(candidate_token)  # Removes duplicates.

        self.add_to_vocab(list(novel_tokens))

    def chunk_bpe(self, actions: torch.Tensor, dones: torch.Tensor, n_tokens_to_add: int):
        """Find the most valuable subsequence of actions from the corpus.
        Args:
            actions (torch.Tensor[batch_size, traj_length]): Batch of sequence of actions.
            dones (torch.Tensor[batch_size, traj_length]): Batch of sequence of terminations.
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
        novel_tokens = set(tokenizer.get_vocab().keys()) - set(self.actions)
        self.add_to_vocab(list(novel_tokens))

    def chunk_wordpiece(self, actions: torch.Tensor, dones: torch.Tensor, n_tokens_to_add: int):
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

        # Sorts the WP vocab dict by occurance ascending, finds the most useful
        # novel token.
        novel_tokens = set(tokenizer.get_vocab().keys()) - set(self.actions)
        self.add_to_vocab(list(novel_tokens))

    def add_to_vocab(self, tokens: list):
        assert all([x not in self.actions for x in tokens])
        self.actions.extend(tokens)
        self.action_len = torch.cat(
            [self.action_len, torch.tensor([len(x) for x in tokens])], dim=0
        )
        # Reset the action frequency.
        self.action_frequency = torch.cat(
            [self.action_frequency * 0, torch.zeros(len(tokens))], dim=0
        )

    def remove_from_vocab(self, token: str):
        assert isinstance(token, str)
        if token in self.actions:
            idx = self.actions.index(token)
            self.actions.pop(idx)
            self.action_len = torch.cat(
                [self.action_len[:idx], self.action_len[idx + 1 :]], dim=0
            )
            self.action_frequency = torch.cat(
                [self.action_frequency[:idx], self.action_frequency[idx + 1 :]], dim=0
            )

    @abstractmethod
    def get_parent_actions(self, states: torch.Tensor):
        """Get the parent actions of a batch of states.
        Args:
            states (torch.Tensor[batch_size, max_len, state_vocab_dim]): Batch of states.
        """
        NotImplementedError


class BaseUnconditionalEnvironmentModule(BaseEnvironmentModule):
    """A `BaseUnconditionalEnvironmentModule` for defining unconditional environment datamodules.
    Since it's an environment meant for unconditional generation, it doesn't require any dataset, but we'll make a dummy one.
    """

    def __init__(
        self,
        num_train_iterations: int,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(batch_size, num_workers, pin_memory, **kwargs)
        """Initialize the `BaseUnconditionalEnvironmentModule`.
        Args:
            num_train_iterations (int): The number of training iterations per epoch.
            batch_size (int): The batch size. Defaults to 64.
            num_workers (int): The number of workers for the dataloaders. Defaults to 0.
            pin_memory (bool): Whether to pin memory for the dataloaders. Defaults to False.

        """
        self.num_train_iterations = num_train_iterations

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
