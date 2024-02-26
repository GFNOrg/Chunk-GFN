from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class BaseEnvironmentModule(LightningDataModule, ABC):
    """A `BaseEnvironmentModule` for defining environment datamodules."""

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
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
    def get_invalid_actions_mask(self, states: torch.Tensor):
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

    @abstractmethod
    def chunk(self, final_states: torch.Tensor):
        """Find the most valuable token from the corpus.
        Args:
            final_states (torch.Tensor[batch_size, max_len, state_vocab_dim]): Batch of final states.
        """
        NotImplementedError

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
        num_val_iterations: int,
        num_test_iterations: int,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__(batch_size, num_workers, pin_memory)
        """Initialize the `BaseUnconditionalEnvironmentModule`.
        Args:
            num_train_iterations (int): The number of training iterations per epoch.
            num_val_iterations (int): The number of val iterations per epoch.
            num_test_iterations (int): The number of test iterations per epoch.
            batch_size (int): The batch size. Defaults to 64.
            num_workers (int): The number of workers for the dataloaders. Defaults to 0.
            pin_memory (bool): Whether to pin memory for the dataloaders. Defaults to False.
        
        """
        self.num_train_iterations = num_train_iterations
        self.num_val_iterations = num_val_iterations
        self.num_test_iterations = num_test_iterations

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
        self.data_val = DummyDataset(self.num_test_iterations * self.hparams.batch_size)
        self.data_test = DummyDataset(self.num_val_iterations * self.hparams.batch_size)
