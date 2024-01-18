from typing import Any, Dict, Optional

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from chunkgfn.datasets.text_chunk import ChunkDataset


class ChunkModule(LightningDataModule):
    """`LightningDataModule` for the MNIST dataset.

    The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.
    It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a
    fixed-size image. The original black and white images from NIST were size normalized to fit in a 20x20 pixel box
    while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing
    technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of
    mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        dataset: Dict[str, Any],
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `MNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[ChunkDataset] = None
        self.data_val: Optional[ChunkDataset] = None
        self.data_test: Optional[ChunkDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = ChunkDataset(**self.hparams.dataset)

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

    def batch_token2vocab(self, states: torch.Tensor):
        """Convert batch of token indices to list of strings.
        Args:
            states (torch.Tensor[batch_size, max_len, state_vocab_dim]): Batch of token indices.
        """
        strings = []
        for state in states:
            # Cut the state before it arrives at [-1,-1,...]
            nonzero = (state == -1).nonzero()
            if len(nonzero) > 0:
                state = state[: nonzero[0][0]]

            state = torch.argmax(state, dim=-1).tolist()
            # Convert token indices to strings
            strings.append(
                "".join(
                    [
                        self.data_train.token2vocab[t]
                        for t in state
                        if t != self.data_train.eos_token_idx
                    ]
                )
            )
        return strings

    def compute_logreward(self, inputs: torch.Tensor, states: torch.Tensor):
        """Compute the logreward for a batch of sentences.
        Args:
            inputs (torch.Tensor[batch_size, max_len, input_vocab_dim]): Batch of inputs.
            states (torch.Tensor[batch_size, max_len, state_vocab_dim]): Batch of states.
        """
        # Convert token indices to strings
        state_strings = self.batch_token2vocab(states)
        input_strings = self.batch_token2vocab(inputs)
        # Compute reward
        logrewards = []
        for inp, state in zip(input_strings, state_strings):
            str_len = max(len(inp), len(state))
            logreward = 0
            for i in range(str_len):
                if i < len(inp) and i < len(state) and inp[i] == state[i]:
                    logreward += 0
                else:
                    logreward -= 1
            logrewards.append(logreward)
        logrewards = torch.tensor(logrewards)

        return logrewards
