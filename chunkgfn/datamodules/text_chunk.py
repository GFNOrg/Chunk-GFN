from typing import Any, Optional

import torch
from lightning import LightningDataModule
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from torch.utils.data import DataLoader

from chunkgfn.datasets.text_chunk import ChunkDataset, generate_sentence


class ChunkModule(LightningDataModule):
    """A `LightningDataModule` for the chunk dataset."""

    def __init__(
        self,
        batch_size: int = 64,
        max_len: int = 30,
        num_sentences: int = 1000,
        val_ratio: float = 0.2,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize the `ChunkModule`.
        Args:
            batch_size (int): The batch size. Defaults to 64.
            max_len (int): The maximum length of the sentences. Defaults to 30.
            num_sentences (int): The number of sentences to generate. Defaults to 1000.
            val_ratio (float): The ratio of the validation set. Defaults to 0.2.
            num_workers (int): The number of workers for the dataloaders. Defaults to 0.
            pin_memory (bool): Whether to pin memory for the dataloaders. Defaults to False.
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
            # Generate data
            data = [
                generate_sentence(self.hparams.max_len)
                for _ in range(self.hparams.num_sentences)
            ]
            # Split data
            n_val = int(self.hparams.val_ratio * len(data))
            self.data_train = ChunkDataset(data[:-n_val])
            self.data_val = ChunkDataset(data[-n_val:])

            self.s0 = -torch.ones(1 + self.hparams.max_len, 4)

    def is_initial_state(self, state: torch.Tensor) -> bool:
        """Check if the state is the initial state.
        Args:
            state (torch.Tensor[batch_size, max_len, dim]): Batch of states.
        Returns:
            is_initial (bool): Whether the state is the initial state or not.
        """
        return (state == self.s0.to(state)).all(dim=-1).all(dim=-1)

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
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

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
        eos_token = torch.zeros(dim).to(state)
        vocab_tensor = self.data_train.vocab_tensor.to(state.device)
        action_len = self.data_train.action_len.to(state.device)
        eos_token[self.data_train.eos_token_idx] = 1
        # Update the state by filling the current timestep with the sampled action only if it doesn't contain EOS token
        new_state = state.clone()
        start_indices = torch.argmax(((state == -1).all(dim=-1) + 0), dim=-1).to(
            state.device
        )  # Where to start inserting action

        done = torch.where((state == eos_token).all(dim=-1).any(dim=-1), True, False)

        for i in range(bs):
            if not done[i]:
                new_state[
                    i,
                    start_indices[i] : start_indices[i]
                    + int(action_len[forward_action[i]]),
                ] = vocab_tensor[
                    forward_action[i],
                    : int(action_len[forward_action[i]]),
                ]
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
        action_len = self.data_train.action_len.to(state.device)

        where_padding = (state == self.data_train.padding_token.to(state)).all(dim=-1)

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
        new_state[mask] = self.data_train.padding_token.to(state.device)

        return new_state, done

    def get_invalid_actions_mask(self, states: torch.Tensor):
        """Get the invalid actions mask for a batch of states.
        Args:
            states (torch.Tensor[batch_size, max_len, dim]): Batch of states.
        Returns:
            actions_mask (torch.Tensor[batch_size, n_actions]): Invalid actions mask.
        """
        # Convert token indices to strings
        state_strings = self.batch_token2vocab(states, inlcude_eos=False)
        len_tokens_to_go = self.hparams.max_len - torch.tensor(
            [len(s) for s in state_strings]
        )

        actions_mask = len_tokens_to_go.unsqueeze(
            1
        ) >= self.data_train.action_len.unsqueeze(0)
        actions_mask[..., self.data_train.eos_token_idx] = 1
        actions_mask = actions_mask.to(states.device)
        return actions_mask

    def batch_token2vocab(self, states: torch.Tensor, inlcude_eos: bool = False):
        """Convert batch of token indices to list of strings.
        Args:
            states (torch.Tensor[batch_size, max_len, state_vocab_dim]): Batch of token indices.
            inlcude_eos (bool): Whether to include the end-of-string token in the output.
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
                        if t != self.data_train.eos_token_idx or inlcude_eos
                    ]
                )
            )
        return strings

    def batch_token2vocablist(self, states: torch.Tensor):
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
            strings.append([self.data_train.token2vocab[t] for t in state])
        return strings

    def compute_logreward(self, inputs: torch.Tensor, states: torch.Tensor):
        """Compute the logreward for a batch of sentences.
        Args:
            inputs (torch.Tensor[batch_size, max_len, input_vocab_dim]): Batch of inputs.
            states (torch.Tensor[batch_size, max_len, state_vocab_dim]): Batch of states.
        """
        # Convert token indices to strings
        state_strings = self.batch_token2vocab(states, inlcude_eos=False)
        input_strings = self.batch_token2vocab(inputs, inlcude_eos=False)
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

    def compute_accuracy(self, inputs: torch.Tensor, states: torch.Tensor):
        """Compute the accuracy for a batch of sentences.
        Args:
            inputs (torch.Tensor[batch_size, max_len, input_vocab_dim]): Batch of inputs.
            states (torch.Tensor[batch_size, max_len, state_vocab_dim]): Batch of states.
        """
        # Convert token indices to strings
        state_strings = self.batch_token2vocab(states, inlcude_eos=False)
        input_strings = self.batch_token2vocab(inputs, inlcude_eos=False)
        # Compute reward
        accuracies = []
        for inp, state in zip(input_strings, state_strings):
            if inp == state:
                accuracies.append(1.0)
            else:
                accuracies.append(0.0)
        accuracies = torch.tensor(accuracies).to(states.device)

        return accuracies

    def chunk(self, final_states: torch.Tensor):
        """Find the most valuable token from the corpus.
        Args:
            final_states (torch.Tensor[batch_size, max_len, state_vocab_dim]): Batch of final states.
        """
        # Convert token indices to strings
        state_strings = self.batch_token2vocab(final_states, inlcude_eos=False)
        # Apply BPE algorithm to the state_strings and get the most frequent token
        tokenizer = Tokenizer(BPE(self.data_train.vocab, [], unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=self.data_train.vocab_size)
        tokenizer.train_from_iterator(state_strings, trainer=trainer)
        new_token = list(
            set(tokenizer.get_vocab().keys()).difference(
                set(self.data_train.vocab.keys())
            )
        )[0]
        self.data_train.add_to_vocab(new_token)

    def get_parent_actions(self, states: torch.Tensor):
        """Get the parent actions of a batch of states.
        Args:
            states (torch.Tensor[batch_size, max_len, state_vocab_dim]): Batch of states.
        """
        # Convert token indices to strings
        state_strings = self.batch_token2vocablist(states)
        # Get the parent actions
        parent_actions = torch.zeros(
            states.shape[0], len(self.data_train._vocab), dtype=torch.int64
        )
        for i, state in enumerate(state_strings):
            if len(state) > 0:
                last_token = state[-1]
                if last_token == "<EOS>":
                    parent_actions_ = ["<EOS>"]
                else:
                    parent_actions_ = set()
                    for j in range(len(state)):
                        parent_actions_.add("".join(state[-j - 1 :]))
                    parent_actions_ = list(
                        parent_actions_.intersection(set(self.data_train.vocab))
                    )
                parent_actions_ = [
                    self.data_train.vocab2token[a] for a in parent_actions_
                ]
                parent_actions[i, parent_actions_] = 1
            else:
                parent_actions[i] = 1
        return parent_actions
        return parent_actions
