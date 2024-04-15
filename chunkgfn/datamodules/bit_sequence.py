import random

import numpy as np
import torch
from polyleven import levenshtein

from .base_sequence import BaseSequenceModule


class BitSequenceModule(BaseSequenceModule):
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
        atomic_tokens = [
            "<EOS>",
            "0",
            "1",
        ]

        super().__init__(
            atomic_tokens=atomic_tokens,
            max_len=max_len,
            num_train_iterations=num_train_iterations,
            batch_size=batch_size,
            sample_exact_length=sample_exact_length,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **kwargs,
        )

        self.oracle_difficulty = oracle_difficulty
        self.num_modes = num_modes
        self.threshold = threshold

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

    def compute_logreward(self, states: torch.Tensor) -> torch.Tensor:
        """Compute the reward for the given states and action.
        Args:
            states (torch.Tensor[batch_size, max_len, dim]): Batch of states.
        Returns:
            reward (torch.Tensor[batch_size]): Batch of rewards.
        """
        strings = [
            s.replace("<EOS>", "") for s in self.to_strings(states)
        ]  # remove <EOS> tokens for computing reward

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
            s.replace("<EOS>", "") for s in self.to_strings(states)
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
