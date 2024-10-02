import os
from pathlib import Path

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
        num_train_iterations: int,
        threshold: float,
        batch_size: int = 64,
        atomic_tokens: list[str] = ["0", "1"],
        sample_exact_length: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ) -> None:
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        atomic_tokens = [
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

        self.threshold = threshold
        self.modes_path = os.path.join(
            Path(__file__).parent.parent.parent, f"modes_{self.max_len}.txt"
        )
        self.dataset_path = os.path.join(
            Path(__file__).parent.parent.parent, f"modes_{self.max_len}_dataset.txt"
        )
        self.chunks = ["00000000", "11111111", "11110000", "00001111", "00111100"]

        self.create_modes()

    def create_modes(self):
        """Create the modes for the bit-sequence task depending on the oracle difficulty.
        If the difficulty is "medium" then the modes all have the same length `max_len`
        and are unique.
        If the difficulty is "hard" then the modes have different lengths with
        maximum length of `max_len` and minimum length of `max_len//2` and are unique.
        """
        with open(self.modes_path, "r") as f:
            self.modes = f.read().splitlines()
        self.modes = list(self.modes)
        self.len_modes = torch.tensor([len(m) for m in self.modes])

    def shortest_parse(self, vocabulary, string):
        """Compute the shortest parse of a given string using the vocabulary tokens.

        This function finds the minimum number of tokens required to completely parse
        the input `string` using tokens from the vocabulary. It returns
        both the minimum number of tokens and the sequence of tokens that achieves
        this minimum.

        Args:
            string (str): The input string to be parsed.

        Returns:
            tuple:
                - int: The minimum number of tokens required to parse the input string.
                - list of str: The sequence of tokens that forms the shortest parse of the input string.
        """
        min_parses = {string[:i]: float("inf") for i in range(len(string) + 1)}
        min_parses[""] = 0
        best_tokens = {"": []}
        for ln in range(1, len(string) + 1):
            s = string[:ln]
            candidates = []
            for token in vocabulary:
                c = len(token)
                if s[-c:] == token:
                    candidates.append((token, s[:-c]))

            best_token, best_candidate = min(candidates, key=lambda x: min_parses[x[1]])
            best_tokens[s] = best_tokens[s[: -len(best_token)]] + [best_token]
            min_parses[s] = 1 + min_parses[best_candidate]
        return min_parses[string], best_tokens[string]

    def compute_logreward(self, states: torch.Tensor) -> torch.Tensor:
        """Compute the reward for the given states and action.
        Args:
            states (torch.Tensor[batch_size, max_len, dim]): Batch of states.
        Returns:
            reward (torch.Tensor[batch_size]): Batch of rewards.
        """
        rewards = []
        parse_vocabulary = self.chunks + [
            a for a in self.atomic_tokens if a != self.exit_action
        ]
        max_chunks_number = self.max_len // min([len(chunk) for chunk in self.chunks])
        for s in self.to_strings(states):
            string = s.replace(
                self.exit_action, ""
            )  # remove <EOS> tokens for computing reward
            _, best_tokens = self.shortest_parse(parse_vocabulary, string)
            reward = sum([(token in self.chunks) for token in best_tokens])
            reward /= max_chunks_number
            rewards.append(reward)

        rewards = torch.tensor(rewards)
        rewards = rewards.clamp_min(min=1e-4)
        logrewards = rewards.log()
        return logrewards

    def compute_metrics(
        self, states: torch.Tensor, logrewards: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute metrics for the given states.
        Args:
            states (torch.Tensor[batch_size, max_len, dim]): Batch of states.
        Returns:
            metrics (dict[str, torch.Tensor]): Dictionary of metrics.
        """
        strings = [
            s.replace(self.exit_action, "") for s in self.to_strings(states)
        ]  # remove <EOS> tokens
        self.visited.update(set(strings))
        dists = torch.tensor([[levenshtein(s, i) for i in self.modes] for s in strings])
        values, indices = torch.min(dists, dim=-1)
        mode_indices = indices[
            values <= self.threshold
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
        with open(self.dataset_path, "r") as f:
            dataset = f.read().splitlines()

        test_seq = []
        for string in dataset:
            s_idx = torch.tensor(
                [self.atomic_tokens.index(char) for char in string]
                + [self.atomic_tokens.index(self.exit_action)]
            )
            s_tensor = torch.zeros(s_idx.shape[0], len(self.atomic_tokens))
            s_tensor[torch.arange(s_idx.shape[0]), s_idx] = 1
            test_seq.append(s_tensor)
        test_seq = torch.stack(test_seq, dim=0)
        test_rs = self.compute_logreward(test_seq)
        return test_seq, test_rs
