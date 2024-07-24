import gzip
import os
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import marisa_trie
import numpy as np
import torch
from Bio import SeqIO
from polyleven import levenshtein

from chunkgfn.environment.base_sequence import BaseSequenceModule


def find_closest_match(query, trie, max_cost=2):
    # Start by finding candidates that share at least a part of the prefix
    # We generate prefixes of the query to a certain reasonable length
    candidates = set()
    for i in range(1, min(len(query), max_cost + 1) + 1):
        prefix = query[:i]
        for word in trie.keys(prefix):
            candidates.add(word)

    # If no candidates are found directly, consider the entire dictionary
    if not candidates:
        candidates = set(trie.keys())

    # Compute edit distances between the query and each candidate
    closest_match = None
    min_distance = float("inf")
    for candidate in candidates:
        distance = levenshtein(query, candidate)
        if distance < min_distance:
            min_distance = distance
            closest_match = candidate

    return closest_match, min_distance


# def distance_to_closest_exemplar(dataset: list, examples: list, failure: int, cutoff=1e-6):
#     """
#     Fast Similarity Search in Large Dictionaries (Thomas Bocek, Ela Hunt,
#     Burkhard Stiller), 2007.
#     """
#     scores = []
#     for example in examples:
#         candidates = get_close_matches(example, dataset, cutoff=cutoff)
#         if len(candidates) == 0:
#             scores.append(failure)
#         else:
#             distances = [levenshtein(example, c) for c in candidates]
#             scores.append(min(distances))

#     return scores


def open_fasta(filename):
    with gzip.open(filename, "rt") as file:
        # Using SeqIO from Biopython to parse the FASTA formatted data
        dataset = {}
        for record in SeqIO.parse(file, "fasta"):
            dataset[str(record.seq)] = record.id

        return dataset


def default_data_path():
    return os.path.join(
        str(Path(os.path.abspath(__file__)).parent.parent.parent),
        "data/ccds",
    )


class CCDSSequenceModule(BaseSequenceModule):
    """Naive Protein Coding Sequence building environment that constructs human
    protein coding regions from RNA one nucleotide at a time, with tokens extracted
    from the CCDS dataset. Uses the protein coding sequences as the environment modes
    and uses edit distance to the closest CDS a reward.
    """

    def __init__(
        self,
        max_len: int,
        num_train_iterations: int,
        task: str,
        num_modes: int = 100,
        n_held_out: int = 10,
        batch_size: int = 64,
        sample_exact_length: bool = False,
        threshold: float = 0.9,
        num_workers: int = 0,
        pin_memory: bool = False,
        max_length: int = None,
        eps: float = 1e-12,
        **kwargs,
    ) -> None:
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        assert task in ["dna", "rna", "protein", "protein_exons"]

        # Set the tokens and the sequences for the task (RNA vs Proteins).
        dpath = default_data_path()
        if task in ["rna", "dna"]:
            self.dset = open_fasta(
                os.path.join(dpath, "CCDS_nucleotide.current.fna.gz"),
            )
        else:
            if task == "protein_exons":
                self.dset = open_fasta(
                    os.path.join(dpath, "CCDS_protein_exons.current.faa.gz"),
                )
            elif task == "proteins":
                self.dset = open_fasta(
                    os.path.join(dpath, "CCDS_protein.current.faa.gz"),
                )

        self.eps = eps
        self.len_modes = 0
        self.modes = []
        self.num_modes = num_modes
        self.n_held_out = n_held_out
        self.threshold = threshold

        # All unique tokens in this dataset.
        atomic_tokens_npy = np.unique([*"".join(list(self.dset.keys()))])
        atomic_tokens = [str(x) for x in atomic_tokens_npy]

        # "None" performs no truncation.
        if max_len is None:
            self.examples = list(self.dset.keys())
        else:
            self.examples = [x[:max_len] for x in list(self.dset.keys())]
        self.max_example_len = max([len(x) for x in self.examples])

        # Trie is used for fast distance search.
        self.trie = marisa_trie.Trie(self.examples)

        # Create the training modes and the "known" (held out) sequences.
        self.modes = self.examples[: self.num_modes]
        self.known_sequences = self.examples[-self.n_held_out :]
        self.len_modes = torch.tensor([len(m) for m in self.modes])

        super().__init__(
            atomic_tokens=atomic_tokens,
            max_len=max(self.len_modes) if max_len is None else max_len,
            num_train_iterations=num_train_iterations,
            batch_size=batch_size,
            sample_exact_length=sample_exact_length,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **kwargs,
        )

    def normalized_edit_distance(self, strings: list):
        scores, seqs, lengths = [], [], []
        for string in strings:
            matching_seq, score = find_closest_match(string, self.trie, max_cost=1000)
            scores.append(score)
            seqs.append(matching_seq)
            lengths.append(len(matching_seq))

        return torch.tensor(scores), seqs, torch.tensor(lengths)

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

        scores, _, lengths = self.normalized_edit_distance(strings)

        return 1 - scores / lengths

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

        scores, seqs, lengths = self.normalized_edit_distance(strings)
        reward = 1 - scores / lengths

        # Find the modes that are close to the samples, and save them.
        modes_found = []
        for seq, r in zip(seqs, reward):
            if r > self.threshold:
                modes_found.append(seq)

        modes_found = set(modes_found)
        self.discovered_modes.update(modes_found)

        metrics = {
            "num_modes": float(len(self.discovered_modes)),
            "num_visited": float(len(self.visited)),
        }

        return metrics

    def build_test(self):
        """Build the test set based on known valid molecules.
        returns:
            test_seq (list[str]): List of test sequences.
            test_rs (list[float]): List of test logrewards.
        """
        test_seq = []

        for string in self.known_sequences:
            s_idx = torch.tensor(
                [self.atomic_tokens.index(char) for char in string]
                + [self.atomic_tokens.index("<EOS>")]
            )
            s_tensor = torch.zeros(s_idx.shape[0], len(self.atomic_tokens))
            s_tensor[torch.arange(s_idx.shape[0]), s_idx] = 1
            seq = torch.full(
                (self.max_len + 1, s_tensor.shape[1]), -1, dtype=torch.float
            )
            seq[: len(s_tensor)] = s_tensor
            test_seq.append(seq)

        test_seq = torch.stack(test_seq, dim=0)
        test_rs = self.compute_logreward(test_seq)

        return test_seq, test_rs


if __name__ == "__main__":
    from datetime import datetime

    truncate = [None, 1000, 100]
    for t in truncate:
        dset = CCDSSequenceModule(
            max_len=t,
            num_train_iterations=100,
            task="dna",
            batch_size=64,
            sample_exact_length=False if t is None else True,
            num_workers=0,
            pin_memory=False,
            eps=1e-12,
        )
        trie = marisa_trie.Trie(dset.examples[1:])

        for example_idx in [0, -1]:
            now = datetime.now()
            score = find_closest_match(dset.examples[example_idx], trie, max_cost=1000)[
                1
            ]
            end = datetime.now()
            time_taken = end - now
            print(
                "truncate={}, time={}, idx={}, score={}".format(
                    t, time_taken, example_idx, score
                )
            )
