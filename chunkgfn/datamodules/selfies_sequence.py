import random
import string
from typing import Tuple

import numpy as np
import selfies as sf
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem.QED import qed
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

from .base_sequence import BaseSequenceModule

RDLogger.DisableLog("rdApp.*")

# fmt: off
ALPHABET = [
    '[#N]', '[N+1]', '[Br]', '[NH2+1]', '[NH1]',
    '[/C]', '[O-1]', '[=Ring1]', '[P]', '[NH1+1]',
    '[#C]', '[Cl]', '[#Branch2]', '[F]', '[=Branch2]',
    '[C@H1]', '[C@@H1]', '[#Branch1]', '[S]', '[=N]',
    '[Ring2]', '[Branch2]', '[O]', '[=O]', '[N]',
    '[=Branch1]', '[Branch1]', '[Ring1]', '[=C]', '[C]'
]
# fmt: on


class SELFIESSequenceModule(BaseSequenceModule):
    """SELFIES building environment. Uses QED (drug-likeness) score as a reward."""

    def __init__(
        self,
        max_len: int,
        num_train_iterations: int,
        batch_size: int = 64,
        sample_exact_length: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        scaffold_thresholds: Tuple = (0.6, 0.7, 0.8, 0.9),
        **kwargs,
    ) -> None:
        self.save_hyperparameters(logger=False)
        atomic_tokens = list(string.ascii_letters)[: len(ALPHABET)]

        self.tokens2symbols = {tok: sym for tok, sym in zip(atomic_tokens, ALPHABET)}

        self.min_threshold = min(scaffold_thresholds)
        self.discovered_scaffolds = {k: set() for k in scaffold_thresholds}
        self.modes = []
        self.len_modes = 0

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

    def _string_to_selfie(self, s: str) -> str:
        return "".join(self.tokens2symbols[t] for t in s)

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

        qeds = []
        for s in strings:
            selfie = self._string_to_selfie(s)
            smiles = sf.decoder(selfie)
            mol = Chem.MolFromSmiles(smiles)
            qeds.append(qed(mol))
        return torch.tensor(qeds).clip(min=1e-10).log()

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

        qeds = []
        for s in strings:
            selfie = self._string_to_selfie(s)
            smiles = sf.decoder(selfie)
            mol = Chem.MolFromSmiles(smiles)
            q = qed(mol)
            qeds.append(q)

            if q >= self.min_threshold:
                scaffold = MurckoScaffoldSmiles(smiles)
                for th in self.discovered_scaffolds.keys():
                    if q >= th:
                        self.discovered_scaffolds[th].add(scaffold)

        metrics = {
            "avg_qed": np.mean(qeds),
            "min_qed": np.min(qeds),
            "max_qed": np.max(qeds),
        }
        for th, scaffolds in self.discovered_scaffolds.items():
            metrics[f"scaffolds_above_qed_{th}"] = len(scaffolds)

        return metrics

    def build_test(self):
        """Build the test set based on known valid molecules.
        returns:
            test_seq (list[str]): List of test sequences.
            test_rs (list[float]): List of test logrewards.
        """
        test_seq = []

        strings = [
            "".join(
                random.choices(
                    self.atomic_tokens[1:], k=np.random.randint(5, self.max_len)
                )
            )
            for _ in range(1000)
        ]

        for s in strings:
            s_idx = torch.tensor(
                [self.atomic_tokens.index(char) for char in s]
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
