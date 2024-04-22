from pathlib import Path

import numpy as np
import pandas as pd
import torch
from polyleven import levenshtein
from rdkit import Chem, RDLogger
from rdkit.Chem.QED import qed

from .base_sequence import BaseSequenceModule

RDLogger.DisableLog("rdApp.*")

ZINC_PATH = Path(__file__).parent / '..' / '..' / 'data' / 'ZINC-250k.csv'


class SMILESSequenceModule(BaseSequenceModule):
    """Naive SMILES building environment that constructs SMILES' one letter at a time,
    with tokens extracted from ZINC-250k dataset. Does not enforce validity in any way,
    and uses edit distance from the modes from ZINC-250k (the ones with compatible length).
    """

    def __init__(
        self,
        max_len: int,
        num_modes: int,
        num_train_iterations: int,
        batch_size: int = 64,
        sample_exact_length: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        simple: bool = True,
        **kwargs,
    ) -> None:
        if simple:
            # fmt: off
            atomic_tokens = [
                "<EOS>", "(", ")", "1", "2", "=", "B", "C", "F", "H",
                "N", "O", "S", "[", "]", "c", "l", "n", "r", "s"
            ]
            # fmt: on
        else:
            # fmt: off
            atomic_tokens = [
                "<EOS>", "#", "(", ")", "+", "-", "/", "1", "2", "3",
                "4", "5", "6", "7", "8", "=", "@", "B", "C", "F", "H",
                "I", "N", "O", "P", "S", "[", "\\", "]", "c", "l", "n",
                "o", "r", "s"
            ]
            # fmt: on

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

        self.num_modes = num_modes
        self.simple = simple

        self.create_modes()

    def create_modes(self):
        df = pd.read_csv(ZINC_PATH)
        df['smiles'] = df['smiles'].apply(lambda x: x.strip())
        df = df[df['smiles'].apply(len) <= self.max_len]

        if self.simple:
            df = df[df['smiles'].apply(lambda x: all(c in self.atomic_tokens for c in x))]

        self.modes = list(df['smiles'])

        if self.num_modes != -1 and len(df) > self.num_modes:
            self.modes = np.random.choice(self.modes, self.num_modes, replace=False)

        assert len(self.modes) > 0

        print(f"Found {len(self.modes)} modes.")

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
        modes_found = set([s for s in strings if s in self.modes])
        self.discovered_modes.update(modes_found)

        valid_qeds = []
        for s in strings:
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                valid_qeds.append(qed(mol))

        metrics = {
            "num_modes": float(len(self.discovered_modes)),
            "num_visited": float(len(self.visited)),
            "num_valid": float(len(valid_qeds)),
            "avg_valid_qed": np.mean(valid_qeds),
        }

        return metrics

    def build_test(self):
        test_seq = []

        known_molecules = self.modes
        known_molecules = [m for m in known_molecules if len(m) < self.max_len]

        for string in known_molecules:
            s_idx = torch.tensor(
                [self.atomic_tokens.index(char) for char in string]
                + [self.atomic_tokens.index("<EOS>")]
            )
            s_tensor = torch.zeros(s_idx.shape[0], len(self.atomic_tokens))
            s_tensor[torch.arange(s_idx.shape[0]), s_idx] = 1
            seq = torch.full((self.max_len + 1, s_tensor.shape[1]), -1, dtype=torch.float)
            seq[:len(s_tensor)] = s_tensor
            test_seq.append(seq)
        test_seq = torch.stack(test_seq, dim=0)
        test_rs = self.compute_logreward(test_seq)

        return test_seq, test_rs
