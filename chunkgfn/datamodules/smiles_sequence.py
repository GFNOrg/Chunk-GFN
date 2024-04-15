import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem.QED import qed

from .base_sequence import BaseSequenceModule

RDLogger.DisableLog("rdApp.*")


class SMILESSequenceModule(BaseSequenceModule):
    """Naive SMILES building environment that constructs SMILES' one letter at a time,
    with tokens extracted from ZINC-250k dataset. Does not enforce validity in any way,
    and uses QED (drug-likeness) score as a reward if the SMILES is valid, near-zero
    otherwise.
    """

    def __init__(
        self,
        max_len: int,
        num_train_iterations: int,
        batch_size: int = 64,
        sample_exact_length: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        simple: bool = True,
        eps: float = 1e-12,
        **kwargs,
    ) -> None:
        if simple:
            # fmt: off
            atomic_tokens = [
                "<EOS>", "(", ")", "1", "2", "3", "=", "C", "N", "O"
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

        self.eps = eps
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

        rewards = []
        for s in strings:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                rewards.append(self.eps)
            else:
                rewards.append(qed(mol))
        return torch.tensor(rewards)

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

        valid_rewards = []
        for s in strings:
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                valid_rewards.append(qed(mol))

        metrics = {
            "num_valid": float(len(valid_rewards)),
            "avg_valid_reward": np.mean(valid_rewards),
        }

        return metrics

    def build_test(self):
        """Build the test set based on known valid molecules.
        returns:
            test_seq (list[str]): List of test sequences.
            test_rs (list[float]): List of test logrewards.
        """
        test_seq = []

        known_molecules = [
            'CC(=O)OC1=CC=CC=C1C(=O)O',  # aspirin
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # ibuprofen
            'CN1C2CCC1C(C(C2)OC(=O)C3=CC=CC=C3)C(=O)OC',  # cocaine
            'C1=CC=C(C=C1)C=O',  # benzaldehyde
            'CCCC1=CC=C(C=C1)C=O',  # 4-propylbenzaldehyde
            'CC(=O)C'  # acetone
        ]
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
