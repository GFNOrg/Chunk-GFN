import RNA
import torch

from .base_sequence import BaseSequenceModule


def registry():
    """
    Return a dictionary of problems of the form:
    `{
        "problem name": {
            "params": ...,
            "starts": ...
        },
        ...
    }`

    Returns:
        dict: Problems in the registry.

    """
    # RNA target sequences
    targets = [
        "GAACGAGGCACAUUCCGGCUCGCCCGGCCCAUGUGAGCAUGGGCCGGACCCCGUCCGCGCGGGGCCCCCGCGCGGACGGGGGCGAGCCGGAAUGUGCCUC",  # noqa: E501
        "GAGGCACAUUCCGGCUCGCCCCCGUCCGCGCGGGGGCCCCGCGCGGACGGGGUCCGGCCCGCGCGGGGCCCCCGCGCGGGAGCCGGAAUGUGCCUCGUUC",  # noqa: E501
        "CCGGUGAUACUGUUAGUGGUCACGGUGCAUUUAUAGCGCUAAAGUACAGUCUUCCCCUGUUGAACGGCGCCAUUGCAUACAGGGCCAGCCGCGUAACGCC",  # noqa: E501
        "UAAGAGAGCGUAAAAAUAGAGAUAUGUUCUUGGGUCAGGGCUAUGCGUACCCCAUGAGAGUAAAUCAUACCCCCAAUGGGCUUCGGCGGAAAUUCACUUA",  # noqa: E501
    ]

    # Starting sequences of lengths 14, 50, and 100
    starts = {
        14: {
            1: "AUGGGCCGGACCCC",
            2: "GCCCCGCCGGAAUG",
            3: "UCUUGGGGACUUUU",
            4: "GGAUAACAAUUCAU",
            5: "CCCAUGCGCGAUCA",
        },
        50: {
            1: "GAACGAGGCACAUUCCGGCUCGCCCGGCCCAUGUGAGCAUGGGCCGGACC",
            2: "CCGUCCGCGCGGGGCCCCCGCGCGGACGGGGGCGAGCCGGAAUGUGCCUC",
            3: "AUGUUUCUUUUAUUUAUCUGAGCAUGGGCGGGGCAUUUGCCCAUGCAAUU",
            4: "UAAACGAUGCUUUUGCGCCUGCAUGUGGGUUAGCCGAGUAUCAUGGCAAU",
            5: "AGGGAAGAUUAGAUUACUCUUAUAUGACGUAGGAGAGAGUGCGGUUAAGA",
        },
        100: {
            1: "GAACGAGGCACAUUCCGGCUCGCCCGGCCCAUGUGAGCAUGGGCCGGACCCCGUCCGCGCGGGGCCCCCGCGCGGACGGGGGCGAGCCGGAAUGUGCCUC",  # noqa: E501
            2: "AGCAUCUCGCCGUGGGGGCGGGCCCGGCCCAUGUGAGCAUGCGUAGGUUUAUCCCAUAGAGGACCCCGGGAGAACUGUCCAAUUGGCUCCUAGCCCACGC",  # noqa: E501
            3: "GGCGGAUACUAGACCCUAUUGGCCCGGCCCAUGUGAGCAUGGCCCCAGAUCUUCCGCUCACUCGCAUAUUCCCUCCGGUUAAGUUGCCGUUUAUGAAGAU",  # noqa: E501
            4: "UUGCAGGUCCCUACACCUCCGGCCCGGCCCAUGUGACCAUGAAUAGUCCACAUAAAAACCGUGAUGGCCAGUGCAGUUGAUUCCGUGCUCUGUACCCUUU",  # noqa: E501
            5: "UGGCGAUGAGCCGAGCCGCCAUCGGACCAUGUGCAAUGUAGCCGUUCGUAGCCAUUAGGUGAUACCACAGAGUCUUAUGCGGUUUCACGUUGAGAUUGCA",  # noqa: E501
        },
    }

    problems = {}

    # Single target problems - 4 of these
    for t in range(len(targets)):
        for length, start in starts.items():
            name = f"L{length}_RNA{t+1}"
            problems[name] = {
                "params": {"targets": [targets[t]], "seq_length": length},
                "starts": start,
            }

    # Two-target problems
    for t1 in range(len(targets)):
        for t2 in range(t1 + 1, len(targets)):
            for length, start in starts.items():
                name = f"L{length}_RNA{t1+1}+{t2+1}"
                problems[name] = {
                    "params": {
                        "targets": [targets[t1], targets[t2]],
                        "seq_length": length,
                    },
                    "starts": start,
                }

    # Two-target problems with conserved portion
    for t1 in range(len(targets)):
        for t2 in range(t1 + 1, len(targets)):
            name = f"C20_L100_RNA{t1+1}+{t2+1}"
            problems[name] = {
                "params": {
                    "targets": [targets[t1], targets[t2]],
                    "seq_length": 100,
                    "conserved_region": {
                        "start": 21,
                        "pattern": "GCCCGGCCCAUGUGAGCAUG",
                    },
                },
                "starts": starts[100],
            }

    return problems


class RNABindingModule(BaseSequenceModule):
    def __init__(
        self,
        num_train_iterations: int,
        batch_size: int = 64,
        task: str = "L14_RNA1",
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ) -> None:
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        atomic_tokens = ["A", "C", "G", "U"]

        self.task = task
        self.modes = []
        self.len_modes = 0

        params = registry()[task]["params"]
        self.starts = registry()[task]["starts"]
        self.targets = params["targets"]
        self.conserved_region = (
            params["conserved_region"] if "conserved_region" in params else None
        )

        super().__init__(
            atomic_tokens=atomic_tokens,
            max_len=params["seq_length"],
            num_train_iterations=num_train_iterations,
            batch_size=batch_size,
            sample_exact_length=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **kwargs,
        )
        self.norm_values = self.compute_min_binding_energies()

    def compute_min_binding_energies(self):
        """Compute the lowest possible binding energy for each target."""
        complements = {"A": "U", "C": "G", "G": "C", "U": "A"}

        min_energies = []
        for target in self.targets:
            complement = "".join(complements[x] for x in target)[::-1]
            energy = RNA.duplexfold(complement, target).energy
            min_energies.append(energy * self.max_len / len(target))

        return torch.tensor(min_energies)

    def compute_logreward(self, states: torch.Tensor) -> torch.Tensor:
        """Compute the logreward for the given states and action.
        Args:
            states (torch.Tensor[batch_size, max_len, dim]): Batch of states.
        Returns:
            logreward (torch.Tensor[batch_size]): Batch of log rewards.
        """
        sequences = [
            s.replace("<EOS>", "") for s in self.to_strings(states)
        ]  # remove <EOS> tokens for computing reward

        logrewards = []
        for seq in sequences:
            if len(seq) != self.max_len:
                raise ValueError(
                    f"All sequences in `sequences` must be of length {self.max_len}"
                )

            # If `self.conserved_region` is not None, check that the region is conserved
            if self.conserved_region is not None:
                start = self.conserved_region["start"]
                pattern = self.conserved_region["pattern"]

                # If region not conserved, fitness is 0
                if seq[start : start + len(pattern)] != pattern:
                    logrewards.append(0)
                    continue

            # Energy is sum of binding energies across all targets
            energies = torch.tensor(
                [RNA.duplexfold(target, seq).energy for target in self.targets]
            )
            fitness = (energies / self.norm_values).mean().clip(min=1e-10).log()

            logrewards.append(fitness)

        return torch.tensor(logrewards)

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

        metrics = {
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

        starting_sequences = [self.starts[i] for i in self.starts]

        for seq in starting_sequences:
            s_idx = torch.tensor(
                [self.atomic_tokens.index(char) for char in seq]
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
