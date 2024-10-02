import random

import torch
import torch.nn.functional as F
from esm_reward.lm_design import Designer

from .base_sequence import BaseSequenceModule

_SUPPRESS_AAS = {"C"}


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ESMRewardModelWrapper(Designer):
    def __init__(
        self,
        seq_len: int,
        language_model_energy_term_weight,
        ngram_energy_term_weight,
        ngram_orders,
    ):
        torch.nn.Module.__init__(self)

        self.allowed_AA = "".join(
            AA for AA in self.standard_AA if AA not in _SUPPRESS_AAS
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_models()
        self._init_no_target(seq_len)

        self.language_model_energy_term_weight = language_model_energy_term_weight
        self.ngram_energy_term_weight = ngram_energy_term_weight
        self.ngram_orders = ngram_orders

        self.all_esm_toks = self.vocab.all_toks
        self.esm_vocab_char_to_idx = {
            char: idx
            for idx, char in enumerate(self.all_esm_toks)
            if char in self.allowed_AA
        }

    def _encode(self, sequences):
        def convert(token):
            return self.esm_vocab_char_to_idx[token]

        # A token could be invalid, specifically if the token is an
        # end-of-sentence or padding token
        def is_valid(token):
            return token in self.esm_vocab_char_to_idx

        big_list = [[convert(tkn) for tkn in seq if is_valid(tkn)] for seq in sequences]

        int_esm_encoded_seqs = torch.tensor(
            [[convert(tkn) for tkn in seq if is_valid(tkn)] for seq in sequences],
            device=get_device(),
        )

        return F.one_hot(int_esm_encoded_seqs, len(self.all_esm_toks)).float()

    def calc_total_loss(self, sequences):
        return super().calc_total_loss(
            x=self._encode(sequences),
            mask=None,
            LM_w=self.language_model_energy_term_weight,
            struct_w=False,
            ngram_w=self.ngram_energy_term_weight,
            ngram_orders=self.ngram_orders,
        )[0]


class ProteinModule(BaseSequenceModule):
    def __init__(
        self,
        num_train_iterations: int,
        batch_size: int = 64,
        max_len: int = 30,
        atomic_tokens: list[str] = [
            "L",
            "A",
            "G",
            "V",
            "S",
            "E",
            "R",
            "T",
            "I",
            "D",
            "P",
            "K",
            "Q",
            "N",
            "F",
            "Y",
            "M",
            "H",
            "W",
        ],
        language_model_energy_term_weight: float = 1,
        ngram_energy_term_weight: float = 0.5,
        ngram_orders: tuple[int] = (1, 2, 3),
        high_reward_threshold: int | None = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ) -> None:
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.max_len = max_len
        self.language_model_energy_term_weight = language_model_energy_term_weight
        self.ngram_energy_term_weight = ngram_energy_term_weight
        self.ngram_orders = ngram_orders
        self.high_reward_threshold = high_reward_threshold
        self.modes = []
        self.len_modes = 0
        self.esm_reward_calculator = ESMRewardModelWrapper(
            max_len,
            language_model_energy_term_weight,
            ngram_energy_term_weight,
            ngram_orders,
        )

        super().__init__(
            atomic_tokens=atomic_tokens,
            max_len=max_len,
            num_train_iterations=num_train_iterations,
            batch_size=batch_size,
            sample_exact_length=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **kwargs,
        )

        self.high_reward_strings = set()

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
        return self.esm_reward_calculator.calc_total_loss(sequences=sequences).cpu()

    def compute_metrics(
        self, states: torch.Tensor, logrewards: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute metrics for the given states.
        Args:
            states (torch.Tensor[batch_size, max_len, dim]): Batch of states.
            logrewards (torch.Tensor[batch_size]): Batch of logrewards.
        Returns:
            metrics (dict[str, torch.Tensor]): Dictionary of metrics.
        """
        strings = [
            s.replace("<EOS>", "") for s in self.to_strings(states)
        ]  # remove <EOS> tokens
        self.visited.update(set(strings))
        if hasattr(self, "modes"):
            unique_strings = set(strings)
            modes_found = unique_strings.intersection(self.modes)
            self.discovered_modes.update(modes_found)

        rewards = logrewards.exp()
        if self.high_reward_threshold is not None:
            (idx_where_high,) = torch.where(rewards >= self.high_reward_threshold)
            idx_where_high = idx_where_high.tolist()
            high_reward_strings = set([strings[i] for i in idx_where_high])
            self.high_reward_strings.update(high_reward_strings)

        metrics = {
            "num_visited": float(len(self.visited)),
            "num_modes": float(len(self.discovered_modes)),
            f"num_rewards_th_{self.high_reward_threshold}": float(
                len(self.high_reward_strings)
            ),
        }

        return metrics

    def build_test(self):
        """Build the test set based on known valid molecules.
        returns:
            test_seq (list[str]): List of test sequences.
            test_rs (list[float]): List of test logrewards.
        """
        test_seq = []

        if hasattr(self, "dataset"):
            sequences = self.dataset
        else:
            small_vcb = [a for a in self.atomic_tokens if a != "<EOS>"]
            sequences = [
                "".join(random.choices(small_vcb, k=self.max_len)) for _ in range(75)
            ]

        for seq in sequences:
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

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.high_reward_strings = state_dict["high_reward_strings"]

    def state_dict(self):
        state = super().state_dict()
        state["high_reward_strings"] = self.high_reward_strings
        return state
