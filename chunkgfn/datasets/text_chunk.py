import random
from itertools import chain

import torch
from torch.utils.data import Dataset

# Define chunk vocabulary examples
three_letter_chunks = ["AAA", "BBB", "CCC", "ACC", "BCA", "CCB", "AAB", "BBC", "CCA"]
six_letter_chunks = [
    "AAABBB",
    "BBACCC",
    "CCBAAA",
    "ACBBBC",
    "BACACC",
    "CBBAAA",
    "ABCABC",
    "BCACBC",
    "CACABC",
]
nine_letter_chunks = [
    "AAABBBCAC",
    "BBCACCCBB",
    "CCBBBAAAB",
    "ACBBBCBAA",
    "BACACACAC",
    "CBBAAAACA",
    "ABCBCBABA",
    "BCACABCCC",
    "CACABCBBB",
]

# Define rules for generating sentences (for now only if starts then can end with)
rules = [
    ("AAA", ["AAA", "CCC", "BBB", "AAC", "AAABBBCCC"]),
    ("BBB", ["BBB", "AAA", "CCC", "BCA", "CCBAAABBB"]),
    ("CCC", ["CCC", "BBB", "AAA", "ACC", "BBCAACCCC"]),
    ("ACC", ["CBB", "BBB", "CAC", "AAABBBCAC"]),
    ("BCA", ["CAC", "ABC", "BAC", "ACBBBCBAA"]),
    ("CCB", ["BBB", "ABC", "CCA", "ABCBCBABA"]),
]


# Generate a dataset of sentences
def generate_sentence(max_len):
    while True:
        start_chunk, end_chunks = random.choice(rules)
        middle_chunk = random.choice(six_letter_chunks + nine_letter_chunks)
        end_chunk = random.choice(end_chunks)
        sentence = start_chunk + middle_chunk + end_chunk
        if len(sentence) != max_len:
            # If the sentence length is not 30, modify the middle_chunk by adding 3-letter tokens
            required_length = max_len - len(sentence)
            if required_length <= 0:
                continue  # Skip this iteration if the middle_chunk is too long
            middle_chunk += "".join(
                random.choice(three_letter_chunks) for _ in range(required_length // 3)
            )
            sentence = start_chunk + middle_chunk + end_chunk
        return sentence


class ChunkDataset(Dataset):
    def __init__(self, data):
        self._vocab = {"<EOS>": [0], "A": [1], "B": [2], "C": [3]}
        self.vocab2token = {k: i for i, k in enumerate(self.vocab)}
        self.token2vocab = {i: k for i, k in enumerate(self.vocab)}
        self.data = data
        self.action_len = torch.Tensor([0, 1, 1, 1])

    def __len__(self):
        return len(self.data)

    @property
    def vocab_tensor(self):
        dim = len(self.vocab2token)
        max_len = max([len(self._vocab[key]) for key in self._vocab])
        _vocab_tensor = -torch.ones(len(self._vocab), max_len, dim)
        for key in self._vocab:
            idx = self.vocab2token[key]
            _vocab_tensor[idx, : len(self._vocab[key])] = torch.eye(dim)[
                self._vocab[key]
            ]
        return _vocab_tensor

    @property
    def vocab(self):
        return self._vocab

    @property
    def vocab_size(self):
        return len(self._vocab)

    @property
    def eos_token_idx(self):
        return self.vocab2token["<EOS>"]

    def add_to_vocab(self, token):
        if token not in self.vocab:
            self.token2vocab[len(self._vocab)] = token
            self._vocab[token] = list(
                chain.from_iterable([self._vocab[atom] for atom in token])
            )
            self.action_len = torch.cat(
                [self.action_len, torch.Tensor([len(token)])], dim=0
            )

    def __getitem__(self, i):
        vec = torch.tensor(
            [self.vocab2token[k] for k in self.data[i]] + [self.vocab2token["<EOS>"]]
        )
        out = torch.zeros((len(vec), len(self.vocab2token)))
        out[range(len(vec)), vec] = 1
        return out
