import random

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
    def __init__(self, max_len=30, num_sentences=1000):
        self.vocab2token = {k: i for i, k in enumerate(["A", "B", "C"])}
        self.token2vocab = {i: k for i, k in enumerate(["A", "B", "C"])}
        self.num_sentences = num_sentences
        self.data = [generate_sentence(max_len) for _ in range(num_sentences)]

    def __len__(self):
        return self.num_sentences

    def __getitem__(self, i):
        vec = torch.tensor([self.vocab2token[k] for k in self.data[i]])
        out = torch.zeros((len(vec), len(self.vocab2token)))
        out[range(len(vec)), vec] = 1
        return out
