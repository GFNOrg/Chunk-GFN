import os
import gzip
from collections import Counter

import pandas as pd
import seaborn as sns
import numpy as np
from Bio import SeqIO
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import torch
from torch.utils.data import Dataset





def load_fasta_into_dict(data_dir):
    input_file = os.path.join(data_dir, "ccds/CCDS_nucleotide.current.fna.gz")

    records = {}
    with gzip.open(input_file, "rt") as file:
        for record in SeqIO.parse(file, "fasta"):
            chars = np.unique([i for i in str(record.seq)])

            # Only save normal DNA sequences.
            if all(chars == ["A", "C", "G", "T"]) and len(chars) == 4:
                # Mapping of identifier -> sequence.
                records[record.id] = str(record.seq)

    return records


def get_tokenizer(records, use_special_tokens=False, train=False):
    # Initalize the tokenizer.
    tokenizer = Tokenizer(BPE({"T": 0, "C": 1, "A":2, "G":3}, [], unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    if train:
        if use_special_tokens:
            trainer = BpeTrainer(
                special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
            )
        else:
            trainer = BpeTrainer()
        tokenizer.train_from_iterator(records.values(), trainer)

    return tokenizer


def count_instances(records, tokenizer):
    tokenized_texts = [tokenizer.encode(text).tokens for text in records.values()]
    all_tokens = [token for sublist in tokenized_texts for token in sublist]
    token_counts = Counter(all_tokens)

    # Sort the dict by the values (the token counts).
    token_counts = dict(sorted(token_counts.items(), key=lambda item: item[1]))
    return token_counts



def split_strings(input_list: list, max_length: int):
    """Chunks the submitted list into strings with a defined maximum length.

    Accepts a list of strings and a max length. For each string, if it is longer than
    max length, this will truncate each, and append them to the output list.
    """
    output_list = []
    for s in input_list:
        while len(s) > max_length:
            output_list.append(s[:max_length])  # Add chunk.
            s = s[max_length:]  # Truncate on chunk.

        # Remaining chunk - if it exists.
        if s:
            output_list.append(s)

    return output_list


class CCDS(Dataset):
    def __init__(self, batch_size: int = 32, max_len: int = 128):
        self.batch_size = batch_size
        self.max_len = max_len

        data_dir = os.getenv("CHUNKGFN_DATA")
        records = load_fasta_into_dict(data_dir)
        self.tokenizer = get_tokenizer(records, train=False)

        # Stores all of the raw DNA sequences.
        all_sequences = list(records.values())
        all_sequences = split_strings(all_sequences, max_len)
        self.all_sequences = np.array(all_sequences)
        self.n_seq = len(all_sequences)

        # Batch indices, which index shuffled sample indices.
        self.sample_idx = np.arange(self.n_seq)
        np.random.shuffle(self.sample_idx)
        self.batch_idx = np.arange(0, self.n_seq, batch_size)
        self.n_batch = len(self.batch_idx) - 1  # b/c we use the [i] and [i+1] index.

    def get_batch(self, i):
        assert 0 <= i <= self.n_batch - 1  # -1 again to handle 0-indexing.
        batch = self.tokenizer.encode_batch(
            self.all_sequences[
                self.sample_idx[self.batch_idx[i]:self.batch_idx[i+1]]
            ]
        )

        for i in batch:
            i.pad(self.max_len)

        batch = torch.tensor([i.ids for i in batch]).long()

        return batch

    def __len__(self):
        return self.n_batch

    def __getitem__(self, i):
        return self.get_batch(i)

    def __call__(self, i):
        return self.get_batch(i)

    def add_tokens(self, tokens: list):
        self.tokenizer.add_tokens(tokens)

    def id_to_token(self, id: int):
        return self.tokenizer.id_to_token(id)

    def decode(self, ids: list):
        return self.tokenizer.decode(ids)


if __name__ == "__main__":
    dset = CCDS()
    import IPython; IPython.embed()