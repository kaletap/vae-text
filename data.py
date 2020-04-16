from collections import Counter
import os
from io import open
import torch
from string import digits
from typing import List
import torch
from transformers import BertTokenizer

SOS = 0
EOS = 1
UNK = 2
NUM = 3


INITIAL_WORD2IDX = {"<sos>": SOS, "<eos>": EOS, "<unk>": UNK, "<num>": NUM}
INITIAL_IDX2WORD = ["<sos>", "<eos>", "<unk>", "<num>"]


class Dictionary:
    def __init__(self):
        self.word2idx = INITIAL_WORD2IDX.copy()
        self.idx2word = INITIAL_IDX2WORD.copy()
        self.count = dict.fromkeys(self.word2idx.keys(), 1)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            self.count[word] = 1
        else:
            self.count[word] += 1
        return self.word2idx[word]

    def shrink(self, size: int = 10_000):
        """Reduces dictionary to `size` most popular words."""
        most_popular_words = sorted(self.word2idx.keys(), key=lambda k: self.count[k], reverse=True)
        self.count = Counter(most_popular_words)
        self.word2idx = INITIAL_WORD2IDX.copy()
        self.idx2word = INITIAL_IDX2WORD.copy()
        for word in most_popular_words[:size]:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1

    def __getitem__(self, word: str) -> int:
        return self.word2idx.get(word, UNK)

    def __len__(self):
        return len(self.idx2word)


class Corpus:
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'), train=False)
        self.test = self.tokenize(os.path.join(path, 'test.txt'), train=False)

    def tokenize(self, path: str, train: bool = True, shrink: int = None) -> torch.Tensor:
        """Tokenizes a text file and adds words to a dictionary if in train mode."""
        assert os.path.exists(path)
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                tokens = self.tokenizer.encode(line)
                if train:
                    # Add words to the dictionary
                    words = self.tokenizer.decode(tokens)
                    for word in words:
                        self.dictionary.add_word(word)
                idss.append(torch.tensor(tokens))
        if shrink:
            self.dictionary.shrink(size=shrink)
        return torch.cat(idss)
