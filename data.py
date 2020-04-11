import os
from io import open
import torch

SOS = 0
EOS = 1
UNK = 2
NUM = 3


class Dictionary:
    def __init__(self):
        self.word2idx = {"<sos>": SOS, "<eos>": EOS, "<unk>": UNK, "<num>": NUM}
        self.idx2word = ["<sos>", "<eos>", "<unk>", "<num>"]
        self.count = dict.fromkeys(self.word2idx.keys(), 1)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            self.count[word] = 1
        else:
            self.count[word] += 1
        return self.word2idx[word]

    def __getitem__(self, word: str) -> int:
        return self.word2idx.get(word, UNK)

    def __len__(self):
        return len(self.idx2word)


class Corpus:
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'), train=False)
        self.test = self.tokenize(os.path.join(path, 'test.txt'), train=False)

    def tokenize(self, path, train=True):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        if train:
            # Add words to the dictionary
            with open(path, 'r', encoding="utf8") as f:
                for line in f:
                    words = ["<sos>"] + line.split() + ['<eos>']
                    for word in words:
                        self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids
