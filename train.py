import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from flair.embeddings import BertEmbeddings

from data import Corpus
from model import Vae


# Loading corpus

corpus = Corpus("data/wikitext-2")
n_tokens = len(corpus.dictionary)
net = Vae(120, n_tokens=n_tokens)

use_cuda = True
device = torch.device("cuda") if torch.cuda.is_available() and use_cuda else torch.device("cpu")


# Embeddings



# Creating batches of data

def batchify(data: torch.Tensor, bsz: int) -> torch.Tensor:
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


batch_size = 20
eval_batch_size = 10

train_data = batchify(corpus.train, batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

bptt = 35


def get_embedding(index_tensor):
    pass


def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = get_embedding(source[i:i+seq_len])
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


# Training
lr = 0.01

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr)


def train():
    # Turn on training mode which enables dropout.
    net.train()
    total_loss = 0.
    start_time = time.time()
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, targets)
        loss.backward()
