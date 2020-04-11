import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple


INPUT_SIZE = 128
LATENT_DIM = 128
EMBEDDING_DIM = 128


class Vae(nn.Module):
    def __init__(self, hidden_size: int, n_tokens: int):
        super().__init__()
        self.encoder_lstm = nn.LSTM(INPUT_SIZE, hidden_size)
        self.encoder_linear = nn.Linear(hidden_size, LATENT_DIM)
        self.mu = torch.Parameter(torch.randn(LATENT_DIM))
        self.sigma = torch.Parameter(torch.rand(LATENT_DIM))
        self.noise = torch.randn(LATENT_DIM)
        self.decoder_linear = nn.Linear(LATENT_DIM, hidden_size)  # produces hidden state number zero
        self.embedding = nn.Embedding(n_tokens, EMBEDDING_DIM)
        self.decoder_lstm = nn.LSTM(EMBEDDING_DIM, hidden_size)
        self.output_layer = nn.Linear(hidden_size, n_tokens)

    def forward(self, x: torch.Tensor):
        _, (h, c) = self.encoder_lstm(x)
        x = self.encoder_linear(h)
        z = self.mu + self.sigma * self.noise
        x = self.decoder_linear(z)
        emb = self.embedding(x)
        _, (h, _) = self.decoder_lstm(emb)
        y = self.output_layer(h)
        return y
