"""B2: data and VAE for FP4 blocks.

Data point = one MXFP4 block: the 32 codes of a row segment (R10-merged, 15 symbols, txcore.merge) and its
scale byte E. Blocks are taken in raster order (row by row, left to right).

The VAE follows the official full-MNIST model (bits-back/torch_vae/tvae_beta_binomial.py):
  encoder  fc1 -> BN -> ReLU -> (fc21 -> BN = mean, fc22 -> BN -> exp = std)
  decoder  fc3 -> BN -> ReLU -> fc4
  prior N(0, I), posterior diagonal Gaussian, loss = single-sample negative ELBO in bits per dimension.
FP4 adaptations (report, adaptation table):
  * likelihood: fc4 outputs 32 x 15 logits, a categorical per position (paper: Bernoulli / beta-binomial);
  * input: the signed E2M1 level of each code divided by 12 (in [-1, 1], the analogue of pixel / 255);
  * the block scale is an extra input to both the encoder and the decoder: (E - 120) / 4, a scalar.
"""
import os
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, os.path.join(ROOT, "experiments/010-transforms"))
sys.path.insert(0, os.path.join(ROOT, "experiments/010-transforms/tx"))
sys.path.insert(0, os.path.join(ROOT, "experiments/011-limits-and-models"))

NSYM = 15
B = 32
DATA = os.path.join(ROOT, "data/012/b2")
# level of each merged symbol (2 x E2M1 value): 0..7 = +{0,1,2,3,4,6,8,12}, 8..14 = -{1,2,3,4,6,8,12}
LEVEL = np.array([0, 1, 2, 3, 4, 6, 8, 12, -1, -2, -3, -4, -6, -8, -12], dtype=np.float32)


def train_names():
    import tx
    import common as K
    ex = tx.sample_experts()
    return K.names({L: es[:2] for L, es in ex.items()})


def matrix_blocks(name):
    """(sym (n_blocks, 32) uint8, E (n_blocks,) uint8, shape) in raster block order."""
    import tx
    from txcore import merge
    C, E = tx.read(name)
    sym = merge(C).astype(np.uint8)
    r, c = sym.shape
    return sym.reshape(r, c // B, B).reshape(-1, B), np.asarray(E, dtype=np.uint8).reshape(-1), (r, c)


def training_blocks():
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "train_blocks.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["sym"], z["E"]
    S, E = [], []
    for n in train_names():
        s, e, _ = matrix_blocks(n)
        S.append(s)
        E.append(e)
    S, E = np.concatenate(S), np.concatenate(E)
    np.savez(path, sym=S, E=E)
    return S, E


def scale_feature(E):
    return (torch.as_tensor(E).float() - 120.0) / 4.0


class FP4VAE(nn.Module):
    """bn=True: official full-MNIST structure (batch norm); bn=False: official binarised-MNIST structure (no batch norm)."""

    def __init__(self, hidden_dim=200, latent_dim=50, bn=True):
        super().__init__()
        self.hidden_dim, self.latent_dim, self.bn = hidden_dim, latent_dim, bn
        self.register_buffer("level", torch.from_numpy(LEVEL / 12.0))
        norm = nn.BatchNorm1d if bn else (lambda d: nn.Identity())
        self.fc1 = nn.Linear(B + 1, hidden_dim)
        self.bn1 = norm(hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        self.bn21 = norm(latent_dim)
        self.bn22 = norm(latent_dim)
        self.fc3 = nn.Linear(latent_dim + 1, hidden_dim)
        self.bn3 = norm(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, B * NSYM)

    def encode(self, sym, s):
        """sym: (n, 32) long, s: (n,) scale feature -> mean, std."""
        x = torch.cat([self.level[sym], s[:, None]], 1)
        h = F.relu(self.bn1(self.fc1(x)))
        return self.bn21(self.fc21(h)), torch.exp(self.bn22(self.fc22(h)))

    def decode(self, z, s):
        """-> logits (n, 32, 15)."""
        h = F.relu(self.bn3(self.fc3(torch.cat([z, s[:, None]], 1))))
        return self.fc4(h).view(-1, B, NSYM)

    def neg_elbo(self, sym, s, sample=True, beta=1.0):
        """per-block negative ELBO in bits (sum over the 32 positions); beta < 1 only for KL warm-up (extension)."""
        mu, std = self.encode(sym, s)
        z = mu + std * torch.randn_like(std) if sample else mu
        logits = self.decode(z, s)
        ll = -F.cross_entropy(logits.reshape(-1, NSYM), sym.reshape(-1), reduction="none").view(-1, B).sum(1)
        pz = (-0.5 * z * z - 0.5 * np.log(2 * np.pi)).sum(1)
        qz = (-0.5 * ((z - mu) / std) ** 2 - torch.log(std) - 0.5 * np.log(2 * np.pi)).sum(1)
        return -(ll + beta * (pz - qz)) / np.log(2)

    def loss(self, sym, s, beta=1.0):
        return self.neg_elbo(sym, s, beta=beta).mean() / B   # bits per weight, as the official loss


def n_params(m):
    return sum(p.numel() for p in m.parameters())
