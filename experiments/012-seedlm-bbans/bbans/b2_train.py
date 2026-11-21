"""B2: train the FP4 VAE on the 24 training matrices of 010 (10% of the blocks held out for validation).

Optimiser and schedule as the official full-MNIST script (tvae_beta_binomial.py): Adam, lr 5e-4, batch 100,
20 epochs, final parameters saved. Validation: single-sample negative ELBO (bits/weight), batch-norm in eval mode.

usage: python b2_train.py HIDDEN LATENT [EPOCHS]
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fp4vae as V  # noqa: E402

OUT = os.path.join(HERE, "../results/b2")
MODELS = os.path.join(V.DATA, "models")


def split(n, seed=0):
    perm = np.random.default_rng(seed).permutation(n)
    nv = n // 10
    return perm[nv:], perm[:nv]


@torch.no_grad()
def evaluate(model, sym, s, bs=65536):
    model.eval()
    tot = 0.0
    for i in range(0, len(sym), bs):
        torch.manual_seed(1 + i)
        tot += model.neg_elbo(sym[i:i + bs], s[i:i + bs]).sum().item()
    model.train()
    return tot / (len(sym) * V.B)


def main():
    H, D = int(sys.argv[1]), int(sys.argv[2])
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    dev = "cuda"
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)
    S, E = V.training_blocks()
    tr, va = split(len(S))
    symt = torch.from_numpy(S.astype(np.int64)).to(dev)
    st = V.scale_feature(E).to(dev)
    torch.manual_seed(17)
    model = V.FP4VAE(H, D).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    tr_t = torch.from_numpy(tr).to(dev)
    va_t = torch.from_numpy(va).to(dev)
    hist = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = tr_t[torch.randperm(len(tr_t), device=dev)]
        run = 0.0
        nb = 0
        for i in range(0, len(perm) - 99, 100):
            idx = perm[i:i + 100]
            opt.zero_grad(set_to_none=True)
            loss = model.loss(symt[idx], st[idx])
            loss.backward()
            opt.step()
            if nb % 1000 == 0:
                run += loss.item()
            nb += 1
        val = evaluate(model, symt[va_t], st[va_t])
        hist.append(dict(epoch=ep, train_loss_sampled=run / max(1, (nb + 999) // 1000), val_neg_elbo=val,
                         sec=time.time() - t0))
        print(json.dumps(hist[-1]), flush=True)
    path = os.path.join(MODELS, f"vae_h{H}_d{D}_e{epochs}.pt")
    torch.save(model.state_dict(), path)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    rec = dict(hidden=H, latent=D, epochs=epochs, params=V.n_params(model), model_file=path, sha256=sha,
               n_train_blocks=int(len(tr)), n_val_blocks=int(len(va)), val_neg_elbo=hist[-1]["val_neg_elbo"],
               best_val=min(h["val_neg_elbo"] for h in hist), history=hist, wall_s=time.time() - t0)
    with open(os.path.join(OUT, "train.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
