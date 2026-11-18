"""Spec 010 P1: capability probe -- a small neural network predicting each FP4 code from its context.

    PYTHONPATH=. python experiments/010-transforms/probe.py synth     # synthetic checks first
    PYTHONPATH=. python experiments/010-transforms/probe.py real      # train on 24 matrices, test on the other 24

Context of the code at (row r, column c): the previous 32 codes of the row, the 8 codes above in the same
column, the code of w1 at the same weight (w3: w1[r, c]; w2: w1[c, r], i.e. the same neuron and hidden unit;
w1 itself has none, so that the decode order w1 -> w3, w2 stays causal), the block's scale byte, the row index
(r / rows) and the layer index (layer / 39), plus the matrix role (one-hot; needed to tell the three matrices apart).
Everything in the context is decoded before the code, so -log2 p is a valid code length for a decoder that
has the network. The network weights are side information: reported with and without their cost (16 bit each).

Split: per layer the first 2 of the 4 sampled experts train (24 matrices), the other 2 test (24 matrices).
Hyper-parameters (epochs, learning rate) are chosen only on held-out positions of the training matrices.
Test code length: mean -log2 p over 500,000 uniformly drawn positions per test matrix (standard error reported),
compared with the 008 chain's code bits (chain total minus its scale stream) on the same matrices.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tx  # noqa: E402

from redundancy import increment as I  # noqa: E402

torch.set_num_threads(48)
PAD = 16
NPREV, NABOVE = 32, 8
RES = tx.RES


def gather(C, X, E, r, c, role, layer, B=32):
    """Context features for positions (r, c) of codes C; X = cross-matrix codes aligned to C or None."""
    R, W = C.shape
    Cp = np.full((R + NABOVE, W + NPREV), PAD, dtype=np.int64)
    Cp[NABOVE:, NPREV:] = C
    prev = Cp[(r + NABOVE)[:, None], (c + NPREV)[:, None] - np.arange(1, NPREV + 1)[None, :]]
    above = Cp[(r + NABOVE)[:, None] - np.arange(1, NABOVE + 1)[None, :], (c + NPREV)[:, None]]
    cross = X[r, c][:, None] if X is not None else np.full((len(r), 1), PAD, dtype=np.int64)
    toks = np.concatenate([prev, above, cross], axis=1).astype(np.int64)
    sc = E[r, c // B].astype(np.int64)
    num = np.stack([r / R, np.full(len(r), layer / 39.0)], axis=1).astype(np.float32)
    rl = np.zeros((len(r), 3), dtype=np.float32)
    rl[:, role] = 1
    return toks, sc, np.concatenate([num, rl], axis=1), C[r, c].astype(np.int64)


class Net(torch.nn.Module):
    def __init__(self, d=8, h=256):
        super().__init__()
        self.emb = torch.nn.Embedding(PAD + 1, d)
        self.semb = torch.nn.Embedding(256, d)
        n_in = (NPREV + NABOVE + 1) * d + d + 5
        self.mlp = torch.nn.Sequential(torch.nn.Linear(n_in, h), torch.nn.ReLU(), torch.nn.Linear(h, h),
                                       torch.nn.ReLU(), torch.nn.Linear(h, 16))

    def forward(self, toks, sc, num):
        x = torch.cat([self.emb(toks).flatten(1), self.semb(sc), num], dim=1)
        return self.mlp(x)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def bits_of(model, data, bs=65536):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(data[3]), bs):
            t, s, n, y = (torch.from_numpy(a[i:i + bs]) for a in data)
            lp = torch.log_softmax(model(t, s, n), dim=1)
            out.append((-lp.gather(1, y[:, None])[:, 0] / math.log(2)).numpy())
    return np.concatenate(out)


def cat(parts):
    return tuple(np.concatenate([p[k] for p in parts]) for k in range(4))


def train(tr, va, epochs, lr, seed=0, bs=4096, log=None):
    torch.manual_seed(seed)
    model = Net()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(tr[3])
    rng = np.random.default_rng(seed)
    hist = []
    best, best_state = None, None
    for ep in range(epochs):
        model.train()
        idx = rng.permutation(n)
        for i in range(0, n, bs):
            j = idx[i:i + bs]
            t, s, nm, y = (torch.from_numpy(a[j]) for a in tr)
            loss = torch.nn.functional.cross_entropy(model(t, s, nm), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        v = float(bits_of(model, va).mean())
        hist.append(v)
        if log:
            print(f"  lr {lr} epoch {ep + 1}: validation {v:.4f} bit/code", flush=True)
        if best is None or v < best:
            best, best_state = v, {k: x.clone() for k, x in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, hist


def fit(tr, va, log=True, epochs=4):
    """Choose the learning rate and the epoch on the validation positions only."""
    runs = {}
    for lr in (1e-3, 3e-3):
        m, h = train(tr, va, epochs=epochs, lr=lr, log=log)
        runs[lr] = (min(h), m, h)
    lr = min(runs, key=lambda k: runs[k][0])
    return runs[lr][1], {"lr": lr, "validation_bits_per_epoch": {str(k): v[2] for k, v in runs.items()}}


def draw(rng, shape, n):
    R, W = shape
    k = rng.choice(R * W, size=min(n, R * W), replace=False)
    return k // W, k % W


# ------------------------------------------------------------------ synthetic checks

def synth():
    rng = np.random.default_rng(5)
    p = np.array([0.2, 0.15, 0.12, 0.1, 0.08, 0.05, 0.03, 0.02] * 2) / 1.5
    p = p / p.sum()
    H = float(-(p * np.log2(p)).sum())
    E = np.full((512, 16), 120, dtype=np.uint8)
    out = {}
    for kind in ("iid", "markov"):
        mats = []
        for _ in range(5):
            C = rng.choice(16, size=(512, 512), p=p)
            if kind == "markov":     # with probability 1/2 repeat the previous code of the row
                rep = rng.random((512, 512)) < 0.5
                for c in range(1, 512):
                    C[:, c] = np.where(rep[:, c], C[:, c - 1], C[:, c])
            mats.append(C)
        parts = []
        for C in mats:
            r, c = draw(rng, C.shape, 200000)
            parts.append(gather(C, None, E, r, c, 0, 0))
        tr, va, te = cat(parts[:3]), parts[3], parts[4]
        model, info = fit(tr, va, log=False, epochs=8)
        b = bits_of(model, te)
        if kind == "iid":
            target = H
        else:   # H(C_t | C_t-1) for the mixture: repeat with prob 1/2 + 1/2 * p(x)
            q = 0.5 * np.eye(16) + 0.5 * p[None, :]
            target = float(-(p[:, None] * q * np.log2(q)).sum())
        out[kind] = {"test_bits_per_code": float(b.mean()), "target_entropy": target,
                     "pass": bool(abs(b.mean() - target) < 0.05)}
    out["all_pass"] = all(v["pass"] for v in out.values())
    return out


# ------------------------------------------------------------------ real matrices

def load_matrix(L, e, role):
    n = f"L{L}.E{e}."
    C, E = tx.read(n + role)
    X = None
    if role == "w3":
        X = tx.read(n + "w1")[0]
    elif role == "w2":
        X = np.ascontiguousarray(tx.read(n + "w1")[0].T)
    return C, X, E


def real(n_train=400000, n_test=500000):
    ex = tx.sample_experts()
    roles = {"w1": 0, "w2": 1, "w3": 2}
    rng = np.random.default_rng(tx.SEED + 7)
    tr_parts, va_parts, train_names = [], [], []
    t0 = time.time()
    for L, es in ex.items():
        for e in es[:2]:
            for role, ri in roles.items():
                C, X, E = load_matrix(L, e, role)
                r, c = draw(rng, C.shape, n_train + 40000)
                f = gather(C, X, E, r, c, ri, L)
                tr_parts.append(tuple(a[:n_train] for a in f))
                va_parts.append(tuple(a[n_train:] for a in f))
                train_names.append(f"L{L}.E{e}.{role}")
    tr, va = cat(tr_parts), cat(va_parts)
    print(f"training data {len(tr[3])} positions, {time.time() - t0:.0f}s", flush=True)
    model, info = fit(tr, va)
    base = {}
    for line in open(RES / "base.jsonl"):
        r = json.loads(line)
        base[r["task"]] = r["chain"]
    per = {}
    for L, es in ex.items():
        for e in es[2:]:
            for role, ri in roles.items():
                name = f"L{L}.E{e}.{role}"
                C, X, E = load_matrix(L, e, role)
                r, c = draw(rng, C.shape, n_test)
                b = bits_of(model, gather(C, X, E, r, c, ri, L))
                ch = base[name]
                scale = I.dirichlet_bits(E.ravel(), None, 256)
                per[name] = {"weights": int(C.size), "nn_bits_per_code": float(b.mean()),
                             "nn_se": float(b.std(ddof=1) / math.sqrt(len(b))),
                             "chain_code_bits_per_code": (ch["L"][-1] - scale) / C.size,
                             "chain_scale_bits_per_weight": scale / C.size,
                             "chain_total_bits_per_weight": ch["L"][-1] / C.size}
                print(name, {k: round(v, 4) for k, v in per[name].items() if k != "weights"}, flush=True)
    W = sum(v["weights"] for v in per.values())
    nn = sum(v["nn_bits_per_code"] * v["weights"] for v in per.values()) / W
    chn = sum(v["chain_code_bits_per_code"] * v["weights"] for v in per.values()) / W
    se = math.sqrt(sum((v["nn_se"] * v["weights"]) ** 2 for v in per.values())) / W
    npar = n_params(model)
    res = {"train_matrices": train_names, "test_matrices": sorted(per), "n_params": npar,
           "train_positions": int(len(tr[3])), "test_positions_per_matrix": n_test, "fit": info,
           "nn_bits_per_code": nn, "nn_se": se, "chain_code_bits_per_code": chn,
           "net_saving_bits_per_weight_without_model_cost": chn - nn,
           "model_cost_bits_per_weight_16bit": 16.0 * npar / W,
           "net_saving_bits_per_weight_with_model_cost": chn - nn - 16.0 * npar / W,
           "per_matrix": per}
    torch.save(model.state_dict(), RES / "p1_model.pt")
    return res


def main():
    RES.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "synth":
        res = synth()
        json.dump(res, open(RES / "p1_synthetic.json", "w"), indent=1)
    else:
        res = real()
        json.dump(res, open(RES / "p1.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_matrix"}, indent=1))


if __name__ == "__main__":
    main()
