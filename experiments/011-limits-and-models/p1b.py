"""Spec 010 P1b (moved to 011 with the other neural-network work): correction probe on top of the 008 chain.

    PYTHONPATH=. python experiments/011-limits-and-models/p1b.py verify    # per-position chain = closed form, 48 matrices
    PYTHONPATH=. python experiments/011-limits-and-models/p1b.py synth     # V-a, V-b and the zero-init identity check
    PYTHONPATH=. python experiments/011-limits-and-models/p1b.py data      # training samples of the 24 training matrices
    PYTHONPATH=. python experiments/011-limits-and-models/p1b.py train     # offline model + online learning-rate choice
    PYTHONPATH=. python experiments/011-limits-and-models/p1b.py test      # offline and online code length, 24 test matrices
    PYTHONPATH=. python experiments/011-limits-and-models/p1b.py ablate    # only if a gain > 0.005 bit per weight

Chain predictions: `chain_probs` rebuilds, for the patterns the 008 chain actually switched on for a matrix,
the same context levels as increment.fp4_codelen and gives the sequential predictive distribution of every
code (hier_bits computes the same code length in closed form). The probe outputs
logits = log p_chain + f(features) with the last layer of f initialised to 0.

Decode order: the 008 chain codes w1 with w3 as context, so w3 is decoded first, then w1, then w2. To keep
the chain as a special case of the probe, the cross-matrix features follow that order: w1 sees w3 (same
position and the same row +-16), w2 sees w1 (w2[r, c] <-> w1[c, r]; the row c of w1 around column r), w3 sees
no other matrix. (P1 used w1 as context of w3; that is not possible here without dropping the 008 R7 of w1.)
All features use the codes after the R10 merge (+0 = -0), which is what the chain transmits.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import copy
import json
import math
import sys
import multiprocessing
import re
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "010-transforms"))
import tx  # noqa: E402

from redundancy import increment as I  # noqa: E402

RES = Path(__file__).resolve().parent / "results" / "p1b"     # the 008 chain of the 48 matrices stays in tx.RES
P = tx.P
ORDER = tx.ORDER
SWITCH = ["R2", "R3", "R4", "R5", "R6", "R7", "R8"]   # one flag bit each in the chain
NPREV, NABOVE, NWIN = 32, 8, 16
ROLES = {"w1": 0, "w2": 1, "w3": 2}


# ------------------------------------------------------------------ per-position chain predictions

def chain_levels(C, E, flags, partner=None, params=None, seed=0):
    """Context levels of fp4_codelen for the given (active) flags; returns sym, levels, K, side and scale bits."""
    Pp = params or P
    C = np.asarray(C, dtype=np.int64)
    R, W = C.shape
    n = R * W
    Bk = Pp["block"]
    if "R6" in flags:
        raise NotImplementedError("R6 changes the coded positions; not switched on in this sample")
    scale = I.dirichlet_bits(E.ravel(), None, 256) if "R1s" in flags else 8.0 * E.size
    merge = "R10" in flags
    Cm = np.where(C == 8, 0, C) if merge else C
    sym = np.where(Cm > 8, Cm - 1, Cm) if merge else Cm
    K = 15 if merge else 16
    assert "R1c" in flags
    V = I.FP4_VALUES[Cm]
    tau = Pp["pred_tau"] * float(np.sqrt(np.mean(V * V)) or 1.0)
    feats, side = {}, 0.0
    if "R2" in flags:
        su, sinv = np.unique(E, return_inverse=True)
        feats["R2"] = np.repeat(sinv.reshape(E.shape), Bk, axis=1).ravel()
    if "R3" in flags:
        f = np.zeros((R, W), dtype=np.int64)
        for j in range(1, Pp["ctx_order"] + 1):
            prevc = np.full((R, W), 16, dtype=np.int64)
            prevc[:, j:] = Cm[:, :-j]
            f = f * 17 + prevc
        feats["R3"] = f.ravel()
    if "R4" in flags:
        pv = I._row_lp(V, Pp["lp_order"], Pp["lp_rows"])
        A = np.abs(V)
        pa = I._row_lp(A - A.mean(), Pp["lp_order"], Pp["lp_rows"])
        feats["R4"] = (I._tri(pv, tau) * 3 + I._tri(pa, tau)).ravel()
    if "R5" in flags:
        ap, sb = I._lowrank_float(V, Pp["rank"])
        feats["R5"], side = I._tri(ap, tau).ravel(), side + sb
    if "R7" in flags and partner is not None:
        Pm = np.asarray(partner, dtype=np.int64)
        Pm = np.where(Pm == 8, 0, Pm) if merge else Pm
        Vp = I.FP4_VALUES[Pm]
        rs = np.sign((V * Vp).sum(axis=1))
        rs[rs == 0] = 1
        side += I.kt_bits(np.zeros(R), (rs > 0).astype(np.int64))
        feats["R7"] = ((np.sign(Vp) * rs[:, None] + 1).astype(np.int64) * 2 + (np.abs(Vp) >= 2.0)).ravel()
    if "R8" in flags:
        rng = np.random.default_rng(seed)
        X = V.reshape(R * (W // Bk), Bk)
        smp = X[rng.choice(len(X), min(Pp["dict_sample"], len(X)), replace=False)]
        D = I._kmeans(smp, Pp["dict_atoms"], Pp["dict_iters"], rng).astype(np.float16).astype(np.float64)
        d = (X * X).sum(1)[:, None] - 2 * X @ D.T + (D * D).sum(1)[None, :]
        lab = d.argmin(1)
        side += 16.0 * D.size + I.dirichlet_bits(lab, None, Pp["dict_atoms"])
        feats["R8"] = I._tri(D[lab].reshape(R, W), tau).ravel()
    levels = [np.zeros(n, dtype=np.int64)] + [feats[p] for p in ORDER if p in feats]
    return sym.ravel(), levels, K, side, scale


def hier_probs(sym, levels, K, beta, nblocks=32, alpha=0.5, first=256):
    """Per-position predictive distribution (n, K) of the sequential code whose length hier_bits gives in
    closed form: parent predictions frozen at the start of each block, the finest level counted sequentially."""
    sym = np.asarray(sym, dtype=np.int64).ravel()
    n = len(sym)
    edges = I.block_edges(n, nblocks, first)
    blk = np.searchsorted(edges, np.arange(n), side="right") - 1
    prev_id = prev_rows = None
    for lv, ctx in enumerate(levels):
        ctx = np.asarray(ctx, dtype=np.int64).ravel()
        key = ctx if prev_id is None else prev_id * (int(ctx.max()) + 1) + ctx
        u, inv = I.fast_unique(key)
        idv = inv.astype(np.int64)
        U = len(u)
        par = None
        if prev_id is not None:
            par = np.zeros(U, dtype=np.int64)
            par[idv] = prev_id
        rkeys, rinv = I.fast_unique(blk * U + idv)
        rinv = rinv.astype(np.int64)
        Rn = len(rkeys)
        M = np.bincount(rinv * K + sym, minlength=Rn * K).reshape(Rn, K).astype(np.float64)
        rb, rid = rkeys // U, rkeys % U
        order = np.lexsort((rb, rid))
        Mo = M[order]
        excl = np.cumsum(Mo, axis=0) - Mo
        ido = rid[order]
        first_row = np.concatenate([[True], ido[1:] != ido[:-1]])
        g0 = np.maximum.accumulate(np.where(first_row, np.arange(Rn), 0))
        Cc = np.empty_like(Mo)
        Cc[order] = excl - excl[g0]
        if lv == 0:
            prior = np.full((Rn, K), alpha)
        else:
            pkeys, pq, pU = prev_rows
            prior = beta * pq[np.searchsorted(pkeys, rb * pU + par[rid])]
        a = Cc + prior
        if lv == len(levels) - 1:
            srt = np.argsort(rinv, kind="stable")          # positions grouped by (block, context), in coding order
            rs_ = rinv[srt]
            oh = np.zeros((n, K), dtype=np.int32)
            oh[np.arange(n), sym[srt]] = 1
            cum = np.cumsum(oh, axis=0, dtype=np.int32) - oh
            start = np.concatenate([[True], rs_[1:] != rs_[:-1]])
            gi = np.maximum.accumulate(np.where(start, np.arange(n), 0))
            before = cum - cum[gi]
            num = a[rs_] + before
            p = np.empty((n, K), dtype=np.float64)
            p[srt] = num / num.sum(axis=1, keepdims=True)
            return p
        prev_rows = (rkeys, a / a.sum(axis=1, keepdims=True), U)
        prev_id = idv
    raise AssertionError


def chain_probs(C, E, flags, partner=None):
    sym, levels, K, side, scale = chain_levels(C, E, flags, partner)
    p = hier_probs(sym, levels, K, P["beta"], P["nblocks"])
    return sym, p, K, side, scale, levels


# ------------------------------------------------------------------ data access (008 decode order)

def base_chains():
    out = {}
    for line in open(tx.RES / "base.jsonl"):
        r = json.loads(line)
        out[r["task"]] = r["chain"]
    return out


def active(chain):
    return frozenset(p for p in ORDER if chain["used"][p])


def load(name):
    """codes, scales, R7 partner (w1: w3), cross source for the features and its index map."""
    L, e, role = name.split(".")
    pre = f"{L}.{e}."
    C, E = tx.read(name)
    partner = tx.read(pre + "w3")[0] if role == "w1" else None
    cross = None
    if role == "w1":
        cross = ("same", tx.read(pre + "w3")[0])
    elif role == "w2":
        cross = ("transposed", tx.read(pre + "w1")[0])
    return C, E, partner, cross


def merged(C):
    Cm = np.where(C == 8, 0, C)
    return np.where(Cm > 8, Cm - 1, Cm)


# ------------------------------------------------------------------ features

def feature_maker(S, E, cross, role, layer, K):
    """Returns f(r, c) -> dict of arrays for positions (r, c). S: merged symbols (R, W)."""
    R, W = S.shape
    PAD = K
    Sp = np.full((R + NABOVE, W + NPREV), PAD, dtype=np.int64)
    Sp[NABOVE:, NPREV:] = S
    Y = None
    if cross is not None:
        kind, Xc = cross
        Xs = merged(Xc)
        Y = np.full((Xs.shape[0], Xs.shape[1] + 2 * NWIN), PAD, dtype=np.int64)
        Y[:, NWIN:NWIN + Xs.shape[1]] = Xs
    Ef = E.astype(np.float64)
    smean = (Ef.mean(axis=1) - 120.0) / 10.0
    sstd = Ef.std(axis=1) / 10.0
    off = np.arange(-NWIN, NWIN + 1)

    def f(r, c):
        prev = Sp[(r + NABOVE)[:, None], (c + NPREV)[:, None] - np.arange(1, NPREV + 1)[None, :]]
        above = Sp[(r + NABOVE)[:, None] - np.arange(1, NABOVE + 1)[None, :], (c + NPREV)[:, None]]
        if Y is None:
            win = np.full((len(r), 2 * NWIN + 1), PAD, dtype=np.int64)
        elif cross[0] == "same":
            win = Y[r[:, None], (c + NWIN)[:, None] + off[None, :]]
        else:      # w2[r, c] <-> w1[c, r]: row c of w1 around column r
            win = Y[c[:, None], (r + NWIN)[:, None] + off[None, :]]
        toks = np.concatenate([prev, above, win], axis=1).astype(np.int16)
        num = np.zeros((len(r), 7), dtype=np.float32)
        num[:, 0] = r / R
        num[:, 1] = layer / 39.0
        num[:, 2 + role] = 1
        num[:, 5] = smean[r]
        num[:, 6] = sstd[r]
        return {"toks": toks, "sc": E[r, c // 32].astype(np.int16), "pos": (c % 32).astype(np.int16), "num": num}
    return f


# ------------------------------------------------------------------ model

def make_net(K, d=8, h=256):
    import torch

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(K + 1, d)
            self.semb = torch.nn.Embedding(256, d)
            self.pemb = torch.nn.Embedding(32, 4)
            n_in = (NPREV + NABOVE + 2 * NWIN + 1) * d + d + 4 + 7 + K
            self.body = torch.nn.Sequential(torch.nn.Linear(n_in, h), torch.nn.ReLU(), torch.nn.Linear(h, h), torch.nn.ReLU())
            self.out = torch.nn.Linear(h, K)
            torch.nn.init.zeros_(self.out.weight)
            torch.nn.init.zeros_(self.out.bias)
            self.groups = None      # ablation: names of feature groups set to neutral values

        def forward(self, toks, sc, pos, num, logp):
            x = torch.cat([self.emb(toks.long()).flatten(1), self.semb(sc.long()), self.pemb(pos.long()), num,
                           logp.clamp(min=-30.0)], dim=1)
            return logp + self.out(self.body(x))
    return Net()


def n_params(m):
    return sum(p.numel() for p in m.parameters())


ABLATE = {   # feature groups -> how they are neutralised (set to constants) in an ablation
    "row_prev32": lambda b, K: b["toks"][:, :NPREV].fill_(K),
    "column_above8": lambda b, K: b["toks"][:, NPREV:NPREV + NABOVE].fill_(K),
    "cross_window": lambda b, K: b["toks"][:, NPREV + NABOVE:].fill_(K),
    "scale_and_row_scale_stats": lambda b, K: (b["sc"].fill_(0), b["num"][:, 5:7].fill_(0)),
    "position_in_block": lambda b, K: b["pos"].fill_(0),
    "row_layer_role": lambda b, K: b["num"][:, 0:5].fill_(0),
    "log_p_chain_input": None,     # handled in the model call (input copy of log p_chain set to 0)
}


def to_t(batch):
    import torch
    return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in batch.items()}


def logits(model, b, drop=None, K=15):
    import torch
    if drop and drop != "log_p_chain_input":
        b = {k: v.clone() for k, v in b.items()}
        ABLATE[drop](b, K)
    if drop == "log_p_chain_input":
        x = torch.cat([model.emb(b["toks"].long()).flatten(1), model.semb(b["sc"].long()), model.pemb(b["pos"].long()),
                       b["num"], torch.zeros_like(b["logp"])], dim=1)
        return b["logp"] + model.out(model.body(x))
    return model(b["toks"], b["sc"], b["pos"], b["num"], b["logp"])


def nll_bits(model, b, drop=None, K=15):
    import torch
    lp = torch.log_softmax(logits(model, b, drop, K), dim=1)
    return -lp.gather(1, b["y"].long()[:, None])[:, 0] / math.log(2)


# ------------------------------------------------------------------ one matrix: chain probabilities + features

def matrix_stream(name, chain):
    """Everything needed to run the probe over one matrix in coding order."""
    C, E, partner, cross = load(name)
    flags = active(chain)
    sym, p, K, side, scale, _ = chain_probs(C, E, flags, partner)
    R, W = C.shape
    layer = int(name.split(".")[0][1:])
    role = ROLES[name.split(".")[2]]
    fm = feature_maker(sym.reshape(R, W), E, cross, role, layer, K)
    return {"sym": sym, "logp": np.log(np.maximum(p, 1e-300)).astype(np.float32), "K": K, "side": side,
            "scale": scale, "R": R, "W": W, "fm": fm, "flags": sorted(flags),
            "chain_bits": float(-np.log2(p[np.arange(len(sym)), sym]).sum())}


def batch_at(st, idx):
    r, c = idx // st["W"], idx % st["W"]
    b = st["fm"](r, c)
    b["logp"] = st["logp"][idx]
    b["y"] = st["sym"][idx].astype(np.int16)
    return b


# ------------------------------------------------------------------ stage: verify

def verify_one(name):
    t0 = time.process_time()
    chain = base_chains()[name]
    C, E, partner, _ = load(name)
    flags = active(chain)
    sym, p, K, side, scale, levels = chain_probs(C, E, flags, partner)
    seq = float(-np.log2(p[np.arange(len(sym)), sym]).sum())
    closed = I.hier_bits(sym, levels, K, P["beta"], P["nblocks"])
    ref = I.fp4_codelen(C, E, flags, P, ORDER, partner=partner)[0]
    ours = scale + side + seq
    n = C.size
    return {"task": name, "flags": sorted(flags), "seq_code_bits": seq, "closed_form_code_bits": closed,
            "ours_total": ours, "fp4_codelen_total": ref, "chain_L_minus_switch_bits": chain["L"][-1] - len(SWITCH),
            "diff_bits_per_weight": (ours - ref) / n, "seq_vs_closed_bits_per_weight": (seq - closed) / n,
            "diff_vs_008_result_bits_per_weight": (ours - (chain["L"][-1] - len(SWITCH))) / n,
            "cpu": time.process_time() - t0}


def stage_verify(workers=24):
    names = sorted(base_chains())
    out = []
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(verify_one, names):
            out.append(r)
            print(r["task"], f"{r['diff_bits_per_weight']:.2e} {r['seq_vs_closed_bits_per_weight']:.2e} "
                             f"{r['diff_vs_008_result_bits_per_weight']:.2e}", flush=True)
    worst = max(abs(r["diff_bits_per_weight"]) for r in out)
    worst8 = max(abs(r["diff_vs_008_result_bits_per_weight"]) for r in out)
    res = {"matrices": len(out), "max_abs_diff_bits_per_weight_vs_fp4_codelen": worst,
           "max_abs_diff_bits_per_weight_vs_008_stored_result": worst8, "pass": bool(max(worst, worst8) < 0.001),
           "rows": out}
    json.dump(res, open(RES / "verify.json", "w"), indent=1)
    print("max diff", worst, worst8, "pass", res["pass"])


# ------------------------------------------------------------------ training / evaluation helpers

def sample_batches(streams, n_per, rng):
    parts = []
    for st in streams:
        idx = np.sort(rng.choice(st["R"] * st["W"], size=n_per, replace=False))
        parts.append(batch_at(st, idx))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def fit(train, val, K, epochs=4, lrs=(1e-3, 3e-3), bs=4096, seed=0, log=print, threads=48, drop=None):
    import torch
    torch.set_num_threads(threads)
    tr, va = to_t(train), to_t(val)
    n = len(train["y"])
    best_all = None
    hist = {}
    for lr in lrs:
        torch.manual_seed(seed)
        model = make_net(K)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        rng = np.random.default_rng(seed)
        h = [float(eval_bits(model, va, K, drop).mean())]
        best, state = h[0], copy.deepcopy(model.state_dict())
        for ep in range(epochs):
            model.train()
            idx = torch.from_numpy(rng.permutation(n))
            for i in range(0, n, bs):
                j = idx[i:i + bs]
                b = {k: v[j] for k, v in tr.items()}
                loss = nll_bits(model, b, drop, K).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            v = float(eval_bits(model, va, K, drop).mean())
            h.append(v)
            log(f"  lr {lr} epoch {ep + 1}: validation {v:.5f} bit/code (chain {h[0]:.5f})")
            if v < best:
                best, state = v, copy.deepcopy(model.state_dict())
        hist[str(lr)] = h
        if best_all is None or best < best_all[0]:
            best_all = (best, lr, state)
    model = make_net(K)
    model.load_state_dict(best_all[2])
    return model, {"lr": best_all[1], "validation_bits_per_epoch": hist, "epoch0_is_chain": True}


def eval_bits(model, b, K, drop=None, bs=65536):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(b["y"]), bs):
            out.append(nll_bits(model, {k: v[i:i + bs] for k, v in b.items()}, drop, K).numpy())
    return np.concatenate(out)


def run_matrix(model, st, online_lr=None, chunk=8192, drop=None):
    """Code length of the codes of one matrix in coding order; online: predict, then update on that batch."""
    import torch
    K = st["K"]
    n = st["R"] * st["W"]
    m = copy.deepcopy(model)
    opt = torch.optim.Adam(m.parameters(), lr=online_lr) if online_lr else None
    total = 0.0
    chain_bits = st["chain_bits"]          # exact (float64) chain code length of the codes
    for i in range(0, n, chunk):
        idx = np.arange(i, min(n, i + chunk))
        b = to_t(batch_at(st, idx))
        if opt is None:
            with torch.no_grad():
                m.eval()
                total += float(nll_bits(m, b, drop, K).sum())
        else:
            m.train()
            bits = nll_bits(m, b, drop, K)
            total += float(bits.detach().sum())      # prediction made before the update
            opt.zero_grad()
            bits.mean().backward()
            opt.step()
    return total, chain_bits


# ------------------------------------------------------------------ stage: synthetic checks

def synth_matrix(rng, R, W, xor=None, repeat=0.0):
    """Code magnitudes depending on the block scale (structure the chain sees through R2; `repeat` > 0 adds a
    dependence on the previous code, which the chain sees through R3 but learns with a larger learning cost);
    optionally the sign bit = XOR of the sign bits at c - d1 and c - d2 with probability rho (otherwise random)."""
    E = rng.integers(117, 123, size=(R, W // 32)).astype(np.uint8)
    S = np.repeat(E.astype(np.int64), 32, axis=1)
    mag = np.zeros((R, W), dtype=np.int64)
    base = rng.random((R, W))
    for c in range(W):
        lam = 0.8 + 0.4 * (S[:, c] - 117)             # scale -> mean magnitude
        prevm = mag[:, c - 1] if c else np.zeros(R, dtype=np.int64)
        m = np.minimum(7, rng.poisson(lam))
        mag[:, c] = np.where(base[:, c] < repeat, prevm, m)
    sign = rng.integers(0, 2, size=(R, W))
    if xor:
        d1, d2, rho = xor
        for c in range(d2, W):
            s = sign[:, c - d1] ^ sign[:, c - d2]
            sign[:, c] = np.where(rng.random(R) < rho, s, sign[:, c])
    C = mag + 8 * sign
    C = np.where(mag == 0, 0, C)
    return C, E


def synth_stream(C, E, layer=0):
    chain = tx.chain(C, E)
    flags = active(chain)
    sym, p, K, side, scale, _ = chain_probs(C, E, flags)
    R, W = C.shape
    fm = feature_maker(sym.reshape(R, W), E, None, 0, layer, K)
    return {"sym": sym, "logp": np.log(np.maximum(p, 1e-300)).astype(np.float32), "K": K, "side": side, "scale": scale,
            "R": R, "W": W, "fm": fm, "flags": sorted(flags), "chain_L": chain["L"][-1],
            "chain_bits": float(-np.log2(p[np.arange(len(sym)), sym]).sum())}


def stage_synth():
    import torch
    torch.set_num_threads(48)
    rng = np.random.default_rng(11)
    out = {}
    R, W = 512, 4096      # 2 million positions: the chain's learning cost for the R2 contexts is negligible
    for kind, xor in (("V-a", None), ("V-b", (3, 20, 0.9))):
        mats = [synth_matrix(rng, R, W, xor) for _ in range(6)]
        sts = [synth_stream(C, E) for C, E in mats]
        K = sts[0]["K"]
        train = sample_batches(sts[:4], 120000, rng)
        val = sample_batches(sts[4:5], 60000, rng)
        model0 = make_net(K)
        st = sts[5]
        b0, chain_bits = run_matrix(model0, st)
        model, info = fit(train, val, K, epochs=6, log=lambda s: None)
        bo, _ = run_matrix(model, st)
        n = R * W
        rec = {"flags": st["flags"], "chain_code_bits_per_weight": chain_bits / n,
               "init_identity_diff_bits_per_weight": (b0 - chain_bits) / n,
               "offline_gain_bits_per_weight": (chain_bits - bo) / n, "fit": info}
        if xor:
            Cte = mats[5][0]
            nz = (Cte & 7) > 0
            nz[:, :xor[1]] = False
            rho = xor[2] + (1 - xor[2]) / 2      # rule applied with prob rho, otherwise a random sign
            h = -(rho * math.log2(rho) + (1 - rho) * math.log2(1 - rho))
            # the sign of a non-zero code costs 1 bit to the chain (signs are independent of everything it
            # sees); knowing the XOR rule leaves H(rho) -- only when both earlier codes are non-zero
            Sg = Cte >> 3
            both = np.zeros_like(nz)
            both[:, xor[1]:] = ((Cte[:, xor[1] - xor[0]:W - xor[0]] & 7) > 0) & ((Cte[:, :W - xor[1]] & 7) > 0)
            rec["theory_gain_bits_per_weight"] = float((nz & both).sum() * (1 - h) / n)
            rec["note"] = ("spec example: XOR of codes 100 apart; the probe sees only the previous 32 codes of the "
                           "row, so the two codes are at distances 3 and 20 (out of reach of the chain's R3, which "
                           "uses the previous 2)")
        out[kind] = rec
        print(kind, {k: v for k, v in rec.items() if k != "fit"}, flush=True)
    a, b = out["V-a"], out["V-b"]
    out["checks"] = {
        "init_identity": bool(max(abs(a["init_identity_diff_bits_per_weight"]), abs(b["init_identity_diff_bits_per_weight"])) < 1e-4),
        "V-a_gain_about_0": bool(-0.01 < a["offline_gain_bits_per_weight"] < 0.005),
        "V-b_recovers_theory": bool(0.8 * b["theory_gain_bits_per_weight"] < b["offline_gain_bits_per_weight"]
                                    < 1.1 * b["theory_gain_bits_per_weight"])}
    out["all_pass"] = all(out["checks"].values())
    json.dump(out, open(RES / "synthetic.json", "w"), indent=1)
    print(json.dumps(out["checks"]), "all_pass", out["all_pass"])


# ------------------------------------------------------------------ stages on the real sample

def split():
    ex = tx.sample_experts()
    tr = [f"L{L}.E{e}.{r}" for L, es in ex.items() for e in es[:2] for r in ("w1", "w2", "w3")]
    te = [f"L{L}.E{e}.{r}" for L, es in ex.items() for e in es[2:] for r in ("w1", "w2", "w3")]
    return tr, te


def data_one(name, n_train=400000, n_val=40000):
    chain = base_chains()[name]
    st = matrix_stream(name, chain)
    rng = np.random.default_rng([tx.SEED, 11, zlib.crc32(name.encode())])
    idx = rng.choice(st["R"] * st["W"], size=n_train + n_val, replace=False)
    b = batch_at(st, np.sort(idx[:n_train]))
    v = batch_at(st, np.sort(idx[n_train:]))
    np.savez(RES / "data" / f"{name}.npz", **{"t_" + k: x for k, x in b.items()}, **{"v_" + k: x for k, x in v.items()})
    return name


def stage_data(workers=24):
    (RES / "data").mkdir(parents=True, exist_ok=True)
    tr, _ = split()
    with ProcessPoolExecutor(workers) as ex:
        for n in ex.map(data_one, tr):
            print("data", n, flush=True)


def load_data(names):
    t, v = [], []
    for n in names:
        z = np.load(RES / "data" / f"{n}.npz")
        t.append({k[2:]: z[k] for k in z.files if k.startswith("t_")})
        v.append({k[2:]: z[k] for k in z.files if k.startswith("v_")})
    cat = lambda L: {k: np.concatenate([d[k] for d in L]) for k in L[0]}
    return cat(t), cat(v)


def online_one(args):
    name, lr = args
    import torch
    torch.set_num_threads(16)
    model = make_net(15)
    model.load_state_dict(torch.load(RES / "offline_model.pt"))
    st = matrix_stream(name, base_chains()[name])
    bits, chain_bits = run_matrix(model, st, online_lr=lr)
    return {"name": name, "lr": lr, "bits": bits, "chain_bits": chain_bits, "n": st["R"] * st["W"]}


def fit_info_from_log(path):
    """Validation history printed by fit() in the run that saved offline_model.pt (used after the online-lr
    step of that run deadlocked: workers forked after torch had started its thread pool)."""
    hist = {}
    for line in open(path):
        m = re.match(r"\s+lr (\S+) epoch (\d+): validation (\S+) bit/code \(chain (\S+)\)", line)
        if m:
            h = hist.setdefault(m[1], [float(m[4])])
            h.append(float(m[3]))
    best = min(((min(h), lr) for lr, h in hist.items()))
    return {"lr": float(best[1]), "validation_bits_per_epoch": hist, "epoch0_is_chain": True,
            "source": "read back from results/p1b/run_train.log (same run as offline_model.pt)"}


SPAWN = multiprocessing.get_context("spawn")    # torch thread pools do not survive fork


def stage_train():
    import torch
    tr, _ = split()
    if (RES / "offline_model.pt").exists() and (RES / "run_train.log").exists() and not (RES / "fit_info.json").exists():
        info = fit_info_from_log(RES / "run_train.log")
        model = make_net(15)
        model.load_state_dict(torch.load(RES / "offline_model.pt"))
    elif (RES / "fit_info.json").exists():
        info = json.load(open(RES / "fit_info.json"))
        model = make_net(15)
        model.load_state_dict(torch.load(RES / "offline_model.pt"))
    else:
        train, val = load_data(tr)
        print("training positions", len(train["y"]), flush=True)
        model, info = fit(train, val, 15, epochs=4)
        torch.save(model.state_dict(), RES / "offline_model.pt")
    json.dump(info, open(RES / "fit_info.json", "w"), indent=1)
    info["n_params"] = n_params(model)
    # online learning rate: chosen on 2 training matrices only (never on the test matrices)
    cand = [(n, lr) for n in (tr[0], tr[13]) for lr in (1e-4, 3e-4)]
    with ProcessPoolExecutor(4, mp_context=SPAWN) as ex:
        rows = list(ex.map(online_one, cand))
    by = {}
    for r in rows:
        by.setdefault(r["lr"], 0.0)
        by[r["lr"]] += r["bits"] - r["chain_bits"]
    info["online_lr_choice"] = {"rows": rows, "chosen": min(by, key=by.get)}
    json.dump(info, open(RES / "train.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != "validation_bits_per_epoch"}, indent=1), flush=True)


def test_one(args):
    name, lr = args
    import torch
    torch.set_num_threads(4)
    t0 = time.time()
    model = make_net(15)
    model.load_state_dict(torch.load(RES / "offline_model.pt"))
    chain = base_chains()[name]
    st = matrix_stream(name, chain)
    off, chain_bits = run_matrix(model, st)
    on, _ = run_matrix(model, st, online_lr=lr)
    return {"task": name, "n": st["R"] * st["W"], "chain_code_bits": chain_bits, "offline_bits": off, "online_bits": on,
            "chain_total_008": chain["L"][-1], "wall_s": time.time() - t0}


def stage_test(workers=24):
    _, te = split()
    lr = json.load(open(RES / "train.json"))["online_lr_choice"]["chosen"]
    path = RES / "test.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(l)["task"] for l in open(path)}
    todo = [(n, lr) for n in te if n not in done]
    with ProcessPoolExecutor(workers, mp_context=SPAWN) as ex, open(path, "a") as f:
        for r in ex.map(test_one, todo):
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(r["task"], {k: round(v / r["n"], 5) for k, v in r.items() if k.endswith("bits")}, flush=True)


def ablate_train_one(drop):
    """Retrain the offline model with one feature group removed (set to a constant in training and test)."""
    import torch
    tr, _ = split()
    train, val = load_data(tr)
    lr = json.load(open(RES / "train.json"))["lr"]
    model, info = fit(train, val, 15, epochs=4, lrs=(lr,), log=lambda s: None, threads=16, drop=drop)
    torch.save(model.state_dict(), RES / "ablate" / f"{drop}.pt")
    return drop, info


def ablate_eval_one(name):
    import torch
    torch.set_num_threads(4)
    st = matrix_stream(name, base_chains()[name])
    out = {"task": name, "n": st["R"] * st["W"], "chain_bits": st["chain_bits"], "bits": {}}
    for drop in ABLATE:
        model = make_net(15)
        model.load_state_dict(torch.load(RES / "ablate" / f"{drop}.pt"))
        out["bits"][drop] = run_matrix(model, st, drop=drop)[0]
    return out


def stage_ablate(workers=24):
    (RES / "ablate").mkdir(exist_ok=True)
    todo = [d for d in ABLATE if not (RES / "ablate" / f"{d}.pt").exists()]
    infos = {}
    with ProcessPoolExecutor(len(ABLATE), mp_context=SPAWN) as ex:
        for d, info in ex.map(ablate_train_one, todo):
            infos[d] = info
            print("trained without", d, info["validation_bits_per_epoch"], flush=True)
    if infos:
        json.dump(infos, open(RES / "ablate" / "fit.json", "w"), indent=1)
    _, te = split()
    with ProcessPoolExecutor(workers, mp_context=SPAWN) as ex, open(RES / "ablate.jsonl", "w") as f:
        for r in ex.map(ablate_eval_one, te):
            f.write(json.dumps(r) + "\n")
            f.flush()
            print("ablation evaluated", r["task"], flush=True)


def main():
    RES.mkdir(parents=True, exist_ok=True)
    stage = sys.argv[1]
    {"verify": stage_verify, "synth": stage_synth, "data": stage_data, "train": stage_train, "test": stage_test,
     "ablate": stage_ablate}[stage]()


if __name__ == "__main__":
    main()
