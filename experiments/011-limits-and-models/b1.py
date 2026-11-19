"""Spec 011 B1: parametric discrete distributions (latent width -> 15 probabilities on the E2M1 rounding cells).

    PYTHONPATH=. python experiments/011-limits-and-models/b1.py synth
    PYTHONPATH=. python experiments/011-limits-and-models/b1.py run

A code is the rounding of a continuous value u (weight / block scale) to the E2M1 grid; with u ~ Laplace(0, b) or
N(0, b^2), the 15 probabilities are the masses of the rounding cells (|u| edges 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5)
split equally by sign (zero merged). One table per width level (64 levels, log-spaced): decoding is one lookup.

Measured here (bit per code, T1 and T2):
  bound    : each block uses its own best width level and the level costs nothing -- an upper bound of anything
             B1a (causal width) or B1b (sent width) can reach with this family, since both choose one width per
             block or less (B1a per weight from earlier codes; its gain is below this bound as well).
  row      : one width sigma per row (sent, 8 bit per row), block width = sigma / block scale.
  B1b-vq   : the parametric family replaced by 16 tables learned on the training pool (multinomial mixture, EM);
             each block sends its table index, coded adaptively with the previous block's index as context.
             Tables billed at 15 x 12 bit each.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bfeat as BF  # noqa: E402

RES = HERE / "results" / "b"
EDGES = np.array([0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, np.inf])
LEVELS = 2.0 ** (np.arange(-24, 41) / 8.0)        # widths 0.125 .. 32
ALL_W = 557171343360


def table(b, fam):
    from scipy.special import erf
    cdf = 1 - np.exp(-EDGES / b) if fam == "laplace" else erf(EDGES / (b * math.sqrt(2)))
    pm = np.maximum(np.diff(cdf), 1e-12)
    p = np.zeros(15)
    p[0] = pm[0]
    p[1:8] = pm[1:] / 2
    p[8:15] = pm[1:] / 2
    return np.log2(p / p.sum())


TABS = {f: np.stack([table(b, f) for b in LEVELS]) for f in ("laplace", "gauss")}


def block_hist(S):
    R, W = S.shape
    blk = S.reshape(R, W // 32, 32)
    oh = np.zeros((R, W // 32, 15))
    for k in range(15):
        oh[..., k] = (blk == k).sum(-1)
    return oh


def bound_and_row(S, E):
    oh = block_hist(S)
    out = {}
    for fam, T in TABS.items():
        cost = -(oh @ T.T)                                    # (R, NB, L)
        out[f"bound_{fam}"] = float(cost.min(-1).sum())
        # one sigma per row: block level index = s + 8 * (127 - E) on the 1/8-octave grid
        R, NB, L = cost.shape
        tot = 0.0
        for r in range(R):
            base = 8 * (127 - E[r].astype(np.int64)) + 24
            ss = np.arange(-int(base.max()), L - int(base.min()))
            idx = ss[:, None] + base[None, :]
            c = np.where((idx >= 0) & (idx < L), cost[r][np.arange(NB)[None, :], np.clip(idx, 0, L - 1)], 1e9).sum(1)
            tot += c.min() + 8.0                                # sigma level sent with 8 bit
        out[f"row_{fam}"] = float(tot)
    return out


def vq_bits(S, L):
    """codes with the best of the learned tables per block + the index, adaptive with the previous index as context."""
    import common as K
    from redundancy import increment as I
    oh = block_hist(S)
    ll = oh @ L.T                                             # (R, NB, Kc) log2 likelihood
    lab = ll.argmax(-1)
    codes = float(-ll.max(-1).sum())
    prev = np.full(lab.shape, L.shape[0], dtype=np.int64)
    prev[:, 1:] = lab[:, :-1]
    side = I.dirichlet_bits(lab.ravel(), prev.ravel(), L.shape[0])
    return codes, float(side)


def one(args):
    name, Lvq = args
    M = BF.Matrices()
    S, E = M.codes(name)
    out = {"task": name, "n": int(S.size), **bound_and_row(S, E)}
    c, s = vq_bits(S, Lvq)
    out["vq_codes"], out["vq_side"] = c, s
    return out


def _hists(name, per=3000):
    M = BF.Matrices()
    S, _ = M.codes(name)
    h = block_hist(S).reshape(-1, 15)
    return h[np.random.default_rng(0).choice(len(h), per, replace=False)]


def learn_vq(Kc=16, iters=30):
    import common as K
    with ProcessPoolExecutor(24) as ex:
        H = np.concatenate(list(ex.map(_hists, K.pool_round(1))))
    rng = np.random.default_rng(0)
    tabs = H[rng.choice(len(H), Kc, replace=False)] + 0.5
    tabs /= tabs.sum(1, keepdims=True)
    for _ in range(iters):
        lab = (H @ np.log2(tabs).T).argmax(1)
        for k in range(Kc):
            s = H[lab == k].sum(0) + 0.5
            tabs[k] = s / s.sum()
    q = np.maximum(1, np.round(tabs * 4096))
    q[np.arange(Kc), q.argmax(1)] += 4096 - q.sum(1)
    return np.log2(q / 4096)


def stage_run():
    import common as K
    Lvq = learn_vq()
    with ProcessPoolExecutor(48) as ex:
        rows = list(ex.map(one, [(n, Lvq) for n in K.T1 + K.T2]))
    chain = {json.loads(l)["name"]: json.loads(l)["chain_code_bits"] for l in open(BF.D / "build_info.jsonl")}
    out = {"vq_table_cost_bits": 16 * 15 * 12}
    for split, ns in (("T1", K.T1), ("T2", K.T2)):
        sel = [r for r in rows if r["task"] in ns]
        n = sum(r["n"] for r in sel)
        ch = sum(chain[r["task"]] for r in sel)
        res = {"chain_bits_per_code": ch / n}
        for k in ("bound_laplace", "bound_gauss", "row_laplace", "row_gauss"):
            res[k + "_gain"] = (ch - sum(r[k] for r in sel)) / n
        vq = sum(r["vq_codes"] + r["vq_side"] for r in sel)
        res["vq_gain"] = (ch - vq) / n
        res["vq_net_cost_over_test"] = (ch - vq - out["vq_table_cost_bits"]) / n
        res["vq_side_bits_per_code"] = sum(r["vq_side"] for r in sel) / n
        res["vq_codes_only_gain"] = (ch - sum(r["vq_codes"] for r in sel)) / n
        out[split] = res
    json.dump(out, open(RES / "b1.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


def stage_synth():
    """V-a: codes from Laplace(1) (the chain sees everything): the parametric table with the right width is as good
    as the chain (gain about 0). V-b: each block draws b from {0.5, 1, 2, 4} (a hidden block parameter the chain
    cannot see); B1b with that level set (2 bit per block, per-block choice) should recover about
    (chain - exact code length given b - 2 bit per block)."""
    import common as K
    sys.path.insert(0, str(K.ROOT / "experiments" / "010-transforms"))
    import p1b
    out = {}
    for kind in ("V-a", "V-b"):
        rng = np.random.default_rng(81)
        R, W = 1024, 4096
        bs = np.full((R, W // 32), 1.0) if kind == "V-a" else rng.choice([0.5, 1.0, 2.0, 4.0], size=(R, W // 32))
        u = rng.laplace(0, np.repeat(bs, 32, axis=1))
        a = np.abs(u)
        mag = np.searchsorted(EDGES[1:-1], a, side="right")       # 0..7
        sgn = (u < 0).astype(np.int64)
        C = np.where(mag == 0, 0, mag + 8 * sgn)
        E = np.full((R, W // 32), 120, dtype=np.uint8)
        ch = K.tx.chain(C, E)
        sym, p, Kk, side, scale, _ = p1b.chain_probs(C, E, p1b.active(ch))
        chain_bits = float(-np.log2(p[np.arange(len(sym)), sym]).sum())
        S = sym.reshape(R, W)
        oh = block_hist(S)
        levels = [1.0] if kind == "V-a" else [0.5, 1.0, 2.0, 4.0]
        Tl = np.stack([table(b, "laplace") for b in (0.5, 1.0, 2.0, 4.0)])
        true_bits = float(-(oh * Tl[np.searchsorted([0.5, 1.0, 2.0, 4.0], bs)]).sum())
        # B1b with the generator's level set: each block picks its level, sent with log2(#levels) bit
        Tb = np.stack([table(b, "laplace") for b in levels])
        bnd = (-(oh @ Tb.T)).min(-1).sum()
        n = C.size
        side_bits = math.log2(len(levels)) * bs.size
        out[kind] = {"b1b_gain": (chain_bits - bnd - side_bits) / n,
                     "theory_gain": (chain_bits - true_bits - (2.0 * bs.size if kind == "V-b" else 0)) / n,
                     "flags": sorted(p1b.active(ch))}
        print(kind, out[kind], flush=True)
    out["checks"] = {"V-a_about_0": bool(-0.01 < out["V-a"]["b1b_gain"] < 0.005),
                     "V-b_recovers": bool(0.8 * out["V-b"]["theory_gain"] < out["V-b"]["b1b_gain"]
                                          <= out["V-b"]["theory_gain"] + 0.01)}
    out["all_pass"] = all(out["checks"].values())
    json.dump(out, open(RES / "synthetic_b1.json", "w"), indent=1)
    print(out["checks"], out["all_pass"])


if __name__ == "__main__":
    RES.mkdir(parents=True, exist_ok=True)
    {"run": stage_run, "synth": stage_synth}[sys.argv[1]]()
