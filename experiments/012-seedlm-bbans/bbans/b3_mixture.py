"""B3 (extension): BB-ANS with a discrete latent -- the 005 block latent-class mixture.

Model (docs/005, section 3 "B"): for every scale value E a separate mixture; a block belongs to class z with
probability pi_z, then its 32 codes are i.i.d. from theta_z (15 probabilities). Fitted by EM exactly as 005:
input = per-block 15-bin histograms; E-step r_bz ~ pi_z prod_c theta_z(c)^n_bc; M-step pi = mean r,
theta_z ~ sum_b r_bz n_b + 0.5 (the 0.5 only on codes seen under that E); 150 iterations; three starts
(blocks grouped by quantiles of the block mean |v|; the same grouping with the class labels shuffled; random
responsibilities), the one with the highest training likelihood kept; an E with fewer than 500*K blocks uses
the largest K' in {1,2,4,...} with 500*K' blocks or more. Model cost sum_E K_E * 16 * 16 bits (005).

Here the mixture is fitted on the whole matrix that is coded (the model is side information), and the blocks are
coded with BB-ANS: pop z from the exact posterior r(z | block), push the 32 codes with theta_z, push z with pi_E.
Single chain with the official bits-back util (bb_ans_append / categoricals_*); N streams with Craystack.

usage: python b3_mixture.py MATRIX K [N,N,...]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fp4vae as V  # noqa: E402

sys.path.insert(0, os.path.join(V.ROOT, "data/012/src/bits-back_bits-back"))
import rans  # noqa: E402
import util  # noqa: E402

OUT = os.path.join(HERE, "../results/b3")
PREC = 14                     # posterior, prior and observation precision (official observation precision)
N_OTHER = 50
ABSV = np.abs(V.LEVEL)


def _em(n, K, init_r, iters=150):
    """n: (nb, 15) histograms. Returns pi (K,), theta (K, 15), loglik (nats)."""
    seen = n.sum(0) > 0
    r = init_r
    for _ in range(iters):
        pi = r.mean(0)
        th = r.T @ n + 0.5 * seen[None, :]
        th = th / th.sum(1, keepdims=True)
        lt = np.log(np.where(th > 0, th, 1e-300))
        a = n @ lt.T + np.log(np.maximum(pi, 1e-300))[None, :]
        m = a.max(1, keepdims=True)
        p = np.exp(a - m)
        s = p.sum(1, keepdims=True)
        r = p / s
    pi = r.mean(0)
    th = r.T @ n + 0.5 * seen[None, :]
    th = th / th.sum(1, keepdims=True)
    lt = np.log(np.where(th > 0, th, 1e-300))
    a = n @ lt.T + np.log(np.maximum(pi, 1e-300))[None, :]
    m = a.max(1, keepdims=True)
    ll = float((m[:, 0] + np.log(np.exp(a - m).sum(1))).sum())
    return pi, th, ll


def fit_E(n, mabs, K, rng):
    nb = len(n)
    if K == 1:
        return _em(n, 1, np.ones((nb, 1)))
    q = np.quantile(mabs, np.linspace(0, 1, K + 1)[1:-1])
    g = np.searchsorted(q, mabs, side="right")
    best = None
    for start in range(3):
        if start == 0:
            r = np.eye(K)[g]
        elif start == 1:
            r = np.eye(K)[rng.permutation(K)[g]]
        else:
            r = rng.dirichlet(np.ones(K), size=nb)
        out = _em(n, K, r)
        if best is None or out[2] > best[2]:
            best = out
    return best


def fit(sym, E, K, seed=0):
    rng = np.random.default_rng(seed)
    n = np.zeros((len(sym), V.NSYM))
    np.add.at(n, (np.repeat(np.arange(len(sym)), V.B), sym.ravel().astype(np.int64)), 1)
    mabs = ABSV[sym].mean(1)
    model = {}
    for e in np.unique(E):
        idx = np.nonzero(E == e)[0]
        k = K
        while k > 1 and len(idx) < 500 * k:
            k //= 2
        pi, th, ll = fit_E(n[idx], mabs[idx], k, rng)
        model[int(e)] = (pi, th)
    return model, n


def marginal_bits(model, n, E):
    tot = 0.0
    for e, (pi, th) in model.items():
        idx = E == e
        lt = np.log2(np.where(th > 0, th, 1e-300))
        a = n[idx] @ lt.T + np.log2(np.maximum(pi, 1e-300))[None, :]
        m = a.max(1, keepdims=True)
        tot += float(-(m[:, 0] + np.log2(np.exp2(a - m).sum(1))).sum())
    return tot


def posterior(model, n_b, e):
    pi, th = model[e]
    lt = np.log(np.where(th > 0, th, 1e-300))
    a = n_b @ lt.T + np.log(np.maximum(pi, 1e-300))
    p = np.exp(a - a.max())
    return p / p.sum()


def single_chain(model, sym, E, n):
    def codec(b_e, n_b=None):
        pi, th = model[b_e]

        def post_pop(data):
            hist = np.bincount(np.asarray(data, dtype=np.int64), minlength=V.NSYM).astype(float)
            r = posterior(model, hist, b_e)
            return util.categoricals_pop(r[None, :], PREC)

        def lik_append(z):
            return util.categoricals_append(np.repeat(th[int(np.ravel(z)[0])][None, :], V.B, 0), PREC)

        def lik_pop(z):
            return util.categoricals_pop(np.repeat(th[int(np.ravel(z)[0])][None, :], V.B, 0), PREC)

        def post_append(data):
            hist = np.bincount(np.asarray(data, dtype=np.int64), minlength=V.NSYM).astype(float)
            r = posterior(model, hist, b_e)
            return util.categoricals_append(r[None, :], PREC)

        prior_app = util.categoricals_append(pi[None, :], PREC)
        prior_pop = util.categoricals_pop(pi[None, :], PREC)
        return (util.bb_ans_append(post_pop, lik_append, prior_app),
                util.bb_ans_pop(prior_pop, lik_pop, post_append))

    rng = np.random.RandomState(0)
    other = rng.randint(low=1 << 16, high=1 << 31, size=N_OTHER, dtype=np.uint32)
    state = rans.unflatten(other)
    nb = len(sym)
    t0 = time.time()
    for b in range(nb):
        app, _ = codec(int(E[b]))
        state = app(state, sym[b].astype(np.int64))
    t_enc = time.time() - t0
    flat = rans.flatten(state)
    state = rans.unflatten(flat)
    ok = True
    t0 = time.time()
    for k in range(nb):
        b = nb - 1 - k
        _, pop = codec(int(E[b]))
        state, x = pop(state)
        if not np.array_equal(np.asarray(x).ravel(), sym[b].astype(np.int64)):
            ok = False
            break
    t_dec = time.time() - t0
    return dict(N=1, impl="official bits-back", bits_gross=32 * len(flat), bits_net=32 * (len(flat) - N_OTHER),
                roundtrip_ok=ok, other_bits_recovered=bool(np.array_equal(rans.flatten(state), other)),
                encode_s=t_enc, decode_s=t_dec, decode_codes_per_s=nb * V.B / t_dec)


class MixAdapter:
    """Vectorised streams need one class count for all blocks of a step: E values with K_E < K_max are padded with
    zero-probability classes (theta copied from class 0). Craystack gives each zero bucket frequency 1 (of 2^14), so
    a padded class can in principle be drawn from the posterior; encoder and decoder then use the same padded theta."""

    def __init__(self, model):
        kmax = max(len(p) for p, _ in model.values())
        self.model = {}
        for e, (pi, th) in model.items():
            k = len(pi)
            self.model[e] = (np.concatenate([pi, np.zeros(kmax - k)]),
                             np.concatenate([th, np.repeat(th[:1], kmax - k, 0)]))
        self.cs = None

    def _cs(self):
        if self.cs is None:
            from streams import cs
            self.cs = cs
        return self.cs

    def post_codecs(self, X, s, E):
        cs = self._cs()
        R = np.stack([posterior(self.model, np.bincount(x, minlength=V.NSYM).astype(float), int(e))
                      for x, e in zip(X, E)])
        return [cs.Categorical(R, PREC)]

    def prior_codecs(self, s, E):
        cs = self._cs()
        return [cs.Categorical(np.stack([self.model[int(e)][0] for e in E]), PREC)]

    def lik_probs(self, Z, s, E):
        th = np.stack([self.model[int(e)][1][int(z)] for z, e in zip(Z[:, 0], E)])
        return np.repeat(th[:, None, :], V.B, 1)


def main():
    name, K = sys.argv[1], int(sys.argv[2])
    Ns = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else []
    os.makedirs(OUT, exist_ok=True)
    sym, E, _ = V.matrix_blocks(name)
    t0 = time.time()
    model, n = fit(sym, E, K)
    t_fit = time.time() - t0
    model1, _ = fit(sym, E, 1)
    k_used = {e: len(p) for e, (p, _) in model.items()}
    cost = sum(k_used.values()) * 16 * 16
    res = dict(matrix=name, K=K, n_blocks=len(sym), n_weights=len(sym) * V.B, K_per_E=k_used, model_cost_bits=cost,
               marginal_bits=marginal_bits(model, n, E), k1_bits=marginal_bits(model1, n, E),
               k1_cost_bits=len(model1) * 16 * 16, fit_s=t_fit, runs=[])
    if 1 in Ns or not Ns:
        res["runs"].append(single_chain(model, sym, E, n))
        print(json.dumps(res["runs"][-1]), flush=True)
    import streams as ST
    for N in [x for x in Ns if x != 1]:
        r = ST.run(MixAdapter(model), sym, np.zeros(len(sym), np.float32), E.astype(np.int64), N)
        r["impl"] = "craystack"
        res["runs"].append(r)
        print(json.dumps(r), flush=True)
    sys.path.insert(0, os.path.join(V.ROOT, "experiments/011-limits-and-models"))
    import common
    suffix = "_T2" if name in common.T2 else ""
    with open(os.path.join(OUT, f"b3_K{K}{suffix}.jsonl"), "a") as f:
        f.write(json.dumps(res) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "runs"}), flush=True)


if __name__ == "__main__":
    main()
