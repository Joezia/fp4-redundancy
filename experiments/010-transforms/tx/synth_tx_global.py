"""TX stage 2 synthetic checks: V-a (no structure) -> appearance <= 0.01 bit/value, V-b (targeted structure) ->
appearance >= 0.05 bit/value with the best coder."""
import numpy as np

import txcore as T
import tx_global as G
import synth_tx_local as SL

R, W = 256, 1024


def group_check(fn, v, mk_a, mk_b, base, tol=0.01, need=0.05):
    res = {}
    for tag, mk in (("V-a", mk_a), ("V-b", mk_b)):
        ms = mk()
        encs = fn(ms, v)
        n = sum(m.L.size for m in ms)
        rt = True
        tot = {"C0": 0.0, "C1": 0.0}
        raw = {"C0": 0.0, "C1": 0.0}
        side = 0.0
        for m, enc in zip(ms, encs):
            e = T.evaluate(enc)
            rt &= e["roundtrip"]
            side += e["side_bits_C1"]
            r = T.synth_raw(m, "code" if base == "code" else "value")[base]
            for c in ("C0", "C1"):
                tot[c] += e[c]
                raw[c] += r[c]
        res[tag] = {"n": n, "roundtrip": bool(rt), "side_per_value": side / n,
                    "appear_C0": (raw["C0"] - tot["C0"]) / n, "appear_C1": (raw["C1"] - tot["C1"]) / n}
    va, vb = res["V-a"], res["V-b"]
    res["pass_V-a"] = bool(max(va["appear_C0"], va["appear_C1"]) <= tol and va["roundtrip"])
    res["pass_V-b"] = bool(max(vb["appear_C0"], vb["appear_C1"]) >= need and vb["roundtrip"])
    res["pass"] = res["pass_V-a"] and res["pass_V-b"]
    res["criteria"] = {"V-a_max": tol, "V-b_min": need, "base_view": base}
    res["theory_V-b_per_value"] = None
    return res


def iid_group(n=8, seed=0):
    return lambda: [T.synth_mat(L=SL.iid_levels(seed + i, (R, W))) for i in range(n)]


def common_plus_noise(n=8, seed=20):
    """experts = one common matrix + small independent noise (position-wise correlated experts)."""
    def mk():
        rng = np.random.default_rng(seed)
        base = rng.integers(-40, 41, size=(R, W))
        return [T.synth_mat(L=base + rng.integers(-1, 2, size=(R, W))) for _ in range(n)]
    return mk


def shared_lowrank(n=8, k=8, seed=21):
    """every expert's lines lie near one shared k-dim subspace of the line space."""
    def mk():
        rng = np.random.default_rng(seed)
        B = rng.normal(size=(k, R))                      # lines are columns (length R) for this width
        out = []
        for _ in range(n):
            Cf = rng.normal(0, 8, size=(W, k))
            out.append(T.synth_mat(L=np.round((Cf @ B).T + rng.normal(0, 0.5, size=(R, W))).astype(np.int64)))
        return out
    return mk


def tucker_lowrank(n=8, ranks=(2, 16, 16), seed=22):
    def mk():
        rng = np.random.default_rng(seed)
        U0 = rng.normal(size=(n, ranks[0]))
        U1 = rng.normal(size=(R, ranks[1]))
        U2 = rng.normal(size=(W, ranks[2]))
        core = rng.normal(0, 3, size=ranks)
        Xt = np.einsum("ea,ib,jc,abc->eij", U0, U1, U2, core)
        return [T.synth_mat(L=np.round(Xt[e] + rng.normal(0, 0.5, size=(R, W))).astype(np.int64)) for e in range(n)]
    return mk


def sparse_mixture(d=32, seed=23, n=1, unit="block", shape=(R, W)):
    """vectors = A s with sparse integer sources s (90 % zero) and one fixed random mixing matrix A."""
    R, W = shape

    def mk():
        rng = np.random.default_rng(seed)
        A = rng.normal(size=(d, d))
        out = []
        for i in range(n):
            S = np.where(rng.random((R * W // d, d)) < 0.1, rng.integers(-20, 21, size=(R * W // d, d)), 0)
            Xv = np.round(S @ A.T).astype(np.int64)
            out.append(T.synth_mat(L=Xv.reshape(R, W)))
        return out
    return mk


def proto_rows_codes(seed=24, nproto=8, shuffled=True):
    """rows copied from 8 prototype rows with 10 % of the codes replaced, in shuffled row order."""
    rng = np.random.default_rng(seed)
    P = rng.integers(0, 16, size=(nproto, W))
    lab = rng.integers(0, nproto, size=R)
    C = P[lab].copy()
    rep = rng.random(C.shape) < 0.1
    C[rep] = rng.integers(0, 16, size=rep.sum())
    E = np.full((R, W // 32), 120, dtype=np.uint8)
    return T.Mat("synthetic", C, E)


def iid_codes(seed=25):
    rng = np.random.default_rng(seed)
    return T.Mat("synthetic", rng.integers(0, 16, size=(R, W)), np.full((R, W // 32), 120, dtype=np.uint8))


def chain_experts(n=8, seed=26, p=0.1):
    """expert e = expert (e - 1) with p of the codes replaced; experts given in shuffled order."""
    def mk():
        rng = np.random.default_rng(seed)
        s = SL.iid_syms(seed, (R, W))
        seq = [s]
        for _ in range(n - 1):
            t = seq[-1].copy()
            rep = rng.random(t.shape) < p
            t[rep] = SL.iid_syms(seed + 100 + len(seq), (R, W))[rep]
            seq.append(t)
        order = rng.permutation(n)
        return [T.synth_mat(sym=seq[i]) for i in order]
    return mk


def iid_sym_group(n=8, seed=27):
    return lambda: [T.synth_mat(sym=SL.iid_syms(seed + i, (R, W))) for i in range(n)]


def word_rows(seed=28, nwords=4, wl=32):
    """rows built from 4 random words of 32 symbols over a 4-letter alphabet: long repeated contexts inside every
    row (BWT's target), which an order-2 context cannot resolve."""
    rng = np.random.default_rng(seed)
    voc = rng.integers(0, 4, size=(nwords, wl))
    idx = rng.integers(0, nwords, size=(R, W // wl))
    return voc[idx].reshape(R, W)


def checks():
    out = {}
    C = T.synth_check
    for q in (1, 2):
        out[f"A6_klt expert_axis q={q}"] = group_check(G.A6_klt, dict(kind="expert_axis", aligned=False, q=q),
                                                     iid_group(), common_plus_noise(), "aligned")
    out["A6_klt subspace k=8 q=2"] = group_check(G.A6_klt, dict(kind="subspace", k=8, q=2), iid_group(),
                                                shared_lowrank(), "aligned")
    out["A6_klt subspace k=32 q=2"] = group_check(G.A6_klt, dict(kind="subspace", k=32, q=2), iid_group(),
                                                 shared_lowrank(), "aligned")
    out["A7_ica block q=0.25 (single)"] = C(G.A7_ica, dict(unit="block", q=0.25), "value",
                                            lambda: T.synth_mat(L=SL.iid_levels(0, (R, W))),
                                            lambda: sparse_mixture()()[0])
    out["A7_ica seg q=0.25 (single)"] = C(G.A7_ica, dict(unit="seg", q=0.25), "value",
                                          lambda: T.synth_mat(L=SL.iid_levels(0, (2048, 2048))),
                                          lambda: sparse_mixture(d=256, unit="seg", shape=(2048, 2048))()[0], base="aligned")
    out["A7_ica_shared block q=0.25"] = group_check(G.A7_ica_shared, dict(unit="block", q=0.25), iid_group(4),
                                                   sparse_mixture(n=4), "level")
    out["A8_tucker (2,32,32)"] = group_check(G.A8_tucker, dict(ranks=(2, 32, 32)), iid_group(), tucker_lowrank(), "aligned")
    for o in ("spectral", "tsp", "hclust"):
        for c in ("code", "b1"):
            out[f"C3_seriation {o} {c}"] = C(G.C3_seriation, dict(order=o, coding=c), "code" if c == "code" else "value",
                                             iid_codes, lambda: proto_rows_codes(),
                                             base="code" if c == "code" else "aligned")
    for o in ("index", "similar"):
        out[f"C3_expert_order {o}"] = group_check(G.C3_expert_order, dict(order=o, aligned=False), iid_sym_group(),
                                                  chain_experts(), "code")
    for u in ("row", "matrix"):
        for o in ("bwt", "bwt_mtf"):
            out[f"C4_bwt {u} {o}"] = C(G.C4_bwt, dict(unit=u, out=o), "code", lambda: T.synth_mat(sym=SL.iid_syms(0, (R, W))),
                                       lambda: T.synth_mat(sym=word_rows()), base="code_seq")
    return out
