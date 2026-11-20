"""TX stage 1 synthetic checks (docs/010-变换全集实验.md 1.4): V-a iid data -> appearance <= 0.01 bit/value,
V-b the structure each transform targets -> appearance >= 0.05 bit/value (best coder)."""
import math

import numpy as np

import txcore as T
import tx_local as X

R, W = 512, 2048
P_MAG = np.array([0.14, 0.25, 0.20, 0.14, 0.12, 0.09, 0.04, 0.02])


def iid_levels(seed=0, shape=(R, W)):
    rng = np.random.default_rng(seed)
    mag = T.MAG2[rng.choice(8, size=shape, p=P_MAG)]
    return np.where(rng.random(shape) < 0.5, -mag, mag)


def iid_syms(seed=0, shape=(R, W)):
    return T.merge(T.codes_from_levels(iid_levels(seed, shape)))


def smooth(seed=1, shape=(R, W), amp=30.0, noise=1.0):
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(shape[0]) / shape[0], np.arange(shape[1]) / shape[1], indexing="ij")
    f = np.zeros(shape)
    for _ in range(4):
        fy, fx = rng.integers(0, 4, size=2)
        f += rng.normal() * np.cos(2 * math.pi * (fy * yy + fx * xx) + rng.random() * 6.28)
    return np.round(amp * f + rng.normal(0, noise, shape)).astype(np.int64)


def sinusoids(seed=2, shape=(R, W)):
    rng = np.random.default_rng(seed)
    c = np.arange(shape[1])
    rows = []
    for r in range(shape[0]):
        f = rng.integers(1, shape[1] // 2, size=3)
        a = rng.normal(0, 20, size=3)
        ph = rng.random(3) * 6.28
        rows.append(sum(a[i] * np.cos(2 * math.pi * f[i] * c / shape[1] + ph[i]) for i in range(3)))
    return np.round(np.array(rows)).astype(np.int64)


def ramps(seed=3, shape=(R, W)):
    """per 32-block linear ramps (random offset and slope) + small noise."""
    rng = np.random.default_rng(seed)
    g = shape[1] // 32
    off = rng.integers(-20, 21, size=(shape[0], g, 1))
    sl = rng.normal(0, 1.0, size=(shape[0], g, 1))
    t = np.arange(32) - 15.5
    v = off + sl * t + rng.normal(0, 0.7, size=(shape[0], g, 32))
    return np.round(v).reshape(shape).astype(np.int64)


def copy_left_syms(seed=4, p=0.9):
    rng = np.random.default_rng(seed)
    s = iid_syms(seed)
    cp = rng.random(s.shape) < p
    for c in range(1, s.shape[1]):
        s[:, c] = np.where(cp[:, c], s[:, c - 1], s[:, c])
    return s


def column_syms(seed=5, p=0.9):
    """every column has a value; each code copies it with prob p (structure only along columns)."""
    rng = np.random.default_rng(seed)
    colv = iid_syms(seed, (1, W))[0]
    s = iid_syms(seed + 1)
    return np.where(rng.random(s.shape) < p, colv[None, :], s)


def block_vertical_syms(seed=6, p=0.9):
    """each 32-block copies the block above with prob p."""
    rng = np.random.default_rng(seed)
    s = iid_syms(seed)
    g = W // 32
    G = s.reshape(R, g, 32)
    cp = rng.random((R, g)) < p
    for r in range(1, R):
        G[r] = np.where(cp[r][:, None], G[r - 1], G[r])
    return G.reshape(R, W)


def column_type_blocks(seed=12):
    """block columns of two types (all codes 0, or iid non-zero codes): the block above has the same type,
    the block to the left a random one. Only the first codes of a block can profit from a vertical neighbour."""
    rng = np.random.default_rng(seed)
    g = W // 32
    typ = rng.random(g) < 0.5
    nzs = rng.integers(1, 15, size=(R, W))
    return np.where(np.repeat(typ, 32)[None, :], 0, nzs)


def block_patch_types(seed=13, P=4, values=False):
    """2-D patches of P x P blocks share a type (all-zero codes / iid non-zero codes; for values: small / large
    magnitude). Row-major order changes type every P blocks, a space-filling curve about every P*P blocks."""
    rng = np.random.default_rng(seed)
    g = W // 32
    typ = rng.random((R // P + 1, g // P + 1)) < 0.5
    tb = np.repeat(np.repeat(typ, P, axis=0)[:R], P, axis=1)[:, :g]
    t = np.repeat(tb, 32, axis=1)
    if values:
        return np.where(t, rng.integers(-1, 2, size=(R, W)), rng.integers(20, 40, size=(R, W)))
    return np.where(t, 0, rng.integers(1, 15, size=(R, W)))


def column_values(seed=14):
    rng = np.random.default_rng(seed)
    return rng.integers(-30, 31, size=(1, W)) + rng.integers(-1, 2, size=(R, W))


def signed_smooth(seed=1):
    return smooth(seed) * np.tile(X.signs(32), W // 32)[None, :]


def sign_runs_syms(seed=7, shape=(128, 512)):
    rng = np.random.default_rng(seed)
    lv = np.abs(iid_levels(seed, shape))
    sg = np.sign(np.cumsum(rng.random(shape) < 0.02, axis=1) % 2 - 0.5)       # sign flips rarely along a row
    return T.merge(T.codes_from_levels((lv * sg).astype(np.int64)))


def mk_val(fn, *a):
    return lambda: T.synth_mat(L=fn(*a))


def mk_sym(fn, *a, partner=None):
    return lambda: T.synth_mat(sym=fn(*a), partner=partner)


def scale_case(structured):
    """D1 scale planes: matrix of iid codes whose block scales are iid (V-a) or piecewise constant (V-b)."""
    def mk():
        rng = np.random.default_rng(8)
        s = iid_syms(8)
        m = T.synth_mat(sym=s)
        if structured:
            E = 119 + (np.cumsum(rng.random((R, W // 32)) < 0.05, axis=1) % 4)
        else:
            E = rng.integers(119, 123, size=(R, W // 32))
        m.E = E.astype(np.int64)
        return m
    return mk


def scale_check():
    res = {}
    for tag, structured in (("V-a", False), ("V-b", True)):
        m = scale_case(structured)()
        e = T.evaluate(X.D1_split(m, {"kind": "scale_planes"}))
        n = m.E.size
        base = T.scale_bits(m.E)
        res[tag] = {"n": n, "roundtrip": e["roundtrip"], "appear_C0": (base - e["C0"]) / n, "appear_C1": (base - e["C1"]) / n}
    res["pass_V-a"] = max(res["V-a"]["appear_C0"], res["V-a"]["appear_C1"]) <= 0.01
    res["pass_V-b"] = max(res["V-b"]["appear_C0"], res["V-b"]["appear_C1"]) >= 0.05
    res["pass"] = res["pass_V-a"] and res["pass_V-b"] and res["V-a"]["roundtrip"] and res["V-b"]["roundtrip"]
    res["theory_V-b_per_value"] = None
    return res


def checks():
    iid_v = mk_val(iid_levels, 0)
    sm = mk_val(smooth, 1)
    out = {}
    C = T.synth_check
    for v in (dict(dom="L", axis="row", N=8, rsign=False), dict(dom="X", axis="col", N=32, rsign=False)):
        out[f"A1_wht {v}"] = C(X.A1_wht, v, "value", iid_v, sm)
    out["A1_wht random sign N=32 (V-b: fixed sign pattern x smooth)"] = C(
        X.A1_wht, dict(dom="L", axis="row", N=32, rsign=True), "value", iid_v, mk_val(signed_smooth, 1))
    for v in (dict(dom="L", axis="row", N=32, q=1), dict(dom="X", axis="col", N=32, q=2)):
        out[f"A2_dct {v}"] = C(X.A2_dct, v, "value", iid_v, sm)
    out["A3_dft row full q=1"] = C(X.A3_dft, dict(dom="X", axis="row", N="full", q=1), "value", iid_v,
                                   mk_val(sinusoids, 2))
    for w in ("haar", "53", "97"):
        for d in ("inblock", "row", "col", "2d"):
            out[f"A4_dwt {w} {d} lev3"] = C(X.A4_dwt, dict(wavelet=w, dir=d, lev=3), "value", iid_v, sm)
    out["A5_lapped M=8 q=1"] = C(X.A5_lapped, dict(M=8, q=1), "value", iid_v, sm)
    for k in ("col", "2d", "row_inblock"):
        out[f"B1_lorenzo {k}"] = C(X.B1_lorenzo, dict(kind=k), "value", iid_v, sm)
    out["B2_xor left"] = C(X.B2_xor, dict(ref="left"), "code", mk_sym(iid_syms, 0), mk_sym(copy_left_syms, 4))
    out["B2_xor up"] = C(X.B2_xor, dict(ref="up"), "code", mk_sym(iid_syms, 0), mk_sym(column_syms, 5))
    ps = iid_syms(9)
    noisy = np.where(np.random.default_rng(10).random(ps.shape) < 0.9, ps, iid_syms(11))
    out["B2_xor partner"] = C(X.B2_xor, dict(ref="partner"), "code", mk_sym(iid_syms, 0, partner=ps),
                              mk_sym(lambda: noisy, partner=ps))
    out["B3_regress order1 s1"] = C(X.B3_regress, dict(order=1, s=1), "value", iid_v, mk_val(ramps, 3))
    out["B3_regress order2 s1"] = C(X.B3_regress, dict(order=2, s=1), "value", iid_v, mk_val(ramps, 3))
    for k in ("linear", "cubic"):
        for d in ("row", "col", "inblock"):
            out[f"B4_interp {k} {d}"] = C(X.B4_interp, dict(kind=k, dir=d), "value", iid_v, sm)
        out[f"B5_multilevel {k}"] = C(X.B5_multilevel, dict(kind=k), "value", iid_v, sm)
    cs, cv = "code_seq", "level_seq"
    colv = mk_val(column_values, 14)
    out["C_order col code"] = C(X.C_order, dict(kind="col", data="code"), "code", mk_sym(iid_syms, 0),
                                mk_sym(column_syms, 5), base=cs)
    out["C_order col value"] = C(X.C_order, dict(kind="col", data="value"), "value", iid_v, colv, base=cv)
    # orders that change the context of only the first two codes of each block: criterion 0.005 (see report)
    out["C_order blockcol code"] = C(X.C_order, dict(kind="blockcol", data="code"), "code", mk_sym(iid_syms, 0),
                                     mk_sym(column_type_blocks, 12), base=cs, need=0.005)
    for k in ("z", "hilbert"):
        out[f"C_order {k} code code"] = C(X.C_order, dict(kind=k, unit="code", data="code"), "code",
                                          mk_sym(iid_syms, 0), mk_sym(column_syms, 5), base=cs)
        out[f"C_order {k} block code"] = C(X.C_order, dict(kind=k, unit="block", data="code"), "code",
                                           mk_sym(iid_syms, 0), mk_sym(block_patch_types, 13), base=cs, need=0.005)
        out[f"C_order {k} code value"] = C(X.C_order, dict(kind=k, unit="code", data="value"), "value", iid_v, colv,
                                           base=cv)
        out[f"C_order {k} block value"] = C(X.C_order, dict(kind=k, unit="block", data="value"), "value", iid_v,
                                            mk_val(block_patch_types, 13, 4, True), base=cv, need=0.005)
    out["D1_split fields"] = C(X.D1_split, dict(kind="fields"), "code", (lambda: T.synth_mat(sym=iid_syms(0, (1024, 2048)))),
                               mk_sym(sign_runs_syms, 7))
    out["D1_split planes"] = C(X.D1_split, dict(kind="planes"), "code", (lambda: T.synth_mat(sym=iid_syms(0, (1024, 2048)))),
                               mk_sym(sign_runs_syms, 7))
    out["D1_split scale_planes"] = scale_check()
    # theory where it is simple: copy-left symbols, XOR with left -> H of the XOR stream ~ 0.9 * 0 + mixture
    return out
