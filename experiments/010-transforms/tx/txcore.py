"""TX (010 supplement, docs/010-变换全集实验.md): shared infrastructure.

Data views
  code view  : sym = R10-merged symbols 0..14 (code 8 = -0 merged into 0, codes 9..15 -> 8..14).
  value view : L = signed integer level = 2 * E2M1 value, in {-12..12} (one 32-block has one scale);
               X = L * 2^(e_block - e_min): every block aligned to the matrix's smallest scale exponent (int64).
  The scale stream E is always coded first and separately (008: adaptive order-0, 256 symbols).
Evaluation coders (both give ideal adaptive code lengths, closed form, decodable sequentially)
  C0 : order-0 adaptive (no context).
  C1 : context coder; context = two causal neighbours (2-D: left and above; 1-D: the two previous values).
  For integers: binarisation as increment.eg_bits (audio / image chains of 008): bit-length class k (0 = zero,
  adaptive k-ary in context), sign (KT, context k [C1: and neighbour signs]), mantissa bits below the leading one
  (first 3 in context of class, position and prefix [C1: and the neighbour context]; the rest by class, position).
  For symbols: adaptive Dirichlet (alpha 1/2); C1 context = (left symbol, above symbol) or (previous two symbols).
Reference: 008 chain + 011 R9c (odd+nzc+max) on the same matrix, total bits incl. the scale stream (measured).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
E10 = ROOT / "experiments" / "010-transforms"
E11 = ROOT / "experiments" / "011-limits-and-models"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(E10))
import tx  # noqa: E402  (010 tx.py: read, sample_experts)

from redundancy import increment as I  # noqa: E402

RES = E10 / "results" / "tx"
DATA = ROOT / "data" / "010tx"
LN2 = math.log(2)
B = 32
MAG2 = np.array([0, 1, 2, 3, 4, 6, 8, 12], dtype=np.int64)        # 2 * |E2M1 value|
INV_MAG2 = np.full(13, -1, dtype=np.int64)
INV_MAG2[MAG2] = np.arange(8)

# ------------------------------------------------------------------ splits

ROLES = ("w1", "w2", "w3")
LAYERS = tx.LAYERS


def _names(experts):
    return [f"L{L}.E{e}.{r}" for L, es in experts.items() for e in es for r in ROLES]


_SAMPLE = tx.sample_experts()
TRAIN = _names({L: es[:2] for L, es in _SAMPLE.items()})          # 010's first two experts per layer
TEST = _names({L: es[2:] for L, es in _SAMPLE.items()})           # = 011 T1


def t2_names():
    sys.path.insert(0, str(E11))
    import common as K11
    return list(K11.T2)


def cross_experts(n=16):
    """16 experts per layer (fixed seed), disjoint from the 48-matrix sample; first 8 select, last 8 report."""
    out = {}
    for L in LAYERS:
        used = set(_SAMPLE[L])
        perm = [int(e) for e in np.random.default_rng([tx.SEED, 30, L]).permutation(384) if int(e) not in used]
        out[L] = perm[:n]
    return out


def partner_name(name):
    """decode order w3 -> w1 -> w2: w1 sees w3, w2 sees w1 (transposed); w3 has none."""
    L, Ex, r = name.split(".")
    return {"w1": f"{L}.{Ex}.w3", "w2": f"{L}.{Ex}.w1", "w3": None}[r]


# ------------------------------------------------------------------ views

def merge(C):
    Cm = np.where(C == 8, 0, C)
    return np.where(Cm > 8, Cm - 1, Cm).astype(np.int64)


def levels(C):
    C = np.asarray(C, dtype=np.int64)
    return np.where(C >= 8, -1, 1) * MAG2[C & 7]


def codes_from_levels(L):
    """inverse of levels() on R10-merged codes: zero -> code 0."""
    m = INV_MAG2[np.abs(L)]
    assert (m >= 0).all(), "value outside the E2M1 grid"
    return np.where(L < 0, m + 8, m).astype(np.int64)


def aligned(L, E):
    sh = np.repeat(E.astype(np.int64) - int(E.min()), B, axis=1)
    assert sh.max() < 50
    return L << sh


def unaligned(X, E):
    sh = np.repeat(E.astype(np.int64) - int(E.min()), B, axis=1)
    L = X >> sh
    assert ((L << sh) == X).all(), "aligned value not divisible by its block scale"
    return L


class Mat:
    """One matrix in both views (+ the partner's merged codes, oriented like this matrix)."""

    def __init__(self, name=None, C=None, E=None, partner=None, L=None):
        self.name = name
        if L is not None:                       # synthetic integer data (no FP4 codes)
            self.L = np.asarray(L, dtype=np.int64)
            self.E = np.zeros((self.L.shape[0], max(1, self.L.shape[1] // B)), dtype=np.int64) if E is None else E
            self.C = None
            self.sym = None
            self.X = self.L if E is None else aligned(self.L, E)
        else:
            self.C = np.asarray(C, dtype=np.int64)
            self.E = E
            self.sym = merge(self.C)
            self.L = levels(np.where(self.C == 8, 0, self.C))
            self.X = aligned(self.L, E)
        self.partner = partner                   # merged symbols of the partner, same shape (or None)
        self.n = self.L.size

    @classmethod
    def load(cls, name, with_partner=True):
        C, E = tx.read(name)
        P = None
        pn = partner_name(name) if with_partner else None
        if pn is not None:
            Cp, _ = tx.read(pn)
            P = merge(Cp)
            if name.endswith("w2"):
                P = np.ascontiguousarray(P.T)
        return cls(name, C, E, P)

    def check_codes(self, sym=None, L=None, X=None):
        """lossless check against the merged codes (code view) or the values (value view)."""
        if sym is not None:
            return bool(np.array_equal(sym, self.sym))
        if X is not None:
            L = unaligned(X, self.E) if self.C is not None else X
        if self.C is None:
            return bool(np.array_equal(L, self.L))
        return bool(np.array_equal(codes_from_levels(L), np.where(self.C == 8, 0, self.C)))


# ------------------------------------------------------------------ adaptive code lengths (dense counting)

def _lg(alpha, counts):
    return I._lgam(alpha, np.asarray(counts, dtype=np.int64))


def _dense(key):
    key = np.asarray(key, dtype=np.int64).ravel()
    if key.size and key.max() < 60_000_000 and key.min() >= 0:
        return key, int(key.max()) + 1
    u, inv = np.unique(key, return_inverse=True)
    return inv, len(u)


def kt(key, bit):
    key, nk = _dense(key)
    if key.size == 0:
        return 0.0
    bit = np.asarray(bit, dtype=np.int64).ravel()
    n = np.bincount(key, minlength=nk)
    n1 = np.bincount(key, weights=bit, minlength=nk).astype(np.int64)
    return -(_lg(0.5, n - n1) + _lg(0.5, n1) - _lg(1.0, n)) / LN2


def dirichlet(sym, ctx, K, alpha=0.5):
    sym = np.asarray(sym, dtype=np.int64).ravel()
    if sym.size == 0:
        return 0.0
    ctx = np.zeros_like(sym) if ctx is None else np.asarray(ctx, dtype=np.int64).ravel()
    key, nk = _dense(ctx * K + sym)
    cnt = np.bincount(key, minlength=nk)
    cctx, nc = _dense(ctx)
    tot = np.bincount(cctx, minlength=nc)
    return -(_lg(alpha, cnt) - _lg(K * alpha, tot)) / LN2


def bitlen(a):
    a = np.asarray(a, dtype=np.int64)
    assert a.size == 0 or a.max() < (1 << 52)
    return np.frexp(a.astype(np.float64))[1].astype(np.int64)


def neighbours(x, mode, fill):
    """(first, second) causal neighbours of every element: 2-D left / above, or 1-D previous two (flattened)."""
    if mode == "2d":
        assert x.ndim == 2
        a = np.full(x.shape, fill, dtype=np.int64)
        b = np.full(x.shape, fill, dtype=np.int64)
        a[:, 1:] = x[:, :-1]
        b[1:, :] = x[:-1, :]
        return a.ravel(), b.ravel()
    v = x.ravel()
    a = np.full(v.shape, fill, dtype=np.int64)
    b = np.full(v.shape, fill, dtype=np.int64)
    a[1:] = v[:-1]
    b[2:] = v[:-2]
    return a, b


def int_bits(x, coder, mode="2d"):
    """code length (bits) of the integer array x with coder 'C0' or 'C1' (mode '2d' or 'seq' for C1)."""
    x = np.asarray(x, dtype=np.int64)
    v = x.ravel()
    if v.size == 0:
        return 0.0
    a = np.abs(v)
    k = bitlen(a)
    kmax = int(k.max())
    bits = 6.0                                   # kmax (alphabet of the class code)
    Kc = kmax + 1
    sg = np.where(v > 0, 1, np.where(v < 0, 2, 0))
    if coder == "C0":
        ctx = np.zeros_like(k)
        sctx = np.minimum(k, 63)
    else:
        kk = np.minimum(k.reshape(x.shape), 15)
        k1, k2 = neighbours(kk, mode, 16)
        ctx = k1 * 17 + k2
        s1, s2 = neighbours(sg.reshape(x.shape), mode, 0)
        sctx = (np.minimum(k, 15) * 3 + s1) * 3 + s2
    bits += dirichlet(k, ctx, Kc)
    nz = k > 0
    bits += kt(sctx[nz], (v[nz] < 0).astype(np.int64))
    keys, bs = [], []
    for j in range(1, kmax):
        sel = k > j
        if not sel.any():
            continue
        kj, aj = k[sel], a[sel]
        b = (aj >> (kj - 1 - j)) & 1
        if j <= 3:
            pref = (aj >> (kj - j)) & ((1 << j) - 1)
            key = ((ctx[sel] * 64 + kj) * 4 + j) * 8 + pref
        else:
            key = (1 << 22) + kj * 64 + j
        keys.append(key)
        bs.append(b)
    if keys:
        bits += kt(np.concatenate(keys), np.concatenate(bs))
    return float(bits)


def sym_bits(s, K, coder, mode="2d"):
    s = np.asarray(s, dtype=np.int64)
    if coder == "C0":
        return float(dirichlet(s, None, K))
    a, b = neighbours(s, mode, K)
    return float(dirichlet(s, a * (K + 1) + b, K))


def bin_bits(s, coder, mode="2d"):
    return sym_bits(s, 2, coder, mode)


def scale_bits(E):
    """008 scale stream (R1s): adaptive order-0 over 256 values."""
    return float(I.dirichlet_bits(np.asarray(E).ravel(), None, 256))


# ------------------------------------------------------------------ streams and evaluation

class Stream:
    """kind 'int' (any integers) or 'sym' (alphabet K); mode '2d' or 'seq' for C1."""

    def __init__(self, arr, kind="int", mode="2d", K=None, name="", side=False):
        self.arr, self.kind, self.mode, self.K, self.name, self.side = arr, kind, mode, K, name, side

    def bits(self, coder):
        if self.kind == "int":
            return int_bits(self.arr, coder, self.mode)
        return sym_bits(self.arr, self.K, coder, self.mode)


def evaluate(enc):
    """enc: dict(streams=[Stream], side=bits, info={...}, roundtrip=bool, scale=None|bits replacing the scale stream)."""
    out = {"side_bits": float(enc.get("side", 0.0)), "roundtrip": bool(enc["roundtrip"])}
    for c in ("C0", "C1"):
        per = [s.bits(c) for s in enc["streams"]]
        out[c] = float(sum(per)) + out["side_bits"]
        out["side_bits_" + c] = out["side_bits"] + float(sum(b for s, b in zip(enc["streams"], per) if s.side))
        out[c + "_streams"] = {s.name or str(i): float(b) for i, (s, b) in enumerate(zip(enc["streams"], per))}
    nzs = [np.count_nonzero(s.arr) for s in enc["streams"] if not s.side]
    tot = [s.arr.size for s in enc["streams"] if not s.side]
    out["nonzero_frac"] = float(sum(nzs) / sum(tot)) if tot else None
    if enc.get("scale") is not None:
        out["scale_bits"] = float(enc["scale"])
    out.update(enc.get("info", {}))
    return out


def raw_streams(m, view):
    if view == "code":
        return [Stream(m.sym, "sym", "2d", 15, "sym")]
    if view == "level":
        return [Stream(m.L, "int", "2d", name="L")]
    return [Stream(m.X, "int", "2d", name="X")]


def raw_bits(m):
    out = {}
    for view in ("code", "level", "aligned"):
        st = raw_streams(m, view)[0]
        out[view] = {c: st.bits(c) for c in ("C0", "C1")}
    out["code_seq"] = {c: sym_bits(m.sym.ravel(), 15, c, "seq") for c in ("C0", "C1")}
    out["level_seq"] = {c: int_bits(m.L.ravel(), c, "seq") for c in ("C0", "C1")}
    return out


# ------------------------------------------------------------------ coding gain and helpers

def coding_gain(coef, band):
    """AM / GM of the per-subband variances (subbands with zero variance excluded and counted)."""
    coef = np.asarray(coef, dtype=np.float64).ravel()
    band = np.asarray(band, dtype=np.int64).ravel()
    nb = int(band.max()) + 1
    cnt = np.bincount(band, minlength=nb).astype(np.float64)
    s1 = np.bincount(band, weights=coef, minlength=nb)
    s2 = np.bincount(band, weights=coef * coef, minlength=nb)
    ok = cnt > 1
    var = np.where(ok, s2 / np.maximum(cnt, 1) - (s1 / np.maximum(cnt, 1)) ** 2, 0.0)
    pos = ok & (var > 1e-12)
    if pos.sum() == 0:
        return {"coding_gain": None, "zero_var_bands": int(ok.sum())}
    v = var[pos]
    return {"coding_gain": float(v.mean() / math.exp(np.log(v).mean())), "zero_var_bands": int((ok & ~pos).sum())}


def log2_fact(n):
    return math.lgamma(n + 1) / LN2


# ------------------------------------------------------------------ reference (008 chain + 011 R9c)

_REF = None


def reference():
    """{name: {total, scale, code, n}} for T1 and T2 (bits). R9c = 'R9c odd+nzc+max' with its switch bit."""
    global _REF
    if _REF is not None:
        return _REF
    chains = {}
    for p in (E10 / "results" / "base.jsonl", E11 / "results" / "chains.jsonl"):
        for line in open(p):
            r = json.loads(line)
            chains[r["task"]] = r["chain"]["L"][-1]
    out = {}
    for line in open(E11 / "results" / "s0" / "s0_2_rows.jsonl"):
        r = json.loads(line)
        gain = max(r["base_code_bits"] - r["R9c odd+nzc+max"], 0.0) - 1.0
        out[r["task"]] = {"chain_total": chains[r["task"]], "r9c_gain_bits": gain,
                          "total": chains[r["task"]] - gain, "n": r["n"]}
    _REF = out
    return out


# ------------------------------------------------------------------ running

def jsonl_done(path, key="key"):
    if not Path(path).exists():
        return set()
    return {json.loads(l).get(key) for l in open(path)}


def run_jobs(fn, jobs, out_path, workers=48, log=print, done_key="key"):
    """jobs: list of dicts with a unique 'key'; fn(job) -> dict or list of dicts; results appended to out_path
    (skips jobs whose key already appears in the file under done_key)."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    done = jsonl_done(out_path, done_key)
    todo = [j for j in jobs if j["key"] not in done]
    log(f"{out_path}: {len(todo)} jobs ({len(jobs) - len(todo)} done)", flush=True)
    if not todo:
        return
    t0 = time.time()
    with ProcessPoolExecutor(min(workers, len(todo))) as ex, open(out_path, "a") as f:
        futs = {ex.submit(fn, j): j for j in todo}
        for i, fu in enumerate(as_completed(futs)):
            r = fu.result()
            for x in (r if isinstance(r, list) else [r]):
                f.write(json.dumps(x) + "\n")
            f.flush()
            if (i + 1) % max(1, len(todo) // 20) == 0:
                log(f"  {i + 1}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)


def load_rows(path):
    return [json.loads(l) for l in open(path)] if Path(path).exists() else []


# ------------------------------------------------------------------ method evaluation and synthetic checks

RAW_PATH = RES / "raw.jsonl"


def raw_table():
    return {r["key"]: r for r in load_rows(RAW_PATH)}


def appearance(raw, enc_eval, base):
    """displayed structure (bits): raw code length of the transform's input view `base` ('code', 'level' =
    block levels, 'aligned' = aligned values) minus the transformed code length (incl. side), per coder; plus the
    same against the raw code view."""
    out = {c: raw[base][c] - enc_eval[c] for c in ("C0", "C1")}
    out.update({c + "_vs_code": raw["code"][c] - enc_eval[c] for c in ("C0", "C1")})
    return out


def synth_mat(L=None, sym=None, partner=None):
    if sym is not None:
        m = Mat(L=np.zeros(np.shape(sym), dtype=np.int64))
        m.sym = np.asarray(sym, dtype=np.int64)
        m.partner = partner
        return m
    return Mat(L=L)


def synth_raw(m, view):
    if view == "code":
        st, sq = Stream(m.sym, "sym", "2d", 15), Stream(m.sym.ravel(), "sym", "seq", 15)
    else:
        st, sq = Stream(m.L, "int", "2d"), Stream(m.L.ravel(), "int", "seq")
    r = {c: st.bits(c) for c in ("C0", "C1")}
    rs = {c: sq.bits(c) for c in ("C0", "C1")}
    return {"code": r, "level": r, "aligned": r, "code_seq": rs, "level_seq": rs}


def synth_check(fn, variant, view, mk_a, mk_b, theory_b=None, tol=0.01, need=0.05, base=None):
    """V-a: iid data -> appearance (per value) <= tol for both coders.  V-b: targeted structure -> best appearance
    >= need per value (and, when a theory value is given, it is reported). fn(m, variant) -> enc dict."""
    res = {}
    for tag, mk in (("V-a", mk_a), ("V-b", mk_b)):
        m = mk()
        e = evaluate(fn(m, variant))
        ap = appearance(synth_raw(m, view), e, base or ("code" if view == "code" else "level"))
        n = m.sym.size if view == "code" else m.L.size
        res[tag] = {"n": n, "roundtrip": e["roundtrip"], "side_per_value": e["side_bits_C1"] / n,
                    "appear_C0": ap["C0"] / n, "appear_C1": ap["C1"] / n}
    va, vb = res["V-a"], res["V-b"]
    res["theory_V-b_per_value"] = theory_b
    res["pass_V-a"] = bool(max(va["appear_C0"], va["appear_C1"]) <= tol and va["roundtrip"])
    res["pass_V-b"] = bool(max(vb["appear_C0"], vb["appear_C1"]) >= need and vb["roundtrip"])
    res["criteria"] = {"V-a_max": tol, "V-b_min": need, "base_view": base or ("code" if view == "code" else "level")}
    res["pass"] = res["pass_V-a"] and res["pass_V-b"]
    return res


def eval_job(job):
    """job: key, name, module, method, variant. Loads the matrix, runs the method, evaluates C0 / C1."""
    import importlib
    t0 = time.time()
    mod = importlib.import_module(job["module"])
    m = Mat.load(job["name"], with_partner=job.get("partner", False))
    meth = getattr(mod, job["method"])
    enc = meth(m, job["variant"])
    e = evaluate(enc)
    e.update({"key": job["key"], "name": job["name"], "method": job["method"], "variant": job["variant"],
              "vname": job.get("vname"), "view": job["view"], "n": m.n, "cpu_s": time.time() - t0})
    return e
