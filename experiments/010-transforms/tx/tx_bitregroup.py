"""010 supplement: bit regrouping of the FP4 codes -- statistics only (step 1, no coding scheme).

    set -a && source .env && set +a
    PYTHONPATH=. python experiments/010-transforms/tx/tx_bitregroup.py selftest
    PYTHONPATH=. python experiments/010-transforms/tx/tx_bitregroup.py run {deepseek|kimi} [--workers N] [--limit K]
    PYTHONPATH=. python experiments/010-transforms/tx/tx_bitregroup.py summarize

Idea: break the 4-bit code apart (bit 3 = sign, bits 2-1 = exponent, bit 0 = mantissa) and pack the SAME field
of g different weights into one new number of g x b bits; then look at the statistics of these new numbers. A
grouped symbol coded with a static table decodes independently of every other group, so any structure found here
would be usable without serial context.

Fields (b bits per weight): sign, exp_hi, exp_lo, mant (1); exp (2); mag = exponent + mantissa (3); code (4, reference).
Geometries (which g weights form one group; groups never overlap and cover the matrix, tails that do not fill a
group are left out and the covered weight count is recorded):
  blk_adj    : g adjacent codes inside one 32-code block
  blk_stride : g codes inside one block at stride 32/g (positions t, t + 32/g, ...)
  row_xblk   : the same in-block position in g consecutive blocks of the row
  col        : the same column in g consecutive rows
Group sizes: g in {2, 4, 8, 16} with g x b <= 16 bits (so at most 65,536 different new numbers).

Per matrix and configuration (field, geometry, g):
  H_act   plug-in (empirical) entropy of the new numbers, total bits;
  H_shuf  the same after a random permutation of all weight positions of the matrix (each weight keeps its 4 bits),
          which removes every dependence between weights but keeps the marginal and the finite-sample bias;
          MI = H_shuf - H_act is the information shared between the g weights of a group;
  KT_act  adaptive KT code length of the new numbers (every symbol starts at 1/2, one table per matrix), i.e. what a
          zero-order adaptive code of the new numbers would really cost, learning included;
  in-block geometries also: Hc_act = H(new number | block scale byte), Hc_shufE = the same after permuting the codes
          only among positions with the same block scale (keeps what the scale explains, removes the rest), so
          MIc = Hc_shufE - Hc_act is the shared information beyond the scale; KTc_act = KT with one table per scale.
Per field and matrix also the single-weight values (g = 1): H(field), H(field | scale), KT, for reference.
Pooled histograms of the new numbers (configurations with g x b <= 8 bits) are kept per model and role.
Every grouping is checked to be reversible: the four 1-bit fields, regrouped and unpacked, give back the codes.
"""
import argparse
import json
import math
import os
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402
from scipy.special import gammaln  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RES = HERE.parent / "results" / "bitregroup"
SEED = 20261127
B = 32
FIELDS = {"sign": (3, 1), "exp_hi": (2, 1), "exp_lo": (1, 1), "mant": (0, 1), "exp": (1, 2), "mag": (0, 3),
          "code": (0, 4)}                                   # field -> (lowest bit, width)
GEOMS = ["blk_adj", "blk_stride", "row_xblk", "col"]
INBLOCK = {"blk_adj", "blk_stride"}
GS = [2, 4, 8, 16]
HIST_MAX_BITS = 8
MODELS = {
    "deepseek": dict(plan="experiments/008-scale-media/results/fp4/B/plan.json", first=172,
                     items="experiments/008-scale-media/results/fp4_increments_B.json",
                     samples="data/008/fp4/matrices.json", snapshot_env="SNAPSHOT"),
    "kimi": dict(plan="experiments/009-kimi-k3/results/plan_552.json", first=None, items=None,
                 samples="data/009/kimi/matrices.json", snapshot_env="KIMI_SNAPSHOT"),
}


def configs():
    for f, (_, b) in FIELDS.items():
        for g in GS:
            if g * b <= 16:
                for geo in GEOMS:
                    yield f, geo, g


def tasks_of(model):
    m = MODELS[model]
    plan = json.load(open(ROOT / m["plan"]))
    tasks = plan["tasks"][:m["first"]] if m["first"] else plan["tasks"]
    if m["items"]:
        ents = sorted(e for t in tasks for e in t["entries"])
        assert ents == sorted(json.load(open(ROOT / m["items"]))["items"]), "sample differs from 008"
    return tasks


def field(C, f):
    lo, b = FIELDS[f]
    return (C >> lo) & ((1 << b) - 1)


def groups(X, geo, g):
    """X (R, W) -> (n_groups, g) array of member values, plus the row/block index of each group (for the scale)."""
    R, W = X.shape
    nb = W // B
    if geo == "blk_adj":
        G = X.reshape(R, nb, B // g, g)
        blk = np.broadcast_to(np.arange(R * nb).reshape(R, nb, 1), (R, nb, B // g))
        return G.reshape(-1, g), blk.reshape(-1)
    if geo == "blk_stride":
        G = X.reshape(R, nb, g, B // g).transpose(0, 1, 3, 2)
        blk = np.broadcast_to(np.arange(R * nb).reshape(R, nb, 1), (R, nb, B // g))
        return G.reshape(-1, g), blk.reshape(-1)
    if geo == "row_xblk":
        nk = nb // g * g
        G = X[:, :nk * B].reshape(R, nk // g, g, B).transpose(0, 1, 3, 2)
        return G.reshape(-1, g), None
    if geo == "col":
        rk = R // g * g
        G = X[:rk].reshape(rk // g, g, W).transpose(0, 2, 1)
        return G.reshape(-1, g), None
    raise ValueError(geo)


def pack(G, b):
    sh = (np.arange(G.shape[1], dtype=np.int64) * b)
    return (G.astype(np.int64) << sh).sum(1)


def unpack(s, g, b):
    sh = (np.arange(g, dtype=np.int64) * b)
    return (s[:, None] >> sh) & ((1 << b) - 1)


def h_total(cnt):
    """plug-in entropy of a count vector, total bits (n * H)."""
    c = cnt[cnt > 0].astype(np.float64)
    n = c.sum()
    return float(n * math.log2(n) - (c * np.log2(c)).sum())


def kt_total(cnt2d, K):
    """closed-form sequential KT code length, one table per row of cnt2d (alphabet K), total bits."""
    tot = cnt2d.sum(1)
    used = tot > 0
    c = cnt2d[used]
    nz = c[c > 0].astype(np.float64)
    lnp = (gammaln(nz + 0.5) - gammaln(0.5)).sum() - (gammaln(tot[used] + K * 0.5) - gammaln(K * 0.5)).sum()
    return float(-lnp / math.log(2))


def cond_counts(s, ctx, K, values=False):
    """counts (n_ctx_used, K) of s per context value, built sparsely (only contexts that occur)."""
    cu, ci = np.unique(ctx, return_inverse=True)
    cc = np.bincount(ci * K + s, minlength=len(cu) * K).reshape(len(cu), K)
    return (cu, cc) if values else cc


def sparse(cnt):
    nz = np.flatnonzero(cnt)
    return nz.astype(np.int32), cnt[nz].astype(np.int64)


def sparse_cond(cu, cc):
    return {int(e): sparse(row) for e, row in zip(cu, cc)}


def shuffles(C, E, rng):
    """P0: random permutation of all positions; PE: permutation only among positions with the same block scale."""
    R, W = C.shape
    flat = C.ravel()
    P0 = flat[rng.permutation(flat.size)].reshape(R, W)
    Ew = np.repeat(E.astype(np.int64), B, axis=1).ravel()
    by_scale = np.argsort(Ew, kind="stable")
    mixed = np.lexsort((rng.random(flat.size), Ew))
    PE = np.empty_like(flat)
    PE[by_scale] = flat[mixed]
    assert np.array_equal(Ew[by_scale], Ew[mixed])
    return P0, PE.reshape(R, W)


def matrix_stats(C, E, rng):
    R, W = C.shape
    assert W % B == 0
    C = C.astype(np.int64)
    P0, PE = shuffles(C, E, rng)
    Eblk = E.astype(np.int64).ravel()                         # scale of block index r * nb + k
    out, hist = {}, {}
    glob = {}                     # global counts, sparse: "act|cfg", "shuf|cfg" -> (idx, cnt); "actE|cfg", "shufE|cfg" -> {scale: (idx, cnt)}
    # reversibility: every geometry and g, the four 1-bit planes regrouped and unpacked give back the codes
    for geo in GEOMS:
        for g in GS:
            rec = 0
            ref = groups(C, geo, g)[0]
            for f in ("sign", "exp_hi", "exp_lo", "mant"):
                lo, b = FIELDS[f]
                Gf, _ = groups(field(C, f), geo, g)
                rec = rec | (unpack(pack(Gf, b), g, b) << lo)
            assert np.array_equal(rec, ref), (geo, g)
    single = {}
    Ew = np.repeat(E.astype(np.int64), B, axis=1).ravel()
    for f, (_, b) in FIELDS.items():
        K = 1 << b
        s = field(C, f).ravel()
        cnt = np.bincount(s, minlength=K)
        cu, cc = cond_counts(s, Ew, K, values=True)
        glob[f"act|{f}/single/1"] = sparse(cnt)
        glob[f"actE|{f}/single/1"] = sparse_cond(cu, cc)
        single[f] = {"n": int(s.size), "H": h_total(cnt), "Hc": sum(h_total(r) for r in cc),
                     "KT": kt_total(cnt[None], K), "KTc": kt_total(cc, K)}
    for f, geo, g in configs():
        b = FIELDS[f][1]
        K = 1 << (g * b)
        Ga, blk = groups(field(C, f), geo, g)
        sa = pack(Ga, b)
        ca = np.bincount(sa, minlength=K)
        ss = pack(groups(field(P0, f), geo, g)[0], b)
        cs = np.bincount(ss, minlength=K)
        glob[f"act|{f}/{geo}/{g}"] = sparse(ca)
        glob[f"shuf|{f}/{geo}/{g}"] = sparse(cs)
        r = {"n_groups": int(len(sa)), "n_weights": int(len(sa) * g), "bits": g * b,
             "H_act": h_total(ca), "H_shuf": h_total(cs), "KT_act": kt_total(ca[None], K)}
        if geo in INBLOCK:
            ctx = Eblk[blk]
            cu, cc = cond_counts(sa, ctx, K, values=True)
            se = pack(groups(field(PE, f), geo, g)[0], b)
            cue, cce = cond_counts(se, ctx, K, values=True)
            glob[f"actE|{f}/{geo}/{g}"] = sparse_cond(cu, cc)
            glob[f"shufE|{f}/{geo}/{g}"] = sparse_cond(cue, cce)
            r["Hc_act"] = sum(h_total(x) for x in cc)
            r["Hc_shufE"] = sum(h_total(x) for x in cce)
            r["KTc_act"] = kt_total(cc, K)
        out[f"{f}/{geo}/{g}"] = r
        if g * b <= HIST_MAX_BITS:
            hist[f"{f}/{geo}/{g}"] = ca
    return {"single": single, "configs": out}, hist, glob


def expert_task(model, task, half=0):
    from redundancy.adapters.fp4 import FP4
    m = MODELS[model]
    ad = FP4({"samples": str(ROOT / m["samples"]), "snapshot_env": m["snapshot_env"]})
    res, hists, globs = [], {}, []
    for name in task["entries"]:
        t0 = time.time()
        C, E = ad._read(ad.samples[name])
        rng = np.random.default_rng([SEED, zlib.crc32(name.encode())])
        st, hist, glob = matrix_stats(C, E, rng)
        role = name.split(".")[-1]
        globs.append((role, half, glob))
        for k, v in hist.items():
            key = f"{role}|{k}"
            hists[key] = hists.get(key, 0) + v
        res.append({"matrix": name, "layer": task["layer"], "expert": task["expert"], "role": role,
                    "shape": list(C.shape), **st, "seconds": time.time() - t0})
    return res, hists, globs


class Global:
    """pooled counts per (kind, role, half, config); conditional kinds keep one row per scale value."""
    def __init__(self):
        self.d = {}

    def size(self, cfg):
        f, _, g = cfg.split("/")
        return 1 << (int(g) * FIELDS[f][1])

    def add(self, role, half, glob):
        for key, v in glob.items():
            kind, cfg = key.split("|")
            K = self.size(cfg)
            k = (kind, role, half, cfg)
            if kind in ("act", "shuf"):
                a = self.d.setdefault(k, np.zeros(K, dtype=np.int64))
                a[v[0]] += v[1]
            else:
                rows = self.d.setdefault(k, {})
                for e, (idx, cnt) in v.items():
                    a = rows.setdefault(e, np.zeros(K, dtype=np.int64))
                    a[idx] += cnt

    def save(self, path):
        arrs = {}
        for (kind, role, half, cfg), v in self.d.items():
            base = f"{kind}|{role}|{half}|{cfg}"
            if isinstance(v, dict):
                es = sorted(v)
                arrs[base + "|scales"] = np.array(es, dtype=np.int64)
                M = np.stack([v[e] for e in es])
                nz = np.flatnonzero(M)
                arrs[base + "|shape"] = np.array(M.shape, dtype=np.int64)
                arrs[base + "|idx"], arrs[base + "|cnt"] = nz.astype(np.int64), M.ravel()[nz]
            else:
                nz = np.flatnonzero(v)
                arrs[base + "|shape"] = np.array([len(v)], dtype=np.int64)
                arrs[base + "|idx"], arrs[base + "|cnt"] = nz.astype(np.int64), v[nz]
        np.savez_compressed(path, **arrs)


def cmd_run(a):
    RES.mkdir(parents=True, exist_ok=True)
    path = RES / f"bitregroup_{a.model}_items.jsonl"
    hpath = RES / f"hist_{a.model}.npz"
    gpath = RES / f"global_{a.model}.npz"
    GL = Global()
    assert not path.exists(), f"{path} exists; remove it to rerun"
    todo = tasks_of(a.model)
    if a.limit:
        todo = todo[:a.limit]
    print(f"{a.model}: {len(todo)} experts", flush=True)
    t0 = time.time()
    H = {}
    with open(path, "w") as fo, ProcessPoolExecutor(a.workers) as ex:
        halves = [i % 2 for i in range(len(todo))]          # alternate experts: two halves for a held-out check
        for i, (rs, hs, gs) in enumerate(ex.map(expert_task, [a.model] * len(todo), todo, halves), 1):
            for role, half, glob in gs:
                GL.add(role, half, glob)
            for r in rs:
                fo.write(json.dumps(r) + "\n")
            fo.flush()
            for k, v in hs.items():
                H[k] = H.get(k, 0) + v
            if i % 20 == 0:
                print(f"{i}/{len(todo)} {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(hpath, **{k: v for k, v in H.items()})
    GL.save(gpath)
    print(f"RUN_DONE {a.model} wall {time.time() - t0:.0f} s", flush=True)


def pooled(rows, f, geo, g, keys):
    rs = [r["configs"][f"{f}/{geo}/{g}"] for r in rows]
    n = sum(r["n_weights"] for r in rs)
    return {k: sum(r[k] for r in rs) / n for k in keys if k in rs[0]}, n


def load_global(path):
    z = np.load(path)
    d = {}
    for k in z.files:
        if not k.endswith("|shape"):
            continue
        base = k[:-len("|shape")]
        kind, role, half, cfg = base.split("|")
        shape = tuple(z[k])
        M = np.zeros(int(np.prod(shape)), dtype=np.int64)
        M[z[base + "|idx"]] = z[base + "|cnt"]
        M = M.reshape(shape)
        d[(kind, role, int(half), cfg)] = (dict(zip(z[base + "|scales"].tolist(), M)) if base + "|scales" in z.files else M)
    return d


def xent(c, t, K):
    """bits of counts c coded with the static table estimated from counts t (+1/2 per symbol)."""
    q = (t + 0.5) / (t.sum() + 0.5 * K)
    m = c > 0
    return float(-(c[m] * np.log2(q[m])).sum())


def glob_pair(A, Bc, K):
    """A, Bc: the two halves (arrays, or dicts scale -> array). Returns total bits in-sample and held-out, count."""
    if isinstance(A, dict):
        es = set(A) | set(Bc)
        z = np.zeros(K, dtype=np.int64)
        P = [(A.get(e, z), Bc.get(e, z)) for e in es]
    else:
        P = [(A, Bc)]
    ins = sum(h_total(a + b) for a, b in P if (a + b).sum())
    held = sum(xent(a, b, K) + xent(b, a, K) for a, b in P)
    n = sum(int(a.sum() + b.sum()) for a, b in P)
    return ins, held, n


def add_tables(xs):
    if isinstance(xs[0], dict):
        out = {}
        for x in xs:
            for e, v in x.items():
                out[e] = out.get(e, 0) + v
        return out
    return sum(xs)


def global_summary(model):
    d = load_global(RES / f"global_{model}.npz")
    roles = ("w1", "w2", "w3")
    out = {}
    cfgs = [f"{f}/single/1" for f in FIELDS] + [f"{f}/{geo}/{g}" for f, geo, g in configs()]
    for scope in roles + ("shared",):
        rs = roles if scope == "shared" else (scope,)
        res = {}
        for cfg in cfgs:
            f, geo, g = cfg.split("/")
            g = int(g)
            K = 1 << (g * FIELDS[f][1])
            r = {}
            for kind in ("act", "shuf", "actE", "shufE"):
                if cfg.endswith("single/1") and kind.startswith("shuf"):
                    continue
                if (kind, rs[0], 0, cfg) not in d:
                    continue
                A = add_tables([d[(kind, x, 0, cfg)] for x in rs])
                Bc = add_tables([d[(kind, x, 1, cfg)] for x in rs])
                ins, held, n = glob_pair(A, Bc, K)
                r[kind] = {"in_sample": ins / (n * g), "held_out": held / (n * g), "n_weights": n * g}
            if "shuf" in r:
                r["MI_held_out"] = r["shuf"]["held_out"] - r["act"]["held_out"]
            if "shufE" in r:
                r["MIc_held_out"] = r["shufE"]["held_out"] - r["actE"]["held_out"]
            res[cfg] = r
        # whole decompositions: all fields of a weight regrouped the same way, per weight, vs the 4-bit code tables
        dec = {}
        parts = {"4x1bit": ("sign", "exp_hi", "exp_lo", "mant"), "sign+exp+mant": ("sign", "exp", "mant"),
                 "sign+mag": ("sign", "mag")}
        for geo in GEOMS:
            for g in GS:
                for nm, fs in parts.items():
                    keys = [f"{f}/{geo}/{g}" for f in fs]
                    if not all(k in res for k in keys):
                        continue
                    v = {"held_out": sum(res[k]["act"]["held_out"] for k in keys)}
                    if all("actE" in res[k] for k in keys):
                        v["held_out_given_scale"] = sum(res[k]["actE"]["held_out"] for k in keys)
                    v["vs_code_table"] = res["code/single/1"]["act"]["held_out"] - v["held_out"]
                    if "held_out_given_scale" in v:
                        v["vs_code_table_given_scale"] = res["code/single/1"]["actE"]["held_out"] - v["held_out_given_scale"]
                    dec[f"{nm}/{geo}/{g}"] = v
        out[scope] = {"configs": res, "decompositions": dec}
    return out


def cmd_summarize(a):
    allout = {}
    for model in MODELS:
        path = RES / f"bitregroup_{model}_items.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(l) for l in open(path)]
        want = [e for t in tasks_of(model) for e in t["entries"]]
        assert sorted(r["matrix"] for r in rows) == sorted(want), f"{model}: sample incomplete"
        out = {"model": model, "n_matrices": len(rows), "seed": SEED,
               "unit": "bits per weight covered by the groups (one field of one weight = b raw bits); "
                       "MI = H_shuf - H_act, MIc = Hc_shufE - Hc_act",
               "single": {}, "configs": {}}
        for role in ("all", "w1", "w2", "w3"):
            rr = rows if role == "all" else [r for r in rows if r["role"] == role]
            sg = {}
            for f in FIELDS:
                n = sum(r["single"][f]["n"] for r in rr)
                sg[f] = {k: sum(r["single"][f][k] for r in rr) / n for k in ("H", "Hc", "KT", "KTc")}
            out["single"][role] = sg
            cf = {}
            for f, geo, g in configs():
                v, n = pooled(rr, f, geo, g, ("H_act", "H_shuf", "KT_act", "Hc_act", "Hc_shufE", "KTc_act"))
                v["MI"] = v["H_shuf"] - v["H_act"]
                if "Hc_act" in v:
                    v["MIc"] = v["Hc_shufE"] - v["Hc_act"]
                v["n_weights"] = n
                v["KT_minus_single_KT"] = v["KT_act"] - sg[f]["KT"]
                if "KTc_act" in v:
                    v["KTc_minus_single_KTc"] = v["KTc_act"] - sg[f]["KTc"]
                cf[f"{f}/{geo}/{g}"] = v
            out["configs"][role] = cf
        out["seconds_total"] = float(sum(r["seconds"] for r in rows))
        out["global_note"] = ("one static table per scope pooled over all sample matrices (w1 / w2 / w3 separately, or "
                              "shared by the three roles); held_out = experts split in two alternating halves, each half "
                              "coded with the table of the other (+1/2 per symbol), bits per weight; actE = one table "
                              "per block-scale value; MI_held_out = shuffled minus actual")
        out["global"] = global_summary(model)
        json.dump(out, open(RES / f"bitregroup_{model}.json", "w"), indent=1)
        allout[model] = out
        print(f"== {model}: {len(rows)} matrices, cpu {out['seconds_total']:.0f} s")
        print("single (all):", {f: round(v["H"], 5) for f, v in out["single"]["all"].items()})
        for f, geo, g in configs():
            v = out["configs"]["all"][f"{f}/{geo}/{g}"]
            extra = f" MIc {v['MIc']:+.5f} KTc-1 {v['KTc_minus_single_KTc']:+.5f}" if "MIc" in v else ""
            print(f"{f:6s} {geo:10s} g={g:2d}  H {v['H_act']:.5f} MI {v['MI']:+.5f} "
                  f"KT-1 {v['KT_minus_single_KT']:+.5f}{extra}")


def cmd_selftest(a):
    rng = np.random.default_rng(0)
    R, W = 64, 512
    p = np.array([1, 3, 6, 8, 8, 6, 3, 1] * 2, dtype=float)
    E = rng.integers(118, 122, size=(R, W // B))
    cases = {}
    # 1. iid codes: no shared information anywhere
    cases["iid"] = rng.choice(16, size=(R, W), p=p / p.sum())
    # 2. one sign per block with probability 0.9 (sign shared inside blocks)
    C = rng.choice(8, size=(R, W), p=p[:8] / p[:8].sum())
    bs = rng.integers(0, 2, size=(R, W // B, 1))
    flip = rng.random((R, W // B, B)) < 0.1
    sgn = (bs ^ flip).reshape(R, W)
    cases["block_sign"] = C | (sgn << 3)
    # 3. exponent copied from the row above with probability 0.8 (shared down the columns)
    C = rng.choice(16, size=(R, W), p=p / p.sum())
    for i in range(1, R):
        cp = rng.random(W) < 0.8
        C[i, cp] = (C[i, cp] & 0b1001) | (C[i - 1, cp] & 0b0110)
    cases["col_exp"] = C
    # 4. magnitude set by the block scale only (no other dependence): MI > 0 but MIc ~ 0
    Ew = np.repeat(E, B, axis=1)
    lo = rng.choice(8, size=(R, W), p=np.array([1, 3, 6, 8, 8, 6, 3, 1]) / 36)
    hi = rng.choice(8, size=(R, W), p=np.array([8, 8, 6, 3, 1, 1, 1, 1]) / 29)
    mag = np.where(Ew % 2 == 0, lo, hi)
    cases["scale_only"] = mag | (rng.integers(0, 2, size=(R, W)) << 3)
    res = {}
    for k, C in cases.items():
        st, _, _ = matrix_stats(C, E, np.random.default_rng(1))
        cf = st["configs"]
        mi = {c: (v["H_shuf"] - v["H_act"]) / v["n_weights"] for c, v in cf.items()}
        mic = {c: (v["Hc_shufE"] - v["Hc_act"]) / v["n_weights"] for c, v in cf.items() if "Hc_act" in v}
        res[k] = (mi, mic)
        print(f"{k}: sign/blk_adj/8 MI {mi['sign/blk_adj/8']:+.4f}  sign/col/8 {mi['sign/col/8']:+.4f}  "
              f"exp/col/4 {mi['exp/col/4']:+.4f}  exp/blk_adj/4 {mi['exp/blk_adj/4']:+.4f}  "
              f"mag/blk_adj/2 MI {mi['mag/blk_adj/2']:+.4f} MIc {mic['mag/blk_adj/2']:+.4f}")
    mi, mic = res["iid"]
    small = [c for c in mi if int(c.split("/")[2]) * FIELDS[c.split("/")[0]][1] <= 8]
    assert all(abs(mi[c]) < 0.01 for c in small), "iid: MI should be ~0 at <= 8 bits"
    assert all(abs(mic[c]) < 0.02 for c in mic if c in small), "iid: MIc should be ~0"
    mi, _ = res["block_sign"]
    # col groups of neighbouring columns in one block repeat the same row-block signs, so the plug-in MI of col is
    # small but not zero (dependence between groups, not inside them); only a relative check here
    assert mi["sign/blk_adj/8"] > 0.3 and mi["sign/blk_stride/8"] > 0.3 and mi["sign/col/8"] < 0.1 * mi["sign/blk_adj/8"]
    mi, _ = res["col_exp"]
    assert mi["exp/col/4"] > 0.3 and abs(mi["exp/blk_adj/4"]) < 0.02
    mi, mic = res["scale_only"]
    assert mi["mag/blk_adj/2"] > 0.02 and abs(mic["mag/blk_adj/2"]) < 0.01
    # KT on one table equals the literal sequential KT coder
    s = rng.integers(0, 8, size=300)
    t, bits = np.full(8, 0.5), 0.0
    for x in s:
        bits -= math.log2(t[x] / t.sum())
        t[x] += 1
    assert abs(kt_total(np.bincount(s, minlength=8)[None], 8) - bits) < 1e-6
    # global tables: pooled sparse counts round-trip through save/load; held-out MI on block-shared signs
    import tempfile
    GL, direct = Global(), 0
    for i in range(4):
        C = rng.choice(8, size=(R, W), p=p[:8] / p[:8].sum())
        bs = rng.integers(0, 2, size=(R, W // B, 1))
        sg = (bs ^ (rng.random((R, W // B, B)) < 0.1)).reshape(R, W)
        C = C | (sg << 3)
        _, _, glob = matrix_stats(C, E, np.random.default_rng(i))
        GL.add("w1", i % 2, glob)
        direct = direct + np.bincount(pack(groups(field(C, "sign"), "blk_adj", 8)[0], 1), minlength=256)
    with tempfile.TemporaryDirectory() as td:
        GL.save(Path(td) / "g.npz")
        d = load_global(Path(td) / "g.npz")
    assert np.array_equal(d[("act", "w1", 0, "sign/blk_adj/8")] + d[("act", "w1", 1, "sign/blk_adj/8")], direct)
    eA = d[("actE", "w1", 0, "sign/blk_adj/8")]
    assert sum(v.sum() for v in eA.values()) == d[("act", "w1", 0, "sign/blk_adj/8")].sum()
    ins, held, n = glob_pair(d[("act", "w1", 0, "sign/blk_adj/8")], d[("act", "w1", 1, "sign/blk_adj/8")], 256)
    ins_s, held_s, _ = glob_pair(d[("shuf", "w1", 0, "sign/blk_adj/8")], d[("shuf", "w1", 1, "sign/blk_adj/8")], 256)
    print(f"global sign/blk_adj/8: in-sample {ins / (8 * n):.4f} held-out {held / (8 * n):.4f} "
          f"shuffled held-out {held_s / (8 * n):.4f} bit/weight")
    assert held >= ins and (held_s - held) / (8 * n) > 0.3 and abs(held_s / (8 * n) - 1) < 0.01
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("model", choices=list(MODELS))
    r.add_argument("--workers", type=int, default=32)
    r.add_argument("--limit", type=int, default=None)
    sub.add_parser("summarize")
    sub.add_parser("selftest")
    a = ap.parse_args()
    {"run": cmd_run, "summarize": cmd_summarize, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
