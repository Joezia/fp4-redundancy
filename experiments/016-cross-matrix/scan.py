"""Spec 016, part B: whole-model scans (every matrix of the model takes part).

    PYTHONPATH=.:experiments/016-cross-matrix python experiments/016-cross-matrix/scan.py hash    {deepseek|kimi} [--workers N]
    PYTHONPATH=.:experiments/016-cross-matrix python experiments/016-cross-matrix/scan.py queries {deepseek|kimi} [--nq ...]
    PYTHONPATH=.:experiments/016-cross-matrix python experiments/016-cross-matrix/scan.py gpu     {deepseek|kimi} [--workers N]

hash  (B1): exact duplicates of whole matrices, rows, row scales, columns, 32-code blocks and 8x32 patches over the
            whole model, by 64-bit hashes; within-matrix and cross-matrix counts and strict savings.
queries   : random query vectors / patches from the whole model (+ shuffled controls), saved to data/016.
gpu   (B2, B3): every query against every vector / every position (stride 1) of every matrix of the model.
"""
import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

import common as K  # noqa: E402

SEED = 20261125
M64 = np.uint64(0xFFFFFFFFFFFFFFFF)
UNITS = ["matrix", "row_codes", "row_scales", "col_codes", "block32", "patch8x32"]
NPART = 64


def bounded_map(ex, fn, args_list, window=96):
    """ordered map over a process pool with at most `window` tasks in flight (bounded memory)."""
    from collections import deque
    it = iter(args_list)
    q = deque()
    for args in it:
        q.append(ex.submit(fn, *args))
        if len(q) >= window:
            break
    while q:
        r = q.popleft().result()
        for args in it:
            q.append(ex.submit(fn, *args))
            break
        yield r


# ================================================================== B1 hashing
def splitmix(x):
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


_KEYS = splitmix(np.arange(1, 1 << 16, dtype=np.uint64))


def hash_rows(M):
    """64-bit hash of every row of a uint8 matrix whose row length is a multiple of 8."""
    n, nb = M.shape
    w = np.ascontiguousarray(M).view(np.uint64)                    # (n, nb/8)
    h = splitmix(w ^ _KEYS[: w.shape[1]][None, :]).sum(axis=1, dtype=np.uint64)
    return splitmix(h ^ np.uint64(nb))


def unit_hashes(P, E, C):
    """dict unit -> (hashes uint64, costs float64 in bits under the matrix's zero-order code / scale distribution)."""
    R, W = C.shape
    hc = np.bincount(C.ravel(), minlength=16).astype(np.float64)
    cl = np.where(hc > 0, -np.log2(np.maximum(hc, 1) / C.size), 0.0)
    cost = cl[C]                                                      # (R, W)
    he = np.bincount(E.ravel(), minlength=256).astype(np.float64)
    cle = np.where(he > 0, -np.log2(np.maximum(he, 1) / E.size), 0.0)
    nb = W // 32
    out = {}
    import xxhash
    hm = xxhash.xxh64(P.tobytes())
    hm.update(E.tobytes())
    out["matrix"] = (np.array([hm.intdigest()], dtype=np.uint64), np.array([cost.sum() + cle[E].sum()]))
    out["row_codes"] = (hash_rows(P), cost.sum(axis=1))
    Ep = E if E.shape[1] % 8 == 0 else np.pad(E, ((0, 0), (0, 8 - E.shape[1] % 8)))
    out["row_scales"] = (hash_rows(Ep), cle[E].sum(axis=1))
    out["col_codes"] = (hash_rows(np.ascontiguousarray(C.T)), cost.sum(axis=0))
    b = P.reshape(R, nb, 16).view(np.uint64)                          # (R, nb, 2)
    hb = splitmix(splitmix(b[:, :, 0] ^ _KEYS[0]) + b[:, :, 1])       # (R, nb)
    cb = cost.reshape(R, nb, 32).sum(axis=2)
    out["block32"] = (hb.ravel(), cb.ravel())
    hp = splitmix(hb.reshape(R // 8, 8, nb) + _KEYS[1:9][None, :, None]).sum(axis=1, dtype=np.uint64)
    out["patch8x32"] = (splitmix(hp).ravel(), cb.reshape(R // 8, 8, nb).sum(axis=1).ravel())
    return out


def hash_task(model, i, sub, part):
    e = K.entries(model)[i]
    P, E = K.load_packed(model, e)
    C = K.unpack(P)
    res = {"i": i}
    for u, (h, c) in unit_hashes(P, E, C).items():
        uh, first, cnt = np.unique(h, return_index=True, return_counts=True)
        cu = c[first]
        d = cnt > 1
        within = {"n": int(h.size), "dup_units": int((cnt[d] - 1).sum()), "dup_groups": int(d.sum()),
                  "dup_cost_bits": float(((cnt[d] - 1) * cu[d]).sum())}
        if sub > 1 and u in ("block32", "patch8x32"):
            # keep one of `sub` disjoint hash subsets (bits 58-63), so the cross-matrix table fits in memory
            sel = ((uh >> np.uint64(58)) % np.uint64(sub)) == np.uint64(part)
            uh, cu = uh[sel], cu[sel]
        if u in ("block32", "patch8x32"):
            pk = ((uh >> np.uint64(40)) & np.uint64(NPART - 1)).astype(np.int64)
            o = np.argsort(pk, kind="stable")
            res[u] = (within, uh[o], np.minimum(np.round(cu[o] * 4), 65535).astype(np.uint16),
                      np.bincount(pk, minlength=NPART))
        else:
            res[u] = (within, uh, cu.astype(np.float32), None)
    return res


def cmd_hash(a):
    ents = K.entries(a.model)
    n = len(ents) if a.limit is None else a.limit
    sub = a.sub if a.sub else {"deepseek": 2, "kimi": 16}[a.model]
    part = a.part
    keep_quarter = sub > 1
    store = {u: {"h": [], "c": []} if u not in ("block32", "patch8x32") else {"h": [[] for _ in range(NPART)], "c": [[] for _ in range(NPART)]}
             for u in UNITS}
    within = {u: {"n": 0, "dup_units": 0, "dup_groups": 0, "dup_cost_bits": 0.0, "matrices_with_dups": 0} for u in UNITS}
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for k, r in enumerate(bounded_map(ex, hash_task, [(a.model, i, sub, part) for i in range(n)]), 1):
            for u in UNITS:
                w, h, c, pc = r[u]
                for f in ("n", "dup_units", "dup_groups", "dup_cost_bits"):
                    within[u][f] += w[f]
                within[u]["matrices_with_dups"] += int(w["dup_units"] > 0)
                if pc is None:
                    store[u]["h"].append(h); store[u]["c"].append(c)
                else:
                    off = np.concatenate([[0], np.cumsum(pc)])
                    for p in range(NPART):
                        if pc[p]:
                            store[u]["h"][p].append(h[off[p]:off[p + 1]].copy()); store[u]["c"][p].append(c[off[p]:off[p + 1]].copy())
            if k % 500 == 0:
                print(f"{k}/{n} {time.time() - t0:.0f}s", flush=True)
    t_read = time.time() - t0
    print(f"read+hash {t_read:.0f}s", flush=True)
    # totals of units in the whole model (before the Kimi quarter filter) for the reference cost
    out = {"model": a.model, "n_matrices": n, "hash_subsets_for_blocks_patches": sub, "subset_index": part, "units": {},
           "seconds_read_hash": t_read}
    K.DATA.mkdir(parents=True, exist_ok=True)
    for u in UNITS:
        if u in ("block32", "patch8x32"):
            from concurrent.futures import ThreadPoolExecutor
            su = store[u]

            def dups(p):
                # concatenate one partition, free its pieces at once (bounded memory), then find repeated hashes
                h = np.concatenate(su["h"][p]) if su["h"][p] else np.zeros(0, np.uint64)
                c = np.concatenate(su["c"][p]) if su["c"][p] else np.zeros(0, np.uint16)
                su["h"][p] = None; su["c"][p] = None
                uh, first, cnt = np.unique(h, return_index=True, return_counts=True)
                d = cnt > 1
                return h.size, uh[d], cnt[d], c[first[d]].astype(np.float64) / 4
            with ThreadPoolExecutor(4) as tp:
                rs = list(tp.map(dups, range(NPART)))
            store[u] = None
            n_kept = int(sum(r[0] for r in rs))
            dh = np.concatenate([r[1] for r in rs]); dc = np.concatenate([r[2] for r in rs]); dcost = np.concatenate([r[3] for r in rs])
        else:
            h = np.concatenate(store[u]["h"]); c = np.concatenate(store[u]["c"]).astype(np.float64)
            store[u] = None
            uh, first, cnt = np.unique(h, return_index=True, return_counts=True)
            d = cnt > 1
            n_kept = int(h.size); dh, dc, dcost = uh[d], cnt[d], c[first[d]]
        n_total = within[u]["n"] if not (keep_quarter and u in ("block32", "patch8x32")) else None
        n_units_model = within[u]["n"]            # every unit of the model was hashed (before filtering)
        ref = math.log2(max(n_units_model, 2)) + 1
        extra = dc - 1                             # cross-matrix repeats (hashes are unique within a matrix)
        gross = float((extra * dcost).sum())
        netpos = float((extra * np.maximum(0, dcost - ref)).sum())
        n_rep = int(extra.sum())
        flag = float(K.kt_bits(np.array([n_kept - n_rep, n_rep]), 2)) if n_kept else 0.0
        hashspace = 64 - (math.log2(sub) if (keep_quarter and u in ("block32", "patch8x32")) else 0)
        exp_false = n_kept * (n_kept - 1) / 2 / 2.0 ** hashspace
        order = np.argsort(-(extra * dcost))[:2000]
        np.savez(K.DATA / f"b1_{a.model}_s{sub}p{part}_{u}_dups.npz", h=dh[order], cnt=dc[order], cost=dcost[order])
        out["units"][u] = {
            "n_units_model": n_units_model, "n_units_checked_cross": n_kept,
            "within_matrix": within[u],
            "cross_matrix": {"dup_groups": int(dh.size), "repeat_units": n_rep, "gross_cost_bits": gross,
                             "net_positive_bits_after_ref": netpos, "ref_cost_bits_per_repeat": ref,
                             "flag_cost_bits_all_units": flag,
                             "expected_false_hash_collisions": exp_false,
                             "max_group_size": int(dc.max()) if dc.size else 0,
                             "cost_of_top_groups": dcost[order[:20]].tolist(), "size_of_top_groups": dc[order[:20]].tolist()},
            "note": f"cross-matrix check of this unit covers hash subset {part} of {sub} (disjoint subsets; units are assigned by hash, so repeated units fall in the same subset)" if n_total is None else "",
        }
        print(u, json.dumps(out["units"][u]["cross_matrix"])[:400], json.dumps(within[u]), flush=True)
    out["seconds_total"] = time.time() - t0
    K.RES.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(K.RES / f"b1_{a.model}_s{sub}p{part}.json", "w"), indent=1)
    print("HASH_DONE", a.model, f"{time.time() - t0:.0f}s", flush=True)


def verify_task(model, i, targets):
    """locations and raw bytes of the blocks / patches of matrix i whose hash is in `targets` (unit -> uint64 array)."""
    e = K.entries(model)[i]
    P, E = K.load_packed(model, e)
    C = K.unpack(P)
    hs = unit_hashes(P, E, C)
    R, W = C.shape
    nb = W // 32
    out = []
    for u, tg in targets.items():
        h = hs[u][0]
        hit = np.nonzero(np.isin(h, tg))[0]
        for k in hit:
            if u == "block32":
                r, b = divmod(int(k), nb)
                raw = P[r, 16 * b:16 * b + 16].tobytes() + bytes([int(E[r, b])])
                loc = [r, 32 * b]
            else:
                r8, b = divmod(int(k), nb)
                raw = P[8 * r8:8 * r8 + 8, 16 * b:16 * b + 16].tobytes() + E[8 * r8:8 * r8 + 8, b].tobytes()
                loc = [8 * r8, 32 * b]
            out.append({"unit": u, "hash": int(h[k]), "matrix": e["matrix"], "row": loc[0], "col": loc[1],
                        "codes_hex": raw[:-1 if u == "block32" else -8].hex(), "scales_hex": raw[-1 if u == "block32" else -8:].hex(),
                        "codes": C[loc[0]:loc[0] + (1 if u == "block32" else 8), loc[1]:loc[1] + 32].tolist()})
    return out


def cmd_verify(a):
    import glob
    targets = {}
    for u in ("block32", "patch8x32"):
        hs = [np.load(f)["h"] for f in glob.glob(str(K.DATA / f"b1_{a.model}_s*_{u}_dups.npz"))]
        if hs:
            targets[u] = np.unique(np.concatenate(hs))
    print({u: len(v) for u, v in targets.items()}, flush=True)
    ents = K.entries(a.model)
    hits = []
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for k, r in enumerate(bounded_map(ex, verify_task, [(a.model, i, targets) for i in range(len(ents))]), 1):
            hits += r
            if k % 20000 == 0:
                print(f"{k}/{len(ents)} {time.time() - t0:.0f}s", flush=True)
    groups = {}
    for h in hits:
        groups.setdefault((h["unit"], h["hash"]), []).append(h)
    res = []
    for (u, hh), g in groups.items():
        same_codes = len({x["codes_hex"] for x in g}) == 1
        res.append({"unit": u, "hash": hh, "n": len(g), "identical_codes": same_codes,
                    "identical_codes_and_scales": same_codes and len({x["scales_hex"] for x in g}) == 1, "occurrences": g})
    out = {"model": a.model, "groups": res, "n_groups": len(res),
           "n_true_repeats": {u: sum(r["n"] - 1 for r in res if r["unit"] == u and r["identical_codes"]) for u in targets},
           "n_hash_collisions": {u: sum(1 for r in res if r["unit"] == u and not r["identical_codes"]) for u in targets},
           "seconds": time.time() - t0}
    json.dump(out, open(K.RES / f"b1_verify_{a.model}.json", "w"), indent=1)
    print("VERIFY_DONE", a.model, out["n_true_repeats"], out["n_hash_collisions"], flush=True)


# ================================================================== queries
SHAPES = {"1x64": (1, 64), "8x8": (8, 8), "16x16": (16, 16)}


def cmd_queries(a):
    rng = np.random.default_rng(SEED)
    ents = K.entries(a.model)
    mats = rng.choice(len(ents), a.qmats, replace=False)          # random order, so any prefix is a uniform sample
    nv = {"neuron": [], "hidden": []}
    meta = {"neuron": [], "hidden": []}
    patches = {s: [] for s in SHAPES}
    pmeta = {s: [] for s in SHAPES}
    per = {s: max(1, a.nq_patch[k] // a.qmats) for k, s in enumerate(SHAPES)}
    pv = max(1, a.nq_vec // a.qmats)
    for i in mats:
        e = ents[i]
        C, E = K.load(a.model, e)
        Ef = np.repeat(E, 32, axis=1)
        Cn, En = K.neuron_major(C, E, e["role"])
        for typ, (M, ME) in (("neuron", (Cn, En)), ("hidden", (Cn.T, En.T))):
            idx = rng.choice(M.shape[0], pv, replace=False)
            for j in idx:
                nv[typ].append((M[j].copy(), ME[j].copy()))
                meta[typ].append((int(i), int(j)))
        R, W = C.shape
        for s, (h, w) in SHAPES.items():
            for _ in range(per[s]):
                r, c = int(rng.integers(0, R - h + 1)), int(rng.integers(0, W - w + 1))
                patches[s].append((C[r:r + h, c:c + w].copy(), Ef[r:r + h, c:c + w].copy()))
                pmeta[s].append((int(i), r, c))
    out = {}
    for typ in nv:
        Cq = np.stack([x[0] for x in nv[typ]]); Eq = np.stack([x[1] for x in nv[typ]])
        perm = np.stack([rng.permutation(Cq.shape[1]) for _ in range(len(Cq))])
        out[f"{typ}_C"] = Cq; out[f"{typ}_E"] = Eq
        out[f"{typ}_Cs"] = np.take_along_axis(Cq, perm, 1); out[f"{typ}_Es"] = np.take_along_axis(Eq, perm, 1)
        out[f"{typ}_meta"] = np.array(meta[typ])
    for s in SHAPES:
        Cq = np.stack([x[0] for x in patches[s]]); Eq = np.stack([x[1] for x in patches[s]])
        n, h, w = Cq.shape
        perm = np.stack([rng.permutation(h * w) for _ in range(n)])
        out[f"p{s}_C"] = Cq; out[f"p{s}_E"] = Eq
        out[f"p{s}_Cs"] = np.take_along_axis(Cq.reshape(n, -1), perm, 1).reshape(n, h, w)
        out[f"p{s}_Es"] = np.take_along_axis(Eq.reshape(n, -1), perm, 1).reshape(n, h, w)
        out[f"p{s}_meta"] = np.array(pmeta[s])
    K.DATA.mkdir(parents=True, exist_ok=True)
    np.savez(K.DATA / f"queries_{a.model}.npz", **out)
    print({k: v.shape for k, v in out.items()})


# ================================================================== GPU scan (B2 + B3)
def load_task(model, i):
    e = K.entries(model)[i]
    P, E = K.load_packed(model, e)
    return i, P, E


class GPUScan:
    def __init__(self, torch, model, q, nq_vec, nq_patch):
        self.t = torch
        self.model = model
        ents = K.entries(model)
        self.layer = np.array([e["layer"] for e in ents]); self.expert = np.array([e["expert"] for e in ents])
        self.role = np.array([e["role"] for e in ents])
        dev = "cuda"
        self.lut = torch.as_tensor(K.E2M1, device=dev)
        # ---- B2 queries
        self.vec = {}
        for typ in ("neuron", "hidden"):
            n = min(nq_vec, len(q[f"{typ}_C"]))
            C = np.concatenate([q[f"{typ}_C"][:n], q[f"{typ}_Cs"][:n]]); E = np.concatenate([q[f"{typ}_E"][:n], q[f"{typ}_Es"][:n]])
            meta = np.concatenate([q[f"{typ}_meta"][:n], np.full((n, 2), -1)])
            sig = self.vec_signals(torch.as_tensor(C, device=dev), torch.as_tensor(E, device=dev))
            Q = len(C)
            self.vec[typ] = {"n_real": n, "meta": meta, "q": {k: v.T.contiguous() for k, v in sig.items()},
                             "top_v": {k: torch.full((4, Q), -2.0, device=dev) for k in sig},
                             "top_i": {k: torch.full((4, Q), -1, dtype=torch.long, device=dev) for k in sig},
                             "cat_v": {k: torch.full((4, Q), -2.0, device=dev) for k in sig},
                             "cat_i": {k: torch.full((4, Q), -1, dtype=torch.long, device=dev) for k in sig}}
        # ---- B3 queries
        self.pat = {}
        for k, s in enumerate(SHAPES):
            n = min(nq_patch[k], len(q[f"p{s}_C"]))
            C = np.concatenate([q[f"p{s}_C"][:n], q[f"p{s}_Cs"][:n]])
            meta = np.concatenate([q[f"p{s}_meta"][:n], np.full((n, 3), -1)])
            Ct = torch.as_tensor(C, device=dev).long()
            bits = torch.stack([((Ct >> (3 - b)) & 1).half() * 2 - 1 for b in range(4)], dim=1)     # (Q,4,h,w)
            Q = len(C)
            # B3 uses the bit-agreement signal only (v cosine needs a per-position window norm and was too slow
            # for a whole-model scan; v / w are covered by B2 and part A)
            self.pat[s] = {"n_real": n, "meta": meta, "bits": bits.contiguous(memory_format=torch.channels_last),
                           "best": {f"bits_{w}": torch.full((Q,), -2.0, device=dev) for w in ("same", "other")},
                           "where": {f"bits_{w}": torch.full((Q, 3), -1, dtype=torch.long, device=dev) for w in ("same", "other")}}

    def vec_signals(self, C, E):
        """C, E: (n, L) codes and expanded scales -> dict of (n, D) float16 rows (unit norm for v, w)."""
        t = self.t
        C = C.long()
        v = self.lut[C]
        w = v * t.exp2(E.float() - 127)
        out = {}
        for nm, x in (("v", v), ("w", w)):
            out[nm] = (x / x.norm(dim=1, keepdim=True).clamp_min(1e-30)).half()
        bits = t.cat([((C >> (3 - b)) & 1).half() * 2 - 1 for b in range(4)], dim=1)
        out["bits"] = bits / math.sqrt(bits.shape[1])
        return out

    def categories(self, i, meta):
        """per query: 0 same matrix, 1 same expert other matrix, 2 same layer other expert, 3 other layer (-1: null query)."""
        src = meta[:, 0]
        cat = np.full(len(src), 3)
        ok = src >= 0
        s = src[ok]
        c = np.where(self.layer[s] == self.layer[i], np.where(self.expert[s] == self.expert[i], 1, 2), 3)
        c[s == i] = 0
        cat[ok] = c
        # null queries: categorise like their real counterpart (same index offset)
        n = (~ok).sum()
        if n:
            cat[~ok] = cat[:n]
        return cat

    def step(self, i, P, E):
        t = self.t
        dev = "cuda"
        Pt = t.as_tensor(P, device=dev)
        C = t.stack((Pt & 15, Pt >> 4), dim=-1).reshape(P.shape[0], -1)
        Ef = t.as_tensor(E, device=dev).repeat_interleave(32, dim=1)
        role = self.role[i]
        Cn, En = (C.T, Ef.T) if role == "w2" else (C, Ef)
        # ---- B2
        for typ, (M, ME) in (("neuron", (Cn, En)), ("hidden", (Cn.T, En.T))):
            st = self.vec[typ]
            meta = st["meta"]
            cat = self.categories(i, meta)
            cat_t = t.as_tensor(cat, device=dev)
            db = self.vec_signals(M.contiguous(), ME.contiguous())
            selfq = np.nonzero(meta[:, 0] == i)[0]
            for nm, X in db.items():
                S = (X @ st["q"][nm]).float()                            # (n_vec, Q)
                if nm != "bits":
                    S = S.abs()
                if len(selfq):
                    S[t.as_tensor(meta[selfq, 1], device=dev), t.as_tensor(selfq, device=dev)] = -3.0
                tv, ti = t.topk(S, 4, dim=0)
                gi = ti + i * 100000                                      # global id = matrix * 1e5 + vector index
                allv = t.cat([st["top_v"][nm], tv]); alli = t.cat([st["top_i"][nm], gi])
                k = t.topk(allv, 4, dim=0)
                st["top_v"][nm] = k.values; st["top_i"][nm] = t.gather(alli, 0, k.indices)
                mv, mi = tv[0], gi[0]
                cur = st["cat_v"][nm][cat_t, t.arange(len(cat), device=dev)]
                upd = mv > cur
                qi = t.nonzero(upd).flatten()
                st["cat_v"][nm][cat_t[qi], qi] = mv[qi]
                st["cat_i"][nm][cat_t[qi], qi] = mi[qi]
                del S
        # ---- B3 (stored orientation)
        R, W = C.shape
        F = t.nn.functional
        Ct = C.long()
        Xb = t.stack([((Ct >> (3 - b)) & 1).half() * 2 - 1 for b in range(4)])[None]        # (1,4,R,W)
        Xb = Xb.contiguous(memory_format=t.channels_last)
        NEG = -60000.0
        for s, (h, w) in SHAPES.items():
            st = self.pat[s]
            meta = st["meta"]
            Q = len(meta)
            selfq = np.nonzero(meta[:, 0] == i)[0]
            area = h * w
            same = t.zeros(Q, dtype=t.bool, device=dev)
            if len(selfq):
                same[t.as_tensor(selfq, device=dev)] = True
            stripe = max(h, int(1.5e9 // (2 * Q * W)))
            for r0 in range(0, R - h + 1, stripe):
                r1 = min(R, r0 + stripe + h - 1)
                o = F.conv2d(Xb[:, :, r0:r1], st["bits"])[0]                     # (Q, rows, W-w+1) fp16 raw dot
                for q in selfq:
                    _, rq, cq = meta[q]
                    a0, a1 = max(0, rq - h + 1 - r0), min(o.shape[1], rq + h - r0)
                    if a1 > a0:
                        o[q, a0:a1, max(0, cq - w + 1): cq + w] = NEG
                mv = (o.amax(dim=(1, 2)).float() / (4 * area) + 1) / 2              # bit-agreement fraction
                for which, mask in (("same", same), ("other", ~same)):
                    key = f"bits_{which}"
                    upd = mask & (mv > st["best"][key])
                    if upd.any():
                        qs = t.nonzero(upd).flatten()
                        st["best"][key][qs] = mv[qs]
                        ma = o[qs].reshape(len(qs), -1).argmax(dim=1)
                        st["where"][key][qs] = t.stack([t.full_like(ma, i), ma // o.shape[2] + r0, ma % o.shape[2]], dim=1)
                del o

    def state(self):
        out = {}
        for typ, st in self.vec.items():
            for f in ("top_v", "top_i", "cat_v", "cat_i"):
                for nm, x in st[f].items():
                    out[f"vec_{typ}_{f}_{nm}"] = x.cpu().numpy()
            out[f"vec_{typ}_meta"] = st["meta"]; out[f"vec_{typ}_n_real"] = st["n_real"]
        for s, st in self.pat.items():
            for f in ("best", "where"):
                for k, x in st[f].items():
                    out[f"pat_{s}_{f}_{k}"] = x.cpu().numpy()
            out[f"pat_{s}_meta"] = st["meta"]; out[f"pat_{s}_n_real"] = st["n_real"]
        return out

    def load_state(self, z):
        t = self.t
        for typ, st in self.vec.items():
            for f in ("top_v", "top_i", "cat_v", "cat_i"):
                for nm in st[f]:
                    st[f][nm] = t.as_tensor(z[f"vec_{typ}_{f}_{nm}"], device="cuda")
        for s, st in self.pat.items():
            for f in ("best", "where"):
                for k in st[f]:
                    st[f][k] = t.as_tensor(z[f"pat_{s}_{f}_{k}"], device="cuda")


def cmd_gpu(a):
    import torch
    torch.backends.cudnn.benchmark = True
    q = dict(np.load(K.DATA / f"queries_{a.model}.npz"))
    g = GPUScan(torch, a.model, q, a.nq_vec, a.nq_patch)
    tag = a.tag or a.model
    ck = K.DATA / f"scan_{tag}_ckpt.npz"
    start = 0
    if ck.exists():
        z = np.load(ck)
        g.load_state(z); start = int(z["next"])
        print("resume at", start, flush=True)
    ents = K.entries(a.model)
    order = np.arange(len(ents)) if not a.random_order else np.random.default_rng(SEED + 1).permutation(len(ents))
    stop = len(ents) if a.limit is None else min(len(ents), start + a.limit)
    idx = [int(x) for x in order[start:stop]]
    t0 = time.time()
    last = t0
    with ProcessPoolExecutor(a.workers) as ex:
        for k, (i, P, E) in enumerate(bounded_map(ex, load_task, [(a.model, i) for i in idx], window=64), 1):
            g.step(i, P, E)
            done_n = start + k
            if k % 200 == 0 or done_n == stop:
                torch.cuda.synchronize()
                print(f"{done_n}/{len(ents)} {time.time() - t0:.0f}s ({(time.time() - last) / 200:.3f}s/matrix)", flush=True)
                last = time.time()
            if k % 2000 == 0 or done_n == stop:
                np.savez(ck, next=done_n, random_order=a.random_order, **g.state())
    st = g.state()
    st["matrices_scanned_from"] = start; st["matrices_scanned_to"] = stop; st["seconds"] = time.time() - t0
    st["random_order"] = a.random_order; st["scanned_matrix_ids"] = order[:stop]
    np.savez(K.RES / f"scan_{tag}.npz", **st)
    print("GPU_SCAN_DONE", a.model, stop, f"{time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["hash", "queries", "gpu", "verify"])
    ap.add_argument("model", choices=list(K.MODELS))
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--qmats", type=int, default=512)
    ap.add_argument("--nq-vec", type=int, default=2048)
    ap.add_argument("--nq-patch", type=int, nargs=3, default=[128, 128, 64])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--random-order", action="store_true")
    ap.add_argument("--sub", type=int, default=None, help="number of disjoint hash subsets for block32 / patch8x32")
    ap.add_argument("--part", type=int, default=0)
    a = ap.parse_args()
    {"hash": cmd_hash, "queries": cmd_queries, "gpu": cmd_gpu, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    main()
