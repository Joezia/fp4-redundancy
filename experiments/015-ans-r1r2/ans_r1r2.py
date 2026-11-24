"""Spec 015 (ANS, R1 + R2): real static-table rANS coding of every MXFP4 tensor of DeepSeek-V4.1-Flash and Kimi-K3.

    PYTHONPATH=. python experiments/015-ans-r1r2/ans_r1r2.py enumerate {deepseek|kimi}
    PYTHONPATH=. python experiments/015-ans-r1r2/ans_r1r2.py count    {deepseek|kimi} [--workers N]   # pass 1
    PYTHONPATH=. python experiments/015-ans-r1r2/ans_r1r2.py code     {deepseek|kimi} [--workers N]   # pass 2
    PYTHONPATH=. python experiments/015-ans-r1r2/ans_r1r2.py summarize {deepseek|kimi}
    PYTHONPATH=. python experiments/015-ans-r1r2/ans_r1r2.py chain    {deepseek|kimi} [--workers N]
    PYTHONPATH=. python experiments/015-ans-r1r2/ans_r1r2.py selftest

Scheme per tensor: scale stream (E8M0 byte per block, one 256-symbol static table) + code stream (16 codes with -0
mapped to +0 -> 15 symbols, one static 15-symbol table per scale value; the scale stream is decoded first).
Tables: 12-bit frequencies (sum 4096), every symbol that occurs gets at least 1. Coder: ryg_rans rans_byte
(see ans_r1r2.c). Streams of CHUNK = 65536 symbols each (scale streams: 65536 scale bytes; code streams: 65536
codes), each with its own 4-byte flush.

Sharing levels ("one table set" = code tables per scale value + one scale table), each built on all the tensors it
covers: tensor, expert, layer x role, layer, role, model. The DeepSeek MTP block (`mtp.0`) is numbered as layer 40
by redundancy.adapters.fp4.list_matrices and is treated as a layer of its own. Both checkpoints contain no MXFP4
tensor outside the routed experts (checked by `enumerate`).

Byte formats (all sizes reported are lengths of these byte strings, built in memory):
  table set   32-byte bitmap of the scale values present; then one bit-packed block: for each present scale value
              (ascending) its 12-bit (freq - 1); then for each present scale value a 15-bit mask of the code
              symbols present followed by their 12-bit (freq - 1); padded to a whole byte.
              The decoder builds its tables from these bytes (decoding uses the parsed tables).
  tensor head u32 rows, u32 codes per row, u32 table-set id, then u32 length of every stream (scale streams
              first). The number of streams follows from the shape and CHUNK.
  file head   per model and level: magic "FP4A", u8 version, u32 CHUNK, u32 number of tensors, u32 number of
              table sets, u64 offset of every table set and every tensor record.
Not stored: tensor names and the rest of the checkpoint (they are not part of the MXFP4 bytes being compressed).
"""
import argparse
import ctypes
import json
import math
import os
import pickle
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES = HERE / "results"
DATA = ROOT / "data/015"
LIB = DATA / "libans_r1r2.so"
CHUNK = 65536
M = 4096
NS = 15
LEVELS = ["tensor", "expert", "layer_role", "layer", "role", "model"]
MODELS = {"deepseek": dict(env="SNAPSHOT", fmt="deepseek", label="DeepSeek-V4.1-Flash"),
          "kimi": dict(env="KIMI_SNAPSHOT", fmt="kimi", label="Kimi-K3")}

# ---------------------------------------------------------------- C library
_L = None


def lib():
    global _L
    if _L is None:
        DATA.mkdir(parents=True, exist_ok=True)
        src = HERE / "ans_r1r2.c"
        if not LIB.exists() or LIB.stat().st_mtime < src.stat().st_mtime:
            subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", str(LIB), str(src)])
        _L = ctypes.CDLL(str(LIB))
        P = ctypes.c_void_p
        _L.count_tensor.argtypes = [P, P, ctypes.c_int64, P, P]
        _L.encode_tensor.argtypes = [P, P, ctypes.c_int64, ctypes.c_int64, P, P, P, P, P, P]
        _L.encode_tensor.restype = ctypes.c_int64
        _L.decode_verify.argtypes = [P, P, ctypes.c_int64, ctypes.c_int64, P, P, P, P, P, P, P, P, P]
    return _L


def ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


# ---------------------------------------------------------------- data
SFX = ""          # "" = all tensors, "_sample" = the 008 / 009 stratified samples (--sample)


def sample_names(model):
    S = SAMPLES[model]
    plan = json.load(open(ROOT / S["plan"]))
    tasks = plan["tasks"][:S["first"]] if S["first"] else plan["tasks"]
    return {e for t in tasks for e in t["entries"]}


def entries(model, all_tensors=False):
    ms = json.load(open(DATA / f"{model}_matrices.json"))["matrices"]
    if SFX == "_sample" and not all_tensors:
        keep = sample_names(model)
        ms = [m for m in ms if m["matrix"] in keep]
        assert len(ms) == len(keep), "sample tensors missing from the enumeration"
    return ms


_READERS = {}


def reader(model):
    """one adapter (and one snapshot index) per process and model."""
    if model not in _READERS:
        from redundancy.adapters.fp4 import FP4
        _READERS[model] = FP4({"samples": str(DATA / f"{model}_matrices.json"), "snapshot_env": MODELS[model]["env"]})
    return _READERS[model]


def read(ad, s):
    C, E = ad._read(s)
    return np.ascontiguousarray(C, dtype=np.uint8).ravel(), np.ascontiguousarray(E, dtype=np.uint8).ravel()


def count(C, E):
    h = np.zeros((256, NS), dtype=np.int64)
    sh = np.zeros(256, dtype=np.int64)
    lib().count_tensor(ptr(C), ptr(E), C.size, ptr(h), ptr(sh))
    return h, sh


def keys_of(e):
    return {"tensor": e["matrix"], "expert": f"L{e['layer']}.E{e['expert']}", "layer_role": f"L{e['layer']}.{e['role']}",
            "layer": f"L{e['layer']}", "role": e["role"], "model": "model"}


# ---------------------------------------------------------------- tables
def quantize(counts):
    """counts (n,) int -> 12-bit frequencies summing to 4096, >= 1 where counts > 0 (greedy cost-optimal fix-up)."""
    c = np.asarray(counts, dtype=np.float64)
    pres = c > 0
    f = np.zeros(len(c), dtype=np.int64)
    if pres.sum() == 0:
        return f
    if pres.sum() == 1:
        f[pres] = M
        return f
    f[pres] = np.maximum(1, np.floor(c[pres] * M / c.sum())).astype(np.int64)
    d = M - f.sum()
    while d != 0:
        if d > 0:      # add 1 where it saves most: c * log2((f + 1) / f)
            gain = np.where(pres, c * np.log2((f + 1) / np.maximum(f, 1)), -np.inf)
            i = int(np.argmax(gain))
            f[i] += 1
            d -= 1
        else:          # remove 1 where it costs least, keeping f >= 1
            cost = np.where(pres & (f > 1), c * np.log2(f / np.maximum(f - 1, 1)), np.inf)
            i = int(np.argmin(cost))
            f[i] -= 1
            d += 1
    return f


def build_tables(h, sh):
    sf = quantize(sh)
    cf = np.zeros((256, NS), dtype=np.int64)
    for s in np.nonzero(sh)[0]:
        cf[s] = quantize(h[s])
    return cf, sf


class BitW:
    def __init__(self):
        self.acc, self.n, self.out = 0, 0, bytearray()

    def put(self, v, bits):
        self.acc |= int(v) << self.n
        self.n += bits
        while self.n >= 8:
            self.out.append(self.acc & 0xFF)
            self.acc >>= 8
            self.n -= 8

    def bytes(self):
        return bytes(self.out) + (bytes([self.acc & 0xFF]) if self.n else b"")


class BitR:
    def __init__(self, b):
        self.b, self.pos = b, 0

    def get(self, bits):
        v = 0
        for k in range(bits):
            byte = self.b[(self.pos + k) >> 3]
            v |= ((byte >> ((self.pos + k) & 7)) & 1) << k
        self.pos += bits
        return v


def serialize_tables(cf, sf):
    present = np.nonzero(sf)[0]
    bitmap = np.zeros(256, dtype=np.uint8)
    bitmap[present] = 1
    head = np.packbits(bitmap, bitorder="little").tobytes()
    w = BitW()
    for s in present:
        w.put(sf[s] - 1, 12)
    for s in present:
        mask = 0
        for m in range(NS):
            if cf[s, m]:
                mask |= 1 << m
        w.put(mask, NS)
        for m in range(NS):
            if cf[s, m]:
                w.put(cf[s, m] - 1, 12)
    return head + w.bytes()


def parse_tables(b):
    bitmap = np.unpackbits(np.frombuffer(b[:32], dtype=np.uint8), bitorder="little")
    present = np.nonzero(bitmap)[0]
    r = BitR(b[32:])
    sf = np.zeros(256, dtype=np.uint32)
    cf = np.zeros((256, NS), dtype=np.uint32)
    for s in present:
        sf[s] = r.get(12) + 1
    for s in present:
        mask = r.get(NS)
        for m in range(NS):
            if mask >> m & 1:
                cf[s, m] = r.get(12) + 1
    return cf, sf


def cum_of(cf, sf):
    scum = np.concatenate([[0], np.cumsum(sf)[:-1]]).astype(np.uint32)
    ccum = np.zeros_like(cf, dtype=np.uint32)
    ccum[:, 1:] = np.cumsum(cf, axis=1)[:, :-1]
    return ccum.astype(np.uint32), scum


class Tables:
    def __init__(self, raw_bytes):
        self.bytes = raw_bytes
        cf, sf = parse_tables(raw_bytes)
        assert sf.sum() == M and all(cf[s].sum() == M for s in np.nonzero(sf)[0])
        self.cf, self.sf = np.ascontiguousarray(cf), np.ascontiguousarray(sf)
        self.ccum, self.scum = cum_of(self.cf, self.sf)
        self.ccum, self.scum = np.ascontiguousarray(self.ccum), np.ascontiguousarray(self.scum)


def make_tables(h, sh):
    cf, sf = build_tables(h, sh)
    raw = serialize_tables(cf, sf)
    t = Tables(raw)
    assert np.array_equal(t.cf, cf) and np.array_equal(t.sf, sf), "table serialisation round trip"
    return t


# ---------------------------------------------------------------- coding one tensor with one table set
def n_streams(n_codes):
    nb = n_codes // 32
    return -(-nb // CHUNK), -(-n_codes // CHUNK)


def code_tensor(C, E, R, W, T, table_id):
    n = C.size
    ns, nc = n_streams(n)
    out = np.empty(2 * n + 64 * (ns + nc) + 1024, dtype=np.uint8)
    lens = np.zeros(ns + nc, dtype=np.int64)
    t0 = time.perf_counter()
    tot = lib().encode_tensor(ptr(C), ptr(E), n, CHUNK, ptr(T.cf), ptr(T.ccum), ptr(T.sf), ptr(T.scum), ptr(out), ptr(lens))
    t_enc = time.perf_counter() - t0
    assert tot > 0, "table does not cover the data"
    head = np.array([R, W, table_id] + lens.tolist(), dtype="<u4").tobytes()
    ds = np.empty(n // 32, dtype=np.uint8)
    dsym = np.empty(n, dtype=np.uint8)
    mism = np.zeros(2, dtype=np.int64)
    # decode from the stream bytes and the tensor head (lengths parsed back from the head)
    lens2 = np.frombuffer(head, dtype="<u4")[3:].astype(np.int64)
    t0 = time.perf_counter()
    lib().decode_verify(ptr(out), ptr(lens2), n, CHUNK, ptr(T.cf), ptr(T.ccum), ptr(T.sf), ptr(T.scum), ptr(C), ptr(E),
                        ptr(ds), ptr(dsym), ptr(mism))
    t_dec = time.perf_counter() - t0
    return {"scale_bytes": int(lens[:ns].sum()), "code_bytes": int(lens[ns:].sum()), "n_scale_streams": int(ns),
            "n_code_streams": int(nc), "head_bytes": len(head), "scale_mismatch": int(mism[0]),
            "code_mismatch": int(mism[1]), "t_enc": t_enc, "t_dec": t_dec}


# ---------------------------------------------------------------- commands
def cmd_enumerate(a):
    from diagnostics.st import index_snapshot
    from redundancy.adapters.fp4 import list_matrices
    snap = os.environ[MODELS[a.model]["env"]]
    lm = list_matrices(snap, MODELS[a.model]["fmt"])
    idx = index_snapshot(snap)
    mx = [n for n, v in idx.items() if v["dtype"] in ("I8", "U8") and not n.endswith("weight_scale")]
    listed = {m["weight"] for m in lm["matrices"]}
    extra = sorted(set(mx) - listed)
    DATA.mkdir(parents=True, exist_ok=True)
    json.dump(lm, open(DATA / f"{a.model}_matrices.json", "w"))
    n_el = sum(m["codewords"] for m in lm["matrices"])
    info = {"model": MODELS[a.model]["label"], "n_tensors": len(lm["matrices"]), "n_elements": n_el,
            "raw_bytes": n_el // 2 + n_el // 32, "raw_code_bytes": n_el // 2, "raw_scale_bytes": n_el // 32,
            "packed_tensors_in_checkpoint": len(mx), "packed_tensors_not_listed": extra[:20],
            "n_not_listed": len(extra),
            "layers": sorted({m["layer"] for m in lm["matrices"]}), "blocks": sorted({m["block"] for m in lm["matrices"]}),
            "shapes": sorted({tuple(m["codeword_shape"]) for m in lm["matrices"]})}
    RES.mkdir(parents=True, exist_ok=True)
    json.dump(info, open(RES / f"enumerate_{a.model}.json", "w"), indent=1, default=list)
    print(json.dumps(info, default=list)[:800])


def experts_of(model):
    ex = {}
    for e in entries(model):
        ex.setdefault((e["layer"], e["expert"]), []).append(e)
    return list(ex.values())


def count_task(model, group):
    ad = reader(model)
    out = {}
    for e in group:
        C, E = read(ad, e)
        h, sh = count(C, E)
        k = (e["layer"], e["role"])
        if k in out:
            out[k][0] += h
            out[k][1] += sh
        else:
            out[k] = [h, sh]
    return out


def pool_map(fn, args, workers, log_every=200, label=""):
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(fn, *zip(*args), chunksize=4), 1):
            yield r
            if i % log_every == 0:
                print(f"{label} {i}/{len(args)} {time.time() - t0:.0f} s", flush=True)


def cmd_count(a):
    groups = experts_of(a.model)
    t0 = time.time()
    tot = {}
    for r in pool_map(count_task, [(a.model, g) for g in groups], a.workers, label="count"):
        for k, (h, sh) in r.items():
            if k in tot:
                tot[k][0] += h
                tot[k][1] += sh
            else:
                tot[k] = [h, sh]
    pickle.dump(tot, open(DATA / f"{a.model}_counts_layer_role{SFX}.pkl", "wb"))
    print(f"COUNT_DONE {a.model} {len(groups)} experts wall {time.time() - t0:.0f} s", flush=True)
    json.dump({"wall_s": time.time() - t0, "workers": a.workers}, open(RES / f"count_time_{a.model}{SFX}.json", "w"))


def global_tables(model):
    """table sets of the levels layer x role, layer, role, model from the pass-1 counts."""
    tot = pickle.load(open(DATA / f"{model}_counts_layer_role{SFX}.pkl", "rb"))
    agg = {"layer_role": {}, "layer": {}, "role": {}, "model": {}}
    for (L, role), (h, sh) in tot.items():
        for lvl, key in (("layer_role", f"L{L}.{role}"), ("layer", f"L{L}"), ("role", role), ("model", "model")):
            d = agg[lvl]
            if key in d:
                d[key][0] = d[key][0] + h
                d[key][1] = d[key][1] + sh
            else:
                d[key] = [h.copy(), sh.copy()]
    return {lvl: {k: serialize_tables(*build_tables(h, sh)) for k, (h, sh) in d.items()} for lvl, d in agg.items()}


_G = {}


def _init_code(model, sfx):
    global SFX
    SFX = sfx
    _G["model"] = model
    _G["tables"] = {lvl: {k: Tables(b) for k, b in d.items()} for lvl, d in
                    pickle.load(open(DATA / f"{model}_global_tables{SFX}.pkl", "rb")).items()}
    _G["ids"] = {lvl: {k: i for i, k in enumerate(sorted(d))} for lvl, d in _G["tables"].items()}


def code_task(group):
    model = _G["model"]
    ad = reader(model)
    t_read = time.perf_counter()
    data = [(e, *read(ad, e)) for e in group]
    t_read = time.perf_counter() - t_read
    hs = [count(C, E) for _, C, E in data]
    ex_T = make_tables(sum(h for h, _ in hs), sum(s for _, s in hs))
    recs = []
    for (e, C, E), (h, sh) in zip(data, hs):
        R, W = e["codeword_shape"]
        tT = make_tables(h, sh)
        k = keys_of(e)
        rec = {"matrix": e["matrix"], "layer": e["layer"], "expert": e["expert"], "role": e["role"], "shape": [R, W],
               "raw_bytes": C.size // 2 + C.size // 32, "levels": {}}
        for lvl in LEVELS:
            if lvl == "tensor":
                T, tid = tT, 0
            elif lvl == "expert":
                T, tid = ex_T, 0
            else:
                T, tid = _G["tables"][lvl][k[lvl]], _G["ids"][lvl][k[lvl]]
            r = code_tensor(C, E, R, W, T, tid)
            if lvl == "tensor":
                r["table_bytes"] = len(tT.bytes)
            recs_lvl = r
            rec["levels"][lvl] = recs_lvl
        rec["expert_table_bytes"] = len(ex_T.bytes)
        rec["t_read"] = t_read / len(group)
        recs.append(rec)
    return recs


def cmd_code(a):
    gt = global_tables(a.model)
    pickle.dump(gt, open(DATA / f"{a.model}_global_tables{SFX}.pkl", "wb"))
    groups = experts_of(a.model)
    path = RES / f"ans_{a.model}{SFX}_items.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(l)["matrix"] for l in open(path)}
    todo = [g for g in groups if not all(e["matrix"] in done for e in g)]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{a.model}: {len(todo)} experts to code", flush=True)
    t0 = time.time()
    with open(path, "a") as fo, ProcessPoolExecutor(a.workers, initializer=_init_code, initargs=(a.model, SFX)) as ex:
        for i, recs in enumerate(ex.map(code_task, todo, chunksize=2), 1):
            for r in recs:
                fo.write(json.dumps(r) + "\n")
            if i % 200 == 0:
                fo.flush()
                print(f"code {i}/{len(todo)} {time.time() - t0:.0f} s", flush=True)
    wall = time.time() - t0
    print(f"CODE_DONE {a.model} wall {wall:.0f} s", flush=True)
    tf = RES / f"code_time_{a.model}{SFX}.json"
    prev = json.load(open(tf)) if tf.exists() else {"runs": []}
    prev["runs"].append({"wall_s": wall, "workers": a.workers, "experts": len(todo)})
    json.dump(prev, open(tf, "w"), indent=1)


def file_head_bytes(n_tensors, n_tables):
    return 4 + 1 + 4 + 4 + 4 + 8 * (n_tensors + n_tables)


def cmd_summarize(a):
    if SFX == "_sample":
        return summarize_sample(a)
    ens = entries(a.model)
    want = [e["matrix"] for e in ens]
    rows = {}
    for l in open(RES / f"ans_{a.model}{SFX}_items.jsonl"):
        r = json.loads(l)
        rows[r["matrix"]] = r
    missing = [m for m in want if m not in rows]
    assert not missing, f"{len(missing)} tensors missing"
    rows = [rows[m] for m in want]
    gt = pickle.load(open(DATA / f"{a.model}_global_tables{SFX}.pkl", "rb"))
    raw = sum(r["raw_bytes"] for r in rows)
    n_el = sum(r["shape"][0] * r["shape"][1] for r in rows)
    out = {"model": MODELS[a.model]["label"], "n_tensors": len(rows), "n_elements": n_el, "raw_bytes": raw,
           "chunk": CHUNK, "levels": {}, "failed": []}
    experts = {}
    for r in rows:
        experts.setdefault((r["layer"], r["expert"]), r["expert_table_bytes"])
    for lvl in LEVELS:
        code = sum(r["levels"][lvl]["code_bytes"] for r in rows)
        scale = sum(r["levels"][lvl]["scale_bytes"] for r in rows)
        head = sum(r["levels"][lvl]["head_bytes"] for r in rows)
        nstreams = sum(r["levels"][lvl]["n_scale_streams"] + r["levels"][lvl]["n_code_streams"] for r in rows)
        if lvl == "tensor":
            tbytes, ntab = sum(r["levels"][lvl]["table_bytes"] for r in rows), len(rows)
        elif lvl == "expert":
            tbytes, ntab = sum(experts.values()), len(experts)
        else:
            tbytes, ntab = sum(len(b) for b in gt[lvl].values()), len(gt[lvl])
        fhead = file_head_bytes(len(rows), ntab)
        total = code + scale + head + tbytes + fhead
        bad = [r["matrix"] for r in rows if r["levels"][lvl]["scale_mismatch"] or r["levels"][lvl]["code_mismatch"]]
        out["failed"] += [(lvl, m) for m in bad]
        t_enc = sum(r["levels"][lvl]["t_enc"] for r in rows)
        t_dec = sum(r["levels"][lvl]["t_dec"] for r in rows)
        out["levels"][lvl] = {
            "n_tables": ntab, "code_stream_bytes": code, "scale_stream_bytes": scale, "flush_bytes": 4 * nstreams,
            "n_streams": nstreams, "tensor_head_bytes": head, "file_head_bytes": fhead, "table_bytes": tbytes,
            "total_bytes": total, "saved_bytes": raw - total, "ratio": total / raw, "saving": 1 - total / raw,
            "bits_per_weight": 8 * total / n_el, "decode_ok": not bad, "n_failed": len(bad),
            "encode_cpu_s": t_enc, "decode_cpu_s": t_dec,
            "encode_MBps_per_thread": raw / 1e6 / t_enc, "decode_MBps_per_thread": raw / 1e6 / t_dec}
    out["read_cpu_s"] = sum(r["t_read"] for r in rows)
    json.dump(out, open(RES / f"ans_{a.model}_summary.json", "w"), indent=1)
    # per-tensor file (columnar)
    cols = {"matrix": [r["matrix"] for r in rows], "raw_bytes": [r["raw_bytes"] for r in rows],
            "shape": [r["shape"] for r in rows]}
    for lvl in LEVELS:
        for f in ("code_bytes", "scale_bytes", "head_bytes"):
            cols[f"{lvl}.{f}"] = [r["levels"][lvl][f] for r in rows]
        cols[f"{lvl}.decode_ok"] = [int(not (r["levels"][lvl]["scale_mismatch"] or r["levels"][lvl]["code_mismatch"]))
                                    for r in rows]
    cols["tensor.table_bytes"] = [r["levels"]["tensor"]["table_bytes"] for r in rows]
    cols["expert.table_bytes_of_its_expert"] = [r["expert_table_bytes"] for r in rows]
    json.dump({"summary": out, "per_tensor": cols,
               "note": "per-tensor total at level L = code_bytes + scale_bytes + head_bytes (+ table_bytes for "
                       "level tensor); shared tables and the file head are counted once per level in summary"},
              open(RES / f"ans_{a.model}.json", "w"))
    print(json.dumps({k: v for k, v in out.items() if k != "failed"}, indent=1)[:3000])
    print("failed:", len(out["failed"]))


def head_bytes_exact(R, W):
    ns, nc = n_streams(R * W)
    return 12 + 4 * (ns + nc), ns + nc


def summarize_sample(a):
    """Extrapolate from the stratified sample: per (layer, role) stratum, the sampled ratio of code / scale stream
    bytes to raw bytes times the stratum's full raw bytes. Tensor heads, flush bytes, numbers of tables and the
    file head are exact (they follow from the real shapes and counts). Table bytes: tensor and expert levels use
    the mean table size of the sampled tensors / experts in the stratum (layer x role, resp. layer) times the real
    number of tensors / experts; the shared levels use the tables built from the sample, one per real key."""
    allm = entries(a.model, all_tensors=True)
    smp = entries(a.model)
    rows = {}
    for l in open(RES / f"ans_{a.model}{SFX}_items.jsonl"):
        r = json.loads(l)
        rows[r["matrix"]] = r
    assert all(m["matrix"] in rows for m in smp), "sample incomplete"
    rows = [rows[m["matrix"]] for m in smp]
    gt = pickle.load(open(DATA / f"{a.model}_global_tables{SFX}.pkl", "rb"))
    full, n_el, raw = {}, 0, 0
    exp_per_layer, head_all, nstr_all = {}, 0, 0
    for m in allm:
        R, W = m["codeword_shape"]
        k = (m["layer"], m["role"])
        rb = R * W // 2 + R * W // 32
        f = full.setdefault(k, [0, 0])
        f[0] += 1
        f[1] += rb
        n_el += R * W
        raw += rb
        exp_per_layer.setdefault(m["layer"], set()).add(m["expert"])
        hb, ns = head_bytes_exact(R, W)
        head_all += hb
        nstr_all += ns
    strata = {}
    for r in rows:
        strata.setdefault((r["layer"], r["role"]), []).append(r)
    assert set(strata) == set(full), "every layer x role stratum must be sampled"
    real_keys = {"layer_role": {f"L{L}.{ro}" for L, ro in full}, "layer": {f"L{L}" for L, _ in full},
                 "role": {ro for _, ro in full}, "model": {"model"}}
    out = {"model": MODELS[a.model]["label"], "mode": "stratified sample, extrapolated", "n_tensors": len(allm),
           "n_tensors_sampled": len(rows), "n_elements": n_el, "raw_bytes": raw,
           "raw_bytes_sampled": sum(r["raw_bytes"] for r in rows), "chunk": CHUNK, "levels": {}, "failed": []}
    for lvl in LEVELS:
        est = {"code_bytes": 0.0, "scale_bytes": 0.0}
        for k, rs in strata.items():
            sraw = sum(r["raw_bytes"] for r in rs)
            for f in est:
                est[f] += sum(r["levels"][lvl][f] for r in rs) / sraw * full[k][1]
        if lvl == "tensor":
            tbytes = sum(np.mean([r["levels"]["tensor"]["table_bytes"] for r in rs]) * full[k][0] for k, rs in strata.items())
            ntab = len(allm)
        elif lvl == "expert":
            tbytes, ntab = 0.0, 0
            for L, exs in exp_per_layer.items():
                sel = {}
                for r in rows:
                    if r["layer"] == L:
                        sel[r["expert"]] = r["expert_table_bytes"]
                tbytes += np.mean(list(sel.values())) * len(exs)
                ntab += len(exs)
        else:
            assert set(gt[lvl]) == real_keys[lvl]
            tbytes, ntab = sum(len(b) for b in gt[lvl].values()), len(gt[lvl])
        fhead = file_head_bytes(len(allm), ntab)
        total = est["code_bytes"] + est["scale_bytes"] + head_all + tbytes + fhead
        bad = [r["matrix"] for r in rows if r["levels"][lvl]["scale_mismatch"] or r["levels"][lvl]["code_mismatch"]]
        out["failed"] += [(lvl, m) for m in bad]
        sraw = sum(r["raw_bytes"] for r in rows)
        t_enc = sum(r["levels"][lvl]["t_enc"] for r in rows)
        t_dec = sum(r["levels"][lvl]["t_dec"] for r in rows)
        out["levels"][lvl] = {
            "n_tables": ntab, "code_stream_bytes": est["code_bytes"], "scale_stream_bytes": est["scale_bytes"],
            "flush_bytes": 4 * nstr_all, "n_streams": nstr_all, "tensor_head_bytes": head_all,
            "file_head_bytes": fhead, "table_bytes": float(tbytes), "total_bytes": total, "saved_bytes": raw - total,
            "ratio": total / raw, "saving": 1 - total / raw, "bits_per_weight": 8 * total / n_el,
            "sample_ratio_streams_and_heads": sum(r["levels"][lvl]["code_bytes"] + r["levels"][lvl]["scale_bytes"]
                                                  + r["levels"][lvl]["head_bytes"] for r in rows) / sraw,
            "decode_ok_sample": not bad, "n_failed_sample": len(bad), "encode_cpu_s_sample": t_enc,
            "decode_cpu_s_sample": t_dec, "encode_MBps_per_thread": sraw / 1e6 / t_enc,
            "decode_MBps_per_thread": sraw / 1e6 / t_dec}
    json.dump(out, open(RES / f"ans_{a.model}{SFX}_summary.json", "w"), indent=1)
    cols = {"matrix": [r["matrix"] for r in rows], "raw_bytes": [r["raw_bytes"] for r in rows],
            "shape": [r["shape"] for r in rows]}
    for lvl in LEVELS:
        for f in ("code_bytes", "scale_bytes", "head_bytes"):
            cols[f"{lvl}.{f}"] = [r["levels"][lvl][f] for r in rows]
        cols[f"{lvl}.decode_ok"] = [int(not (r["levels"][lvl]["scale_mismatch"] or r["levels"][lvl]["code_mismatch"]))
                                    for r in rows]
    cols["tensor.table_bytes"] = [r["levels"]["tensor"]["table_bytes"] for r in rows]
    cols["expert.table_bytes_of_its_expert"] = [r["expert_table_bytes"] for r in rows]
    json.dump({"summary": out, "per_tensor_sampled": cols}, open(RES / f"ans_{a.model}{SFX}.json", "w"))
    print(json.dumps({k: v for k, v in out.items() if k != "failed"}, indent=1)[:3500])
    print("failed:", len(out["failed"]))


# ---------------------------------------------------------------- chain comparison (sample tensors)
SAMPLES = {"deepseek": dict(plan="experiments/008-scale-media/results/fp4/B/plan.json", first=172,
                            params="experiments/008-scale-media/results/fp4/increment_dev/params.json"),
           "kimi": dict(plan="experiments/009-kimi-k3/results/plan_552.json", first=None,
                        params="experiments/009-kimi-k3/results/increment_dev/params.json")}


def chain_task(model, name):
    from redundancy import increment as I
    P = dict(I.FP4_DEFAULT)
    P.update(json.load(open(ROOT / SAMPLES[model]["params"])))
    ad = reader(model)
    s = ad.samples[name]
    C, E = ad._read(s)
    order = ["R1c", "R1s", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R10"]
    bits, _ = I.fp4_codelen(C.astype(np.int64), E, frozenset({"R1c", "R1s", "R2", "R10"}), P, order)
    return name, float(bits)


def cmd_chain(a):
    plan = json.load(open(ROOT / SAMPLES[a.model]["plan"]))
    tasks = plan["tasks"][:SAMPLES[a.model]["first"]] if SAMPLES[a.model]["first"] else plan["tasks"]
    names = [e for t in tasks for e in t["entries"]]
    t0 = time.time()
    res = dict(pool_map(chain_task, [(a.model, n) for n in names], a.workers, label="chain"))
    rows = {}
    for l in open(RES / f"ans_{a.model}{SFX}_items.jsonl"):
        r = json.loads(l)
        if r["matrix"] in res:
            rows[r["matrix"]] = r
    assert len(rows) == len(names)
    n_el = sum(rows[n]["shape"][0] * rows[n]["shape"][1] for n in names)
    chain_bytes = sum(res.values()) / 8
    out = {"n_tensors": len(names), "n_elements": n_el, "chain_R1c_R1s_R2_R10_bytes": chain_bytes,
           "chain_bits_per_weight": 8 * chain_bytes / n_el, "wall_s": time.time() - t0, "levels": {}}
    for lvl in LEVELS:
        streams = sum(rows[n]["levels"][lvl]["code_bytes"] + rows[n]["levels"][lvl]["scale_bytes"] for n in names)
        head = sum(rows[n]["levels"][lvl]["head_bytes"] for n in names)
        tab = sum(rows[n]["levels"]["tensor"]["table_bytes"] for n in names) if lvl == "tensor" else None
        out["levels"][lvl] = {"stream_bytes": streams, "head_bytes": head, "tensor_table_bytes": tab,
                              "streams_minus_chain_bytes": streams - chain_bytes,
                              "streams_minus_chain_bpw": 8 * (streams - chain_bytes) / n_el}
    json.dump(out, open(RES / f"chain_compare_{a.model}.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


def cmd_selftest(a):
    rng = np.random.default_rng(0)
    for trial in range(4):
        R, W = 37, 32 * (5 + trial)
        E = rng.choice([118, 119, 120, 121], size=(R, W // 32), p=[.1, .4, .4, .1]).astype(np.uint8)
        p = np.array([30, 20, 10, 6, 4, 3, 2, 1, 5, 20, 10, 6, 4, 3, 2, 1], dtype=float)
        C = rng.choice(16, size=(R, W), p=p / p.sum()).astype(np.uint8)
        Cf, Ef = C.ravel(), E.ravel()
        h, sh = count(Cf, Ef)
        T = make_tables(h, sh)
        global CHUNK
        old = CHUNK
        CHUNK = 97                               # many small streams in the test
        r = code_tensor(Cf, Ef, R, W, T, 0)
        CHUNK = old
        m = np.where(Cf == 8, 0, np.where(Cf > 8, Cf - 1, Cf))
        ent = -sum(np.log2(T.cf[e, s] / M) for e, s in zip(np.repeat(Ef, 32), m)) / 8
        ents = -sum(np.log2(T.sf[e] / M) for e in Ef) / 8
        print(f"trial {trial}: mismatches {r['scale_mismatch']} {r['code_mismatch']}; code bytes {r['code_bytes']} "
              f"(ideal {ent:.1f} + flush {4 * r['n_code_streams']}), scale bytes {r['scale_bytes']} (ideal {ents:.1f})")
        assert r["scale_mismatch"] == 0 and r["code_mismatch"] == 0
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("enumerate", "count", "code", "summarize", "chain"):
        p = sub.add_parser(c)
        p.add_argument("model", choices=list(MODELS))
        p.add_argument("--workers", type=int, default=64)
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--sample", action="store_true", help="only the 008 / 009 stratified sample; summarize extrapolates")
    sub.add_parser("selftest")
    a = ap.parse_args()
    global SFX
    SFX = "_sample" if getattr(a, "sample", False) else ""
    {"enumerate": cmd_enumerate, "count": cmd_count, "code": cmd_code, "summarize": cmd_summarize,
     "chain": cmd_chain, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
