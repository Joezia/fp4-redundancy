"""Run the incremental-saving chains on a dataset (spec 008 Part 2 and V6).

    python -m redundancy.increment_run run --dataset kodak --split report --params P.json --out DIR [--workers N]
    python -m redundancy.increment_run devsel --dataset kodak --out DIR [--workers N] [--limit N]
    python -m redundancy.increment_run fp4 --plan B/plan.json --params P.json --out DIR [--workers N]
    python -m redundancy.increment_run aggregate --dir DIR --name <label> --out <json>

Every finished item is appended at once to DIR/items.jsonl (resumable). An item's record has, for each
order, the code lengths L(M0..Mk) in bits, the increments, the raw bits and the real codec's size.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from . import increment as I  # noqa: E402
from .adapters import media  # noqa: E402

ORDERS = {
    "default": ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R10"],
    "alt": ["R10", "R1", "R2", "R4", "R3", "R5", "R6", "R7", "R8"],
    "alt_after_R1": ["R1", "R10", "R2", "R4", "R3", "R5", "R6", "R7", "R8"],
}
FP4_ORDERS = {
    "default": ["R1c", "R1s", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R10"],
    "alt": ["R10", "R1c", "R1s", "R2", "R4", "R3", "R5", "R6", "R7", "R8"],
    "alt_after_R1": ["R1c", "R1s", "R10", "R2", "R4", "R3", "R5", "R6", "R7", "R8"],
}


def defaults(dataset):
    return dict(I.IMAGE_DEFAULT if media.kind(dataset) == "image" else I.AUDIO_DEFAULT)


def codelen_fn(dataset, arr, P):
    if media.kind(dataset) == "image":
        return lambda f: I.image_codelen(arr, f, P)
    return lambda f: I.audio_codelen(arr, f, P)


def strip(r):
    return {k: r[k] for k in ("order", "raw_bits", "L", "increment", "total", "used", "if_forced")} | {
        "info": r["details"][-1] if r.get("details") else {}}


def item_task(dataset, path, P, orders, max_seconds):
    t0 = time.process_time()
    arr, meta = media.load(dataset, path, max_seconds)
    raw = media.raw_bits(dataset, arr)
    out = {"item": os.path.basename(path), "path": path, "meta": meta, "raw_bits": raw, "params": P, "chains": {}}
    for name in orders:
        out["chains"][name] = strip(I.run_chain(codelen_fn(dataset, arr, P), ORDERS[name], raw))
    out["codec_bits"] = media.codec_bits(dataset, arr, meta)
    out["cpu_seconds"] = time.process_time() - t0
    return out


def fp4_task(entries, partner_of, P, orders, samples, seed, snapshot_env="SNAPSHOT"):
    from .adapters.fp4 import FP4
    t0 = time.process_time()
    ad = FP4({"samples": samples, "snapshot_env": snapshot_env})
    codes = {}
    out = []
    for n in entries:
        C, E = ad._read(ad.samples[n])
        codes[n] = (C.astype(np.int64), E)
    for n in entries:
        C, E = codes[n]
        pn = partner_of.get(n)
        part = codes[pn][0] if pn in codes else None
        rec = {"item": n, "partner": pn if part is not None else None, "raw_bits": C.size * 4.25, "params": P,
               "chains": {}}
        for name in orders:
            cache = {}
            order = FP4_ORDERS[name]
            rec["chains"][name] = strip(I.run_chain(
                lambda f: I.fp4_codelen(C, E, f, P, order, partner=part, seed=seed, cache=cache), order, C.size * 4.25))
        out.append(rec)
    dt = time.process_time() - t0
    for r in out:
        r["cpu_seconds"] = dt / len(out)
    return out


def _done(path):
    done = set()
    if path.exists():
        for line in open(path):
            try:
                done.add(json.loads(line)["item"])
            except Exception:
                pass
    return done


def _pool(jobs, workers, path, stop_at=None):
    """jobs: list of (key, fn, args). Appends each result (dict or list of dicts) to path."""
    log = open(path.parent / "progress.log", "a")
    with open(path, "a") as fo, ProcessPoolExecutor(workers) as ex:
        it = iter(jobs)
        pend = {}

        def sub():
            if stop_at and time.time() > stop_at:
                return
            try:
                key, fn, args = next(it)
            except StopIteration:
                return
            pend[ex.submit(fn, *args)] = key

        for _ in range(workers):
            sub()
        while pend:
            fin, _ = wait(pend, return_when=FIRST_COMPLETED)
            for f in fin:
                key = pend.pop(f)
                try:
                    r = f.result()
                    for x in (r if isinstance(r, list) else [r]):
                        fo.write(json.dumps(x) + "\n")
                    fo.flush()
                    os.fsync(fo.fileno())
                    msg = f"{time.strftime('%H:%M:%S')} done {key}"
                except Exception:
                    msg = f"{time.strftime('%H:%M:%S')} FAILED {key}\n{traceback.format_exc()}"
                print(msg, flush=True)
                log.write(msg + "\n")
                log.flush()
                sub()


def cmd_run(a):
    P = defaults(a.dataset)
    if a.params:
        P.update(json.load(open(a.params)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "items.jsonl"
    done = _done(path)
    fs = media.items(a.dataset, a.split)
    if a.limit:
        fs = fs[:a.limit]
    orders = a.orders.split(",")
    jobs = [(os.path.basename(f), item_task, (a.dataset, f, P, orders, a.max_seconds)) for f in fs
            if os.path.basename(f) not in done]
    # largest first (better packing)
    jobs.sort(key=lambda j: -os.path.getsize(j[2][1]))
    print(f"{len(jobs)} items to run of {len(fs)}", flush=True)
    _pool(jobs, a.workers, path, a.stop_at)


def cmd_fp4(a):
    plan = json.load(open(a.plan))
    P = dict(I.FP4_DEFAULT)
    if a.params:
        P.update(json.load(open(a.params)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "items.jsonl"
    done = _done(path)
    for d in (a.skip.split(",") if a.skip else []):
        done |= _done(Path(d) / "items.jsonl")
    orders = a.orders.split(",")
    tasks = plan["tasks"][:a.first] if a.first else plan["tasks"]
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        tasks = tasks[i::n]
    jobs = []
    for t in tasks:
        ents = t["entries"]
        if all(e in done for e in ents):
            continue
        w1 = [e for e in ents if e.endswith(".w1")]
        w3 = [e for e in ents if e.endswith(".w3")]
        partner = {w1[0]: w3[0]} if w1 and w3 else {}
        jobs.append((t["id"], fp4_task, (ents, partner, P, orders, a.samples, plan.get("seed", 0), a.snapshot_env)))
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"{len(jobs)} experts to run", flush=True)
    _pool(jobs, a.workers, path, a.stop_at)


# ---------------------------------------------------------------- parameter selection on dev items

GRIDS = {
    "audio": [("R2", "frame", [512, 1024, 4096]), ("R3", "ctx_order", [1, 2]), ("R4", "lpc_order", [8, 16, 32]),
              ("R5", "rank", [1, 2, 4]), ("R6", "lzp_k", [2, 4, 8]), ("R7", "ltp_block", [512, 1024, 2048]),
              ("R8", "sparse_m", [1, 4, 16])],
    "image": [("R2", "block", [16, 32, 64]), ("R3", "ctx_neighbors", [2, 3, 4]), ("R4", "lp_window_rows", [16, 64, 512]),
              ("R5", "rank", [1, 2, 4]), ("R6", "lzp_k", [3, 6, 12]), ("R7", "bm_block", [8, 16, 32]),
              ("R8", "sparse_m", [1, 2, 4])],
}


def _dev_cost(dataset, f, P, flags, max_seconds):
    arr, meta = media.load(dataset, f, max_seconds)
    return codelen_fn(dataset, arr, P)(frozenset(flags))[0]


def cmd_devsel(a):
    """Greedy, in the default order: for each pattern's parameter, the value that minimises the summed code
    length of the model that first switches that pattern on, over the dev items (earlier choices fixed)."""
    P = defaults(a.dataset)
    fs = media.items(a.dataset, "dev")
    if a.limit:
        step = max(1, len(fs) // a.limit)
        fs = fs[::step][:a.limit]
    order = ORDERS["default"]
    log = {"items": [os.path.basename(f) for f in fs], "max_seconds": a.max_seconds, "steps": []}
    with ProcessPoolExecutor(a.workers) as ex:
        for pat, key, grid in GRIDS[media.kind(a.dataset)]:
            flags = order[:order.index(pat) + 1]
            costs = {}
            for v in grid:
                Pv = dict(P)
                Pv[key] = v
                costs[v] = sum(ex.map(_dev_cost, [a.dataset] * len(fs), fs, [Pv] * len(fs), [flags] * len(fs),
                                      [a.max_seconds] * len(fs)))
            best = min(costs, key=costs.get)
            P[key] = best
            log["steps"].append({"pattern": pat, "parameter": key, "bits": {str(k): v for k, v in costs.items()},
                                 "chosen": best})
            print(pat, key, {k: round(v) for k, v in costs.items()}, "->", best, flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump(P, open(Path(a.out) / "params.json", "w"), indent=1)
    json.dump(log, open(Path(a.out) / "devsel.json", "w"), indent=1)


FP4_GRID = [("R3", "ctx_order", [1, 2]), ("R4", "lp_order", [2, 4, 8]), ("R5", "rank", [1, 2]),
            ("R6", "lzp_k", [8, 12, 16]), ("R8", "dict_atoms", [16, 64])]


def _fp4_dev_cost(name, partner, P, flags, samples, snapshot_env="SNAPSHOT"):
    from .adapters.fp4 import FP4
    ad = FP4({"samples": samples, "snapshot_env": snapshot_env})
    C, E = ad._read(ad.samples[name])
    part = ad._read(ad.samples[partner])[0].astype(np.int64) if partner else None
    return I.fp4_codelen(C.astype(np.int64), E, frozenset(flags), P, FP4_ORDERS["default"], partner=part)[0]


def cmd_fp4_devsel(a):
    """FP4 parameters on dev matrices (the pilot matrices, not part of the reported sample); the
    hierarchical prior strength beta is chosen first on the model with R1-R3."""
    P = dict(I.FP4_DEFAULT)
    names = a.entries.split(",")
    part = {n: n[:-3] + ".w3" if n.endswith(".w1") else None for n in names}
    order = FP4_ORDERS["default"]
    log = {"items": names, "steps": []}
    grid = [("R3", "beta", [16.0, 64.0, 256.0])] + FP4_GRID
    with ProcessPoolExecutor(a.workers) as ex:
        for pat, key, vals in grid:
            flags = order[:order.index(pat) + 1]
            costs = {}
            for v in vals:
                Pv = dict(P)
                Pv[key] = v
                costs[v] = sum(ex.map(_fp4_dev_cost, names, [part[n] for n in names], [Pv] * len(names),
                                      [flags] * len(names), [a.samples] * len(names),
                                      [a.snapshot_env] * len(names)))
            best = min(costs, key=costs.get)
            P[key] = best
            log["steps"].append({"pattern": pat, "parameter": key, "bits": {str(k): v for k, v in costs.items()},
                                 "chosen": best})
            print(pat, key, {k: round(v) for k, v in costs.items()}, "->", best, flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump(P, open(Path(a.out) / "params.json", "w"), indent=1)
    json.dump(log, open(Path(a.out) / "devsel.json", "w"), indent=1)


# ---------------------------------------------------------------- aggregation

def aggregate(rows, name, fp4=False):
    orders = sorted({o for r in rows for o in r["chains"]})
    out = {"dataset": name, "n_items": len(rows), "unit": "% of raw bits", "orders": {}}
    raw = sum(r["raw_bits"] for r in rows)
    for o in orders:
        rr = [r["chains"][o] for r in rows if o in r["chains"]]
        c = I.combine(rr)
        inc = {k: 100 * v for k, v in c["increment"].items()}
        out["orders"][o] = {"order": c["order"], "raw_bits": c["raw_bits"], "L_bits": c["L"],
                            "increment_raw": inc, "increment_truncated": {k: max(0.0, v) for k, v in inc.items()},
                            "total": 100 * c["total"], "n_items": c["n_items"]}
    codecs = {}
    for r in rows:
        for k, v in (r.get("codec_bits") or {}).items():
            codecs[k] = codecs.get(k, 0.0) + v
    out["codec_saving"] = {k: 100 * (1 - v / raw) for k, v in codecs.items()}
    out["raw_bits"] = raw
    out["cpu_seconds"] = sum(r.get("cpu_seconds", 0) for r in rows)
    return out


def cmd_aggregate(a):
    rows = [json.loads(line) for d in a.dir.split(",") for line in open(Path(d) / "items.jsonl")]
    if a.items:
        keep = set(a.items.split(","))
        rows = [r for r in rows if r["item"] in keep]
    res = aggregate(rows, a.name)
    res["items"] = [r["item"] for r in rows]
    json.dump(res, open(a.out, "w"), indent=1)
    o = res["orders"].get("default")
    if o:
        print(a.name, {k: round(v, 3) for k, v in o["increment_raw"].items()}, "total", round(o["total"], 3),
              res["codec_saving"])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--dataset", required=True)
    r.add_argument("--split", required=True)
    r.add_argument("--params", default=None)
    r.add_argument("--out", required=True)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--orders", default="default,alt,alt_after_R1")
    r.add_argument("--max-seconds", type=float, default=None)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--stop-at", type=float, default=None)
    d = sub.add_parser("devsel")
    d.add_argument("--dataset", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--workers", type=int, default=8)
    d.add_argument("--limit", type=int, default=None)
    d.add_argument("--max-seconds", type=float, default=None)
    f = sub.add_parser("fp4")
    f.add_argument("--plan", required=True)
    f.add_argument("--params", default=None)
    f.add_argument("--samples", default="data/008/fp4/matrices.json")
    f.add_argument("--out", required=True)
    f.add_argument("--workers", type=int, default=8)
    f.add_argument("--orders", default="default,alt,alt_after_R1")
    f.add_argument("--limit", type=int, default=None)
    f.add_argument("--stop-at", type=float, default=None)
    f.add_argument("--shard", default=None, help="i/n: every n-th task starting at i")
    f.add_argument("--first", type=int, default=None, help="only the first N tasks of the plan")
    f.add_argument("--skip", default=None, help="other output dirs whose items count as done")
    f.add_argument("--snapshot-env", default="SNAPSHOT", help="environment variable holding the snapshot path")
    h = sub.add_parser("fp4-devsel")
    h.add_argument("--entries", required=True)
    h.add_argument("--samples", default="data/008/fp4/matrices.json")
    h.add_argument("--out", required=True)
    h.add_argument("--workers", type=int, default=8)
    h.add_argument("--snapshot-env", default="SNAPSHOT", help="environment variable holding the snapshot path")
    g = sub.add_parser("aggregate")
    g.add_argument("--dir", required=True)
    g.add_argument("--name", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--items", default=None)
    a = ap.parse_args()
    {"run": cmd_run, "devsel": cmd_devsel, "fp4": cmd_fp4, "fp4-devsel": cmd_fp4_devsel,
     "aggregate": cmd_aggregate}[a.cmd](a)


if __name__ == "__main__":
    main()
