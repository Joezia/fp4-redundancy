"""Spec 013: standalone saving of each redundancy pattern (no chained accumulation).

    PYTHONPATH=. python experiments/013-standalone/standalone.py run  {deepseek|kimi} [--workers N]
    PYTHONPATH=. python experiments/013-standalone/standalone.py summarize {deepseek|kimi}
    PYTHONPATH=. python experiments/013-standalone/standalone.py media-run {kodak|sqam|librispeech} [--workers N]
    PYTHONPATH=. python experiments/013-standalone/standalone.py media-summarize {kodak|sqam|librispeech}

Samples and parameters are those of 008 (DeepSeek, plan B, first 172 experts = the 516 matrices of
results/fp4_increments_B.json) and 009 (Kimi-K3, plan_552, 552 matrices); the parameters are the ones each
selected on its dev set (increment_dev/params.json), not re-selected. Only the `redundancy` package is used
(`increment.run_chain`, `increment.fp4_codelen`, the FP4 adapter that reads the snapshot).

Per matrix:
  - the chain [R1c, R1s]  -> increments of R1c and R1s relative to the raw 4.25 bit/weight;
  - for each Rk in R2, R3, R4, R5, R6, R7, R8, R10 the chain [R1c, R1s, Rk] -> the increment of Rk with its
    on/off switch working (run_chain's rule: one flag bit per matrix, on only if it shortens the code),
    the forced-on value (`if_forced`, negative values kept) and whether the switch turned it off.
    R10 is not switchable in run_chain (as in 008/009), so its forced value equals its normal value.
The partner matrix (R7) is defined for w1 only (the w3 matrix of the same expert), as in 008/009.
Increments are in fractions of the raw bits per matrix; the summary pools them over all matrices by raw bits.

Media (008 Part 2, same report splits and the parameters each dataset selected on its dev split): Kodak 13-24
(12 images, 8-bit RGB), SQAM even tracks (35, 16-bit PCM), LibriSpeech test-clean (2,620 files), full length.
The 008 media chain starts with a single R1 step, so the chains are [R1] and [R1, Rk] for Rk in R2..R8, R10;
values are in % of the raw bits and in bits per sample (audio, raw 16) or per channel value (image, raw 8).
"""
import argparse
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402

from redundancy import increment as I  # noqa: E402
from redundancy.adapters import media  # noqa: E402
from redundancy.adapters.fp4 import FP4  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
RAW = 4.25
PATTERNS = ["R2", "R3", "R4", "R5", "R6", "R7", "R8", "R10"]
MODELS = {
    "deepseek": dict(plan="experiments/008-scale-media/results/fp4/B/plan.json", first=172,
                     items="experiments/008-scale-media/results/fp4_increments_B.json",
                     params="experiments/008-scale-media/results/fp4/increment_dev/params.json",
                     samples="data/008/fp4/matrices.json", snapshot_env="SNAPSHOT",
                     default_rows=["experiments/008-scale-media/results/fp4/" + d + "/items.jsonl"
                                   for d in ("increment_B", "increment_B_s1", "increment_B_s2", "increment_B_s3")]),
    "kimi": dict(plan="experiments/009-kimi-k3/results/plan_552.json", first=None, items=None,
                 params="experiments/009-kimi-k3/results/increment_dev/params.json",
                 samples="data/009/kimi/matrices.json", snapshot_env="KIMI_SNAPSHOT",
                 default_rows=["experiments/009-kimi-k3/results/increment_552/items.jsonl",
                               "experiments/009-kimi-k3/results/increment_36/items.jsonl"]),
}


def tasks_of(model):
    m = MODELS[model]
    plan = json.load(open(m["plan"]))
    tasks = plan["tasks"][:m["first"]] if m["first"] else plan["tasks"]
    ents = [e for t in tasks for e in t["entries"]]
    if m["items"]:
        assert sorted(ents) == sorted(json.load(open(m["items"]))["items"]), "sample differs from 008"
    return tasks, plan.get("seed", 0)


def params_of(model):
    P = dict(I.FP4_DEFAULT)
    P.update(json.load(open(MODELS[model]["params"])))
    return P


def chain_record(r, p):
    return {"L": r["L"], "increment": r["increment"][p], "forced": r["if_forced"][p], "used": r["used"][p]}


def expert_task(model, entries, P, seed):
    t0 = time.process_time()
    m = MODELS[model]
    ad = FP4({"samples": m["samples"], "snapshot_env": m["snapshot_env"]})
    codes = {n: ad._read(ad.samples[n]) for n in entries}
    w1 = [e for e in entries if e.endswith(".w1")]
    w3 = [e for e in entries if e.endswith(".w3")]
    partner_of = {w1[0]: w3[0]} if w1 and w3 else {}
    out = []
    for n in entries:
        C, E = codes[n]
        C = C.astype(np.int64)
        pn = partner_of.get(n)
        part = codes[pn][0].astype(np.int64) if pn in codes else None
        raw = C.size * RAW
        cache, memo = {}, {}

        def codelen_for(order):
            def f(flags):
                key = (tuple(q for q in order if q in flags), flags)    # levels depend only on this
                if key not in memo:
                    memo[key] = I.fp4_codelen(C, E, flags, P, order, partner=part, seed=seed, cache=cache)
                return memo[key]
            return f

        rec = {"item": n, "partner": pn if part is not None else None, "raw_bits": raw, "n_weights": int(C.size)}
        o1 = ["R1c", "R1s"]
        r1 = I.run_chain(codelen_for(o1), o1, raw)
        rec["R1"] = {"L": r1["L"], "R1c": r1["increment"]["R1c"], "R1s": r1["increment"]["R1s"]}
        rec["chains"] = {}
        for p in PATTERNS:
            o = ["R1c", "R1s", p]
            r = I.run_chain(codelen_for(o), o, raw)
            assert abs(r["L"][2] - r1["L"][2]) < 1e-6 * raw, "R1 prefix differs between chains"
            rec["chains"][p] = chain_record(r, p)
        out.append(rec)
    dt = time.process_time() - t0
    for r in out:
        r["cpu_seconds"] = dt / len(out)
    return out


def items_path(model):
    return RES / f"standalone_{model}_items.jsonl"


def cmd_run(a):
    RES.mkdir(parents=True, exist_ok=True)
    tasks, seed = tasks_of(a.model)
    P = params_of(a.model)
    path = items_path(a.model)
    done = set()
    if path.exists():
        done = {json.loads(l)["item"] for l in open(path)}
    todo = [t for t in tasks if not all(e in done for e in t["entries"])]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{a.model}: {len(todo)} experts to run of {len(tasks)}", flush=True)
    t0 = time.time()
    with open(path, "a") as fo, ProcessPoolExecutor(a.workers) as ex:
        it = iter(todo)
        pend = set()
        for _ in range(a.workers):
            t = next(it, None)
            if t is not None:
                pend.add(ex.submit(expert_task, a.model, t["entries"], P, seed))
        n_done = 0
        while pend:
            fin, pend = wait(pend, return_when=FIRST_COMPLETED)
            for f in fin:
                for r in f.result():
                    fo.write(json.dumps(r) + "\n")
                fo.flush()
                n_done += 1
                if n_done % 10 == 0:
                    print(f"{n_done}/{len(todo)} experts, {time.time() - t0:.0f} s", flush=True)
                t = next(it, None)
                if t is not None:
                    pend.add(ex.submit(expert_task, a.model, t["entries"], P, seed))
    print(f"RUN_DONE {a.model} wall {time.time() - t0:.0f} s", flush=True)


def default_rows(model):
    rows = {}
    for f in MODELS[model]["default_rows"]:
        if os.path.exists(f):
            for l in open(f):
                try:
                    r = json.loads(l)
                except json.JSONDecodeError:
                    continue
                rows.setdefault(r["item"], r)
    return rows


def stats(values_bpw):
    v = np.asarray(values_bpw)
    return {"median_bpw": float(np.median(v)), "min_bpw": float(v.min()), "max_bpw": float(v.max()),
            "median_pct": float(100 * np.median(v) / RAW), "min_pct": float(100 * v.min() / RAW),
            "max_pct": float(100 * v.max() / RAW)}


def cmd_summarize(a):
    tasks, _ = tasks_of(a.model)
    want = [e for t in tasks for e in t["entries"]]
    rows = {}
    for l in open(items_path(a.model)):
        r = json.loads(l)
        rows[r["item"]] = r
    missing = [e for e in want if e not in rows]
    assert not missing, f"{len(missing)} matrices missing"
    rows = [rows[e] for e in want]
    raw = sum(r["raw_bits"] for r in rows)
    pooled = lambda vals: float(sum(r["raw_bits"] * v for r, v in zip(rows, vals)) / raw)  # noqa: E731
    table = {}
    for p in ("R1c", "R1s"):
        vals = [r["R1"][p] for r in rows]
        f = pooled(vals)
        table[p] = {"chain": ["R1c", "R1s"], "pct": 100 * f, "bpw": RAW * f, **stats([RAW * v for v in vals])}
    r1_total = pooled([r["R1"]["R1c"] + r["R1"]["R1s"] for r in rows])
    for p in PATTERNS:
        inc = [r["chains"][p]["increment"] for r in rows]
        frc = [r["chains"][p]["forced"] for r in rows]
        f, ff = pooled(inc), pooled(frc)
        table[p] = {"chain": ["R1c", "R1s", p], "pct": 100 * f, "bpw": RAW * f,
                    "forced_pct": 100 * ff, "forced_bpw": RAW * ff,
                    "switched_off": int(sum(not r["chains"][p]["used"] for r in rows)),
                    "switchable": p in I.SWITCHABLE,
                    **stats([RAW * v for v in inc]),
                    "forced_stats": stats([RAW * v for v in frc])}
    # check: R2 in [R1c, R1s, R2] against R2 in the 008/009 default order (R2 directly after R1c, R1s)
    dr = default_rows(a.model)
    have = [r for r in rows if r["item"] in dr]
    diffs = [RAW * (r["chains"]["R2"]["increment"] - dr[r["item"]]["chains"]["default"]["increment"]["R2"])
             for r in have]
    raw_h = sum(r["raw_bits"] for r in have)
    p_new = sum(r["raw_bits"] * r["chains"]["R2"]["increment"] for r in have) / raw_h
    p_old = sum(r["raw_bits"] * dr[r["item"]]["chains"]["default"]["increment"]["R2"] for r in have) / raw_h
    check = {"n_matrices_compared": len(have), "source": MODELS[a.model]["default_rows"],
             "R2_standalone_bpw": RAW * p_new, "R2_default_order_bpw": RAW * p_old,
             "pooled_difference_bpw": RAW * (p_new - p_old),
             "max_abs_per_matrix_difference_bpw": float(np.max(np.abs(diffs))) if diffs else None}
    per_matrix = {r["item"]: {"raw_bits": r["raw_bits"], "partner": r["partner"],
                              "R1c_bpw": RAW * r["R1"]["R1c"], "R1s_bpw": RAW * r["R1"]["R1s"],
                              **{f"{p}_bpw": RAW * r["chains"][p]["increment"] for p in PATTERNS},
                              **{f"{p}_forced_bpw": RAW * r["chains"][p]["forced"] for p in PATTERNS},
                              **{f"{p}_on": r["chains"][p]["used"] for p in PATTERNS},
                              "L_R1": r["R1"]["L"], **{f"L_{p}": r["chains"][p]["L"] for p in PATTERNS}}
                  for r in rows}
    out = {"model": a.model, "n_matrices": len(rows), "params": params_of(a.model),
           "plan": MODELS[a.model]["plan"], "unit": "bpw = bit/weight (raw 4.25); pct = % of 4.25 bit; "
           "pooled over all matrices by raw bits; median/min/max over matrices",
           "R1_total_bpw": RAW * r1_total, "table": table, "R2_check": check,
           "cpu_seconds_total": float(sum(r["cpu_seconds"] for r in rows)), "per_matrix": per_matrix}
    json.dump(out, open(RES / f"standalone_{a.model}.json", "w"), indent=1)
    print(f"{a.model}: {len(rows)} matrices; R1 total {RAW * r1_total:.4f} bpw")
    print("pattern | pct | bpw | forced bpw | off | median | min | max")
    for p, t in table.items():
        print(f"{p} | {t['pct']:.3f}% | {t['bpw']:.5f} | {t.get('forced_bpw', float('nan')):.5f} | "
              f"{t.get('switched_off', '-')} | {t['median_bpw']:.5f} | {t['min_bpw']:.5f} | {t['max_bpw']:.5f}")
    print("R2 check:", check)


# ---------------------------------------------------------------- media (008 Part 2)
M8 = "experiments/008-scale-media/results"
MEDIA = {
    "kodak": dict(split="report", params=f"{M8}/kodak/dev/params.json", default_rows=[f"{M8}/kodak/all/items.jsonl"],
                  ref=f"{M8}/kodak_increments.json", unit="bit/channel value", raw_per=8),
    "sqam": dict(split="report", params=f"{M8}/sqam/dev/params.json", default_rows=[f"{M8}/sqam/report/items.jsonl"],
                 ref=f"{M8}/sqam_increments.json", unit="bit/sample", raw_per=16),
    "librispeech": dict(split="report", params=f"{M8}/librispeech/dev/params.json",
                        default_rows=[f"{M8}/librispeech/report/items.jsonl"], ref=f"{M8}/librispeech_increments.json",
                        unit="bit/sample", raw_per=16),
}


def media_params(ds):
    P = dict(I.IMAGE_DEFAULT if media.kind(ds) == "image" else I.AUDIO_DEFAULT)
    P.update(json.load(open(MEDIA[ds]["params"])))
    return P


def media_task(ds, path, P):
    t0 = time.process_time()
    arr, meta = media.load(ds, path, None)
    raw = media.raw_bits(ds, arr)
    fn = I.image_codelen if media.kind(ds) == "image" else I.audio_codelen
    memo = {}

    def codelen(flags):                      # media code lengths do not depend on the order
        if flags not in memo:
            memo[flags] = fn(arr, flags, P)
        return memo[flags]

    r1 = I.run_chain(codelen, ["R1"], raw)
    rec = {"item": os.path.basename(path), "path": path, "meta": meta, "raw_bits": raw,
           "R1": {"L": r1["L"], "R1": r1["increment"]["R1"]}, "chains": {}}
    for p in PATTERNS:
        o = ["R1", p]
        r = I.run_chain(codelen, o, raw)
        assert abs(r["L"][1] - r1["L"][1]) < 1e-6 * raw
        rec["chains"][p] = chain_record(r, p)
    rec["cpu_seconds"] = time.process_time() - t0
    return [rec]


def media_items_path(ds):
    return RES / f"standalone_{ds}_items.jsonl"


def cmd_media_run(a):
    RES.mkdir(parents=True, exist_ok=True)
    P = media_params(a.model)
    fs = media.items(a.model, MEDIA[a.model]["split"])
    path = media_items_path(a.model)
    done = {json.loads(l)["item"] for l in open(path)} if path.exists() else set()
    todo = [f for f in fs if os.path.basename(f) not in done]
    todo.sort(key=lambda f: -os.path.getsize(f))           # largest first
    if a.limit:
        todo = todo[:a.limit]
    print(f"{a.model}: {len(todo)} items to run of {len(fs)}", flush=True)
    t0 = time.time()
    with open(path, "a") as fo, ProcessPoolExecutor(a.workers) as ex:
        it = iter(todo)
        pend = set()
        for _ in range(a.workers):
            f = next(it, None)
            if f is not None:
                pend.add(ex.submit(media_task, a.model, f, P))
        n_done = 0
        while pend:
            fin, pend = wait(pend, return_when=FIRST_COMPLETED)
            for fu in fin:
                for r in fu.result():
                    fo.write(json.dumps(r) + "\n")
                fo.flush()
                n_done += 1
                if n_done % 100 == 0:
                    print(f"{n_done}/{len(todo)} items, {time.time() - t0:.0f} s", flush=True)
                f = next(it, None)
                if f is not None:
                    pend.add(ex.submit(media_task, a.model, f, P))
    print(f"RUN_DONE {a.model} wall {time.time() - t0:.0f} s", flush=True)


def cmd_media_summarize(a):
    ds = a.model
    m = MEDIA[ds]
    per = m["raw_per"]
    want = [os.path.basename(f) for f in media.items(ds, m["split"])]
    rows = {}
    for l in open(media_items_path(ds)):
        r = json.loads(l)
        rows[r["item"]] = r
    missing = [e for e in want if e not in rows]
    assert not missing, f"{len(missing)} items missing"
    rows = [rows[e] for e in want]
    ref_items = json.load(open(m["ref"]))["items"]
    assert sorted(ref_items) == sorted(want), "sample differs from 008"
    raw = sum(r["raw_bits"] for r in rows)
    pooled = lambda vals: float(sum(r["raw_bits"] * v for r, v in zip(rows, vals)) / raw)  # noqa: E731

    def st(vals):
        v = np.asarray(vals)
        return {"median_pct": float(100 * np.median(v)), "min_pct": float(100 * v.min()),
                "max_pct": float(100 * v.max()), "median_per_unit": float(per * np.median(v)),
                "min_per_unit": float(per * v.min()), "max_per_unit": float(per * v.max())}

    vals = [r["R1"]["R1"] for r in rows]
    f = pooled(vals)
    table = {"R1": {"chain": ["R1"], "pct": 100 * f, "per_unit": per * f, **st(vals)}}
    for p in PATTERNS:
        inc = [r["chains"][p]["increment"] for r in rows]
        frc = [r["chains"][p]["forced"] for r in rows]
        f, ff = pooled(inc), pooled(frc)
        table[p] = {"chain": ["R1", p], "pct": 100 * f, "per_unit": per * f, "forced_pct": 100 * ff,
                    "forced_per_unit": per * ff, "switched_off": int(sum(not r["chains"][p]["used"] for r in rows)),
                    "switchable": p in I.SWITCHABLE, **st(inc), "forced_stats": st(frc)}
    dr = default_rows_from(m["default_rows"])
    have = [r for r in rows if r["item"] in dr]
    diffs = [per * (r["chains"]["R2"]["increment"] - dr[r["item"]]["chains"]["default"]["increment"]["R2"]) for r in have]
    raw_h = sum(r["raw_bits"] for r in have)
    p_new = sum(r["raw_bits"] * r["chains"]["R2"]["increment"] for r in have) / raw_h
    p_old = sum(r["raw_bits"] * dr[r["item"]]["chains"]["default"]["increment"]["R2"] for r in have) / raw_h
    check = {"n_items_compared": len(have), "source": m["default_rows"], "R2_standalone_pct": 100 * p_new,
             "R2_default_order_pct": 100 * p_old, "pooled_difference_pct": 100 * (p_new - p_old),
             "max_abs_per_item_difference_per_unit": float(np.max(np.abs(diffs))) if diffs else None}
    per_item = {r["item"]: {"raw_bits": r["raw_bits"], "meta": r["meta"], "R1_pct": 100 * r["R1"]["R1"],
                            **{f"{p}_pct": 100 * r["chains"][p]["increment"] for p in PATTERNS},
                            **{f"{p}_forced_pct": 100 * r["chains"][p]["forced"] for p in PATTERNS},
                            **{f"{p}_on": r["chains"][p]["used"] for p in PATTERNS},
                            "L_R1": r["R1"]["L"], **{f"L_{p}": r["chains"][p]["L"] for p in PATTERNS}}
                for r in rows}
    out = {"dataset": ds, "split": m["split"], "n_items": len(rows), "params": media_params(ds),
           "unit": f"pct = % of raw bits; per_unit = {m['unit']} (raw {per}); pooled over all items by raw bits; "
                   "median/min/max over items", "table": table, "R2_check": check,
           "cpu_seconds_total": float(sum(r["cpu_seconds"] for r in rows)), "per_item": per_item}
    json.dump(out, open(RES / f"standalone_{ds}.json", "w"), indent=1)
    print(f"{ds}: {len(rows)} items")
    print(f"pattern | pct | {m['unit']} | forced pct | off | median pct | min pct | max pct")
    for p, t in table.items():
        print(f"{p} | {t['pct']:.3f}% | {t['per_unit']:.4f} | {t.get('forced_pct', float('nan')):.3f}% | "
              f"{t.get('switched_off', '-')} | {t['median_pct']:.3f} | {t['min_pct']:.3f} | {t['max_pct']:.3f}")
    print("R2 check:", check)


def default_rows_from(files):
    rows = {}
    for f in files:
        if os.path.exists(f):
            for l in open(f):
                try:
                    r = json.loads(l)
                except json.JSONDecodeError:
                    continue
                rows.setdefault(r["item"], r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("model", choices=list(MODELS))
    r.add_argument("--workers", type=int, default=48)
    r.add_argument("--limit", type=int, default=None)
    s = sub.add_parser("summarize")
    s.add_argument("model", choices=list(MODELS))
    mr = sub.add_parser("media-run")
    mr.add_argument("model", choices=list(MEDIA))
    mr.add_argument("--workers", type=int, default=48)
    mr.add_argument("--limit", type=int, default=None)
    ms = sub.add_parser("media-summarize")
    ms.add_argument("model", choices=list(MEDIA))
    a = ap.parse_args()
    {"run": cmd_run, "summarize": cmd_summarize, "media-run": cmd_media_run,
     "media-summarize": cmd_media_summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
