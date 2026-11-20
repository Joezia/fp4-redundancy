"""TX runner: synthetic checks, runs on the 48-matrix sample, T2 check, summaries.

    PYTHONPATH=. python experiments/010-transforms/tx/tx_run.py synth  <stage>      # V-a / V-b of every method
    PYTHONPATH=. python experiments/010-transforms/tx/tx_run.py run    <stage> [methods...]
    PYTHONPATH=. python experiments/010-transforms/tx/tx_run.py t2     <stage>      # best variant (chosen on train) on T2
    PYTHONPATH=. python experiments/010-transforms/tx/tx_run.py summary <stage>

stage 1 = tx_local. Results: results/tx/<stage>/<method>.jsonl (one line per matrix and variant),
results/tx/<stage>/summary.json, results/tx/synthetic_<stage>.json.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import importlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import txcore as T  # noqa: E402

STAGES = {"1": "tx_local", "2": "tx_global", "3": "tx_nonlinear"}


def vname(v):
    return ",".join(f"{k}={v[k]}" for k in sorted(v))


def mod_of(stage):
    return importlib.import_module(STAGES[stage])


def jobs_for(stage, methods, names):
    M = mod_of(stage)
    V = M.variants()
    out = []
    for meth in methods:
        for v in V[meth]:
            for n in names:
                out.append({"key": f"{meth}|{vname(v)}|{n}", "name": n, "module": STAGES[stage], "method": meth,
                            "variant": v, "vname": vname(v), "view": M.VIEW[meth],
                            "partner": meth in getattr(M, "NEED_PARTNER", ("B2_xor",))})
    return out


def run(stage, methods=None, names=None, workers=64, tag=""):
    M = mod_of(stage)
    methods = methods or list(M.variants())
    names = names or (T.TRAIN + T.TEST)
    for meth in methods:
        T.run_jobs(T.eval_job, jobs_for(stage, [meth], names), T.RES / f"stage{stage}" / f"{meth}{tag}.jsonl",
                   workers=workers)


def summarize_rows(rows, M, meth, raw, ref):
    """per variant and split: bits per weight of C0 / C1 / side, appearance vs the input view and the code view,
    net gain vs the reference (test splits only), coding gain, nonzero fraction, round trips."""
    by = {}
    for r in rows:
        by.setdefault(r["vname"], []).append(r)
    out = {}
    for vn, rs in by.items():
        v = rs[0]["variant"]
        bview = M.base_view(meth, v)
        d = {"variant": v, "base_view": bview}
        for split, names in (("train", T.TRAIN), ("test", T.TEST), ("T2", T2N)):
            sel = [r for r in rs if r["name"] in names]
            if meth == "B2_xor" and v.get("ref") == "partner":
                sel = [r for r in sel if not r.get("not_applicable")]
            if not sel:
                continue
            N = sum(r["n"] for r in sel)
            s = {"matrices": len(sel), "roundtrip_all": all(r["roundtrip"] for r in sel)}
            for c in ("C0", "C1"):
                s[c] = sum(r[c] for r in sel) / N
                s["side_" + c] = sum(r["side_bits_" + c] for r in sel) / N
                s["appear_" + c] = sum(raw[r["name"]][bview][c] - r[c] for r in sel) / N
                s["appear_vs_code_" + c] = sum(raw[r["name"]]["code"][c] - r[c] for r in sel) / N
            if v.get("kind") == "scale_planes":
                for c in ("C0", "C1"):
                    s["appear_" + c] = sum(raw[r["name"]]["scale_bits"] - r[c] for r in sel) / N
                    s.pop("appear_vs_code_" + c)
            cg = [r.get("coding_gain") for r in sel if r.get("coding_gain") is not None]
            s["coding_gain_mean"] = float(np.mean(cg)) if cg else None
            nzf = [r["nonzero_frac"] for r in sel if r.get("nonzero_frac") is not None]
            s["nonzero_frac_mean"] = float(np.mean(nzf)) if nzf else None
            if all(r["name"] in ref for r in sel):
                tot_ref = sum(ref[r["name"]]["total"] for r in sel)
                if v.get("kind") == "scale_planes":     # planes replace the scale stream; codes keep the best raw coder
                    tot = sum(min(r["C0"], r["C1"]) + min(raw[r["name"]]["code"]["C0"], raw[r["name"]]["code"]["C1"])
                              for r in sel)
                else:
                    tot = sum(min(r["C0"], r["C1"]) + raw[r["name"]]["scale_bits"] for r in sel)
                s["total_bpw"] = tot / N
                s["reference_bpw"] = tot_ref / N
                s["net_gain"] = (tot_ref - tot) / N
            d[split] = s
        out[vn] = d
    return out


T2N = []


def best_variant(summ):
    """chosen on the training matrices: smallest min(C0, C1) bits per weight (scale stream is common)."""
    ok = {k: d for k, d in summ.items() if "train" in d and d["train"]["roundtrip_all"]}
    return min(ok, key=lambda k: min(ok[k]["train"]["C0"], ok[k]["train"]["C1"])) if ok else None


def summary(stage):
    global T2N
    T2N = T.t2_names()
    M = mod_of(stage)
    raw = T.raw_table()
    ref = T.reference()
    out = {}
    for meth in M.variants():
        rows = T.load_rows(T.RES / f"stage{stage}" / f"{meth}.jsonl") + \
            T.load_rows(T.RES / f"stage{stage}" / f"{meth}_T2.jsonl")
        if not rows:
            out[meth] = {"status": "not run"}
            continue
        for r in rows:
            r.setdefault("vname", None)
        sm = summarize_rows(rows, M, meth, raw, ref)
        excl = getattr(M, "exclude_from_best", lambda m_, v_: False)
        bv = best_variant({k: d for k, d in sm.items() if not excl(meth, d["variant"])})
        gk = getattr(M, "GROUP_KEYS", {}).get(meth)
        best_by_group = {}
        if gk:
            gs = {}
            for k, d in sm.items():
                gs.setdefault(",".join(f"{a}={d['variant'].get(a)}" for a in gk), {})[k] = d
            for gname, sub in gs.items():
                b = best_variant(sub)
                if b:
                    best_by_group[gname] = b
        out[meth] = {"view": M.VIEW[meth], "best_on_train": bv, "best_by_group_on_train": best_by_group,
                     "variants": sm}
    path = T.RES / f"stage{stage}" / "summary.json"
    json.dump(out, open(path, "w"), indent=1)
    for meth, d in out.items():
        if "variants" not in d:
            print(meth, d)
            continue
        b = d["variants"][d["best_on_train"]]
        te = b.get("test", {})
        print(f"{meth:14s} best {d['best_on_train']:40s} test C0 {te.get('C0', float('nan')):.4f} C1 {te.get('C1', float('nan')):.4f} "
              f"appear C0 {te.get('appear_C0', float('nan')):+.4f} C1 {te.get('appear_C1', float('nan')):+.4f} "
              f"net {te.get('net_gain', float('nan')):+.4f} rt {te.get('roundtrip_all')}")


def t2(stage, workers=64):
    """the train-selected best variant of each method (and of each group) on T2."""
    M = mod_of(stage)
    summ = json.load(open(T.RES / f"stage{stage}" / "summary.json"))
    names = T.t2_names()
    for meth, d in summ.items():
        if "variants" not in d:
            continue
        keep = {d["best_on_train"], *d.get("best_by_group_on_train", {}).values()}
        vs = [d["variants"][k]["variant"] for k in keep if k]
        jobs = []
        for v in vs:
            for n in names:
                jobs.append({"key": f"{meth}|{vname(v)}|{n}", "name": n, "module": STAGES[stage], "method": meth,
                             "variant": v, "vname": vname(v), "view": M.VIEW[meth],
                             "partner": meth in getattr(M, "NEED_PARTNER", ("B2_xor",))})
        T.run_jobs(T.eval_job, jobs, T.RES / f"stage{stage}" / f"{meth}_T2.jsonl", workers=workers)


def synth(stage):
    S = importlib.import_module(f"synth_{STAGES[stage]}")
    res = S.checks()
    res["all_pass"] = all(r["pass"] for k, r in res.items() if isinstance(r, dict) and "pass" in r)
    json.dump(res, open(T.RES / f"synthetic_stage{stage}.json", "w"), indent=1)
    for k, r in res.items():
        if isinstance(r, dict) and "pass" in r:
            print(f"{k:40s} pass {r['pass']}  V-a C0 {r['V-a']['appear_C0']:+.4f} C1 {r['V-a']['appear_C1']:+.4f}  "
                  f"V-b C0 {r['V-b']['appear_C0']:+.4f} C1 {r['V-b']['appear_C1']:+.4f}  theory {r.get('theory_V-b_per_value')}")
    print("all_pass", res["all_pass"])


if __name__ == "__main__":
    cmd, stage = sys.argv[1], sys.argv[2]
    if cmd == "run":
        run(stage, sys.argv[3:] or None)
    elif cmd == "summary":
        summary(stage)
    elif cmd == "t2":
        t2(stage)
    elif cmd == "synth":
        synth(stage)
