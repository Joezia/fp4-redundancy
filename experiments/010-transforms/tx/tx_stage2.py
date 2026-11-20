"""TX stage 2 runner (global transforms).

    PYTHONPATH=. python experiments/010-transforms/tx/tx_stage2.py prep        # raw C0 / C1 of the cross-expert and layer samples
    PYTHONPATH=. python experiments/010-transforms/tx/tx_stage2.py ref         # chain + R9c reference on the reported matrices
    PYTHONPATH=. python experiments/010-transforms/tx/tx_stage2.py single      # per-matrix methods (A7, C3 seriation, C4) on train / test
    PYTHONPATH=. python experiments/010-transforms/tx/tx_stage2.py groups      # A6, A7 shared, A8, C3 expert order at 8 experts
    PYTHONPATH=. python experiments/010-transforms/tx/tx_stage2.py sweep       # A6 / A8 best variants at 2, 4, 16 experts; A8 layer mode
    PYTHONPATH=. python experiments/010-transforms/tx/tx_stage2.py summary
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import txcore as T  # noqa: E402
import tx_global as G  # noqa: E402
import tx_run  # noqa: E402

OUT = T.RES / "stage2"
REF_X = T.RES / "reference_cross.jsonl"


def all_cross_names():
    cs = G.cross_sets()
    names = [n for d in cs.values() for k in ("xsel", "xrep") for n in d[k]]
    names += [n for ns in G.layer_sets().values() for n in ns]
    return sorted(set(names))


def report_names():
    cs = G.cross_sets()
    return sorted({n for d in cs.values() for n in d["xrep"]} | {n for ns in G.layer_sets().values() for n in ns})


def prep():
    import stage0
    T.run_jobs(stage0.raw_job, [{"key": n, "name": n} for n in all_cross_names()], T.RAW_PATH, workers=32)


def ref_job(job):
    """008 chain (w1 with w3 as R7 partner) + R9c (odd + nzc + max, one switch bit), as 010 base / 011 S0-2."""
    sys.path.insert(0, str(T.E11))
    import p1b
    import s0
    import tx
    name = job["name"]
    C, E = tx.read(name)
    pn = name.rsplit(".", 1)[0] + ".w3" if name.endswith("w1") else None
    part = tx.read(pn)[0] if pn else None
    ch = tx.chain(C, E, part)
    if "R6" in p1b.active(ch):            # 011's R9c implementation cannot code with R6: chain only (flagged)
        return {"key": name, "task": name, "chain_total": ch["L"][-1], "r9c_gain_bits": None,
                "total": ch["L"][-1], "n": int(C.size), "r9c_unavailable": "R6 active in the chain"}
    r = s0.r9b_bits(C, E, p1b.active(ch), part, {"R9c odd+nzc+max": s0.COMBOS["R9c odd+nzc+max"]}, "after_R2", seq=False)
    gain = max(r["base_code_bits"] - r["R9c odd+nzc+max"], 0.0) - 1.0
    return {"key": name, "task": name, "chain_total": ch["L"][-1], "r9c_gain_bits": gain,
            "total": ch["L"][-1] - gain, "n": int(C.size)}


def ref():
    T.run_jobs(ref_job, [{"key": n, "name": n} for n in report_names()], REF_X, workers=32)


def reference_all():
    r = dict(T.reference())
    for row in T.load_rows(REF_X):
        r[row["task"]] = row
    return r


def single():
    tx_run.run("2", names=T.TRAIN + T.TEST, workers=48)


def group_jobs(method, variants, nshare, sets=("xsel", "xrep")):
    jobs = []
    for (L, r), d in G.cross_sets().items():
        if nshare == 16:
            names = d["xsel"] + d["xrep"]
            jobs.append({"key": f"{method}|n16|L{L}.{r}", "gkey": f"{method}|n16|L{L}.{r}", "method": method,
                         "variants": variants, "names": names, "split": "xrep", "nshare": 16, "report_only": d["xrep"]})
            continue
        for s in sets:
            for g in range(8 // nshare):
                names = d[s][g * nshare:(g + 1) * nshare]
                gk = f"{method}|n{nshare}|{s}|L{L}.{r}|g{g}"
                jobs.append({"key": gk, "gkey": gk, "method": method, "variants": variants, "names": names, "split": s,
                             "nshare": nshare})
    return jobs


def groups(only=None, kind_only=None, workers=6):
    for meth, vs in G.GROUP_VARIANTS.items():
        if only and meth != only:
            continue
        if meth == "A6_klt":                    # split the two kinds so that the jobs are of similar size
            for kind in ("expert_axis", "subspace"):
                if kind_only and kind != kind_only:
                    continue
                T.run_jobs(G.group_job, group_jobs(meth, [v for v in vs if v["kind"] == kind], 8),
                           OUT / f"{meth}_{kind}.jsonl", workers=workers, done_key="gkey")
        else:
            T.run_jobs(G.group_job, group_jobs(meth, vs, 8), OUT / f"{meth}.jsonl", workers=workers, done_key="gkey")


def group_rows(meth):
    files = sorted(OUT.glob(f"{meth}*.jsonl"))
    return [r for f in files for r in T.load_rows(f)]


def best_group_variant(rows, pred=lambda v: True):
    """chosen on the selection experts (xsel) at 8 experts: smallest min(C0, C1) per weight."""
    agg = {}
    for r in rows:
        if r["split"] != "xsel" or r["nshare"] != 8 or not pred(r["variant"]):
            continue
        a = agg.setdefault(r["vname"], [0.0, 0, r["variant"], True])
        a[0] += min(r["C0"], r["C1"])
        a[1] += r["n"]
        a[3] &= r["roundtrip"]
    ok = {k: a for k, a in agg.items() if a[3]}
    if not ok:
        return None
    k = min(ok, key=lambda k: ok[k][0] / ok[k][1])
    return ok[k][2]


def sweep():
    picks = {}
    a6 = group_rows("A6_klt")
    for kind in ("expert_axis", "subspace"):
        v = best_group_variant(a6, lambda v, kind=kind: v["kind"] == kind)
        if v:
            picks[f"A6_klt_{kind}"] = ("A6_klt", v)
    v = best_group_variant(group_rows("A8_tucker"))
    if v:
        picks["A8_tucker"] = ("A8_tucker", v)
    for tag, (meth, v) in picks.items():
        for ns in (2, 4, 16):
            T.run_jobs(G.group_job, group_jobs(meth, [v], ns), OUT / f"sweep_{tag}_n{ns}.jsonl", workers=24,
                       done_key="gkey")
    jobs = []
    for (L, e, r), names in G.layer_sets().items():
        gk = f"A8_tucker|layers|L{L}.E{e}.{r}"
        jobs.append({"key": gk, "gkey": gk, "method": "A8_tucker", "variants": G.GROUP_VARIANTS["A8_tucker"],
                     "names": names, "split": "layers", "nshare": 4})
    T.run_jobs(G.group_job, jobs, OUT / "layers_A8_tucker.jsonl", workers=10, done_key="gkey")
    json.dump({k: v[1] for k, v in picks.items()}, open(OUT / "sweep_picks.json", "w"), indent=1)


# ------------------------------------------------------------------ summary

def agg(rows, raw, ref, base_of):
    """bits per weight: C0, C1, side, appearance (vs the input view and vs the code view), net gain (when every
    matrix has a reference), round trips."""
    N = sum(r["n"] for r in rows)
    if not N:
        return None
    s = {"matrices": len(rows), "roundtrip_all": all(r["roundtrip"] for r in rows)}
    for c in ("C0", "C1"):
        s[c] = sum(r[c] for r in rows) / N
        s["side_" + c] = sum(r["side_bits_" + c] for r in rows) / N
        s["appear_" + c] = sum(raw[r["name"]][base_of(r)][c] - r[c] for r in rows) / N
        s["appear_vs_code_" + c] = sum(raw[r["name"]]["code"][c] - r[c] for r in rows) / N
    cg = [r.get("coding_gain") for r in rows if r.get("coding_gain") is not None]
    s["coding_gain_mean"] = float(np.mean(cg)) if cg else None
    nz = [r["nonzero_frac"] for r in rows if r.get("nonzero_frac") is not None]
    s["nonzero_frac_mean"] = float(np.mean(nz)) if nz else None
    if all(r["name"] in ref for r in rows):
        tot = sum(min(r["C0"], r["C1"]) + raw[r["name"]]["scale_bits"] for r in rows)
        tref = sum(ref[r["name"]]["total"] for r in rows)
        s["total_bpw"], s["reference_bpw"], s["net_gain"] = tot / N, tref / N, (tref - tot) / N
    for k in ("energy_share_first", "energy_share_k", "relative_error"):
        vals = [r[k] for r in rows if k in r]
        if vals:
            s[k + "_mean"] = float(np.mean(vals))
    return s


def summary():
    raw = T.raw_table()
    ref = reference_all()
    out = {"note": "cross-expert / cross-layer methods are measurements of shared information, not codecs that follow "
                   "the repository rule of coding every target independently"}
    # per-matrix methods via the stage-1 summariser
    tx_run.T2N = T.t2_names()
    import tx_global as M
    for meth in M.variants():
        rows = T.load_rows(OUT / f"{meth}.jsonl") + T.load_rows(OUT / f"{meth}_T2.jsonl")
        if not rows:
            out[meth] = {"status": "not run"}
            continue
        sm = tx_run.summarize_rows(rows, M, meth, raw, ref)
        out[meth] = {"best_on_train": tx_run.best_variant(sm), "variants": sm}
    # group methods
    base = {"A6_klt": lambda r: "aligned", "A8_tucker": lambda r: "aligned", "C3_expert_order": lambda r: "code",
            "A7_ica_shared": lambda r: "level" if r["variant"]["unit"] == "block" else "aligned"}
    for meth in G.GROUP_VARIANTS:
        rows = group_rows(meth)
        if not rows:
            out[meth] = {"status": "not run"}
            continue
        by = {}
        for r in rows:
            by.setdefault((r["vname"], r["nshare"], r["split"]), []).append(r)
        res = {}
        for (vn, ns, sp), rs in by.items():
            res.setdefault(vn, {"variant": rs[0]["variant"]})[f"n{ns}_{sp}"] = agg(rs, raw, ref, base[meth])
        best = best_group_variant(rows)
        out[meth] = {"best_on_xsel": tx_run.vname(best) if best else None, "variants": res}
    sw = {}
    for f in sorted(OUT.glob("sweep_*_n*.jsonl")):
        rows = T.load_rows(f)
        if rows:
            meth = rows[0]["method"]
            sw[f.stem] = {sp: agg([r for r in rows if r["split"] == sp], raw, ref, base[meth]) for sp in ("xsel", "xrep")
                          if any(r["split"] == sp for r in rows)}
    out["sweep"] = sw
    lay = T.load_rows(OUT / "layers_A8_tucker.jsonl")
    if lay:
        by = {}
        for r in lay:
            by.setdefault(r["vname"], []).append(r)
        out["A8_tucker_layer_mode"] = {vn: agg(rs, raw, ref, base["A8_tucker"]) for vn, rs in by.items()}
    json.dump(out, open(OUT / "summary.json", "w"), indent=1)
    print(json.dumps({k: (v.get("best_on_train") or v.get("best_on_xsel")) if isinstance(v, dict) else v
                      for k, v in out.items() if k not in ("sweep", "note")}, indent=1))


def t2():
    """train-selected best variant of the per-matrix methods on T2 (C3: best order per coding)."""
    s = json.load(open(OUT / "summary.json"))
    names = T.t2_names()
    for meth in G.variants():
        d = s.get(meth, {})
        if "variants" not in d:
            continue
        keep = {d["best_on_train"]}
        if meth == "C3_seriation":
            for c in ("code", "b1", "a4"):
                sub = {k: x for k, x in d["variants"].items() if x["variant"]["coding"] == c}
                keep.add(tx_run.best_variant(sub))
        vs = [d["variants"][k]["variant"] for k in keep if k]
        jobs = [{"key": f"{meth}|{tx_run.vname(v)}|{n}", "name": n, "module": "tx_global", "method": meth, "variant": v,
                 "vname": tx_run.vname(v), "view": G.VIEW[meth], "partner": False} for v in vs for n in names]
        T.run_jobs(T.eval_job, jobs, OUT / f"{meth}_T2.jsonl", workers=48)


if __name__ == "__main__":
    if sys.argv[1] == "groups" and len(sys.argv) > 2:
        groups(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None, workers=5)
    else:
        {"prep": prep, "ref": ref, "single": single, "groups": groups, "sweep": sweep, "summary": summary, "t2": t2}[sys.argv[1]]()
