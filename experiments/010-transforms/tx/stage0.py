"""TX stage 0: views, raw C0 / C1 code lengths, reference, coder checks.

    PYTHONPATH=. python experiments/010-transforms/tx/stage0.py synth   # coder and view checks (synthetic)
    PYTHONPATH=. python experiments/010-transforms/tx/stage0.py raw     # raw C0 / C1 on train + test (+ T2)
    PYTHONPATH=. python experiments/010-transforms/tx/stage0.py ref     # reference (008 chain + R9c), measured
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import txcore as T  # noqa: E402


def raw_job(job):
    m = T.Mat.load(job["name"], with_partner=False)
    rb = T.raw_bits(m)
    ok_code = m.check_codes(sym=T.merge(T.codes_from_levels(m.L)))
    ok_val = m.check_codes(X=m.X)
    return {"key": job["key"], "name": job["name"], "n": m.n, "scale_bits": T.scale_bits(m.E), **rb,
            "views_roundtrip": bool(ok_code and ok_val), "emin": int(m.E.min()), "emax": int(m.E.max()),
            "aligned_max_abs": int(np.abs(m.X).max())}


def stage_raw(extra_t2=True):
    names = T.TRAIN + T.TEST + (T.t2_names() if extra_t2 else [])
    T.run_jobs(raw_job, [{"key": n, "name": n} for n in names], T.RAW_PATH, workers=48)


def stage_ref():
    ref = T.reference()
    raw = T.raw_table()
    out = {}
    for split, names in (("T1", T.TEST), ("T2", T.t2_names())):
        rows = []
        for n in names:
            r = dict(ref[n])
            if n in raw:
                r["scale_bits"] = raw[n]["scale_bits"]
                r["code_part"] = r["total"] - r["scale_bits"]
            rows.append((n, r))
        N = sum(r["n"] for _, r in rows)
        s = {"matrices": len(rows), "weights": N,
             "chain_total_bpw": sum(r["chain_total"] for _, r in rows) / N,
             "r9c_gain_bpw": sum(r["r9c_gain_bits"] for _, r in rows) / N,
             "reference_total_bpw": sum(r["total"] for _, r in rows) / N}
        if all("scale_bits" in r for _, r in rows):
            s["scale_bpw"] = sum(r["scale_bits"] for _, r in rows) / N
            s["reference_code_part_bpw"] = sum(r["code_part"] for _, r in rows) / N
            s["raw"] = {v: {c: sum(raw[n][v][c] for n, _ in rows) / N for c in ("C0", "C1")}
                        for v in ("code", "level", "aligned")}
        out[split] = {"summary": s, "per_matrix": dict(rows)}
    json.dump(out, open(T.RES / "reference.json", "w"), indent=1)
    for k, v in out.items():
        print(k, json.dumps(v["summary"], indent=1))


# ------------------------------------------------------------------ synthetic checks of the coders and views

def H(p):
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def stage_synth():
    rng = np.random.default_rng(0)
    out = {}
    n = 400 * 1000
    # 1. iid symbols: C0 and C1 ~ n H(p)
    p = rng.dirichlet(np.ones(15) * 0.7)
    s = rng.choice(15, size=(400, 1000), p=p)
    out["sym_iid"] = {"H": H(p), "C0": T.sym_bits(s, 15, "C0") / n, "C1": T.sym_bits(s, 15, "C1") / n}
    # 2. symbols copying the left neighbour with prob 0.8: C1 ~ H(X | left) < C0 ~ H(p)
    s2 = s.copy()
    cp = rng.random(s.shape) < 0.8
    for c in range(1, 1000):
        s2[:, c] = np.where(cp[:, c], s2[:, c - 1], s2[:, c])
    pm = np.bincount(s2.ravel(), minlength=15) / n
    hc = 0.0          # H(X | left) = sum_l p(l) H(0.8 delta_l + 0.2 p)
    for l in range(15):
        q = 0.2 * p.copy()
        q[l] += 0.8
        hc += pm[l] * H(q)
    out["sym_copy_left"] = {"H_marginal": H(pm), "H_cond_left": hc, "C0": T.sym_bits(s2, 15, "C0") / n,
                            "C1": T.sym_bits(s2, 15, "C1") / n}
    # 3. iid integers (two-sided geometric): C0 ~ C1 ~ entropy
    g = rng.geometric(0.15, size=(400, 1000)) - 1
    v = np.where(rng.random(g.shape) < 0.5, -g, g)
    vals, cnt = np.unique(v, return_counts=True)
    out["int_iid"] = {"H": H(cnt / n), "C0": T.int_bits(v, "C0") / n, "C1": T.int_bits(v, "C1") / n}
    # 4. integers with 2-D smooth magnitude (scale field): C1 < C0
    sc = np.exp(np.cumsum(rng.normal(0, 0.05, (400, 1000)), axis=1) + np.cumsum(rng.normal(0, 0.05, (400, 1)), axis=0))
    w = np.round(rng.normal(0, 1, (400, 1000)) * 8 * sc / sc.mean()).astype(np.int64)
    out["int_smooth_scale"] = {"C0": T.int_bits(w, "C0") / n, "C1": T.int_bits(w, "C1") / n}
    # 5. views: random codes -> levels -> codes, aligned -> unaligned
    C = rng.integers(0, 16, size=(64, 256))
    E = rng.integers(119, 123, size=(64, 8)).astype(np.uint8)
    m = T.Mat("synthetic", C, E)
    out["views"] = {"codes_from_levels": m.check_codes(sym=T.merge(T.codes_from_levels(m.L))),
                    "aligned_roundtrip": m.check_codes(X=m.X), "minus_zero_merged": bool((m.sym[C == 8] == 0).all())}
    chk = {
        "sym_iid": abs(out["sym_iid"]["C0"] - out["sym_iid"]["H"]) < 0.01 and abs(out["sym_iid"]["C1"] - out["sym_iid"]["H"]) < 0.08,
        "sym_copy_left": abs(out["sym_copy_left"]["C1"] - hc) < 0.05 and out["sym_copy_left"]["C0"] - out["sym_copy_left"]["C1"] > 1.0,
        "int_iid": abs(out["int_iid"]["C0"] - out["int_iid"]["H"]) < 0.02 and abs(out["int_iid"]["C1"] - out["int_iid"]["H"]) < 0.08,
        "int_smooth_scale": out["int_smooth_scale"]["C0"] - out["int_smooth_scale"]["C1"] > 0.1,
        "views": all(out["views"].values()),
    }
    out["checks"] = chk
    out["all_pass"] = all(chk.values())
    T.RES.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(T.RES / "synthetic_stage0.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"synth": stage_synth, "raw": stage_raw, "ref": stage_ref}[sys.argv[1]]()
