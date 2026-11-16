"""Acceptance checks of spec 007 section 7; writes validation.json.

    python -m redundancy.tests.acceptance --results <FP4 results> --rerun <second FP4 results> \
        --work <scratch dir> --out experiments/007-figures/validation.json

1 synthetic verdicts (from the synthetic part of --results)
2 consistency with the result files of specs 001-006 (FP4 only; this file is the one place where
  the FP4 meaning of the old result files is spelled out)
3 portability: two temporary .npy arrays through the array_file adapter, run + plot
4 layer isolation (same test as test_isolation.py)
5 reproducibility: --results and --rerun must be byte-identical
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "redundancy"
PY = sys.executable
FORBIDDEN = ("scale", "e2m1", "expert", "w1", "w2", "w3", "mtp")
GENERIC_FILES = ["plot.py", "baselines.py", "breakeven.py"]


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- 1 synthetic

MUST_PASS = ("R4", "R5", "R6", "R7a_unit", "R7a_row", "R7a_partner", "R7b", "R8")


def check_synthetic(results):
    man = load(Path(results) / "manifest.json")
    rows = []
    for key, v in sorted(man["synthetic"].items()):
        none, has = v["none"]["verdict"], v["has"]["verdict"]
        want = "present_breakeven" if key in MUST_PASS else ("present", "present_breakeven")
        ok_has = has == want if isinstance(want, str) else has in want
        rows.append({"key": key, "none": none, "has": has, "pass": none == "none" and ok_has})
    return {"rows": rows, "pass": all(r["pass"] for r in rows) and len(rows) == 13}


# ----------------------------------------------------------------------------- 2 consistency

def _csv(p):
    with open(ROOT / p, newline="") as f:
        return list(csv.DictReader(f))


def references():
    """{quantity: {matrix: value}} from the result files of specs 001-006."""
    ref = {k: {} for k in ("H0_001", "H_002", "I_scale_002", "mi_same_001", "mi_same_S2_001", "acf_abs_001",
                           "top_share_003", "top_share_S4_003", "Kstar_003", "nn_median_003", "ratio64_003",
                           "r2_64_2_003", "r2_64_2_006", "frac_no_odd_005")}
    for r in _csv("experiments/r1-r4/r1_histograms.csv"):
        if r["item"] == "A1" and r["group_or_context"] == "H0":
            ref["H0_001"][r["matrix"]] = float(r["value"])
    ce = {}
    for r in _csv("experiments/002-ledger/cond_entropy.csv"):
        if r["quantity"] in ("H(C)", "H(C|E)") and not r["E"]:
            ce.setdefault(r["matrix"], {})[r["quantity"]] = float(r["value"])
    for m, d in ce.items():
        ref["H_002"][m] = d["H(C)"]
        ref["I_scale_002"][m] = d["H(C)"] - d["H(C|E)"]
    for r in _csv("experiments/r1-r4/r3_mi_lag.csv"):
        if (r["item"], r["variable"], r["direction"], r["lag"], r["subset"], r["baseline"]) == \
                ("C1", "C", "row", "1", "same", "S2"):
            ref["mi_same_001"][r["matrix"]] = float(r["value"])
            ref["mi_same_S2_001"][r["matrix"]] = float(r["baseline_mean"])
    for r in _csv("experiments/r1-r4/r4_linear.csv"):
        if (r["item"], r["variable"], r["direction"], r["lag"], r["subset"]) == ("D1", "|v|", "row", "1", "all"):
            ref["acf_abs_001"][r["matrix"]] = float(r["value"])
    s4 = {}
    for r in _csv("experiments/003-r5-r8/r5_spectrum.csv"):
        if r["variable"] == "energy_share_topk" and r["domain"] == "v" and r["k"] == "1":
            if r["group_or_context"] == "real":
                ref["top_share_003"][r["matrix"]] = float(r["value"])
            elif r["group_or_context"] == "S4":
                s4.setdefault(r["matrix"], []).append(float(r["value"]))
    ref["top_share_S4_003"] = {m: float(np.mean(v)) for m, v in s4.items()}
    for r in _csv("experiments/003-r5-r8/r6_repeats.csv"):
        if r["item"] == "R6.2" and r["variable"] == "K_star" and r["group_or_context"] == "":
            ref["Kstar_003"][r["matrix"]] = None if r["value"] == "" else int(float(r["value"]))
    for r in _csv("experiments/003-r5-r8/r7_blocks.csv"):
        if r["item"] == "R7.1" and r["variable"] == "best_median" and r["group_or_context"] == "real":
            ref["nn_median_003"][r["matrix"]] = float(r["value"])
        if r["item"] == "R7.2" and r["variable"] == "ratio" and r["domain"] == "v" and r["k"] == "64":
            ref["ratio64_003"][r["matrix"]] = float(r["value"])
    for r in _csv("experiments/003-r5-r8/r8_sparse.csv"):
        if r["item"] == "R8.2" and r["variable"] == "R2_dict" and r["k"] == "2":
            ref["r2_64_2_003"][r["matrix"]] = float(r["value"])
    for r in _csv("experiments/006-dict-sweep/sweep.csv"):
        if r["K"] == "64" and r["s"] == "2":
            ref["r2_64_2_006"][r["matrix"]] = float(r["R2_real"])
    agg = {}
    for r in _csv("experiments/005-r9/a_grid.csv"):
        if r["E"] != "all":
            n = int(r["n_blocks"])
            a = agg.setdefault(r["matrix"], [0.0, 0])
            a[0] += float(r["frac0_real"]) * n
            a[1] += n
    ref["frac_no_odd_005"] = {m: a[0] / a[1] for m, a in agg.items()}
    return ref


def check_fp4(results):
    res = Path(results) / "data"
    ent = {f: load(res / f"{f}.json")["entries"] for f in
           ("R1", "R2", "R3", "R4", "R5", "R6", "R7a_unit", "R7b", "R8", "R9_odd_count")}
    ref = references()
    new = {
        "H0": {m: r["H"] for m, r in ent["R1"].items()},
        "I_scale": {m: r["groups"]["scale"]["mi"] for m, r in ent["R2"].items()},
        "mi_same": {m: r["mi_same_unit"][0] for m, r in ent["R3"].items()},
        "mi_same_S2": {m: r["base_same_unit_mean"][0] for m, r in ent["R3"].items()},
        "acf_abs": {m: r["abs"][0] for m, r in ent["R4"].items()},
        "top_share": {m: r["share"][0] for m, r in ent["R5"].items()},
        "top_share_S4": {m: r["base_mean"][0] for m, r in ent["R5"].items()},
        "Kstar": {m: r["Kstar"] for m, r in ent["R6"].items()},
        "nn_median": {m: r["real"]["median"] for m, r in ent["R7a_unit"].items()},
        "ratio64": {m: r["ratio"][r["ks"].index(64)] for m, r in ent["R7b"].items()},
        "r2_64_2": {m: r["r2"][r["keys"].index("64,2")] for m, r in ent["R8"].items()
                    if r.get("applicable", True)},
        "frac_no_odd": {m: r["observed"][0] for m, r in ent["R9_odd_count"].items()},
    }
    spec = [  # (quantity, new key, reference key, tolerance or "equal", source)
        ("zero-order entropy", "H0", "H0_001", 5e-6, "001 r1_histograms (stored to 6 significant digits)"),
        ("zero-order entropy", "H0", "H_002", 1e-6, "002 cond_entropy H(C) (full precision, same quantity)"),
        ("MI with the scale group", "I_scale", "I_scale_002", 1e-4, "002 H(C) - H(C|E)"),
        ("MI of adjacent symbols, same unit", "mi_same", "mi_same_001", 1e-4, "001 C1 row d=1 same"),
        ("same, after shuffling inside units", "mi_same_S2", "mi_same_S2_001", 1e-4, "001 C1 S2 baseline mean"),
        ("|v| lag-1 autocorrelation", "acf_abs", "acf_abs_001", 1e-3, "001 D1 |v| row d=1"),
        ("strongest layer energy share", "top_share", "top_share_003", 1e-4, "003 R5.1 top-1 share"),
        ("its baseline (sign shuffle)", "top_share_S4", "top_share_S4_003", 1e-4, "003 R5.1 S4 mean"),
        ("longest repeat length (on the k grid)", "Kstar", "Kstar_003", "equal", "003 R6.2 K*"),
        ("median nearest-neighbour equal positions", "nn_median", "nn_median_003", "equal", "003 R7.1"),
        ("k-means distortion ratio, K = 64", "ratio64", "ratio64_003", 0.005, "003 R7.2 v"),
        ("dictionary R2, K = 64, s = 2", "r2_64_2", "r2_64_2_003", 0.005, "003 R8.2"),
        ("dictionary R2, K = 64, s = 2", "r2_64_2", "r2_64_2_006", 0.005, "006 sweep"),
        ("share of units without odd levels", "frac_no_odd", "frac_no_odd_005", 1e-4, "005 a_grid"),
    ]
    out = []
    for q, nk, rk, tol, src in spec:
        common = sorted(set(new[nk]) & set(ref[rk]))
        diffs = {}
        for m in common:
            a, b = new[nk][m], ref[rk][m]
            diffs[m] = (0.0 if a == b else float("inf")) if tol == "equal" and (a is None or b is None) \
                else (abs(a - b) if a is not None and b is not None else float("inf"))
        mx = max(diffs.values()) if diffs else None
        bad = [m for m, d in diffs.items() if (d > 0 if tol == "equal" else d > tol)]
        out.append({"quantity": q, "source": src, "tolerance": tol, "n_matrices": len(common),
                    "max_abs_diff": mx, "n_outside": len(bad), "outside": {m: [new[nk][m], ref[rk][m]] for m in bad},
                    "pass": bool(common) and not bad})
    return {"rows": out, "pass": all(r["pass"] for r in out)}


# ----------------------------------------------------------------------------- 3 portability

PORTABLE_INDICATORS = load(PKG / "configs" / "synthetic.json")["indicators"]


def make_arrays(work, seed=20261118):
    work.mkdir(parents=True, exist_ok=True)
    g = np.random.default_rng(seed)
    uni = g.integers(0, 256, size=(256, 512), dtype=np.uint8)
    walk = np.cumsum(g.standard_normal((256, 512)), axis=1)
    k = 8
    smooth = np.stack([np.convolve(r, np.ones(k) / k, mode="same") for r in walk])
    np.save(work / "uniform_uint8.npy", uni)
    np.save(work / "smooth.npy", smooth)
    return work / "uniform_uint8.npy", work / "smooth.npy"


def portable_config(work, files):
    cfg = {"name": "array_test", "label": "array_file test", "seed": 7, "workers": 2,
           "adapter": {"type": "array_file", "files": {Path(f).stem: str(f) for f in files}, "bins": 64},
           "indicators": PORTABLE_INDICATORS}
    work.mkdir(parents=True, exist_ok=True)
    p = work / "array_test.json"
    with open(p, "w") as f:
        json.dump(cfg, f, indent=1)
    return p


def run_cmd(args):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    r = subprocess.run([PY, "-m", *args], cwd=ROOT, env=env, capture_output=True, text=True)
    return r.returncode, r.stdout[-2000:] + r.stderr[-4000:]


def check_portable(work, synthetic_results=None):
    work = Path(work)
    uni, smooth = make_arrays(work)
    out = {"pass": True, "datasets": {}}
    for name, f in (("uniform", uni), ("smooth", smooth)):
        cfgp = portable_config(work / name, [f])
        rdir, fdir = work / name / "results", work / name / "figures"
        code, log = run_cmd(["redundancy.run", "--config", str(cfgp), "--out", str(rdir)])
        pargs = ["redundancy.plot", "--results", str(rdir), "--out", str(fdir)]
        if synthetic_results:
            pargs += ["--synthetic", str(synthetic_results)]
        code2, log2 = (1, "") if code else run_cmd(pargs)
        d = {"run_ok": code == 0, "plot_ok": code2 == 0}
        if code == 0:
            man = load(rdir / "manifest.json")
            d["verdicts"] = man["verdicts"]
            pngs = sorted(p.stem for p in fdir.glob("*.png")) if code2 == 0 else []
            want = [f for f in man["figures"] if f != "R10"]
            from redundancy.style import file_name
            d["figures"] = pngs
            d["all_figures"] = all(file_name(f) in pngs for f in want) and "overview" in pngs
            na = [f for f, v in man["verdicts"].items() if v == "not_applicable"]
            d["not_applicable"] = na
            if name == "uniform":
                d["pass"] = all(v in ("none", "not_applicable") for v in man["verdicts"].values())
            else:
                d["pass"] = man["verdicts"].get("R4") == "present_breakeven"
            d["pass"] = d["pass"] and d["all_figures"] and "R7a_partner" in na and code2 == 0
        else:
            d["pass"] = False
            d["log"] = log
        if code2:
            d["plot_log"] = log2
        out["datasets"][name] = d
        out["pass"] &= d["pass"]
    return out


# ----------------------------------------------------------------------------- 4 isolation

def check_isolation():
    import ast
    import re
    files = sorted((PKG / "indicators").glob("*.py")) + [PKG / f for f in GENERIC_FILES]
    pat = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(FORBIDDEN) + r")(?![A-Za-z0-9])", re.IGNORECASE)
    hits = []
    for f in files:
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            for m in pat.finditer(line):
                hits.append(f"{f.relative_to(ROOT)}:{i}: {m.group(0)}")
    tree = ast.parse((PKG / "plot.py").read_text(encoding="utf-8"))
    bad_imports = []
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or "")] + [a.name for a in node.names]
        for n in names:
            if any(k in n for k in ("adapters", "indicators", "fp4", "synthetic", "array_file", "diagnostics")):
                bad_imports.append(n)
    return {"files": [str(f.relative_to(ROOT)) for f in files], "forbidden_words": list(FORBIDDEN),
            "hits": hits, "plot_bad_imports": bad_imports, "pass": not hits and not bad_imports}


# ----------------------------------------------------------------------------- 5 reproducibility

def compare_dirs(a, b):
    a, b = Path(a), Path(b)
    fa = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file())
    fb = sorted(p.relative_to(b) for p in b.rglob("*") if p.is_file())
    diff = [str(p) for p in fa if p in fb and (a / p).read_bytes() != (b / p).read_bytes()]
    only = sorted(set(map(str, fa)) ^ set(map(str, fb)))
    return {"n_files": len(fa), "different": diff, "only_in_one": only, "pass": bool(fa) and not diff and not only}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--rerun", default=None)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    v = {"1_synthetic": check_synthetic(args.results), "2_consistency": check_fp4(args.results),
         "3_portable": check_portable(args.work, args.results), "4_isolation": check_isolation()}
    if args.rerun:
        v["5_reproducible"] = compare_dirs(args.results, args.rerun)
    v["all_pass"] = all(x["pass"] for x in v.values())
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(v, f, indent=1, ensure_ascii=False, default=lambda o: None)
    for k, x in v.items():
        if k != "all_pass":
            print(k, "PASS" if x["pass"] else "FAIL")
    print("all_pass", v["all_pass"])


if __name__ == "__main__":
    main()
