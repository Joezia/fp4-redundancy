"""Spec 004: select the section-2 samples and run A/C1 (per expert), B (per layer x family),
C2 (round trip) and D (probe x variant) on the model.

Refuses to start unless experiments/004-gaps/validation.json reports all_pass. Writes
samples_004.json into --out, then one pickle per job into --work (outside the repository;
resumable). Aggregation: python -m diagnostics.gaps_report --work <dir>.

Usage: python -m diagnostics.gaps_run --snapshot $SNAPSHOT --work <dir> [--jobs 20]
"""
import argparse
import json
import os
import pickle
import re
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402

from . import gaps, ledger, r58  # noqa: E402
from .measure import unpack_codes  # noqa: E402
from .st import index_snapshot, read_bytes  # noqa: E402

FP4 = re.compile(r"^(layers|mtp)\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.weight$")
ROLES = ("w1", "w2", "w3")
B_LAYERS = (0, 13, 26, 39)
ROUNDTRIP = ("L0.E164.w1", "L13.E135.w2", "L39.E183.w3")
FAM_ROLE = {"a": "w1", "g": "w3", "o": "w2"}


def mname(prefix, layer, expert, role):
    return f"{'L' if prefix == 'layers' else 'MTP'}{layer}.E{expert}.{role}"


def tensor(index, prefix, layer, expert, role):
    w = f"{prefix}.{layer}.ffn.experts.{expert}.{role}.weight"
    s = w[: -len("weight")] + "scale"
    we, se = index[w], index[s]
    return {"matrix": mname(prefix, layer, expert, role), "prefix": prefix, "layer": layer,
            "expert": expert, "role": role, "weight": w, "scale": s, "weight_file": we["file"],
            "weight_shape": we["shape"], "scale_shape": se["shape"],
            "codeword_shape": [we["shape"][0], 2 * we["shape"][1]]}


def select(index, orig):
    experts = {}
    for k in index:
        m = FP4.match(k)
        if m:
            experts.setdefault((m.group(1), int(m.group(2))), set()).add(int(m.group(3)))
    main = sorted(l for p, l in experts if p == "layers")
    mtp = sorted(l for p, l in experts if p == "mtp")
    assert main == list(range(len(main)))
    full = []
    for prefix, layers, base in (("layers", main, 20261113), ("mtp", mtp, 20261113 + 1000)):
        for layer in layers:
            pool = sorted(experts[(prefix, layer)])
            drawn = [int(x) for x in np.random.default_rng(base + layer).choice(pool, 2, replace=False)]
            for e in sorted(drawn):
                full.append({"prefix": prefix, "layer": layer, "expert": e, "drawn": drawn,
                             "seed": base + layer, "n_pool": len(pool)})
    orig_by_layer = {}
    for s in orig:
        orig_by_layer.setdefault(s["layer"], [])
        if s["expert"] not in orig_by_layer[s["layer"]]:
            orig_by_layer[s["layer"]].append(s["expert"])
    in_layer = {}
    for layer in B_LAYERS:
        o3 = sorted(orig_by_layer[layer])
        rest = sorted(set(experts[("layers", layer)]) - set(o3))
        new = [int(x) for x in np.random.default_rng(20261114 + layer).choice(rest, 5, replace=False)]
        in_layer[layer] = {"original": o3, "new_drawn": new, "order": o3 + sorted(new),
                           "fit": (o3 + sorted(new))[:6], "eval": (o3 + sorted(new))[6:]}
    probe = [{"prefix": "layers", "layer": l, "expert": orig_by_layer[l][0]} for l in B_LAYERS]
    orig_experts = [{"prefix": "layers", "layer": l, "expert": e}
                    for l in sorted(orig_by_layer) for e in sorted(orig_by_layer[l])]
    tensors = {}

    def add(prefix, layer, e):
        for r in ROLES:
            t = tensor(index, prefix, layer, e, r)
            tensors[t["matrix"]] = t

    for x in full + probe + orig_experts:
        add(x["prefix"], x["layer"], x["expert"])
    for layer, v in in_layer.items():
        for e in v["order"]:
            add("layers", layer, e)
    for name in ROUNDTRIP:
        assert name in tensors
    return {"main_layers": len(main), "mtp_layers": mtp, "full_layer": full, "in_layer": in_layer,
            "probe": probe, "original": orig_experts, "roundtrip": list(ROUNDTRIP),
            "n_tensors": len(tensors), "numpy_version": np.__version__,
            "rng": {"full_layer": "default_rng(20261113 + layer) (mtp: 20261113 + 1000 + index)"
                                  ".choice(sorted routed experts, 2, replace=False)",
                    "in_layer": "default_rng(20261114 + layer).choice(sorted experts minus the 3 "
                                "original, 5, replace=False); order = original 3 (sorted) then "
                                "new 5 (sorted); fit = first 6, eval = last 2"},
            "tensors": tensors}


def load(index, t, merged=True):
    C = unpack_codes(read_bytes(index[t["weight"]]))
    E = read_bytes(index[t["scale"]])
    assert list(C.shape) == t["codeword_shape"] and E.shape == (C.shape[0], C.shape[1] // gaps.B)
    return (ledger.merge_zero(C) if merged else C), E


def job(spec, snapshot, sel, work):
    t0 = time.time()
    kind, name = spec["kind"], spec["name"]
    index = index_snapshot(snapshot)
    T = sel["tensors"]
    res = {"spec": spec}
    if kind == "a":
        fams, led = {}, {}
        for fam, role in FAM_ROLE.items():
            t = T[mname(spec["prefix"], spec["layer"], spec["expert"], role)]
            C, E = load(index, t, merged=False)
            led[role] = ledger.ledger_matrix(C, E, gaps.B, gaps.rng_for(spec["seed"], 50, ROLES.index(role)))
            led[role]["matrix"] = t["matrix"]
            fams[fam] = gaps.family(ledger.merge_zero(C), E, role)
            del C
        res["ledger"] = led
        res["A"] = gaps.a_expert(fams)
    elif kind == "b":
        role = FAM_ROLE[spec["family"]]
        order = sel["in_layer"][spec["layer"]]["order"]
        g = gaps.rng_for(spec["seed"], 60)
        real, s4 = [], []
        for e in order:
            Cp, E = load(index, T[mname("layers", spec["layer"], e, role)])
            real.append((Cp, E))
            s4.append(r58.shuffle(Cp, E, "S4", g))
        res["B"] = gaps.b_layer(real, s4, role, spec["seed"])
    elif kind == "c2":
        t = T[spec["matrix"]]
        C, E = load(index, t, merged=False)
        res["C2"] = gaps.c2_roundtrip(C, E)
        r = ledger.ledger_matrix(C, E, gaps.B, gaps.rng_for(spec["seed"], 51))
        res["ledger_row4"] = ledger.ledger_rows(r, None)[4]
    elif kind == "d":
        Cp, E = load(index, T[spec["matrix"]])
        if spec["variant"] != "real":
            Cp, E = r58.shuffle(Cp, E, spec["variant"], gaps.rng_for(spec["seed"], 70))
        res["D"] = gaps.d_probe(Cp, E, spec["seed"])
    res["seconds"] = time.time() - t0
    out = Path(work) / f"{name}.pkl"
    tmp = out.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(res, f)
    tmp.rename(out)
    return name, res["seconds"]


def make_jobs(sel):
    jobs = []
    for f in ("a", "g", "o"):
        for layer in B_LAYERS:
            jobs.append({"kind": "b", "name": f"b_L{layer}_{f}", "layer": layer, "family": f,
                         "seed": 400 + 10 * layer + "ago".index(f)})
    for i, m in enumerate(sel["roundtrip"]):
        jobs.append({"kind": "c2", "name": f"c2_{m}", "matrix": m, "seed": 300 + i})
    i = 0
    for p in sel["probe"]:
        for role in ROLES:
            m = mname(p["prefix"], p["layer"], p["expert"], role)
            for v in ("real", "S2", "S0E"):
                jobs.append({"kind": "d", "name": f"d_{m}_{v}", "matrix": m, "variant": v,
                             "seed": 500 + i})
                i += 1
    seen = set()
    for setname, lst in (("full", sel["full_layer"]), ("orig", sel["original"])):
        for x in lst:
            key = (x["prefix"], x["layer"], x["expert"])
            if key in seen:
                continue
            seen.add(key)
            n = len(seen)
            jobs.append({"kind": "a", "name": f"a_{mname(*key, 'x')[:-2]}", "prefix": x["prefix"],
                         "layer": x["layer"], "expert": x["expert"], "seed": 1000 + n,
                         "sets": [s for s, l in (("full", sel["full_layer"]), ("orig", sel["original"]))
                                  if any((y["prefix"], y["layer"], y["expert"]) == key for y in l)]})
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--orig", default="experiments/r1-r4/samples.json")
    ap.add_argument("--validation", default="experiments/004-gaps/validation.json")
    ap.add_argument("--out", default="experiments/004-gaps")
    ap.add_argument("--jobs", type=int, default=20)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--kinds", nargs="*")
    args = ap.parse_args()
    with open(args.validation) as f:
        if not json.load(f)["all_pass"]:
            raise SystemExit("section 7 checks have failures; refusing to read model data")
    with open(args.orig) as f:
        orig = json.load(f)["matrices"]
    index = index_snapshot(args.snapshot)
    sel = select(index, orig)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / "samples_004.json", "w") as f:
        json.dump(sel, f, indent=1)
    jobs = make_jobs(sel)
    Path(args.work).mkdir(parents=True, exist_ok=True)
    with open(Path(args.work) / "jobs.json", "w") as f:
        json.dump(jobs, f, indent=1)
    todo = [j for j in jobs if not (Path(args.work) / f"{j['name']}.pkl").exists()]
    if args.only:
        todo = [j for j in todo if j["name"] in args.only]
    if args.kinds:
        todo = [j for j in todo if j["kind"] in args.kinds]
    print(f"{len(todo)} jobs to run ({len(jobs)} in total; {sel['n_tensors']} tensors)", flush=True)
    t0 = time.time()
    failures = {}
    with ProcessPoolExecutor(min(args.jobs, max(1, len(todo)))) as ex:
        futs = {ex.submit(job, j, args.snapshot, sel, args.work): j["name"] for j in todo}
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                n, sec = fut.result()
                print(f"done {n}: {sec:.0f}s (elapsed {time.time() - t0:.0f}s)", flush=True)
            except Exception:
                failures[name] = traceback.format_exc()
                print(f"FAILED {name}\n{failures[name]}", flush=True)
    with open(Path(args.work) / "failures.json", "w") as f:
        json.dump(failures, f, indent=1)
    print(f"{len(todo) - len(failures)} ok, {len(failures)} failed, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
