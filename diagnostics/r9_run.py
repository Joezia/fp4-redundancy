"""Spec 005: run the original design (sections 2-4) on the section-1 samples.

Phases (each resumable; one pickle per job in --work, outside the repository):
  main    A1-A3 + B1 + B2 per full-layer / probe matrix, C1 per original matrix, C2 on the probe
          matrices (and 3 S0E), A4 synthetic rules, B3 data preparation.
  b3fit   B3 EM fits: (scope, group, direction, E value, K, initialisation).
  b3eval  B3 evaluation of the shared models on the other expert of every layer.
  cconf   C2 original design on the C confirmation set (run once, after exploration).
Refuses to start unless experiments/005-r9/validation.json reports all_pass.

Usage: python -m diagnostics.r9_run --snapshot $SNAPSHOT --work <dir> --phase main [--jobs 100]
"""
import argparse
import json
import os
import pickle
import time
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

from . import r9  # noqa: E402
from .gaps_run import load, mname  # noqa: E402
from .r58 import shuffle  # noqa: E402
from .st import index_snapshot  # noqa: E402

ROLES = ("w1", "w2", "w3")
B3_K = (1, 4, 16, 32)
B3_GROUPS = ("all", "w1", "w2", "w3")
A4_DISTS = ("normal", "laplace", "t4")
A4_TAUS = (0.0, 0.3)
A4_RULES = ("F", "C", "C+")
SAMPLES = "experiments/004-gaps/samples_004.json"


def seed_of(name):
    return zlib.crc32(name.encode()) & 0x7FFFFFFF


def names(sel):
    """Matrix lists of section 1 (and the section-5 split)."""
    full = [mname(x["prefix"], x["layer"], x["expert"], r) for x in sel["full_layer"] for r in ROLES]
    probe = [mname(x["prefix"], x["layer"], x["expert"], r) for x in sel["probe"] for r in ROLES]
    orig = [mname(x["prefix"], x["layer"], x["expert"], r) for x in sel["original"] for r in ROLES]
    by_layer = {}
    for x in sel["original"]:
        by_layer.setdefault(x["layer"], []).append(x["expert"])
    c_conf = [mname("layers", l, sorted(e)[1], r) for l, e in sorted(by_layer.items()) for r in ROLES]
    slots = {}
    for x in sel["full_layer"]:
        slots.setdefault((x["prefix"], x["layer"]), []).append(x["expert"])
    b3 = {}
    for scope, prefix in (("main", "layers"), ("mtp", "mtp")):
        for slot in (0, 1):
            b3[(scope, slot)] = [mname(prefix, l, sorted(e)[slot], r)
                                 for (p, l), e in sorted(slots.items()) if p == prefix for r in ROLES]
    b_conf = [m for m in full if m not in probe]
    return {"full": full, "probe": probe, "orig": orig, "c_dev": probe, "c_conf": c_conf,
            "b_dev": probe, "b_conf": b_conf, "b3": b3,
            "c2_s0e": [m for m in probe if m.endswith(".w1")][:3]}


# ----------------------------------------------------------------------------- jobs

def job_ab(spec, index, T):
    Cp, E = load(index, T[spec["matrix"]])
    seed = spec["seed"]
    Cs = shuffle(Cp, E, "S0E", r9.rng_for(seed, 1))[0]
    rows, mat = r9.a_matrix(Cp, E, Cs)
    out = {"A_rows": rows, "A": mat, "rules": r9.rule_stats(Cp, E)}
    out["B1"] = r9.b1_matrix(Cp, E, seed)
    out["B1_S0E"] = r9.b1_matrix(Cs, E, seed, Ks=(1, 2, 8, 32))
    out["B2"] = r9.b2_classes(Cp, E, seed, 4) + r9.b2_classes(Cp, E, seed, 8)
    out["shape"] = list(Cp.shape)
    return out


def job_c1(spec, index, T):
    Cp, E = load(index, T[spec["matrix"]])
    seed = spec["seed"]
    Cs = shuffle(Cp, E, "S0E", r9.rng_for(seed, 1))[0]
    return {"C1": r9.c1_matrix(Cp, E, seed, Cs=Cs), "C1b": r9.c1b_matrix(E, seed, Cp.size)}


def job_c2(spec, index, T):
    Cp, E = load(index, T[spec["matrix"]])
    if spec["variant"] == "S0E":
        Cp = shuffle(Cp, E, "S0E", r9.rng_for(spec["seed"], 1))[0]
    return {"C2": r9.inr_fit(Cp, E, spec["seed"])}


def job_a4(spec, index, T):
    out = []
    for i, dist in enumerate(A4_DISTS):
        for j, tau in enumerate(A4_TAUS):
            Wt = r9.synth_weights(r9.rng_for(77, i, j), dist, tau)
            for rule in A4_RULES:
                Cp, E = r9.quant_rule(Wt, rule)
                from .ledger import merge_zero
                Cp = merge_zero(Cp)
                out.append({"dist": dist, "tau": tau, "rule": rule, **r9.rule_stats(Cp, E)})
    return {"A4": out}


def b3_dir(work, scope, slot):
    return Path(work) / f"b3_{scope}_{slot}"


def job_b3prep(spec, index, T):
    d = b3_dir(spec["work"], spec["scope"], spec["slot"])
    d.mkdir(parents=True, exist_ok=True)
    Hs, Es, roles, mats = [], [], [], []
    for i, m in enumerate(spec["matrices"]):
        Cp, E = load(index, T[m])
        Hs.append(r9.block_hist(Cp))
        Es.append(E.ravel())
        roles.append(np.full(E.size, ROLES.index(m.split(".")[-1]), dtype=np.uint8))
        mats.append(np.full(E.size, i, dtype=np.int32))
    np.save(d / "H.npy", np.concatenate(Hs))
    np.save(d / "E.npy", np.concatenate(Es))
    np.save(d / "role.npy", np.concatenate(roles))
    np.save(d / "mat.npy", np.concatenate(mats))
    E = np.concatenate(Es)
    vals, cnt = np.unique(E, return_counts=True)
    return {"matrices": spec["matrices"], "E_counts": {int(v): int(c) for v, c in zip(vals, cnt)}}


def b3_select(d, group, e):
    E = np.load(d / "E.npy", mmap_mode="r")
    sel = np.asarray(E) == e
    if group != "all":
        role = np.load(d / "role.npy", mmap_mode="r")
        sel &= np.asarray(role) == ROLES.index(group)
    idx = np.nonzero(sel)[0]
    H = np.load(d / "H.npy", mmap_mode="r")
    return np.asarray(H[idx])


def unique_rows(H):
    v = np.ascontiguousarray(H).view(np.dtype((np.void, 16))).ravel()
    u, cnt = np.unique(v, return_counts=True)
    return np.frombuffer(u.tobytes(), dtype=np.uint8).reshape(-1, 16), cnt


def job_b3fit(spec, index, T):
    d = b3_dir(spec["work"], spec["scope"], spec["fit_slot"])
    H = b3_select(d, spec["group"], spec["E"])
    n = len(H)
    U, cnt = unique_rows(H)
    del H
    Hs = U[:, r9.CP15]
    key = (U.astype(np.float64) @ r9.ABSV) / r9.B
    Kk = r9.k_eff(n, spec["K"])
    m = r9.em_fit(Hs, Kk, key, r9.rng_for(spec["seed"], spec["init"]), inits=(spec["init"],),
                  wts=cnt)
    return {"model": {"logpi": m["logpi"], "logth": m["logth"]}, "ll": m["ll"], "n": n,
            "n_unique": len(U), "K_E": Kk}


def job_b3eval(spec, index, T):
    work = Path(spec["work"])
    d = b3_dir(work, spec["scope"], 1 - spec["fit_slot"])
    H = np.load(d / "H.npy", mmap_mode="r")
    E = np.load(d / "E.npy")
    role = np.load(d / "role.npy")
    mat = np.load(d / "mat.npy")
    nmat = len(spec["eval_matrices"])
    out = {}
    for group in B3_GROUPS:
        for K in B3_K:
            bits = np.zeros(nmat)
            unseen = np.zeros(nmat, dtype=np.int64)
            kes = {}
            for e in np.unique(E):
                sel = E == e
                if group != "all":
                    sel &= role == ROLES.index(group)
                idx = np.nonzero(sel)[0]
                if not len(idx):
                    continue
                best = None
                for init in ((0,) if K == 1 else (0, 1, 2)):
                    p = work / f"{b3fit_name(spec['scope'], group, spec['fit_slot'], int(e), K, init)}.pkl"
                    if p.exists():
                        with open(p, "rb") as f:
                            r = pickle.load(f)
                        if best is None or r["ll"] > best["ll"]:
                            best = r
                if best is None or best["n"] == 0:
                    b = np.full(len(idx), np.log2(15.0) * r9.B)
                    unseen += np.bincount(mat[idx], minlength=nmat)
                else:
                    kes[int(e)] = best["K_E"]
                    hh = np.asarray(H[idx])[:, r9.CP15]
                    b = r9.mix_bits(hh, best["model"])
                bits += np.bincount(mat[idx], weights=b, minlength=nmat)
            nblk = np.bincount(mat, minlength=nmat) if group == "all" else \
                np.bincount(mat[role == ROLES.index(group)], minlength=nmat)
            out[(group, K)] = {"bits": bits, "n_blocks": nblk, "unseen_blocks": unseen, "K_E": kes}
    return {"eval_matrices": spec["eval_matrices"], "results": out}


def b3fit_name(scope, group, fit_slot, e, K, init):
    return f"b3fit_{scope}_{group}_s{fit_slot}_E{e}_K{K}_i{init}"


JOBS = {"ab": job_ab, "c1": job_c1, "c2": job_c2, "a4": job_a4, "b3prep": job_b3prep,
        "b3fit": job_b3fit, "b3eval": job_b3eval}


def job(spec, snapshot, sel, work):
    t0 = time.time()
    index = index_snapshot(snapshot) if spec["kind"] not in ("a4", "b3fit", "b3eval") else None
    res = {"spec": spec, **JOBS[spec["kind"]](spec, index, sel["tensors"])}
    res["seconds"] = time.time() - t0
    out = Path(work) / f"{spec['name']}.pkl"
    tmp = out.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(res, f)
    tmp.rename(out)
    return spec["name"], res["seconds"]


def make_jobs(sel, phase, work):
    nm = names(sel)
    jobs = []
    if phase == "main":
        jobs.append({"kind": "a4", "name": "a4", "seed": 77})
        for scope in ("main", "mtp"):
            for slot in (0, 1):
                jobs.append({"kind": "b3prep", "name": f"b3prep_{scope}_{slot}", "scope": scope,
                             "slot": slot, "matrices": nm["b3"][(scope, slot)], "work": work})
        for m in nm["probe"]:
            jobs.append({"kind": "c2", "name": f"c2_{m}_real", "matrix": m, "variant": "real",
                         "seed": seed_of(m)})
        for m in nm["c2_s0e"]:
            jobs.append({"kind": "c2", "name": f"c2_{m}_S0E", "matrix": m, "variant": "S0E",
                         "seed": seed_of(m)})
        for m in nm["full"] + [x for x in nm["probe"] if x not in nm["full"]]:
            jobs.append({"kind": "ab", "name": f"ab_{m}", "matrix": m, "seed": seed_of(m)})
        for m in nm["orig"]:
            jobs.append({"kind": "c1", "name": f"c1_{m}", "matrix": m, "seed": seed_of(m)})
    elif phase == "b3fit":
        for scope in ("main", "mtp"):
            for slot in (0, 1):
                with open(Path(work) / f"b3prep_{scope}_{slot}.pkl", "rb") as f:
                    prep = pickle.load(f)
                Ecount = prep["E_counts"]
                for group in B3_GROUPS:
                    for e in sorted(Ecount):
                        for K in B3_K:
                            for init in ((0,) if K == 1 else (0, 1, 2)):
                                n = b3fit_name(scope, group, slot, e, K, init)
                                jobs.append({"kind": "b3fit", "name": n, "scope": scope,
                                             "group": group, "fit_slot": slot, "E": e, "K": K,
                                             "init": init, "seed": seed_of(n), "work": work,
                                             "size": Ecount[e] * K})
        jobs.sort(key=lambda j: -j["size"])
    elif phase == "b3eval":
        for scope in ("main", "mtp"):
            for slot in (0, 1):
                jobs.append({"kind": "b3eval", "name": f"b3eval_{scope}_fit{slot}", "scope": scope,
                             "fit_slot": slot, "eval_matrices": nm["b3"][(scope, 1 - slot)],
                             "work": work})
    elif phase == "cconf":
        for m in nm["c_conf"]:
            jobs.append({"kind": "c2", "name": f"c2conf_{m}_real", "matrix": m, "variant": "real",
                         "seed": seed_of(m)})
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--phase", required=True, choices=("main", "b3fit", "b3eval", "cconf"))
    ap.add_argument("--validation", default="experiments/005-r9/validation.json")
    ap.add_argument("--jobs", type=int, default=100)
    ap.add_argument("--kinds", nargs="*")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    with open(args.validation) as f:
        if not json.load(f)["all_pass"]:
            raise SystemExit("section 7 checks have failures; refusing to read model data")
    with open(SAMPLES) as f:
        sel = json.load(f)
    work = str(Path(args.work).resolve())
    Path(work).mkdir(parents=True, exist_ok=True)
    jobs = make_jobs(sel, args.phase, work)
    with open(Path(work) / f"jobs_{args.phase}.json", "w") as f:
        json.dump(jobs, f, indent=1)
    todo = [j for j in jobs if not (Path(work) / f"{j['name']}.pkl").exists()]
    if args.kinds:
        todo = [j for j in todo if j["kind"] in args.kinds]
    if args.only:
        todo = [j for j in todo if j["name"] in args.only]
    # longest first: B1 jobs and B3 preparation before the short ones
    pri = {"b3prep": 0, "c2": 1, "ab": 2, "c1": 3, "a4": 4, "b3fit": 0, "b3eval": 0}
    todo.sort(key=lambda j: pri[j["kind"]])
    print(f"{len(todo)} jobs to run ({len(jobs)} in phase {args.phase})", flush=True)
    t0 = time.time()
    failures = {}
    with ProcessPoolExecutor(min(args.jobs, max(1, len(todo)))) as ex:
        futs = {ex.submit(job, j, args.snapshot, sel, work): j["name"] for j in todo}
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                n, sec = fut.result()
                print(f"done {n}: {sec:.0f}s (elapsed {time.time() - t0:.0f}s)", flush=True)
            except Exception:
                failures[name] = traceback.format_exc()
                print(f"FAILED {name}\n{failures[name]}", flush=True)
    with open(Path(work) / f"failures_{args.phase}.json", "w") as f:
        json.dump(failures, f, indent=1)
    print(f"{len(todo) - len(failures)} ok, {len(failures)} failed, {time.time() - t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
