"""Spec 003: run the R5-R8/RX measurements on the 001 sample (36 matrices, 75 pairs).

Writes experiments/003-r5-r8/pairs.json, then one pickle per matrix / pair job into --work (outside
the repository; resumable: finished jobs are skipped). Refuses to start unless the section-7
validation.json reports all_pass. Aggregation: python -m diagnostics.r58_report --work <dir>.

Usage: python -m diagnostics.r58_run --snapshot $SNAPSHOT --work <dir> [--jobs 48]
"""
import argparse
import json
import os
import pickle
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

from . import r58  # noqa: E402
from .measure import unpack_codes  # noqa: E402
from .st import index_snapshot, read_bytes  # noqa: E402

META = ("matrix", "layer", "expert", "role")


def make_pairs(samples):
    by = {(s["layer"], s["expert"], s["role"]): s["matrix"] for s in samples}
    layers = sorted({s["layer"] for s in samples})
    experts = {l: sorted({s["expert"] for s in samples if s["layer"] == l}) for l in layers}
    pairs = []
    for l in layers:
        for role in ("w1", "w2", "w3"):
            ex = experts[l]
            for i in range(len(ex)):
                for j in range(i + 1, len(ex)):
                    pairs.append({"group": "P1", "a": by[(l, ex[i], role)], "b": by[(l, ex[j], role)]})
    for l in layers:
        for e in experts[l]:
            pairs.append({"group": "P2", "a": by[(l, e, "w1")], "b": by[(l, e, "w3")]})
    for role in ("w1", "w2", "w3"):
        for la, lb in zip(layers, layers[1:]):
            for i in range(3):
                pairs.append({"group": "P3", "a": by[(la, experts[la][i], role)],
                              "b": by[(lb, experts[lb][i], role)]})
    return pairs


def load(index, s):
    C = unpack_codes(read_bytes(index[s["weight"]]))
    E = read_bytes(index[s["scale"]])
    assert list(C.shape) == s["codeword_shape"] and E.shape == (C.shape[0], C.shape[1] // r58.B)
    return C, E


def job(kind, idx, snapshot, payload, work):
    t0 = time.time()
    out = Path(work) / f"{kind}{idx:03d}.pkl"
    index = index_snapshot(snapshot)
    if kind == "m":
        s = payload
        C, E = load(index, s)
        res = r58.measure_matrix(C, E, {k: s[k] for k in META}, seed=idx)
        res["meta"] = {k: s[k] for k in META}
    else:
        sa, sb, group = payload
        CA, EA = load(index, sa)
        CB, EB = load(index, sb)
        res = r58.measure_pair(CA, EA, CB, EB, {k: sa[k] for k in META}, {k: sb[k] for k in META},
                               group, seed=1000 + idx)
        res["meta"] = {"a": sa["matrix"], "b": sb["matrix"], "group": group}
    res["seconds"] = time.time() - t0
    tmp = out.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(res, f)
    tmp.rename(out)
    return f"{kind}{idx:03d}", res["seconds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--samples", default="experiments/r1-r4/samples.json")
    ap.add_argument("--validation", default="experiments/003-r5-r8/validation.json")
    ap.add_argument("--out", default="experiments/003-r5-r8")
    ap.add_argument("--jobs", type=int, default=48)
    ap.add_argument("--only", nargs="*", help="job names such as m000 p012")
    args = ap.parse_args()
    with open(args.validation) as f:
        if not json.load(f)["all_pass"]:
            raise SystemExit("section 7 checks have failures; refusing to read model data")
    with open(args.samples) as f:
        samples = json.load(f)["matrices"]
    pairs = make_pairs(samples)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / "pairs.json", "w") as f:
        json.dump({"n": len(pairs), "counts": {g: sum(p["group"] == g for p in pairs)
                                               for g in ("P1", "P2", "P3")},
                   "rule": {"P1": "same layer, same role, different experts",
                            "P2": "same expert, w1 vs w3",
                            "P3": "same role, adjacent sampled layers, same position in the "
                                  "layer's sorted expert list of samples.json"},
                   "pairs": pairs}, f, indent=1)
    by = {s["matrix"]: s for s in samples}
    Path(args.work).mkdir(parents=True, exist_ok=True)
    todo = [("m", i, s) for i, s in enumerate(samples)]
    todo += [("p", i, (by[p["a"]], by[p["b"]], p["group"])) for i, p in enumerate(pairs)]
    todo = [t for t in todo if not (Path(args.work) / f"{t[0]}{t[1]:03d}.pkl").exists()]
    if args.only:
        todo = [t for t in todo if f"{t[0]}{t[1]:03d}" in args.only]
    print(f"{len(todo)} jobs to run ({len(samples)} matrices, {len(pairs)} pairs in total)", flush=True)
    t0 = time.time()
    failures = {}
    with ProcessPoolExecutor(min(args.jobs, max(1, len(todo)))) as ex:
        futs = {ex.submit(job, k, i, args.snapshot, p, args.work): f"{k}{i:03d}" for k, i, p in todo}
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
