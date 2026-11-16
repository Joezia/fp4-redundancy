"""Spec 006: run the dictionary sweep on the 004 probe sample (12 matrices x real / S0E / S2).

Refuses to start unless experiments/006-dict-sweep/validation.json reports all_pass. One small JSON
per (matrix, variant) job in --work (outside the repository; resumable). The 003 matrix index is the
seed, so the block sample and the shuffles are the ones 003 used.
Usage: python -m diagnostics.dsweep_run --snapshot $SNAPSHOT --work <dir> [--jobs 36]
"""
import argparse
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

from . import dsweep  # noqa: E402
from .gaps_run import mname  # noqa: E402
from .ledger import merge_zero  # noqa: E402
from .r58_run import load  # noqa: E402
from .st import index_snapshot  # noqa: E402

SAMPLES_004 = "experiments/004-gaps/samples_004.json"
SAMPLES_001 = "experiments/r1-r4/samples.json"


def probe_matrices():
    """[(003 index, sample dict)] of the 004 probe sample (12 matrices)."""
    with open(SAMPLES_004) as f:
        probe = json.load(f)["probe"]
    with open(SAMPLES_001) as f:
        orig = json.load(f)["matrices"]
    by = {s["matrix"]: (i, s) for i, s in enumerate(orig)}
    return [by[mname(p["prefix"], p["layer"], p["expert"], r)] for p in probe for r in ("w1", "w2", "w3")]


def job(idx, s, variant, snapshot, work):
    t0 = time.time()
    index = index_snapshot(snapshot)
    C, E = load(index, s)
    res = dsweep.sweep_variant(merge_zero(C), E, idx, variant)
    out = {"matrix": s["matrix"], "seed": idx, "variant": variant, "main": res["main"],
           "n_fit": res["n_fit"], "n_eval": res["n_eval"],
           "dict": [[K, sv, v] for (K, sv), v in res["dict"].items()],
           "pca": [[sv, v] for sv, v in res["pca"].items()], "seconds": time.time() - t0}
    p = Path(work) / f"{s['matrix']}_{variant}.json"
    with open(p.with_suffix(".tmp"), "w") as f:
        json.dump(out, f)
    p.with_suffix(".tmp").rename(p)
    return out["seconds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--jobs", type=int, default=36)
    args = ap.parse_args()
    with open("experiments/006-dict-sweep/validation.json") as f:
        if not json.load(f)["all_pass"]:
            raise SystemExit("section 4 checks have failures; refusing to read model data")
    Path(args.work).mkdir(parents=True, exist_ok=True)
    todo = [(i, s, v) for i, s in probe_matrices() for v in dsweep.VARIANTS
            if not (Path(args.work) / f"{s['matrix']}_{v}.json").exists()]
    print(f"{len(todo)} jobs", flush=True)
    t0, nfail = time.time(), 0
    with ProcessPoolExecutor(min(args.jobs, max(1, len(todo)))) as ex:
        futs = {ex.submit(job, i, s, v, args.snapshot, args.work): f"{s['matrix']}_{v}" for i, s, v in todo}
        for fut in as_completed(futs):
            try:
                print(f"done {futs[fut]}: {fut.result():.0f}s (elapsed {time.time() - t0:.0f}s)", flush=True)
            except Exception:
                nfail += 1
                print(f"FAILED {futs[fut]}\n{traceback.format_exc()}", flush=True)
    print(f"{len(todo) - nfail} ok, {nfail} failed, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
