"""Run section-7 measurements on the 36 sampled matrices (one process per matrix).

Refuses to start unless validation.json reports all section-8 checks passed.
Per-matrix results (rows + 16-bin histogram) are pickled to --work (outside the repo); only the
aggregated small files are written to experiments/r1-r4/ by diagnostics.report.

Usage: python -m diagnostics.run_model --snapshot $SNAPSHOT --work <dir> [--jobs 36] [--only NAME]
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
    os.environ.setdefault(_v, "1")

from .measure import measure_matrix, unpack_codes  # noqa: E402
from .st import index_snapshot, read_bytes  # noqa: E402

B = 32
SEED_BASE = 20261112


def work_one(snapshot, sample, idx, work):
    t0 = time.time()
    index = index_snapshot(snapshot)
    packed = read_bytes(index[sample["weight"]])
    E = read_bytes(index[sample["scale"]])
    C = unpack_codes(packed)
    assert list(C.shape) == sample["codeword_shape"]
    meta = {"matrix": sample["matrix"], "layer": sample["layer"], "expert": sample["expert"],
            "role": sample["role"]}
    res = measure_matrix(C, E, B, meta, seed=SEED_BASE * 100 + idx)
    res["meta"] = meta
    res["seconds"] = time.time() - t0
    with open(Path(work) / f"{sample['matrix']}.pkl", "wb") as f:
        pickle.dump(res, f)
    return sample["matrix"], res["seconds"], len(res["rows"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--samples", default="experiments/r1-r4/samples.json")
    ap.add_argument("--validation", default="experiments/r1-r4/validation.json")
    ap.add_argument("--work", required=True)
    ap.add_argument("--jobs", type=int, default=36)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    with open(args.validation) as f:
        if not json.load(f)["all_pass"]:
            raise SystemExit("section 8 validation has failures; refusing to read model data")
    with open(args.samples) as f:
        samples = json.load(f)["matrices"]
    Path(args.work).mkdir(parents=True, exist_ok=True)
    todo = [(i, s) for i, s in enumerate(samples) if not args.only or s["matrix"] in args.only]
    failures = {}
    with ProcessPoolExecutor(min(args.jobs, len(todo))) as ex:
        futs = {ex.submit(work_one, args.snapshot, s, i, args.work): s["matrix"] for i, s in todo}
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                n, sec, nrows = fut.result()
                print(f"done {n}: {sec:.0f}s, {nrows} rows", flush=True)
            except Exception:
                failures[name] = traceback.format_exc()
                print(f"FAILED {name}\n{failures[name]}", flush=True)
    with open(Path(args.work) / "failures.json", "w") as f:
        json.dump(failures, f, indent=1)
    print(f"{len(todo) - len(failures)} ok, {len(failures)} failed")


if __name__ == "__main__":
    main()
