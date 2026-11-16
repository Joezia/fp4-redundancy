"""Resumable large-sample run of the indicators (spec 008 Part 1).

    python -m redundancy.sweep run --config <cfg> --plan <plan.json> --out <dir> [--workers N] [--stop-at EPOCH]
    python -m redundancy.sweep aggregate --config <cfg> --out <dir> --results <results dir> [--plan <plan.json>]

`run` executes the tasks of a plan in order with a process pool. A task is a list of entries plus the
figures to compute for them (e.g. the cheap set or everything). Every finished entry is appended at once
to <dir>/entries.jsonl (one JSON line per entry, with CPU seconds per indicator); entries already in that
file are skipped, so a crashed or stopped run continues where it ended without recomputing.
`--stop-at` stops submitting new tasks after that wall-clock time; tasks already running finish.

`aggregate` turns entries.jsonl into the result files of `redundancy.run` (data/<figure>.json and a
manifest), so the existing plot CLI draws them. A figure uses every entry for which it was computed.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait  # noqa: E402
from pathlib import Path  # noqa: E402

from . import adapters, indicators  # noqa: E402
from .run import clean, dump, record  # noqa: E402

ALL = ["R1", "R2", "R3", "R4", "R5", "R6", "R7a_unit", "R7a_row", "R7a_partner", "R7b", "R8", "R9", "R10"]
FIGURES = ["R1", "R2", "R3", "R4", "R5", "R6", "R7a_unit", "R7a_row", "R7a_partner", "R7b", "R8", "R9_max_level",
           "R9_odd_count", "R10"]
R7A_PART = {"R7a_unit": "unit_part", "R7a_row": "row_part", "R7a_partner": "partner_part"}


def compute_one(fig, entry, cfg, seed):
    """{figure: result} for one requested figure id (R7a parts are computed separately)."""
    if fig in R7A_PART:
        mod = indicators.MODULES["R7a"]
        return {fig: getattr(mod, R7A_PART[fig])(entry, cfg["indicators"]["R7a"], seed)}
    return indicators.MODULES[fig].compute(entry, cfg["indicators"][fig], seed)


def task(cfg, names, figs):
    out = []
    ad = adapters.get(cfg["adapter"])
    for n in names:
        w0, c0 = time.time(), time.process_time()
        e = ad.load(n)
        secs = {"load": time.process_time() - c0}
        res = {}
        for f in figs:
            if f not in cfg["indicators"] and f not in R7A_PART:
                continue
            c1 = time.process_time()
            res.update(compute_one(f, e, cfg, cfg["seed"]))
            secs[f] = time.process_time() - c1
        out.append({"entry": n, "figs": figs, "results": clean(res), "attrs": e.attrs, "cpu_seconds": secs,
                    "cpu_total": time.process_time() - c0, "wall": time.time() - w0,
                    "maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024})
        del e
    return out


def load_jsonl(path, keep_results=True):
    rows = {}
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # a line cut by a crash
                continue
            if not keep_results:
                r.pop("results", None)
            prev = rows.get(r["entry"])
            if prev is None:
                rows[r["entry"]] = r
            else:  # several runs of one entry (e.g. cheap set first, the rest later): merge
                prev["figs"] = sorted(set(prev["figs"]) | set(r["figs"]))
                if keep_results:
                    prev["results"].update(r["results"])
                prev["cpu_seconds"].update(r["cpu_seconds"])
    return rows


def run(args):
    cfg = json.load(open(args.config))
    plan = json.load(open(args.plan))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "entries.jsonl"
    done = {n: set(r["figs"]) for n, r in load_jsonl(path, keep_results=False).items()}
    todo = []
    for t in plan["tasks"]:
        figs = t["figs"]
        names = [n for n in t["entries"] if not set(figs) <= done.get(n, set())]
        if names:
            todo.append((t["id"], names, figs))
    print(f"{len(todo)} tasks to run ({len(plan['tasks'])} in plan), {args.workers} workers", flush=True)
    t0 = time.time()
    log = open(out / "progress.log", "a")
    with open(path, "a") as fo, ProcessPoolExecutor(args.workers) as ex:
        pending, it = {}, iter(todo)

        def submit():
            if args.stop_at and time.time() > args.stop_at:
                return False
            try:
                tid, names, figs = next(it)
            except StopIteration:
                return False
            pending[ex.submit(task, cfg, names, figs)] = tid
            return True

        for _ in range(args.workers):
            if not submit():
                break
        while pending:
            fin, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in fin:
                tid = pending.pop(fut)
                try:
                    rows = fut.result()
                    for r in rows:
                        fo.write(json.dumps(r, sort_keys=True) + "\n")
                    fo.flush()
                    os.fsync(fo.fileno())
                    msg = f"{time.strftime('%H:%M:%S')} done {tid}: {sum(r['cpu_total'] for r in rows):.0f} cpu-s"
                except Exception:
                    msg = f"{time.strftime('%H:%M:%S')} FAILED {tid}\n{traceback.format_exc()}"
                print(msg, flush=True)
                log.write(msg + "\n")
                log.flush()
                submit()
    print(f"finished in {time.time() - t0:.0f}s", flush=True)


def aggregate(args):
    cfg = json.load(open(args.config))
    rows = {}
    for d in args.out.split(","):
        for n, r in load_jsonl(Path(d) / "entries.jsonl").items():
            if n in rows:
                rows[n]["results"].update(r["results"])
                rows[n]["figs"] = sorted(set(rows[n]["figs"]) | set(r["figs"]))
            else:
                rows[n] = r
    order = None
    if args.plan:
        order = [n for t in json.load(open(args.plan))["tasks"] for n in t["entries"]]
    names = [n for n in (order or sorted(rows)) if n in rows]
    res = Path(args.results)
    manifest = {"dataset": cfg.get("name", "data"), "label": cfg.get("label", "data"), "config": cfg,
                "figures": [], "synthetic": {}, "verdicts": {}, "keys": {}, "n_entries": {}}
    for fig in FIGURES:
        items = [(n, rows[n]["results"][fig]) for n in names if fig in rows[n]["results"]]
        if not items:
            continue
        sel = [n for n, _ in items]
        icfg_key = "R7a" if fig.startswith("R7a") else fig.split("_")[0]
        if icfg_key not in cfg["indicators"]:
            continue
        rec = record(fig, "data", manifest["label"], items, cfg, {"entry_attrs": {n: rows[n]["attrs"] for n in sel}})
        dump(res / "data" / f"{fig}.json", rec)
        manifest["figures"].append(fig)
        manifest["verdicts"][fig] = rec["verdict"]
        manifest["keys"][fig] = rec["key"]
        manifest["n_entries"][fig] = len(items)
        print(f"{fig}: {len(items)} entries, verdict {rec['verdict']}", flush=True)
    dump(res / "manifest.json", manifest)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run")
    a.add_argument("--config", required=True)
    a.add_argument("--plan", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--workers", type=int, default=8)
    a.add_argument("--stop-at", type=float, default=None)
    b = sub.add_parser("aggregate")
    b.add_argument("--config", required=True)
    b.add_argument("--out", required=True, help="one or more run directories, comma separated")
    b.add_argument("--results", required=True)
    b.add_argument("--plan", default=None)
    args = ap.parse_args()
    {"run": run, "aggregate": aggregate}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
