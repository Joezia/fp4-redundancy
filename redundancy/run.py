"""Compute every indicator and write result files (spec 007 section 2).

    python -m redundancy.run --config redundancy/configs/<name>.json --out <results dir>

Writes <out>/data/<figure>.json (the measured dataset, when the config has an adapter),
<out>/synthetic/<key>_{none,has}.json (the two synthetic panels of every figure) and
<out>/manifest.json. All randomness comes from the seeds in the config; numerical libraries are
pinned to one thread so that two runs give byte-identical files. Progress goes to stdout only.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from . import adapters, indicators  # noqa: E402
from .adapters.synthetic import KEYS, Synthetic  # noqa: E402
from .indicators.common import na  # noqa: E402


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(clean(obj), sort_keys=True, indent=1, ensure_ascii=False))
        f.write("\n")


def module_of(fig):
    if fig.startswith("R9"):
        return indicators.MODULES["R9"]
    if fig.startswith("R7a"):
        return indicators.MODULES["R7a"]
    return indicators.MODULES[fig]


def enabled(cfg):
    ic = cfg.get("indicators", {})
    return [k for k in indicators.ORDER if k in ic and ic[k].get("enabled", True)]


def task_synthetic(cfg, key, side):
    t0 = time.time()
    e = Synthetic(cfg["synthetic"]).entry(key, side)
    ind, fig = KEYS[key]
    icfg = {k: v for k, v in cfg["indicators"].get(ind, {}).items() if k != "entries"}  # data-only filter
    res = indicators.MODULES[ind].compute(e, icfg, cfg["seed"])
    return {"fig": fig, "entry": e.name, "result": res[fig], "attrs": e.attrs, "seconds": time.time() - t0}


def task_data(cfg, name):
    t0 = time.time()
    e = adapters.get(cfg["adapter"]).load(name)
    out = {}
    for ind in enabled(cfg):
        out.update(indicators.MODULES[ind].compute(e, cfg["indicators"][ind], cfg["seed"]))
    return {"entry": name, "results": out, "attrs": e.attrs, "seconds": time.time() - t0}


def record(fig, dataset, label, items, cfg, extra=None):
    mod = module_of(fig)
    icfg = cfg["indicators"].get(fig.split("_")[0] if not fig.startswith("R7a") else "R7a", {})
    agg = mod.aggregate(fig, items, icfg, cfg["seed"])
    out = {"figure": fig, "pattern": indicators.pattern_of(fig), "dataset": dataset, "dataset_label": label,
           "verdict": agg["verdict"], "key": agg.get("key", ""), "summary": agg.get("summary", {}),
           "detail": agg.get("detail", {}), "entries": dict(items)}
    if extra:
        out.update(extra)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    out = Path(args.out)
    for sub in ("data", "synthetic"):
        if (out / sub).exists():
            shutil.rmtree(out / sub)
    workers = args.workers or cfg.get("workers", 8)
    adapter = adapters.get(cfg["adapter"]) if cfg.get("adapter") else None
    names = adapter.entries() if adapter else []
    syn_tasks = Synthetic(cfg["synthetic"]).tasks() if cfg.get("synthetic") else []
    syn_tasks = [(k, s) for k, s in syn_tasks if KEYS[k][0] in enabled(cfg)]
    print(f"{len(names)} data entries, {len(syn_tasks)} synthetic datasets, {workers} workers", flush=True)
    t0 = time.time()
    data_res, syn_res, failed = {}, {}, []
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(task_data, cfg, n): ("data", n) for n in names}
        futs.update({ex.submit(task_synthetic, cfg, k, s): ("syn", (k, s)) for k, s in syn_tasks})
        for fut in as_completed(futs):
            kind, key = futs[fut]
            try:
                r = fut.result()
            except Exception:
                failed.append(str(key))
                print(f"FAILED {key}\n{traceback.format_exc()}", flush=True)
                continue
            (data_res if kind == "data" else syn_res)[key] = r
            print(f"done {key}: {r['seconds']:.0f}s (elapsed {time.time() - t0:.0f}s)", flush=True)
    if failed:
        print(f"{len(failed)} tasks failed; no result files written", flush=True)
        sys.exit(1)

    manifest = {"dataset": cfg.get("name", "data"), "label": cfg.get("label", cfg.get("name", "data")),
                "config": cfg, "figures": [], "synthetic": {}, "verdicts": {}, "keys": {}}
    for (key, side), r in sorted(syn_res.items()):
        fig = r["fig"]
        rec = record(fig, f"synthetic/{key}_{side}", side, [(r["entry"], r["result"])], cfg,
                     {"description": r["attrs"].get("description", ""), "side": side})
        dump(out / "synthetic" / f"{key}_{side}.json", rec)
        manifest["synthetic"].setdefault(key, {})[side] = {"file": f"synthetic/{key}_{side}.json",
                                                           "verdict": rec["verdict"], "figure": fig}
    if names:
        figs = []
        for n in names:
            for fig in data_res[n]["results"]:
                if fig not in figs:
                    figs.append(fig)
        order = {k: i for i, k in enumerate(indicators.ORDER)}
        figs.sort(key=lambda f: (order[f.split("_")[0] if not f.startswith("R7a") else "R7a"], f))
        for fig in figs:
            items = [(n, data_res[n]["results"].get(fig, na("not computed for this entry"))) for n in names]
            rec = record(fig, "data", manifest["label"], items, cfg,
                         {"entry_attrs": {n: data_res[n]["attrs"] for n in names}})
            dump(out / "data" / f"{fig}.json", rec)
            manifest["figures"].append(fig)
            manifest["verdicts"][fig] = rec["verdict"]
            manifest["keys"][fig] = rec["key"]
    dump(out / "manifest.json", manifest)
    print(f"written {out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
