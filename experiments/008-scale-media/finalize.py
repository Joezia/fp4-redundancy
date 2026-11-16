"""Spec 008: aggregate every result, write validation.json, the summary figures and the report tables.

    python experiments/008-scale-media/finalize.py [--skip-part1]

All numbers in the generated tables come from the result files; nothing is typed in by hand except the
slide-13 hand numbers (given in the spec) used for the V6 comparison.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import os
from pathlib import Path

import numpy as np

from redundancy import increment as I
from redundancy import increment_plot
from redundancy.adapters.media import items as media_items
from redundancy.increment_run import aggregate

E = Path("experiments/008-scale-media")
R = E / "results"
F = E / "figures"
SLIDE13 = {"R1c": 0.106, "R1s": 0.219, "R2": 0.127, "R7": 0.012, "R10": 0.117, "total": 0.581}  # bit per weight


def rows_of(d):
    p = Path(d) / "items.jsonl"
    out = {}
    if p.exists():
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[r["item"]] = r
    return list(out.values())


def media_results():
    res = {}
    spec = [("kodak", "all", "Kodak", lambda r: int(r["item"][5:7]) >= 13, "report: kodim13-24"),
            ("sqam", "report", "SQAM", None, "report: even tracks"),
            ("librispeech", "report", "LibriSpeech", None, "report: test-clean")]
    for ds, split, label, keep, note in spec:
        rows = rows_of(R / ds / split)
        if not rows:
            res[ds] = None
            continue
        sel = [r for r in rows if keep(r)] if keep else rows
        agg = aggregate(sel, label)
        agg["split"] = note
        agg["items"] = sorted(r["item"] for r in sel)
        if ds == "kodak":
            agg["all_24"] = aggregate(rows, "Kodak (all 24)")
        else:
            n_total = len(media_items(ds, "report"))
            agg["coverage"] = {"items_done": len(sel), "items_in_split": n_total,
                               "seconds_done": sum(r["meta"]["seconds_used"] for r in sel)}
        agg["params"] = sel[0]["params"]
        agg["switch_off_counts"] = {p: sum(1 for r in sel if not r["chains"]["default"]["used"].get(p, True))
                                    for p in sel[0]["chains"]["default"]["used"]}
        agg["if_forced"] = {o: {p: 100 * float(np.average([r["chains"][o]["if_forced"][p] for r in sel],
                                                         weights=[r["raw_bits"] for r in sel]))
                                for p in sel[0]["chains"][o]["if_forced"]} for o in sel[0]["chains"]}
        json.dump(agg, open(R / f"{ds}_increments.json", "w"), indent=1)
        res[ds] = agg
    return res


def fp4_results():
    out = {}
    shards = [R / "fp4" / d for d in ("increment_B", "increment_B_s1", "increment_B_s2", "increment_B_s3")]
    for name, dirs in [("v6_36", [R / "fp4" / "v6"]), ("B", shards)]:
        merged = {}
        for d in dirs:          # the B chain ran in four shards; each matrix appears once
            for r in rows_of(d):
                merged[r["item"]] = r
        rows = list(merged.values())
        if not rows:
            out[name] = None
            continue
        agg = aggregate(rows, "FP4")
        agg["items"] = sorted(r["item"] for r in rows)
        agg["params"] = rows[0]["params"]
        agg["switch_off_counts"] = {p: sum(1 for r in rows if not r["chains"]["default"]["used"].get(p, True))
                                    for p in rows[0]["chains"]["default"]["used"]}
        agg["if_forced"] = {o: {p: 100 * float(np.average([r["chains"][o]["if_forced"][p] for r in rows if o in r["chains"]],
                                                         weights=[r["raw_bits"] for r in rows if o in r["chains"]]))
                                for p in rows[0]["chains"][o]["if_forced"]} for o in rows[0]["chains"]}
        # bit per weight
        for o, v in agg["orders"].items():
            v["increment_bits_per_weight"] = {k: x * 4.25 / 100 for k, x in v["increment_raw"].items()}
            v["total_bits_per_weight"] = v["total"] * 4.25 / 100
        json.dump(agg, open(R / f"fp4_increments_{name}.json", "w"), indent=1)
        out[name] = agg
    return out


def v6_check(v6):
    if not v6:
        return {"name": "V6 FP4 (007's 36 matrices) vs slide 13 hand numbers", "pass": None, "note": "not finished"}
    o = v6["orders"]["default"]
    bw = o["increment_bits_per_weight"]
    ours = {"R1c": bw["R1c"], "R1s": bw["R1s"], "R2": bw["R2"], "R7": bw["R7"], "R10": bw["R10"],
            "total": o["total_bits_per_weight"]}
    diff = {k: ours[k] - SLIDE13[k] for k in SLIDE13}
    rest = {k: bw[k] for k in bw if k not in ("R1c", "R1s", "R2", "R7", "R10")}
    expl = None
    if (E / "dev" / "r3_blockshuffle.json").exists():
        expl = {"total_excess": "comes from R3 (slide 13 counts R3 as 0); after the scale context of R2, the two "
                                "previous codes still carry block-level information: the R3 increment survives a "
                                "within-block shuffle (dev/r3_blockshuffle.json)",
                "check": json.load(open(E / "dev" / "r3_blockshuffle.json"))["rows"]}
    return {"name": "V6 FP4: the 36 matrices of 007 recomputed with the chain module, compared with the slide-13 hand numbers",
            "explanation": expl,
            "unit": "bit per weight", "slide13": SLIDE13, "chain": ours, "difference": diff,
            "other_patterns": rest, "n_matrices": v6["n_items"],
            "tolerance": "|difference| <= 0.01 bit per weight per item; larger differences are explained in report.md",
            "over_tolerance": {k: v for k, v in diff.items() if abs(v) > 0.01},
            "pass": all(abs(v) <= 0.01 for v in diff.values())}


def validation(v6):
    v1d = json.load(open(E / "validation_1d.json"))
    img = json.load(open(E / "dev" / "validation_image.json"))
    checks = dict(v1d["checks"])
    checks.update(img["checks"])     # the image checks rerun after the image-plane fix replace the earlier ones
    checks["V6"] = v6_check(v6)
    out = {"unit": "% of raw bits unless stated", "zero_tolerance_percent": v1d["zero_tolerance"],
           "switch_rule": "patterns R2-R8 have one on/off bit per item; the encoder switches a pattern on only if "
                          "that lowers the code length (kept for later steps); 'if_forced_on' lists the increments "
                          "with the pattern always on",
           "all_V1_V5_pass": all(checks[k]["pass"] for k in ("V1", "V2", "V3", "V4", "V5")),
           "checks": checks}
    json.dump(out, open(E / "validation.json", "w"), indent=1)
    return out


def figures(media, fp4):
    F.mkdir(parents=True, exist_ok=True)
    names = {"kodak": ("Kodak (kodim13-24)", "kodak_summary.png"), "sqam": ("SQAM (even tracks)", "sqam_summary.png"),
             "librispeech": ("LibriSpeech (test-clean)", "librispeech_summary.png")}
    for ds, (label, fn) in names.items():
        if media.get(ds):
            increment_plot.draw(media[ds], label, F / fn)
    src = fp4.get("B") or fp4.get("v6_36")
    if src:
        label = f"FP4 ({src['n_items']} matrices)"
        increment_plot.draw(src, label, F / "fp4_summary.png")


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    media = media_results()
    fp4 = fp4_results()
    val = validation(fp4.get("v6_36"))
    figures(media, fp4)
    short = {ds: (None if a is None else {"default": a["orders"]["default"]["increment_raw"],
                                          "total": a["orders"]["default"]["total"], "codec": a["codec_saving"],
                                          "n": a["n_items"]}) for ds, a in media.items()}
    print(json.dumps(short, indent=1))
    print(json.dumps(val["checks"]["V6"], indent=1))


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- report tables (markdown, numbers from results)

PATS = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R10"]


def f2(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}"


def row_media(name, a):
    o = a["orders"]["default"]
    inc = o["increment_raw"]
    codec = "; ".join(f"{k} {v:.2f}" for k, v in a["codec_saving"].items())
    return f"| {name} | " + " | ".join(f2(inc.get(p)) for p in PATS) + f" | {o['total']:.2f} | {codec} |"


def row_fp4(name, a):
    o = a["orders"]["default"]
    inc = o["increment_raw"]
    r1 = f"{f2(inc['R1c'])} + {f2(inc['R1s'])}"
    return (f"| {name} | {r1} | " + " | ".join(f2(inc.get(p)) for p in PATS[1:]) +
            f" | {o['total']:.2f} | — |")


def tables(media, fp4):
    L = []
    head = "| 数据集 | " + " | ".join(PATS) + " | 合计 | 真实 codec |"
    L += [head, "| --- | " + " | ".join(["---:"] * len(PATS)) + " | ---: | --- |"]
    if fp4.get("B"):
        L.append(row_fp4(f"FP4（B 层 {fp4['B']['n_items']} 个矩阵；R1 = 码值 + scale）", fp4["B"]))
    for ds, nm in [("sqam", "SQAM（偶数曲目）"), ("librispeech", "LibriSpeech（test-clean）"), ("kodak", "Kodak（kodim13–24）")]:
        if media.get(ds):
            L.append(row_media(nm, media[ds]))
    if media.get("kodak"):
        L.append(row_media("Kodak（全部 24 张）", media["kodak"]["all_24"]))
    L.append("")
    # second order and the variant
    L += ["| 数据集 | 顺序 | 各项增量（%，按该顺序） | 合计 |", "| --- | --- | --- | ---: |"]
    for ds, a in [("FP4（007 的 36 个矩阵）", fp4.get("v6_36")), ("SQAM", media.get("sqam")),
                  ("LibriSpeech", media.get("librispeech")), ("Kodak（13–24）", media.get("kodak"))]:
        if not a:
            continue
        for o in ("alt", "alt_after_R1", "default"):
            if o not in a["orders"]:
                continue
            v = a["orders"][o]
            s = ", ".join(f"{k} {x:.2f}" for k, x in v["increment_raw"].items())
            L.append(f"| {ds} | {o} | {s} | {v['total']:.2f} |")
    L.append("")
    # switches and forced values
    L += ["| 数据集 | 被开关关掉的条目数（默认顺序） | 强制打开时的增量（%） |", "| --- | --- | --- |"]
    for ds, a in [("FP4 B 层", fp4.get("B")), ("FP4 36 个", fp4.get("v6_36")), ("SQAM", media.get("sqam")),
                  ("LibriSpeech", media.get("librispeech")), ("Kodak（13–24）", media.get("kodak"))]:
        if not a:
            continue
        off = ", ".join(f"{k} {v}/{a['n_items']}" for k, v in a["switch_off_counts"].items() if v)
        fo = ", ".join(f"{k} {v:.2f}" for k, v in a["if_forced"]["default"].items())
        L.append(f"| {ds} | {off or '无'} | {fo} |")
    return "\n".join(L) + "\n"


def run_tables():
    media = {ds: (json.load(open(R / f"{ds}_increments.json")) if (R / f"{ds}_increments.json").exists() else None)
             for ds in ("kodak", "sqam", "librispeech")}
    fp4 = {n: (json.load(open(R / f"fp4_increments_{n}.json")) if (R / f"fp4_increments_{n}.json").exists() else None)
           for n in ("B", "v6_36")}
    open(E / "report_tables.md", "w").write(tables(media, fp4))


if __name__ == "__main__":
    run_tables()
