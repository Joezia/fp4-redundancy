"""Spec 016: summarise every result file that exists into results/summary.json (and print the main tables).

    PYTHONPATH=.:experiments/016-cross-matrix python experiments/016-cross-matrix/summarize.py
"""
import json
from pathlib import Path

import numpy as np

import common as K

OUT = {}


def q(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], dtype=float)
    if not v.size:
        return None
    return {"n": int(v.size), "median": float(np.median(v)), "p90": float(np.quantile(v, .9)),
            "max": float(v.max()), "min": float(v.min()), "mean": float(v.mean())}


def get(d, path):
    for p in path.split("."):
        if d is None or p not in d:
            return None
        d = d[p]
    return d


def part_a(model):
    p = K.RES / f"pairs_gpu_{model}.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in open(p)]
    cats = sorted({r["cat"] for r in rows})
    res = {"n_pairs": len(rows), "seconds": sum(r["seconds"] for r in rows), "by_cat": {}}
    keys_sp = ["gain_code", "gain_code_given_own_scale", "gain_bit_sign", "gain_bit_e1", "gain_bit_e0", "gain_bit_m",
               "gain_scale_per_weight"]
    keys_gs = ["eq_max_z_nonzero_shift", "gain_code_at_best_shift_per_code", "eq_frac_at_best_shift_exact"] + \
              [f"{s}_max_z_nonzero_shift" for s in ("sign", "e1", "e0", "m", "v", "absv", "w")] + \
              [f"{s}_z_at_zero_shift" for s in ("eq", "sign", "e1", "e0", "m", "v", "absv", "w")]
    for c in cats:
        rs = [r for r in rows if r["cat"] == c]
        d = {"n": len(rs)}
        for tag in ("real", "shuffled"):
            for k in keys_sp:
                d[f"same_position.{k}.{tag}"] = q([get(r, f"{tag}.same_position.{k}") for r in rs])
            for k in keys_gs:
                d[f"global_shift.{k}.{tag}"] = q([get(r, f"{tag}.global_shift.{k}") for r in rs])
        for T in (32, 128):
            for tag in ("real", "shuffled"):
                tiles = [t for r in rs for t in r["local"][f"T{T}_{tag}"]]
                d[f"local.T{T}.eq_max.{tag}"] = q([t["eq_max"] - t["eq_p0"] for t in tiles])
                d[f"local.T{T}.evidence_bits.{tag}"] = q([t["evidence_bits"] for t in tiles])
                d[f"local.T{T}.frac_evidence_pos.{tag}"] = float(np.mean([t["evidence_bits"] > 0 for t in tiles]))
                d[f"local.T{T}.bits_max.{tag}"] = q([t["bits_max"] for t in tiles])
        # shifts found (real)
        d["eq_best_shifts_real"] = [get(r, "real.global_shift.eq_argmax_shift") for r in rs][:30]
        res["by_cat"][c] = d
    # strongest tiles overall
    tiles = [(t["evidence_bits"], r["cat"], r["a"], r["b"], T, t) for r in rows for T in (32, 128) for t in r["local"][f"T{T}_real"]]
    tiles.sort(key=lambda x: -x[0])
    res["top_tiles"] = [{"evidence_bits": e, "cat": c, "a": a, "b": b, "T": T, **t} for e, c, a, b, T, t in tiles[:15]]
    return res


def part_a_comp(model):
    p = K.RES / f"pairs_comp_{model}.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in open(p)]
    res = {"n_pairs": len(rows), "by_cat": {}}
    for c in sorted({r["cat"] for r in rows}):
        rs = [r for r in rows if r["cat"] == c]
        d = {}
        for rep in ("packed", "unpacked"):
            for cn in ("zstd19", "xz9e"):
                k = f"{rep}_{cn}"
                d[f"{k}.gain"] = q([r[k]["cond_gain_bits_per_code"] for r in rs])
                d[f"{k}.gain_control"] = q([r[k]["cond_gain_bits_per_code_control"] for r in rs])
                d[f"{k}.gain_minus_control"] = q([r[k]["cond_gain_bits_per_code"] - r[k]["cond_gain_bits_per_code_control"] for r in rs])
                d[f"{k}.NCD"] = q([r[k]["NCD"] for r in rs]); d[f"{k}.NCD_control"] = q([r[k]["NCD_control"] for r in rs])
        res["by_cat"][c] = d
    return res


def names(model):
    return [e["matrix"] for e in K.entries(model)]


def part_b_scan(model):
    for f in (K.RES / f"scan_{model}.npz", K.DATA / f"scan_{model}_ckpt.npz"):
        if f.exists():
            z = dict(np.load(f, allow_pickle=True))
            src = f.name
            break
    else:
        return None
    nm = names(model)
    n_scanned = int(z["matrices_scanned_to"]) if "matrices_scanned_to" in z else int(z["next"])
    res = {"source": src, "matrices_scanned": n_scanned, "matrices_total": len(nm),
           "random_order": bool(z.get("random_order", False)), "vec": {}, "pat": {}}
    for typ in ("neuron", "hidden"):
        n = int(z[f"vec_{typ}_n_real"])
        meta = z[f"vec_{typ}_meta"]
        d = {"n_real_queries": n}
        for sig in ("v", "w", "bits"):
            tv = z[f"vec_{typ}_top_v_{sig}"]; ti = z[f"vec_{typ}_top_i_{sig}"]
            cv = z[f"vec_{typ}_cat_v_{sig}"]
            real, null = tv[0, :n], tv[0, n:]
            if sig == "bits":                      # stored as dot of +-1 / (4L) -> bit-agreement fraction
                real, null, cv = (real + 1) / 2, (null + 1) / 2, (cv + 1) / 2
            d[sig] = {"top1_real": q(real), "top1_null": q(null),
                      "real_above_null_max": int((real > null.max()).sum()),
                      "by_category_real": {c: q(cv[k, :n]) for k, c in enumerate(["same_matrix", "same_expert", "same_layer", "other_layer"])},
                      "by_category_null": {c: q(cv[k, n:]) for k, c in enumerate(["same_matrix", "same_expert", "same_layer", "other_layer"])}}
            if sig != "bits":
                # lossy view: mean energy fraction removed by predicting the vector from its partner with optimal scale
                d[sig]["mean_cos2_real"] = {"best": float((real ** 2).mean()), **{c: float((cv[k, :n] ** 2).mean()) for k, c in
                                            enumerate(["same_matrix", "same_expert", "same_layer", "other_layer"])}}
                d[sig]["mean_cos2_null"] = {"best": float((null ** 2).mean()), **{c: float((cv[k, n:] ** 2).mean()) for k, c in
                                            enumerate(["same_matrix", "same_expert", "same_layer", "other_layer"])}}
            order = np.argsort(-real)[:10]
            d[sig]["strongest"] = [{"query": f"{nm[meta[j, 0]]}[{meta[j, 1]}]", "value": float(real[j]),
                                    "hit": f"{nm[int(ti[0, j]) // 100000]}[{int(ti[0, j]) % 100000}]"} for j in order]
        res["vec"][typ] = d
    for s in ("1x64", "8x8", "16x16"):
        if f"pat_{s}_n_real" not in z:
            continue
        n = int(z[f"pat_{s}_n_real"])
        meta = z[f"pat_{s}_meta"]
        d = {"n_real_queries": n}
        for w in ("other", "same"):
            b = z[f"pat_{s}_best_bits_{w}"]
            d[f"bits_{w}_real"] = q(b[:n]); d[f"bits_{w}_null"] = q(b[n:])
        b = z[f"pat_{s}_best_bits_other"]; wh = z[f"pat_{s}_where_bits_other"]
        d["real_other_above_null_max"] = int((b[:n] > b[n:].max()).sum())
        order = np.argsort(-b[:n])[:10]
        d["strongest_other"] = [{"query": f"{nm[meta[j, 0]]}@{meta[j, 1]},{meta[j, 2]}", "bits": float(b[j]),
                                 "hit": f"{nm[wh[j, 0]]}@{wh[j, 1]},{wh[j, 2]}"} for j in order]
        res["pat"][s] = d
    return res


def main():
    for model in K.MODELS:
        m = {}
        for name, f in (("A_gpu", part_a), ("A_comp", part_a_comp), ("B2_B3", part_b_scan)):
            r = f(model)
            if r is not None:
                m[name] = r
        import glob
        parts = sorted(glob.glob(str(K.RES / f"b1_{model}_s*p*.json")))
        if parts:
            ps = [json.load(open(f)) for f in parts]
            b1 = {"files": [Path(f).name for f in parts], "units": {}}
            for u in ps[0]["units"]:
                small = u not in ("block32", "patch8x32")
                src = ps[:1] if small else ps              # small units are complete in every part
                cm = [x["units"][u]["cross_matrix"] for x in src]
                b1["units"][u] = {"n_units_model": ps[0]["units"][u]["n_units_model"],
                                  "n_units_checked_cross": sum(x["units"][u]["n_units_checked_cross"] for x in src),
                                  "within_matrix": ps[0]["units"][u]["within_matrix"],
                                  "cross_repeat_units_by_hash": sum(c["repeat_units"] for c in cm),
                                  "cross_dup_groups_by_hash": sum(c["dup_groups"] for c in cm),
                                  "expected_false_hash_collisions": sum(c["expected_false_hash_collisions"] for c in cm),
                                  "gross_cost_bits": sum(c["gross_cost_bits"] for c in cm),
                                  "net_positive_bits_after_ref": sum(c["net_positive_bits_after_ref"] for c in cm),
                                  "flag_cost_bits_all_units": sum(c["flag_cost_bits_all_units"] for c in cm),
                                  "max_group_size": max(c["max_group_size"] for c in cm)}
            b1["subsets"] = [(x["hash_subsets_for_blocks_patches"], x["subset_index"]) for x in ps]
            b1["seconds"] = sum(x["seconds_total"] for x in ps)
            m["B1"] = b1
        if (K.RES / f"b1_verify_{model}.json").exists():
            v = json.load(open(K.RES / f"b1_verify_{model}.json"))
            m["B1_verify"] = {k: v[k] for k in ("n_groups", "n_true_repeats", "n_hash_collisions", "seconds")}
            m["B1_verify"]["true_groups"] = [{k: g[k] for k in ("unit", "n", "identical_codes", "identical_codes_and_scales")}
                                             | {"where": [(o["matrix"], o["row"], o["col"]) for o in g["occurrences"]],
                                                "codes": g["occurrences"][0]["codes"]}
                                             for g in v["groups"] if g["identical_codes"]]
        for t, fn in (("twins", f"twins_{model}.json"), ("twins_null", f"twins_{model}_null.json")):
            if (K.RES / fn).exists():
                m[t] = json.load(open(K.RES / fn))
        if (K.RES / f"joint_{model}.jsonl").exists():
            m["C_joint"] = [json.loads(l) for l in open(K.RES / f"joint_{model}.jsonl")]
        OUT[model] = m
    json.dump(OUT, open(K.RES / "summary.json", "w"), indent=1)
    print(json.dumps(OUT, indent=1)[:20000])


if __name__ == "__main__":
    main()
