"""Spec 009: check the Kimi-K3 reading path of the FP4 adapter.

    PYTHONPATH=. python experiments/009-kimi-k3/check_adapter.py [--real L1.E0.w1,...]

1. Synthetic: random E2M1 values are packed exactly as compressed-tensors `pack_fp4_to_uint8` does
   (magnitude index | sign << 3; element 2i in the low nibble) and written, with E8M0 scales, to a tiny
   safetensors file under Kimi-K3 tensor names. `list_matrices(fmt="kimi")` + `FP4._read` must give back
   the same codes and scales, and the dequantised values lut[code] * 2^(scale - 127) must equal the input.
2. Real (needs KIMI_SNAPSHOT): for the named matrices, re-pack the unpacked codes and compare with the
   stored bytes; report the sign balance, the largest level per 32-block, the scale range, and the
   correlation of corresponding W1 and W3 rows (expected clearly > 0 if rows and roles are read right).
Writes experiments/009-kimi-k3/adapter_check.json.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import struct
import tempfile

import numpy as np

from redundancy.adapters.fp4 import E2M1, FP4, list_matrices

OUT = "experiments/009-kimi-k3/adapter_check.json"
LUT = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def ct_pack(values):
    """numpy transcription of compressed_tensors pack_fp4_to_uint8 (nearest magnitude, sign bit 3)."""
    idx = np.abs(np.abs(values)[..., None] - LUT).argmin(-1) + (np.signbit(values).astype(np.int64) << 3)
    flat = idx.reshape(-1, 2)
    return (flat[:, 0] | (flat[:, 1] << 4)).astype(np.uint8).reshape(values.shape[0], values.shape[1] // 2)


def write_st(path, tensors):
    header, off, blobs = {}, 0, []
    for name, arr in tensors.items():
        b = np.ascontiguousarray(arr).tobytes()
        header[name] = {"dtype": "U8", "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    h = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def synthetic():
    rng = np.random.default_rng(9)
    pre = "language_model.model.layers.{}.block_sparse_moe.experts.{}.w{}."
    tensors, truth = {}, {}
    for L, e, r, shape in [(1, 0, 1, (6, 64)), (1, 0, 2, (8, 32)), (1, 0, 3, (6, 64)), (5, 7, 1, (4, 96))]:
        mag = rng.integers(0, 8, size=shape)
        sign = rng.integers(0, 2, size=shape)
        vals = LUT[mag] * np.where(sign == 1, -1.0, 1.0)
        vals[sign == 1 & (mag == 0)] = -0.0
        scale = rng.integers(110, 130, size=(shape[0], shape[1] // 32)).astype(np.uint8)
        tensors[pre.format(L, e, r) + "weight_packed"] = ct_pack(vals)
        tensors[pre.format(L, e, r) + "weight_scale"] = scale
        truth[f"L{L}.E{e}.w{r}"] = (vals, scale)
    with tempfile.TemporaryDirectory() as d:
        write_st(os.path.join(d, "model-00001-of-00001.safetensors"), tensors)
        lst = list_matrices(d, "kimi")
        sp = os.path.join(d, "samples.json")
        json.dump(lst, open(sp, "w"))
        ad = FP4({"samples": sp, "snapshot": d})
        res = {}
        for name, (vals, scale) in truth.items():
            C, E = ad._read(ad.samples[name])
            deq = E2M1[C] * np.exp2(np.repeat(E.astype(np.float64), 32, axis=1) - 127)
            ref = vals * np.exp2(np.repeat(scale.astype(np.float64), 32, axis=1) - 127)
            res[name] = {"codes_match_values": bool(np.array_equal(E2M1[C], vals) and
                                                    np.array_equal(np.signbit(E2M1[C]), np.signbit(vals))),
                         "scales_match": bool(np.array_equal(E, scale)),
                         "dequant_match": bool(np.array_equal(deq, ref))}
        listed = sorted(m["matrix"] for m in lst["matrices"])
    ok = listed == sorted(truth) and all(all(v.values()) for v in res.values())
    return {"pass": ok, "listed": listed, "per_matrix": res}


def real(names):
    ad = FP4({"samples": "data/009/kimi/matrices.json", "snapshot_env": "KIMI_SNAPSHOT"})
    from diagnostics.st import index_snapshot, read_bytes
    idx = index_snapshot(ad.snapshot)
    out = {}
    codes = {}
    for n in names:
        s = ad.samples[n]
        C, E = ad._read(s)
        raw = read_bytes(idx[s["weight"]])
        repack = (C[:, 0::2] | (C[:, 1::2] << 4)).astype(np.uint8)
        lvl = (C & 7).reshape(C.shape[0], -1, 32).max(axis=2)
        out[n] = {"shape": list(C.shape), "repack_equals_stored_bytes": bool(np.array_equal(repack, raw)),
                  "negative_fraction_of_nonzero": float(((C >= 9)).sum() / max(1, ((C & 7) > 0).sum())),
                  "zero_code_fraction": float(np.isin(C, [0, 8]).mean()),
                  "block_max_level_hist": {str(LUT[k]): float((lvl == k).mean()) for k in range(8)},
                  "scale_min": int(E.min()), "scale_median": float(np.median(E)), "scale_max": int(E.max())}
        codes[n] = C
    for n in names:
        if n.endswith(".w1") and n[:-3] + ".w3" in codes:
            a, b = E2M1[codes[n]], E2M1[codes[n[:-3] + ".w3"]]
            a = a - a.mean(1, keepdims=True)
            b = b - b.mean(1, keepdims=True)
            r = (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))
            rng = np.random.default_rng(0)
            bp = b[rng.permutation(b.shape[0])]
            r0 = (a * bp).sum(1) / np.sqrt((a * a).sum(1) * (bp * bp).sum(1))
            out[n]["w1_w3_row_abs_corr_median"] = float(np.median(np.abs(r)))
            out[n]["w1_w3_row_abs_corr_median_random_pairing"] = float(np.median(np.abs(r0)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="")
    a = ap.parse_args()
    res = {"synthetic": synthetic()}
    print("synthetic:", "PASS" if res["synthetic"]["pass"] else "FAIL", flush=True)
    if a.real:
        res["real"] = real(a.real.split(","))
        print(json.dumps(res["real"], indent=1))
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
