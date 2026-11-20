"""Spec 011 line C5: general-purpose strong compressors on the same code streams, as an independent check.

    set -a && source .env && set +a
    PYTHONPATH=. python experiments/011-limits-and-models/c5_generic.py prep          # input files
    PYTHONPATH=. python experiments/011-limits-and-models/c5_generic.py chain         # chain references
    PYTHONPATH=. python experiments/011-limits-and-models/c5_generic.py fast          # xz / zstd / bzip2, 48 matrices
    PYTHONPATH=. python experiments/011-limits-and-models/c5_generic.py strong <cmix|nncp> [jobs]
    PYTHONPATH=. python experiments/011-limits-and-models/c5_generic.py report

Input formats (one file per matrix):
  a : the merged 15-symbol codes, one byte each, row-major (exactly what the chain's code part transmits);
  b : the same with the block's scale byte placed before each 32-code block (33 bytes per block).
References, on the same matrix:
  chain_codes        : the 008 chain, code part, with the partner (as everywhere in 011; build_info.jsonl);
  chain_codes_nopart : the chain without the partner matrix (the compressors do not see the partner either);
  chain_scales       : the chain's scale stream; format b is compared with chain_codes_nopart + chain_scales.
Every output is decompressed and compared byte for byte with its input.
Tools (not committed): data/011/tools/cmix (GitHub byronknoll/cmix, commit recorded), nncp-2024-06-05 (bellard.org).
NNCP runs on 6 full matrices of T1 (experts L0.E191 and L26.E336), on the CPU (the GPUs are taken). cmix runs on the
first 512 rows of the same 6 matrices (reduced version: on the full matrices it ran at about 4-5 % per 26 minutes under
load, i.e. about 18 hours for encoding plus decoding), compared with the chain recomputed on the same 512 rows.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as K  # noqa: E402

D = Path(os.environ.get("D011", str(HERE.parents[1] / "data" / "011")))
W5 = D / "c5"
TOOLS = D / "tools"
RES = HERE / "results" / "c"
STRONG_SET = [f"{e}.{r}" for e in ("L0.E191", "L26.E336") for r in ("w1", "w2", "w3")]
CMIX = TOOLS / "cmix" / "cmix"
NNCP = TOOLS / "nncp-2024-06-05" / "nncp"
NNCP_THREADS = 4


CROP = 512


def inp(name, fmt, crop=False):
    return W5 / ("in512" if crop else "in") / f"{name}.{fmt}"


def stage_crop():
    """first CROP rows of the strong set: input files and the chain (without partner) on the crop."""
    (W5 / "in512").mkdir(parents=True, exist_ok=True)
    for name in STRONG_SET:
        for fmt in ("a", "b"):
            src = np.fromfile(inp(name, fmt), dtype=np.uint8)
            z = np.load(D / "codes" / f"{name}.npz")
            W = z["sym"].shape[1]
            per_row = W if fmt == "a" else W // 32 * 33
            src[:CROP * per_row].tofile(inp(name, fmt, crop=True))
    with ProcessPoolExecutor(6) as ex:
        rows = list(ex.map(chain_crop_one, STRONG_SET))
    with open(RES / "c5_chain_crop.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("CROP_DONE")


def chain_crop_one(name):
    import p1b
    C, E = K.read(name)
    C, E = C[:CROP], E[:CROP]
    ch = K.tx.chain(C, E)
    sym, p, Kk, side, scale, _ = p1b.chain_probs(C, E, p1b.active(ch))
    return {"task": name, "n": int(C.size), "chain_codes_nopart": float(-np.log2(p[np.arange(len(sym)), sym]).sum()),
            "chain_side_nopart": float(side), "chain_scales": float(scale)}


def prep_one(name):
    z = np.load(D / "codes" / f"{name}.npz")
    s, E = z["sym"], z["E"]
    s.tofile(inp(name, "a"))
    R, W = s.shape
    b = np.concatenate([E.reshape(R, W // 32, 1), s.reshape(R, W // 32, 32)], axis=2)
    b.tofile(inp(name, "b"))
    return name


def stage_prep():
    (W5 / "in").mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(24) as ex:
        list(ex.map(prep_one, K.T1 + K.T2))
    print("PREP_DONE")


def chain_one(name):
    import p1b
    C, E = K.read(name)
    ch = K.tx.chain(C, E)                                  # flags chosen without the partner
    sym, p, Kk, side, scale, _ = p1b.chain_probs(C, E, p1b.active(ch))
    z = np.load(D / "codes" / f"{name}.npz")
    assert (sym.reshape(C.shape) == z["sym"]).all()
    return {"task": name, "n": int(C.size), "chain_codes_nopart": float(-np.log2(p[np.arange(len(sym)), sym]).sum()),
            "chain_side_nopart": float(side), "chain_scales": float(scale)}


def stage_chain():
    info = {json.loads(l)["name"]: json.loads(l) for l in open(D / "build_info.jsonl")}
    with ProcessPoolExecutor(48) as ex:
        rows = list(ex.map(chain_one, K.T1 + K.T2))
    for r in rows:
        r["chain_codes"] = info[r["task"]]["chain_code_bits"]
    with open(RES / "c5_chain.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("CHAIN_DONE")


FAST = {"xz": (["xz", "-9e", "-T1", "-c"], ["xz", "-d", "-c"]),
        "zstd": (["zstd", "--ultra", "-22", "--long=27", "-c", "-q"], ["zstd", "-d", "--long=27", "-c", "-q"]),
        "bzip2": (["bzip2", "-9", "-c"], ["bzip2", "-d", "-c"])}


def run_pipe(cmd, src, dst):
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        subprocess.run(cmd, stdin=fi, stdout=fo, check=True)


def same(p, q):
    return Path(p).read_bytes() == Path(q).read_bytes()


def fast_one(args):
    name, fmt, tool = args
    src = inp(name, fmt)
    out = W5 / "fast" / f"{name}.{fmt}.{tool}"
    back = W5 / "fast" / f"{name}.{fmt}.{tool}.back"
    c, d = FAST[tool]
    t0 = time.time()
    run_pipe(c, src, out)
    t1 = time.time()
    run_pipe(d, out, back)
    t2 = time.time()
    ok = same(src, back)
    back.unlink()
    return {"task": name, "fmt": fmt, "tool": tool, "bytes": out.stat().st_size, "in_bytes": src.stat().st_size,
            "lossless": ok, "enc_s": t1 - t0, "dec_s": t2 - t1}


def stage_fast():
    (W5 / "fast").mkdir(parents=True, exist_ok=True)
    jobs = [(n, f, t) for n in K.T1 + K.T2 for f in ("a", "b") for t in FAST]
    with ProcessPoolExecutor(32) as ex:
        rows = list(ex.map(fast_one, jobs))
    with open(RES / "c5_fast.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("FAST_DONE", all(r["lossless"] for r in rows))


def strong_one(args):
    tool, name, fmt = args
    src = inp(name, fmt, crop=(tool == "cmix"))
    od = W5 / (tool + ("512" if tool == "cmix" else ""))
    out, back = od / f"{name}.{fmt}.{tool}", od / f"{name}.{fmt}.{tool}.back"
    if tool == "cmix":
        enc, dec = [str(CMIX), "-c", str(src), str(out)], [str(CMIX), "-d", str(out), str(back)]
    else:
        enc = [str(NNCP), "-q", "-T", str(NNCP_THREADS), "c", str(src), str(out)]
        dec = [str(NNCP), "-q", "-T", str(NNCP_THREADS), "d", str(out), str(back)]
    log = open(od / f"{name}.{fmt}.log", "w")
    t0 = time.time()
    subprocess.run(enc, check=True, stdout=log, stderr=subprocess.STDOUT)
    t1 = time.time()
    rec = {"task": name, "fmt": fmt, "tool": tool, "bytes": out.stat().st_size, "in_bytes": src.stat().st_size, "enc_s": t1 - t0}
    with open(RES / f"c5_{tool}.jsonl", "a") as f:          # the size is known now; the check follows
        f.write(json.dumps({**rec, "stage": "encoded"}) + "\n")
    subprocess.run(dec, check=True, stdout=log, stderr=subprocess.STDOUT)
    rec["dec_s"] = time.time() - t1
    rec["lossless"] = same(src, back)
    back.unlink()
    with open(RES / f"c5_{tool}.jsonl", "a") as f:
        f.write(json.dumps({**rec, "stage": "verified"}) + "\n")
    return rec


def stage_strong(tool, jobs):
    (W5 / (tool + ("512" if tool == "cmix" else ""))).mkdir(parents=True, exist_ok=True)
    work = [(tool, n, f) for n in STRONG_SET for f in ("a", "b")]
    with ProcessPoolExecutor(jobs) as ex:
        for r in ex.map(strong_one, work):
            print(tool, r, flush=True)
    print("STRONG_DONE", tool, flush=True)


def tool_versions():
    v = {}
    try:
        v["cmix_commit"] = subprocess.run(["git", "-C", str(TOOLS / "cmix"), "log", "-1", "--format=%H"], capture_output=True,
                                          text=True).stdout.strip()
    except Exception:
        v["cmix_commit"] = None
    tar = TOOLS / "nncp-2024-06-05.tar.gz"
    v["nncp_tar_sha256"] = hashlib.sha256(tar.read_bytes()).hexdigest() if tar.exists() else None
    for t in ("xz", "zstd", "bzip2"):
        v[t] = subprocess.run([t, "--version"], capture_output=True, text=True).stdout.splitlines()[:1] or \
            subprocess.run([t, "--version"], capture_output=True, text=True).stderr.splitlines()[:1]
    return v


def stage_report():
    ch_full = {json.loads(l)["task"]: json.loads(l) for l in open(RES / "c5_chain.jsonl")}
    ch_crop = {json.loads(l)["task"]: json.loads(l) for l in open(RES / "c5_chain_crop.jsonl")} if (RES / "c5_chain_crop.jsonl").exists() else {}
    out = {"versions": tool_versions(), "tools": {}, "note": "cmix: first 512 rows, compared with the chain on the same rows "
           "(no partner); chain_bits_per_code for cmix is also the no-partner crop chain"}
    rows = [json.loads(l) for l in open(RES / "c5_fast.jsonl")]
    for tool in ("cmix", "nncp"):
        p = RES / f"c5_{tool}.jsonl"
        if p.exists():
            rows += [r for r in map(json.loads, open(p)) if r.get("stage") == "verified"]
    for tool in sorted({r["tool"] for r in rows}):
        ch = ch_crop if tool == "cmix" else ch_full
        for fmt in ("a", "b"):
            sel = [r for r in rows if r["tool"] == tool and r["fmt"] == fmt]
            if not sel:
                continue
            for split, ns in (("T1", K.T1), ("T2", K.T2), ("strong6", STRONG_SET)):
                s = [r for r in sel if r["task"] in ns]
                if not s:
                    continue
                n = sum(ch[r["task"]]["n"] for r in s)
                bits = sum(8 * r["bytes"] for r in s)
                ref = sum(ch[r["task"]]["chain_codes_nopart"] + (ch[r["task"]]["chain_scales"] if fmt == "b" else 0) for r in s)
                ref_p = sum(ch[r["task"]].get("chain_codes", ch[r["task"]]["chain_codes_nopart"]) + (ch[r["task"]]["chain_scales"] if fmt == "b" else 0) for r in s)
                out["tools"][f"{tool}/{fmt}/{split}"] = {
                    "matrices": len(s), "complete": len(s) == len(ns), "all_lossless": all(r["lossless"] for r in s),
                    "bits_per_code": bits / n, "chain_nopart_bits_per_code": ref / n, "chain_bits_per_code": ref_p / n,
                    "gain_vs_chain_nopart": (ref - bits) / n, "gain_vs_chain": (ref_p - bits) / n,
                    "enc_codes_per_s": n / sum(r["enc_s"] for r in s), "dec_codes_per_s": n / sum(r["dec_s"] for r in s)}
    json.dump(out, open(RES / "c5_generic.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    RES.mkdir(parents=True, exist_ok=True)
    st = sys.argv[1]
    if st == "strong":
        stage_strong(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 12)
    else:
        {"prep": stage_prep, "chain": stage_chain, "crop": stage_crop, "fast": stage_fast, "report": stage_report}[st]()
