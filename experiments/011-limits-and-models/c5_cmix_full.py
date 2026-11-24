"""Spec 011 line C5 supplement: summary of cmix on one full matrix (run by c5_cmix_full.sh).

    python experiments/011-limits-and-models/c5_cmix_full.py

Reads data/011/c5/cmix_full/ (events and /usr/bin/time -v files) and writes results/c/c5_cmix_full.jsonl,
one record per format. The references are taken from existing C5/A1 result files, not recomputed:
  chain_nopart        : results/c/c5_chain.jsonl chain_codes_nopart (format a reference)
  chain_nopart+scales : the same + chain_scales (format b reference)
  chain (partner)     : results/c/c5_chain.jsonl chain_codes (+ chain_scales for b)
  nncp                : results/c/c5_nncp.jsonl, same full matrix, same format
  a1                  : results/a/a1.jsonl, the 160k network (as in the C5 text), test bits on the same matrix
                        (+ chain_scales for b); A1 uses the partner matrix
  cmix_first512       : results/c/c5_cmix.jsonl on the first 512 rows of the same matrix, with the crop chain
A format that has not finished is written with status "unfinished" and no bits.
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OD = ROOT / "data" / "011" / "c5" / "cmix_full"
RES = HERE / "results" / "c"
NAME = "L0.E191.w1"


def jl(p):
    return [json.loads(l) for l in open(p)]


def events(fmt):
    p = OD / f"{NAME}.{fmt}.events"
    ev = {}
    if p.exists():
        for l in open(p):
            k, *v = l.split()
            ev[k] = v
    return ev


def gnu_time(p):
    if not p.exists():
        return None
    t = p.read_text()
    m = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", t)
    if m is None:                                          # still running: the file is written at the end
        return None
    parts = [float(x) for x in m.group(1).split(":")]
    wall = sum(x * 60 ** i for i, x in enumerate(reversed(parts)))
    rss = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", t).group(1))
    code = int(re.search(r"Exit status: (\d+)", t).group(1))
    return {"wall_s": wall, "max_rss_kb": rss, "exit_status": code}


def main():
    ch = {r["task"]: r for r in jl(RES / "c5_chain.jsonl")}[NAME]
    crop = {r["task"]: r for r in jl(RES / "c5_chain_crop.jsonl")}[NAME]
    a1 = [r for r in jl(HERE / "results" / "a" / "a1.jsonl") if r["size"] == "160k"]
    assert len(a1) == 1
    a1 = a1[0]["test"][NAME]
    n = ch["n"]
    assert a1["n"] == n == 2304 * 5120
    out = []
    for fmt in ("a", "b"):
        sc = ch["chain_scales"] if fmt == "b" else 0.0
        nn = [r for r in jl(RES / "c5_nncp.jsonl") if r["task"] == NAME and r["fmt"] == fmt and r["stage"] == "verified"]
        cm = [r for r in jl(RES / "c5_cmix.jsonl") if r["task"] == NAME and r["fmt"] == fmt and r["stage"] == "verified"]
        ref = {"chain_nopart": (ch["chain_codes_nopart"] + sc) / n, "chain_partner": (ch["chain_codes"] + sc) / n,
               "nncp": nn[0]["bytes"] * 8 / n, "nncp_lossless": nn[0]["lossless"], "a1_160k": (a1["bits"] + sc) / n,
               "cmix_first512": cm[0]["bytes"] * 8 / crop["n"],
               "chain_nopart_first512": (crop["chain_codes_nopart"] + (crop["chain_scales"] if fmt == "b" else 0)) / crop["n"]}
        ev = events(fmt)
        src = ROOT / "data" / "011" / "c5" / "in" / f"{NAME}.{fmt}"
        rec = {"task": NAME, "fmt": fmt, "tool": "cmix", "rows": 2304, "cols": 5120, "n_codes": n,
               "in_bytes": src.stat().st_size, "in_sha256": ev.get("SRC_SHA256", [None])[0],
               "unit": "bits per code; format b includes the scale bytes; references for b include chain_scales",
               "refs": ref, "enc_time": gnu_time(OD / f"{NAME}.{fmt}.enc.time"),
               "dec_time": gnu_time(OD / f"{NAME}.{fmt}.dec.time")}
        if "DONE" in ev:
            b = int(ev["ENC_END"][1])
            assert b == (OD / f"{NAME}.{fmt}.cmix").stat().st_size
            rec.update(status="done", bytes=b, bits_per_code=b * 8 / n, lossless=ev["LOSSLESS"][0] == "1",
                       gain_vs_chain_nopart=ref["chain_nopart"] - b * 8 / n, gain_vs_nncp=ref["nncp"] - b * 8 / n,
                       gain_vs_a1=ref["a1_160k"] - b * 8 / n,
                       full_minus_first512=b * 8 / n - ref["cmix_first512"])
        else:
            rec.update(status="unfinished", stage=[k for k in ("ENC_START", "ENC_END", "DEC_START", "DEC_END") if k in ev])
        out.append(rec)
    with open(RES / "c5_cmix_full.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    for r in out:
        print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
