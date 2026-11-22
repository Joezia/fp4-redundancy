"""LLM.265 (Xu et al., MICRO 2025) on FP4 matrices: the original lossy pipeline, then made lossless.

Lossy part (follows the released code, pytorch_nvenc/utils/eval_compress_quarot_model_variable_bitrate*.py):
  - input: the dequantised FP4 matrix (code value x 2^(E-127)); w2 (down projection) is transposed first, w1/w3
    (gate/up) are not, as the original does for down_proj vs gate/up_proj;
  - CWQuantization: per row asymmetric min-max to uint8 (scale = (max-min)/255, zero = round(-min/scale)), with
    scale and zero kept as FP16 (the original computes them in FP16 and does not count them; we report both);
  - FixedTiling: the 2304 x 5120 image is cut into 4 frames of 2304 x 1280 (the original uses 2048 x 2048 for the
    LLaMA shapes; 5120 is not a multiple of 2048, so the frame width is changed, no padding is needed);
  - each frame is encoded on its own (a new encoder per tile, so intra only) with NVENC HEVC, preset P7,
    tuning HQ (the original's settings). Rate: the original sets a VBR average bitrate (max = 10 x average).
    Through ffmpeg, VBR on a one-frame stream ignores the target (synthetic test: 0.25-4 bit/weight targets all
    gave 3.5 bit/weight) and max bitrates above ~800 Mbit/s are rejected, so the sweep uses constant QP
    (target 'qpN'), which is what a one-frame VBR stream ends up choosing anyway; the actual bits are reported.
    The original uses NV12 + monoChromeEncoding; ffmpeg does not expose monochrome HEVC, so the chroma planes are
    a constant 128 and their cost is included in the bitstream size;
  - the bitstream is decoded with ffmpeg's software HEVC decoder (standard, bit-exact) and dequantised.
  Not reproduced: QuaRot rotation done by the released code before compression (not in the paper text), and the
  per-layer bitrate schedule B = k*l + b (needs model accuracy); every matrix uses the same target here.
Lossless part (as SeedLM S3-b): the true codes (R10-merged, 15 symbols) are coded in raster order with an
adaptive count model whose context is (interval of w_hat on the FP4 grid of the weight's own block scale with S
sub-bins, block scale byte E). S is chosen on train8 per target and applied to T1/T2, where the codes are really
encoded and decoded with racoder.c and compared bit for bit; the HEVC bitstream is decoded twice independently
and the two reconstructions must be identical.
Total = HEVC bitstream + FP16 scale/zero per row + conditional code + 008 scale stream (reference.json).

usage: python l265_run.py SET TARGET[,TARGET...]       (SET = train8 | T1 | T2; TARGET = qpN | raw8 | bits/weight)
  raw8 skips HEVC: the uint8 image itself is the side information (8 bit/weight), as step (1) of the paper's Fig. 2.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../seedlm"))
import s3_run as R  # noqa: E402
import s3b_lossless as B  # noqa: E402
import racoder  # noqa: E402
from txcore import dirichlet  # noqa: E402

FFMPEG = os.environ.get("FFMPEG", os.path.expanduser("~/.local/share/miniforge3/envs/llm265/bin/ffmpeg"))
OUT = os.path.join(HERE, "../results/llm265")
FPS = 30
MAXRATE = 800_000_000      # NVENC rejects a max bitrate above ~800 Mbit/s (HEVC high tier); 10 x average is capped
TILE_W = 1280
SUBS = B.SUBS
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("L265_GPU", "1"))


def cw_quant(X):
    """per row min-max uint8 (CWQuantization); scale and zero stored as FP16."""
    mn = X.min(axis=1)
    mx = X.max(axis=1)
    scale = (np.maximum(mx - mn, 1e-5) / 255).astype(np.float16)
    s32 = scale.astype(np.float32)
    zero = np.round(-mn / s32).astype(np.float16)
    z32 = zero.astype(np.float32)
    q = np.clip(np.round(X / s32[:, None]) + z32[:, None], 0, 255).astype(np.uint8)
    return q, s32, z32


def dequant(q, s32, z32):
    return (s32[:, None] * (q.astype(np.float32) - z32[:, None])).astype(np.float32)


def nv12(y):
    h, w = y.shape
    return y.tobytes() + bytes([128]) * (h // 2 * w)


def encode_frame(y, target):
    """target 'qpN': constant QP N (used for the sweep); a number: VBR at that many bits/weight (original knob)."""
    h, w = y.shape
    if target.startswith("qp"):
        rc = ["-rc", "constqp", "-qp", target[2:]]
    else:
        br = int(round(float(target) * h * w * FPS))
        rc = ["-rc", "vbr", "-b:v", str(min(br, MAXRATE)), "-maxrate", str(min(10 * br, MAXRATE))]
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "nv12", "-s", f"{w}x{h}",
           "-r", str(FPS), "-i", "-", "-frames:v", "1", "-c:v", "hevc_nvenc", "-gpu", "0", "-preset", "p7",
           "-tune", "hq", *rc, "-bf", "0", "-f", "hevc", "-"]
    p = subprocess.run(cmd, input=nv12(y), capture_output=True, env=ENV)
    if p.returncode != 0 or not p.stdout:
        raise RuntimeError(p.stderr.decode()[-2000:])
    return p.stdout


def decode_frame(bs, h, w):
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-c:v", "hevc", "-f", "hevc", "-i", "-", "-f", "rawvideo",
           "-pix_fmt", "nv12", "-"]
    p = subprocess.run(cmd, input=bs, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode()[-2000:])
    raw = np.frombuffer(p.stdout, dtype=np.uint8)
    assert raw.size == h * w * 3 // 2, (raw.size, h, w)
    return raw[:h * w].reshape(h, w)


def hevc_roundtrip(q, target):
    """encode each tile, decode it twice; returns (decoded image, bitstream bytes, both decodes equal)."""
    h, w = q.shape
    assert w % TILE_W == 0
    out = np.empty_like(q)
    nbytes = 0
    same = True
    for j in range(0, w, TILE_W):
        tile = np.ascontiguousarray(q[:, j:j + TILE_W])
        bs = encode_frame(tile, target)
        nbytes += len(bs)
        d1 = decode_frame(bs, h, TILE_W)
        d2 = decode_frame(bytes(bs), h, TILE_W)
        same &= bool(np.array_equal(d1, d2))
        out[:, j:j + TILE_W] = d1
    return out, nbytes, same


def run_matrix(name, target, which, chosen=None):
    W, sym, scale = R.load(name)
    tr = name.endswith("w2")
    X = W.T if tr else W
    q, s32, z32 = cw_quant(X)
    t0 = time.time()
    if target == "raw8":
        qh, hevc_bits, same = q, 8 * q.size, True
    else:
        qh, nbytes, same = hevc_roundtrip(q, target)
        hevc_bits = 8 * nbytes
    t_hevc = time.time() - t0
    Xh = dequant(qh, s32, z32)
    What = Xh.T if tr else Xh
    n = W.size
    rows = q.shape[0]
    qp_bits = 32 * rows
    W64 = W.astype(np.float64)
    Wq = dequant(q, s32, z32)
    res = dict(matrix=name, target=target, n=int(n), rows=int(rows), transposed=tr,
               hevc_bits=int(hevc_bits), qparam_bits=int(qp_bits),
               lossy_bpw=hevc_bits / n, lossy_bpw_with_qparams=(hevc_bits + qp_bits) / n,
               decode_twice_equal=same, hevc_s=t_hevc,
               mse_uint8=float(np.mean((qh.astype(np.float64) - q) ** 2)),
               rel_mse=float(((W64 - What) ** 2).sum() / (W64 ** 2).sum()),
               rel_mse_8bit_only=float(((W64 - (Wq.T if tr else Wq)) ** 2).sum() / (W64 ** 2).sum()),
               code_mismatch=float(np.mean(R.nearest_code(What.astype(np.float64), scale) != sym)),
               ideal_bits={})
    u = What.astype(np.float64) / scale
    E_w = np.round(np.log2(scale)).astype(np.int64) + 127
    for sb in SUBS:
        res["ideal_bits"][str(sb)] = float(dirichlet(sym.ravel(), B.contexts(u, E_w, sb).ravel(), 15))
    if chosen is not None:
        ctx = B.contexts(u, E_w, chosen).ravel()
        t0 = time.time()
        words, nctx = racoder.encode(sym.ravel(), ctx)
        t_enc = time.time() - t0
        t0 = time.time()
        dec = racoder.decode(words, ctx, n, nctx)
        t_dec = time.time() - t0
        ok = bool(np.array_equal(dec, sym.ravel())) and same
        ref = json.load(open(B.REF))[which]["per_matrix"][name]
        code_bits = 32 * len(words)
        total = hevc_bits + qp_bits + code_bits + ref["scale_bits"]
        res.update(sub=chosen, code_bits_actual=int(code_bits), roundtrip_ok=ok, scale_bits=ref["scale_bits"],
                   total_bits=float(total), ref_total_bits=ref["total"], net_gain_bits=float(ref["total"] - total),
                   bpw_code=code_bits / n, bpw_total=float(total / n),
                   bpw_total_without_qparams=float((total - qp_bits) / n),
                   net_gain_bpw=float((ref["total"] - total) / n), encode_s=t_enc, decode_s=t_dec)
    return res


def done_keys(path):
    out = set()
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            out.add((r["matrix"], str(r["target"])))
    return out


def main():
    which = sys.argv[1]
    targets = sys.argv[2].split(",")
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"l265_{which}.jsonl")
    done = done_keys(path)
    for target in targets:
        chosen = None
        if which != "train8":
            tr = [json.loads(l) for l in open(os.path.join(OUT, "l265_train8.jsonl"))]
            tr = [r for r in tr if str(r["target"]) == target]
            assert len(tr) == 8, f"need the 8 training matrices first ({target}: {len(tr)})"
            chosen = int(min(SUBS, key=lambda s: sum(r["ideal_bits"][str(s)] for r in tr)))
        for name in R.names(which):
            if (name, target) in done:
                continue
            r = run_matrix(name, target, which, chosen)
            with open(path, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(json.dumps({k: v for k, v in r.items() if k != "ideal_bits"}), flush=True)


if __name__ == "__main__":
    main()
