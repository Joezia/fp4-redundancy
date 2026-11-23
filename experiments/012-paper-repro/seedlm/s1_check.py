"""S1 checks: paper examples, maximal length, GPU vs numpy, and comparison with the third-party
reproduction qiuyu-ren/seedlm-repro (@5f42af6) on the same inputs.

Writes results/s1/s1_check.json.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import slm as S  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
OUT = os.path.join(HERE, "../results/s1")
os.makedirs(OUT, exist_ok=True)
REPRO = os.path.join(ROOT, "data/012/src/qiuyu-ren_seedlm-repro")


def rel(W, R):
    return float(np.linalg.norm(W - R) / np.linalg.norm(W))


def main():
    res = {}
    # 1. Figure 4 / A.2: K = 3, seed 4 -> V(4) = [[2,5],[6,7],[3,1],[4,2]]
    v = S.lfsr_sequence(3, 4, 8).reshape(4, 2)
    res["fig4_V4"] = v.tolist()
    res["fig4_ok"] = v.tolist() == [[2, 5], [6, 7], [3, 1], [4, 2]]
    res["fig4_V_by_seed_ok"] = S.V_by_seed(3, 4, 2, [4])[0].tolist() == v.tolist()
    # 2. maximal length of every Table 6 tap set
    t0 = time.time()
    res["maximal"] = {K: bool(S.is_maximal(K)) for K in sorted(S.TAPS)}
    res["maximal_s"] = time.time() - t0
    # 3. pinv memory (paper: "at most 6.3MB")
    for (C, P) in [(8, 3), (12, 4)]:
        res[f"pinv_bytes_fp32_C{C}P{P}"] = ((1 << 16) - 1) * C * P * 4
    # 4. GPU vs numpy, and vs the third-party reproduction, on Gaussian blocks
    import torch
    import seedlm_gpu as G
    sys.path.insert(0, REPRO)
    import importlib
    R_compress = importlib.import_module("seedlm.compress")  # third party package (ours is slm.py)
    rng = np.random.default_rng(0)
    for (C, P, K) in [(8, 3, 16), (12, 4, 16)]:
        cb = S.Codebook(K, C, P, device="cuda")
        gcb = G.GPUCodebook(cb)
        nb = 2048
        Wm = rng.standard_normal((nb * C,)).astype(np.float32).reshape(-1, C)
        for mode in ("paper", "range"):
            key = f"C{C}P{P}K{K}_{mode}"
            t0 = time.time()
            s_n, q_n, e_n, err_n = S.search_numpy(Wm[:256], cb, mode)
            t_np = time.time() - t0
            t0 = time.time()
            s_g, q_g, e_g, err_g = G.search(Wm, gcb, mode)
            torch.cuda.synchronize()
            t_gpu = time.time() - t0
            Rg = S.reconstruct(s_g, q_g, e_g, K, C, P)
            # third party: its compress() takes a whole tensor and returns cached-mode offsets
            cfg = R_compress.SeedLMConfig(C=C, P=P, K=K)
            t0 = time.time()
            ct = R_compress.compress(Wm[:256].reshape(-1), cfg, exponent_mode=mode)
            t_tp = time.time() - t0
            tp_seeds = ct.hardware_seeds()
            Rt = R_compress.decompress(ct).reshape(-1, C)
            res[key] = dict(
                gpu_vs_numpy_seed_agree=float(np.mean(s_g[:256] == s_n)),
                gpu_vs_numpy_err_maxrel=float(np.max(np.abs(err_g[:256] - err_n) / np.maximum(err_n, 1e-30))),
                thirdparty_seed_agree=float(np.mean(tp_seeds == s_g[:256])),
                rel_err_ours_2048=rel(Wm, Rg), rel_err_ours_256=rel(Wm[:256], Rg[:256]),
                rel_err_thirdparty_256=rel(Wm[:256], Rt),
                floor_frac=float(np.mean(e_g == -8)),
                t_numpy_256=t_np, t_gpu_2048=t_gpu, t_thirdparty_256=t_tp,
                bits_per_element=S.bits_per_element(C, P, K))
            print(key, json.dumps(res[key]), flush=True)
    json.dump(res, open(os.path.join(OUT, "s1_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
