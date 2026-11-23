"""S2 (part 2): reconstruction-level check on real LLM weights against the third-party reproduction.

The paper's models (Llama 2 / 3) are not accessible here (report section 1). Instead, 14 weight tensors of a model
already in the local Hugging Face cache (Qwen3-4B-Instruct-2507, layers 0 and 18: q/k/v/o, gate/up/down) are
sliced like the third party does for Llama (the first 32,768 elements of each tensor, bf16 -> float32), and both
implementations compress them with both Table 1 configurations and both exponent rules. Reported: relative
error ||W - W_hat|| / ||W|| of each, seed agreement, and the fraction of blocks at the exponent floor.
"""
import glob
import importlib
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import slm as S  # noqa: E402
import seedlm_gpu as G  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, os.path.join(ROOT, "data/012/src/qiuyu-ren_seedlm-repro"))
TP = importlib.import_module("seedlm.compress")
OUT = os.path.join(HERE, "../results/s1")
SNAP = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/"))[0]


def tensors():
    from safetensors import safe_open
    want = [f"model.layers.{L}.{m}.weight" for L in (0, 18)
            for m in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                      "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")]
    out = {}
    for f in sorted(glob.glob(SNAP + "*.safetensors")):
        with safe_open(f, "pt") as sf:
            for k in sf.keys():
                if k in want:
                    out[k] = sf.get_tensor(k).float().reshape(-1)[:32768].numpy()
    return out


def rel(W, R):
    return float(np.linalg.norm(W.astype(np.float64) - R) / np.linalg.norm(W.astype(np.float64)))


def main():
    os.makedirs(OUT, exist_ok=True)
    T = tensors()
    path = os.path.join(OUT, "s2_realweights.jsonl")
    for (C, P, K) in [(8, 3, 16), (12, 4, 16)]:
        gcb = G.GPUCodebook(S.Codebook(K, C, P, device="cuda"))
        for name, w in T.items():
            for mode in ("range", "paper"):
                blocks = w.reshape(-1, C) if len(w) % C == 0 else np.concatenate(
                    [w, np.zeros((-len(w)) % C, np.float32)]).reshape(-1, C)
                s, q, e, err = G.search(blocks, gcb, mode)
                ours = S.reconstruct(s, q, e, K, C, P).reshape(-1)[:len(w)]
                t0 = time.time()
                ct = TP.compress(w, TP.SeedLMConfig(C=C, P=P, K=K), exponent_mode=mode)
                t_tp = time.time() - t0
                theirs = TP.decompress(ct).reshape(-1).astype(np.float64)
                r = dict(tensor=name, C=C, P=P, K=K, mode=mode, n=len(w), std=float(w.std()),
                         rel_err_ours=rel(w, ours), rel_err_thirdparty=rel(w, theirs),
                         seed_agree=float(np.mean(ct.hardware_seeds() == s)),
                         recon_max_abs_diff=float(np.max(np.abs(ours - theirs))),
                         floor_frac=float(np.mean(e == -8)), thirdparty_s=t_tp)
                with open(path, "a") as f:
                    f.write(json.dumps(r) + "\n")
                print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
