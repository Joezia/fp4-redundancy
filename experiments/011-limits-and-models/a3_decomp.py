"""Spec 011 A3: where does the MLP's gain sit? Split the code length of every code into three factors --
zero or not; sign (if non-zero); magnitude (given non-zero and sign) -- for the chain and for the best A1 MLP,
and report the gain per factor and per position in the block. Model: data/011/models/a1_160k_1000000000.pt.

    PYTHONPATH=. python experiments/011-limits-and-models/a3_decomp.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import amodel as A  # noqa: E402
import common as K  # noqa: E402


def factors(lp, y):
    """lp: (n, 15) natural-log probabilities; y: symbols. returns bits of (zero?, sign, magnitude)."""
    p = lp.exp()
    pz = p[:, 0]
    ppos = p[:, 1:8].sum(1)
    pneg = p[:, 8:15].sum(1)
    nz = y > 0
    b_zero = torch.where(nz, -torch.log2(1 - pz), -torch.log2(pz))
    neg = y >= 8
    b_sign = torch.where(nz, torch.where(neg, -torch.log2(pneg / (ppos + pneg)), -torch.log2(ppos / (ppos + pneg))), torch.zeros_like(pz))
    py = p.gather(1, y[:, None])[:, 0]
    b_mag = torch.where(nz, -torch.log2(py / torch.where(neg, pneg, ppos)), torch.zeros_like(pz))
    return b_zero, b_sign, b_mag


def main():
    dev = "cuda:0"
    names = K.T1
    store = A.Store(dev, names=names)
    model = A.MLP(**A.SIZES["160k"]).to(dev)
    model.load_state_dict(torch.load(A.D / "models" / "a1_160k_1000000000.pt"))
    model.eval()
    tot = {k: 0.0 for k in ("chain_zero", "chain_sign", "chain_mag", "mlp_zero", "mlp_sign", "mlp_mag")}
    bypos = np.zeros((32, 2))
    n_all = 0
    with torch.no_grad():
        for name in names:
            mid = store.ids[name]
            lp_all = np.load(A.D / "test" / f"{name}.npy", mmap_mode="r")
            n = len(lp_all)
            for i in range(0, n, 1 << 18):
                j = torch.arange(i, min(n, i + (1 << 18)), device=dev)
                m = torch.full_like(j, mid)
                lp = torch.from_numpy(np.array(lp_all[i:i + len(j)])).to(dev)
                y = A.sym_at(store, m, j)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = model(store.features(m, j), lp)
                lm = torch.log_softmax(lg.float(), 1)
                for tag, L in (("chain", lp), ("mlp", lm)):
                    z, s, g = factors(L, y)
                    tot[tag + "_zero"] += float(z.sum())
                    tot[tag + "_sign"] += float(s.sum())
                    tot[tag + "_mag"] += float(g.sum())
                bc = -lp.gather(1, y[:, None])[:, 0] / math.log(2)
                bm = -lm.gather(1, y[:, None])[:, 0] / math.log(2)
                pos = (j % int(store.W[mid])) % 32
                bypos[:, 0] += torch.bincount(pos, weights=bc - bm, minlength=32).cpu().numpy()
                bypos[:, 1] += torch.bincount(pos, minlength=32).cpu().numpy()
            n_all += n
    out = {"matrices": names, "n": n_all,
           "gain_bits_per_weight": {f: (tot["chain_" + f] - tot["mlp_" + f]) / n_all for f in ("zero", "sign", "mag")},
           "chain_bits_per_weight": {f: tot["chain_" + f] / n_all for f in ("zero", "sign", "mag")},
           "gain_by_position_in_block": (bypos[:, 0] / bypos[:, 1]).tolist()}
    json.dump(out, open(HERE / "results" / "a" / "a3_decomp.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "matrices"}, indent=1))


if __name__ == "__main__":
    main()
