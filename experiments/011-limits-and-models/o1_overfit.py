"""Spec 011 line O (over-fitting probe): train the A1 correction MLP on every position of the matrices it will
compress (T1, 24 matrices, 283 M positions), for several epochs with no early stopping, then measure
  (1) the code length of those same T1 matrices (memorisation allowed, model not billed), and
  (2) the code length of the 24 T2 matrices it never saw (model billed at 16 bit per parameter over the whole model).
Model: amodel.MLP (logits = log p_chain + f(features), output layer starting at 0), sizes of A1 plus a 40 M one.
Init: 'scratch' (random) or 'a1' (the A1 model trained on 1e9 pool positions).

    PYTHONPATH=. python experiments/011-limits-and-models/o1_overfit.py <size> <scratch|a1> <epochs>
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import amodel as A  # noqa: E402
import common as K  # noqa: E402

RES = HERE / "results" / "o"
ALL_W = 557171343360
SIZES = dict(A.SIZES)
SIZES["40M"] = dict(d=32, h=3200, depth=4)
LR = {"40k": 3e-3, "160k": 3e-3, "640k": 3e-3, "2.56M": 1e-3, "10M": 1e-3, "40M": 5e-4}


def load_train(store, names, dev):
    """chain log-probabilities of every position of the training matrices on the GPU (fp16) + offsets."""
    mids, offs = [], [0]
    for n in names:
        lp = np.load(A.D / "test" / f"{n}.npy", mmap_mode="r")
        mids.append(store.ids[n])
        offs.append(offs[-1] + len(lp))
    LP = torch.empty((offs[-1], 15), dtype=torch.float16, device=dev)      # filled in place (no second copy)
    for n, a, b in zip(names, offs[:-1], offs[1:]):
        lp = np.load(A.D / "test" / f"{n}.npy", mmap_mode="r")
        for i in range(0, b - a, 1 << 22):
            LP[a + i:a + min(b - a, i + (1 << 22))] = torch.from_numpy(np.asarray(lp[i:i + (1 << 22)], dtype=np.float16)).to(dev)
    return LP, torch.tensor(mids, device=dev), torch.tensor(offs, device=dev)


def locate(g, mids, offs):
    j = torch.searchsorted(offs, g, right=True) - 1
    return mids[j], g - offs[j]


def subsample_bits(model, store, LP, mids, offs, sel):
    model.eval()
    tot = chain = 0.0
    with torch.no_grad():
        for i in range(0, len(sel), 1 << 18):
            g = sel[i:i + (1 << 18)]
            m, j = locate(g, mids, offs)
            lp = LP[g].float()
            y = A.sym_at(store, m, j)
            tot += float(A.bits_of(model, store.features(m, j), lp, y).sum())
            chain += float((-lp.gather(1, y[:, None])[:, 0] / math.log(2)).sum())
    model.train()
    return tot / len(sel), chain / len(sel)


def main(size, init, epochs, bs=16384, log=print):
    dev = torch.device("cuda")
    torch.manual_seed(0)
    RES.mkdir(parents=True, exist_ok=True)
    tag = f"o1_{size}_{init}_e{epochs}"
    store = A.Store(dev, K.T1 + K.T2)
    model = A.MLP(**SIZES[size]).to(dev)
    if init == "a1":
        model.load_state_dict(torch.load(A.D / "models" / f"a1_{size}_1000000000.pt", map_location=dev))
    npar = A.n_params(model)
    LP, mids, offs = load_train(store, K.T1, dev)
    N = int(offs[-1])
    g = torch.Generator().manual_seed(1)                                             # shuffles on the CPU (GPU memory)
    sel = torch.randperm(N, generator=g)[:4_000_000].sort().values.to(dev)           # fixed 4 M-position monitor
    lr = LR[size] if init == "scratch" else LR[size] / 3
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    steps_ep = N // bs
    total = steps_ep * epochs
    t0 = time.time()
    hist = [{"epoch": 0, "train_sub_bits": subsample_bits(model, store, LP, mids, offs, sel)[0]}]
    chain_sub = subsample_bits(model, store, LP, mids, offs, sel)[1]
    log(f"{tag}: params {npar}, positions {N}, epochs {epochs}, lr {lr}; start {hist[0]['train_sub_bits']:.5f} chain {chain_sub:.5f}", flush=True)
    s = 0
    for ep in range(1, epochs + 1):
        perm = torch.randperm(N, generator=g).to(torch.int32).to(dev)
        for k in range(steps_ep):
            f = s / total
            cur = lr * (s + 1) / max(1, total // 100) if s < total // 100 else lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * f)))
            for pg in opt.param_groups:
                pg["lr"] = cur
            gi = perm[k * bs:(k + 1) * bs].long()
            m, j = locate(gi, mids, offs)
            y = A.sym_at(store, m, j)
            loss = A.bits_of(model, store.features(m, j), LP[gi].float(), y).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            s += 1
        del perm
        tb, _ = subsample_bits(model, store, LP, mids, offs, sel)
        hist.append({"epoch": ep, "train_sub_bits": tb, "wall_s": time.time() - t0})
        log(f"  epoch {ep}: train (4M sample) {tb:.5f}  gain {chain_sub - tb:.5f}  {time.time() - t0:.0f}s", flush=True)
    path = A.D / "models" / f"{tag}.pt"
    torch.save(model.state_dict(), path)
    del LP
    torch.cuda.empty_cache()
    test = A.evaluate_test(model, store, K.T1 + K.T2, dev)
    out = {"tag": tag, "size": size, "init": init, "epochs": epochs, "params": npar, "cfg": model.cfg, "lr": lr, "bs": bs,
           "train_positions": N, "samples_seen": s * bs, "model_file": str(path), "sha256": A.sha256(path),
           "history": hist, "chain_sub_bits": chain_sub, "wall_s": time.time() - t0, "test": test}
    for split, names in (("T1_trained", K.T1), ("T2_unseen", K.T2)):
        n = sum(test[x]["n"] for x in names)
        bits = sum(test[x]["bits"] for x in names)
        chain = sum(test[x]["chain_bits_exact"] for x in names)
        out[split] = {"n": n, "bits_per_code": bits / n, "chain_bits_per_code": chain / n, "gain": (chain - bits) / n,
                      "model_bpw_over_whole_model": 16.0 * npar / ALL_W,
                      "net_gain_over_whole_model": (chain - bits) / n - 16.0 * npar / ALL_W,
                      "model_bpw_over_these_matrices": 16.0 * npar / n}
    with open(RES / "o1.jsonl", "a") as fo:
        fo.write(json.dumps(out) + "\n")
    log(f"{tag}: T1 (trained) gain {out['T1_trained']['gain']:.5f}   T2 (unseen) gain {out['T2_unseen']['gain']:.5f} "
        f"net {out['T2_unseen']['net_gain_over_whole_model']:.5f}   {out['wall_s']:.0f}s", flush=True)
    return out


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
