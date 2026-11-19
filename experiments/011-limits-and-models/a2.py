"""Spec 011 A2: causal Transformer over a whole row (A2a), with 2-D context (A2b), online learning (A2c).

    PYTHONPATH=. python experiments/011-limits-and-models/a2.py synth               # V-a / V-b (XOR at distances 50, 100)
    PYTHONPATH=. python experiments/011-limits-and-models/a2.py train <a|b> <1M|10M> [tokens]
    PYTHONPATH=. python experiments/011-limits-and-models/a2.py online <model tag> <offline|random>

A row of a matrix (5120 codes for w1/w3, 2304 for w2) is one sequence; no segmentation. The input at position t:
the previous code (a start symbol at t = 0), the scale byte of t's block, t's position in the block, the row's
role / layer / relative row index, log p_chain at t (15 values) and a fixed sinusoidal position code.
A2b adds, at every position, the 8 codes above in the same column and the 33 codes of the partner matrix
around it (w1: w3 at (r, c +- 16); w2: w1 at (c, r +- 16)), all decoded earlier.
Output: logits = log p_chain + head(h_t); the head starts at 0 (the model starts as the chain).
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as Fn

RES = HERE / "results" / "a"
SIZES = {"1M": dict(d=128, layers=4, heads=4), "10M": dict(d=320, layers=8, heads=8)}
LR = {"1M": 1e-3, "10M": 5e-4}


class RowModel(torch.nn.Module):
    def __init__(self, d, layers, heads, two_d=False, K=15):
        super().__init__()
        self.cfg = {"d": d, "layers": layers, "heads": heads, "two_d": two_d}
        self.code = torch.nn.Embedding(17, d)            # 0..14 symbols, 15 = pad, 16 = start
        self.scale = torch.nn.Embedding(256, d)
        self.pos = torch.nn.Embedding(32, d)
        self.rowf = torch.nn.Linear(6, d)
        self.lp = torch.nn.Linear(K, d)
        self.two_d = two_d
        if two_d:
            self.ctx_emb = torch.nn.Embedding(16, 8)
            self.ctx = torch.nn.Linear(41 * 8, d)
        self.blocks = torch.nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.norm = torch.nn.LayerNorm(d)
        self.head = torch.nn.Linear(d, K)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, prevc, sc, pos, rowf, logp, ctx=None):
        B, L = prevc.shape
        x = self.code(prevc) + self.scale(sc) + self.pos(pos) + self.rowf(rowf)[:, None, :] + self.lp(logp.clamp(min=-30))
        x = x + sinusoid(L, x.shape[-1], x.device)[None]
        if self.two_d:
            x = x + self.ctx(self.ctx_emb(ctx).flatten(2))
        for b in self.blocks:
            x = b(x)
        return logp + self.head(self.norm(x)).float()


class Block(torch.nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1, self.n2 = torch.nn.LayerNorm(d), torch.nn.LayerNorm(d)
        self.qkv = torch.nn.Linear(d, 3 * d)
        self.o = torch.nn.Linear(d, d)
        self.ff = torch.nn.Sequential(torch.nn.Linear(d, 4 * d), torch.nn.GELU(), torch.nn.Linear(4 * d, d))
        self.h = heads

    def forward(self, x):
        B, L, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, L, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = Fn.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, L, d))
        return x + self.ff(self.n2(x))


_SIN = {}


def sinusoid(L, d, dev):
    key = (L, d, str(dev))
    if key not in _SIN:
        pos = torch.arange(L, device=dev, dtype=torch.float32)[:, None]
        i = torch.arange(d // 2, device=dev, dtype=torch.float32)[None]
        ang = pos / (10000 ** (2 * i / d))
        _SIN[key] = torch.cat([ang.sin(), ang.cos()], 1)
    return _SIN[key]


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ------------------------------------------------------------------ rows from the GPU store

def row_inputs(store, mids, rows, logp, two_d):
    """mids, rows: lists (same width W); logp: (B, W, 15) float tensor on the GPU."""
    import amodel as A
    dev = store.dev
    W = int(store.W[mids[0]])
    mid = torch.tensor(mids, device=dev)[:, None].expand(-1, W).reshape(-1)
    idx = (torch.tensor(rows, device=dev)[:, None] * W + torch.arange(W, device=dev)[None]).reshape(-1)
    f = store.features(mid, idx)
    y = A.sym_at(store, mid, idx).view(len(mids), W)
    prevc = torch.cat([torch.full((len(mids), 1), 16, device=dev, dtype=torch.long), y[:, :-1]], 1)
    sc = f["sc"][:, 0].long().view(len(mids), W)
    pos = f["pos"].long().view(len(mids), W)
    num = f["num"].view(len(mids), W, -1)[:, 0]        # row fraction, layer, scale mean / std, role (3)
    rowf = torch.cat([num[:, :2], num[:, 4:7], torch.zeros_like(num[:, :1])], 1)
    ctx = None
    if two_d:
        ctx = torch.cat([f["above"], f["cross"]], 1).long().view(len(mids), W, -1)
    return prevc, sc, pos, rowf, ctx, y


def bits(model, store, mids, rows, logp, two_d):
    prevc, sc, pos, rowf, ctx, y = row_inputs(store, mids, rows, logp, two_d)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg = model(prevc, sc, pos, rowf, logp, ctx)
    lp = torch.log_softmax(lg.float(), -1)
    return -lp.gather(-1, y[..., None])[..., 0] / math.log(2), y


class Rows:
    def __init__(self, store, names, rows_dir):
        self.store = store
        self.items = []
        self.lp = {}
        for n in names:
            mid = store.ids[n]
            self.lp[mid] = np.load(rows_dir / f"{n}.npy", mmap_mode="r")
            self.items += [(mid, r) for r in range(int(store.R[mid]))]
        self.byW = {}
        for mid, r in self.items:
            self.byW.setdefault(int(store.W[mid]), []).append((mid, r))

    def logp(self, mid, r):
        W = int(self.store.W[mid])
        return np.asarray(self.lp[mid][r * W:(r + 1) * W], dtype=np.float32)


def batch_size(W, tokens=65536):
    return max(1, tokens // W)


def evaluate_rows(model, store, data, sel, two_d, dev):
    model.eval()
    tot = ch = n = 0.0
    with torch.no_grad():
        for W, lst in sel.items():
            bs = batch_size(W)
            for i in range(0, len(lst), bs):
                part = lst[i:i + bs]
                lp = torch.from_numpy(np.stack([data.logp(m, r) for m, r in part])).to(dev)
                b, y = bits(model, store, [m for m, _ in part], [r for _, r in part], lp, two_d)
                tot += float(b.sum())
                ch += float((-lp.gather(-1, y[..., None])[..., 0] / math.log(2)).sum())
                n += b.numel()
    model.train()
    return tot / n, ch / n


def train(model, store, data, val, val_sel, tokens, lr, two_d, dev, log=print, eval_every=50_000_000, patience=4,
          time_limit=None):
    rng = np.random.default_rng(0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    Ws = sorted(data.byW)
    weights = np.array([len(data.byW[W]) * W for W in Ws], dtype=np.float64)
    weights /= weights.sum()
    seen, step, nxt = 0, 0, eval_every
    best = (evaluate_rows(model, store, val, val_sel, two_d, dev)[0], 0, {k: v.clone() for k, v in model.state_dict().items()})
    hist = [(0, best[0])]
    bad = 0
    t0 = time.time()
    total_steps = tokens // 65536
    while seen < tokens:
        W = Ws[rng.choice(len(Ws), p=weights)]
        lst = data.byW[W]
        bs = batch_size(W)
        part = [lst[i] for i in rng.integers(0, len(lst), bs)]
        f = step / max(1, total_steps)
        cur = lr * min(1.0, (step + 1) / max(1, total_steps // 50)) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(f, 1.0))))
        for g in opt.param_groups:
            g["lr"] = cur
        lp = torch.from_numpy(np.stack([data.logp(m, r) for m, r in part])).to(dev)
        b, _ = bits(model, store, [m for m, _ in part], [r for _, r in part], lp, two_d)
        loss = b.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        seen += b.numel()
        step += 1
        if seen >= nxt or seen >= tokens:
            nxt += eval_every
            v, vc = evaluate_rows(model, store, val, val_sel, two_d, dev)
            hist.append((seen, v))
            log(f"    seen {seen / 1e6:.0f}M  val {v:.5f} (chain {vc:.5f})  {seen / (time.time() - t0) / 1e6:.3f}M tok/s")
            if v < best[0] - 1e-5:
                best, bad = (v, seen, {k: x.clone() for k, x in model.state_dict().items()}), 0
            else:
                bad += 1
                if bad >= patience:
                    log("    early stop")
                    break
            if time_limit and time.time() - t0 > time_limit:
                log("    time limit")
                break
    model.load_state_dict(best[2])
    return {"best_val": best[0], "best_at": best[1], "seen": seen, "history": hist, "wall_s": time.time() - t0, "lr": lr}


class TestRows:
    """log p_chain of a test matrix (float32, every position)."""

    def __init__(self, store, D, names):
        self.store = store
        self.lp = {store.ids[n]: np.load(D / "test" / f"{n}.npy", mmap_mode="r") for n in names}

    def logp(self, mid, r):
        W = int(self.store.W[mid])
        return np.asarray(self.lp[mid][r * W:(r + 1) * W])


def evaluate_test(model, store, D, names, two_d, dev):
    data = TestRows(store, D, names)
    info = {json.loads(l)["name"]: json.loads(l) for l in open(D / "build_info.jsonl")}
    out = {}
    for n in names:
        mid = store.ids[n]
        sel = {int(store.W[mid]): [(mid, r) for r in range(int(store.R[mid]))]}
        b, _ = evaluate_rows(model, store, data, sel, two_d, dev)
        N = int(store.R[mid] * store.W[mid])
        out[n] = {"n": N, "bits": b * N, "chain_bits_exact": info[n]["chain_code_bits"]}
    return out


def online(model, store, D, name, two_d, dev, lr, rows_per_step=8):
    """Predict, then update: rows in coding order, rows_per_step rows per step; only the predictions made before
    each update are counted."""
    data = TestRows(store, D, [name])
    mid = store.ids[name]
    R = int(store.R[mid])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    tot = 0.0
    model.train()
    for i in range(0, R, rows_per_step):
        rows = list(range(i, min(R, i + rows_per_step)))
        lp = torch.from_numpy(np.stack([data.logp(mid, r) for r in rows])).to(dev)
        b, _ = bits(model, store, [mid] * len(rows), rows, lp, two_d)
        tot += float(b.detach().sum())
        loss = b.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return tot


# ------------------------------------------------------------------ stages

def summarize(test, names, npar):
    n = sum(test[k]["n"] for k in names)
    b = sum(test[k]["bits"] for k in names)
    c = sum(test[k]["chain_bits_exact"] for k in names)
    return {"n": n, "bits_per_code": b / n, "chain_bits_per_code": c / n, "gain": (c - b) / n,
            "net_gain_cost_over_test": (c - b - 16.0 * npar) / n, "net_gain_cost_over_whole_model": (c - b) / n - 16.0 * npar / 557171343360}


def stage_train(variant, size, tokens):
    import amodel as A
    import common as K
    dev = "cuda:0"
    two_d = variant == "b"
    store = A.Store(dev)
    data = Rows(store, K.pool_round(1), A.D / "rows")
    val = Rows(store, K.validation(), A.D / "rows")
    rng = np.random.default_rng(5)
    val_sel = {W: [lst[i] for i in sorted(rng.choice(len(lst), min(len(lst), 600), replace=False))] for W, lst in val.byW.items()}
    torch.manual_seed(0)
    model = RowModel(**SIZES[size], two_d=two_d).to(dev)
    npar = n_params(model)
    tag = f"a2{variant}_{size}"
    print(f"== {tag}: {npar} parameters, {tokens:.2e} tokens, lr {LR[size]}  {time.strftime('%H:%M:%S')}", flush=True)
    info = train(model, store, data, val, val_sel, tokens, LR[size], two_d, dev, time_limit=6 * 3600)
    (A.D / "models").mkdir(exist_ok=True)
    path = A.D / "models" / f"{tag}.pt"
    torch.save(model.state_dict(), path)
    t0 = time.time()
    test = evaluate_test(model, store, A.D, K.T1 + K.T2, two_d, dev)
    res = {"tag": tag, "variant": variant, "size": size, "params": npar, "cfg": model.cfg, "tokens": tokens,
           "model_file": str(path.relative_to(A.D.parent.parent)), "sha256": A.sha256(path), "train": info,
           "test_wall_s": time.time() - t0, "test": test,
           "T1": summarize(test, K.T1, npar), "T2": summarize(test, K.T2, npar)}
    with open(RES / "a2.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")
    print(f"   T1 gain {res['T1']['gain']:.5f}  T2 gain {res['T2']['gain']:.5f}", flush=True)


def stage_online(tag, start, lr=None):
    import amodel as A
    import common as K
    dev = "cuda:0"
    rec = [json.loads(l) for l in open(RES / "a2.jsonl") if json.loads(l)["tag"] == tag][-1]
    two_d = rec["variant"] == "b"
    store = A.Store(dev, names=K.T1 + K.T2 + K.validation())
    info = {json.loads(l)["name"]: json.loads(l) for l in open(A.D / "build_info.jsonl")}

    def fresh():
        torch.manual_seed(0)
        m = RowModel(**SIZES[rec["size"]], two_d=two_d).to(dev)
        if start == "offline":
            m.load_state_dict(torch.load(A.D / "models" / f"{tag}.pt"))
        return m
    # online learning rate: chosen on two validation matrices only
    if lr is None:
        cand = {}
        vnames = [K.validation()[0], K.validation()[13]]
        for c in (1e-5, 3e-5, 1e-4) if start == "offline" else (1e-4, 3e-4, 1e-3):
            tot = ch = 0.0
            for vn in vnames:
                # validation matrices have float16 rows; write a temporary float32 test file view
                tot += online_val(fresh(), store, A.D, vn, two_d, dev, c)
                ch += info[vn]["chain_code_bits"] if vn in info else 0.0
            cand[c] = tot
            print(f"   online lr {c}: {tot:.0f} bits", flush=True)
        lr = min(cand, key=cand.get)
    out = {}
    path = RES / f"a2c_{tag}_{start}.jsonl"
    done = {json.loads(l)["task"] for l in open(path)} if path.exists() else set()
    for n in K.T1 + K.T2:
        if n in done:
            continue
        t0 = time.time()
        b = online(fresh(), store, A.D, n, two_d, dev, lr)
        r = {"task": n, "n": int(store.R[store.ids[n]] * store.W[store.ids[n]]), "bits": b,
             "chain_bits_exact": info[n]["chain_code_bits"], "lr": lr, "wall_s": time.time() - t0}
        with open(path, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(n, f"{(r['chain_bits_exact'] - b) / r['n']:.5f}", flush=True)


def online_val(model, store, D, name, two_d, dev, lr, rows_per_step=8):
    data = Rows(store, [name], D / "rows")
    mid = store.ids[name]
    R = int(store.R[mid])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    tot = 0.0
    for i in range(0, R, rows_per_step):
        rows = list(range(i, min(R, i + rows_per_step)))
        lp = torch.from_numpy(np.stack([data.logp(mid, r) for r in rows])).to(dev)
        b, _ = bits(model, store, [mid] * len(rows), rows, lp, two_d)
        tot += float(b.detach().sum())
        opt.zero_grad(set_to_none=True)
        b.mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return tot


if __name__ == "__main__":
    st = sys.argv[1]
    RES.mkdir(parents=True, exist_ok=True)
    if st == "train":
        stage_train(sys.argv[2], sys.argv[3], int(float(sys.argv[4])) if len(sys.argv) > 4 else 10 ** 9)
    elif st == "online":
        stage_online(sys.argv[2], sys.argv[3])
    elif st == "synth":
        import a2synth
        a2synth.main()
