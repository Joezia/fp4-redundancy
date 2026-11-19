"""Spec 011 line A: GPU feature gathering, correction MLP (logits = log p_chain + f), training and evaluation.

Every matrix of data/011/registry.json lives on the GPU as one padded uint8 array of merged symbols (15 = pad):
8 rows of padding on top, 32 columns on the left, 16 on the right. Features of the code at (r, c), all decoded
before it in the order w3 -> w1 -> w2 (row-major inside a matrix, scales first):
  prev  : the 32 previous codes of the row (k = 1..32; k <= position-in-block means the same 32-block)
  above : the 8 codes above in the same column
  cross : the other matrix (w1: w3 at (r, c +- 16); w2: w1 at (c, r +- 16); w3: none -> pad), 33 codes
  scale : scale byte of the block and of the blocks left and right of it; mean and std of the row's scales
  pos   : position in the block (0..31), row / rows; layer / 42 and role (one-hot)
  logp  : log p_chain (15), also an input of f
Feature groups can be masked (ablation, context-length scan) -- see MASKS.
"""
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import os
D = Path(os.environ.get("D011", str(Path(__file__).resolve().parents[2] / "data" / "011")))
PAD = 15
NP, NA, NW = 32, 8, 16
TOP, LEFT, RIGHT = 8, 32, 16
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Store:
    """All matrices on the GPU, plus the per-row scale statistics."""

    def __init__(self, dev, names=None):
        self.dev = dev
        self.reg = json.load(open(D / "registry.json"))
        if names is not None:
            keep = set(names)
            need = set()
            for r in self.reg:
                if r["name"] in keep:
                    need.add(r["id"])
                    if r["cross"] is not None:
                        need.add(r["cross"])
        else:
            need = {r["id"] for r in self.reg}
        n = len(self.reg)
        self.R = np.zeros(n, dtype=np.int64)
        self.W = np.zeros(n, dtype=np.int64)
        self.off = np.zeros(n, dtype=np.int64)
        self.eoff = np.zeros(n, dtype=np.int64)
        bufs, ebufs, mean, std, roff = [], [], [], [], np.zeros(n, dtype=np.int64)
        o = eo = ro = 0
        for r in self.reg:
            i = r["id"]
            if i not in need:
                continue
            z = np.load(D / "codes" / f"{r['name']}.npz")
            s, E = z["sym"], z["E"]
            R, W = s.shape
            P = np.full((R + TOP, LEFT + W + RIGHT), PAD, dtype=np.uint8)
            P[TOP:, LEFT:LEFT + W] = s
            self.R[i], self.W[i], self.off[i], self.eoff[i], roff[i] = R, W, o, eo, ro
            bufs.append(P.ravel())
            ebufs.append(E.ravel())
            Ef = E.astype(np.float32)
            mean.append((Ef.mean(1) - 120.0) / 10.0)
            std.append(Ef.std(1) / 10.0)
            o += P.size
            eo += E.size
            ro += R
        t = lambda a, dt: torch.from_numpy(np.ascontiguousarray(a)).to(dev, dt)
        self.buf = t(np.concatenate(bufs), torch.uint8)
        self.ebuf = t(np.concatenate(ebufs), torch.uint8)
        self.smean = t(np.concatenate(mean), torch.float32)
        self.sstd = t(np.concatenate(std), torch.float32)
        self.g_R, self.g_W, self.g_off, self.g_eoff, self.g_roff = (t(a, torch.int64) for a in (self.R, self.W, self.off, self.eoff, roff))
        self.g_layer = t(np.array([r["layer"] for r in self.reg]), torch.float32)
        self.g_role = t(np.array([{"w1": 0, "w2": 1, "w3": 2}[r["role"]] for r in self.reg]), torch.int64)
        self.g_cross = t(np.array([(-1 if r["cross"] is None else r["cross"]) for r in self.reg]), torch.int64)
        self.g_ckind = t(np.array([(0 if r["cross"] is None else 1 if r["cross_kind"] == "same" else 2) for r in self.reg]), torch.int64)
        self.ids = {r["name"]: r["id"] for r in self.reg}

    def features(self, mid, idx, mask=None):
        """mid, idx: int64 tensors on the GPU (matrix id, flat position). Returns a dict of tensors."""
        W = self.g_W[mid]
        r, c = idx // W, idx % W
        Wp = W + LEFT + RIGHT
        base = self.g_off[mid] + (r + TOP) * Wp + (c + LEFT)
        k = torch.arange(1, NP + 1, device=self.dev)
        prev = self.buf[base[:, None] - k[None, :]]
        ka = torch.arange(1, NA + 1, device=self.dev)
        above = self.buf[base[:, None] - ka[None, :] * Wp[:, None]]
        cm = self.g_cross[mid]
        kind = self.g_ckind[mid]
        cmc = torch.where(cm < 0, mid, cm)
        Wx = self.g_W[cmc] + LEFT + RIGHT
        rr = torch.where(kind == 2, c, r)
        cc = torch.where(kind == 2, r, c)
        cb = self.g_off[cmc] + (rr + TOP) * Wx + (cc + LEFT)
        cb = torch.where(kind == 0, self.g_off[mid] + LEFT, cb)      # no cross matrix: a padding row
        ow = torch.arange(-NW, NW + 1, device=self.dev)
        cross = self.buf[cb[:, None] + ow[None, :]]
        cross = torch.where((kind == 0)[:, None], torch.full_like(cross, PAD), cross)
        nb = W // 32
        b = c // 32
        eb = self.g_eoff[mid] + r * nb
        sc = self.ebuf[eb + b]
        scl = torch.where(b > 0, self.ebuf[eb + (b - 1).clamp(min=0)], torch.zeros_like(sc))
        scr = torch.where(b < nb - 1, self.ebuf[eb + (b + 1).clamp(max=10 ** 9).minimum(nb - 1)], torch.zeros_like(sc))
        pos = c % 32
        row = self.g_roff[mid] + r
        num = torch.stack([r.float() / self.g_R[mid].float(), self.g_layer[mid] / 42.0, self.smean[row], self.sstd[row]], 1)
        role = torch.nn.functional.one_hot(self.g_role[mid], 3).float()
        f = {"prev": prev, "above": above, "cross": cross, "sc": torch.stack([sc, scl, scr], 1), "pos": pos,
             "num": torch.cat([num, role], 1)}
        if mask:
            apply_mask(f, mask)
        return f


MASK_GROUPS = ["inblock", "crossblock", "above", "partner", "scale", "position"]


def apply_mask(f, mask):
    """mask: set of groups to neutralise, and/or ('ctx', L) = keep only the L previous codes of the row."""
    k = torch.arange(1, NP + 1, device=f["prev"].device)[None, :]
    pos = f["pos"][:, None]
    for m in mask:
        if m == "inblock":
            f["prev"] = torch.where(k <= pos, torch.full_like(f["prev"], PAD), f["prev"])
        elif m == "crossblock":
            f["prev"] = torch.where(k > pos, torch.full_like(f["prev"], PAD), f["prev"])
        elif m == "above":
            f["above"] = torch.full_like(f["above"], PAD)
        elif m == "partner":
            f["cross"] = torch.full_like(f["cross"], PAD)
        elif m == "scale":
            f["sc"] = torch.zeros_like(f["sc"])
            f["num"][:, 2:4] = 0
        elif m == "position":
            f["pos"] = torch.zeros_like(f["pos"])
            f["num"][:, 0] = 0
        elif isinstance(m, tuple) and m[0] == "ctx":
            f["prev"] = torch.where(k > m[1], torch.full_like(f["prev"], PAD), f["prev"])
        else:
            raise ValueError(m)


class MLP(torch.nn.Module):
    def __init__(self, d=8, h=256, depth=2, K=15):
        super().__init__()
        self.cfg = {"d": d, "h": h, "depth": depth}
        self.emb = torch.nn.Embedding(PAD + 1, d)
        self.semb = torch.nn.Embedding(256, d)
        self.pemb = torch.nn.Embedding(32, 4)
        n_in = (NP + NA + 2 * NW + 1) * d + 3 * d + 4 + 7 + K
        layers = [torch.nn.Linear(n_in, h), torch.nn.ReLU()]
        for _ in range(depth - 1):
            layers += [torch.nn.Linear(h, h), torch.nn.ReLU()]
        self.body = torch.nn.Sequential(*layers)
        self.out = torch.nn.Linear(h, K)
        torch.nn.init.zeros_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)

    def forward(self, f, logp):
        toks = torch.cat([f["prev"], f["above"], f["cross"]], 1).long()
        x = torch.cat([self.emb(toks).flatten(1), self.semb(f["sc"].long()).flatten(1), self.pemb(f["pos"].long()),
                       f["num"], logp.clamp(min=-30.0)], 1)
        return logp + self.out(self.body(x)).float()


def n_params(m):
    return sum(p.numel() for p in m.parameters())


SIZES = {"40k": dict(d=4, h=96, depth=2), "160k": dict(d=8, h=192, depth=2), "640k": dict(d=8, h=448, depth=3),
         "2.56M": dict(d=16, h=896, depth=3), "10M": dict(d=16, h=1600, depth=4)}


def bits_of(model, f, logp, y):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg = model(f, logp)
    lp = torch.log_softmax(lg.float(), 1)
    return -lp.gather(1, y[:, None])[:, 0] / math.log(2)


class Train:
    """1e9 shuffled positions on disk (memory-mapped); the first N rows form the training set of size N."""

    def __init__(self, store, N):
        a = D / "train_all"
        self.mid = np.load(a / "mid.npy", mmap_mode="r")
        self.idx = np.load(a / "idx.npy", mmap_mode="r")
        self.logp = np.load(a / "logp.npy", mmap_mode="r")
        self.N = min(N, len(self.mid))
        self.store = store

    def batch(self, i, bs, dev):
        j = i % self.N
        sl = slice(j, min(j + bs, self.N))
        mid = torch.from_numpy(self.mid[sl].astype(np.int64)).to(dev, non_blocking=True)
        idx = torch.from_numpy(self.idx[sl].astype(np.int64)).to(dev, non_blocking=True)
        lp = torch.from_numpy(np.asarray(self.logp[sl])).to(dev, non_blocking=True).float()
        return mid, idx, lp


def sym_at(store, mid, idx):
    W = store.g_W[mid]
    r, c = idx // W, idx % W
    Wp = W + LEFT + RIGHT
    return store.buf[store.g_off[mid] + (r + TOP) * Wp + (c + LEFT)].long()


def load_val(store, dev):
    reg = {r["name"]: r for r in store.reg}
    mids, idxs, lps = [], [], []
    for r in store.reg:
        if r["split"] != "val":
            continue
        z = np.load(D / "val" / f"{r['name']}.npz")
        mids.append(np.full(len(z["idx"]), r["id"]))
        idxs.append(z["idx"].astype(np.int64))
        lps.append(z["logp"])
    t = lambda a, dt: torch.from_numpy(np.concatenate(a)).to(dev, dt)
    return t(mids, torch.int64), t(idxs, torch.int64), t(lps, torch.float32)


def evaluate_val(model, store, val, mask=None, bs=65536):
    mid, idx, lp = val
    model.eval()
    tot = 0.0
    chain = 0.0
    with torch.no_grad():
        for i in range(0, len(mid), bs):
            m, j, l = mid[i:i + bs], idx[i:i + bs], lp[i:i + bs]
            y = sym_at(store, m, j)
            tot += float(bits_of(model, store.features(m, j, mask), l, y).sum())
            chain += float((-l.gather(1, y[:, None])[:, 0] / math.log(2)).sum())
    model.train()
    return tot / len(mid), chain / len(mid)


def train(model, store, N, dev, lr, budget=None, bs=16384, mask=None, eval_every=20_000_000, patience=4, log=print,
          time_limit=None):
    """Adam, 1% warm-up then cosine decay to lr/20 over the budget (samples seen); early stopping on validation."""
    data = Train(store, N)
    budget = budget or max(data.N, 300_000_000)
    steps = budget // bs
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    val = load_val(store, dev)
    best = (evaluate_val(model, store, val, mask)[0], 0, {k: v.detach().clone() for k, v in model.state_dict().items()})
    hist = [(0, best[0])]
    bad = 0
    t0 = time.time()
    seen = 0
    nxt = eval_every
    for s in range(steps):
        f = s / steps
        cur = lr * (s + 1) / max(1, steps // 100) if s < steps // 100 else lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * f)))
        for g in opt.param_groups:
            g["lr"] = cur
        mid, idx, lp = data.batch(seen, bs, dev)
        y = sym_at(store, mid, idx)
        loss = bits_of(model, store.features(mid, idx, mask), lp, y).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        seen += len(mid)
        if seen >= nxt or s == steps - 1:
            nxt += eval_every
            v, vc = evaluate_val(model, store, val, mask)
            hist.append((seen, v))
            log(f"    seen {seen / 1e6:.0f}M  val {v:.5f} (chain {vc:.5f})  {seen / (time.time() - t0) / 1e6:.2f}M/s")
            if v < best[0] - 1e-5:
                best, bad = (v, seen, {k: x.detach().clone() for k, x in model.state_dict().items()}), 0
            else:
                bad += 1
                if bad >= patience:
                    log("    early stop")
                    break
            if time_limit and time.time() - t0 > time_limit:
                log("    time limit")
                break
    model.load_state_dict(best[2])
    return {"best_val": best[0], "best_at_seen": best[1], "seen": seen, "history": hist, "wall_s": time.time() - t0,
            "N": data.N, "budget": budget, "lr": lr, "bs": bs}


def evaluate_test(model, store, names, dev, mask=None, chunk=1 << 18, max_rows=None):
    """Every position of each test matrix (or the first max_rows rows); returns per-matrix bits."""
    info = {}
    for line in open(D / "build_info.jsonl"):
        r = json.loads(line)
        info[r["name"]] = r
    out = {}
    model.eval()
    with torch.no_grad():
        for name in names:
            mid_ = store.ids[name]
            lp_all = np.load(D / "test" / f"{name}.npy", mmap_mode="r")
            n = len(lp_all)
            if max_rows:
                n = min(n, int(max_rows * store.W[mid_]))
            tot = chain = 0.0
            for i in range(0, n, chunk):
                j = torch.arange(i, min(n, i + chunk), device=dev)
                m = torch.full_like(j, mid_)
                lp = torch.from_numpy(np.asarray(lp_all[i:i + len(j)])).to(dev)
                y = sym_at(store, m, j)
                tot += float(bits_of(model, store.features(m, j, mask), lp, y).sum())
                chain += float((-lp.gather(1, y[:, None])[:, 0] / math.log(2)).sum())
            out[name] = {"n": n, "bits": tot, "chain_bits_f32": chain,
                         "chain_bits_exact": info[name]["chain_code_bits"] if n == len(lp_all) else None}
    model.train()
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
