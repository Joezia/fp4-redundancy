"""TX stage 3: non-linear transforms (GPU training, CPU float64 coding).

E1 normalising flow on 32-blocks (RealNVP affine couplings), trained on the training matrices by maximum
   likelihood of the dequantised block levels; coding: z = f(x), Q = round(z / q), x_hat = round(f^-1(Q q)),
   residual exact (approximation + residual).
E2 integer discrete flow (Hoogeboom et al. 2019): additive couplings with rounding (straight-through), fixed
   permutations, factorised discrete-logistic-mixture prior; exactly invertible on integers. Code length: the learned
   prior (static, model billed) and, for comparison, C0 / C1 on the output.
E3 block autoencoder: latent of 2-8 dims, uniform-noise relaxation, trained on rate (factorised latent prior) +
   residual bits (discrete logistic per dim); coding: latent ints + exact residual, with C0 / C1.
E4 coordinate network (SIREN): (1) one small network per matrix on (row, col) -> aligned value; (2) one trunk shared
   by 8 experts of a layer with a per-expert modulation vector. Approximation + residual.
All networks are billed at 16 bit per parameter: E1-E3 over the test matrices and over the whole model (one shared
network), E4 (1) over its matrix, (2) over its 8 experts.
"""
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import txcore as T
import tx_local as X

ALL_W = 557171343360
MODELS = T.DATA / "models"


def dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def nparams(net):
    return sum(p.numel() for p in net.parameters())


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def train_blocks(names, per_matrix=200000, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for n in names:
        m = T.Mat.load(n, with_partner=False)
        B = m.L.reshape(-1, 32)
        out.append(B[rng.choice(len(B), min(per_matrix, len(B)), replace=False)].astype(np.int8))
    return np.concatenate(out)


def split_val(B, frac=0.1, seed=1):
    rng = np.random.default_rng(seed)
    p = rng.permutation(len(B))
    k = int(len(B) * frac)
    return B[p[k:]], B[p[:k]]


# ------------------------------------------------------------------ E1 RealNVP

class Coupling(nn.Module):
    def __init__(self, d, mask, h):
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 2 * d))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        st = self.net(x * self.mask)
        s, t = st.chunk(2, -1)
        s = torch.tanh(s) * (1 - self.mask)
        t = t * (1 - self.mask)
        return x * torch.exp(s) + t, s.sum(-1)

    def inverse(self, y):
        st = self.net(y * self.mask)
        s, t = st.chunk(2, -1)
        s = torch.tanh(s) * (1 - self.mask)
        t = t * (1 - self.mask)
        return (y - t) * torch.exp(-s)


class Flow(nn.Module):
    def __init__(self, d=32, layers=8, h=128):
        super().__init__()
        masks = []
        for l in range(layers):
            m = torch.zeros(d)
            if l % 2 == 0:
                m[::2] = 1
            else:
                m[: d // 2] = 1
            masks.append(m if (l // 2) % 2 == 0 else 1 - m)
        self.scale = nn.Parameter(torch.zeros(d))
        self.layers = nn.ModuleList([Coupling(d, m, h) for m in masks])
        self.cfg = {"d": d, "layers": layers, "h": h}

    def forward(self, x):
        ld = self.scale.sum().expand(x.shape[0])
        x = x * torch.exp(self.scale)
        for c in self.layers:
            x, l = c(x)
            ld = ld + l
        return x, ld

    def inverse(self, z):
        for c in reversed(self.layers):
            z = c.inverse(z)
        return z * torch.exp(-self.scale)


def train_flow(Btr, Bva, layers, h, steps=6000, bs=4096, lr=1e-3, log=print):
    d = dev()
    net = Flow(32, layers, h).to(d)
    net.scale.data.fill_(-math.log(float(np.std(Btr)) + 1e-6))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    Xt = torch.tensor(Btr, dtype=torch.float32, device=d)
    Xv = torch.tensor(Bva[:200000], dtype=torch.float32, device=d)
    g = torch.Generator(device=d).manual_seed(0)

    def nll(x):
        x = x + torch.rand(x.shape, device=d, generator=g) - 0.5
        z, ld = net(x)
        lp = -0.5 * (z * z).sum(-1) - 16 * math.log(2 * math.pi) + ld
        return -lp.mean() / (32 * math.log(2))           # bits per value (dequantised density)

    for it in range(steps):
        idx = torch.randint(0, len(Xt), (bs,), device=d, generator=g)
        loss = nll(Xt[idx])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        sched.step()
        if (it + 1) % 2000 == 0:
            with torch.no_grad():
                log(f"   flow L{layers} h{h} step {it + 1} train {loss.item():.4f} val {nll(Xv).item():.4f} bit/value", flush=True)
    with torch.no_grad():
        val = float(np.mean([nll(Xv[i:i + 50000]).item() for i in range(0, len(Xv), 50000)]))
    return net.cpu().double(), val


def flow_steps(net, B, q, n=2000, seed=0):
    """per-dimension quantisation steps in z: q * RMS over blocks of |dz_d / dx| (row norm of the Jacobian), so that
    one step in z_d corresponds to about q units of x. Sent as 32 fp16 values (side)."""
    rng = np.random.default_rng(seed)
    S = torch.tensor(B[rng.choice(len(B), min(n, len(B)), replace=False)], dtype=torch.float64)
    J = torch.func.vmap(torch.func.jacrev(lambda x: net(x[None])[0][0]))(S)          # (n, 32, 32)
    g = torch.sqrt((J * J).sum(-1).mean(0)).detach().numpy()
    return np.asarray(q * g, dtype=np.float16).astype(np.float64)


def E1_flow(m, v, net):
    """approximation + residual through the flow (float64 on the CPU, deterministic); steps per z dimension."""
    B = m.L.reshape(-1, 32)
    st = flow_steps(net, B, v["q"])
    with torch.no_grad():
        z = torch.cat([net(torch.tensor(B[i:i + 200000], dtype=torch.float64))[0] for i in range(0, len(B), 200000)]).numpy()
        Q = np.round(z / st).astype(np.int64)
        xh = torch.cat([net.inverse(torch.tensor(Q[i:i + 200000] * st, dtype=torch.float64))
                        for i in range(0, len(Q), 200000)]).numpy()
    xh = np.round(xh).astype(np.int64)
    res = B - xh
    rec = (xh + res).reshape(m.L.shape)
    R = m.L.shape[0]
    lay = X.band_layout(Q.reshape(R, -1, 32))
    info = T.coding_gain(z, np.broadcast_to(np.arange(32), z.shape))
    info["residual_nonzero"] = float(np.count_nonzero(res) / res.size)
    return {"streams": [T.Stream(lay, "int", "2d", name="z"), T.Stream(res.reshape(m.L.shape), "int", "2d", name="residual")],
            "side": 16.0 * 32, "roundtrip": m.check_codes(L=rec), "info": info}


# ------------------------------------------------------------------ E2 integer discrete flow

class AddCoupling(nn.Module):
    def __init__(self, d, h, perm):
        super().__init__()
        self.register_buffer("perm", perm)
        self.register_buffer("iperm", torch.argsort(perm))
        self.net = nn.Sequential(nn.Linear(d // 2, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, d // 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def t(self, a):
        return self.net(a)

    def forward(self, x):
        x = x[:, self.perm]
        a, b = x.chunk(2, -1)
        t = self.t(a)
        rt = t + (torch.round(t) - t).detach()          # straight-through rounding
        return torch.cat([a, b + rt], -1)

    def inverse(self, y):
        a, b = y.chunk(2, -1)
        b = b - torch.round(self.t(a))
        return torch.cat([a, b], -1)[:, self.iperm]


class IDF(nn.Module):
    def __init__(self, d=32, layers=8, h=128, mix=3, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.layers = nn.ModuleList([AddCoupling(d, h, torch.randperm(d, generator=g)) for _ in range(layers)])
        self.mu = nn.Parameter(torch.randn(d, mix) * 0.5)
        self.logs = nn.Parameter(torch.zeros(d, mix) + math.log(2.0))
        self.logit = nn.Parameter(torch.zeros(d, mix))
        self.cfg = {"d": d, "layers": layers, "h": h, "mix": mix}

    def forward(self, x):
        for c in self.layers:
            x = c(x)
        return x

    def inverse(self, y):
        for c in reversed(self.layers):
            y = c.inverse(y)
        return y

    def log2p(self, y):
        """factorised discrete logistic mixture: log2 P(y_d) summed over dims."""
        y = y.unsqueeze(-1)
        s = torch.exp(self.logs)
        up = torch.sigmoid((y + 0.5 - self.mu) / s)
        lo = torch.sigmoid((y - 0.5 - self.mu) / s)
        p = (up - lo).clamp(min=1e-12)
        lw = torch.log_softmax(self.logit, -1)
        return torch.logsumexp(torch.log(p) + lw, -1).sum(-1) / math.log(2)


def train_idf(Btr, Bva, layers, h, steps=6000, bs=4096, lr=1e-3, log=print):
    d = dev()
    net = IDF(32, layers, h).to(d)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    Xt = torch.tensor(Btr, dtype=torch.float32, device=d)
    Xv = torch.tensor(Bva[:200000], dtype=torch.float32, device=d)
    g = torch.Generator(device=d).manual_seed(0)
    for it in range(steps):
        idx = torch.randint(0, len(Xt), (bs,), device=d, generator=g)
        loss = -net.log2p(net(Xt[idx])).mean() / 32
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        sched.step()
        if (it + 1) % 2000 == 0:
            with torch.no_grad():
                vl = -net.log2p(net(Xv)).mean().item() / 32
            log(f"   idf L{layers} h{h} step {it + 1} train {loss.item():.4f} val {vl:.4f} bit/value", flush=True)
    with torch.no_grad():
        val = -net.log2p(net(Xv)).mean().item() / 32
    return net.cpu().double(), val


def E2_idf(m, v, net):
    B = m.L.reshape(-1, 32)
    with torch.no_grad():
        ys, bits = [], 0.0
        for i in range(0, len(B), 200000):
            y = net(torch.tensor(B[i:i + 200000], dtype=torch.float64))
            bits += -net.log2p(y).sum().item()
            ys.append(y)
        Y = torch.cat(ys)
        rec = torch.cat([net.inverse(Y[i:i + 200000]) for i in range(0, len(Y), 200000)]).numpy()
    Yi = np.round(Y.numpy()).astype(np.int64)
    assert np.abs(Y.numpy() - Yi).max() < 1e-6
    ok = m.check_codes(L=np.round(rec).astype(np.int64).reshape(m.L.shape))
    R = m.L.shape[0]
    lay = X.band_layout(Yi.reshape(R, -1, 32))
    info = T.coding_gain(Yi, np.broadcast_to(np.arange(32), Yi.shape))
    info["prior_bits"] = bits
    return {"streams": [T.Stream(lay, "int", "2d", name="y")], "roundtrip": ok, "info": info}


# ------------------------------------------------------------------ E3 block autoencoder

class DLogistic(nn.Module):
    """per-dimension discrete logistic (location 0 for the residual, learnable for latents)."""

    def __init__(self, d, loc=True):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(d)) if loc else None
        self.logs = nn.Parameter(torch.zeros(d))

    def bits(self, y):
        mu = self.mu if self.mu is not None else 0.0
        s = torch.exp(self.logs)
        p = torch.sigmoid((y + 0.5 - mu) / s) - torch.sigmoid((y - 0.5 - mu) / s)
        return -torch.log2(p.clamp(min=1e-12)).sum(-1)


class AE(nn.Module):
    def __init__(self, k, h=256):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(32, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, k))
        self.dec = nn.Sequential(nn.Linear(k, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 32))
        self.zp = DLogistic(k)
        self.rp = DLogistic(32, loc=False)
        self.cfg = {"k": k, "h": h}

    def loss(self, x, g):
        z = self.enc(x)
        zt = z + torch.rand(z.shape, device=z.device, generator=g) - 0.5
        xh = self.dec(zt)
        r = x - xh
        return (self.zp.bits(zt) + self.rp.bits(r)).mean() / 32


def train_ae(Btr, Bva, k, h, steps=6000, bs=4096, lr=1e-3, log=print):
    d = dev()
    net = AE(k, h).to(d)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    Xt = torch.tensor(Btr, dtype=torch.float32, device=d)
    Xv = torch.tensor(Bva[:200000], dtype=torch.float32, device=d)
    g = torch.Generator(device=d).manual_seed(0)
    for it in range(steps):
        idx = torch.randint(0, len(Xt), (bs,), device=d, generator=g)
        loss = net.loss(Xt[idx], g)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (it + 1) % 2000 == 0:
            with torch.no_grad():
                log(f"   ae k{k} h{h} step {it + 1} train {loss.item():.4f} val {net.loss(Xv, g).item():.4f} bit/value", flush=True)
    with torch.no_grad():
        val = net.loss(Xv, g).item()
    return net.cpu().double(), val


def E3_ae(m, v, net):
    B = m.L.reshape(-1, 32)
    with torch.no_grad():
        Z = torch.cat([net.enc(torch.tensor(B[i:i + 200000], dtype=torch.float64)) for i in range(0, len(B), 200000)])
        Q = torch.round(Z)
        xh = torch.cat([net.dec(Q[i:i + 200000]) for i in range(0, len(Q), 200000)]).numpy()
    Q = Q.numpy().astype(np.int64)
    xh = np.round(xh).astype(np.int64)
    res = B - xh
    rec = (xh + res).reshape(m.L.shape)
    R = m.L.shape[0]
    k = Q.shape[1]
    return {"streams": [T.Stream(X.band_layout(Q.reshape(R, -1, k)), "int", "2d", name="latent"),
                        T.Stream(res.reshape(m.L.shape), "int", "2d", name="residual")],
            "roundtrip": m.check_codes(L=rec), "info": {"residual_nonzero": float(np.count_nonzero(res) / res.size)}}


# ------------------------------------------------------------------ E4 coordinate networks (SIREN)

class Siren(nn.Module):
    def __init__(self, h=64, depth=3, w0=30.0, nmod=0, nexp=1):
        super().__init__()
        self.w0 = w0
        dims = [2] + [h] * depth
        self.lin = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.out = nn.Linear(h, 1)
        with torch.no_grad():
            self.lin[0].weight.uniform_(-1 / 2, 1 / 2)
            for l in self.lin[1:]:
                l.weight.uniform_(-math.sqrt(6 / h) / w0, math.sqrt(6 / h) / w0)
        self.mod = nn.Parameter(torch.zeros(nexp, depth, h)) if nmod else None      # per-expert shift (FiLM)
        self.cfg = {"h": h, "depth": depth, "w0": w0, "nmod": nmod, "nexp": nexp}

    def forward(self, rc, e=None):
        x = rc
        for i, l in enumerate(self.lin):
            y = l(x)
            if self.mod is not None:
                y = y + self.mod[e, i]
            x = torch.sin(self.w0 * y) if i == 0 else torch.sin(self.w0 * y)
        return self.out(x).squeeze(-1)


def coords(R, W, d):
    r = torch.arange(R, device=d, dtype=torch.float32) / (R - 1) * 2 - 1
    c = torch.arange(W, device=d, dtype=torch.float32) / (W - 1) * 2 - 1
    return r, c


def fit_siren(Xs, h, steps, lr=1e-4, bs=65536, log=print, shared=False):
    """Xs: list of aligned-value matrices (same shape); shared = one trunk + per-expert modulation."""
    d = dev()
    n = len(Xs)
    net = Siren(h, nmod=1 if shared else 0, nexp=n).to(d)
    R, W = Xs[0].shape
    r, c = coords(R, W, d)
    Y = torch.tensor(np.stack(Xs), dtype=torch.float32, device=d)
    sd = Y.std().item() + 1e-6
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    g = torch.Generator(device=d).manual_seed(0)
    for it in range(steps):
        e = torch.randint(0, n, (bs,), device=d, generator=g)
        i = torch.randint(0, R, (bs,), device=d, generator=g)
        j = torch.randint(0, W, (bs,), device=d, generator=g)
        pred = net(torch.stack([r[i], c[j]], -1), e if shared else None)
        loss = F.mse_loss(pred, Y[e, i, j] / sd)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (it + 1) % 1000 == 0:
            log(f"   siren h{h} n{n} step {it + 1} mse/var {loss.item():.4f}", flush=True)
    return net.cpu().double(), sd


def siren_predict(net, sd, R, W, e=None):
    r, c = coords(R, W, torch.device("cpu"))
    out = np.empty((R, W))
    with torch.no_grad():
        for a in range(0, R, 256):
            rr = r[a:a + 256].double()[:, None].expand(-1, W)
            cc = c.double()[None, :].expand(len(rr), -1)
            rc = torch.stack([rr.reshape(-1), cc.reshape(-1)], -1)
            ee = None if e is None else torch.full((len(rc),), e, dtype=torch.long)
            out[a:a + 256] = (net(rc, ee) * sd).numpy().reshape(len(rr), W)
    return out


def E4_code(m, pred, side):
    xh = np.round(pred).astype(np.int64)
    res = m.X - xh
    rec = xh + res
    mse = float(((m.X - pred) ** 2).mean() / max(m.X.astype(np.float64).var(), 1e-30))
    return {"streams": [T.Stream(res, "int", "2d", name="residual")], "side": side, "roundtrip": m.check_codes(X=rec),
            "info": {"relative_mse": mse}}


# ------------------------------------------------------------------ E1b iterative Gaussianisation (RBIG, Laparra et al. 2011)

class RBIG:
    """K iterations of (orthogonal PCA rotation, per-dimension marginal Gaussianisation by a monotone piecewise-linear
    map through 257 empirical quantiles of the dequantised data). Float64, deterministic; parameters billed at 16 bit
    (rotations 32 x 32, two 257-knot tables per dimension)."""

    def __init__(self, K):
        self.K = K
        self.R, self.kx, self.kz = [], [], []

    def fit(self, B, seed=0):
        from scipy.stats import norm
        rng = np.random.default_rng(seed)
        x = B.astype(np.float64) + rng.random(B.shape) - 0.5
        p = (np.arange(257) + 0.5) / 257
        zq = norm.ppf(p)
        for _ in range(self.K):
            ev, V = np.linalg.eigh(np.cov(x.T))
            R = V[:, ::-1].copy()
            y = x @ R
            kx = np.quantile(y, p, axis=0).T                  # (32, 257)
            for d in range(32):                               # strictly increasing knots
                kx[d] = np.maximum.accumulate(kx[d] + np.arange(257) * 1e-9)
            self.R.append(R)
            self.kx.append(kx)
            self.kz.append(np.tile(zq, (32, 1)))
            x = self._marg(y, kx, self.kz[-1])
        return self

    @staticmethod
    def _marg(y, kx, kz):
        out = np.empty_like(y)
        for d in range(y.shape[1]):
            a, b = kx[d], kz[d]
            o = np.interp(y[:, d], a, b)
            lo, hi = y[:, d] < a[0], y[:, d] > a[-1]
            sl0 = (b[1] - b[0]) / (a[1] - a[0])
            sl1 = (b[-1] - b[-2]) / (a[-1] - a[-2])
            o[lo] = b[0] + (y[lo, d] - a[0]) * sl0
            o[hi] = b[-1] + (y[hi, d] - a[-1]) * sl1
            out[:, d] = o
        return out

    def forward(self, x):
        x = x.astype(np.float64)
        for R, kx, kz in zip(self.R, self.kx, self.kz):
            x = self._marg(x @ R, kx, kz)
        return x

    def inverse(self, z):
        for R, kx, kz in zip(self.R[::-1], self.kx[::-1], self.kz[::-1]):
            z = self._marg(z, kz, kx) @ R.T
        return z

    def jac_rownorm(self, x):
        """RMS over samples of the row norms of dz/dx (numerical, central differences)."""
        eps = 1e-3
        J = np.zeros((len(x), 32, 32))
        for j in range(32):
            e = np.zeros(32)
            e[j] = eps
            J[:, :, j] = (self.forward(x + e) - self.forward(x - e)) / (2 * eps)
        return np.sqrt((J * J).sum(-1).mean(0))

    def nparams(self):
        return self.K * (32 * 32 + 2 * 32 * 257)

    def quantised(self):
        """parameters rounded to fp16 (what the decoder receives); knots kept increasing."""
        q = RBIG(self.K)
        q.R = [np.asarray(R, np.float16).astype(np.float64) for R in self.R]
        q.kx = [np.maximum.accumulate(np.asarray(k, np.float16).astype(np.float64) + np.arange(257) * 1e-6) for k in self.kx]
        q.kz = [np.maximum.accumulate(np.asarray(k, np.float16).astype(np.float64) + np.arange(257) * 1e-6) for k in self.kz]
        return q


def E1_rbig(m, v, g):
    """approximation + residual through RBIG; z steps q * row norm of the Jacobian (as E1_flow)."""
    B = m.L.reshape(-1, 32)
    rng = np.random.default_rng(0)
    st = np.asarray(v["q"] * g.jac_rownorm(B[rng.choice(len(B), 2000, replace=False)].astype(np.float64)),
                    np.float16).astype(np.float64)
    z = np.concatenate([g.forward(B[i:i + 200000]) for i in range(0, len(B), 200000)])
    Q = np.round(z / st).astype(np.int64)
    xh = np.round(np.concatenate([g.inverse(Q[i:i + 200000] * st) for i in range(0, len(Q), 200000)])).astype(np.int64)
    res = B - xh
    rec = (xh + res).reshape(m.L.shape)
    R = m.L.shape[0]
    info = T.coding_gain(z, np.broadcast_to(np.arange(32), z.shape))
    info["residual_nonzero"] = float(np.count_nonzero(res) / res.size)
    return {"streams": [T.Stream(X.band_layout(Q.reshape(R, -1, 32)), "int", "2d", name="z"),
                        T.Stream(res.reshape(m.L.shape), "int", "2d", name="residual")],
            "side": 16.0 * 32, "roundtrip": m.check_codes(L=rec), "info": info}
