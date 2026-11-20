"""Spec 011 line C4: global latents (hyperprior) -- side information about whole lines of the current matrix.

    PYTHONPATH=. python experiments/011-limits-and-models/c4_hyperprior.py synth
    PYTHONPATH=. python experiments/011-limits-and-models/c4_hyperprior.py run [config ...]

Needs c3_scope.py prep (the whole-line summaries: 15 symbol frequencies + log2 RMS of each row and column).
Bounds (side information free, not causal -- what any per-line latent could at most bring):
  oracle_h  : the summary of the code's own hidden line (whole column of w1 / w3, whole row of w2)
  oracle_n  : the summary of its own neuron line
  oracle_hn : both
Coded (Balle 2018 style hyperprior, everything billed): an encoder maps a line's summary to k = 4 numbers, rounded
to integers and sent first; their code length comes from a learned factorised prior (per dimension a mixture of 3
discretised logistics); the correction MLP sees the decoded latents. Training uses uniform noise instead of
rounding; the objective is code bits + latent bits spread over the line (both in bits, no trade-off weight).
At test time the latents are rounded and their exact code length is added (each hidden line has 2,304 codes and
each neuron line 5,120 in every role, so spreading the line's bits over its codes adds up to the exact total).
  coded_h, coded_n, coded_hn, coded_m (one latent per matrix, from the whole-matrix frequencies).
Bits-back coding (Bit-Swap, HiLLoC) is not done; it could at most win back the latent bits, which are reported.
The low-rank variant is done only if C1 decided so (results/c/c1_dequant.json, c4_lowrank_decision).
Results: results/c/c4_hyperprior.jsonl, synthetic check results/c/synthetic_c4.json.
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

RES = HERE / "results" / "c"
N_REAL = 160 * 10 ** 6
KZ = 4
LINE_LEN = {"h": 2304, "n": 5120}
CONFIGS = {"base": ("none", []), "oracle_h": ("oracle", ["h"]), "oracle_n": ("oracle", ["n"]), "oracle_hn": ("oracle", ["h", "n"]),
           "coded_h": ("coded", ["h"]), "coded_n": ("coded", ["n"]), "coded_hn": ("coded", ["h", "n"]), "coded_m": ("coded", ["m"])}


def build(kind, lines, dev, line_len=None):
    import torch
    import amodel as A
    import cx

    class Prior(torch.nn.Module):
        """factorised prior over k integer latents: per dimension a mixture of 3 discretised logistics."""

        def __init__(self, k, m=3):
            super().__init__()
            self.logit = torch.nn.Parameter(torch.zeros(k, m))
            self.mu = torch.nn.Parameter(torch.linspace(-1, 1, m).repeat(k, 1))
            self.logs = torch.nn.Parameter(torch.zeros(k, m))

        def bits(self, z):                               # z: (B, k) -> (B,) bits
            w = torch.log_softmax(self.logit, -1)
            s = torch.exp(self.logs).clamp(min=1e-3)
            zz = z[..., None]
            up = torch.sigmoid((zz + 0.5 - self.mu) / s)
            lo = torch.sigmoid((zz - 0.5 - self.mu) / s)
            p = (torch.exp(w) * (up - lo).clamp(min=1e-9)).sum(-1).clamp(min=1e-9)
            return -(torch.log2(p)).sum(-1)

    class Model(cx.MLPx):
        def __init__(self):
            n_cont = 16 * len(lines) if kind == "oracle" else KZ * len(lines)
            super().__init__(n_cont=n_cont, **A.SIZES["160k"])
            self.kind, self.lines = kind, lines
            if kind == "coded":
                self.enc = torch.nn.ModuleDict({l: torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, KZ))
                                                for l in lines})
                self.prior = torch.nn.ModuleDict({l: Prior(KZ) for l in lines})
            self.rate_pp = None

        def forward(self, f, logp):
            src = {"h": "self_h", "n": "self_n", "m": "self_m"}
            if self.kind == "oracle":
                f["cont"] = torch.cat([f[src[l]] for l in self.lines], 1)
            elif self.kind == "coded":
                parts, rate = [], 0.0
                for l in self.lines:
                    z = self.enc[l](f[src[l]]) * 4.0
                    zq = z + (torch.rand_like(z) - 0.5) if self.training else torch.round(z)
                    parts.append(zq)
                    rate = rate + self.prior[l].bits(zq) / f["len_" + l]
                f["cont"] = torch.cat(parts, 1)
                self.rate_pp = rate
            return super().forward(f, logp)

    torch.manual_seed(0)
    return Model().to(dev)


def patch_bits():
    """A.bits_of(...) + the latent bits spread over the line (used by A.train / evaluate_val / evaluate_test)."""
    import amodel as A
    orig = A.bits_of
    if getattr(A, "_c4_patched", False):
        return

    def bits_of(model, f, logp, y):
        b = orig(model, f, logp, y)
        r = getattr(model, "rate_pp", None)
        return b if r is None else b + r.float()

    A.bits_of = bits_of
    A._c4_patched = True


def self_extra(store, mid, idx, r, c, f, lv=None, mfreq=None):
    """adds f['self_h'], f['self_n'] (whole-line summaries of this matrix), f['self_m'] and the line lengths."""
    import torch
    lv(store, mid, idx, r, c, f)
    f["len_h"] = torch.where(f["role"] == 1, store.g_W[mid], store.g_R[mid]).float()
    f["len_n"] = torch.where(f["role"] == 1, store.g_R[mid], store.g_W[mid]).float()
    f["len_m"] = (store.g_R[mid] * store.g_W[mid]).float()
    f["self_m"] = mfreq[mid]


def matrix_freq(store, dev):
    import torch
    import amodel as A
    out = np.zeros((len(store.reg), 16), dtype=np.float32)
    for r in store.reg:
        z = np.load(A.D / "codes" / f"{r['name']}.npz")
        cnt = np.bincount(z["sym"].ravel(), minlength=15)[:15]
        out[r["id"], :15] = cnt / cnt.sum()
    return torch.from_numpy(out).to(dev)


def attach(store, dev):
    import functools
    import c3_scope as C3
    lv = C3.Levels(store, ["self"], dev)
    store.extras = [functools.partial(self_extra, lv=lv, mfreq=matrix_freq(store, dev))]


def train_eval(cfg, store, N, dev, out_path, tests, budget=None, log=print, record_extra=None):
    import torch
    import amodel as A
    import cx
    kind, lines = CONFIGS[cfg]
    patch_bits()
    model = build(kind, lines, dev)
    rec = {"tag": "c4_" + cfg, "config": cfg, "kind": kind, "lines": lines, **(record_extra or {})}
    tmp = Path(str(out_path) + ".tmp")
    res = cx.run(model, store, N, dev, 3e-3, tmp, rec, tests, log=log, budget=budget)
    tmp.unlink()
    if kind == "coded":
        lb = latent_bits(model, store, tests)
        for split in ("T1", "T2"):
            if split in res:
                import common as K
                ns = [x for x in getattr(K, split) if x in lb]
                res[split]["latent_bits_per_code"] = sum(lb[x] for x in ns) / sum(res["test"][x]["n"] for x in ns)
        res["latent_bits"] = lb
    with open(out_path, "a") as f:
        f.write(json.dumps(res) + "\n")
    del model
    torch.cuda.empty_cache()
    return res


def latent_bits(model, store, names):
    """exact bits of the rounded latents of each matrix (all its lines), for reporting the side information."""
    import torch
    lv = store.extras[0].keywords["lv"]
    mf = store.extras[0].keywords["mfreq"]
    out = {}
    model.eval()
    with torch.no_grad():
        for name in names:
            mid = store.ids[name]
            R, W = int(store.R[mid]), int(store.W[mid])
            role = store.reg[mid]["role"]
            rows = lv.rows[int(lv.o["rows"][mid]): int(lv.o["rows"][mid]) + R * 16].view(R, 16).float()
            cols = lv.cols[int(lv.o["cols"][mid]): int(lv.o["cols"][mid]) + W * 16].view(W, 16).float()
            src = {"h": rows if role == "w2" else cols, "n": cols if role == "w2" else rows, "m": mf[mid][None, :]}
            out[name] = float(sum(model.prior[l].bits(torch.round(model.enc[l](src[l]) * 4.0)).sum() for l in model.lines))
    model.train()
    return out


def stage_run(names):
    import torch
    import common as K
    import cx
    dev = "cuda:0"
    out = RES / "c4_hyperprior.jsonl"
    done = {json.loads(l)["config"] for l in open(out)} if out.exists() else set()
    store = cx.StoreX(dev)
    attach(store, dev)
    for cfg in names:
        if cfg in done:
            continue
        print(f"== {cfg}  {time.strftime('%H:%M:%S')}", flush=True)
        train_eval(cfg, store, N_REAL, dev, out, K.T1 + K.T2, log=lambda s: print(s, flush=True))
    print("C4_DONE", flush=True)


def stage_synth():
    """V-b: every hidden dim (column, 512 x 2048, role w3) of every matrix draws one of 4 types, lambda x {0.5, 1, 1.5,
    2}, independently in each matrix -- only the matrix's own column shows it. Floor = exact code length given the
    types; a coded latent must also pay for the type (2 bit per column at most). V-a: no types.
    Checks: V-a coded_h about 0 (>= -0.005); V-b oracle_h recovers >= 80% of (chain - floor), coded_h recovers
    >= 70% of (chain - floor - 2 bit per column), and coded_h <= oracle_h."""
    sd = HERE.parents[1] / "data" / "011_synth_c4"
    os.environ["D011"] = str(sd)
    import torch
    import amodel as A
    import cx
    import c3_scope as C3
    dev = "cuda:0"
    out = {}
    R, W = 512, 2048
    for kind in ("V-a", "V-b"):
        rng = np.random.default_rng(71)
        mats, truth = [], {}
        for i in range(8):
            mult = rng.choice([0.5, 1.0, 1.5, 2.0], size=W) if kind == "V-b" else np.ones(W)
            E = rng.integers(117, 123, size=(R, W // 32)).astype(np.uint8)
            lam = (0.8 + 0.4 * (np.repeat(E.astype(np.float64), 32, axis=1) - 117)) * mult[None, :]
            C = cx.poisson_codes(rng, lam)
            name = f"L0.E{i}.w3"
            mats.append({"name": name, "layer": 0, "expert": i, "role": "w3", "C": C, "E": E,
                         "split": "pool" if i < 6 else "val" if i == 6 else "T1"})
            truth[name] = (C, lam)
        chain = cx.build_synth(sd, mats, seed=3)
        (sd / "ref").mkdir(exist_ok=True)
        json.dump({}, open(sd / "ref" / "index.json", "w"))
        C3.stage_prep(workers=8)
        A.D = sd
        store = cx.StoreX(dev)
        attach(store, dev)
        t = "L0.E7.w3"
        n = truth[t][0].size
        rec = {"theory": (chain[t] - cx.poisson_bits(*truth[t])) / n, "type_bits_per_code": (2.0 * W / n) if kind == "V-b" else 0.0}
        p = RES / f"synthetic_c4_{kind}.jsonl"
        if p.exists():
            p.unlink()
        for cfg in ("base", "oracle_h", "coded_h"):
            r = train_eval(cfg, store, 6 * R * W, dev, p, [t], budget=60_000_000, log=lambda s: None)
            rec[cfg] = (r["test"][t]["chain_bits_exact"] - r["test"][t]["bits"]) / n
        out[kind] = rec
        print(kind, rec, flush=True)
        del store
    vb = out["V-b"]
    out["checks"] = {"V-a_coded_about_0": bool(-0.005 <= out["V-a"]["coded_h"] < 0.005),
                     "V-b_oracle_recovers": bool(0.8 * vb["theory"] <= vb["oracle_h"] <= vb["theory"] + 2e-3),
                     "V-b_coded_recovers": bool(vb["coded_h"] >= 0.7 * (vb["theory"] - vb["type_bits_per_code"])),
                     "V-b_coded_le_oracle": bool(vb["coded_h"] <= vb["oracle_h"] + 1e-3)}
    out["all_pass"] = all(out["checks"].values())
    json.dump(out, open(RES / "synthetic_c4.json", "w"), indent=1)
    print(out["checks"], out["all_pass"], flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "synth":
        stage_synth()
    else:
        stage_run(sys.argv[2:] or list(CONFIGS))
