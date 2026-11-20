"""Spec 011 line C3: widen what the model can see, level by level -- matrix, expert, layer, whole model.

    PYTHONPATH=. python experiments/011-limits-and-models/c3_scope.py prep        # CPU: summaries -> data/011/c3/
    PYTHONPATH=. python experiments/011-limits-and-models/c3_scope.py causal      # M-level features never see undecoded codes
    PYTHONPATH=. python experiments/011-limits-and-models/c3_scope.py synth
    PYTHONPATH=. python experiments/011-limits-and-models/c3_scope.py run [config ...]

A "summary" of a set of codes = the 15 symbol frequencies (+ for decoded lines of other matrices, the log2 RMS of
the dequantised values). Summaries ignore the order of rows / neurons, which is arbitrary in an expert.
Axes: hidden dim h = column of w1 / w3, row of w2; neuron n = row of w1 / w3, column of w2.
Levels (all decoded before the code: scales first, w3 -> w1 -> w2 inside an expert; the reference experts of
c_ref.py and the other layers are taken as decoded earlier -- this crosses experts and layers, against the
deployment rule of coding each target on its own, so L and G are measurements of information, not a codec):
  base : the A1 / A3 features (160k MLP)
  M    : this matrix, causally: the current row up to the last multiple of 64 columns before the code; the current
         column over the rows before the last multiple of 64 rows; all rows before that multiple of 64.
  E    : the same expert's decoded matrices, whole lines: w1 sees w3's neuron line n and hidden line h; w2 sees
         the neuron line n and hidden line h of both w1 and w3; w3 has nothing at this level.
  L    : the layer's 8 reference experts, hidden line h, each role.
  G    : every other layer's reference experts, hidden line h, each role; and the two adjacent layers.
Training: the A3 protocol (160k MLP, first 1.6e8 shuffled positions, 3e8 samples, lr 3e-3, seed 0), each
configuration from scratch. Results: results/c/c3_scope.jsonl, results/c/synthetic_c3.json.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

RES = HERE / "results" / "c"
VAL = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)
SNAP = 64
N_REAL = 160 * 10 ** 6
CONFIGS = {"base": [], "M": ["M"], "M+E": ["M", "E"], "M+E+L": ["M", "E", "L"], "M+E+L+G": ["M", "E", "L", "G"],
           "L": ["L"], "G": ["G"], "E": ["E"]}
WIDTH = {"M": 15 + 1 + 15 + 1 + 15, "E": 4 * 16 + 4, "L": 3 * 16, "G": 6 * 16}


def Dd():
    return Path(os.environ.get("D011", str(HERE.parents[1] / "data" / "011")))


# ------------------------------------------------------------------ CPU: summaries

def onehot_counts(sym, axis):
    """counts of the 15 symbols along `axis` -> array with a trailing 15."""
    out = np.zeros(sym.shape[1 - axis] if sym.ndim == 2 else (), dtype=np.int64)
    return np.stack([(sym == k).sum(axis) for k in range(15)], -1)


def m_arrays(sym):
    """causal snapshots: RP (R, W/64, 15) row prefix up to 64 j columns; CP (R/64, W, 15) column prefix over the first
    64 i rows; MP (R/64, 15) all of the first 64 i rows. Frequencies x 255 as uint8 (counts are known: 64 j, 64 i, 64 i W)."""
    R, W = sym.shape
    oh = np.zeros((R, W, 15), dtype=np.uint16)
    for k in range(15):
        oh[..., k] = sym == k
    rc = np.concatenate([np.zeros((R, 1, 15), np.int64), np.cumsum(oh.reshape(R, W // SNAP, SNAP, 15).sum(2), 1)[:, :-1]], 1)
    RP = np.round(255 * rc / np.maximum(1, SNAP * np.arange(W // SNAP))[None, :, None]).astype(np.uint8)
    cc = np.concatenate([np.zeros((1, W, 15), np.int64), np.cumsum(oh.reshape(R // SNAP, SNAP, W, 15).sum(1), 0)[:-1]], 0)
    CP = np.round(255 * cc / np.maximum(1, SNAP * np.arange(R // SNAP))[:, None, None]).astype(np.uint8)
    mc = cc.sum(1)
    MP = np.round(255 * mc / np.maximum(1, SNAP * W * np.arange(R // SNAP))[:, None]).astype(np.uint8)
    return RP, CP, MP


def line_summaries(sym, E):
    """whole rows and whole columns: (R, 16), (W, 16) float16 = 15 frequencies, scaled log2 RMS of dequantised values."""
    X = VAL[sym] * np.repeat(np.exp2(E.astype(np.float32) - 127), 32, axis=1)
    out = []
    for axis in (1, 0):
        cnt = onehot_counts(sym, axis).astype(np.float64)
        fr = cnt / cnt.sum(-1, keepdims=True)
        rms = np.sqrt((X.astype(np.float64) ** 2).mean(axis))
        lr = (np.log2(np.maximum(rms, 1e-30)) + 7.0) / 4.0
        out.append(np.concatenate([fr, lr[:, None]], 1).astype(np.float16))
    return out


def hidden_line_stats(path, role):
    """for a reference matrix: per hidden dim, symbol counts and sum of squared dequantised values."""
    z = np.load(path)
    sym, E = z["sym"], z["E"]
    X2 = (VAL[sym] * np.repeat(np.exp2(E.astype(np.float32) - 127), 32, axis=1)).astype(np.float64) ** 2
    ax = 1 if role == "w2" else 0                  # sum over neurons
    return onehot_counts(sym, ax), X2.sum(ax), sym.shape[ax]


def _ref_one(args):
    L, e, r, D = args
    c, s2, n = hidden_line_stats(D / "ref" / f"L{L}.E{e}.{r}.npz", r)
    return L, r, c, s2, n


def _reg_one(args):
    name, D = args
    z = np.load(D / "codes" / f"{name}.npz")
    sym, E = z["sym"], z["E"]
    RP, CP, MP = m_arrays(sym)
    rows, cols = line_summaries(sym, E)
    return name, RP, CP, MP, rows, cols


def summarize_lines(cnt, s2, n):
    fr = cnt / np.maximum(1, cnt.sum(-1, keepdims=True))
    rms = np.sqrt(s2 / np.maximum(1, n))
    lr = (np.log2(np.maximum(rms, 1e-30)) + 7.0) / 4.0
    return np.concatenate([fr, lr[..., None]], -1).astype(np.float16)


def stage_prep(workers=32):
    D = Dd()
    out = D / "c3"
    out.mkdir(exist_ok=True)
    reg = json.load(open(D / "registry.json"))
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        res = {r[0]: r[1:] for r in ex.map(_reg_one, [(r["name"], D) for r in reg])}
    print(f"registry summaries {time.time() - t0:.0f}s", flush=True)
    offs = {}
    for k, key in enumerate(("RP", "CP", "MP", "rows", "cols")):
        o, parts = 0, []
        for r in reg:
            a = res[r["name"]][k]
            offs.setdefault(r["name"], {})[key] = o
            parts.append(a.reshape(-1))
            o += a.size
        np.save(out / f"{key}.npy", np.concatenate(parts))
    json.dump(offs, open(out / "offsets.json", "w"))
    # layer / model level from the reference experts
    idx = json.load(open(D / "ref" / "index.json"))
    nL = 43
    cnt = np.zeros((nL, 3, 5120, 15))
    s2 = np.zeros((nL, 3, 5120))
    nn = np.zeros((nL, 3))
    roles = ("w1", "w2", "w3")
    jobs = [(int(L), e, r, D) for L, es in idx.items() for e in es for r in roles]
    with ProcessPoolExecutor(workers) as ex:
        for L, r, c, q, n in ex.map(_ref_one, jobs):
            j = roles.index(r)
            H = c.shape[0]
            cnt[L, j, :H] += c
            s2[L, j, :H] += q
            nn[L, j] += n
    Ls = summarize_lines(cnt, s2, nn[:, :, None])
    present = np.array([str(L) in idx for L in range(nL)])
    Gall = np.zeros_like(Ls)
    Gadj = np.zeros_like(Ls)
    for L in range(nL):
        others = [k for k in range(nL) if k != L and present[k]]
        adj = [k for k in (L - 1, L + 1) if 0 <= k < nL and present[k]]
        if others:
            Gall[L] = summarize_lines(cnt[others].sum(0), s2[others].sum(0), nn[others].sum(0)[:, None])
        if adj:
            Gadj[L] = summarize_lines(cnt[adj].sum(0), s2[adj].sum(0), nn[adj].sum(0)[:, None])
    np.savez(out / "LG.npz", L=Ls, Gall=Gall, Gadj=Gadj)
    print("C3_PREP_DONE", f"{time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ GPU: features

class Levels:
    """Loads the summaries to the device and builds f['cont'] for the requested levels."""

    def __init__(self, store, levels, dev):
        import torch
        D = Dd()
        c3 = D / "c3"
        self.levels = levels
        reg = store.reg
        offs = json.load(open(c3 / "offsets.json"))
        t = lambda a, dt: torch.from_numpy(np.ascontiguousarray(a)).to(dev, dt)
        n = len(reg)
        g = lambda key: t(np.array([offs[r["name"]][key] for r in reg]), torch.int64)
        self.o = {k: g(k) for k in ("RP", "CP", "MP", "rows", "cols")}
        if "M" in levels:
            self.RP, self.CP, self.MP = (t(np.load(c3 / f"{k}.npy"), torch.uint8) for k in ("RP", "CP", "MP"))
        if "E" in levels or "self" in levels:
            self.rows = t(np.load(c3 / "rows.npy"), torch.float16)
            self.cols = t(np.load(c3 / "cols.npy"), torch.float16)
        byname = {r["name"]: r["id"] for r in reg}
        sib = np.full((n, 2), -1, dtype=np.int64)          # (w3 sibling, w1 sibling) visible to this matrix
        for r in reg:
            L_, e_, role = r["name"].split(".")
            pre = f"{L_}.{e_}."
            if role in ("w1", "w2"):
                sib[r["id"], 0] = byname.get(pre + "w3", -1)
            if role == "w2":
                sib[r["id"], 1] = byname.get(pre + "w1", -1)
        self.sib = t(sib, torch.int64)
        if "L" in levels or "G" in levels:
            z = np.load(c3 / "LG.npz")
            self.LG = {k: t(z[k], torch.float16) for k in ("L", "Gall", "Gadj")}
        self.dev = dev

    def __call__(self, store, mid, idx, r, c, f):
        import torch
        W = store.g_W[mid]
        R = store.g_R[mid]
        out = []
        if "M" in self.levels:
            j = c // SNAP
            i = r // SNAP
            nj = W // SNAP
            rp = self.RP[(self.o["RP"][mid] + (r * nj + j) * 15)[:, None] + torch.arange(15, device=self.dev)]
            cp = self.CP[(self.o["CP"][mid] + (i * W + c) * 15)[:, None] + torch.arange(15, device=self.dev)]
            mp = self.MP[(self.o["MP"][mid] + i * 15)[:, None] + torch.arange(15, device=self.dev)]
            out += [rp.float() / 255, torch.log1p((SNAP * j).float())[:, None] / 10, cp.float() / 255,
                    torch.log1p((SNAP * i).float())[:, None] / 10, mp.float() / 255]
        if "E" in self.levels or "self" in self.levels:
            ar = torch.arange(16, device=self.dev)

            def line(m, which, k):
                """summary of row k (which = 0) or column k (which = 1) of matrix m (m >= 0)."""
                mm = m.clamp(min=0)
                if which == 0:
                    v = self.rows[(self.o["rows"][mm] + k * 16)[:, None] + ar]
                else:
                    v = self.cols[(self.o["cols"][mm] + k * 16)[:, None] + ar]
                return torch.where((m >= 0)[:, None], v.float(), torch.zeros_like(v, dtype=torch.float32))

            if "E" in self.levels:
                n_, h_ = f["n"], f["h"]
                for s in (0, 1):                               # w3 sibling, w1 sibling (both are w1 / w3-shaped)
                    m = self.sib[mid, s]
                    out += [line(m, 0, n_.clamp(max=2303)), line(m, 1, h_.clamp(max=5119)), (m >= 0).float()[:, None],
                            (m >= 0).float()[:, None]]
            if "self" in self.levels:                         # C4 oracle: this matrix's own whole lines (not causal)
                w2 = f["role"] == 1
                hl = torch.where(w2[:, None], line(mid, 0, r), line(mid, 1, c))
                nl = torch.where(w2[:, None], line(mid, 1, c), line(mid, 0, r))
                f["self_h"], f["self_n"] = hl, nl
                if "self_h" in self.levels:
                    out.append(hl)
                if "self_n" in self.levels:
                    out.append(nl)
        if "L" in self.levels:
            out.append(self.LG["L"][f["L"], :, f["h"]].float().flatten(1))
        if "G" in self.levels:
            out += [self.LG["Gall"][f["L"], :, f["h"]].float().flatten(1), self.LG["Gadj"][f["L"], :, f["h"]].float().flatten(1)]
        if out:
            f["cont"] = torch.cat(out, 1)


def width(levels):
    return sum(WIDTH[k] for k in levels if k in WIDTH)


# ------------------------------------------------------------------ checks

def stage_causal(tests=40):
    """M-level features at (r, c) recomputed after every code at or after (r, c) in the row-major order is replaced
    by random codes must be unchanged."""
    import common as K
    rng = np.random.default_rng(7)
    D = Dd()
    report = []
    for name in (K.T1[0], K.T1[1]):
        sym = np.load(D / "codes" / f"{name}.npz")["sym"]
        R, W = sym.shape
        RP, CP, MP = m_arrays(sym)
        for _ in range(tests // 2):
            r, c = int(rng.integers(R)), int(rng.integers(W))
            s2 = sym.copy().reshape(-1)
            t = r * W + c
            s2[t:] = rng.integers(0, 15, size=s2.size - t)
            RP2, CP2, MP2 = m_arrays(s2.reshape(R, W))
            i, j = r // SNAP, c // SNAP
            same = bool((RP[r, j] == RP2[r, j]).all() and (CP[i, c] == CP2[i, c]).all() and (MP[i] == MP2[i]).all())
            report.append({"matrix": name, "r": r, "c": c, "unchanged": same})
    ok = all(x["unchanged"] for x in report)
    RES.mkdir(parents=True, exist_ok=True)
    json.dump({"pass": ok, "tests": report}, open(RES / "c3_causal_check.json", "w"), indent=1)
    print("causal pass:", ok, len(report))


def make_model(levels, dev, extra_embs=None, seed=0):
    import torch
    import amodel as A
    import cx
    torch.manual_seed(seed)
    return cx.MLPx(embs=extra_embs, n_cont=width(levels), **A.SIZES["160k"]).to(dev)


def stage_run(names):
    import torch
    import common as K
    import cx
    dev = "cuda:0"
    out = RES / "c3_scope.jsonl"
    done = {json.loads(l)["config"] for l in open(out)} if out.exists() else set()
    store = cx.StoreX(dev)
    for cfg in names:
        if cfg in done:
            continue
        base_cfg, _, sd = cfg.partition("@s")          # "<config>@s<seed>": the same with another initialisation seed
        lv = CONFIGS[base_cfg]
        seed = int(sd) if sd else 0
        store.extras = [Levels(store, lv, dev)] if lv else []
        print(f"== {cfg}  {time.strftime('%H:%M:%S')}", flush=True)
        model = make_model(lv, dev, seed=seed)
        cx.run(model, store, N_REAL, dev, 3e-3, out, {"tag": "c3_" + cfg.replace("+", "_").replace("@", "_"), "config": cfg,
                                                      "levels": lv, "seed": seed},
               K.T1 + K.T2, log=lambda s: print(s, flush=True))
        del model
        store.extras = []
        torch.cuda.empty_cache()
    print("C3_DONE", flush=True)


def stage_synth(kinds=("V-a", "V-b-M", "V-b-E", "V-b-L"), tag=""):
    """Three structures, each visible only from one level (magnitudes Poisson(lambda), lambda from the block scale
    times a boost of 2 on some lines; the exact code length of the generator is the floor):
      V-b-M : 10% of hidden dims boosted, different in every matrix -> only this matrix's earlier rows show it;
      V-b-E : 10% of hidden dims boosted in both w3 and w1 of an expert (different in every expert) -> w1 sees it
              in w3's whole hidden line. (A first version boosted neurons instead; that is visible in w1's own row
              prefix, so base already had it and the check could not show the E level -- replaced.)
      V-b-L : 10% of hidden dims boosted, the same in every expert of the layer -> the reference experts show it.
      V-a   : no boost.
    Each: 6 training experts, 1 validation, 1 test (w3 and w1, 512 x 2048 / 2048-wide), 8 reference experts.
    Checks: V-a every level about 0; in V-b-X the level X (on top of base) recovers >= 70% of (chain - floor) on the
    matrices where the structure is present, and beats base by at least half of that."""
    sd = HERE.parents[1] / "data" / ("011_synth_c3" + tag)
    os.environ["D011"] = str(sd)
    import torch
    import amodel as A
    import cx
    dev = "cuda:0"
    out = {}
    R, W = 512, 2048
    for kind in kinds:
        rng = np.random.default_rng(61)
        shared = np.where(rng.random(W) < 0.1, 2.0, 1.0) if kind == "V-b-L" else np.ones(W)
        mats, truth = [], {}

        def gen(Lb):
            E = rng.integers(117, 123, size=(R, W // 32)).astype(np.uint8)
            lam = (0.8 + 0.4 * (np.repeat(E.astype(np.float64), 32, axis=1) - 117)) * Lb
            return cx.poisson_codes(rng, lam), E, lam

        def boost_pair():
            hb = np.where(rng.random(W) < 0.1, 2.0, 1.0) if kind == "V-b-M" else np.ones(W)
            return hb, np.ones(R)

        for i in range(16):                      # 0-5 pool, 6 val, 7 test, 8-15 reference
            hb_e = np.where(rng.random(W) < 0.1, 2.0, 1.0) if kind == "V-b-E" else None
            for role in ("w3", "w1"):
                hb, nb = boost_pair()
                if hb_e is not None:
                    hb = hb_e                         # the same hidden dims boosted in w3 and w1 of an expert
                Lb = shared[None, :] * hb[None, :] * nb[:, None]
                C, E, lam = gen(Lb)
                name = f"L0.E{i}.{role}"
                if i >= 8:
                    (sd / "ref").mkdir(parents=True, exist_ok=True)
                    S = np.where(C == 8, 0, C)
                    np.savez(sd / "ref" / f"{name}.npz", sym=np.where(S > 8, S - 1, S).astype(np.uint8), E=E)
                    continue
                split = "pool" if i < 6 else "val" if i == 6 else "T1"
                mats.append({"name": name, "layer": 0, "expert": i, "role": role, "C": C, "E": E, "split": split,
                             "cross": f"L0.E{i}.w3" if role == "w1" else None, "cross_kind": "same" if role == "w1" else None})
                truth[name] = (C, lam)
        # reference index: only w1 / w3 exist here, write an empty w2 for the loader
        for i in range(8, 16):
            z = np.zeros((W, R), dtype=np.uint8)
            np.savez(sd / "ref" / f"L0.E{i}.w2.npz", sym=z, E=np.full((W, R // 32), 120, np.uint8))
        json.dump({"0": list(range(8, 16))}, open(sd / "ref" / "index.json", "w"))
        chain = cx.build_synth(sd, mats, seed=2)
        stage_prep(workers=8)
        A.D = sd
        store = cx.StoreX(dev)
        tests = ["L0.E7.w3", "L0.E7.w1"]
        n = sum(truth[t][0].size for t in tests)
        floor = sum(cx.poisson_bits(*truth[t]) for t in tests)
        ch = sum(chain[t] for t in tests)
        rec = {"theory": (ch - floor) / n}
        if kind == "V-b-E":                        # the structure only helps w1 (it is in w3 itself as well)
            n1 = truth["L0.E7.w1"][0].size
            rec["theory_w1"] = (chain["L0.E7.w1"] - cx.poisson_bits(*truth["L0.E7.w1"])) / n1
        lv_of = {"V-a": ["M", "E", "L"], "V-b-M": ["M"], "V-b-E": ["E"], "V-b-L": ["L"]}
        for cfg, lv in (("base", []), (kind, lv_of[kind])):
            store.extras = [Levels(store, lv, dev)] if lv else []
            model = make_model(lv, dev)
            A.train(model, store, 6 * 2 * R * W, dev, 3e-3, budget=60_000_000, eval_every=5_000_000, log=lambda s: None)
            te = A.evaluate_test(model, store, tests, dev)
            rec[cfg] = sum(te[t]["chain_bits_exact"] - te[t]["bits"] for t in tests) / n
            rec[cfg + "_w1"] = (te["L0.E7.w1"]["chain_bits_exact"] - te["L0.E7.w1"]["bits"]) / te["L0.E7.w1"]["n"]
            del model
            torch.cuda.empty_cache()
        out[kind] = rec
        print(kind, rec, flush=True)
        del store
    if tag:
        return out
    ch = {}
    ch["V-a_about_0"] = bool(all(-0.01 < out["V-a"][k] < 0.005 for k in ("base", "V-a")))
    for k in ("V-b-M", "V-b-L"):
        t = out[k]["theory"]
        ch[f"{k}_recovers"] = bool(out[k][k] >= 0.7 * t and out[k][k] <= t + 2e-3)
        ch[f"{k}_beats_base"] = bool(out[k][k] - out[k]["base"] >= 0.5 * (t - out[k]["base"]) - 1e-9)
    t = out["V-b-E"]["theory_w1"]
    ch["V-b-E_recovers_w1"] = bool(out["V-b-E"]["V-b-E_w1"] >= 0.7 * t and out["V-b-E"]["V-b-E_w1"] <= t + 2e-3)
    ch["V-b-E_beats_base_w1"] = bool(out["V-b-E"]["V-b-E_w1"] - out["V-b-E"]["base_w1"] >= 0.5 * (t - out["V-b-E"]["base_w1"]) - 1e-9)
    out["checks"] = ch
    out["all_pass"] = all(ch.values())
    RES.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(RES / "synthetic_c3.json", "w"), indent=1)
    print(ch, out["all_pass"], flush=True)


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "run":
        stage_run(sys.argv[2:] or list(CONFIGS))
    else:
        {"prep": stage_prep, "causal": stage_causal, "synth": stage_synth}[st]()
