"""Spec 011 line C1: the dequantised weights in the continuous domain -- low rank and per-dimension structure.

    PYTHONPATH=. python experiments/011-limits-and-models/c1_dequant.py run
    PYTHONPATH=. python experiments/011-limits-and-models/c1_dequant.py figures

Dequantised weight = value(merged code) * 2^(E - 127). Axes: w1 / w3 are (2304 neurons) x (5120 hidden dims), w2 is
(5120 hidden dims) x (2304 neurons); the hidden dims (residual stream) are shared by every expert and layer, the
neuron order is arbitrary (a joint permutation of w1/w3 rows and w2 columns leaves the network unchanged).

Measured on the 48 test matrices (T1 + T2), with the 1,032 reference matrices (c_ref.py) for cross-expert statistics:
  spectrum     : energy share of the top k singular values (k = 1, 4, 16, 64, 256), real vs the same matrix with its
                 entries randomly permuted (same values, no structure).
  lowrank_k    : a code-domain estimate of what a rank-k latent could save: the rank-k approximation (factors rounded
                 to fp16) divided by the block scale, cut into 16 cells (the E2M1 rounding cells, signed), is used as a
                 context of an adaptive code; gain = bits(code | scale byte) - bits(code | scale byte, cell) - the
                 factors (16 bit per entry, k * (R + W)). The same on the permuted matrix shows how much a rank-k fit
                 of pure noise appears to gain. Baseline is a scale-only adaptive code, not the chain (an indicator).
  hidden / neuron RMS : RMS of each hidden-dim line and each neuron line; coefficient of variation vs permuted;
                 correlations of log RMS across experts of a layer, adjacent layers, roles; neuron lines across
                 w1 rows / w3 rows / w2 columns of one expert.
  hidden_ctx   : a code-domain preview of C3's layer level: the mean |value| of the codes (scale-normalised) at
                 each hidden dim over the layer's 8 reference experts (same role), cut into 16 quantile cells, as a
                 context of an adaptive code: gain = bits(code | scale byte) - bits(code | scale byte, cell). Control:
                 the same with the hidden dims of the reference statistic randomly permuted.
"""
import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as K  # noqa: E402

D = Path(os.environ.get("D011", str(HERE.parents[1] / "data" / "011")))
RES = HERE / "results" / "c"
FIG = HERE / "figures"
VAL = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)
KS = (1, 4, 16, 64, 256)
EDGES = np.array([-5, -3.5, -2.5, -1.75, -1.25, -0.75, -0.25, 0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])   # 16 cells


def load(name, ref=False):
    z = np.load((D / "ref" if ref else D / "codes") / f"{name}.npz")
    return z["sym"], z["E"]


def scale_of(E):
    return np.repeat(np.exp2(E.astype(np.float32) - 127), 32, axis=1)


def deq(sym, E):
    return VAL[sym] * scale_of(E)


def hidden_axis(name):
    return 0 if name.endswith("w2") else 1      # axis along which a hidden-dim line runs ... see lines()


def lines(name, X):
    """(hidden-dim lines, neuron lines) as rows of 2-D arrays."""
    return (X, X.T) if name.endswith("w2") else (X.T, X)


def dbits(sym, ctx, K_=15):
    from redundancy import increment as I
    return float(I.dirichlet_bits(sym.ravel().astype(np.int64), ctx.ravel().astype(np.int64), K_))


def rms(A):
    return np.sqrt((A.astype(np.float64) ** 2).mean(1))


def one_test(name):
    sym, E = load(name)
    R, W = sym.shape
    n = sym.size
    X = deq(sym, E)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    Xp = X.ravel()[perm].reshape(R, W)
    symp = sym.ravel()[perm].reshape(R, W)
    Ep_scale = scale_of(E).ravel()[perm].reshape(R, W)      # the permuted matrix keeps each value with its scale
    sc = scale_of(E)
    ectx = np.repeat(E.astype(np.int64), 32, axis=1)
    ectxp = ectx.ravel()[perm].reshape(R, W)
    out = {"task": name, "R": R, "W": W}
    base = dbits(sym, ectx)
    basep = dbits(symp, ectxp)
    out["bits_scale_ctx"] = base / n
    for tag, M, S_, s_, ec, b0 in (("real", X, sc, sym, ectx, base), ("permuted", Xp, Ep_scale, symp, ectxp, basep)):
        U, s, Vt = np.linalg.svd(M.astype(np.float32), full_matrices=False)
        e = s.astype(np.float64) ** 2
        out[f"energy_{tag}"] = {k: float(e[:k].sum() / e.sum()) for k in KS}
        out[f"sv_{tag}"] = (s[:512] / s[0]).astype(float).tolist()[::4]
        for k in KS:
            Uk = (U[:, :k] * s[:k]).astype(np.float16).astype(np.float32)
            Vk = Vt[:k].astype(np.float16).astype(np.float32)
            cell = np.searchsorted(EDGES, (Uk @ Vk) / S_)
            b1 = dbits(s_, ec * 16 + cell)
            out.setdefault(f"lowrank_{tag}", {})[k] = {"gain_before_factors": (b0 - b1) / n,
                                                       "factor_bits_per_code": 16.0 * k * (R + W) / n,
                                                       "net_gain": (b0 - b1) / n - 16.0 * k * (R + W) / n}
    hl, nl = lines(name, X)
    hlp, nlp = lines(name, Xp)
    cv = lambda v: float(v.std() / v.mean())
    out["hidden_rms_cv"] = {"real": cv(rms(hl)), "permuted": cv(rms(hlp))}
    out["neuron_rms_cv"] = {"real": cv(rms(nl)), "permuted": cv(rms(nlp))}
    # code-domain preview of the layer level: reference experts' mean |value| per hidden dim (same role)
    L = int(name.split(".")[0][1:])
    role = name.split(".")[2]
    ref = json.load(open(D / "ref" / "index.json"))[str(L)]
    stat = np.zeros(5120)
    for e in ref:
        s2, _ = load(f"L{L}.E{e}.{role}", ref=True)
        hl2, _ = lines(role, np.abs(VAL[s2]))
        stat += hl2.mean(1)
    stat /= len(ref)
    hctx = {}
    for tag, st in (("ref_stat", stat), ("ref_stat_hidden_permuted", stat[rng.permutation(5120)])):
        q = np.searchsorted(np.quantile(st, np.linspace(0, 1, 17)[1:-1]), st)          # 16 cells
        cellh = q[None, :] if role != "w2" else q[:, None]
        cellh = np.broadcast_to(cellh, sym.shape)
        hctx[tag] = (base - dbits(sym, ectx * 16 + cellh)) / n
    out["hidden_ctx_gain"] = hctx
    out["hidden_rms"] = rms(hl).astype(float).tolist()
    return out


def ref_hidden_rms(name):
    sym, E = load(name, ref=True)
    hl, nl = lines(name, deq(sym, E))
    return name, np.log(rms(hl) + 1e-30), np.log(rms(nl) + 1e-30)


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def stage_run():
    RES.mkdir(parents=True, exist_ok=True)
    names = K.T1 + K.T2
    with ProcessPoolExecutor(12) as ex:
        rows = list(ex.map(one_test, names))
    with open(D / "c1_rows_full.jsonl", "w") as f:          # with the 5,120 hidden-line RMS values (not committed)
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(RES / "c1_rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "hidden_rms"}) + "\n")
    # cross-expert / cross-layer / cross-role correlations of log hidden RMS on the reference experts
    idx = json.load(open(D / "ref" / "index.json"))
    refn = [f"L{L}.E{e}.{r}" for L, es in idx.items() for e in es for r in K.ROLES]
    with ProcessPoolExecutor(32) as ex:
        H = {n: (h, nn) for n, h, nn in ex.map(ref_hidden_rms, refn)}
    out = {"per_matrix": "c1_rows.jsonl"}
    same_layer, roles_w1w3, roles_w1w2, neur_w1w3, neur_w1w2 = [], [], [], [], []
    layer_mean = {}
    for L, es in idx.items():
        for r in K.ROLES:
            hs = [H[f"L{L}.E{e}.{r}"][0] for e in es]
            same_layer += [corr(hs[i], hs[j]) for i in range(len(hs)) for j in range(i + 1, len(hs))]
            layer_mean[(int(L), r)] = np.mean(hs, 0)
        for e in es:
            p = f"L{L}.E{e}."
            roles_w1w3.append(corr(H[p + "w1"][0], H[p + "w3"][0]))
            roles_w1w2.append(corr(H[p + "w1"][0], H[p + "w2"][0]))
            neur_w1w3.append(corr(H[p + "w1"][1], H[p + "w3"][1]))
            neur_w1w2.append(corr(H[p + "w1"][1], H[p + "w2"][1]))
    adj = [corr(layer_mean[(L, r)], layer_mean[(L + 1, r)]) for L in range(42) for r in K.ROLES]
    far = [corr(layer_mean[(L, r)], layer_mean[(L + 10, r)]) for L in range(32) for r in K.ROLES]
    med = lambda v: {"median": float(np.median(v)), "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)), "n": len(v)}
    out["log_hidden_rms_corr"] = {"same_layer_other_expert_same_role": med(same_layer), "adjacent_layer_mean": med(adj),
                                  "layer_plus_10_mean": med(far), "w1_vs_w3_same_expert": med(roles_w1w3),
                                  "w1_vs_w2_same_expert": med(roles_w1w2)}
    out["log_neuron_rms_corr"] = {"w1_row_vs_w3_row": med(neur_w1w3), "w1_row_vs_w2_col": med(neur_w1w2)}
    # aggregates over T1 / T2 (code-weighted)
    for split, ns in (("T1", K.T1), ("T2", K.T2)):
        s = [r for r in rows if r["task"] in ns]
        n = np.array([r["R"] * r["W"] for r in s], float)
        wavg = lambda f: float((np.array([f(r) for r in s]) * n).sum() / n.sum())
        agg = {"bits_scale_ctx": wavg(lambda r: r["bits_scale_ctx"])}
        for tag in ("real", "permuted"):
            agg[f"energy_{tag}"] = {k: float(np.mean([r[f"energy_{tag}"][str(k)] if str(k) in r[f"energy_{tag}"] else r[f"energy_{tag}"][k] for r in s])) for k in KS}
            agg[f"lowrank_{tag}"] = {k: {m: wavg(lambda r, k=k, m=m: (r[f"lowrank_{tag}"].get(str(k)) or r[f"lowrank_{tag}"][k])[m])
                                         for m in ("gain_before_factors", "factor_bits_per_code", "net_gain")} for k in KS}
        agg["hidden_rms_cv"] = {t: float(np.mean([r["hidden_rms_cv"][t] for r in s])) for t in ("real", "permuted")}
        agg["neuron_rms_cv"] = {t: float(np.mean([r["neuron_rms_cv"][t] for r in s])) for t in ("real", "permuted")}
        agg["hidden_ctx_gain"] = {t: wavg(lambda r, t=t: r["hidden_ctx_gain"][t]) for t in ("ref_stat", "ref_stat_hidden_permuted")}
        out[split] = agg
    best = max(out["T1"]["lowrank_real"].items(), key=lambda kv: kv[1]["net_gain"])
    out["c4_lowrank_decision"] = {"best_k": best[0], "T1_net_gain": best[1]["net_gain"], "threshold": 0.005,
                                  "do_lowrank_variant": bool(best[1]["net_gain"] > 0.005)}
    json.dump(out, open(RES / "c1_dequant.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_matrix"}, indent=1))


def stage_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [json.loads(l) for l in open(D / "c1_rows_full.jsonl")]
    FIG.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for r in rows[:6]:
        x = np.arange(len(r["sv_real"])) * 4 + 1
        ax.plot(x, r["sv_real"], lw=1, label=r["task"])
        ax.plot(x, r["sv_permuted"], lw=1, ls="--", color="gray")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Singular value / largest")
    ax.set_title("Dequantised weights: real (colour) vs entries permuted (grey)")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "c1_spectrum.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for r in rows[:6]:
        h = np.sort(np.array(r["hidden_rms"]))[::-1]
        ax.plot(h / np.median(h), lw=1, label=r["task"])
    ax.set_yscale("log")
    ax.set_xlabel("Hidden dims, sorted by RMS")
    ax.set_ylabel("RMS / median")
    ax.set_title("RMS of each hidden-dim line")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "c1_hidden_rms.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    {"run": stage_run, "figures": stage_figures}[sys.argv[1]]()
