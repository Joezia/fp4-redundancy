"""Spec 014, supplement (user request): 3-D magnitude surfaces of FP4 expert weights, drawn like Figure 1 of SmoothQuant
(absolute value over input channel x output channel).

Values: dequantized |w| = |v| 2^(E-127). Storage is PyTorch layout (rows = output channels, columns = input channels):
w1 / w3 (neurons x hidden): in = hidden dimension, out = neuron; w2 (hidden x neurons): in = neuron, out = hidden.
A full matrix has 5-12 million elements, too many for a surface. The neuron axis is kept at full resolution and the
hidden axis is pooled in groups of 32 (for w1 / w3 these are exactly the scale blocks): each point is the RMS of 32
values of one neuron. (Max pooling was tried first and discarded: FP4 values take few levels, so nearly every cell's
maximum is the same grid value and the surface is flat.)
Below each real surface the same matrix with all elements randomly permuted (seed 20261123) is drawn with the same
pooling and z range, as the "no structure" reference.
Each figure is written as PNG and PDF and only the smaller file is kept (user request).

    PYTHONPATH=. python experiments/014-heatmaps/heatmap3d.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from redundancy.adapters.fp4 import E2M1, FP4  # noqa: E402

import heatmaps as H  # noqa: E402

OUT = H.FIG / "3d"
CELL = 32
SEED = 20261123
PANELS = {
    "deepseek": ["L0.E323.w1", "L20.E86.w1", "L20.E86.w2", "L42.E8.w1", "L42.E8.w2"],
    "kimi": ["L46.E484.w1", "L46.E484.w2", "L1.E236.w1", "L1.E236.w2"],
}
NOTE = {
    "L0.E323.w1": "layer 0: hidden-dim profile shared by experts",
    "L20.E86.w1": "typical", "L20.E86.w2": "typical",
    "L42.E8.w1": "strongest DeepSeek expert", "L42.E8.w2": "strongest DeepSeek expert",
    "L46.E484.w1": "typical", "L46.E484.w2": "typical",
    "L1.E236.w1": "strongest Kimi-K3 expert", "L1.E236.w2": "strongest Kimi-K3 expert",
}


def load(model, name):
    m = H.MODELS[model]
    ad = FP4({"samples": str(H.ROOT / m["samples"]), "snapshot_env": m["snapshot_env"]})
    C, E = ad._read(ad.samples[name])
    return np.abs(E2M1[C.astype(np.int64)] * np.exp2(np.repeat(E.astype(np.float64), 32, axis=1) - 127))


def pool(a, name):
    """RMS over groups of CELL along the hidden axis (columns of w1 / w3, rows of w2); neurons kept one by one."""
    R, W = a.shape
    if name.endswith("w2"):
        return np.sqrt((a * a).reshape(R // CELL, CELL, W).mean(axis=1)), (CELL, 1)
    return np.sqrt((a * a).reshape(R, W // CELL, CELL).mean(axis=2)), (1, CELL)


def axes_names(name, shape):
    R, W = shape
    if name.endswith("w2"):
        return f"In channel (neuron, {W})", f"Out channel (hidden, {R})"
    return f"In channel (hidden, {W})", f"Out channel (neuron, {R})"


def surface(ax, P, step, zmax, title, xl, yl):
    ny, nx = P.shape
    X, Y = np.meshgrid(np.arange(nx) * step[1], np.arange(ny) * step[0])
    ax.plot_surface(X, Y, P, cmap="coolwarm", vmin=0, vmax=zmax, rcount=ny, ccount=nx, linewidth=0,
                    antialiased=False)
    ax.set_zlim(0, zmax)
    ax.set_xlabel(xl, fontsize=7, labelpad=2)
    ax.set_ylabel(yl, fontsize=7, labelpad=2)
    ax.set_zlabel(f"RMS of |w| over {CELL} hidden dims", fontsize=7, labelpad=2)
    ax.tick_params(labelsize=6, pad=0)
    ax.view_init(elev=28, azim=-58)
    ax.set_title(title, fontsize=8)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    info = {"cell": CELL, "seed": SEED, "panels": {}}
    for model, names in PANELS.items():
        label = H.MODELS[model]["label"]
        fig = plt.figure(figsize=(4.2 * len(names), 7.8))
        for j, name in enumerate(names):
            a = load(model, name)
            s = a.ravel()[np.random.default_rng(SEED).permutation(a.size)].reshape(a.shape)
            (P, step), (Ps, _) = pool(a, name), pool(s, name)
            zmax = float(max(P.max(), Ps.max()))
            xl, yl = axes_names(name, a.shape)
            ax = fig.add_subplot(2, len(names), j + 1, projection="3d")
            surface(ax, P, step, zmax, f"{label} {name} ({a.shape[0]}x{a.shape[1]})\n{NOTE[name]}", xl, yl)
            ax = fig.add_subplot(2, len(names), len(names) + j + 1, projection="3d")
            surface(ax, Ps, step, zmax, f"{name} shuffled (seed {SEED})", xl, yl)
            info["panels"][name] = {"model": model, "shape": list(a.shape), "zmax": zmax,
                                    "max_abs": float(a.max()), "median_abs": float(np.median(a)),
                                    "pooled_max_real": float(P.max()), "pooled_median_real": float(np.median(P)),
                                    "pooled_median_shuffled": float(np.median(Ps))}
            # single panel, larger
            f1 = plt.figure(figsize=(7, 5.5))
            ax1 = f1.add_subplot(1, 1, 1, projection="3d")
            surface(ax1, P, step, zmax, f"{label} {name}: dequantized |w| (exact in BF16), RMS over {CELL} hidden dims", xl, yl)
            H.savefig(f1, OUT / f"{model}_{name}")
        fig.suptitle(f"{label}: |dequantized weight| over input x output channel (top: real; bottom: all elements "
                     f"shuffled; each point = RMS of {CELL} values of one neuron along the hidden axis; same z range per column)", fontsize=10)
        kept = H.savefig(fig, OUT / f"{model}_overview")
        info[f"{model}_overview"] = kept
        print(model, kept, flush=True)
    json.dump(info, open(H.RES / "heatmap3d.json", "w"), indent=1)
    print(json.dumps(info["panels"], indent=0))


if __name__ == "__main__":
    main()
