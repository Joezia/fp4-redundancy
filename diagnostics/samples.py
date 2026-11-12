"""Section 3: sample selection from the safetensors headers. Writes samples.json."""
import argparse
import json
import re
from pathlib import Path

import numpy as np

from .st import index_snapshot

ROUTED = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w\d+)\.weight$")
ROLES = ("w1", "w2", "w3")


def fp4_weights(index):
    """FP4 weights = I8 tensors of routed/MTP experts; each has a sibling `.scale` (F8_E8M0)."""
    return {k: v for k, v in index.items() if v["dtype"] == "I8"}


def select(index):
    layers = sorted({int(m.group(1)) for k in index if (m := re.match(r"^layers\.(\d+)\.", k))})
    L = len(layers)
    assert layers == list(range(L)), "layer ids are not contiguous"
    chosen_layers = sorted({0, (L - 1) // 3, 2 * (L - 1) // 3, L - 1})
    experts_by_layer = {}
    for k in index:
        m = ROUTED.match(k)
        if m:
            experts_by_layer.setdefault(int(m.group(1)), set()).add(int(m.group(2)))
    samples = []
    for layer in chosen_layers:
        pool = sorted(experts_by_layer[layer])
        rng = np.random.default_rng(20261112 + layer)
        drawn = [int(x) for x in rng.choice(pool, size=3, replace=False)]
        for expert in sorted(drawn):
            for role in ROLES:
                w = f"layers.{layer}.ffn.experts.{expert}.{role}.weight"
                s = w[: -len("weight")] + "scale"
                if w not in index:
                    continue
                we, se = index[w], index[s]
                rows, cols_b = we["shape"]
                samples.append({
                    "matrix": f"L{layer}.E{expert}.{role}", "layer": layer, "expert": expert,
                    "role": role, "drawn_order": drawn,
                    "weight": w, "weight_file": we["file"], "weight_dtype": we["dtype"],
                    "weight_shape": we["shape"], "scale": s, "scale_file": se["file"],
                    "scale_dtype": se["dtype"], "scale_shape": se["shape"],
                    "codeword_shape": [rows, 2 * cols_b], "codewords": rows * 2 * cols_b,
                })
    return {"L": L, "layers": chosen_layers, "n_routed_experts_per_layer":
            {str(l): len(experts_by_layer[l]) for l in chosen_layers},
            "rng": "numpy.random.default_rng(20261112 + layer).choice(sorted expert ids, 3, replace=False)",
            "numpy_version": np.__version__, "matrices": samples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out", default="experiments/r1-r4/samples.json")
    args = ap.parse_args()
    out = select(index_snapshot(args.snapshot))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(len(out["matrices"]), "matrices; layers", out["layers"])


if __name__ == "__main__":
    main()
