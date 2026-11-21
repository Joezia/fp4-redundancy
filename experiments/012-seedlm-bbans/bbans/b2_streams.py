"""B2: BB-ANS with the FP4 VAE over N parallel streams (Craystack), with the network on CPU or GPU.

usage: python b2_streams.py MODEL_FILE MATRIX N[,N...] cpu|cuda [n_blocks]
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fp4vae as V  # noqa: E402
import streams as ST  # noqa: E402
from streams import cs  # noqa: E402
from b2_code import load_model, PRIOR_PREC, Q_PREC  # noqa: E402

OUT = os.path.join(HERE, "../results/b2/streams")


class VAEAdapter:
    def __init__(self, model, device):
        self.m = model.to(device)
        self.dev = device
        self.d = model.latent_dim
        self.centres = cs.std_gaussian_centres(PRIOR_PREC)

    @torch.no_grad()
    def post_codecs(self, X, s, E):
        mu, std = self.m.encode(torch.as_tensor(X, device=self.dev), torch.as_tensor(s, device=self.dev))
        mu, std = mu.double().cpu().numpy(), std.double().cpu().numpy()
        return [cs.DiagGaussian_StdBins(mu[:, j], std[:, j], Q_PREC, PRIOR_PREC) for j in range(self.d)]

    def prior_codecs(self, s, E):
        return [cs.Uniform(PRIOR_PREC)] * self.d

    @torch.no_grad()
    def lik_probs(self, Z, s, E):
        y = torch.as_tensor(self.centres[Z], dtype=torch.float32, device=self.dev)
        logits = self.m.decode(y, torch.as_tensor(s, device=self.dev))
        return torch.softmax(logits.double(), -1).cpu().numpy()


def main():
    model_file, name, Ns, device = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")], sys.argv[4]
    limit = int(sys.argv[5]) if len(sys.argv) > 5 else None
    torch.set_num_threads(1)
    if device == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
    os.makedirs(OUT, exist_ok=True)
    model = load_model(model_file)
    ad = VAEAdapter(model, device)
    sym, E, _ = V.matrix_blocks(name)
    if limit:
        sym, E = sym[:limit], E[:limit]
    sf = V.scale_feature(E).numpy().astype(np.float32)
    tag = os.path.basename(model_file).replace(".pt", "")
    for N in Ns:
        res = ST.run(ad, sym, sf, E, N)
        res.update(matrix=name, model_file=model_file, device=device, limit=limit, n_weights=len(sym) * V.B)
        with open(os.path.join(OUT, f"{tag}.jsonl"), "a") as f:
            f.write(json.dumps(res) + "\n")
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
