"""B2: BB-ANS coding of one FP4 matrix with the trained VAE, single chain, official bits-back code.

All blocks of the matrix (raster order) are pushed onto one BB-ANS chain with util.vae_append /
util.vae_pop of bits-back/bits-back (unmodified), exactly as torch_vae/torch_mnist_compress.py does for
images: prior precision 8, posterior precision 14, observation precision 14, 50 random 32-bit words of
initial 'other bits'. The observation codec is the official util.categoricals_append/pop (15 classes).
The block scale is bound into the encoder/decoder closures (it is decoded first, from the 008 scale stream).
Decoding pops every block and checks it bit for bit; the initial words must come back unchanged.

usage: python b2_code.py MODEL_FILE MATRIX [n_blocks]
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fp4vae as V  # noqa: E402

BB = os.path.join(V.ROOT, "data/012/src/bits-back_bits-back")
sys.path.insert(0, BB)
import rans  # noqa: E402
import util  # noqa: E402

OUT = os.path.join(HERE, "../results/b2/code")
PRIOR_PREC, Q_PREC, OBS_PREC = 8, 14, 14       # torch_mnist_compress.py
N_OTHER = 50


def load_model(path):
    sd = torch.load(path, map_location="cpu")
    H = sd["fc1.weight"].shape[0]
    D = sd["fc21.weight"].shape[0]
    m = V.FP4VAE(H, D, bn="bn1.weight" in sd)
    m.load_state_dict(sd)
    m.eval()
    return m


def msg_bits(state):
    head, tail = state
    n = 0
    while tail:
        n += 1
        tail = tail[1]
    return 32 * n + head.bit_length()


def make_codec(model, scale):
    s = torch.tensor([scale], dtype=torch.float32)
    D = model.latent_dim

    @torch.no_grad()
    def rec_net(data):
        mu, std = model.encode(torch.as_tensor(np.asarray(data, dtype=np.int64)).view(1, -1), s)
        return mu.numpy(), std.numpy()

    @torch.no_grad()
    def gen_net(z):
        logits = model.decode(torch.as_tensor(np.asarray(z, dtype=np.float32)).view(1, D), s)
        return torch.softmax(logits.double(), -1).numpy()[0]           # (32, 15)

    def obs_append(probs):
        def append(state, data):
            return util.categoricals_append(probs, OBS_PREC)(state, np.int64(data))
        return append

    def obs_pop(probs):
        def pop(state):
            return util.categoricals_pop(probs, OBS_PREC)(state)
        return pop

    app = util.vae_append((1, D), gen_net, rec_net, obs_append, PRIOR_PREC, Q_PREC)
    pop = util.vae_pop((1, D), gen_net, rec_net, obs_pop, PRIOR_PREC, Q_PREC)
    return app, pop


@torch.no_grad()
def neg_elbo_bits(model, sym, sf, samples=8):
    st = torch.from_numpy(sym.astype(np.int64))
    s = torch.as_tensor(sf)
    torch.manual_seed(0)
    tot = 0.0
    for i in range(0, len(st), 16384):
        acc = 0.0
        for _ in range(samples):
            acc = acc + model.neg_elbo(st[i:i + 16384], s[i:i + 16384]).double().sum().item()
        tot += acc / samples
    return tot


def main():
    model_file, name = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    torch.set_num_threads(1)
    os.makedirs(OUT, exist_ok=True)
    model = load_model(model_file)
    sym, E, shape = V.matrix_blocks(name)
    if limit:
        sym, E = sym[:limit], E[:limit]
    n = len(sym)
    sf = V.scale_feature(E).numpy()
    rng = np.random.RandomState(0)
    other = rng.randint(low=1 << 16, high=1 << 31, size=N_OTHER, dtype=np.uint32)
    state = rans.unflatten(other)
    init_bits = msg_bits(state)
    t0 = time.time()
    for b in range(n):
        app, _ = make_codec(model, float(sf[b]))
        state = app(state, sym[b])
    t_enc = time.time() - t0
    flat = rans.flatten(state)
    gross = 32 * len(flat)
    net_official = 32 * (len(flat) - N_OTHER)
    exact = msg_bits(state)
    # decode
    state = rans.unflatten(flat)
    t0 = time.time()
    ok = True
    for k in range(n):
        b = n - 1 - k
        _, pop = make_codec(model, float(sf[b]))
        state, x = pop(state)
        if not np.array_equal(np.asarray(x).ravel(), sym[b].astype(np.int64)):
            ok = False
            print("MISMATCH", name, b, flush=True)
            break
    t_dec = time.time() - t0
    ok_other = bool(np.array_equal(rans.flatten(state), other))
    elbo = neg_elbo_bits(model, sym, sf)
    res = dict(matrix=name, model_file=model_file, n_blocks=n, n_weights=n * V.B, init_bits=init_bits,
               bits_gross=gross, bits_net_official=net_official, bits_net_exact=exact - init_bits,
               neg_elbo_bits=elbo, roundtrip_ok=ok, other_bits_recovered=ok_other,
               encode_s=t_enc, decode_s=t_dec, decode_codes_per_s=n * V.B / t_dec if t_dec else None)
    tag = os.path.basename(model_file).replace(".pt", "")
    with open(os.path.join(OUT, f"{tag}{'_lim' + str(limit) if limit else ''}.jsonl"), "a") as f:
        f.write(json.dumps(res) + "\n")
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
