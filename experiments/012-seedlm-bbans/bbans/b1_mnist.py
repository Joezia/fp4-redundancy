"""B1: reproduce BB-ANS Table 2 with the official code (bits-back/bits-back @ 45aef5a).

Only the official modules are used (rans.py, util.py, torch_vae/*). The model set-up lines are
copied from torch_vae/torch_bin_mnist_compress.py and torch_vae/torch_mnist_compress.py
(precisions, latent shapes, saved parameter files, initial random words). What differs from the
official scripts:
  * number of images: the whole MNIST test set (10,000) instead of 200 / 100, and, as in the
    paper's Figure 3, a concatenation of three shuffled copies of the test set;
  * we additionally record the negative ELBO and the exact message size after every image.

usage: python b1_mnist.py bin|full test|test3x
"""
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/012/src/bits-back_bits-back'))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results/b1'))
os.chdir(REPO)
sys.path.insert(0, REPO)

import rans  # noqa: E402
import util  # noqa: E402
from torch_vae import tvae_utils  # noqa: E402
from torch_vae.tvae_binary import BinaryVAE  # noqa: E402
from torch_vae.tvae_beta_binomial import BetaBinomialVAE, beta_binomial_log_pdf  # noqa: E402
from torchvision import datasets, transforms  # noqa: E402
from torch.distributions import Bernoulli, Normal  # noqa: E402


def msg_bits(state):
    """Exact size of an rANS message in bits: 32 per tail word plus the bit length of the head."""
    head, tail = state
    n = 0
    while tail:
        n += 1
        tail = tail[1]
    return 32 * n + head.bit_length()


def setup(kind):
    if kind == 'bin':
        # torch_bin_mnist_compress.py, lines 12-46
        prior_precision, obs_precision, q_precision = 8, 12, 14
        latent_shape = (40,)
        model = BinaryVAE(hidden_dim=100, latent_dim=40)
        model.load_state_dict(torch.load('torch_vae/saved_params/torch_binary_vae_params_random'))
        obs_append = tvae_utils.bernoulli_obs_append(obs_precision)
        obs_pop = tvae_utils.bernoulli_obs_pop(obs_precision)
        n_other = 20
    else:
        # torch_mnist_compress.py, lines 12-38
        prior_precision, obs_precision, q_precision = 8, 14, 14
        latent_dim = 50
        latent_shape = (1, latent_dim)
        model = BetaBinomialVAE(hidden_dim=200, latent_dim=latent_dim)
        model.load_state_dict(torch.load('torch_vae/saved_params/torch_vae_beta_binomial_params',
                                         map_location=lambda storage, location: storage))
        model.eval()
        obs_append = tvae_utils.beta_binomial_obs_append(255, obs_precision)
        obs_pop = tvae_utils.beta_binomial_obs_pop(255, obs_precision)
        n_other = 50
    rec_net = tvae_utils.torch_fun_to_numpy_fun(model.encode)
    gen_net = tvae_utils.torch_fun_to_numpy_fun(model.decode)
    vae_append = util.vae_append(latent_shape, gen_net, rec_net, obs_append, prior_precision, q_precision)
    vae_pop = util.vae_pop(latent_shape, gen_net, rec_net, obs_pop, prior_precision, q_precision)
    return model, vae_append, vae_pop, n_other, dict(prior_precision=prior_precision,
                                                     obs_precision=obs_precision, q_precision=q_precision,
                                                     latent_shape=list(latent_shape))


def load_images(kind, mode):
    mnist = datasets.MNIST('data/mnist', train=False, download=True,
                           transform=transforms.Compose([transforms.ToTensor()]))
    images = mnist.data
    if kind == 'bin':
        # official: stochastic binarisation, torch seed 17 (set when tvae_binary is imported)
        torch.manual_seed(17)
        images = Bernoulli(images.float() / 255.).sample()
        flat = [im.view(-1) for im in images]
    else:
        flat = [im.float().view(1, -1) for im in images]
    if mode == 'test3x':
        rs = np.random.RandomState(0)
        order = np.concatenate([rs.permutation(len(flat)) for _ in range(3)])
        flat = [flat[i] for i in order]
    return flat


def neg_elbo(kind, model, images, n_samples=20):
    """Returns (official test() value, Monte-Carlo negative ELBO), both in bits per dimension.

    The official test() calls model.loss() in eval mode, where reparameterize returns the posterior
    mean (z = mu). The Monte-Carlo value samples z ~ q(z|x) (n_samples draws; batch norm in eval mode).
    """
    x = torch.stack([im.view(-1) for im in images])
    model.eval()
    off = []
    mc = []
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for i in range(0, len(x), 100):
            xb = x[i:i + 100]
            off.append(model.loss(xb).item() * len(xb))
            mu, std = model.encode(xb)
            acc = []
            for _ in range(n_samples):
                z = mu + std * torch.randn(mu.shape, generator=g)
                if kind == 'bin':
                    l = Bernoulli(model.decode(z)).log_prob(xb).sum(1)
                else:
                    a, b = model.decode(z)
                    l = beta_binomial_log_pdf(xb, torch.full_like(xb, 255.), a, b).sum(1)
                pz = Normal(0., 1.).log_prob(z).sum(1)
                qz = Normal(mu, std).log_prob(z).sum(1)
                acc.append(-(l + pz - qz))
            mc.append(torch.stack(acc).mean(0).sum().item())
    d = 784 * len(x)
    # model.loss() is already a per-dimension mean (bits), so only the image count divides it
    return sum(off) / len(x), sum(mc) / d * np.log2(np.e)


def main():
    kind, mode = sys.argv[1], sys.argv[2]
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    model, vae_append, vae_pop, n_other, cfg = setup(kind)
    images = load_images(kind, mode)
    n = len(images)
    off_elbo, mc_elbo = neg_elbo(kind, model, images[:10000] if mode == 'test' else images)

    rng = np.random.RandomState(0)
    other_bits = rng.randint(low=1 << 16, high=1 << 31, size=n_other, dtype=np.uint32)
    state = rans.unflatten(other_bits)
    init_bits = msg_bits(state)

    sizes = []
    t0 = time.time()
    for i, image in enumerate(images):
        state = vae_append(state, image)
        if i % 100 == 0 or i == n - 1:  # msg_bits walks the cons list, so sample it
            sizes.append(msg_bits(state))
        if i % 1000 == 0:
            print('encoded', i, sizes[-1], time.time() - t0, flush=True)
    t_enc = time.time() - t0
    flat = rans.flatten(state)
    total_bits_words = 32 * len(flat)

    # measure the clean bits consumed by the first image's posterior pop, on a fresh message
    big = rans.unflatten(np.random.RandomState(1).randint(1 << 16, 1 << 31, size=200, dtype=np.uint32))
    s0 = msg_bits(big)
    post_pop_bits = None
    try:
        # re-run the official append step by step: post_pop is the first stage of bb_ans_append
        from util import gaussian_latent_cdf, gaussian_latent_ppf, non_uniforms_pop
        rec = tvae_utils.torch_fun_to_numpy_fun(model.encode)
        m, s = rec(images[0])
        m, s = np.ravel(m), np.ravel(s)
        cdfs = [gaussian_latent_cdf(a, b, cfg['prior_precision'], cfg['q_precision']) for a, b in zip(m, s)]
        ppfs = [gaussian_latent_ppf(a, b, cfg['prior_precision'], cfg['q_precision']) for a, b in zip(m, s)]
        st, _ = non_uniforms_pop(cfg['q_precision'], ppfs, cdfs)(big)
        post_pop_bits = s0 - msg_bits(st)
    except Exception as e:  # noqa: BLE001
        print('post-pop measurement failed', e)

    # decode everything and check bit-exactness
    state = rans.unflatten(flat)
    t0 = time.time()
    ok = True
    for k in range(n):
        state, im = vae_pop(state)
        orig = images[n - k - 1].numpy()
        if not np.array_equal(np.asarray(orig).ravel(), np.asarray(im).ravel()):
            ok = False
            print('MISMATCH at', n - k - 1)
            break
        if k % 1000 == 0:
            print('decoded', k, time.time() - t0, flush=True)
    t_dec = time.time() - t0
    rec_other = rans.flatten(state)
    ok_other = np.array_equal(rec_other, other_bits)

    dims = 784 * n
    res = dict(kind=kind, mode=mode, n_images=n, cfg=cfg, n_other_words=n_other, init_bits=init_bits,
               net_bits_official=32 * (len(flat) - len(other_bits)),
               bpd_official=32 * (len(flat) - len(other_bits)) / dims,
               bpd_exact_net=(sizes[-1] - init_bits) / dims,
               bpd_with_initial=total_bits_words / dims,
               neg_elbo_official_test=off_elbo, neg_elbo_mc20=mc_elbo,
               first_image_posterior_pop_bits=post_pop_bits,
               first_image_increment_bits=sizes[0] - init_bits,
               roundtrip_ok=bool(ok), other_bits_recovered=bool(ok_other),
               encode_s=t_enc, decode_s=t_dec,
               sizes_every_100=sizes)
    with open(os.path.join(OUT, f'b1_{kind}_{mode}.json'), 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'sizes_every_100'}, indent=1))


if __name__ == '__main__':
    main()
