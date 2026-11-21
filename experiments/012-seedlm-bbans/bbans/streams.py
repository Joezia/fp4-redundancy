"""BB-ANS over N parallel ANS streams with Craystack (j-towns/craystack @ 992e14d, unmodified).

The message head has shape (N,): N independent rANS states sharing one tail, as in Craystack's vectorised rANS.
Step t codes blocks t*N .. t*N+N-1, one block per stream (the last step uses the first r streams). Within a
step every codec call is vectorised over the streams. BB-ANS order per block, as bits-back util.bb_ans_append:
pop the latent from the posterior, push the 32 codes with the likelihood, push the latent with the prior.

A model adapter provides, for a batch of blocks (X (n, 32) int, s (n,) scale feature, E (n,) raw scale):
  post_codecs(X, s, E)  -> list of Craystack codecs, one per latent dimension (pop order)
  prior_codecs(s, E)    -> list of Craystack codecs, one per latent dimension
  lik_probs(Z, s, E)    -> (n, 32, 15) float64 probabilities given the latent symbols Z (n, n_latent)
"""
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, os.path.join(ROOT, "data/012/src/j-towns_craystack"))
import craystack as cs  # noqa: E402
from craystack import rans as vrans  # noqa: E402

OBS_PREC = 14


def _sub(message, r):
    head, tail = message
    return (head[:r], tail), head[r:]


def _join(message, rest):
    head, tail = message
    return (np.concatenate([head, rest]), tail)


def push_step(message, model, X, s, E, r):
    """encode blocks X (r of them) on the first r streams."""
    sub, rest = _sub(message, r)
    z = []
    for c in model.post_codecs(X, s, E):
        sub, zj = c.pop(sub)
        z.append(np.asarray(zj, dtype=np.int64))
    Z = np.stack(z, 1)
    P = model.lik_probs(Z, s, E)
    for k in reversed(range(X.shape[1])):
        sub, = cs.Categorical(P[:, k], OBS_PREC).push(sub, np.uint64(X[:, k]))
    pri = model.prior_codecs(s, E)
    for j in reversed(range(Z.shape[1])):
        sub, = pri[j].push(sub, np.uint64(Z[:, j]))
    return _join(sub, rest)


def pop_step(message, model, s, E, r, n_pos=32):
    sub, rest = _sub(message, r)
    z = []
    for c in model.prior_codecs(s, E):
        sub, zj = c.pop(sub)
        z.append(np.asarray(zj, dtype=np.int64))
    Z = np.stack(z, 1)
    P = model.lik_probs(Z, s, E)
    xs = []
    for k in range(n_pos):
        sub, xk = cs.Categorical(P[:, k], OBS_PREC).pop(sub)
        xs.append(np.asarray(xk, dtype=np.int64))
    X = np.stack(xs, 1)
    post = model.post_codecs(X, s, E)
    for j in reversed(range(Z.shape[1])):
        sub, = post[j].push(sub, np.uint64(Z[:, j]))
    return _join(sub, rest), X


def message_bits(message):
    return 32 * len(cs.flatten(message))


def run(model, sym, sf, E, N, check=True):
    """Encode all blocks of one matrix on N streams, decode, verify. Returns a dict of measurements."""
    n = len(sym)
    steps = [(t, min(N, n - t)) for t in range(0, n, N)]
    message = cs.base_message((N,))
    base_bits = message_bits(message)
    warnings.simplefilter("ignore")
    borrowed_before = _borrow_counter[0]
    t0 = time.time()
    for t, r in steps:
        message = push_step(message, model, sym[t:t + r].astype(np.int64), sf[t:t + r], E[t:t + r], r)
    t_enc = time.time() - t0
    flat = cs.flatten(message)
    gross = 32 * len(flat)
    borrowed = _borrow_counter[0] - borrowed_before
    ok = None
    t_dec = None
    if check:
        message = cs.unflatten(flat, (N,))
        t0 = time.time()
        ok = True
        for t, r in reversed(steps):
            message, X = pop_step(message, model, sf[t:t + r], E[t:t + r], r)
            if not np.array_equal(X, sym[t:t + r].astype(np.int64)):
                ok = False
                break
        t_dec = time.time() - t0
    return dict(N=N, n_blocks=n, bits_gross=gross, base_bits=base_bits, borrowed_words=int(borrowed),
                roundtrip_ok=ok, encode_s=t_enc, decode_s=t_dec,
                decode_codes_per_s=(n * sym.shape[1] / t_dec) if t_dec else None)


# count the 32-bit words Craystack generates when a stream pops from an empty tail (the 'initial bits')
_borrow_counter = [0]
_orig_slice = vrans.stack_slice


def _counting_slice(stack, n):
    m = n
    s = stack
    while m > 0 and s:
        arr, s = s
        m -= len(arr)
    if m > 0:
        _borrow_counter[0] += m
    return _orig_slice(stack, n)


vrans.stack_slice = _counting_slice
