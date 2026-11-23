"""ctypes wrapper of racoder.c (compiled to data/012/bin/libracoder.so; the binary is not committed)."""
import ctypes
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.abspath(os.path.join(HERE, "../../../data/012/bin/libracoder.so"))
if not os.path.exists(LIB):
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    subprocess.check_call(["gcc", "-O2", "-shared", "-fPIC", "-o", LIB, os.path.join(HERE, "racoder.c")])
_L = ctypes.CDLL(LIB)
_P = ctypes.POINTER
_L.rac_encode.restype = ctypes.c_int64
_L.rac_encode.argtypes = [_P(ctypes.c_int32), _P(ctypes.c_int32), ctypes.c_int64, ctypes.c_int32, _P(ctypes.c_uint32)]
_L.rac_decode.restype = ctypes.c_int
_L.rac_decode.argtypes = [_P(ctypes.c_uint32), ctypes.c_int64, _P(ctypes.c_int32), ctypes.c_int64, ctypes.c_int32,
                          _P(ctypes.c_int32)]


def _p(a, t):
    return a.ctypes.data_as(_P(t))


def encode(sym, ctx):
    sym = np.ascontiguousarray(sym, dtype=np.int32).ravel()
    ctx = np.ascontiguousarray(ctx, dtype=np.int32).ravel()
    nctx = int(ctx.max()) + 1 if ctx.size else 1
    out = np.empty(len(sym) + 2, dtype=np.uint32)
    w = _L.rac_encode(_p(sym, ctypes.c_int32), _p(ctx, ctypes.c_int32), len(sym), nctx, _p(out, ctypes.c_uint32))
    return out[:w].copy(), nctx


def decode(words, ctx, n, nctx):
    ctx = np.ascontiguousarray(ctx, dtype=np.int32).ravel()
    words = np.ascontiguousarray(words, dtype=np.uint32)
    sym = np.empty(n, dtype=np.int32)
    rc = _L.rac_decode(_p(words, ctypes.c_uint32), len(words), _p(ctx, ctypes.c_int32), n, nctx,
                       _p(sym, ctypes.c_int32))
    if rc != 0:
        raise RuntimeError(f"rac_decode failed ({rc})")
    return sym
