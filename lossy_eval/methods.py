"""Lossy method interface and the reference perturbations used to calibrate the framework.

A method maps one original expert matrix to a reconstruction W_hat of the same shape.
The framework calls `reconstruct(ctx)` once per (layer, expert, role); the returned
tensor is used in place of the original weights in the perturbed stream only.
"""
from dataclasses import dataclass, field
import hashlib
import torch


@dataclass
class MatrixCtx:
    model: str
    layer: int
    expert: int
    role: str                 # gpt-oss: gate_up / down ; DeepSeek, Kimi: w1 / w3 / w2
    W: torch.Tensor           # float32 [out, in], storage orientation (rows = output features)
    codes: torch.Tensor       # uint8 [out, in] E2M1 codes
    scales: torch.Tensor      # uint8 [out, in/32] E8M0 exponents
    extra: dict = field(default_factory=dict)

    def seed(self, salt: str) -> int:
        h = hashlib.sha256(f"{salt}|{self.model}|{self.layer}|{self.expert}|{self.role}".encode()).digest()
        return int.from_bytes(h[:8], "little") & ((1 << 63) - 1)


class LossyMethod:
    """Base class. Subclasses set `name` and implement `reconstruct`."""
    name = "base"
    identity = False

    def params(self) -> dict:
        return {}

    def reconstruct(self, ctx: MatrixCtx):
        """Return W_hat (tensor, same shape as ctx.W) or (W_hat, info dict).
        `info` may carry e.g. {"bits_per_weight": ...}; rates are only an x-axis."""
        raise NotImplementedError


class Identity(LossyMethod):
    name = "identity"
    identity = True

    def reconstruct(self, ctx):
        return ctx.W.clone(), {"bits_per_weight": 4.25}


class WhiteNoise(LossyMethod):
    """W_hat = W + sigma * N(0, 1), sigma^2 = 10^(nmse_db/10) * mean(W^2), per matrix."""
    name = "white_noise"

    def __init__(self, nmse_db: float, seed: str = "wn"):
        self.nmse_db, self.salt = float(nmse_db), seed

    def params(self):
        return {"nmse_db": self.nmse_db}

    def reconstruct(self, ctx):
        g = torch.Generator(device=ctx.W.device).manual_seed(ctx.seed(self.salt))
        sigma = (10 ** (self.nmse_db / 10) * ctx.W.pow(2).mean()).sqrt()
        noise = torch.randn(ctx.W.shape, generator=g, device=ctx.W.device, dtype=torch.float32)
        return ctx.W + sigma * noise, {}


class BlockRTN(LossyMethod):
    """Round-to-nearest to a symmetric k-bit uniform grid per 32-element block of a row,
    absmax scale kept in bf16 (rate k + 16/32 bits per weight). k=3: levels -3..3 (7 used);
    k=2: levels {-1.5, -0.5, 0.5, 1.5}."""
    name = "block_rtn"

    def __init__(self, bits: int, block: int = 32):
        assert bits in (2, 3, 4)
        self.bits, self.block = bits, block

    def params(self):
        return {"bits": self.bits, "block": self.block}

    def reconstruct(self, ctx):
        out, inn = ctx.W.shape
        x = ctx.W.reshape(out, inn // self.block, self.block)
        amax = x.abs().amax(-1, keepdim=True)
        if self.bits == 2:
            s = (amax / 1.5).to(torch.bfloat16).float().clamp_min(1e-30)
            q = (torch.floor(x / s) + 0.5).clamp(-1.5, 1.5)
        else:
            qmax = 2 ** (self.bits - 1) - 1
            s = (amax / qmax).to(torch.bfloat16).float().clamp_min(1e-30)
            q = torch.round(x / s).clamp(-qmax, qmax)
        return (q * s).reshape(out, inn), {"bits_per_weight": self.bits + 16 / self.block}


def build(spec: str) -> LossyMethod:
    """'identity' | 'white_noise:-10' | 'block_rtn:3'"""
    name, _, arg = spec.partition(":")
    if name == "identity":
        return Identity()
    if name == "white_noise":
        return WhiteNoise(float(arg))
    if name == "block_rtn":
        return BlockRTN(int(arg))
    raise ValueError(f"unknown method {spec}")
