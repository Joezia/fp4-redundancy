"""MXFP4 (E2M1 codes + one E8M0 scale per 32 values) helpers.

Layout used by the gpt-oss / DeepSeek / Kimi checkpoints: blocks uint8 [..., G, 16]
(two codes per byte, low nibble first), scales uint8 [..., G]; value = E2M1(code) * 2**(scale-127).
The last two dims (G, 16) unpack to G*32 consecutive elements of one storage row.
"""
import torch

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def unpack_codes(blocks: torch.Tensor) -> torch.Tensor:
    """uint8 [..., G, 16] -> uint8 codes [..., G*32] (low nibble first)."""
    b = blocks.to(torch.uint8)
    out = torch.stack([b & 0x0F, b >> 4], dim=-1)  # [..., G, 16, 2]
    return out.reshape(*b.shape[:-2], b.shape[-2] * 32)


_LUT = {}


def _byte_lut(device):
    """[256, 2] float32: the two E2M1 values (low nibble, high nibble) of every byte."""
    lut = _LUT.get(device)
    if lut is None:
        v = torch.tensor(E2M1, dtype=torch.float32)
        b = torch.arange(256)
        lut = _LUT[device] = torch.stack([v[b & 15], v[b >> 4]], 1).to(device)
    return lut


def dequant(blocks: torch.Tensor, scales: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    """uint8 blocks [..., G, 16], scales [..., G] -> [..., G*32] in `dtype` (exact in float32)."""
    v = _byte_lut(blocks.device)[blocks.long()]                       # [..., G, 16, 2]
    e = torch.exp2(scales.to(torch.float32) - 127)                     # E8M0 -> 2^(s-127), exact
    w = v.reshape(*scales.shape, 32) * e.unsqueeze(-1)
    return w.reshape(*scales.shape[:-1], scales.shape[-1] * 32).to(dtype)
