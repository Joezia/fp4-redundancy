"""gpt-oss adapter for the layer-sequential engine.

Uses the transformers GptOss modules (attention with sinks, RMSNorm, rotary, router) and
replaces only the expert computation: experts stay packed MXFP4 on the GPU and are
dequantized to bf16 per expert when used (A6000 has no FP4 tensor cores), or come from a
dense W_hat for the perturbed stream. Expert math matches GptOssExperts in transformers.
"""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors import safe_open

from .mxfp4 import dequant, unpack_codes

ROOT = Path(__file__).resolve().parents[1] / "data" / "lossy_eval" / "models"


class GptOss:
    name = "gpt-oss-20b"
    roles = ("gate_up", "down")

    def __init__(self, path=None, device="cuda", attn_impl="sdpa"):
        from transformers import AutoConfig, AutoTokenizer
        from transformers.models.gpt_oss import modeling_gpt_oss as M
        self.M = M
        self.path = Path(path or ROOT / "gpt-oss-20b")
        self.dev = torch.device(device)
        cfg = AutoConfig.from_pretrained(self.path)
        cfg._attn_implementation = "eager"
        if hasattr(cfg, "quantization_config"):
            del cfg.quantization_config
        self.cfg = cfg
        self.tok = AutoTokenizer.from_pretrained(self.path)
        self.n_layers, self.n_experts, self.top_k = cfg.num_hidden_layers, cfg.num_local_experts, cfg.num_experts_per_tok
        self.limit, self.alpha = cfg.swiglu_limit, 1.702
        with torch.device("meta"):
            model = M.GptOssForCausalLM(cfg)
        idx = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]
        dense, self.experts = {}, [dict() for _ in range(self.n_layers)]
        by_file = {}
        for k, f in idx.items():
            by_file.setdefault(f, []).append(k)
        for f, keys in by_file.items():
            with safe_open(self.path / f, framework="pt", device="cpu") as fh:
                for k in keys:
                    t = fh.get_tensor(k)
                    if ".mlp.experts." in k:
                        l = int(k.split(".")[2])
                        self.experts[l][k.split(".mlp.experts.")[1]] = t.pin_memory()  # streamed per layer
                    else:
                        dense[k] = t
        missing, unexpected = model.load_state_dict(dense, strict=False, assign=True)
        missing = [m for m in missing if ".mlp.experts." not in m]
        assert not missing and not unexpected, (missing, unexpected)
        model.model.rotary_emb = M.GptOssRotaryEmbedding(config=cfg).to(self.dev)
        for n, p in model.named_parameters():
            if ".mlp.experts." not in n:
                p.data = p.data.to(self.dev)
        self.model = model.eval()
        self.layers = model.model.layers
        self._mask_cache = {}
        self._gpu_l, self._gpu = None, None
        self.attn_impl = attn_impl

    # ------------------------------------------------------------ per-layer expert residency
    def load_layer(self, l):
        if self._gpu_l != l:
            self._gpu = {k: v.to(self.dev, non_blocking=True) for k, v in self.experts[l].items()}
            self._gpu_l = l
        return self._gpu

    def release_layer(self):
        self._gpu_l, self._gpu = None, None

    # ------------------------------------------------------------ stream pieces
    def embed(self, ids):
        return self.model.model.embed_tokens(ids)

    def _masks(self, T):
        if T not in self._mask_cache:
            i = torch.arange(T, device=self.dev)
            d = i[:, None] - i[None, :]
            neg = torch.finfo(torch.bfloat16).min
            full = torch.where(d >= 0, 0.0, neg).to(torch.bfloat16)[None, None]
            slide = torch.where((d >= 0) & (d < self.cfg.sliding_window), 0.0, neg).to(torch.bfloat16)[None, None]
            pos = i[None]
            Tp = (T + 7) // 8 * 8  # efficient-attention bias needs an aligned last dim
            sb = torch.full((T, Tp), float("-inf"), device=self.dev, dtype=torch.bfloat16)
            sb[:, :T] = torch.where((d >= 0) & (d < self.cfg.sliding_window), 0.0, float("-inf"))
            self._mask_cache[T] = {"full_attention": full, "sliding_attention": slide, "pos": pos,
                                   "sliding_bias": sb[None, None, :, :T]}
            if len(self._mask_cache) > 64:
                self._mask_cache.pop(next(iter(self._mask_cache)))
        return self._mask_cache[T]

    def attn(self, l, h):
        """Residual stream h [B,T,H] -> h + attention(norm(h)).
        'eager': transformers' GptOssAttention (bf16 softmax over [scores, sink]).
        'sdpa' : same projections/rotary; PyTorch memory-efficient attention returns the
                 log-sum-exp, and the sink is applied exactly: out * sigmoid(lse - sink)."""
        L = self.layers[l]
        m = self._masks(h.shape[1])
        pe = self.model.model.rotary_emb(h, m["pos"])
        x = L.input_layernorm(h)
        if self.attn_impl == "eager":
            a, _ = L.self_attn(hidden_states=x, position_embeddings=pe, attention_mask=m[self.cfg.layer_types[l]])
            return h + a
        at = L.self_attn
        B, T, _ = x.shape
        q = at.q_proj(x).view(B, T, -1, at.head_dim).transpose(1, 2)
        k = at.k_proj(x).view(B, T, -1, at.head_dim).transpose(1, 2)
        v = at.v_proj(x).view(B, T, -1, at.head_dim).transpose(1, 2)
        q, k = self.M.apply_rotary_pos_emb(q, k, *pe)
        g = at.num_key_value_groups
        k, v = k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1)
        if self.cfg.layer_types[l] == "full_attention":
            out, lse = torch.ops.aten._scaled_dot_product_efficient_attention(
                q, k, v, None, True, 0.0, True, scale=at.scaling)[:2]
        else:
            bias = m["sliding_bias"].expand(B, q.shape[1], T, T)
            out, lse = torch.ops.aten._scaled_dot_product_efficient_attention(
                q, k, v, bias, True, 0.0, False, scale=at.scaling)[:2]
        lse = lse[:, :, :T]
        out = out * torch.sigmoid(lse - at.sinks.float()[None, :, None]).to(out.dtype)[..., None]
        a = at.o_proj(out.transpose(1, 2).reshape(B, T, -1))
        return h + a

    def moe_in(self, l, h):
        return self.layers[l].post_attention_layernorm(h)

    def route(self, l, x):
        r = self.layers[l].mlp.router
        logits = F.linear(x, r.weight, r.bias)
        val, idx = torch.topk(logits, self.top_k, dim=-1)
        return idx, torch.softmax(val, dim=1, dtype=val.dtype)

    # ------------------------------------------------------------ expert weights
    def matrices(self, l):
        """Yield (expert, role, blocks, scales) in storage orientation [out, in]."""
        ex = self.load_layer(l)
        for e in range(self.n_experts):
            for role, key in (("gate_up", "gate_up_proj"), ("down", "down_proj")):
                yield e, role, ex[f"{key}_blocks"][e], ex[f"{key}_scales"][e]

    def orig_weight(self, l, e, role):
        key = "gate_up_proj" if role == "gate_up" else "down_proj"
        ex = self.load_layer(l)
        return dequant(ex[f"{key}_blocks"][e], ex[f"{key}_scales"][e], torch.bfloat16)

    def expert_fn(self, l, e, x, Wgu, Wd):
        ex = self.load_layer(l)
        gu = x @ Wgu.T + ex["gate_up_proj_bias"][e]
        gate, up = gu[..., ::2], gu[..., 1::2]
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        g = (up + 1) * (gate * torch.sigmoid(gate * self.alpha))
        return g @ Wd.T + ex["down_proj_bias"][e]

    def final_logits(self, h):
        return self.model.lm_head(self.model.model.norm(h)).float()

    def codes_scales(self, blocks, scales):
        return unpack_codes(blocks), scales
