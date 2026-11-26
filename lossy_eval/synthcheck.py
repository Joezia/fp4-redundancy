"""Synthetic check of the layer-sequential engine on a tiny random gpt-oss model.

Builds a 4-layer gpt-oss (hidden 64, 4 experts, top-2, sliding window 8) with random weights,
quantizes the experts to MXFP4 and writes a checkpoint in the official layout. The engine's
original stream must match transformers' GptOssForCausalLM (experts = dequantized MXFP4) run
as one ordinary forward per sequence. Needs only the gpt-oss tokenizer files.

  PYTHONPATH=. python -m lossy_eval.synthcheck <workdir>
"""
import json, shutil, sys
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file

from .mxfp4 import E2M1, dequant
from .methods import Identity, WhiteNoise
from .engine import Engine
from . import tasks as T

SRC = Path(__file__).resolve().parents[1] / "data" / "lossy_eval" / "models" / "gpt-oss-20b"


def quant_mxfp4(W):
    """W float32 [..., out, in] -> blocks uint8 [..., out, in/32, 16], scales uint8 [..., out, in/32]."""
    *p, o, i = W.shape
    x = W.reshape(*p, o, i // 32, 32)
    amax = x.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    e = torch.ceil(torch.log2(amax / 6.0))
    v = x / torch.exp2(e)
    grid = torch.tensor(E2M1[:8])
    code = (v.abs().unsqueeze(-1) - grid).abs().argmin(-1)
    code = code + 8 * ((v < 0) & (code > 0)).long()
    code = code.to(torch.uint8).reshape(*p, o, i // 32, 16, 2)
    blocks = code[..., 0] | (code[..., 1] << 4)
    scales = (e[..., 0] + 127).to(torch.uint8)
    return blocks, scales


def main():
    work = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/lossy_eval_synth")
    work.mkdir(parents=True, exist_ok=True)
    from transformers import AutoConfig, GptOssForCausalLM
    cfg = AutoConfig.from_pretrained(SRC)
    del cfg.quantization_config
    cfg.update(dict(hidden_size=64, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                    num_key_value_heads=2, head_dim=16, num_local_experts=4, num_experts_per_tok=2,
                    experts_per_token=2, sliding_window=8,
                    layer_types=["sliding_attention", "full_attention"] * 2))
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    m = GptOssForCausalLM(cfg).to(torch.bfloat16)
    sd = {}
    for k, v in m.state_dict().items():
        if ".mlp.experts." in k:
            continue
        sd[k] = (torch.randn_like(v.float()) * (0.5 if "norm" not in k else 0.1) + (1.0 if "norm" in k else 0)).to(v.dtype) if v.ndim else v
    for k in sd:
        if k.endswith("sinks"):
            sd[k] = torch.randn_like(sd[k])
    m.load_state_dict(sd, strict=False)
    for l, layer in enumerate(m.model.layers):
        ex = layer.mlp.experts
        for name in ("gate_up_proj", "down_proj"):
            W = torch.randn(getattr(ex, name).shape).transpose(1, 2).contiguous() * 0.2  # [E, out, in]
            b, s = quant_mxfp4(W)
            sd[f"model.layers.{l}.mlp.experts.{name}_blocks"] = b
            sd[f"model.layers.{l}.mlp.experts.{name}_scales"] = s
            getattr(ex, name).data = dequant(b, s, torch.float32).transpose(1, 2).to(torch.bfloat16).contiguous()
            bias = torch.randn(getattr(ex, name + "_bias").shape) * 0.1
            sd[f"model.layers.{l}.mlp.experts.{name}_bias"] = bias.to(torch.bfloat16)
            getattr(ex, name + "_bias").data = bias.to(torch.bfloat16)
    sd = {k: v.contiguous() for k, v in sd.items()}
    save_file(sd, work / "model-00000-of-00001.safetensors")
    (work / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": {k: "model-00000-of-00001.safetensors" for k in sd}}))
    d = cfg.to_dict()
    d["quantization_config"] = {"quant_method": "mxfp4"}
    (work / "config.json").write_text(json.dumps(d))
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        shutil.copy(SRC / f, work / f)

    from .gptoss import GptOss
    from transformers import Mxfp4Config
    A = GptOss(work, attn_impl="eager")  # bit-exact comparison uses transformers' eager attention
    # reference: transformers' own loader on the same checkpoint (MXFP4 dequantized to bf16)
    m = GptOssForCausalLM.from_pretrained(work, dtype=torch.bfloat16, device_map=str(A.dev),
                                          quantization_config=Mxfp4Config(dequantize=True),
                                          attn_implementation="eager").eval()
    print("reference rotary inv_freq dtype:", m.model.rotary_emb.inv_freq.dtype)
    rng = np.random.default_rng(1)
    seqs = []
    for n in (5, 9, 23, 40, 17):
        ids = rng.integers(0, 1000, n).tolist()
        seqs.append(T.Seq(tokens=ids, score_pos=list(range(n - 1)), targets=[[x] for x in ids[1:]], req=[[-1]] * (n - 1)))
    quiet = lambda *_: None
    tgt = lambda s: torch.tensor(s.tokens[1:], device=A.dev)[:, None]
    # (a) one sequence per forward: engine vs transformers
    worst_single = 0.0
    for s in seqs:
        r1 = Engine(A, Identity(), log=quiet).run([s])[0]["lp_ref"]
        with torch.no_grad():
            lg = torch.log_softmax(m(torch.tensor([s.tokens], device=A.dev)).logits[0, :-1].float(), -1)
        hf = lg.gather(1, tgt(s))[:, 0].cpu().numpy()
        worst_single = max(worst_single, float(np.abs(hf - np.array([x[0] for x in r1])).max()))
    # (b) all sequences in one right-padded batch: engine vs transformers with attention_mask
    Tm = max(len(s.tokens) for s in seqs)
    ids = torch.full((len(seqs), Tm), A.cfg.pad_token_id, device=A.dev)
    am = torch.zeros((len(seqs), Tm), dtype=torch.long, device=A.dev)
    for i, s in enumerate(seqs):
        ids[i, :len(s.tokens)] = torch.tensor(s.tokens, device=A.dev); am[i, :len(s.tokens)] = 1
    with torch.no_grad():
        lb = torch.log_softmax(m(ids, attention_mask=am).logits.float(), -1)
    res, st, _ = Engine(A, Identity(), tok_budget=10**6, log=quiet).run(seqs)
    worst_batch, worst_hf_pad, off = 0.0, 0.0, 0
    for i, s in enumerate(seqs):
        hf = lb[i, :len(s.tokens) - 1].gather(1, tgt(s))[:, 0].cpu().numpy()
        mine = np.array([x[0] for x in res["lp_ref"][off: off + len(s.score_pos)]])
        worst_batch = max(worst_batch, float(np.abs(hf - mine).max()))
        with torch.no_grad():
            l1 = torch.log_softmax(m(torch.tensor([s.tokens], device=A.dev)).logits[0, :-1].float(), -1)
        worst_hf_pad = max(worst_hf_pad, float(np.abs(hf - l1.gather(1, tgt(s))[:, 0].cpu().numpy()).max()))
        off += len(s.score_pos)
    print("engine vs transformers, single sequences, max |d logp|:", worst_single)
    print("engine vs transformers, padded batch,     max |d logp|:", worst_batch)
    print("(transformers padded batch vs its own single runs, bf16 kernel rounding:", worst_hf_pad, ")")
    worst = max(worst_single, worst_batch)
    print("identity: max KL", float(res["kl"].max()), "layer err", float(st.local_err.sum() + st.h_err.sum()))
    res2, st2, w2 = Engine(A, WhiteNoise(-10)).run(seqs)
    wdb = 10 * np.log10(sum(r["err"] for r in w2) / sum(r["pow"] for r in w2))
    print(f"white noise -10 dB: weight NMSE {wdb:.2f} dB, mean KL {res2['kl'].mean():.4f}, "
          f"local MoE NMSE by layer {np.round(st2.as_dict()['local_moe_nmse_db'], 2).tolist()}")
    ok = worst == 0.0 and float(res["kl"].max()) == 0.0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
