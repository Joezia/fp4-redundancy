"""Correctness checks for the framework (gpt-oss-20b).

  PYTHONPATH=. python -m lossy_eval.selfcheck [--hf-cpu]

1. MXFP4 dequant == transformers convert_moe_packed_tensors (one layer, all experts).
2. White-noise method hits the requested weight NMSE; block RTN reconstruction error is plausible.
3. Identity method: perturbed stream bit-identical to the original (KL = 0, no flips, zero layer error).
4. Right padding does not change a sequence's log-likelihoods (alone vs batched with a longer one).
5. (--hf-cpu) Engine original-stream logits vs transformers' own GptOssForCausalLM run on CPU
   with dequantized experts, two sequences (one longer than the 128-token sliding window).
"""
import argparse, json
import numpy as np
import torch

from .gptoss import GptOss
from .engine import Engine
from .methods import Identity, WhiteNoise, BlockRTN, MatrixCtx
from .mxfp4 import dequant
from . import tasks as T

TEXTS = ["The capital of France is Paris. It is known for the Eiffel Tower and",
         " ".join(["In 1905 Albert Einstein published four papers that changed physics."] * 3) +
         " The first paper explained the photoelectric effect, the second Brownian motion, the third special relativity,"
         " and the fourth the equivalence of mass and energy. Together they are known as the annus mirabilis papers,"
         " and historians still debate how a patent clerk produced them in a single year while working full time."]


def seqs_of(tok, texts):
    out = []
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        out.append(T.Seq(tokens=ids, score_pos=list(range(len(ids) - 1)), targets=[[x] for x in ids[1:]],
                         group="check", req=[[-1]] * (len(ids) - 1)))
    return out


def attn_check(A, layers=(0, 1, 10, 11), T=2048):
    torch.manual_seed(0)
    out = {}
    with torch.no_grad():
        for l in layers:
            h = torch.randn(1, T, A.cfg.hidden_size, device=A.dev, dtype=torch.bfloat16) * 0.5
            L, at = A.layers[l], A.layers[l].self_attn
            m = A._masks(T)
            pe = A.model.model.rotary_emb(h, m["pos"])
            x = L.input_layernorm(h)
            q = at.q_proj(x).view(1, T, -1, at.head_dim).transpose(1, 2)
            k = at.k_proj(x).view(1, T, -1, at.head_dim).transpose(1, 2)
            v = at.v_proj(x).view(1, T, -1, at.head_dim).transpose(1, 2)
            q, k = A.M.apply_rotary_pos_emb(q, k, *pe)
            g = at.num_key_value_groups
            q, k, v = q.double(), k.repeat_interleave(g, 1).double(), v.repeat_interleave(g, 1).double()
            sc = (q @ k.transpose(-1, -2)) * at.scaling
            i = torch.arange(T, device=A.dev)
            d = i[:, None] - i[None, :]
            ok = (d >= 0) if A.cfg.layer_types[l] == "full_attention" else (d >= 0) & (d < A.cfg.sliding_window)
            sc = sc.masked_fill(~ok, float("-inf"))
            sink = at.sinks.double()[None, :, None, None].expand(1, -1, T, 1)
            p = torch.softmax(torch.cat([sc, sink], -1), -1)[..., :-1]
            ref = at.o_proj((p @ v).to(torch.bfloat16).transpose(1, 2).reshape(1, T, -1)).float()
            row = {}
            for impl in ("eager", "sdpa"):
                A.attn_impl = impl
                a = (A.attn(l, h) - h).float()
                row[impl] = float((a - ref).norm() / ref.norm())
            A.attn_impl = "sdpa"
            out[f"layer{l}_{A.cfg.layer_types[l]}"] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-cpu", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rep = {}
    A = GptOss()
    # 1 dequant
    from transformers.integrations.mxfp4 import convert_moe_packed_tensors
    ok = True
    ex = A.load_layer(0)
    for key in ("gate_up_proj", "down_proj"):
        for e in range(A.n_experts):  # one expert at a time (transformers' routine needs large temporaries)
            b, s = ex[f"{key}_blocks"][e:e + 1], ex[f"{key}_scales"][e:e + 1]
            ref = convert_moe_packed_tensors(b, s, dtype=torch.bfloat16).transpose(1, 2)  # back to [E, out, in]
            ok &= bool(torch.equal(ref, dequant(b, s, torch.bfloat16)))
            exact = dequant(b, s, torch.float32)
            ok &= bool(torch.equal(exact.to(torch.bfloat16).float(), exact))  # bf16 holds MXFP4 exactly
    rep["dequant_equal_transformers"] = ok
    print("1 dequant:", ok, flush=True)
    # 2 methods on one matrix
    e, role, blk, sc = next(A.matrices(3))
    W = dequant(blk, sc).float()
    ctx = MatrixCtx(A.name, 3, e, role, W, None, sc)
    for db in (-20, -10, -2.4):
        Wh, _ = WhiteNoise(db).reconstruct(ctx)
        got = float(10 * torch.log10((Wh - W).pow(2).sum() / W.pow(2).sum()))
        rep[f"white_noise_{db}"] = got
        print(f"2 white noise target {db} dB -> {got:.3f} dB", flush=True)
    for k in (4, 3, 2):
        Wh, info = BlockRTN(k).reconstruct(ctx)
        got = float(10 * torch.log10((Wh - W).pow(2).sum() / W.pow(2).sum()))
        rep[f"block_rtn_{k}"] = got
        print(f"2 block RTN {k} bit -> {got:.2f} dB ({info['bits_per_weight']} bpw)", flush=True)
    # 2b attention implementations vs a float64 reference on the same bf16 q/k/v
    rep["attention_rel_err_vs_float64"] = attn_check(A)
    print("2b attention rel. error vs float64:", rep["attention_rel_err_vs_float64"], flush=True)
    # 3 identity + 4 padding
    seqs = seqs_of(A.tok, TEXTS)
    res, st, wrows = Engine(A, Identity()).run(seqs)
    lr = np.concatenate([x for x in res["lp_ref"]])
    lp = np.concatenate([x for x in res["lp_pert"]])
    d = st.as_dict()
    rep["identity_kl_max"] = float(res["kl"].max())
    rep["identity_top1"] = float(res["agree"].mean())
    rep["identity_logp_equal"] = bool(np.array_equal(lr, lp))
    rep["identity_weight_err"] = float(sum(r["err"] for r in wrows))
    rep["identity_layer_err"] = float(st.local_err.sum() + st.moe_err.sum() + st.h_err.sum() + st.route_diff.sum())
    print("3 identity:", {k: rep[k] for k in rep if k.startswith("identity")}, flush=True)
    res1, _, _ = Engine(A, Identity()).run(seqs[:1])
    a = res1["lp_ref"][: len(seqs[0].score_pos)]
    b = res["lp_ref"][: len(seqs[0].score_pos)]
    diff = float(np.max(np.abs(np.concatenate(a) - np.concatenate(b))))
    rep["padding_max_abs_logp_diff"] = diff
    print("4 padding: max |dlogp| alone vs padded batch =", diff, flush=True)
    # 5 transformers reference on CPU
    if args.hf_cpu:
        from transformers import AutoModelForCausalLM, Mxfp4Config
        del A
        torch.cuda.empty_cache()
        torch.set_num_threads(32)
        m = AutoModelForCausalLM.from_pretrained(GptOss.__init__.__defaults__[0] or str(T.DATA.parent / "models" / "gpt-oss-20b"),
                                                 dtype=torch.bfloat16, device_map="cpu",
                                                 quantization_config=Mxfp4Config(dequantize=True),
                                                 attn_implementation="eager").eval()
        kls, ags, mx = [], [], []
        off = 0
        for s in seqs:
            with torch.no_grad():
                lg = m(torch.tensor([s.tokens])).logits[0, :-1].float()
            hf = torch.log_softmax(lg, -1)
            gl = np.array([x[0] for x in res["lp_ref"][off: off + len(s.score_pos)]])
            hf_t = hf.gather(1, torch.tensor(s.tokens[1:])[:, None])[:, 0].numpy()
            mx.append(float(np.max(np.abs(gl - hf_t))))
            off += len(s.score_pos)
        rep["hf_cpu_max_abs_target_logp_diff"] = mx
        print("5 transformers CPU vs engine: max |dlogp(target)| per sequence =", mx, flush=True)
    if args.out:
        open(args.out, "w").write(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
