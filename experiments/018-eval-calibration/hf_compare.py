"""018 add-on: the engine's original-stream output vs transformers' own model, and the numeric floor.

  CUDA_VISIBLE_DEVICES=<free gpu> PYTHONPATH=. python experiments/018-eval-calibration/hf_compare.py [--fp32 2]

Same 2048-token windows (the first N of WikiText-2 test and of C4 validation, as in the main runs) through:
  eng_sdpa       engine, SDPA attention, all windows in one run (MoE over all tokens together; the setting of the main runs)
  eng_sdpa_alone engine, SDPA, each window run on its own (different GEMM shapes -> different bf16 rounding)
  eng_eager      engine, transformers' eager attention, all windows together
  hf_bf16        transformers GptOssForCausalLM.from_pretrained(Mxfp4Config(dequantize=True)), bf16, eager,
                 split over the visible GPUs with device_map (bf16 matmul on this CPU, without AVX512-BF16, is too slow)
  hf_fp32        (optional, --fp32 N: first N windows of each set) the same in float32, a near-exact reference; on the GPUs
                 with the layers that do not fit offloaded to CPU (accelerate), TF32 disabled (the CPU forward ran single-threaded)
For every pair: mean KL over all positions, top-1 agreement, mean/max |d log p(next token)|, and each side's perplexity.
Pairs between two mathematically identical bf16 computations give the floor below which a weight perturbation
cannot be told apart from rounding.
"""
import argparse, gc, json, os, time
from pathlib import Path
import numpy as np
import torch

from lossy_eval.gptoss import GptOss
from lossy_eval.engine import Engine
from lossy_eval.methods import Identity
from lossy_eval import tasks as T

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"


def engine_logp(A, seqs, impl, alone=False):
    A.attn_impl = impl
    out = [None] * len(seqs)
    groups = [[i] for i in range(len(seqs))] if alone else [list(range(len(seqs)))]
    for g in groups:
        E = Engine(A, Identity(), log=lambda *a: None)
        E.keep_ref = []
        E.run([seqs[i] for i in g])
        n = len(seqs[g[0]].score_pos)
        for slots, lp in E.keep_ref:
            for q, s in enumerate(slots):
                i, p = g[s // n], s % n
                if out[i] is None:
                    out[i] = torch.empty(n, lp.shape[1])
                out[i][p] = lp[q]
    return out


def hf_logp(path, seqs, dtype, log, gpu_mem=None, cpu_mem=None):
    from transformers import AutoModelForCausalLM, Mxfp4Config
    if gpu_mem:
        dm, mm = "auto", {i: gpu_mem for i in range(torch.cuda.device_count())}
        if cpu_mem:
            mm["cpu"] = cpu_mem
    else:
        dm, mm = "cpu", None
    m = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, device_map=dm, max_memory=mm,
                                             quantization_config=Mxfp4Config(dequantize=True),
                                             attn_implementation="eager").eval()
    out = []
    for k, s in enumerate(seqs):
        t0 = time.time()
        with torch.no_grad():
            lg = m(torch.tensor([s.tokens], device=m.device)).logits[0, :-1].float().cpu()
        out.append(torch.log_softmax(lg, -1))
        log(f"  hf {dtype} window {k + 1}/{len(seqs)}: {time.time() - t0:.0f}s")
    del m
    gc.collect()
    torch.cuda.empty_cache()
    return out


def compare(a, b, seqs, idx):
    kl, ag, d, nll_a, nll_b = [], [], [], [], []
    for i in idx:
        x, y = a[i], b[i]
        tg = torch.tensor(seqs[i].tokens[1:])[:, None]
        kl.append((x.exp() * (x - y)).sum(-1))
        ag.append(x.argmax(-1) == y.argmax(-1))
        ta, tb = x.gather(1, tg)[:, 0], y.gather(1, tg)[:, 0]
        d.append((ta - tb).abs()); nll_a.append(-ta); nll_b.append(-tb)
    kl, ag, d = torch.cat(kl), torch.cat(ag).float(), torch.cat(d)
    return dict(kl_mean=float(kl.mean()), kl_p99=float(kl.quantile(0.99)), kl_max=float(kl.max()),
                top1_agree=float(ag.mean()), dlogp_mean=float(d.mean()), dlogp_max=float(d.max()),
                ppl_a=float(torch.cat(nll_a).mean().exp()), ppl_b=float(torch.cat(nll_b).mean().exp()), n=int(kl.numel()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=4, help="windows per dataset")
    ap.add_argument("--fp32", type=int, default=0, help="also run transformers in float32 on the first N windows per dataset")
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--gpu-mem", default="23GiB", help="per-GPU budget for transformers bf16 (device_map over visible GPUs)")
    ap.add_argument("--fp32-gpu-mem", default="16GiB", help="per-GPU budget for transformers fp32 (rest offloaded to CPU)")
    ap.add_argument("--cache", default=None, help="directory for per-configuration log-probs, so a rerun skips finished stages")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    logf = open(OUT / "hf_compare.log", "a")

    def log(m):
        m = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    A = GptOss()
    path = str(A.path) if hasattr(A, "path") else str(T.DATA.parent / "models" / "gpt-oss-20b")
    sets = {d: T.ppl_windows(A.tok, d, 2048, args.windows) for d in ("wikitext2", "c4")}
    seqs = sets["wikitext2"] + sets["c4"]
    idx = {"wikitext2": list(range(args.windows)), "c4": list(range(args.windows, 2 * args.windows)),
           "all": list(range(2 * args.windows))}
    R = {}
    cache = Path(args.cache) if args.cache else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)

    def stage(name, fn):
        f = cache / f"{name}.pt" if cache else None
        if f and f.exists():
            R[name] = torch.load(f); log(f"{name} loaded from cache"); return
        t0 = time.time(); R[name] = fn(); log(f"{name} {time.time() - t0:.0f}s")
        if f:
            torch.save(R[name], f)

    stage("eng_sdpa", lambda: engine_logp(A, seqs, "sdpa"))
    stage("eng_sdpa_alone", lambda: engine_logp(A, seqs, "sdpa", alone=True))
    stage("eng_eager", lambda: engine_logp(A, seqs, "eager"))
    del A
    gc.collect()
    torch.cuda.empty_cache()
    stage("hf_bf16", lambda: hf_logp(path, seqs, torch.bfloat16, log, args.gpu_mem))
    pairs = [("hf_bf16", "eng_eager"), ("hf_bf16", "eng_sdpa"), ("eng_eager", "eng_sdpa"), ("eng_sdpa", "eng_sdpa_alone")]
    if args.fp32:
        sub = list(range(args.fp32)) + list(range(args.windows, args.windows + args.fp32))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        def fp32():
            got = hf_logp(path, [seqs[i] for i in sub], torch.float32, log, args.fp32_gpu_mem, "120GiB")
            out = [None] * len(seqs)
            for i, x in zip(sub, got):
                out[i] = x
            return out
        stage("hf_fp32", fp32)
        pairs += [("hf_fp32", "hf_bf16"), ("hf_fp32", "eng_eager"), ("hf_fp32", "eng_sdpa"), ("hf_fp32", "eng_sdpa_alone")]
    rep = dict(windows=args.windows, fp32_windows=args.fp32, pairs={})
    for a, b in pairs:
        for name, ix in idx.items():
            ix = [i for i in ix if R[a][i] is not None and R[b][i] is not None]
            if not ix:
                continue
            r = compare(R[a], R[b], seqs, ix)
            rep["pairs"][f"{a} vs {b} | {name}"] = r
            log(f"{a} vs {b} | {name}: KL {r['kl_mean']:.5f} (p99 {r['kl_p99']:.4f}, max {r['kl_max']:.3f}) top-1 {r['top1_agree']:.4f} "
                f"|dlogp| mean {r['dlogp_mean']:.4f} max {r['dlogp_max']:.3f} ppl {r['ppl_a']:.2f} / {r['ppl_b']:.2f} (n={r['n']})")
    (OUT / "hf_compare.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
