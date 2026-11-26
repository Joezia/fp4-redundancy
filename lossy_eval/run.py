"""Evaluate one or more lossy methods against the original model.

  PYTHONPATH=. python -m lossy_eval.run --model gpt-oss-20b --methods identity white_noise:-10 \
      --suite wikitext2 c4 arc_easy arc_challenge hellaswag piqa winogrande mmlu --out <dir>

Writes <out>/<method>.json (summary: A weight error, B layer stats, C KL/top-1/PPL, D accuracy
and flips) and <out>/<method>.npz (per scored position KL/top-1, per request log-likelihoods).
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch

from . import tasks as T
from .engine import Engine
from .methods import build

PPL_SETS = {"wikitext2", "c4"}


def build_inputs(tok, suite, seqlen, ppl_windows, mc_limit):
    seqs, meta = [], {}
    for name in suite:
        start = len(seqs)
        if name in PPL_SETS:
            s = T.ppl_windows(tok, name, seqlen, ppl_windows)
            seqs += s
            meta[name] = dict(kind="ppl", seqs=(start, len(seqs)))
        else:
            items = T.mc_items(name, mc_limit)
            s, reqs = T.mc_sequences(tok, items)
            seqs += s
            meta[name] = dict(kind="mc", seqs=(start, len(seqs)), items=items, reqs=reqs)
    return seqs, meta


def summarize(seqs, meta, res, st, wrows, method):
    out = dict(method=method.name, params=method.params())
    # A: weights
    W = dict(err=sum(r["err"] for r in wrows), pow=sum(r["pow"] for r in wrows), n=sum(r["n"] for r in wrows))
    by_role = {}
    for r in wrows:
        d = by_role.setdefault(r["role"], [0.0, 0.0])
        d[0] += r["err"]; d[1] += r["pow"]
    db = lambda e, p: (float(10 * np.log10(e / p)) if e > 0 else None)
    out["A_weights"] = dict(nmse_db=db(W["err"], W["pow"]), n_weights=W["n"],
                            by_role={k: db(*v) for k, v in by_role.items()},
                            matrix_nmse_db_median=(float(np.median([db(r["err"], r["pow"]) for r in wrows]))
                                                   if W["err"] > 0 else None))
    bpw = [r.get("bits_per_weight") for r in wrows if "bits_per_weight" in r]
    if bpw:
        out["A_weights"]["bits_per_weight"] = float(np.mean(bpw))
    out["B_layers"] = st.as_dict()
    # C / D
    C, D = {}, {}
    pos_seq = res["seq"]
    for name, m in meta.items():
        a, b = m["seqs"]
        sel = (pos_seq >= a) & (pos_seq < b)
        kl, ag = res["kl"][sel], res["agree"][sel]
        entry = dict(kl_mean=float(kl.mean()), kl_p99=float(np.quantile(kl, 0.99)), top1_agree=float(ag.mean()),
                     positions=int(sel.sum()))
        if m["kind"] == "ppl":
            idx = np.where(sel)[0]
            nll_r = -np.array([res["lp_ref"][i][0] for i in idx], dtype=np.float64)
            nll_p = -np.array([res["lp_pert"][i][0] for i in idx], dtype=np.float64)
            entry.update(ppl_ref=float(np.exp(nll_r.mean())), ppl_pert=float(np.exp(nll_p.mean())),
                         windows=b - a)
            C[name] = entry
        else:
            reqs, items = m["reqs"], m["items"]
            ll_r, ll_p = np.zeros(len(reqs)), np.zeros(len(reqs))
            for i in range(a, b):
                s = seqs[i]
                o = int(np.searchsorted(pos_seq, i))  # positions of seq i are contiguous, in order
                for q, rids in enumerate(s.req):
                    for c, r in enumerate(rids):
                        ll_r[r] += res["lp_ref"][o + q][c]
                        ll_p[r] += res["lp_pert"][o + q][c]
            sr, sp = T.mc_scores(items, reqs, ll_r), T.mc_scores(items, reqs, ll_p)
            for key, metric in (("pred", "acc"), ("pred_norm", "acc_norm")):
                cr, cp = sr[key] == sr["label"], sp[key] == sp["label"]
                entry[f"{metric}_ref"] = float(cr.mean())
                entry[f"{metric}_pert"] = float(cp.mean())
                entry[f"{metric}_flip_right_to_wrong"] = int((cr & ~cp).sum())
                entry[f"{metric}_flip_wrong_to_right"] = int((~cr & cp).sum())
                entry[f"{metric}_answer_changed"] = int((sr[key] != sp[key]).sum())
            entry["n_items"] = len(items)
            C[name] = {k: entry[k] for k in ("kl_mean", "kl_p99", "top1_agree", "positions")}
            D[name] = {k: v for k, v in entry.items() if k not in C[name]}
            m["_ll"] = (ll_r, ll_p)
    out["C_output"] = C
    out["D_tasks"] = D
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-oss-20b")
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--suite", nargs="+", default=["wikitext2"])
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--ppl-windows", type=int, default=None)
    ap.add_argument("--mc-limit", type=int, default=None)
    ap.add_argument("--chunk-tokens", type=int, default=120_000)
    ap.add_argument("--tok-budget", type=int, default=16384)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)
    if args.model == "gpt-oss-20b":
        from .gptoss import GptOss
        A = GptOss()
    else:
        raise ValueError(args.model)
    log(f"model loaded; GPU mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    seqs, meta = build_inputs(A.tok, args.suite, args.seqlen, args.ppl_windows, args.mc_limit)
    ntok = sum(len(s.tokens) for s in seqs)
    log(f"{len(seqs)} sequences, {ntok} tokens, {sum(len(s.score_pos) for s in seqs)} scored positions")
    for spec in args.methods:
        method = build(spec)
        t = time.time()
        res, st, wrows = Engine(A, method, args.chunk_tokens, args.tok_budget, log=log).run(seqs)
        summ = summarize(seqs, meta, res, st, wrows, method)
        summ.update(model=A.name, spec=spec, suite=args.suite, seqlen=args.seqlen, ppl_windows=args.ppl_windows,
                    mc_limit=args.mc_limit, tokens=ntok, seconds=time.time() - t)
        tag = spec.replace(":", "_")
        (out / f"{tag}.json").write_text(json.dumps(summ, indent=1))
        lls = {f"ll_{k}_{w}": v["_ll"][j] for k, v in meta.items() if "_ll" in v for j, w in enumerate(("ref", "pert"))}
        np.savez_compressed(out / f"{tag}.npz", kl=res["kl"].astype(np.float32), agree=res["agree"], seq=res["seq"], **lls)
        A_ = summ["A_weights"]
        line = f"{spec}: weight NMSE {A_['nmse_db']} dB; " + "; ".join(
            f"{k} KL {v['kl_mean']:.4f} top1 {v['top1_agree']:.4f}" + (f" ppl {v['ppl_ref']:.3f}->{v['ppl_pert']:.3f}" if "ppl_ref" in v else "")
            for k, v in summ["C_output"].items())
        line += "; " + "; ".join(f"{k} acc {v['acc_ref']:.4f}->{v['acc_pert']:.4f}" for k, v in summ["D_tasks"].items())
        log(line + f" ({summ['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
