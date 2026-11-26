"""Layer-sequential evaluation: original and perturbed streams advance together.

For each chunk of sequences, every layer is applied to all batches of the chunk before the
next layer. Per layer the engine records
  * local MoE error: same (original-stream) input, original vs perturbed experts, same routing;
  * per-expert local error on the tokens routed to that expert (unweighted expert output);
  * accumulated error: perturbed vs original residual stream after the layer;
  * routing change: tokens whose top-k expert set differs between the two streams.
At the end, logits at scored positions give KL(p_orig || p_pert), top-1 agreement and the
log-likelihood of target tokens under both streams.
"""
import math, time
import numpy as np
import torch

from .methods import MatrixCtx


def plan_batches(seqs, tok_budget=16384, attn_budget=6_000_000):
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i].tokens))
    batches, cur, T = [], [], 0
    for i in order:
        t = len(seqs[i].tokens)
        nT = max(T, t)
        if cur and ((len(cur) + 1) * nT > tok_budget or (len(cur) + 1) * nT * nT > attn_budget):
            batches.append(cur)
            cur, nT = [], t
        cur.append(i)
        T = nT
    if cur:
        batches.append(cur)
    return batches


def plan_chunks(seqs, batches, chunk_tokens):
    chunks, cur, n = [], [], 0
    for b in batches:
        bt = len(b) * max(len(seqs[i].tokens) for i in b)
        if cur and n + bt > chunk_tokens:
            chunks.append(cur)
            cur, n = [], 0
        cur.append(b)
        n += bt
    if cur:
        chunks.append(cur)
    return chunks


def _sq(a, b, rows=32768):
    """sum((a-b)^2), sum(b^2) in float64, computed in row slices to bound temporaries."""
    e = p = 0.0
    for i in range(0, a.shape[0], rows):
        x, y = a[i:i + rows].float(), b[i:i + rows].float()
        e += float((x - y).pow(2).sum()); p += float(y.pow(2).sum())
    return e, p


class LayerStats:
    def __init__(self, n_layers, n_experts):
        z = lambda *s: np.zeros(s, dtype=np.float64)
        self.local_err, self.local_pow = z(n_layers), z(n_layers)
        self.moe_err, self.moe_pow = z(n_layers), z(n_layers)
        self.h_err, self.h_pow, self.h_cos = z(n_layers), z(n_layers), z(n_layers)
        self.route_diff, self.route_overlap, self.ntok = z(n_layers), z(n_layers), z(n_layers)
        self.ex_err, self.ex_pow, self.ex_cnt = z(n_layers, n_experts), z(n_layers, n_experts), z(n_layers, n_experts)

    def as_dict(self):
        db = lambda e, p: (10 * np.log10(np.maximum(e, 1e-300) / np.maximum(p, 1e-300))).tolist()
        n = np.maximum(self.ntok, 1)
        return dict(
            local_moe_nmse_db=db(self.local_err, self.local_pow),
            moe_out_nmse_db=db(self.moe_err, self.moe_pow),
            hidden_nmse_db=db(self.h_err, self.h_pow),
            hidden_cos=(self.h_cos / n).tolist(),
            route_changed_frac=(self.route_diff / n).tolist(),
            route_overlap=(self.route_overlap / n).tolist(),
            tokens=self.ntok.tolist(),
            expert_local_nmse_db=db(self.ex_err, self.ex_pow),
            expert_tokens=self.ex_cnt.tolist(),
        )


class Engine:
    def __init__(self, adapter, method, chunk_tokens=48_000, tok_budget=16384, logit_chunk=512, log=print):
        self.A, self.method = adapter, method
        self.chunk_tokens, self.tok_budget, self.logit_chunk = chunk_tokens, tok_budget, logit_chunk
        self.keep_ref = None  # set to a list to collect (slots, full original-stream log-probs on CPU)
        self.log = log
        self.wstats = None
        self._what_cpu = None
        self._worig = None

    # ------------------------------------------------------------ perturbed weights
    def _build_what(self, l):
        """Dense bf16 W_hat for every (expert, role) of layer l (on GPU) + weight error stats."""
        A, out = self.A, {}
        rows, errs = [], []
        for e, role, blocks, scales in A.matrices(l):
            W = (self._worig[(e, role)] if self._worig else A.orig_weight(l, e, role)).float()
            codes, sc = A.codes_scales(blocks, scales)
            r = self.method.reconstruct(MatrixCtx(A.name, l, e, role, W, codes, sc))
            Wh, info = (r if isinstance(r, tuple) else (r, {}))
            assert Wh.shape == W.shape
            Wh = Wh.to(torch.bfloat16)
            errs.append(torch.stack([(Wh.float() - W).pow(2).sum(), W.pow(2).sum()]))
            rows.append(dict(layer=l, expert=e, role=role, n=W.numel(),
                             **{k: v for k, v in info.items() if isinstance(v, (int, float))}))
            out[(e, role)] = Wh
        ep = torch.stack(errs).double().cpu().numpy()
        for r, (er, pw) in zip(rows, ep):
            r["err"], r["pow"] = float(er), float(pw)
        return out, rows

    def _what(self, l, cache):
        if self._what_cpu is not None and l in self._what_cpu:
            return {k: v.to(self.A.dev, non_blocking=True) for k, v in self._what_cpu[l].items()}
        w, rows = self._build_what(l)
        if self.wstats is not None and l not in self._wdone:
            self.wstats.extend(rows)
            self._wdone.add(l)
        if cache:
            self._what_cpu.setdefault(l, {k: v.cpu() for k, v in w.items()})
        return w

    # ------------------------------------------------------------ MoE
    @staticmethod
    def _dispatch(idx, n_experts):
        """Group (token, slot) pairs by expert with one sort: yields (e, tok, slot)."""
        flat = idx.reshape(-1)
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=n_experts).cpu().tolist()
        k = idx.shape[1]
        tok_all, slot_all = order // k, order % k
        o = 0
        for e, c in enumerate(counts):
            if c:
                yield e, tok_all[o:o + c], slot_all[o:o + c]
            o += c

    def _moe_pair(self, l, x, What, st, ex_stats):
        """x [N,H] original-stream MoE input. Returns y (original experts), y_hat (W_hat), idx."""
        A = self.A
        idx, w = A.route(l, x)
        y = torch.zeros_like(x)
        yh = torch.zeros_like(x)
        exs = torch.zeros(A.n_experts, 3, device=x.device, dtype=torch.float64)
        for e, tok, k in self._dispatch(idx, A.n_experts):
            xe = x[tok]
            ye = A.expert_fn(l, e, xe, A.orig_weight(l, e, "gate_up"), A.orig_weight(l, e, "down"))
            if self.method.identity and What is None:
                yhe = ye
            else:
                yhe = A.expert_fn(l, e, xe, What[(e, "gate_up")], What[(e, "down")])
            if ex_stats:
                exs[e, 0] += (yhe.float() - ye.float()).pow(2).sum()
                exs[e, 1] += ye.float().pow(2).sum()
                exs[e, 2] += len(tok)
            g = w[tok, k, None]
            y.index_add_(0, tok, (ye * g).to(x.dtype))
            yh.index_add_(0, tok, (yhe * g).to(x.dtype))
        if ex_stats:
            exs = exs.cpu().numpy()
            st.ex_err[l] += exs[:, 0]; st.ex_pow[l] += exs[:, 1]; st.ex_cnt[l] += exs[:, 2]
        return y, yh, idx

    def _moe(self, l, x, What):
        A = self.A
        idx, w = A.route(l, x)
        y = torch.zeros_like(x)
        for e, tok, k in self._dispatch(idx, A.n_experts):
            ye = A.expert_fn(l, e, x[tok], What[(e, "gate_up")], What[(e, "down")])
            y.index_add_(0, tok, (ye * w[tok, k, None]).to(x.dtype))
        return y, idx

    # ------------------------------------------------------------ main
    @torch.no_grad()
    def run(self, seqs):
        A = self.A
        L = A.n_layers
        st = LayerStats(L, A.n_experts)
        self.wstats, self._wdone = [], set()
        batches = plan_batches(seqs, self.tok_budget)
        chunks = plan_chunks(seqs, batches, self.chunk_tokens)
        cache = len(chunks) > 1
        self._what_cpu = {} if cache else None
        npos = sum(len(s.score_pos) for s in seqs)
        res = dict(kl=np.zeros(npos), agree=np.zeros(npos, dtype=bool), seq=np.zeros(npos, dtype=np.int64),
                   lp_ref=[None] * npos, lp_pert=[None] * npos)
        offs = np.cumsum([0] + [len(s.score_pos) for s in seqs])
        t0 = time.time()
        for ci, chunk in enumerate(chunks):
            hs = []
            for b in chunk:
                T = max(len(seqs[i].tokens) for i in b)
                ids = torch.full((len(b), T), A.cfg.pad_token_id, dtype=torch.long)
                mask = torch.zeros((len(b), T), dtype=torch.bool)
                for j, i in enumerate(b):
                    ids[j, :len(seqs[i].tokens)] = torch.tensor(seqs[i].tokens)
                    mask[j, :len(seqs[i].tokens)] = True
                ids, mask = ids.to(A.dev), mask.to(A.dev)
                h = A.embed(ids)
                hs.append([h, h.clone(), mask])
            for l in range(L):
                What = self._what(l, cache)
                # attention per padded batch; MoE once over all real tokens of the chunk
                mids, xr, xp = [], [], []
                for hr, hp, mask in hs:
                    mr, mp = A.attn(l, hr), A.attn(l, hp)
                    mids.append((mr, mp))
                    xr.append(A.moe_in(l, mr)[mask]); xp.append(A.moe_in(l, mp)[mask])
                xr, xp = torch.cat(xr), torch.cat(xp)
                y, yl, idx_r = self._moe_pair(l, xr, What, st, True)
                yp, idx_p = self._moe(l, xp, What)
                del xr, xp
                e1, p1 = _sq(yl, y); e2, _ = _sq(yp, y)
                st.local_err[l] += e1; st.local_pow[l] += p1
                st.moe_err[l] += e2; st.moe_pow[l] += p1
                sr, sp = torch.sort(idx_r, 1).values, torch.sort(idx_p, 1).values
                st.route_diff[l] += float((sr != sp).any(1).sum())
                ov = (idx_r[:, :, None] == idx_p[:, None, :]).any(2).float().sum(1) / A.top_k
                st.route_overlap[l] += float(ov.sum())
                st.ntok[l] += y.shape[0]
                o = 0
                for hb, (mr, mp) in zip(hs, mids):
                    mask = hb[2]
                    n = int(mask.sum())
                    mr[mask] += y[o:o + n]
                    mp[mask] += yp[o:o + n]
                    o += n
                    e3, p3 = _sq(mp[mask], mr[mask])
                    st.h_err[l] += e3; st.h_pow[l] += p3
                    st.h_cos[l] += float(torch.nn.functional.cosine_similarity(mr[mask].float(), mp[mask].float(), dim=1).sum())
                    hb[0], hb[1] = mr, mp
                del y, yl, yp, mids
                del What
                self._worig = None
                A.release_layer()
            # logits at scored positions
            for b, (hr, hp, mask) in zip(chunk, hs):
                rows, cols, slot, cands = [], [], [], []
                for j, i in enumerate(b):
                    for q, p in enumerate(seqs[i].score_pos):
                        rows.append(j); cols.append(p); slot.append(offs[i] + q); cands.append(seqs[i].targets[q])
                        res["seq"][offs[i] + q] = i
                rows_t, cols_t = torch.tensor(rows, device=A.dev), torch.tensor(cols, device=A.dev)
                for s0 in range(0, len(rows), self.logit_chunk):
                    sl = slice(s0, s0 + self.logit_chunk)
                    lr = torch.log_softmax(A.final_logits(hr[rows_t[sl], cols_t[sl]]), -1)
                    lp = torch.log_softmax(A.final_logits(hp[rows_t[sl], cols_t[sl]]), -1)
                    if self.keep_ref is not None:
                        self.keep_ref.append((np.asarray(slot[sl]), lr.cpu()))
                    kl = (lr.exp() * (lr - lp)).sum(-1)
                    ag = lr.argmax(-1) == lp.argmax(-1)
                    kl, ag = kl.cpu().numpy(), ag.cpu().numpy()
                    cc = cands[sl]
                    width = max(len(c) for c in cc)
                    ct = torch.tensor([c + [c[0]] * (width - len(c)) for c in cc], device=A.dev)
                    gr = torch.gather(lr, 1, ct).cpu().numpy()
                    gp = torch.gather(lp, 1, ct).cpu().numpy()
                    for q, s in enumerate(slot[sl]):
                        res["kl"][s], res["agree"][s] = kl[q], ag[q]
                        n = len(cc[q])
                        res["lp_ref"][s], res["lp_pert"][s] = gr[q, :n], gp[q, :n]
            del hs
            torch.cuda.empty_cache()
            self.log(f"chunk {ci + 1}/{len(chunks)} done, {time.time() - t0:.0f}s")
        self._what_cpu = None
        return res, st, self.wstats
