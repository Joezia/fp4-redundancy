"""Evaluation inputs: perplexity windows and multiple-choice log-likelihood requests.

Prompt formats and scoring follow lm-evaluation-harness task definitions (arc_easy,
arc_challenge, hellaswag, piqa, winogrande, mmlu 0-shot): acc = argmax of summed
log-likelihood of the continuation; acc_norm divides by the continuation's UTF-8 byte length.
"""
import gzip, json, re
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data" / "lossy_eval" / "datasets"


def _parquet(pattern):
    import pandas as pd
    files = sorted(DATA.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


# ---------------------------------------------------------------- sequences


@dataclass
class Seq:
    tokens: list               # token ids fed to the model
    score_pos: list            # positions p whose logits predict a target
    targets: list              # per position: list of candidate token ids (len 1 for PPL/most MC)
    group: str = ""            # dataset / task name
    req: list = field(default_factory=list)  # per position: request id(s) the target belongs to


def ppl_windows(tok, name: str, seqlen: int = 2048, max_windows: int | None = None):
    if name == "wikitext2":
        df = _parquet("wikitext/wikitext-2-raw-v1/test-*.parquet")
        text = "\n\n".join(df["text"].tolist())
    elif name == "c4":
        docs = []
        with gzip.open(next(DATA.glob("c4/en/c4-validation.00000-of-00008.json.gz")), "rt") as f:
            for line in f:
                docs.append(json.loads(line)["text"])
                if len(docs) >= 3000:
                    break
        text = "\n\n".join(docs)
    else:
        raise ValueError(name)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    n = len(ids) // seqlen
    if max_windows:
        n = min(n, max_windows)
    out = []
    for i in range(n):
        w = ids[i * seqlen:(i + 1) * seqlen]
        out.append(Seq(tokens=w, score_pos=list(range(seqlen - 1)), targets=[[t] for t in w[1:]],
                       group=name, req=[[-1]] * (seqlen - 1)))
    return out


# ---------------------------------------------------------------- multiple choice


@dataclass
class MCItem:
    task: str
    doc: int
    context: list            # one context per choice (winogrande) or a single shared one
    continuations: list      # one per choice
    label: int


def _hs_pre(text):
    text = text.strip().replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    return text.replace("  ", " ")


def mc_items(task: str, limit: int | None = None):
    items = []
    if task in ("arc_easy", "arc_challenge"):
        sub = "ARC-Easy" if task == "arc_easy" else "ARC-Challenge"
        df = _parquet(f"arc/{sub}/test-*.parquet")
        for i, r in df.iterrows():
            labels = list(r["choices"]["label"])
            items.append(MCItem(task, i, [f"Question: {r['question']}\nAnswer:"],
                                [" " + c for c in r["choices"]["text"]], labels.index(r["answerKey"])))
    elif task == "hellaswag":
        df = _hellaswag_val()
        for i, r in df.iterrows():
            ctx = r["ctx_a"] + " " + r["ctx_b"].capitalize()
            q = _hs_pre(r["activity_label"] + ": " + ctx)
            items.append(MCItem(task, i, [q], [" " + _hs_pre(e) for e in r["endings"]], int(r["label"])))
    elif task == "piqa":
        df = _piqa_val()
        for i, r in df.iterrows():
            items.append(MCItem(task, i, [f"Question: {r['goal']}\nAnswer:"],
                                [" " + r["sol1"], " " + r["sol2"]], int(r["label"])))
    elif task == "winogrande":
        df = _parquet("winogrande/winogrande_xl/validation-*.parquet")
        for i, r in df.iterrows():
            s = r["sentence"]
            idx = s.index("_")
            cont = " " + s[idx + 1:].strip()
            items.append(MCItem(task, i, [s[:idx] + r["option1"], s[:idx] + r["option2"]],
                                [cont, cont], int(r["answer"]) - 1))
    elif task == "mmlu":
        df = _parquet("mmlu/all/test-*.parquet")
        for i, r in df.iterrows():
            subj = r["subject"].replace("_", " ")
            ch = list(r["choices"])
            prompt = (f"The following are multiple choice questions (with answers) about {subj}.\n\n"
                      f"{r['question'].strip()}\nA. {ch[0]}\nB. {ch[1]}\nC. {ch[2]}\nD. {ch[3]}\nAnswer:")
            items.append(MCItem(task, i, [prompt], [" A", " B", " C", " D"], int(r["answer"])))
    else:
        raise ValueError(task)
    if limit and len(items) > limit:
        rng = np.random.default_rng(20261126)
        keep = sorted(rng.choice(len(items), limit, replace=False))
        items = [items[k] for k in keep]
    return items


def _hellaswag_val():
    try:
        return _parquet("hellaswag/**/validation-*.parquet")
    except FileNotFoundError:
        import pandas as pd
        f = next(DATA.glob("hellaswag/**/*val*.jsonl"))
        return pd.read_json(f, lines=True)


def _piqa_val():
    try:
        return _parquet("piqa/**/validation-*.parquet")
    except FileNotFoundError:
        import pandas as pd
        d = next(DATA.glob("piqa/**/dev.jsonl")).parent
        df = pd.read_json(d / "dev.jsonl", lines=True)
        df["label"] = [int(x) for x in (d / "dev-labels.lst").read_text().split()]
        return df


def encode_pair(tok, context: str, continuation: str):
    """lm-eval convention: trailing spaces of the context move to the continuation;
    continuation tokens = enc(context + continuation)[len(enc(context)):]."""
    n_sp = len(context) - len(context.rstrip())
    if n_sp:
        continuation = context[-n_sp:] + continuation
        context = context[:-n_sp]
    whole = tok(context + continuation, add_special_tokens=False)["input_ids"]
    ctx = tok(context, add_special_tokens=False)["input_ids"]
    return whole[:len(ctx)], whole[len(ctx):]


def mc_sequences(tok, items):
    """Turn MC items into scored sequences. Requests whose choices share the context and
    each choice is one token are merged into a single sequence (e.g. MMLU letters)."""
    seqs, reqs = [], []          # reqs[r] = (item index, choice index, n_tokens, n_bytes)
    for k, it in enumerate(items):
        pairs = [encode_pair(tok, it.context[min(c, len(it.context) - 1)], it.continuations[c])
                 for c in range(len(it.continuations))]
        same_ctx = len(it.context) == 1 and all(p[0] == pairs[0][0] for p in pairs)
        if same_ctx and all(len(p[1]) == 1 for p in pairs):
            rid = []
            for c, (ctx, cont) in enumerate(pairs):
                rid.append(len(reqs))
                reqs.append((k, c, 1, len(it.continuations[c].encode())))
            ctx = pairs[0][0]
            seqs.append(Seq(tokens=ctx, score_pos=[len(ctx) - 1], targets=[[p[1][0] for p in pairs]],
                            group=it.task, req=[rid]))
            continue
        for c, (ctx, cont) in enumerate(pairs):
            r = len(reqs)
            reqs.append((k, c, len(cont), len(it.continuations[c].encode())))
            toks = (ctx + cont)
            start = len(ctx) - 1
            seqs.append(Seq(tokens=toks, score_pos=list(range(start, start + len(cont))),
                            targets=[[t] for t in cont], group=it.task, req=[[r]] * len(cont)))
    return seqs, reqs


def mc_scores(items, reqs, req_logp):
    """req_logp[r] = summed log-likelihood. Returns per-item predictions for acc and acc_norm."""
    n = len(items)
    ll = [np.full(len(it.continuations), -np.inf) for it in items]
    lln = [np.full(len(it.continuations), -np.inf) for it in items]
    for r, (k, c, ntok, nbytes) in enumerate(reqs):
        ll[k][c] = req_logp[r]
        lln[k][c] = req_logp[r] / max(nbytes, 1)
    pred = np.array([int(np.argmax(x)) for x in ll])
    pred_n = np.array([int(np.argmax(x)) for x in lln])
    label = np.array([it.label for it in items])
    return dict(pred=pred, pred_norm=pred_n, label=label, n=n)
