"""TX stage 3 runner (non-linear transforms; GPU training, CPU coding).

    PYTHONPATH=. python experiments/010-transforms/tx/tx_stage3.py synth        # V-a / V-b for E1-E4
    PYTHONPATH=. python experiments/010-transforms/tx/tx_stage3.py train        # E1 / E2 / E3 grids on the training matrices
    PYTHONPATH=. python experiments/010-transforms/tx/tx_stage3.py code         # chosen nets (and q grid) on train / test / T2
    PYTHONPATH=. python experiments/010-transforms/tx/tx_stage3.py e4           # coordinate networks (4 test matrices; shared 8 experts)
    PYTHONPATH=. python experiments/010-transforms/tx/tx_stage3.py summary
Selection uses only the training matrices (validation = 10 % of their blocks held out from training; q chosen by
coding 6 training matrices).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import txcore as T  # noqa: E402
import tx_nonlinear as NL  # noqa: E402
import synth_tx_local as SL  # noqa: E402

OUT = T.RES / "stage3"
GRID = {"E1_flow": [dict(layers=l, h=h) for l in (4, 8) for h in (64, 256)],
        "E2_idf": [dict(layers=l, h=h) for l in (4, 8) for h in (64, 256)],
        "E3_ae": [dict(k=k, h=h) for k in (2, 4, 8) for h in (128, 512)]}
QGRID = {"E1_flow": [0.5, 1.0, 2.0, 4.0], "E1_rbig": [1.0, 2.0, 4.0, 8.0]}
TRAINERS = {"E1_flow": NL.train_flow, "E2_idf": NL.train_idf, "E3_ae": NL.train_ae}


def sel6():
    return T.TRAIN[::4]


def tag(meth, cfg):
    return meth + "_" + "_".join(f"{k}{cfg[k]}" for k in sorted(cfg))


def train():
    torch.set_num_threads(16)
    OUT.mkdir(parents=True, exist_ok=True)
    NL.MODELS.mkdir(parents=True, exist_ok=True)
    log_path = OUT / "train.jsonl"
    done = {r["tag"] for r in T.load_rows(log_path)}
    t0 = time.time()
    B = NL.train_blocks(T.TRAIN)
    Btr, Bva = NL.split_val(B)
    print(f"blocks: train {len(Btr)} val {len(Bva)} ({time.time() - t0:.0f}s)", flush=True)
    for meth, grid in GRID.items():
        for cfg in grid:
            tg = tag(meth, cfg)
            if tg in done:
                continue
            t1 = time.time()
            a = (cfg["layers"], cfg["h"]) if meth != "E3_ae" else (cfg["k"], cfg["h"])
            net, val = TRAINERS[meth](Btr, Bva, *a)
            path = NL.MODELS / f"{tg}.pt"
            torch.save(net.state_dict(), path)
            row = {"tag": tg, "method": meth, "cfg": cfg, "val_bits_per_value": val, "params": NL.nparams(net),
                   "model_file": str(path), "sha256": NL.sha256(path), "train_s": time.time() - t1}
            with open(log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({k: row[k] for k in ("tag", "val_bits_per_value", "params", "train_s")}), flush=True)


def load_net(meth, cfg):
    net = {"E1_flow": lambda: NL.Flow(32, cfg["layers"], cfg["h"]), "E2_idf": lambda: NL.IDF(32, cfg["layers"], cfg["h"]),
           "E3_ae": lambda: NL.AE(cfg["k"], cfg["h"])}[meth]().double()
    net.load_state_dict(torch.load(NL.MODELS / f"{tag(meth, cfg)}.pt"))
    net.eval()
    return net


def code_job(job):
    torch.set_num_threads(4)
    t0 = time.time()
    m = T.Mat.load(job["name"], with_partner=False)
    v = dict(job.get("v", {}))
    if job["method"] == "E1_rbig":
        net = load_rbig(job["cfg"]["K"])
        e = T.evaluate(NL.E1_rbig(m, v, net))
        e["params"] = net.nparams()
    else:
        net = load_net(job["method"], job["cfg"])
        fn = {"E1_flow": NL.E1_flow, "E2_idf": NL.E2_idf, "E3_ae": NL.E3_ae}[job["method"]]
        e = T.evaluate(fn(m, v, net))
        e["params"] = NL.nparams(net)
    e.update({"key": job["key"], "name": job["name"], "method": job["method"], "cfg": job["cfg"], "variant": {**job["cfg"], **v},
              "vname": job["key"].rsplit("|", 1)[0], "n": m.n, "cpu_s": time.time() - t0})
    return e


def rbig_path(K):
    return NL.MODELS / f"E1_rbig_K{K}.npz"


def fit_rbig():
    """RBIG on the same training blocks as the networks; fp16-rounded parameters saved."""
    NL.MODELS.mkdir(parents=True, exist_ok=True)
    B = None
    for K in (1, 2):
        p = rbig_path(K)
        if p.exists():
            continue
        if B is None:
            B, _ = NL.split_val(NL.train_blocks(T.TRAIN))
        g = NL.RBIG(K).fit(B[:2000000]).quantised()
        np.savez(p, K=K, R=np.stack(g.R), kx=np.stack(g.kx), kz=np.stack(g.kz))
        with open(OUT / "train.jsonl", "a") as f:
            f.write(json.dumps({"tag": f"E1_rbig_K{K}", "method": "E1_rbig", "cfg": {"K": K}, "params": g.nparams(),
                                "model_file": str(p), "sha256": NL.sha256(p), "val_bits_per_value": None}) + "\n")


def load_rbig(K):
    z = np.load(rbig_path(K))
    g = NL.RBIG(K)
    g.R, g.kx, g.kz = list(z["R"]), list(z["kx"]), list(z["kz"])
    return g


def best_cfgs():
    rows = T.load_rows(OUT / "train.jsonl")
    out = {}
    for meth in GRID:
        rs = [r for r in rows if r["method"] == meth]
        if meth == "E3_ae":                         # every latent size is reported (spec: 2-8 dims); best h per k
            for k in (2, 4, 8):
                rk = [r for r in rs if r["cfg"]["k"] == k]
                if rk:
                    out[f"E3_ae_k{k}"] = min(rk, key=lambda r: r["val_bits_per_value"])["cfg"]
        elif rs:
            out[meth] = min(rs, key=lambda r: r["val_bits_per_value"])["cfg"]
    return out


def code():
    fit_rbig()
    picks = best_cfgs()
    picks.pop("E1_flow", None)                      # failed its V-b check: not used to code real data (see report)
    picks["E1_rbig"] = None                         # K chosen together with q on the 6 training matrices
    json.dump(picks, open(OUT / "picks.json", "w"), indent=1)
    def cfgs_of(key, cfg):
        return [{"K": 1}, {"K": 2}] if key == "E1_rbig" else [cfg]

    def meth_of(key):
        return "E3_ae" if key.startswith("E3_ae") else key

    def vname_of(key, cfg, v):
        return f"{key}|" + ",".join(f"{a}={b}" for a, b in sorted({**cfg, **v}.items()))

    jobs = []
    for key, cfg in picks.items():
        meth = meth_of(key)
        qs = QGRID.get(meth, [None])
        cands = [(c, {} if q is None else {"q": q}) for c in cfgs_of(key, cfg) for q in qs]
        if len(cands) > 1:
            for c, v in cands:
                for n in sel6():
                    jobs.append({"key": f"{vname_of(key, c, v)}|{n}", "name": n, "method": meth, "cfg": c, "v": v})
    T.run_jobs(code_job, jobs, OUT / "code.jsonl", workers=16)
    rows = T.load_rows(OUT / "code.jsonl")
    final, chosen = [], {}
    for key, cfg in picks.items():
        meth = meth_of(key)
        qs = QGRID.get(meth, [None])
        cands = [(c, {} if q is None else {"q": q}) for c in cfgs_of(key, cfg) for q in qs]

        def tot(cv):
            vn = vname_of(key, *cv)
            rs = [r for r in rows if r["vname"] == vn and r["name"] in sel6()]
            return sum(min(r["C0"], r["C1"]) for r in rs) / max(sum(r["n"] for r in rs), 1) if rs else float("inf")
        c, v = min(cands, key=tot) if len(cands) > 1 else cands[0]
        chosen[key] = {"cfg": c, "v": v}
        for n in T.TRAIN + T.TEST + T.t2_names():
            final.append({"key": f"{vname_of(key, c, v)}|{n}", "name": n, "method": meth, "cfg": c, "v": v})
    json.dump(chosen, open(OUT / "chosen.json", "w"), indent=1)
    T.run_jobs(code_job, final, OUT / "code.jsonl", workers=16)


# ------------------------------------------------------------------ E4

E4_TEST = ["L0.E191.w1", "L13.E127.w2", "L26.E336.w3", "L39.E70.w1"]
E4_SEL = ["L0.E167.w1", "L26.E259.w3"]


def e4():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "e4.jsonl"
    done = {r["key"] for r in T.load_rows(path)}
    torch.set_num_threads(16)

    def one(names, h, steps, split, shared):
        key = f"E4|{'shared' if shared else 'single'}|h{h}|s{steps}|" + "+".join(names)
        if key in done:
            return
        ms = [T.Mat.load(n, with_partner=False) for n in names]
        t0 = time.time()
        net, sd = NL.fit_siren([m.X for m in ms], h, steps, shared=shared)
        npar = NL.nparams(net)
        for e, m in enumerate(ms):
            pred = NL.siren_predict(net, sd, *m.X.shape, e if shared else None)
            ev = T.evaluate(NL.E4_code(m, pred, 16.0 * npar / len(ms)))
            ev.update({"key": key + "|" + m.name, "group": key, "name": m.name, "method": "E4_inr",
                       "variant": {"h": h, "steps": steps, "shared": shared}, "split": split, "params": npar,
                       "n": m.n, "fit_s": time.time() - t0})
            with open(path, "a") as f:
                f.write(json.dumps(ev) + "\n")
        print(key, f"{time.time() - t0:.0f}s", flush=True)

    for h in (32, 128):                                         # choose the width on two training matrices
        for n in E4_SEL:
            one([n], h, 3000, "train", False)
    rows = [r for r in T.load_rows(path) if r["split"] == "train"]
    best_h = min((32, 128), key=lambda h: sum(min(r["C0"], r["C1"]) for r in rows
                                              if r["variant"]["h"] == h))
    for n in E4_TEST:
        one([n], best_h, 3000, "test", False)
    import tx_global as G
    cs = G.cross_sets()
    one(cs[(0, "w1")]["xrep"], best_h, 6000, "xrep", True)
    json.dump({"best_h": best_h}, open(OUT / "e4_pick.json", "w"))


# ------------------------------------------------------------------ synthetic checks

def synth():
    torch.set_num_threads(16)
    rng = np.random.default_rng(0)
    out = {}
    nb = 60000
    A = rng.normal(size=(32, 4))                      # fixed structure shared by training and coding data
    D = rng.normal(size=(4, 32))

    def iid_blocks(n):
        return SL.iid_levels(int(rng.integers(1 << 30)), (n, 32))

    def corr_blocks(n):
        """rank-4 correlated blocks plus full-rank noise (std 1.5)."""
        return np.round(rng.normal(size=(n, 4)) @ A.T * 6 + rng.normal(0, 1.5, size=(n, 32))).astype(np.int64)

    def manifold_blocks(n):
        t = rng.random((n, 2)) * 6.28
        return np.round(np.concatenate([np.sin(t), np.cos(t)], 1) @ D * 8 + rng.normal(0, 0.3, size=(n, 32))).astype(np.int64)

    def coded(meth, net, Bm, v):
        m = T.synth_mat(L=Bm.reshape(-1, 32 * 8))
        fn = {"E1_flow": NL.E1_flow, "E2_idf": NL.E2_idf, "E3_ae": NL.E3_ae}[meth]
        e = T.evaluate(fn(m, v, net))
        raw = T.synth_raw(m, "value")["level"]
        n = m.L.size
        return {"roundtrip": e["roundtrip"], "appear_C0": (raw["C0"] - e["C0"]) / n, "appear_C1": (raw["C1"] - e["C1"]) / n,
                **({"appear_prior": (raw["C0"] - e["prior_bits"]) / n} if "prior_bits" in e else {})}

    for meth, (struct, a, vs) in {"E1_flow": (corr_blocks, (4, 128), [{"q": q} for q in (0.5, 1.0, 2.0)]),
                                  "E2_idf": (corr_blocks, (4, 128), [{}]),
                                  "E3_ae": (manifold_blocks, (2, 128), [{}])}.items():
        res = {}
        for tg, gen in (("V-a", iid_blocks), ("V-b", struct)):
            B = gen(nb)
            Btr, Bva = NL.split_val(B)
            net, val = TRAINERS[meth](Btr, Bva, *a, steps=3000)
            Bc = gen(16384)
            cands = [dict(coded(meth, net, Bc, v), variant=v) for v in vs]
            res[tg] = max(cands, key=lambda r: max(x for k, x in r.items() if k.startswith("appear")))
        best = lambda r: max(x for k, x in r.items() if k.startswith("appear"))
        res["pass_V-a"] = best(res["V-a"]) <= 0.01 and res["V-a"]["roundtrip"]
        res["pass_V-b"] = best(res["V-b"]) >= 0.05 and res["V-b"]["roundtrip"]
        res["pass"] = bool(res["pass_V-a"] and res["pass_V-b"])
        out[meth] = res
        print(meth, json.dumps(res), flush=True)
    res = {}
    for tg, gen in (("V-a", iid_blocks), ("V-b", corr_blocks)):
        g = NL.RBIG(1).fit(gen(nb)).quantised()
        m = T.synth_mat(L=gen(16384).reshape(-1, 256))
        raw = T.synth_raw(m, "value")["level"]
        cands = []
        for q in (1.0, 2.0, 4.0):
            e = T.evaluate(NL.E1_rbig(m, {"q": q}, g))
            cands.append({"roundtrip": e["roundtrip"], "appear_C0": (raw["C0"] - e["C0"]) / m.L.size,
                          "appear_C1": (raw["C1"] - e["C1"]) / m.L.size, "variant": {"K": 1, "q": q}})
        res[tg] = max(cands, key=lambda r: max(r["appear_C0"], r["appear_C1"]))
    res["pass_V-a"] = max(res["V-a"]["appear_C0"], res["V-a"]["appear_C1"]) <= 0.01 and res["V-a"]["roundtrip"]
    res["pass_V-b"] = max(res["V-b"]["appear_C0"], res["V-b"]["appear_C1"]) >= 0.05 and res["V-b"]["roundtrip"]
    res["pass"] = bool(res["pass_V-a"] and res["pass_V-b"])
    out["E1_rbig"] = res
    print("E1_rbig", json.dumps(res), flush=True)
    # E4: smooth matrix (V-b) vs iid levels (V-a)
    res = {}
    for tg, L in (("V-a", SL.iid_levels(0, (256, 1024))), ("V-b", SL.smooth(1, (256, 1024)))):
        m = T.synth_mat(L=L)
        net, sd = NL.fit_siren([m.X], 64, 3000)
        pred = NL.siren_predict(net, sd, *m.X.shape)
        e = T.evaluate(NL.E4_code(m, pred, 16.0 * NL.nparams(net)))
        raw = T.synth_raw(m, "value")["level"]
        n = m.L.size
        res[tg] = {"roundtrip": e["roundtrip"], "appear_C0": (raw["C0"] - e["C0"]) / n, "appear_C1": (raw["C1"] - e["C1"]) / n,
                   "relative_mse": e["relative_mse"]}
    res["pass_V-a"] = max(res["V-a"]["appear_C0"], res["V-a"]["appear_C1"]) <= 0.01
    res["pass_V-b"] = max(res["V-b"]["appear_C0"], res["V-b"]["appear_C1"]) >= 0.05
    res["pass"] = bool(res["pass_V-a"] and res["pass_V-b"] and res["V-a"]["roundtrip"] and res["V-b"]["roundtrip"])
    out["E4_inr"] = res
    print("E4_inr", json.dumps(res), flush=True)
    out["all_pass"] = all(r["pass"] for r in out.values() if isinstance(r, dict))
    out["all_pass_except_E1_flow"] = all(r["pass"] for k, r in out.items() if isinstance(r, dict) and k != "E1_flow")
    out["note"] = "appearance excludes the network cost (shared nets are billed over the whole model in the real runs)"
    json.dump(out, open(T.RES / "synthetic_stage3.json", "w"), indent=1)
    print("all_pass", out["all_pass"])



# ------------------------------------------------------------------ summary

def summary():
    """test / train / T2: C0, C1, appearance vs block levels, net gain with the network billed at 16 bit per parameter
    (a) over the 24 test matrices, (b) over the whole model (5.57e11 weights). E4 separately."""
    raw = T.raw_table()
    ref = T.reference()
    chosen = json.load(open(OUT / "chosen.json"))
    rows = T.load_rows(OUT / "code.jsonl")
    T2 = T.t2_names()
    out = {"note": "E1_flow (RealNVP) failed its V-b check and was not used to code real data; its validation "
                   "likelihood is in train.jsonl", "chosen": chosen, "methods": {}}
    for key, ch in chosen.items():
        vn = f"{key}|" + ",".join(f"{a}={b}" for a, b in sorted({**ch['cfg'], **ch['v']}.items()))
        rs = [r for r in rows if r["vname"] == vn]
        d = {}
        for split, names in (("train", T.TRAIN), ("test", T.TEST), ("T2", T2)):
            sel = [r for r in rs if r["name"] in names]
            if not sel:
                continue
            N = sum(r["n"] for r in sel)
            npar = sel[0]["params"]
            s = {"matrices": len(sel), "params": npar, "roundtrip_all": all(r["roundtrip"] for r in sel)}
            for c in ("C0", "C1"):
                s[c] = sum(r[c] for r in sel) / N
                s["appear_" + c] = sum(raw[r["name"]]["level"][c] - r[c] for r in sel) / N
            if all("prior_bits" in r for r in sel):
                s["prior"] = sum(r["prior_bits"] for r in sel) / N
                s["appear_prior"] = sum(raw[r["name"]]["level"]["C0"] - r["prior_bits"] for r in sel) / N
            s["model_bpw_over_test"] = 16.0 * npar / N
            s["model_bpw_over_model"] = 16.0 * npar / NL.ALL_W
            s["coding_gain_mean"] = float(np.mean([r["coding_gain"] for r in sel if r.get("coding_gain")])) \
                if any(r.get("coding_gain") for r in sel) else None
            if all(r["name"] in ref for r in sel):
                best = sum(min(r["C0"], r["C1"], r.get("prior_bits", float("inf"))) + raw[r["name"]]["scale_bits"] for r in sel) / N
                tref = sum(ref[r["name"]]["total"] for r in sel) / N
                s["reference_bpw"] = tref
                s["net_gain_model_over_test"] = tref - best - s["model_bpw_over_test"]
                s["net_gain_model_over_model"] = tref - best - s["model_bpw_over_model"]
            d[split] = s
        out["methods"][key] = d
    e4 = T.load_rows(OUT / "e4.jsonl")
    if e4:
        g = {}
        for r in e4:
            g.setdefault((r["split"], json.dumps(r["variant"], sort_keys=True)), []).append(r)
        out["E4_inr"] = {}
        for (sp, vk), rs in g.items():
            N = sum(r["n"] for r in rs)
            s = {"variant": json.loads(vk), "matrices": [r["name"] for r in rs], "params": rs[0]["params"],
                 "roundtrip_all": all(r["roundtrip"] for r in rs),
                 "relative_mse_mean": float(np.mean([r["relative_mse"] for r in rs])),
                 "side_bpw": sum(r["side_bits"] for r in rs) / N}
            for c in ("C0", "C1"):
                s[c] = sum(r[c] for r in rs) / N
                s["appear_" + c] = sum(raw[r["name"]]["aligned"][c] - r[c] for r in rs) / N
            rf = {**ref, **{r["task"]: r for r in T.load_rows(T.RES / "reference_cross.jsonl")}}
            if all(r["name"] in rf for r in rs):
                s["net_gain"] = sum(rf[r["name"]]["total"] - min(r["C0"], r["C1"]) - raw[r["name"]]["scale_bits"] for r in rs) / N
            out["E4_inr"][f"{sp}|{vk}"] = s
    json.dump(out, open(OUT / "summary.json", "w"), indent=1)
    for k, d in out["methods"].items():
        te = d.get("test", {})
        print(f"{k:10s} test C0 {te.get('C0', 0):.4f} C1 {te.get('C1', 0):.4f} prior {te.get('prior', float('nan')):.4f} "
              f"ap0 {te.get('appear_C0', 0):+.4f} ap1 {te.get('appear_C1', 0):+.4f} net(test) {te.get('net_gain_model_over_test', 0):+.4f} "
              f"net(model) {te.get('net_gain_model_over_model', 0):+.4f} T2 net {d.get('T2', {}).get('net_gain_model_over_model', float('nan')):+.4f} rt {te.get('roundtrip_all')}")
    for k, s in out.get("E4_inr", {}).items():
        print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in s.items() if a != "matrices"})

if __name__ == "__main__":
    {"synth": synth, "train": train, "code": code, "e4": e4, "summary": summary}[sys.argv[1]]()
