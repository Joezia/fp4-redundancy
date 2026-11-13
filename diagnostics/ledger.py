"""Spec 002: ledger of R1 + zero merge + scale conditioning + scale stream (statistics only).

`ledger_matrix` is the single per-matrix code path for both the section-4 checks and the model
matrices. `shared_lengths` computes the leave-one-out shared code tables (M4 schemes b/c) from the
per-matrix count tables, and `ledger_rows` assembles the M5 rows. All quantities are plug-in, in
bits; per-weight values divide by N codewords, per-scale values by N/B scales.
"""
import numpy as np

from . import est
from .measure import E2M1, shuffle

RAW = 4.0  # bit per codeword
TABLE_BITS = 12  # bits per stored probability (spec M4)
REL_CLIP = 3  # E_rel range [-3, +3]
ALPHA = 0.5  # add-alpha for shared tables
CP_SYMS = np.array([c for c in range(16) if c != 8])  # the 15 values of C'
ESC = 2 * REL_CLIP + 1  # shared scale table: symbols -3..+3 then one escape symbol
REPS = 3

ROW_NAMES = {
    1: "仅R1：H(C)，scale 不压缩",
    2: "加零合并：H(C')",
    3: "加 scale 条件：H(C'|E)",
    4: "加 scale 流：H(C'|E) + H0(E)",
    5: "第4行加 m_max（含标签成本）",
    6: "第4行改用全局共用码表（方案 b）",
    "6abs": "补充：第4行改用按绝对 E 的全局共用码表",
}


def merge_zero(C):
    """C -> C': codeword 8 (-0) becomes 0 (+0); every other codeword is unchanged."""
    Cp = C.copy()
    Cp[Cp == 8] = 0
    return Cp


def cond_entropy(ctx, nctx, tgt, k=16):
    """H(T | X) = H(X, T) - H(X) (plug-in), plus the number of occurring contexts."""
    key = ctx.astype(np.int64).ravel() * k + tgt.astype(np.int64).ravel()
    joint = np.bincount(key, minlength=nctx * k)
    marg = joint.reshape(nctx, k).sum(axis=1)
    return est.entropy_counts(joint) - est.entropy_counts(marg), int((marg > 0).sum())


def table_cost(n_ctx, alphabet, n):
    """Per-symbol cost of storing one probability table per context: (A-1) * 12 bits each."""
    return n_ctx * (alphabet - 1) * TABLE_BITS / n


def _m2(C, Einv_w, K_E, Einv_b, B):
    """H(C'|E), H(C'|E,m_max), H(m_max|E) and the M2 incremental net benefit for one matrix."""
    R, W = C.shape
    Cp = merge_zero(C).ravel()
    mm = (C & 7).reshape(R, W // B, B).max(axis=2).ravel()
    h_e, _ = cond_entropy(Einv_w, K_E, Cp)
    h_em, k_em = cond_entropy(np.repeat(Einv_b * 8 + mm, B), K_E * 8, Cp)
    h_mm_e, _ = cond_entropy(Einv_b, K_E, mm, k=8)
    return {"H_Cp_E": h_e, "H_Cp_E_mmax": h_em, "H_mmax_E": h_mm_e, "K_E_mmax": k_em,
            "inc_net": h_e - h_em - h_mm_e / B, "mmax": mm}


def ledger_matrix(C, E, B, rng):
    """Per-matrix M1-M3 quantities and the count tables needed for M4/M5.

    C: (R, W) uint8 codewords 0..15; E: (R, W/B) uint8 E8M0 bytes for the blocks of C.
    """
    R, W = C.shape
    assert W % B == 0 and E.shape == (R, W // B)
    N = C.size
    nS = E.size
    C = np.ascontiguousarray(C)
    Cf = C.ravel()
    Cp = merge_zero(C).ravel()
    m = Cf & 7
    s = Cf >> 3
    uE, Einv_b = np.unique(E.ravel(), return_inverse=True)
    K_E = len(uE)
    Einv_w = np.repeat(Einv_b, B)

    # --- M1: the 8 conditional entropies
    mm = (C & 7).reshape(R, W // B, B).max(axis=2).ravel()
    mm_w = np.repeat(mm, B)
    em_w = np.repeat(Einv_b * 8 + mm, B)
    ent = {}
    for tag, X in (("C", Cf), ("Cp", Cp)):
        ent[f"H({tag})"] = est.entropy_counts(np.bincount(X, minlength=16))
        ent[f"H({tag}|E)"] = cond_entropy(Einv_w, K_E, X)[0]
        ent[f"H({tag}|m_max)"] = cond_entropy(mm_w, 8, X)[0]
        ent[f"H({tag}|E,m_max)"] = cond_entropy(em_w, K_E * 8, X)[0]
    k_em = int((np.bincount(Einv_b * 8 + mm, minlength=K_E * 8) > 0).sum())

    # zero merge identity terms and mutual informations
    z = m == 0
    P_m0 = float(z.mean())
    H_s_m0 = est.entropy_counts(np.bincount(s[z], minlength=2)) if z.any() else float("nan")
    tab_C = np.bincount(Einv_w * 16 + Cf, minlength=K_E * 16).reshape(K_E, 16)
    tab_Cp = tab_C.copy()
    tab_Cp[:, 0] += tab_Cp[:, 8]
    tab_Cp[:, 8] = 0
    I_C_E = est.mi_table(tab_C)[0]
    I_Cp_E = est.mi_table(tab_Cp)[0]

    # per scale value: P(m=0|E), H(s|m=0,E), 15-bin histogram of C'
    per_scale = []
    for i, e in enumerate(uE):
        row = tab_C[i]
        n_e = int(row.sum())
        n0 = int(row[0] + row[8])
        per_scale.append({
            "E": int(e), "n_weights": n_e, "n_scales": int((Einv_b == i).sum()),
            "P(m=0|E)": n0 / n_e,
            "H(s|m=0,E)": est.entropy_counts(row[[0, 8]]) if n0 > 0 else float("nan"),
            "hist_Cp": tab_Cp[i, CP_SYMS].astype(np.int64),
            "hist_m": np.array([row[k] + row[k + 8] for k in range(8)], dtype=np.int64),
        })

    # --- M2: m_max increment; S0 shuffles (codewords permuted over the matrix, E fixed)
    m2 = _m2(C, Einv_w, K_E, Einv_b, B)
    m2.pop("mmax")
    s0 = []
    for _ in range(REPS):
        X = shuffle(C, "S0", rng, B)
        r = _m2(X, Einv_w, K_E, Einv_b, B)
        Xp = merge_zero(X).ravel()
        r["I(Cp;E)"] = est.mi_table(np.bincount(Einv_w * 16 + Xp, minlength=K_E * 16)
                                    .reshape(K_E, 16))[0]
        r.pop("mmax")
        s0.append(r)

    # --- M3: scale stream
    E_counts = np.bincount(Einv_b, minlength=K_E)
    H0_E = est.entropy_counts(E_counts)
    H_mmax = est.entropy_counts(np.bincount(mm, minlength=8))

    # E_rel tables for the shared code tables (M4 b/c)
    mode = int(uE[np.argmax(E_counts)])  # ties -> smallest scale byte
    rel = np.clip(uE.astype(np.int64) - mode, -REL_CLIP, REL_CLIP) + REL_CLIP
    tab_rel = np.zeros((2 * REL_CLIP + 1, 16), dtype=np.int64)
    np.add.at(tab_rel, rel, tab_Cp)
    d = uE.astype(np.int64) - mode
    sym = np.where(np.abs(d) > REL_CLIP, ESC, d + REL_CLIP)
    scale_rel = np.zeros(ESC + 1, dtype=np.int64)
    np.add.at(scale_rel, sym, E_counts)

    # absolute-E tables (supplement, not in spec 002: shared tables keyed by the scale byte itself)
    tab_abs = np.zeros((256, 16), dtype=np.int64)
    tab_abs[uE] = tab_Cp
    scale_abs = np.zeros(256, dtype=np.int64)
    scale_abs[uE] = E_counts

    return {
        "N": N, "n_scales": nS, "B": B, "K_E": K_E, "K_E_mmax": k_em, "ent": ent,
        "P_m0": P_m0, "H_s_m0": H_s_m0, "I_C_E": I_C_E, "I_Cp_E": I_Cp_E,
        "per_scale": per_scale, "m2": m2, "s0": s0, "H0_E": H0_E, "H_mmax": H_mmax,
        "mode_E": mode, "tab_rel": tab_rel, "scale_rel": scale_rel,
        "n_escape": int(scale_rel[ESC]), "tab_abs": tab_abs, "scale_abs": scale_abs,
    }


# ----------------------------------------------------------------------------- M4 shared tables

def shared_lengths(results, groups=None, key="rel"):
    """Leave-one-out cross-entropies under tables pooled from the other matrices.

    key="rel" (spec M4): contexts/symbols are E_rel = E - mode, clipped to [-3, +3], scale symbols
    -3..+3 plus an escape. key="abs" (supplement): contexts/symbols are the 256 scale bytes; no mode
    and no escape are needed.

    results: list of ledger_matrix outputs. groups: optional list of group labels (scheme c);
    only matrices with the same label (other than the tested one) contribute.
    Returns per matrix: codeword bit/weight under P(C'|E_rel), scale bit/scale under P(E_rel)
    (with escapes paying 8 raw bits and 8 bits per matrix for the mode), and the training size.
    """
    out = []
    n = len(results)
    for i, r in enumerate(results):
        idx = [j for j in range(n) if j != i and (groups is None or groups[j] == groups[i])]
        if not idx:
            out.append(None)
            continue
        tk, sk = ("tab_rel", "scale_rel") if key == "rel" else ("tab_abs", "scale_abs")
        tr = sum(results[j][tk] for j in idx)[:, CP_SYMS].astype(np.float64)
        p = (tr + ALPHA) / (tr.sum(axis=1, keepdims=True) + ALPHA * len(CP_SYMS))
        te = r[tk][:, CP_SYMS]
        cw = float(-(te * np.log2(p)).sum() / r["N"])
        trs = sum(results[j][sk] for j in idx).astype(np.float64)
        ps = (trs + ALPHA) / (trs.sum() + ALPHA * len(trs))
        tes = r[sk]
        bits = float(-(tes * np.log2(ps)).sum())
        esc = int(tes[ESC]) if key == "rel" else 0
        if key == "rel":
            bits += 8.0 * esc + 8.0
        out.append({"cw_bits": cw, "scale_bits": bits / r["n_scales"],
                    "scale_escapes": esc, "train_matrices": len(idx)})
    return out


# ----------------------------------------------------------------------------- M5 ledger

def ledger_rows(r, shared_b, shared_abs=None):
    """The six M5 rows for one matrix. Each row: component costs in bit/weight (scale terms in
    bit/scale before the /B), compressed size and saving vs 4.25."""
    B, N, nS, K = r["B"], r["N"], r["n_scales"], r["K_E"]
    e = r["ent"]
    raw_scale = 8.0
    sc_a = (r["H0_E"], K * TABLE_BITS / nS)
    rows = {
        1: dict(cw=e["H(C)"], cw_tab=table_cost(1, 16, N), sc=raw_scale, sc_tab=0.0),
        2: dict(cw=e["H(Cp)"], cw_tab=table_cost(1, 15, N), sc=raw_scale, sc_tab=0.0),
        3: dict(cw=e["H(Cp|E)"], cw_tab=table_cost(K, 15, N), sc=raw_scale, sc_tab=0.0),
        4: dict(cw=e["H(Cp|E)"], cw_tab=table_cost(K, 15, N), sc=sc_a[0], sc_tab=sc_a[1]),
        5: dict(cw=e["H(Cp|E,m_max)"], cw_tab=table_cost(r["K_E_mmax"], 15, N),
                sc=sc_a[0], sc_tab=sc_a[1],
                lab=r["m2"]["H_mmax_E"] / B, lab_tab=table_cost(K, 8, N)),
    }
    if shared_b is not None:
        rows[6] = dict(cw=shared_b["cw_bits"], cw_tab=0.0, sc=shared_b["scale_bits"], sc_tab=0.0)
    else:
        rows[6] = None
    rows["6abs"] = (dict(cw=shared_abs["cw_bits"], cw_tab=0.0, sc=shared_abs["scale_bits"],
                         sc_tab=0.0) if shared_abs is not None else None)
    out = {}
    for k in rows:
        v = rows[k]
        if v is None:
            out[k] = None
            continue
        v.setdefault("lab", 0.0)
        v.setdefault("lab_tab", 0.0)
        comp = v["cw"] + v["cw_tab"] + (v["sc"] + v["sc_tab"]) / B + v["lab"] + v["lab_tab"]
        v["compressed"] = comp
        v["saving"] = (RAW + 8.0 / B) - comp
        out[k] = v
    return out


def sum_of_parts(r):
    """The 001 item sum: (4-H(C)) + P(m=0)H(s|m=0) + I(C;E) + (8-H0(E))/B."""
    B = r["B"]
    return ((RAW - r["ent"]["H(C)"]) + r["P_m0"] * r["H_s_m0"] + r["I_C_E"]
            + (8.0 - r["H0_E"]) / B)


def value_check():
    """U1: v(C') == v(C) for all 16 codewords (+0 and -0 compare equal)."""
    c = np.arange(16, dtype=np.uint8)
    return bool(np.all(E2M1[merge_zero(c)] == E2M1[c]))
