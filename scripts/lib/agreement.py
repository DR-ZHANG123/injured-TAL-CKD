"""同一参与者两种测量（单核 / 单细胞）的一致性：ICC、Bland–Altman、配对 bootstrap 比较两个谱系的 Spearman ρ。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


def icc(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """两向随机效应单次测量 ICC：绝对一致 ICC(A,1) 与一致性 ICC(C,1)（McGraw & Wong 1996）。"""
    Y = np.column_stack([x, y])
    n, k = Y.shape
    gm = Y.mean()
    msr = k * ((Y.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((Y.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    icc_a = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
    icc_c = (msr - mse) / (msr + (k - 1) * mse)
    return float(icc_a), float(icc_c)


def bland_altman(sn: np.ndarray, sc: np.ndarray) -> dict:
    d = sc - sn
    b, s = d.mean(), d.std(ddof=1)
    return {"ba_bias_sc_minus_sn": b, "ba_loa_low": b - 1.96 * s, "ba_loa_high": b + 1.96 * s}


def _rho(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def boot_rho_diff(d: pd.DataFrame, ref: str, other: str, n_boot: int, rng: np.random.Generator) -> dict:
    """d 含列 <lin>_sn / <lin>_sc；同一批参与者上 ρ_ref − ρ_other，参与者配对 bootstrap 百分位 CI。"""
    a1, a2, b1, b2 = (d[f"{ref}_sn"].values, d[f"{ref}_sc"].values, d[f"{other}_sn"].values, d[f"{other}_sc"].values)
    obs = _rho(a1, a2) - _rho(b1, b2)
    n = len(d)
    bs = np.empty(n_boot)
    for i in range(n_boot):
        j = rng.integers(0, n, n)
        bs[i] = _rho(a1[j], a2[j]) - _rho(b1[j], b2[j])
    bs = bs[np.isfinite(bs)]
    return {"rho_ref": _rho(a1, a2), "rho_other": _rho(b1, b2), "rho_diff": obs,
            "ci_low": float(np.quantile(bs, .025)), "ci_high": float(np.quantile(bs, .975)),
            "p_boot_two_sided": float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))}


def lineage_fractions(cnt: pd.DataFrame, lineages: dict, min_cells: int) -> pd.DataFrame:
    names = {s for v in lineages.values() for s in v["num"] + v["den"]}
    c = cnt.reindex(columns=sorted(set(cnt.columns) | names), fill_value=0)
    out = {}
    for lin, v in lineages.items():
        den = c[v["den"]].sum(1)
        out[lin] = (c[v["num"]].sum(1) / den).where(den >= min_cells)
    return pd.DataFrame(out)


def agreement_rows(sn: pd.DataFrame, sc: pd.DataFrame, lab: pd.Series, lineages: list[str], ref: str,
                   eps: float, n_boot: int, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """每个谱系的一致性指标（各自可用参与者，及与 ref 谱系共同可用的参与者）+ ref 与其他谱系 ρ 差的 bootstrap。"""
    lg = lambda v: np.log((v + eps) / (1 - v + eps))
    ids = sn.index.intersection(sc.index)
    w = pd.concat([sn.loc[ids].add_suffix("_sn"), sc.loc[ids].add_suffix("_sc")], axis=1)
    w["dx"] = lab.reindex(ids)
    rows, diffs = [], []
    for subset, wd in (("all", w), ("disease_only", w[w.dx.isin(["DKD", "HKD"])]),
                       ("non_reference", w[w.dx != "REF"])):
        for lin in lineages:
            for pairing, cols in (("own_available", [f"{lin}_sn", f"{lin}_sc"]),
                                  (f"with_{ref}", [f"{lin}_sn", f"{lin}_sc", f"{ref}_sn", f"{ref}_sc"])):
                d = wd.dropna(subset=cols)
                if len(d) < 5:
                    continue
                x, y = lg(d[f"{lin}_sn"].values), lg(d[f"{lin}_sc"].values)
                ia, ic = icc(x, y)
                rho, p = spearmanr(d[f"{lin}_sn"], d[f"{lin}_sc"])
                rows.append({"subset": subset, "lineage": lin, "participants": pairing, "n": len(d),
                             "spearman": rho, "p_spearman": p, "pearson_logit": float(np.corrcoef(x, y)[0, 1]),
                             "icc_absolute_logit": ia, "icc_consistency_logit": ic,
                             "median_sn": d[f"{lin}_sn"].median(), "median_sc": d[f"{lin}_sc"].median(),
                             **bland_altman(x, y)})
            if lin != ref:
                d = wd.dropna(subset=[f"{lin}_sn", f"{lin}_sc", f"{ref}_sn", f"{ref}_sc"])
                if len(d) >= 5:
                    diffs.append({"subset": subset, "reference_lineage": ref, "other_lineage": lin, "n": len(d),
                                  **boot_rho_diff(d, ref, lin, n_boot, rng)})
    return pd.DataFrame(rows), pd.DataFrame(diffs)
