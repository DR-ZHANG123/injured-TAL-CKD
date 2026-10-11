"""Stage 22 辅助函数：TAL 状态比例、组间比较、eGFR 模型、基因程序得分（参与者水平）。"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests

from lib.pbstats import hedges_g


def logit(v: pd.Series, eps: float = 1e-3) -> pd.Series:
    return np.log((v + eps) / (1 - v + eps))


def bin_mid(s) -> float:
    """“50-59 …” → 55；“>60” 等开放分档 → NaN（与 stage 16 相同）。"""
    m = re.match(r"\s*\[?'?(\d+)-(\d+)", str(s))
    return (int(m.group(1)) + int(m.group(2)) + 1) / 2 if m else np.nan


def labels(don: pd.DataFrame, groups: dict, comparators: dict) -> pd.Series:
    """REF/DKD/HKD 按 kpmp.groups；其余参与者中再按 comparators 规则标 AKI、DM_resilient。"""
    lab = pd.Series("other", index=don.index, dtype=object)
    for rules, only_other in ((groups, False), (comparators, True)):
        for name, rule in rules.items():
            (col, val), = rule.items()
            hit = don[col].astype(str).isin(val if isinstance(val, list) else [val])
            if only_other:
                hit &= lab.eq("other")
            lab[hit] = name
    return lab


def fractions(cnt: pd.DataFrame, den: list[str], cort: list[str], med: list[str], min_cells: int,
              min_injured: int) -> pd.DataFrame:
    c = cnt.reindex(columns=sorted(set(cnt.columns) | set(den) | set(cort) | set(med)), fill_value=0)
    n_tal, n_cort = c[den].sum(1), c[cort].sum(1)
    a, f = c["aTAL"], c["frTAL"]
    ok, okc, oki = n_tal >= min_cells, n_cort >= min_cells, (a + f) >= min_injured
    return pd.DataFrame({
        "n_tal": n_tal, "n_aTAL": a, "n_frTAL": f, "n_tal_cortical": n_cort,
        "aTAL_frac": (a / n_tal).where(ok), "frTAL_frac": (f / n_tal).where(ok),
        "injured_frac": ((a + f) / n_tal).where(ok),
        "frTAL_share_of_injured": (f / (a + f)).where(ok & oki),
        "injured_frac_cortical": ((a + f) / n_cort).where(okc),
        "medullary_share": (c[med].sum(1) / n_tal).where(ok)})


def _covars(d: pd.DataFrame) -> pd.DataFrame:
    return d.assign(age=d.Age.map(bin_mid), male=(d.sex.astype(str).str.lower().str.startswith("m")).astype(float))


def contrasts(df: pd.DataFrame, measures: list[str], pairs: list[list[str]], modality: str) -> pd.DataFrame:
    """组间：中位数、Mann–Whitney、logit 尺度 Hedges g；OLS logit ~ 组 + 年龄档中点 + 性别 的组效应。"""
    rows = []
    d0 = _covars(df)
    for m in measures:
        for a, b in pairs:
            s = d0[d0.dx.isin([a, b])].dropna(subset=[m])
            x, y = s.loc[s.dx == a, m], s.loc[s.dx == b, m]
            if len(x) < 3 or len(y) < 3:
                continue
            r = {"modality": modality, "measure": m, "contrast": f"{a}_vs_{b}", "n_a": len(x), "n_b": len(y),
                 "median_a": x.median(), "median_b": y.median(),
                 "hedges_g_logit": float(hedges_g(logit(x).values[:, None], logit(y).values[:, None])[0]),
                 "p_mannwhitney": mannwhitneyu(x, y).pvalue}
            sa = s.dropna(subset=["age"]).assign(yy=lambda t: logit(t[m]), grp=lambda t: (t.dx == a).astype(float))
            if sa.grp.nunique() == 2 and len(sa) >= 8:
                f = smf.ols("yy ~ grp + age + male", sa).fit()
                r.update({"n_adj": len(sa), "beta_logit_adj_age_sex": f.params["grp"], "p_adj_age_sex": f.pvalues["grp"]})
            rows.append(r)
    out = pd.DataFrame(rows)
    out["fdr_mannwhitney"] = out.groupby(["modality", "measure"]).p_mannwhitney.transform(
        lambda p: multipletests(p, method="fdr_bh")[1])
    return out


def egfr_models(d: pd.DataFrame, ycol: str, name: str, transform, extra: list[str] | None = None,
                cohort: str = "snRNA") -> list[dict]:
    """y ~ eGFR/10 + 诊断（+ extra）；另做 + 年龄 + 性别；Spearman 整体及各诊断内。"""
    extra = extra or []
    d = _covars(d).dropna(subset=[ycol, "egfr_mid"] + extra).copy()
    d["y"], d["e10"] = transform(d[ycol]), d.egfr_mid / 10
    rho, p = spearmanr(d[ycol], d.egfr_mid)
    base = {"cohort": cohort, "measure": name, "adjusted_for": "+".join(["dx"] + extra)}
    row = {**base, "subset": "all", "n": len(d), "spearman": rho, "p_spearman": p}
    rhs = " + ".join(["e10", "C(dx)"] + extra)
    f = smf.ols(f"y ~ {rhs}", d).fit()
    row.update({"beta_per10": f.params["e10"], "ci_low": f.conf_int().loc["e10", 0],
                "ci_high": f.conf_int().loc["e10", 1], "p": f.pvalues["e10"]})
    da = d.dropna(subset=["age"])
    if len(da) >= 10:
        fa = smf.ols(f"y ~ {rhs} + age + male", da).fit()
        row.update({"n_age_sex": len(da), "beta_per10_age_sex": fa.params["e10"], "p_age_sex": fa.pvalues["e10"]})
    for e in extra:
        row[f"beta_{e}"], row[f"p_{e}"] = f.params[e], f.pvalues[e]
    rows = [row]
    for dx, s in d.groupby("dx"):
        if len(s) >= 5:
            r, pp = spearmanr(s[ycol], s.egfr_mid)
            fs = smf.ols(" ~ ".join(["y", " + ".join(["e10"] + extra)]), s).fit() if len(s) >= 8 else None
            rows.append({**base, "subset": dx, "n": len(s), "spearman": r, "p_spearman": pp,
                         "beta_per10": fs.params["e10"] if fs is not None else np.nan,
                         "p": fs.pvalues["e10"] if fs is not None else np.nan})
    return rows


def zscore_mean(m: pd.DataFrame, genes: list[str]) -> pd.Series:
    g = [x for x in genes if x in m.columns]
    z = (m[g] - m[g].mean()) / m[g].std().replace(0, np.nan)
    return z.mean(1)


def gene_egfr(m: pd.DataFrame, d: pd.DataFrame, genes: list[str], dx_col: str) -> pd.DataFrame:
    """逐基因：rank(表达) ~ rank(eGFR) + 诊断（与 stage 15 相同的秩回归），BH-FDR。"""
    d = d.reindex(m.index).dropna(subset=["egfr_mid", dx_col])
    rows = []
    for g in genes:
        if g not in m.columns:
            continue
        x = pd.DataFrame({"y": m.loc[d.index, g].rank(), "e": d.egfr_mid.rank(), "dx": d[dx_col].astype(str)})
        f = smf.ols("y ~ e + C(dx)", x).fit()
        rows.append({"gene": g, "beta_rank": f.params["e"], "p": f.pvalues["e"]})
    out = pd.DataFrame(rows).set_index("gene")
    out["fdr"] = multipletests(out.p, method="fdr_bh")[1]
    out.attrs["n"] = len(d)
    return out
