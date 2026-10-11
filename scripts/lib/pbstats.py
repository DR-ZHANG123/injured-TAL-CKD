"""供体级 pseudobulk 统计：logCPM、Hedges g、跨队列一致性及供体标签置换零分布。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def state_matrix(pb: pd.DataFrame, counts: pd.DataFrame, subclasses: list[str], min_cells: int) -> pd.DataFrame:
    """把若干亚群的原始计数在供体内相加；只保留合计细胞数 ≥ min_cells 的供体。返回 供体 × 基因 logCPM。"""
    sub = [s for s in subclasses if s in counts.columns]
    ncell = counts[sub].sum(1)
    donors = ncell.index[ncell >= min_cells]
    rows = []
    for d in donors:
        keys = [f"{d}|{s}" for s in sub if f"{d}|{s}" in pb.index]
        rows.append(pb.loc[keys].sum(0).rename(d))
    if not rows:
        return pd.DataFrame()
    m = pd.DataFrame(rows)
    return np.log2(m.div(m.sum(1), axis=0) * 1e6 + 1)


def hedges_g(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """逐列 Hedges g（x 组相对 y 组），带小样本校正。"""
    n1, n2 = len(x), len(y)
    s = np.sqrt(((n1 - 1) * x.var(0, ddof=1) + (n2 - 1) * y.var(0, ddof=1)) / (n1 + n2 - 2))
    d = (x.mean(0) - y.mean(0)) / np.where(s > 0, s, np.nan)
    return d * (1 - 3 / (4 * (n1 + n2) - 9))


def group_effect(m: pd.DataFrame, labels: pd.Series, a: str, b: str) -> pd.Series:
    lab = labels.reindex(m.index)
    x, y = m[lab == a].values, m[lab == b].values
    return pd.Series(hedges_g(x, y), index=m.columns)


def concordance(g_disc: pd.Series, m_rep: pd.DataFrame, labels_rep: pd.Series, a: str, b: str,
                top: int, n_perm: int, rng: np.random.Generator) -> dict:
    """复现队列中 a vs b 的效应与发现效应的一致性（全部基因 Spearman；发现 top 基因的同向比例），
    零分布来自复现队列内 a/b 标签置换。"""
    genes = g_disc.dropna().index.intersection(m_rep.columns)
    lab = labels_rep.reindex(m_rep.index)
    keep = lab.isin([a, b])
    M, L = m_rep.loc[keep, genes], lab[keep].values
    gd = g_disc[genes]
    topg = gd.abs().sort_values(ascending=False).index[:top]
    ti = genes.get_indexer(topg)

    def stats(labels: np.ndarray) -> tuple[float, float]:
        gr = pd.Series(hedges_g(M.values[labels == a], M.values[labels == b]), index=genes)
        ok = gr.notna()
        rho = gd[ok].rank().corr(gr[ok].rank())
        same = float((np.sign(gr.values[ti]) == np.sign(gd.values[ti])).mean())
        return rho, same

    rho, same = stats(L)
    null = np.array([stats(rng.permutation(L)) for _ in range(n_perm)])
    return {"n_genes": len(genes), "n_a": int((L == a).sum()), "n_b": int((L == b).sum()),
            "spearman": rho, "spearman_null_mean": float(np.nanmean(null[:, 0])),
            "p_spearman": float((1 + (null[:, 0] >= rho).sum()) / (n_perm + 1)),
            "top_same_direction": same, "top_null_mean": float(null[:, 1].mean()),
            "p_top": float((1 + (null[:, 1] >= same).sum()) / (n_perm + 1))}
