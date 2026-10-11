"""KPMP 参照的细胞状态注释迁移（相关法）与损伤 TAL 基因程序的供体内配对比较。

与 stage 09 相同的做法：参照 = KPMP 各 SubclassLevel2 亚群全部供体计数之和的 log1p CP10k；
注释基因 = 每个亚群相对其余参照亚群中位数最特异的前 n 个基因；每个细胞在这些基因上与参照质心的
Pearson 相关取最大者。TAL 状态参照只在 TAL 状态之间计算特异性，用于区分 TAL 内部的正常 / 损伤 / 退行状态。
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import ttest_1samp

from lib.pbstats import state_matrix


def reference(pb: pd.DataFrame, n_per: int, states: list[str] | None = None, drop_regex: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    """pb：供体|亚群 × 基因 的原始计数。states 给定时只在这些亚群之间计算特异性。"""
    sub = pb.index.str.split("|").str[1]
    agg = pb.groupby(sub).sum()
    agg = agg[agg.sum(1) > 0]
    if states is not None:
        agg = agg.loc[[s for s in states if s in agg.index]]
    logcpm = np.log1p(agg.div(agg.sum(1), axis=0) * 1e4)
    cols = logcpm.columns
    if drop_regex:
        pat = re.compile(drop_regex)
        cols = pd.Index([g for g in cols if not pat.match(g)])
    spec = logcpm[cols] - logcpm[cols].median(0)
    genes = sorted({g for s in spec.index for g in spec.loc[s].nlargest(n_per).index})
    return logcpm[genes], genes


def annotate(m: sp.csr_matrix, feats: pd.Index, ref: pd.DataFrame, genes: list[str],
             chunk: int = 50000) -> tuple[np.ndarray, np.ndarray]:
    """m：细胞 × 基因原始计数。返回 每个细胞相关最高的参照亚群 与 该相关值。"""
    first = pd.Series(np.arange(len(feats)), index=feats)[~feats.duplicated()]   # 重名基因取第一个
    gi = first.reindex(genes).fillna(-1).astype(int).values
    ok = gi >= 0
    r = ref.loc[:, np.array(genes)[ok]].values
    r = (r - r.mean(1, keepdims=True)) / (r.std(1, keepdims=True) + 1e-6)
    lib = np.asarray(m.sum(1)).ravel()
    sub = m[:, gi[ok]]
    labs, scores = [], []
    for s in range(0, m.shape[0], chunk):
        x = sub[s:s + chunk].toarray().astype(np.float32)
        x = np.log1p(x / lib[s:s + chunk, None] * 1e4)
        x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)
        c = x @ r.T / x.shape[1]
        labs.append(ref.index.values[c.argmax(1)])
        scores.append(c.max(1))
    return np.concatenate(labs), np.concatenate(scores)


def pseudobulk(m: sp.csr_matrix, feats: pd.Index, unit: np.ndarray, label: np.ndarray) -> pd.DataFrame:
    """单元|状态 × 基因 原始计数之和（重名基因合并）。"""
    key = pd.Series(unit).astype(str) + "|" + pd.Series(label).astype(str)
    keys = pd.Index(sorted(key.unique()))
    ind = sp.csr_matrix((np.ones(len(key)), (keys.get_indexer(key), np.arange(len(key)))),
                        shape=(len(keys), len(key)))
    pb = pd.DataFrame((ind @ m).toarray().astype(np.float32), index=keys, columns=feats)
    return pb.T.groupby(level=0).sum().T


def paired_programme(pb: pd.DataFrame, cnt: pd.DataFrame, inj: list[str], norm: list[str], min_cells: int,
                     donor_of_unit: pd.Series | None = None) -> pd.DataFrame:
    """每个单元内 损伤 − 正常 的 log2CPM 差；同一供体多个单元（如单核 + 单细胞）先平均，再跨供体单样本 t 检验。
    表达过滤与 stage 15 相同：任一状态的平均 log2CPM ≥ 1。"""
    a, n = state_matrix(pb, cnt, inj, min_cells), state_matrix(pb, cnt, norm, min_cells)
    units = a.index.intersection(n.index)
    genes = a.columns.intersection(n.columns)
    a, n = a.loc[units, genes], n.loc[units, genes]
    expressed = genes[(a.mean() >= 1) | (n.mean() >= 1)]
    diff = a[expressed] - n[expressed]
    if donor_of_unit is not None:
        diff = diff.groupby(donor_of_unit.reindex(diff.index).values).mean()
    res = pd.DataFrame({"log2fc": diff.mean(), "frac_donors_up": (diff > 0).mean()}, index=expressed)
    if len(diff) >= 3:
        t, p = ttest_1samp(diff.values, 0.0)
        res["t"], res["p"] = t, p
    res.attrs["n_donors"] = len(diff)
    res.attrs["n_units"] = len(units)
    return res


def programme_concordance(disc: pd.DataFrame, rep: pd.DataFrame, fdr: float, min_lfc: float, rep_lfc: float,
                          exclude: set[str] | None = None) -> dict:
    """与 stage 15 相同：全部共有基因的 log2FC Pearson r；发现命中（FDR < fdr 且 |log2FC| ≥ min_lfc）中
    复现同向且 |log2FC| ≥ rep_lfc 的比例（未在复现队列表达的命中计为未复现）。exclude：剔除的基因（注释迁移所用基因）。"""
    d = disc if not exclude else disc.loc[disc.index.difference(list(exclude))]
    common = d.index.intersection(rep.index)
    hits = d[(d.fdr < fdr) & (d.log2fc.abs() >= min_lfc)]
    rl = rep.log2fc.reindex(hits.index)
    ok = (np.sign(rl) == np.sign(hits.log2fc)) & (rl.abs() >= rep_lfc)
    up = hits.log2fc > 0
    return {"n_genes_common": len(common), "lfc_pearson_r": float(d.log2fc[common].corr(rep.log2fc[common])),
            "lfc_spearman_rho": float(d.log2fc[common].corr(rep.log2fc[common], method="spearman")),
            "n_hits": len(hits), "n_hits_expressed": int(rl.notna().sum()), "rep_rate": float(ok.mean()),
            "rep_rate_expressed": float(ok[rl.notna()].mean()),
            "n_up_hits": int(up.sum()), "rep_rate_up": float(ok[up].mean()),
            "n_down_hits": int((~up).sum()), "rep_rate_down": float(ok[~up].mean())}
