"""Stage 15：修复失败小管状态的关键基因——发现、独立复现与临床关联。

发现（KPMP 单核）：供体内配对比较 损伤状态 vs 同谱系正常状态 的 pseudobulk logCPM（配对 t，BH-FDR）。
复现：KPMP 单细胞中未出现在单核的供体（同样配对比较）；GSE271188（stage 09 注释迁移后的 pseudobulk）。
临床：损伤状态 pseudobulk 中的基因表达与 eGFR（分档中点）的关系，校正诊断类别（KPMP 单核）。
遗传：stage 07f 中有至少一种遗传证据的基因做标注。
输出：results/15_key_genes/{<state>_discovery.tsv, <state>_key_genes.tsv, summary.tsv}
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.pbstats import state_matrix  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

STAGE = "15_key_genes"
EGFR_COL = "Baseline eGFR (ml/min/1.73m2) (Binned)"


def paired(pb: pd.DataFrame, cnt: pd.DataFrame, inj: list[str], norm: list[str], min_cells: int,
           donors: pd.Index | None = None) -> pd.DataFrame:
    a, n = state_matrix(pb, cnt, inj, min_cells), state_matrix(pb, cnt, norm, min_cells)
    d = a.index.intersection(n.index)
    if donors is not None:
        d = d.intersection(donors)
    genes = a.columns.intersection(n.columns)
    a, n = a.loc[d, genes], n.loc[d, genes]
    if len(d) < 3:                                   # 供体不足：返回空表（该复现集不可用）
        res = pd.DataFrame(columns=["log2fc", "t", "p", "mean_injured", "frac_donors_up", "fdr"], dtype=float)
        res.attrs["n_donors"] = len(d)
        return res
    expressed = genes[(a.mean() >= 1) | (n.mean() >= 1)]
    diff = a[expressed] - n[expressed]
    t, p = ttest_rel(a[expressed], n[expressed])
    res = pd.DataFrame({"log2fc": diff.mean(), "t": t, "p": p, "mean_injured": a[expressed].mean(),
                        "frac_donors_up": (diff > 0).mean()}, index=expressed)
    res["fdr"] = multipletests(res.p.fillna(1), method="fdr_bh")[1]
    res.attrs["n_donors"] = len(d)
    return res


def egfr_assoc(pb, cnt, don, inj, min_cells, genes) -> pd.DataFrame:
    m = state_matrix(pb, cnt, inj, min_cells)
    eg = don[EGFR_COL].map(lambda s: (lambda x: (int(x.group(1)) + int(x.group(2))) / 2 if x else np.nan)(
        re.match(r"(\d+)-(\d+)", str(s))))
    dx = don["ConditionCategory"].astype(str)
    d = pd.DataFrame({"egfr": eg, "dx": dx}).reindex(m.index).dropna()
    rows = []
    for g in genes:
        if g not in m.columns:
            continue
        x = d.assign(y=m.loc[d.index, g].rank(), e=d.egfr.rank())
        f = smf.ols("y ~ e + C(dx)", x).fit()
        rows.append({"gene": g, "egfr_beta_rank": float(f.params["e"]), "egfr_p": float(f.pvalues["e"])})
    if not rows:                                     # 没有复现基因时返回空表
        out = pd.DataFrame(columns=["egfr_beta_rank", "egfr_p", "egfr_fdr"], dtype=float)
        out.attrs["n_donors"] = len(d)
        return out
    out = pd.DataFrame(rows).set_index("gene")
    out["egfr_fdr"] = multipletests(out.egfr_p, method="fdr_bh")[1]
    out.attrs["n_donors"] = len(d)
    return out


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    kg = cfg["key_genes"]
    mc = cfg["kpmp"]["min_cells"]
    res3 = ROOT / "results" / "03_kpmp_pseudobulk"
    proc = ROOT / cfg["paths"]["processed"]
    sn = pd.read_parquet(proc / "kpmp" / "snRNA_pseudobulk.parquet")
    sn_cnt = pd.read_csv(res3 / "snRNA_cell_counts.tsv", sep="\t", index_col=0)
    sn_don = pd.read_csv(res3 / "snRNA_donors.tsv", sep="\t", index_col=0)
    sc = pd.read_parquet(proc / "kpmp" / "scRNA_pseudobulk.parquet")
    sc_cnt = pd.read_csv(res3 / "scRNA_cell_counts.tsv", sep="\t", index_col=0)
    g2 = pd.read_parquet(proc / "gse271188" / "pseudobulk.parquet")
    g2_cnt = pd.read_csv(ROOT / "results" / "09_gse271188_replication" / "cell_counts.tsv", sep="\t", index_col=0)
    for t in (sn_cnt, sn_don, sc_cnt):
        t.index = t.index.astype(str)
    genetic = pd.read_csv(ROOT / "results" / "07f_genetic_convergence" / "genetic_evidence_by_gene.tsv", sep="\t")
    genetic = genetic.groupby("symbol").agg(genetic_traits=("trait", lambda x: ";".join(sorted(set(x)))),
                                            genetic_lines=("n_lines", "max"))
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)

    summary, outs = [], []
    for state, spec in kg["states"].items():
        inj, norm = spec["injured"], spec["normal"]
        disc = paired(sn, sn_cnt, inj, norm, mc)
        rep_sc = paired(sc, sc_cnt, inj, norm, mc, donors=sc_cnt.index.difference(sn_cnt.index))
        rep_g2 = paired(g2, g2_cnt, inj, norm, kg["min_cells_gse271188"])
        disc.to_csv(out / f"{state}_discovery.tsv", sep="\t")
        hits = disc[(disc.fdr < kg["fdr"]) & (disc.log2fc.abs() >= kg["min_lfc"])].index
        tab = disc.loc[hits, ["log2fc", "fdr", "frac_donors_up"]].join(
            rep_sc[["log2fc", "fdr"]].add_suffix("_kpmp_sc")).join(rep_g2[["log2fc", "fdr"]].add_suffix("_gse271188"))
        same = lambda c: np.sign(tab[c]) == np.sign(tab.log2fc)
        tab["replicated_kpmp_sc"] = same("log2fc_kpmp_sc") & (tab.log2fc_kpmp_sc.abs() >= kg["rep_lfc"])
        tab["replicated_gse271188"] = same("log2fc_gse271188") & (tab.log2fc_gse271188.abs() >= kg["rep_lfc"])
        tab["replicated_both"] = tab.replicated_kpmp_sc & tab.replicated_gse271188
        eg = egfr_assoc(sn, sn_cnt, sn_don, inj, mc, tab.index[tab.replicated_both])
        tab = tab.join(eg).join(genetic)
        tab = tab.sort_values(["replicated_both", "log2fc"], ascending=[False, False])
        tab.to_csv(out / f"{state}_key_genes.tsv", sep="\t")
        outs += [out / f"{state}_discovery.tsv", out / f"{state}_key_genes.tsv"]
        up = tab[tab.log2fc > 0]
        summary.append({"state": state, "n_donors_disc": disc.attrs["n_donors"], "n_donors_kpmp_sc": rep_sc.attrs["n_donors"],
                        "n_donors_gse271188": rep_g2.attrs["n_donors"], "n_hits": len(tab), "n_up": len(up),
                        "rep_rate_kpmp_sc": float(tab.replicated_kpmp_sc.mean()),
                        "rep_rate_gse271188": float(tab.replicated_gse271188.mean()),
                        "n_replicated_both": int(tab.replicated_both.sum()),
                        "n_up_replicated_egfr_fdr05": int(((up.replicated_both) & (up.egfr_fdr < 0.05)).sum()),
                        "lfc_corr_kpmp_sc": float(disc.log2fc.corr(rep_sc.log2fc.reindex(disc.index))),
                        "lfc_corr_gse271188": float(disc.log2fc.corr(rep_g2.log2fc.reindex(disc.index)))})
        print(summary[-1], flush=True)
        print(tab[tab.replicated_both & (tab.log2fc > 0)].head(25)[["log2fc", "log2fc_kpmp_sc", "log2fc_gse271188",
              "egfr_beta_rank", "egfr_fdr", "genetic_traits"]].round(3).to_string(), flush=True)
    pd.DataFrame(summary).to_csv(out / "summary.tsv", sep="\t", index=False)
    write_provenance(STAGE, inputs=[res3 / "snRNA_cell_counts.tsv", res3 / "scRNA_cell_counts.tsv"],
                     outputs=outs + [out / "summary.tsv"], seed=cfg["seed"])


if __name__ == "__main__":
    main()
