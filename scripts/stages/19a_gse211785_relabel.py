"""Stage 19a：GSE211785（Susztak 队列，不含 KPMP 样本）中损伤 TAL 扩增与损伤 TAL 基因程序的独立复现。

存放对象 layers/counts 为原始 UMI 计数（.X / .raw 为归一化值），全部分析用原始计数。
1. 主分析：原作者 TAL 簇（C_TAL、M_TAL）中的细胞，按技术用匹配的 KPMP 参照（单核 → snRNA、单细胞 → scRNA），
   在只于 TAL 状态之间计算的特异基因上做相关注释，得到 C-TAL / M-TAL / C/M-TAL / aTAL / frTAL / 退行 / 增殖 TAL。
   敏感性：全部 RNA 细胞按 KPMP 全亚群参照注释（stage 09 做法），TAL 谱系 = 注释为任一 TAL 状态的细胞。
2. 损伤 TAL（aTAL+frTAL）占 TAL 的比例，一个供体一个观测（优先单核文库；或两种技术计数相加）；logit ~ 诊断 + 技术；技术内 Mann–Whitney。
3. 损伤 TAL 基因程序：每个供体内 损伤 vs 正常 TAL 的 log2CPM 差（同一供体两种技术先平均），
   与 KPMP 单核发现（stage 15）比较：log2FC 相关、复现率；再剔除注释迁移所用基因后重算。
输出：results/19a_gse211785_relabel/*.tsv；data/processed/gse211785_tal/{pseudobulk.parquet, cell_labels.parquet}
"""
from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from lib.tal_transfer import annotate, paired_programme, programme_concordance, pseudobulk, reference  # noqa: E402

STAGE = "19a_gse211785_relabel"


def load_counts(path: Path, rows: np.ndarray) -> sp.csr_matrix:
    with h5py.File(path, "r") as f:
        g = f["layers/counts"]
        m = sp.csr_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=tuple(g.attrs["shape"]))
    return m[rows]


def label_cells(m, feats, tech, refs, min_r):
    lab = np.empty(m.shape[0], dtype=object)
    score = np.zeros(m.shape[0])
    for t, (ref, genes) in refs.items():
        i = np.where(tech == t)[0]
        if len(i):
            lab[i], score[i] = annotate(m[i], feats, ref, genes)
    return np.where(score >= min_r, lab, "unassigned"), score


def fractions(cnt: pd.DataFrame, units: pd.DataFrame, inj, den, min_cells, eps) -> pd.DataFrame:
    c = cnt.reindex(columns=sorted(set(cnt.columns) | set(den)), fill_value=0)
    d = units.join(pd.DataFrame({"n_tal": c[den].sum(1), "n_injured": c[inj].sum(1),
                                 "n_aTAL": c.get("aTAL", 0), "n_frTAL": c.get("frTAL", 0)}))
    d = d[d.n_tal >= min_cells].copy()
    d["frac"] = d.n_injured / d.n_tal
    d["logit"] = np.log((d.frac + eps) / (1 - d.frac + eps))
    return d


def per_donor(cnt: pd.DataFrame, units: pd.DataFrame, den, min_cells, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """一个供体一个观测：prefer_SN = 有合格单核文库时取单核，否则取单细胞；pooled = 两种技术的细胞计数相加。"""
    u = units.reindex(cnt.index)
    n = cnt.reindex(columns=den, fill_value=0).sum(1)
    if mode == "prefer_SN":
        ok = u[n >= min_cells].copy()
        ok["rank"] = (ok.tech != "SN_RNA").astype(int)
        pick = ok.sort_values("rank").groupby("sample").head(1).index
        return cnt.loc[pick], u.loc[pick]
    c = cnt.groupby(u["sample"].values).sum()
    meta = u.groupby("sample").agg(tech=("tech", lambda x: "+".join(sorted(set(x)))), group=("group", "first"))
    meta["sample"] = meta.index
    return c, meta


def abundance(d: pd.DataFrame, contrasts, analysis: str) -> list[dict]:
    rows = []
    for a_, b_ in contrasts:
        s = d[d.group.isin([a_, b_])].copy()
        s["is_a"] = (s.group == a_).astype(int)
        f = smf.ols("logit ~ is_a + C(tech)", s).fit()
        rec = {"analysis": analysis, "contrast": f"{a_}_vs_{b_}", "n_donors_a": s[s.is_a == 1]["sample"].nunique(),
               "n_donors_b": s[s.is_a == 0]["sample"].nunique(), "n_units": len(s),
               "median_frac_a": float(s.frac[s.is_a == 1].median()), "median_frac_b": float(s.frac[s.is_a == 0].median()),
               "coef_logit": float(f.params["is_a"]), "ci_low": float(f.conf_int().loc["is_a", 0]),
               "ci_high": float(f.conf_int().loc["is_a", 1]), "p": float(f.pvalues["is_a"])}
        for t in sorted(s.tech.unique()):
            x, y = s.frac[(s.tech == t) & (s.is_a == 1)], s.frac[(s.tech == t) & (s.is_a == 0)]
            rec[f"n_{t}"] = f"{len(x)}v{len(y)}"
            rec[f"median_a_{t}"], rec[f"median_b_{t}"] = (float(x.median()) if len(x) else np.nan,
                                                          float(y.median()) if len(y) else np.nan)
            rec[f"p_mw_{t}"] = float(mannwhitneyu(x, y).pvalue) if len(x) >= 2 and len(y) >= 2 else np.nan
        rows.append(rec)
    return rows


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    rc, g = cfg["replication2"], cfg["replication2"]["gse211785"]
    h5 = ROOT / cfg["paths"]["raw"] / "gse211785" / "susztak_preSCVI.h5ad"
    out = ROOT / "results" / STAGE
    proc = ROOT / cfg["paths"]["processed"] / "gse211785_tal"
    out.mkdir(parents=True, exist_ok=True)
    proc.mkdir(parents=True, exist_ok=True)

    a = ad.read_h5ad(h5, backed="r")
    obs = a.obs[["sample", "tech", "group", "Cluster_Idents", "age", "sex"]].copy()
    feats = pd.Index(a.var_names.astype(str))
    a.file.close()
    obs["lib"] = obs.index.str.split("_").str[0]
    keep = obs.tech.isin(g["techs"]).values & ~obs.lib.isin(g["exclude_libs"]).values
    rows = np.where(keep)[0]
    m = load_counts(h5, rows)
    obs = obs.iloc[rows].copy()
    obs["n_umi"] = np.asarray(m.sum(1)).ravel()
    ok = (obs.n_umi >= rc["min_umi"]).values
    m, obs = m[ok], obs[ok]
    for c in ("sample", "tech", "group"):
        obs[c] = obs[c].astype(str)
    obs["unit"] = obs["sample"] + "|" + obs["tech"]
    units = obs.drop_duplicates("unit").set_index("unit")[["sample", "tech", "group", "age", "sex"]]
    print("cells kept", m.shape, flush=True)

    pbs = {mod: pd.read_parquet(ROOT / cfg["paths"]["processed"] / "kpmp" / f"{mod}_pseudobulk.parquet")
           for mod in ("snRNA", "scRNA")}
    ts = rc["tal_states"]
    tal_refs = {t: reference(pbs[mod], ts["n_per"], ts[mod], ts["drop_gene_regex"]) for t, mod in g["reference_by_tech"].items()}
    all_refs = {t: reference(pbs[mod], rc["all_states"]["n_per"]) for t, mod in g["reference_by_tech"].items()}
    lt_genes = sorted(set().union(*[set(v[1]) for v in tal_refs.values()]))
    pd.DataFrame([(t, x) for t, (_, gl) in tal_refs.items() for x in gl], columns=["tech", "gene"]).to_csv(
        out / "reference_genes_tal.tsv", sep="\t", index=False)

    # 主分析：原作者 TAL 簇内的 TAL 状态注释
    tal = obs.Cluster_Idents.isin(g["tal_clusters"]).values
    lab, score = label_cells(m[tal], feats, obs.tech.values[tal], tal_refs, ts["min_r"])
    # 敏感性：全部 RNA 细胞的全亚群注释
    lab_all, score_all = label_cells(m, feats, obs.tech.values, all_refs, rc["all_states"]["min_r"])
    cells = pd.DataFrame({"unit": obs.unit.values, "cluster": obs.Cluster_Idents.astype(str).values,
                          "label_all": lab_all, "r_all": score_all}, index=obs.index)
    cells["label_tal"], cells["r_tal"] = None, np.nan
    cells.loc[tal, "label_tal"], cells.loc[tal, "r_tal"] = lab, score
    cells.to_parquet(proc / "cell_labels.parquet")

    cnt = pd.crosstab(cells.unit[tal], cells.label_tal[tal])
    cnt_all = pd.crosstab(cells.unit, cells.label_all)
    cnt.to_csv(out / "state_counts_tal.tsv", sep="\t")
    cnt_all.to_csv(out / "state_counts_allstates.tsv", sep="\t")
    tal_all = rc["tal_all_sn"] + [s for s in rc["tal_all_sc"] if s not in rc["tal_all_sn"]]
    xt = pd.crosstab(cells.cluster, cells.label_all)
    xt[[c for c in xt.columns if c in tal_all or c == "MD"]].to_csv(out / "cluster_by_allstate_tal_label.tsv", sep="\t")
    pd.crosstab(cells.cluster[tal], cells.label_tal[tal]).to_csv(out / "cluster_by_tal_label.tsv", sep="\t")
    summ = cells[tal].groupby("unit").agg(n_tal_cluster=("r_tal", "size"), median_r=("r_tal", "median"),
                                          unassigned=("label_tal", lambda x: float((x == "unassigned").mean())))
    units.join(summ).to_csv(out / "units.tsv", sep="\t")

    den = [s for s in tal_all]
    mc, eps, inj = rc["min_tal_cells"], rc["logit_eps"], rc["tal_injured"]
    d0 = fractions(cnt, units, inj, den, mc, eps)                                  # 每个供体 × 技术（描述用）
    d0 = d0.join(fractions(cnt_all, units, inj, den, mc, eps)[["n_tal", "n_injured", "frac"]].add_suffix("_allstates"))
    d0.rename_axis("unit").to_csv(out / "unit_fractions.tsv", sep="\t")
    tabs = {}
    for name, c_ in [("tal_clusters_tal_reference", cnt), ("all_cells_all_reference", cnt_all)]:
        for mode in ("prefer_SN", "pooled"):
            cc, uu = per_donor(c_, units, den, mc, mode)
            tabs[f"{name}|{mode}"] = fractions(cc, uu, inj, den, mc, eps)
    df = pd.concat({k: v for k, v in tabs.items()}, names=["analysis", "unit"]).reset_index()
    df.to_csv(out / "donor_fractions.tsv", sep="\t", index=False)
    ab = [r for k, v in tabs.items() for r in abundance(v, g["contrasts"], k)]
    pd.DataFrame(ab).to_csv(out / "abundance.tsv", sep="\t", index=False)
    print(pd.DataFrame(ab)[["analysis", "contrast", "n_donors_a", "n_donors_b", "median_frac_a", "median_frac_b",
                            "coef_logit", "p"]].round(4).to_string(index=False), flush=True)

    # 基因程序
    pb = pseudobulk(m[tal], feats, cells.unit.values[tal], cells.label_tal.values[tal])
    pb.to_parquet(proc / "pseudobulk.parquet")
    disc = pd.read_csv(ROOT / "results" / "15_key_genes" / f"{rc['programme']['state']}_discovery.tsv", sep="\t", index_col=0)
    pr = rc["programme"]
    conc = []
    for tech in ("both", "SN_RNA", "SC_RNA"):
        sel = cnt.index if tech == "both" else cnt.index[units.reindex(cnt.index).tech == tech]
        rep = paired_programme(pb, cnt.loc[sel], rc["tal_injured"], rc["tal_normal"], rc["min_state_cells"],
                               donor_of_unit=units["sample"])
        if tech == "both":
            rep.to_csv(out / "programme_lfc.tsv", sep="\t")
        for excl, name in [(None, "all_genes"), (set(lt_genes), "without_label_transfer_genes")]:
            r = programme_concordance(disc, rep, pr["fdr"], pr["min_lfc"], pr["rep_lfc"], excl)
            conc.append({"technology": tech, "gene_set": name, "n_donors": rep.attrs["n_donors"],
                         "n_units": rep.attrs["n_units"], **r})
    pd.DataFrame(conc).to_csv(out / "programme_concordance.tsv", sep="\t", index=False)
    print(pd.DataFrame(conc).round(3).to_string(index=False), flush=True)
    outs = [out / f for f in ("units.tsv", "unit_fractions.tsv", "state_counts_tal.tsv", "state_counts_allstates.tsv", "donor_fractions.tsv",
                              "abundance.tsv", "programme_lfc.tsv", "programme_concordance.tsv", "reference_genes_tal.tsv",
                              "cluster_by_tal_label.tsv", "cluster_by_allstate_tal_label.tsv")]
    write_provenance(STAGE, inputs=[ROOT / "results" / "15_key_genes" / f"{pr['state']}_discovery.tsv"],
                     outputs=outs + [proc / "pseudobulk.parquet"], seed=cfg["seed"],
                     extra={"counts_source": "layers/counts (raw UMI) of GSE211785_Susztak_SC_SN_ATAC_merged_PreSCVI_final.h5ad",
                            "excluded_libraries": g["exclude_libs"], "n_cells": int(m.shape[0]),
                            "n_tal_cluster_cells": int(tal.sum()), "n_label_transfer_genes": len(lt_genes)})


if __name__ == "__main__":
    main()
