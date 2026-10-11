"""Stage 16：主图 2 / 3 所需的供体级表。

1. KPMP 单核 / 单细胞：每个供体的损伤 TAL（aTAL+frTAL）占 TAL、损伤 PT（aPT+frPT）占 PT（与 stage 04 相同的分母，
   按模态补齐退行/增殖亚群名），诊断与 eGFR 分组；单细胞标注是否为单核中未出现的独立供体。
2. 同一供体单核 vs 单细胞比例一致性（Spearman / Pearson，logit）。
3. GSE211785：每个供体 × 技术的 iPT 比例；GSE271188：每个供体的损伤 TAL / 损伤 PT 比例及 DKD / 原发肾小球疾病 / 对照比较。
4. KPMP 单核：损伤 TAL / PT 比例与 eGFR（分档中点）的关系，整体与校正诊断。
5. 损伤 TAL 基因程序：两个复现集均复现的上调基因，在损伤 TAL pseudobulk 中逐基因跨供体 z 标准化后取均值，与 eGFR 的关系。
6. 关键基因发现 vs 复现 log2FC 的相关；热图用前 N 个复现上调基因。
输出：results/16_fig_tables/*.tsv
"""
from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu, pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.pbstats import hedges_g, state_matrix  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

STAGE = "16_fig_tables"
EGFR_COL = "Baseline eGFR (ml/min/1.73m2) (Binned)"
EXTRA = ["dPT", "cycPT", "dTAL", "dM-TAL", "dC-TAL", "cycTAL"]


def logit(v: pd.Series, eps: float = 1e-3) -> pd.Series:
    return np.log((v + eps) / (1 - v + eps))


def egfr_mid(s) -> float:
    import re
    m = re.match(r"(\d+)-(\d+)", str(s))
    return (int(m.group(1)) + int(m.group(2)) + 1) / 2 if m else np.nan


def kpmp_fractions(counts: pd.DataFrame, k: dict) -> pd.DataFrame:
    st = k["states"]
    need = {s for v in st.values() for s in v} | set(EXTRA)
    c = counts.reindex(columns=sorted(set(counts.columns) | need), fill_value=0)
    pt = c[st["PT_normal"] + st["PT_injured"] + ["dPT", "cycPT"]].sum(1)
    tal = c[st["TAL_normal"] + st["TAL_injured"] + ["dTAL", "dM-TAL", "dC-TAL", "cycTAL"]].sum(1)
    return pd.DataFrame({
        "n_pt": pt, "pt_injured_frac": (c[st["PT_injured"]].sum(1) / pt).where(pt >= k["min_cells"]),
        "n_tal": tal, "tal_injured_frac": (c[st["TAL_injured"]].sum(1) / tal).where(tal >= k["min_cells"])})


def dx_labels(donors: pd.DataFrame, groups: dict) -> pd.Series:
    lab = pd.Series("other", index=donors.index, dtype=object)
    for name, rule in groups.items():
        (col, val), = rule.items()
        lab[donors[col].astype(str).isin(val if isinstance(val, list) else [val])] = name
    return lab


def egfr_group(mid: pd.Series, ft: dict) -> pd.Series:
    return pd.cut(mid, ft["egfr_edges"], right=False, labels=ft["egfr_labels"]).astype(object)


def kpmp_tables(cfg: dict, ft: dict, res3: Path) -> pd.DataFrame:
    k = cfg["kpmp"]
    rows = []
    sn_ids = None
    for mod in ("snRNA", "scRNA"):
        cnt = pd.read_csv(res3 / f"{mod}_cell_counts.tsv", sep="\t", index_col=0, dtype={0: str})
        don = pd.read_csv(res3 / f"{mod}_donors.tsv", sep="\t", index_col=0, dtype=str)
        cnt.index, don.index = cnt.index.astype(str), don.index.astype(str)
        f = kpmp_fractions(cnt, k).join(dx_labels(don, k["groups"]).rename("dx"))
        f["modality"] = mod
        if mod == "snRNA":
            sn_ids = set(cnt.index)
            f["egfr_bin"] = don[EGFR_COL].reindex(f.index)
            f["egfr_mid"] = f.egfr_bin.map(egfr_mid)
            f["egfr_group"] = egfr_group(f.egfr_mid, ft)
            f["independent_of_snRNA"] = False
        else:
            f["independent_of_snRNA"] = ~f.index.isin(sn_ids)
        rows.append(f.rename_axis("donor_id").reset_index())
    return pd.concat(rows, ignore_index=True)


def sn_sc_agreement(fr: pd.DataFrame) -> pd.DataFrame:
    sn = fr[fr.modality == "snRNA"].set_index("donor_id")
    sc = fr[fr.modality == "scRNA"].set_index("donor_id")
    rows, pts = [], []
    for m, lab in [("tal_injured_frac", "TAL"), ("pt_injured_frac", "PT")]:
        d = pd.concat([sn[m].rename("sn"), sc[m].rename("sc"), sn.dx], axis=1, join="inner").dropna(subset=["sn", "sc"])
        rho, p = spearmanr(d.sn, d.sc)
        r, pr = pearsonr(logit(d.sn), logit(d.sc))
        rows.append({"lineage": lab, "n_donors": len(d), "spearman": rho, "p_spearman": p,
                     "pearson_logit": r, "p_pearson_logit": pr})
        pts.append(d.assign(lineage=lab).rename_axis("donor_id").reset_index())
    return pd.DataFrame(rows), pd.concat(pts, ignore_index=True)


def gse211785(cfg: dict, ft: dict) -> pd.DataFrame:
    a = ad.read_h5ad(ROOT / cfg["paths"]["raw"] / "gse211785" / "susztak_preSCVI.h5ad", backed="r")
    obs = a.obs[["tech", "sample", "group", "Cluster_Idents"]].copy()
    obs["lib"] = obs.index.str.split("_").str[0]
    obs = obs[obs.tech.isin(["SN_RNA", "SC_RNA"]) & ~obs.lib.isin(ft["gse211785_exclude_libs"])]
    ct = obs.groupby(["sample", "tech", "Cluster_Idents"], observed=True).size().unstack(fill_value=0)
    inj = ct.reindex(columns=ft["gse211785_pt_injured"], fill_value=0).sum(1)
    pt = ct.reindex(columns=ft["gse211785_pt_normal"] + ft["gse211785_pt_injured"], fill_value=0).sum(1)
    grp = obs.drop_duplicates(["sample", "tech"]).set_index(["sample", "tech"])["group"].astype(str)
    d = pd.DataFrame({"n_pt": pt, "n_ipt": inj}).join(grp)
    d = d[(d.n_pt >= cfg["kpmp"]["min_cells"]) & d.group.isin(["DKD", "HKD", "Control"])]
    d["ipt_frac"] = d.n_ipt / d.n_pt
    return d.reset_index()


def gse271188(cfg: dict, ft: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    res = ROOT / "results" / "09_gse271188_replication"
    cnt = pd.read_csv(res / "cell_counts.tsv", sep="\t", index_col=0)
    don = pd.read_csv(res / "donors.tsv", sep="\t", index_col=0)
    ax = cfg["histology"]["axes"]
    c = cnt.reindex(columns=sorted(set(cnt.columns) | set(ax["tal_den"]) | set(ax["pt_den"])), fill_value=0)
    mc = cfg["kpmp"]["min_cells"]
    d = pd.DataFrame({"n_tal": c[ax["tal_den"]].sum(1), "n_pt": c[ax["pt_den"]].sum(1)})
    d["tal_injured_frac"] = (c[ax["tal_num"]].sum(1) / d.n_tal).where(d.n_tal >= mc)
    d["pt_injured_frac"] = (c[ax["pt_num"]].sum(1) / d.n_pt).where(d.n_pt >= mc)
    d = d.join(don[["diagnosis"]])
    d["group"] = d.diagnosis.where(~d.diagnosis.isin(ft["gse271188_glomerular"]), "GLOM_DISEASE")
    rows = []
    for m in ("tal_injured_frac", "pt_injured_frac"):
        for a_, b_ in [("DKD", "GLOM_DISEASE"), ("DKD", "Control"), ("GLOM_DISEASE", "Control")]:
            x, y = d.loc[d.group == a_, m].dropna(), d.loc[d.group == b_, m].dropna()
            rows.append({"measure": m, "contrast": f"{a_}_vs_{b_}", "n_a": len(x), "n_b": len(y),
                         "median_a": x.median(), "median_b": y.median(),
                         "hedges_g_logit": float(hedges_g(logit(x).values[:, None], logit(y).values[:, None])[0]),
                         "p_mannwhitney": mannwhitneyu(x, y).pvalue})
    return d.rename_axis("gsm").reset_index(), pd.DataFrame(rows)


def egfr_models(d: pd.DataFrame, ycol: str, name: str, transform) -> list[dict]:
    """y 与 eGFR 中点：Spearman（全部及各诊断内）；OLS y ~ eGFR/10 + 诊断。"""
    d = d.dropna(subset=[ycol, "egfr_mid"]).copy()
    d["y"], d["e10"] = transform(d[ycol]), d.egfr_mid / 10
    rho, p = spearmanr(d[ycol], d.egfr_mid)
    rows = [{"measure": name, "subset": "all", "n": len(d), "spearman": rho, "p_spearman": p}]
    f = smf.ols("y ~ e10 + C(dx)", d).fit()
    rows[0].update({"ols_beta_per10_adj_dx": f.params["e10"], "ols_p_adj_dx": f.pvalues["e10"]})
    for dx, s in d.groupby("dx"):
        if len(s) >= 5:
            r, pp = spearmanr(s[ycol], s.egfr_mid)
            rows.append({"measure": name, "subset": dx, "n": len(s), "spearman": r, "p_spearman": pp})
    return rows


def programme(cfg: dict, ft: dict, fr_sn: pd.DataFrame, kg: pd.DataFrame) -> pd.DataFrame:
    genes = kg.index[kg.replicated_both & (kg.log2fc > 0)]
    spec = cfg["key_genes"]["states"][ft["programme_state"]]
    pb = pd.read_parquet(ROOT / cfg["paths"]["processed"] / "kpmp" / "snRNA_pseudobulk.parquet")
    cnt = pd.read_csv(ROOT / "results/03_kpmp_pseudobulk/snRNA_cell_counts.tsv", sep="\t", index_col=0, dtype={0: str})
    cnt.index = cnt.index.astype(str)
    m = state_matrix(pb, cnt, spec["injured"], cfg["kpmp"]["min_cells"])
    g = [x for x in genes if x in m.columns]
    z = (m[g] - m[g].mean()) / m[g].std()
    s = pd.DataFrame({"programme_score": z.mean(1), "n_genes": len(g)})
    s.index = s.index.astype(str)
    return s.join(fr_sn.set_index("donor_id")[["dx", "egfr_bin", "egfr_mid", "egfr_group", "n_tal", "tal_injured_frac"]])


def main() -> None:
    cfg = load_config()
    ft = cfg["fig_tables"]
    set_global_seed(cfg["seed"])
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    res3 = ROOT / "results" / "03_kpmp_pseudobulk"
    outs = {}

    fr = kpmp_tables(cfg, ft, res3)
    outs["kpmp_donor_fractions.tsv"] = fr
    agree, agree_pts = sn_sc_agreement(fr)
    outs["sn_vs_sc_agreement.tsv"], outs["sn_vs_sc_donors.tsv"] = agree, agree_pts
    outs["gse211785_donor_fractions.tsv"] = gse211785(cfg, ft)
    outs["gse271188_donor_fractions.tsv"], outs["gse271188_abundance.tsv"] = gse271188(cfg, ft)

    sn = fr[(fr.modality == "snRNA") & fr.dx.isin(ft["dx_order"])]
    eg = egfr_models(sn, "tal_injured_frac", "tal_injured_frac", logit) + \
        egfr_models(sn, "pt_injured_frac", "pt_injured_frac", logit)

    kg = pd.read_csv(ROOT / "results/15_key_genes" / f"{ft['programme_state']}_key_genes.tsv", sep="\t", index_col=0)
    ps = programme(cfg, ft, fr[fr.modality == "snRNA"], kg)
    outs["tal_programme_score.tsv"] = ps.rename_axis("donor_id").reset_index()
    ps3 = ps[ps.dx.isin(ft["dx_order"])]
    eg += egfr_models(ps3, "programme_score", "tal_programme_score", lambda v: v)
    outs["egfr_associations.tsv"] = pd.DataFrame(eg)

    rep = []
    for col, lab in [("log2fc_kpmp_sc", "KPMP_scRNA"), ("log2fc_gse271188", "GSE271188")]:
        d = kg[["log2fc", col]].dropna()
        r, p = pearsonr(d.log2fc, d[col])
        rcol = "replicated_kpmp_sc" if lab == "KPMP_scRNA" else "replicated_gse271188"
        rep.append({"replication_set": lab, "n_discovery_genes": len(kg), "n_measured": len(d), "pearson": r, "p": p,
                    "replication_rate": kg[rcol].mean(), "n_replicated": int(kg[rcol].sum())})
    up = kg[kg.replicated_both & (kg.log2fc > 0)]
    rep.append({"replication_set": "both_up", "n_discovery_genes": int((kg.log2fc > 0).sum()), "n_measured": len(up),
                "n_replicated": len(up), "n_egfr_fdr05": int((up.egfr_fdr < 0.05).sum()),
                "n_egfr_fdr05_negative": int(((up.egfr_fdr < 0.05) & (up.egfr_beta_rank < 0)).sum())})
    outs["key_gene_replication.tsv"] = pd.DataFrame(rep)
    lf = up[["log2fc", "log2fc_kpmp_sc", "log2fc_gse271188"]]
    top = up.assign(min_log2fc=lf.min(1)).sort_values("min_log2fc", ascending=False).head(ft["heatmap_top_n"])
    outs["key_gene_heatmap.tsv"] = top[["log2fc", "log2fc_kpmp_sc", "log2fc_gse271188", "min_log2fc",
                                        "egfr_beta_rank", "egfr_fdr"]].rename_axis("gene").reset_index()

    paths = []
    for name, df in outs.items():
        p = out / name
        df.to_csv(p, sep="\t", index=False)
        paths.append(p)
    print(agree.round(3).to_string(index=False))
    print(outs["gse271188_abundance.tsv"].round(4).to_string(index=False))
    print(outs["egfr_associations.tsv"].round(4).to_string(index=False))
    print(outs["key_gene_replication.tsv"].round(3).to_string(index=False))
    write_provenance(STAGE, inputs=[res3 / "snRNA_cell_counts.tsv", res3 / "scRNA_cell_counts.tsv",
                                    res3 / "snRNA_donors.tsv", res3 / "scRNA_donors.tsv",
                                    ROOT / "results/09_gse271188_replication/cell_counts.tsv",
                                    ROOT / "results/15_key_genes" / f"{ft['programme_state']}_key_genes.tsv"],
                     outputs=paths, seed=cfg["seed"], extra={"params": ft})


if __name__ == "__main__":
    main()
