"""Stage 19b：GSE195460（DKD 5 名 vs 肿瘤肾切除对照 6 名，肾皮质单核 RNA）中损伤 TAL 扩增、临床关联与基因程序复现。

1. 注释迁移与 stage 09 相同：KPMP 单核全亚群参照，每个细胞核取相关最高的亚群（r < min_r 为未注释）。
   TAL 谱系 = 注释为任一 TAL 状态的细胞核；敏感性：TAL 谱系细胞核再用只在 TAL 状态之间计算的参照细分。
2. 每个供体：损伤 TAL（aTAL+frTAL）占 TAL；DKD vs 对照（Mann–Whitney、logit 的 Hedges g，及校正文库化学的 OLS）；
   与 eGFR、IFTA（系列矩阵；对照仅 Control6 有临床数据）的 Spearman 相关。
3. 损伤 TAL 基因程序：供体内 损伤 vs 正常 TAL 配对 log2CPM 差，与 KPMP 单核发现（stage 15）比较；剔除注释迁移所用基因后重算。
输出：results/19b_gse195460/*.tsv；data/processed/gse195460/{pseudobulk.parquet, cell_labels.parquet}
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.geo import read_series_matrix  # noqa: E402
from lib.pbstats import hedges_g  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from lib.tal_transfer import annotate, paired_programme, programme_concordance, pseudobulk, reference  # noqa: E402

STAGE = "19b_gse195460"


def read_10x_h5(path: Path) -> tuple[sp.csr_matrix, pd.Index, pd.Index]:
    with h5py.File(path, "r") as f:
        g = f["matrix"]
        m = sp.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=tuple(g["shape"][:])).T.tocsr()
        feats = pd.Index([x.decode() for x in g["features/name"][:]])
        bc = pd.Index([x.decode() for x in g["barcodes"][:]])
    return m, feats, bc


def clinical(path: Path, ifta_ord: dict) -> pd.DataFrame:
    smp, _ = read_series_matrix(path)
    ch = smp.filter(like="char_").fillna("").agg(" | ".join, axis=1)
    d = pd.DataFrame({"title": smp["title"],
                      "donor": smp["title"].str.split("_").str[0],
                      "egfr": ch.str.extract(r"egfr: ([\d.]+)", expand=False).astype(float),
                      "ifta": ch.str.extract(r"ifta: ([^|]+?)\s*(?:\||$)", expand=False),
                      "gs": ch.str.extract(r"glomerulosclerosis: ([^|]+?)\s*(?:\||$)", expand=False),
                      "rna": smp["title"].str.contains("snRNA")})
    d = d.sort_values("rna", ascending=False).drop_duplicates("donor")      # 同一供体优先取 RNA 文库的记录
    d["ifta_ord"] = d.ifta.map(ifta_ord)
    return d.set_index("donor")[["egfr", "ifta", "ifta_ord", "gs"]]


def frac_table(cnt: pd.DataFrame, inj, den, min_cells, eps) -> pd.DataFrame:
    c = cnt.reindex(columns=sorted(set(cnt.columns) | set(den)), fill_value=0)
    d = pd.DataFrame({"n_tal": c[den].sum(1), "n_injured": c[inj].sum(1), "n_aTAL": c["aTAL"], "n_frTAL": c["frTAL"]})
    d["frac"] = (d.n_injured / d.n_tal).where(d.n_tal >= min_cells)
    d["logit"] = np.log((d.frac + eps) / (1 - d.frac + eps))
    return d


def tests(d: pd.DataFrame, analysis: str) -> list[dict]:
    x, y = d[d.group == "DKD"].dropna(subset=["frac"]), d[d.group == "Control"].dropna(subset=["frac"])
    f = smf.ols("logit ~ C(group, Treatment('Control')) + C(chem)", d.dropna(subset=["frac"])).fit()
    k = [p for p in f.params.index if p.startswith("C(group")][0]
    rows = [{"analysis": analysis, "test": "DKD_vs_Control", "n": f"{len(x)}v{len(y)}",
             "median_dkd": float(x.frac.median()), "median_control": float(y.frac.median()),
             "hedges_g_logit": float(hedges_g(x.logit.values[:, None], y.logit.values[:, None])[0]),
             "p_mannwhitney": float(mannwhitneyu(x.frac, y.frac).pvalue),
             "coef_logit_adj_chem": float(f.params[k]), "p_adj_chem": float(f.pvalues[k])}]
    for var in ("egfr", "ifta_ord"):
        for sub, s in [("all_with_clinical", d), ("DKD_only", d[d.group == "DKD"])]:
            s = s.dropna(subset=["frac", var])
            if len(s) >= 4:
                r, p = spearmanr(s.frac, s[var])
                rows.append({"analysis": analysis, "test": f"spearman_{var}_{sub}", "n": str(len(s)),
                             "spearman_rho": float(r), "p_spearman": float(p)})
    return rows


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    rc, g = cfg["replication2"], cfg["replication2"]["gse195460"]
    raw = ROOT / g["raw_dir"]
    out = ROOT / "results" / STAGE
    proc = ROOT / cfg["paths"]["processed"] / "gse195460"
    out.mkdir(parents=True, exist_ok=True)
    proc.mkdir(parents=True, exist_ok=True)

    pbk = pd.read_parquet(ROOT / cfg["paths"]["processed"] / "kpmp" / "snRNA_pseudobulk.parquet")
    ref, genes = reference(pbk, rc["all_states"]["n_per"])
    ts = rc["tal_states"]
    tref, tgenes = reference(pbk, ts["n_per"], ts["snRNA"], ts["drop_gene_regex"])
    tal_states = rc["tal_all_sn"]

    mats, cells, summ = [], [], []
    for lib in g["libraries"]:
        path = next(raw.glob(f"*_{lib}_filtered_feature_bc_matrix.h5"))
        m, feats, bc = read_10x_h5(path)
        keep = np.asarray(m.sum(1)).ravel() >= rc["min_umi"]
        m, bc = m[keep], bc[keep]
        lab, r = annotate(m, feats, ref, genes)
        lab = np.where(r >= rc["all_states"]["min_r"], lab, "unassigned")
        istal = np.isin(lab, tal_states)
        tl, tr = annotate(m[istal], feats, tref, tgenes) if istal.any() else (np.array([]), np.array([]))
        lab_t = np.full(len(lab), None, dtype=object)
        lab_t[istal] = np.where(tr >= ts["min_r"], tl, "unassigned")
        mats.append((m, feats))
        cells.append(pd.DataFrame({"donor": lib, "label_all": lab, "r_all": r, "label_tal": lab_t},
                                  index=lib + "_" + bc))
        summ.append({"donor": lib, "n_nuclei": int(keep.sum()), "median_r": float(np.median(r)),
                     "unassigned": float((lab == "unassigned").mean()), "n_tal_labelled": int(istal.sum())})
        print(lib, summ[-1], flush=True)
    feats0 = pd.Index(sorted(set.intersection(*[set(f) for _, f in mats])))   # 各文库共有基因（重名取第一个）
    m = sp.vstack([x[:, pd.Series(np.arange(len(f)), index=f)[~f.duplicated()].reindex(feats0).values]
                   for x, f in mats]).tocsr()
    cells = pd.concat(cells)
    cells.to_parquet(proc / "cell_labels.parquet")

    clin = clinical(raw / "GSE195460_series_matrix.txt.gz", g["ifta_ord"])
    donors = pd.DataFrame(g["libraries"]).T.join(clin).join(pd.DataFrame(summ).set_index("donor"))
    cnt = pd.crosstab(cells.donor, cells.label_all)
    cnt_t = pd.crosstab(cells.donor[cells.label_tal.notna()], cells.label_tal[cells.label_tal.notna()])
    cnt.to_csv(out / "state_counts_allstates.tsv", sep="\t")
    cnt_t.to_csv(out / "state_counts_tal_refined.tsv", sep="\t")

    d1 = donors.join(frac_table(cnt, rc["tal_injured"], tal_states, rc["min_tal_cells"], rc["logit_eps"]))
    d2 = frac_table(cnt_t, rc["tal_injured"], tal_states, rc["min_tal_cells"], rc["logit_eps"])
    df = d1.join(d2[["n_tal", "n_injured", "frac"]].add_suffix("_tal_refined"))
    df.rename_axis("donor").to_csv(out / "donor_fractions.tsv", sep="\t")
    print(df[["group", "chem", "egfr", "ifta", "n_tal", "frac", "frac_tal_refined"]].round(3).to_string(), flush=True)
    res = tests(d1, "all_state_labels") + tests(donors.join(d2), "tal_refined_labels")
    pd.DataFrame(res).to_csv(out / "abundance_clinical.tsv", sep="\t", index=False)
    print(pd.DataFrame(res).round(4).to_string(index=False), flush=True)

    pb = pseudobulk(m, feats0, cells.donor.values, cells.label_all.values)
    pb.to_parquet(proc / "pseudobulk.parquet")
    tal_mask = cells.label_tal.notna().values
    pb_t = pseudobulk(m[tal_mask], feats0, cells.donor.values[tal_mask], cells.label_tal.values[tal_mask])
    disc = pd.read_csv(ROOT / "results" / "15_key_genes" / f"{rc['programme']['state']}_discovery.tsv", sep="\t", index_col=0)
    pr = rc["programme"]
    conc = []
    for name, p_, c_, lt in [("all_state_labels", pb, cnt, set(genes)),
                             ("tal_refined_labels", pb_t, cnt_t, set(genes) | set(tgenes))]:
        rep = paired_programme(p_, c_, rc["tal_injured"], rc["tal_normal"], rc["min_state_cells"])
        rep.to_csv(out / f"programme_lfc_{name}.tsv", sep="\t")
        for excl, gs in [(None, "all_genes"), (lt, "without_label_transfer_genes")]:
            conc.append({"labels": name, "gene_set": gs, "n_donors": rep.attrs["n_donors"],
                         **programme_concordance(disc, rep, pr["fdr"], pr["min_lfc"], pr["rep_lfc"], excl)})
    pd.DataFrame(conc).to_csv(out / "programme_concordance.tsv", sep="\t", index=False)
    print(pd.DataFrame(conc).round(3).to_string(index=False), flush=True)
    outs = [out / f for f in ("state_counts_allstates.tsv", "state_counts_tal_refined.tsv", "donor_fractions.tsv",
                              "abundance_clinical.tsv", "programme_lfc_all_state_labels.tsv",
                              "programme_lfc_tal_refined_labels.tsv", "programme_concordance.tsv")]
    write_provenance(STAGE, inputs=sorted(raw.glob("*.h5")) + [raw / "GSE195460_series_matrix.txt.gz"],
                     outputs=outs + [proc / "pseudobulk.parquet"], seed=cfg["seed"],
                     extra={"n_label_transfer_genes_all": len(genes), "n_label_transfer_genes_tal": len(tgenes),
                            "clinical_note": "eGFR/IFTA from GEO characteristics; controls 1-5 have none"})


if __name__ == "__main__":
    main()
