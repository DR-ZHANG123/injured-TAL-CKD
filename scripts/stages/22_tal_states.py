"""Stage 22：把损伤 TAL 拆成适应性（aTAL）与修复失败（frTAL）两种状态，在参与者水平回答：

1. 每名参与者 aTAL、frTAL 占全部 TAL 的比例，及 frTAL 在损伤 TAL 中的份额；DKD / HKD / 参照差异；
   与 eGFR 的关系（校正诊断；另加年龄、性别）；单核 vs 单细胞同一参与者一致性；单细胞独立参与者复现。
2. 程序–eGFR 关联是否来自组成：580 基因程序得分分别在 aTAL、frTAL pseudobulk 中计算并与 eGFR 关联；
   合并损伤 pseudobulk 的得分再校正 frTAL 份额；逐基因 eGFR 关联在各状态中的情况（含同一批参与者的配对比较）。
3. TAL 功能基因与尿液损伤标志物：发现（单核配对）与复现（单细胞独立参与者、GSE271188）中的 log2FC / FDR；
   参与者水平 EGF、UMOD（全部 TAL 及正常 TAL pseudobulk）与损伤 TAL 比例和 eGFR 的关系。
4. 只用皮质 TAL 分母的损伤比例 vs 主定义。
5. 对照人群：AKI 与 Diabetes Mellitus - Resilient 参与者 vs 参照、DKD、HKD。
输出：results/22_tal_states/*.tsv
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, spearmanr, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import tal_states_lib as tl  # noqa: E402
from lib.pbstats import state_matrix  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

STAGE = "22_tal_states"
EGFR_COL = "Baseline eGFR (ml/min/1.73m2) (Binned)"
FRACS = ["aTAL_frac", "frTAL_frac", "injured_frac", "frTAL_share_of_injured", "injured_frac_cortical"]
RES3 = ROOT / "results" / "03_kpmp_pseudobulk"


def _stage15():
    spec = importlib.util.spec_from_file_location("stage15", Path(__file__).with_name("15_key_genes.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_kpmp(cfg: dict, ts: dict) -> dict:
    k, clin = cfg["kpmp"], pd.read_csv(ROOT / "metadata/kpmp_participants_clinical_bins.tsv", sep="\t", dtype=str)
    rep_egfr = clin.set_index("redcap_id").baseline_egfr.map(tl.bin_mid)
    out = {}
    for mod, den, cort in (("snRNA", ts["tal_den_sn"], ts["cortical_den_sn"]),
                           ("scRNA", ts["tal_den_sc"], ts["cortical_den_sc"])):
        cnt = pd.read_csv(RES3 / f"{mod}_cell_counts.tsv", sep="\t", index_col=0, dtype={0: str})
        don = pd.read_csv(RES3 / f"{mod}_donors.tsv", sep="\t", index_col=0, dtype=str)
        cnt.index, don.index = cnt.index.astype(str), don.index.astype(str)
        f = tl.fractions(cnt, den, cort, ts["medullary"], k["min_cells"], ts["min_injured_cells"])
        f = f.join(tl.labels(don, k["groups"], ts["comparators"]).rename("dx"))
        f = f.join(don[["ConditionCategory", "Age", "sex"]])
        f["tissue"] = don.get("TissueCollection", don.get("Tissue.Type"))
        if mod == "snRNA":
            f["egfr_mid"] = don[EGFR_COL].map(tl.bin_mid)
            f["egfr_source"] = "atlas"
            f["independent_of_snRNA"] = False
            sn_ids = set(cnt.index)
        else:
            f["egfr_mid"] = rep_egfr.reindex(f.index).values
            f["egfr_source"] = "repository"
            f["independent_of_snRNA"] = ~f.index.isin(sn_ids)
        f["modality"] = mod
        out[mod] = (f, cnt)
    return out


def agreement(sn: pd.DataFrame, sc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in FRACS:
        d = pd.concat([sn[m].rename("sn"), sc[m].rename("sc")], axis=1, join="inner").dropna()
        rho, p = spearmanr(d.sn, d.sc)
        r, pr = pearsonr(tl.logit(d.sn), tl.logit(d.sc))
        rows.append({"measure": m, "n": len(d), "spearman": rho, "p_spearman": p, "pearson_logit": r, "p_pearson_logit": pr})
    return pd.DataFrame(rows)


def group_summary(f: pd.DataFrame, modality: str) -> pd.DataFrame:
    g = f.groupby("dx")[FRACS + ["medullary_share"]]
    s = pd.concat({"n": g.count(), "median": g.median(), "q25": g.quantile(0.25), "q75": g.quantile(0.75)}, axis=1)
    s = s.stack(level=1, future_stack=True).rename_axis(["dx", "measure"]).reset_index()
    return s.assign(modality=modality)


def fraction_egfr(f: pd.DataFrame, dx3: list[str], cohort: str) -> list[dict]:
    d = f[f.dx.isin(dx3)]
    rows = []
    for m in FRACS:
        rows += tl.egfr_models(d, m, m, tl.logit, cohort=cohort)
    d = d.assign(frTAL_share_logit=tl.logit(d.frTAL_share_of_injured))
    rows += tl.egfr_models(d, "injured_frac", "injured_frac", tl.logit, ["frTAL_share_logit"], cohort=cohort)
    return rows


def programme_part(cfg, ts, pb, cnt, f, genes, dx3):
    mc = cfg["kpmp"]["min_cells"]
    mats = {s: state_matrix(pb, cnt, v, mc) for s, v in ts["programme_states"].items()}
    scores = pd.DataFrame({f"score_{s}": tl.zscore_mean(m, genes) for s, m in mats.items()})
    raw = pd.DataFrame({f"mean_logcpm_{s}": m[[g for g in genes if g in m.columns]].mean(1) for s, m in mats.items()})
    d = f.join(scores).join(raw)
    d["frTAL_share_logit"] = tl.logit(d.frTAL_share_of_injured)
    d["matched"] = d.score_aTAL.notna() & d.score_frTAL.notna()
    d3 = d[d.dx.isin(dx3)]
    eg = []
    for subset, dd in (("all_available", d3), ("matched_aTAL_and_frTAL", d3[d3.matched])):
        for s in ts["programme_states"]:
            for r in tl.egfr_models(dd, f"score_{s}", f"score_{s}", lambda v: v):
                eg.append({**r, "donor_set": subset})
        for r in tl.egfr_models(dd, "score_pooled", "score_pooled", lambda v: v, ["frTAL_share_logit"]):
            eg.append({**r, "donor_set": subset})
    # frTAL 与 aTAL 在同一参与者中的程序基因表达（原始 logCPM，同尺度）
    pm = d[d.matched]
    a, b = mats["aTAL"].loc[pm.index], mats["frTAL"].loc[pm.index]
    g = [x for x in genes if x in a.columns and x in b.columns]
    gene_diff = (b[g] - a[g]).mean()
    w = wilcoxon(pm.mean_logcpm_frTAL, pm.mean_logcpm_aTAL)
    paired = pd.DataFrame([{"n_participants": len(pm), "n_genes": len(g),
                            "median_mean_logcpm_aTAL": pm.mean_logcpm_aTAL.median(),
                            "median_mean_logcpm_frTAL": pm.mean_logcpm_frTAL.median(),
                            "median_paired_diff_frTAL_minus_aTAL": (pm.mean_logcpm_frTAL - pm.mean_logcpm_aTAL).median(),
                            "frac_participants_frTAL_higher": float((pm.mean_logcpm_frTAL > pm.mean_logcpm_aTAL).mean()),
                            "p_wilcoxon": w.pvalue, "n_genes_higher_in_frTAL": int((gene_diff > 0).sum()),
                            "median_gene_diff_frTAL_minus_aTAL": gene_diff.median()}])
    # 逐基因 eGFR 关联：stage 15 定义（全部参与者，ConditionCategory）与 DKD/HKD/参照子集；全部可用 vs 配对同一批参与者
    gl, summ = [], []
    for dxdef, dcol, keep in (("all_participants_ConditionCategory", "ConditionCategory", None), ("DKD_HKD_REF", "dx", dx3)):
        base = d if keep is None else d[d.dx.isin(keep)]
        for subset, donors in (("all_available", None), ("matched_aTAL_and_frTAL", base.index[base.matched])):
            betas = {}
            for s, m in mats.items():
                mm = m if donors is None else m.loc[m.index.intersection(donors)]
                r = tl.gene_egfr(mm.loc[mm.index.intersection(base.index)], base, genes, dcol)
                betas[s] = r.beta_rank
                gl.append(r.assign(state=s, dx_definition=dxdef, donor_set=subset).reset_index())
                summ.append({"state": s, "dx_definition": dxdef, "donor_set": subset, "n_participants": r.attrs["n"],
                             "n_genes": len(r), "n_fdr05": int((r.fdr < ts["fdr"]).sum()),
                             "n_fdr05_negative": int(((r.fdr < ts["fdr"]) & (r.beta_rank < 0)).sum()),
                             "n_p05_negative": int(((r.p < 0.05) & (r.beta_rank < 0)).sum()),
                             "frac_beta_negative": float((r.beta_rank < 0).mean()),
                             "median_beta_rank": float(r.beta_rank.median())})
            bb = pd.DataFrame(betas).dropna()
            summ[-1]["beta_corr_aTAL_vs_frTAL"] = float(bb.aTAL.corr(bb.frTAL))
    return d, eg, paired, pd.concat(gl, ignore_index=True), pd.DataFrame(summ)


def gene_lookup(cfg, ts, s15, sn, sn_cnt, sc, sc_cnt, g2, g2_cnt) -> pd.DataFrame:
    kg, mc = cfg["key_genes"], cfg["kpmp"]["min_cells"]
    norm = kg["states"]["aTAL_frTAL"]["normal"]
    cats = {g: "TAL_function" for g in ts["function_genes"]} | {g: "urinary_injury_marker" for g in ts["urinary_markers"]}
    rows = []
    stored = pd.read_csv(ROOT / "results/15_key_genes/aTAL_frTAL_discovery.tsv", sep="\t", index_col=0)
    for state, inj in ts["discovery_states"].items():
        sets = {"KPMP_snRNA_discovery": s15.paired(sn, sn_cnt, inj, norm, mc),
                "KPMP_scRNA_independent": s15.paired(sc, sc_cnt, inj, norm, mc, donors=sc_cnt.index.difference(sn_cnt.index)),
                "GSE271188": s15.paired(g2, g2_cnt, inj, norm, kg["min_cells_gse271188"])}
        if state == "aTAL_frTAL":
            dd = sets["KPMP_snRNA_discovery"].log2fc.reindex(stored.index)
            assert np.allclose(dd, stored.log2fc, atol=1e-5, equal_nan=True), "discovery 与 stage 15 不一致"
        disc = sets["KPMP_snRNA_discovery"]
        for g, cat in cats.items():
            r = {"gene": g, "category": cat, "state": state}
            for name, t in sets.items():
                r[f"n_{name}"] = t.attrs["n_donors"]
                for c in ("log2fc", "fdr", "frac_donors_up"):
                    r[f"{c}_{name}"] = t[c].get(g, np.nan) if len(t) else np.nan
            ld = r["log2fc_KPMP_snRNA_discovery"]
            r["discovery_sig"] = bool(r["fdr_KPMP_snRNA_discovery"] < kg["fdr"]) if pd.notna(ld) else False
            r["discovery_key_gene"] = r["discovery_sig"] and abs(ld) >= kg["min_lfc"]
            for name in ("KPMP_scRNA_independent", "GSE271188"):
                lr = r[f"log2fc_{name}"]
                r[f"replicated_{name}"] = bool(pd.notna(ld) and pd.notna(lr) and np.sign(lr) == np.sign(ld)
                                               and abs(lr) >= kg["rep_lfc"])
            r["n_discovery_genes_tested"] = len(disc)
            rows.append(r)
    return pd.DataFrame(rows)


def participant_genes(cfg, ts, pb, cnt, f, dx3):
    mc, norm = cfg["kpmp"]["min_cells"], cfg["key_genes"]["states"]["aTAL_frTAL"]["normal"]
    tabs, assoc, eg = [], [], []
    for pool, subs in (("all_TAL", ts["tal_den_sn"]), ("normal_TAL", norm)):
        m = state_matrix(pb, cnt, subs, mc)
        g = [x for x in ts["participant_genes"] if x in m.columns]
        t = m[g].add_prefix(f"{pool}_")
        tabs.append(t)
        d = f[f.dx.isin(dx3)].join(t, how="inner").assign(inj_logit=lambda x: tl.logit(x.injured_frac))
        for gene in g:
            col = f"{pool}_{gene}"
            dd = d.dropna(subset=[col, "inj_logit"])
            rho, p = spearmanr(dd[col], dd.injured_frac)
            fit = smf.ols(f"Q('{col}') ~ inj_logit + C(dx)", dd).fit()
            assoc.append({"pool": pool, "gene": gene, "n": len(dd), "spearman_vs_injured_frac": rho, "p_spearman": p,
                          "beta_per_logit_injured_adj_dx": fit.params["inj_logit"], "p_adj_dx": fit.pvalues["inj_logit"]})
            for name, ex in ((col, []), (col, ["inj_logit"])):
                for r in tl.egfr_models(d, name, f"{gene}_{pool}", lambda v: v, ex):
                    eg.append(r)
    return pd.concat(tabs, axis=1), pd.DataFrame(assoc), eg


def state_concordance(sn, sn_cnt, sc, sc_cnt, norm: list[str], mc: int) -> pd.DataFrame:
    """aTAL / frTAL 标签在单核与单细胞间是否对应：各状态相对正常 TAL 的参与者平均 log2FC 谱、
    以及 frTAL − aTAL 谱，在两平台都表达（平均 logCPM ≥ 1）的基因上做 Pearson 相关。"""
    prof, expr = {}, {}
    for mod, pb, cnt in (("sn", sn, sn_cnt), ("sc", sc, sc_cnt)):
        m = {s: state_matrix(pb, cnt, v, mc) for s, v in (("aTAL", ["aTAL"]), ("frTAL", ["frTAL"]), ("normal", norm))}
        for s in ("aTAL", "frTAL"):
            d = m[s].index.intersection(m["normal"].index)
            prof[f"{s}_vs_normal_{mod}"] = (m[s].loc[d] - m["normal"].loc[d]).mean()
        d = m["aTAL"].index.intersection(m["frTAL"].index)
        prof[f"frTAL_vs_aTAL_{mod}"] = (m["frTAL"].loc[d] - m["aTAL"].loc[d]).mean()
        prof[f"n_paired_{mod}"] = len(d)
        expr[mod] = pd.concat([m["aTAL"].mean(), m["frTAL"].mean()], axis=1).mean(1)
    genes = expr["sn"].index[expr["sn"] >= 1].intersection(expr["sc"].index[expr["sc"] >= 1])
    rows = []
    for a in ("aTAL_vs_normal_sn", "frTAL_vs_normal_sn", "frTAL_vs_aTAL_sn"):
        for b in ("aTAL_vs_normal_sc", "frTAL_vs_normal_sc", "frTAL_vs_aTAL_sc"):
            x = pd.concat([prof[a], prof[b]], axis=1, join="inner").loc[lambda t: t.index.isin(genes)].dropna()
            r, p = pearsonr(x.iloc[:, 0], x.iloc[:, 1])
            rows.append({"snRNA_profile": a, "scRNA_profile": b, "n_genes": len(x), "pearson": r, "p": p,
                         "n_participants_aTAL_and_frTAL_sn": prof["n_paired_sn"],
                         "n_participants_aTAL_and_frTAL_sc": prof["n_paired_sc"]})
    return pd.DataFrame(rows)


def cortical_vs_main(fr: dict, dx3: list[str]) -> pd.DataFrame:
    rows = []
    for mod, (f, _) in fr.items():
        for subset, d in (("all", f), ("DKD_HKD_REF", f[f.dx.isin(dx3)])):
            d = d.dropna(subset=["injured_frac", "injured_frac_cortical"])
            rho, p = spearmanr(d.injured_frac, d.injured_frac_cortical)
            r, pr = pearsonr(tl.logit(d.injured_frac), tl.logit(d.injured_frac_cortical))
            ms = spearmanr(d.medullary_share, d.injured_frac)
            rows.append({"modality": mod, "subset": subset, "n": len(d), "spearman": rho, "p_spearman": p,
                         "pearson_logit": r, "p_pearson_logit": pr,
                         "median_ratio_cortical_over_main": float((d.injured_frac_cortical / d.injured_frac).median()),
                         "spearman_medullary_share_vs_injured": ms.statistic, "p_medullary": ms.pvalue})
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    ts, dx3 = cfg["tal_states"], cfg["fig_tables"]["dx_order"]
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    proc = ROOT / cfg["paths"]["processed"]
    s15 = _stage15()
    fr = load_kpmp(cfg, ts)
    (fsn, sn_cnt), (fsc, sc_cnt) = fr["snRNA"], fr["scRNA"]
    outs = {}

    # 1 / 4 / 5：比例、组间、eGFR
    outs["participant_fractions.tsv"] = pd.concat([fsn, fsc]).rename_axis("participant").reset_index()
    outs["group_summary.tsv"] = pd.concat([group_summary(fsn, "snRNA"), group_summary(fsc, "scRNA"),
                                           group_summary(fsc[fsc.independent_of_snRNA], "scRNA_independent")])
    meas = FRACS + ["medullary_share"]
    outs["group_contrasts.tsv"] = pd.concat([tl.contrasts(fsn, meas, ts["contrasts"], "snRNA"),
                                             tl.contrasts(fsc, meas, ts["contrasts"], "scRNA"),
                                             tl.contrasts(fsc[fsc.independent_of_snRNA], meas, ts["contrasts"],
                                                          "scRNA_independent")])
    outs["sn_vs_sc_agreement.tsv"] = agreement(fsn, fsc)
    eg = fraction_egfr(fsn, dx3, "snRNA") + fraction_egfr(fsc[fsc.independent_of_snRNA], dx3, "scRNA_independent")
    outs["cortical_vs_main.tsv"] = cortical_vs_main(fr, dx3)

    # 2：程序组成
    kg = pd.read_csv(ROOT / "results/15_key_genes" / f"{cfg['fig_tables']['programme_state']}_key_genes.tsv",
                     sep="\t", index_col=0)
    genes = list(kg.index[kg.replicated_both & (kg.log2fc > 0)])
    sn = pd.read_parquet(proc / "kpmp" / "snRNA_pseudobulk.parquet")
    d, eg_p, paired, gl, gsum = programme_part(cfg, ts, sn, sn_cnt, fsn, genes, dx3)
    outs["programme_scores.tsv"] = d[[c for c in d.columns if c.startswith(("score_", "mean_logcpm_"))] +
                                     ["dx", "egfr_mid", "frTAL_share_of_injured", "matched"]].rename_axis("participant").reset_index()
    outs["programme_state_paired.tsv"], outs["programme_gene_egfr.tsv"] = paired, gl
    outs["programme_gene_egfr_summary.tsv"] = gsum

    # 3：TAL 功能基因与尿液标志物
    sc = pd.read_parquet(proc / "kpmp" / "scRNA_pseudobulk.parquet")
    g2 = pd.read_parquet(proc / "gse271188" / "pseudobulk.parquet")
    g2_cnt = pd.read_csv(ROOT / "results/09_gse271188_replication/cell_counts.tsv", sep="\t", index_col=0)
    outs["tal_gene_lookup.tsv"] = gene_lookup(cfg, ts, s15, sn, sn_cnt, sc, sc_cnt, g2, g2_cnt)
    outs["sn_sc_state_profile_concordance.tsv"] = state_concordance(
        sn, sn_cnt, sc, sc_cnt, cfg["key_genes"]["states"]["aTAL_frTAL"]["normal"], cfg["kpmp"]["min_cells"])
    del sc, g2
    pg, pa, eg_g = participant_genes(cfg, ts, sn, sn_cnt, fsn, dx3)
    outs["participant_tal_genes.tsv"] = pg.join(fsn[["dx", "egfr_mid", "injured_frac"]]).rename_axis("participant").reset_index()
    outs["tal_gene_participant_assoc.tsv"] = pa
    outs["egfr_associations.tsv"] = pd.DataFrame(eg + eg_p + eg_g)

    paths = []
    for name, df in outs.items():
        p = out / name
        df.to_csv(p, sep="\t", index=False)
        paths.append(p)
    pd.set_option("display.width", 250)
    for n in ("sn_vs_sc_agreement.tsv", "sn_sc_state_profile_concordance.tsv", "programme_state_paired.tsv", "programme_gene_egfr_summary.tsv",
              "cortical_vs_main.tsv", "tal_gene_participant_assoc.tsv"):
        print(f"== {n}\n", outs[n].round(4).to_string(index=False), flush=True)
    write_provenance(STAGE, inputs=[RES3 / "snRNA_cell_counts.tsv", RES3 / "scRNA_cell_counts.tsv",
                                    RES3 / "snRNA_donors.tsv", RES3 / "scRNA_donors.tsv",
                                    ROOT / "metadata/kpmp_participants_clinical_bins.tsv",
                                    ROOT / "results/09_gse271188_replication/cell_counts.tsv",
                                    ROOT / "results/15_key_genes/aTAL_frTAL_key_genes.tsv",
                                    ROOT / "results/15_key_genes/aTAL_frTAL_discovery.tsv"],
                     outputs=paths, seed=cfg["seed"], extra={"params": ts, "n_programme_genes": len(genes)})


if __name__ == "__main__":
    main()
