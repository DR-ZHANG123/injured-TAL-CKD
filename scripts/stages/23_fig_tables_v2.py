"""Stage 23：Fig 3 / Fig 4（v2）所需、前序 stage 未直接存档的两张表。

1. programme_null_draws.tsv：stage 18 的表达 × 长度匹配随机基因集零分布的逐次抽样值（每个基因集 × 性状 1000 次的平均 MAGMA Z）。
   完全复用 stage 18 的函数、种子与抽样顺序；写出前核对均值 / 标准差与 results/18_programme_genetics/empirical_null.tsv 一致。
2. genetic_support_genes.tsv：健康 TAL 基因（stage 18 的 tal_healthy 集）与损伤 TAL 上调程序中，
   在肾脏 eQTL 的 SMR 中达到显著（07e，Bonferroni）的基因；附 HEIDI、coloc PP4（07d/07f）、
   SMR 效应方向（统一为“表达越高，肾功能越好”为正）与损伤 TAL 相对正常 TAL 的 log2FC（stage 15 发现与复现）。
3. egf_egfr_standardized.tsv：TAL 中 EGF 表达与 eGFR 的标准化回归系数（两者在分析样本内 z 化），
   依次校正：无、诊断、诊断 + 损伤 TAL 比例（logit），及加年龄、性别；全部 TAL 与正常 TAL 两种 pseudobulk。
   诊断及诊断 + 损伤比例两个模型的 P 值与 results/22_tal_states/egfr_associations.tsv 核对一致。
参数：config/run.yaml → fig_tables_v2:
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import tal_states_lib as tl  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

STAGE = "23_fig_tables_v2"
R18 = ROOT / "results" / "18_programme_genetics"


def _stage18():
    spec = importlib.util.spec_from_file_location("stage18", Path(__file__).with_name("18_programme_genetics.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def null_draws(cfg: dict, s18) -> pd.DataFrame:
    """按 stage 18 main() 的同一顺序重放随机数，返回逐次零分布与观测值。"""
    pg = cfg["programme_genetics"]
    rng = np.random.default_rng(cfg["seed"])
    sets = s18.build_sets(pg)
    expr = s18.kidney_expression(cfg, pg["min_cells_total"])
    zs = {}
    for t in pg["traits"]:
        mg = pd.read_csv(ROOT / "results" / "07b_magma" / f"{t}_genes.tsv", sep="\t")
        zs[t] = mg.dropna(subset=["SYMBOL"]).drop_duplicates("SYMBOL").set_index("SYMBOL")
    common = expr.index.intersection(set.intersection(*[set(m.index) for m in zs.values()]))
    z = pd.DataFrame({t: zs[t].loc[common, "ZSTAT"] for t in pg["traits"]})
    ref = zs[pg["traits"][0]].loc[common]
    cov = pd.DataFrame({"expr_bin": pd.qcut(expr[common].rank(method="first"), pg["n_bins_expr"], labels=False),
                        "len_bin": pd.qcut((ref.STOP - ref.START).rank(method="first"), pg["n_bins_length"],
                                           labels=False)}, index=common)
    cov["bin"] = cov.expr_bin * 100 + cov.len_bin
    rows = []
    for name, genes in sets.items():
        obs, null = s18.matched_null(z, cov, genes, pg, rng)
        for j, t in enumerate(pg["traits"]):
            rows.append(pd.DataFrame({"gene_set": name, "trait": t, "draw": np.arange(len(null)),
                                      "null_mean_z": null[:, j], "observed_mean_z": obs[j]}))
    out = pd.concat(rows, ignore_index=True)
    chk = out.groupby(["trait", "gene_set"]).agg(m=("null_mean_z", "mean"), s=("null_mean_z", "std"),
                                                 o=("observed_mean_z", "first"))
    ref_tab = pd.read_csv(R18 / "empirical_null.tsv", sep="\t").set_index(["trait", "gene_set"])
    chk = chk.join(ref_tab[["null_mean", "null_sd", "mean_z"]])
    dev = np.abs(chk[["m", "s", "o"]].values - chk[["null_mean", "null_sd", "mean_z"]].values).max()
    if not dev < 1e-9:
        raise SystemExit(f"零分布与 stage 18 不一致（最大偏差 {dev:.3g}）")
    print(f"零分布与 stage 18 一致（最大偏差 {dev:.2e}）", flush=True)
    return out


def best_smr(symbols: set[str], traits: list[str]) -> pd.DataFrame:
    """每个 性状 × 基因：优先 SMR 显著且 HEIDI 通过的组织中 P 最小者，否则 SMR 显著者中 P 最小者（同 stage 18）。"""
    smr = pd.concat([pd.read_csv(f, sep="\t") for f in sorted((ROOT / "results" / "07e_smr").glob("*_smr.tsv"))])
    smr = smr[smr.symbol.isin(symbols) & smr.trait.isin(traits) & smr.smr_sig.eq(True)]
    smr = smr.assign(rank_key=~smr.heidi_pass.eq(True)).sort_values(["rank_key", "p_smr"])
    return smr.drop_duplicates(["trait", "symbol"])


def support_table(cfg: dict) -> pd.DataFrame:
    p = cfg["fig_tables_v2"]
    adverse = cfg["programme_genetics"]["adverse_sign"]
    gs = pd.read_csv(R18 / "gene_sets.tsv", sep="\t")
    member = {}
    for name in p["support_sets"]:
        for g in gs.symbol[gs.gene_set == name]:
            member.setdefault(g, name)
    sm = best_smr(set(member), p["support_traits"])
    sm["gene_set"] = sm.symbol.map(member)
    sm["z_smr"] = sm.b_smr / sm.se_smr
    # 统一方向：正值 = 表达越高，肾功能越好（eGFR 升高或 UACR 降低）
    sm["z_protective"] = -sm.z_smr * sm.trait.map(adverse)
    ev = pd.read_csv(ROOT / "results" / "07f_genetic_convergence" / "genetic_evidence_by_gene.tsv", sep="\t")
    sm = sm.merge(ev[["trait", "symbol", "magma_p", "coloc_pp4_max", "coloc_pp4_ge_flag", "n_lines"]],
                  on=["trait", "symbol"], how="left")
    # 每个基因保留一个性状：优先 HEIDI 通过，再取 SMR P 最小
    sm = sm.sort_values(["rank_key", "p_smr"]).drop_duplicates("symbol")
    disc = pd.read_csv(ROOT / "results" / "15_key_genes" / f"{p['tal_state']}_discovery.tsv", sep="\t", index_col=0)
    key = pd.read_csv(ROOT / "results" / "15_key_genes" / f"{p['tal_state']}_key_genes.tsv", sep="\t", index_col=0)
    sm["log2fc_injured_vs_normal"] = sm.symbol.map(disc.log2fc)
    sm["fdr_injured_vs_normal"] = sm.symbol.map(disc.fdr)
    sm["replicated_both"] = sm.symbol.map(key.replicated_both).eq(True)
    sm["lost_in_injury"] = sm.log2fc_injured_vs_normal < 0
    sm["higher_expression_protective"] = sm.z_protective > 0
    sm["loss_harmful_concordant"] = sm.lost_in_injury == sm.higher_expression_protective
    cols = ["symbol", "gene_set", "trait", "eqtl", "b_smr", "se_smr", "z_smr", "z_protective", "p_smr", "heidi_pass",
            "p_heidi", "coloc_pp4_max", "coloc_pp4_ge_flag", "magma_p", "n_lines", "log2fc_injured_vs_normal",
            "fdr_injured_vs_normal", "replicated_both", "lost_in_injury", "higher_expression_protective",
            "loss_harmful_concordant"]
    return sm[cols].sort_values("p_smr").reset_index(drop=True)


def egf_models(cfg: dict) -> pd.DataFrame:
    p = cfg["fig_tables_v2"]
    r22 = ROOT / "results" / "22_tal_states"
    g = pd.read_csv(r22 / "participant_tal_genes.tsv", sep="\t", index_col=0)
    f = pd.read_csv(r22 / "participant_fractions.tsv", sep="\t")
    f = f[f.modality == "snRNA"].set_index("participant")[["Age", "sex"]]
    d = g[g.dx.isin(p["dx_groups"])].join(f)
    d["inj_logit"] = tl.logit(d.injured_frac)   # 与 stage 22 相同的 logit 定义
    d["age"] = d.Age.astype(str).str.extract(r"(\d+)-(\d+)").astype(float).sum(1).add(1).div(2).where(
        d.Age.astype(str).str.match(r"\s*\d+-\d+"))
    d["male"] = d.sex.astype(str).str.lower().str.startswith("m").astype(float)
    models = {"none": [], "dx": ["C(dx)"], "dx+injured_frac": ["C(dx)", "inj_logit"],
              "dx+age+sex": ["C(dx)", "age", "male"], "dx+age+sex+injured_frac": ["C(dx)", "age", "male", "inj_logit"]}
    rows = []
    ref = pd.read_csv(r22 / "egfr_associations.tsv", sep="\t")
    ref = ref[(ref.subset == "all")].set_index(["measure", "adjusted_for"]).sort_index()
    for pool in ("all_TAL", "normal_TAL"):
        col = f"{pool}_{p['gene']}"
        base = d.dropna(subset=[col, "egfr_mid", "inj_logit"])
        for name, cov in models.items():
            dd = base.dropna(subset=["age"]) if "age" in cov else base
            dd = dd.assign(yz=(dd[col] - dd[col].mean()) / dd[col].std(), ez=(dd.egfr_mid - dd.egfr_mid.mean()) /
                           dd.egfr_mid.std())
            fit = smf.ols("yz ~ " + " + ".join(["ez"] + cov), dd).fit()
            ci = fit.conf_int().loc["ez"]
            row = {"pool": pool, "gene": p["gene"], "model": name, "n": len(dd), "beta_std": fit.params["ez"],
                   "ci_low": ci[0], "ci_high": ci[1], "p": fit.pvalues["ez"],
                   "beta_injured_logit_std": fit.params.get("inj_logit", np.nan),
                   "p_injured_logit": fit.pvalues.get("inj_logit", np.nan)}
            key = {"dx": "dx", "dx+injured_frac": "dx+inj_logit"}.get(name)
            if key:
                pr = ref.loc[(f"{p['gene']}_{pool}", key), "p"]
                if not np.isclose(pr, row["p"], rtol=1e-6):
                    raise SystemExit(f"{pool} {name}：P {row['p']:.6g} 与 stage 22 {pr:.6g} 不一致")
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    s18 = _stage18()

    nd = null_draws(cfg, s18)
    nd.to_csv(out / "programme_null_draws.tsv", sep="\t", index=False, float_format="%.6g")

    sup = support_table(cfg)
    sup.to_csv(out / "genetic_support_genes.tsv", sep="\t", index=False)
    summ = sup.groupby("gene_set").agg(n=("symbol", "size"), n_heidi=("heidi_pass", "sum"),
                                       n_lost=("lost_in_injury", "sum"),
                                       n_protective=("higher_expression_protective", "sum"),
                                       n_concordant=("loss_harmful_concordant", "sum")).reset_index()
    hp = sup[sup.heidi_pass.eq(True)].groupby("gene_set").loss_harmful_concordant.agg(["size", "sum"])
    summ = summ.join(hp.rename(columns={"size": "n_heidi_pass", "sum": "n_concordant_heidi_pass"}), on="gene_set")
    summ.to_csv(out / "genetic_support_summary.tsv", sep="\t", index=False)
    print(summ.to_string(index=False), flush=True)
    print(sup[["symbol", "gene_set", "trait", "z_protective", "heidi_pass", "log2fc_injured_vs_normal"]]
          .to_string(index=False), flush=True)

    egf = egf_models(cfg)
    egf.to_csv(out / "egf_egfr_standardized.tsv", sep="\t", index=False)
    print(egf.round(4).to_string(index=False), flush=True)

    pg = cfg["programme_genetics"]
    p = cfg["fig_tables_v2"]
    inputs = [ROOT / "results/22_tal_states/participant_tal_genes.tsv",
              ROOT / "results/22_tal_states/participant_fractions.tsv",
              ROOT / "results/22_tal_states/egfr_associations.tsv",R18 / "gene_sets.tsv", R18 / "empirical_null.tsv",
              ROOT / "results/15_key_genes" / f"{p['tal_state']}_discovery.tsv",
              ROOT / "results/15_key_genes" / f"{p['tal_state']}_key_genes.tsv",
              ROOT / "results/15_key_genes" / f"{pg['tal_state']}_key_genes.tsv",
              ROOT / "results/15_key_genes" / f"{pg['pt_state']}_key_genes.tsv",
              ROOT / "results/13_gwas_cell_enrichment/specific_genes.tsv",
              ROOT / "results/07f_genetic_convergence/genetic_evidence_by_gene.tsv",
              ROOT / cfg["paths"]["processed"] / "kpmp" / "snRNA_pseudobulk.parquet",
              ROOT / "results/03_kpmp_pseudobulk/snRNA_cell_counts.tsv"]
    inputs += [ROOT / "results/07b_magma" / f"{t}_genes.tsv" for t in pg["traits"]]
    inputs += sorted((ROOT / "results/07e_smr").glob("*_smr.tsv"))
    write_provenance(STAGE, inputs, [out / f for f in ("programme_null_draws.tsv", "genetic_support_genes.tsv",
                                                       "genetic_support_summary.tsv", "egf_egfr_standardized.tsv")],
                     seed=cfg["seed"], extra={"null_reproduces": "results/18_programme_genetics/empirical_null.tsv"})


if __name__ == "__main__":
    main()
