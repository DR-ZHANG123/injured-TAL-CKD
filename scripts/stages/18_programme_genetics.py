"""Stage 18：复现的损伤 TAL 基因程序是否携带肾功能遗传信号（与对照基因集比较）。

基因集（均取自 stage 15 / 13，不按遗传结果挑选）：
  1 tal_injured_up    aTAL_frTAL 关键基因中 replicated_both 且 log2fc>0
  2 tal_injured_down  同上 log2fc<0（健康 TAL 中更高）
  3 tal_healthy       集 2 ∪ stage 13 中 C-TAL / M-TAL 亚群特异基因；3b tal_healthy_specific 仅后者
  4 pt_injured_up     frPT 关键基因中 GSE271188 复现且 log2fc>0
检验：
  a. MAGMA 竞争性基因集分析（--gene-results <trait>.genes.raw --set-annot；默认校正基因大小、密度等）
  b. 敏感性：同一分析以 KPMP 单核平均表达（log）为条件（--gene-covar + condition-hide）
  c. 经验零分布：每个基因集的平均 MAGMA Z vs 按平均表达十分位 × 基因长度十分位匹配的随机基因集
另列出程序基因中有 07f 遗传证据者，并按 SMR 效应方向判断与程序上调/下调是否一致。
输出：results/18_programme_genetics/{gene_set_tests.tsv, empirical_null.tsv, supported_programme_genes.tsv, gene_sets.tsv}
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

STAGE = "18_programme_genetics"
MAGMA_WORK = ROOT / "data/interim/magma"


def kidney_expression(cfg: dict, min_cells_total: int) -> pd.Series:
    """与 stage 13 相同：KPMP 单核各亚群合计 CP10k，跨亚群取平均。"""
    pb = pd.read_parquet(ROOT / cfg["paths"]["processed"] / "kpmp" / "snRNA_pseudobulk.parquet")
    cnt = pd.read_csv(ROOT / "results" / "03_kpmp_pseudobulk" / "snRNA_cell_counts.tsv", sep="\t", index_col=0)
    keep = cnt.columns[cnt.sum() >= min_cells_total]
    agg = pb.groupby(pb.index.str.split("|").str[1]).sum()
    agg = agg.loc[agg.index.isin(keep)]
    return (agg.div(agg.sum(1), axis=0) * 1e4).mean(0)


def build_sets(pg: dict) -> dict[str, set[str]]:
    r15 = ROOT / "results" / "15_key_genes"
    tal = pd.read_csv(r15 / f"{pg['tal_state']}_key_genes.tsv", sep="\t", index_col=0)
    pt = pd.read_csv(r15 / f"{pg['pt_state']}_key_genes.tsv", sep="\t", index_col=0)
    spec = pd.read_csv(ROOT / "results" / "13_gwas_cell_enrichment" / "specific_genes.tsv", sep="\t")
    up = set(tal.index[tal.replicated_both & (tal.log2fc > 0)])
    down = set(tal.index[tal.replicated_both & (tal.log2fc < 0)])
    healthy_spec = set(spec.gene[spec.subclass.isin(pg["healthy_tal_subclasses"])]) - up
    pt_up = set(pt.index[pt[pg["pt_replication"]].astype(bool) & (pt.log2fc > 0)])
    return {"tal_injured_up": up, "tal_injured_down": down, "tal_healthy": down | healthy_spec,
            "tal_healthy_specific": healthy_spec, "pt_injured_up": pt_up}


def symbol_to_entrez() -> dict[str, int]:
    """HGNC 现行符号 → Entrez；缺失时退回 NCBI37.3 基因坐标表中的符号。"""
    loc = pd.read_csv(ROOT / "data/tools/magma/NCBI37.3.gene.loc", sep="\t", header=None,
                      names=["GENE", "CHR", "START", "STOP", "STRAND", "SYMBOL"])
    m = dict(zip(loc.SYMBOL, loc.GENE))
    hg = pd.read_csv(ROOT / "data/raw/annot/hgnc_complete_set.txt", sep="\t", usecols=["symbol", "entrez_id"],
                     low_memory=False).dropna()
    m.update({s: int(e) for s, e in zip(hg.symbol, hg.entrez_id)})
    return m


def run_magma(trait: str, set_file: Path, covar_file: Path, work: Path) -> tuple[Path, Path]:
    magma = str(ROOT / "data/tools/magma/magma")
    raw = str(MAGMA_WORK / f"{trait}.genes.raw")
    outs = []
    for tag, extra in (("default", []),
                       ("expr_conditioned", ["--gene-covar", str(covar_file), "--model", "condition-hide=log_expr"])):
        prefix = work / f"{trait}.{tag}"
        with open(f"{prefix}.stdout", "w") as fh:
            subprocess.run([magma, "--gene-results", raw, "--set-annot", str(set_file)] + extra + ["--out", str(prefix)],
                           check=True, stdout=fh, stderr=subprocess.STDOUT)
        outs.append(Path(f"{prefix}.gsa.out"))
    return outs[0], outs[1]


def read_gsa(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", comment="#")
    return df[df.TYPE == "SET"][["VARIABLE", "NGENES", "BETA", "BETA_STD", "SE", "P"]]


def matched_null(z: pd.DataFrame, cov: pd.DataFrame, genes: set[str], pg: dict,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """返回 (观测平均 Z 向量[性状], 零分布矩阵[n_null × 性状])；匹配按 表达十分位 × 长度十分位 分层抽样。"""
    members = cov.index[cov.index.isin(genes)]
    pool = cov.index.difference(members)
    pool_bins = cov.loc[pool].groupby("bin").groups
    need = cov.loc[members, "bin"].value_counts()
    zv = z.values
    pos = {g: i for i, g in enumerate(z.index)}
    null = np.empty((pg["n_null"], z.shape[1]))
    for k in range(pg["n_null"]):
        idx = []
        for b, n in need.items():
            cand = pool_bins.get(b, pd.Index([]))
            idx += [pos[g] for g in rng.choice(cand, size=min(n, len(cand)), replace=False)]
        null[k] = zv[idx].mean(0)
    return zv[[pos[g] for g in members]].mean(0), null


def supported_genes(sets: dict, pg: dict, ent: dict) -> pd.DataFrame:
    ev = pd.read_csv(ROOT / "results" / "07f_genetic_convergence" / "genetic_evidence_by_gene.tsv", sep="\t")
    prog = {g: "up" for g in sets["tal_injured_up"]} | {g: "down" for g in sets["tal_injured_down"]}
    ev = ev[ev.symbol.isin(prog)].copy()
    ev["programme"] = ev.symbol.map(prog)
    smr = []
    for f in sorted((ROOT / "results" / "07e_smr").glob("*_smr.tsv")):
        smr.append(pd.read_csv(f, sep="\t"))
    smr = pd.concat(smr)
    smr = smr[smr.symbol.isin(ev.symbol)]
    # 每个 性状 × 基因：优先取 SMR 显著且 HEIDI 通过的组织中 P 最小者，否则取 P 最小者
    smr = smr.assign(rank_key=~(smr.smr_sig & smr.heidi_pass)).sort_values(["rank_key", "p_smr"])
    best = smr.drop_duplicates(["trait", "symbol"]).set_index(["trait", "symbol"])
    ev = ev.join(best[["eqtl", "b_smr", "p_smr", "smr_sig", "heidi_pass"]].rename(
        columns={"eqtl": "smr_tissue", "heidi_pass": "smr_tissue_heidi_pass"}), on=["trait", "symbol"])
    sign = ev.trait.map(pg["adverse_sign"])
    adverse = np.sign(ev.b_smr) * sign > 0
    ev["higher_expression_effect"] = np.where(ev.b_smr.isna(), "", np.where(adverse, "adverse", "protective"))
    conc = ((ev.programme == "up") & adverse) | ((ev.programme == "down") & ~adverse)
    ev["direction_concordant_with_injury"] = np.where(ev.b_smr.isna() | ~ev.smr_sig.eq(True), "",
                                                      np.where(conc, "yes", "no"))
    ev["multi_line"] = ev.n_lines >= pg["min_lines"]
    return ev.sort_values(["programme", "n_lines", "magma_p"], ascending=[False, False, True])


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    pg = cfg["programme_genetics"]
    rng = np.random.default_rng(cfg["seed"])
    out = ROOT / "results" / STAGE
    work = ROOT / "data/interim/programme_genetics"
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    sets = build_sets(pg)
    ent = symbol_to_entrez()
    raw_genes = set(pd.read_csv(MAGMA_WORK / f"{pg['traits'][0]}.genes.raw", sep=r"\s+", comment="#",
                                header=None, usecols=[0])[0])
    gs_rows, set_lines = [], []
    for name, genes in sets.items():
        ids = sorted({ent[g] for g in genes if g in ent and ent[g] in raw_genes})
        set_lines.append(" ".join([name] + [str(i) for i in ids]))
        gs_rows += [{"gene_set": name, "symbol": g, "entrez": ent.get(g, np.nan),
                     "in_magma": g in ent and ent[g] in raw_genes} for g in sorted(genes)]
    set_file = work / "gene_sets.txt"
    set_file.write_text("\n".join(set_lines) + "\n")
    gs_tab = pd.DataFrame(gs_rows)
    gs_tab.to_csv(out / "gene_sets.tsv", sep="\t", index=False)
    print(gs_tab.groupby("gene_set").in_magma.agg(["size", "sum"]), flush=True)

    expr = kidney_expression(cfg, pg["min_cells_total"])
    ent_expr = pd.Series({ent[g]: v for g, v in expr.items() if g in ent})
    ent_expr = ent_expr[~ent_expr.index.duplicated()]
    covar_file = work / "kpmp_expression.covar"
    pd.DataFrame({"GENE": ent_expr.index, "log_expr": np.log(ent_expr.values + 1e-3)}) \
        .to_csv(covar_file, sep="\t", index=False)

    tests, zs = [], {}
    for trait in pg["traits"]:
        d, c = run_magma(trait, set_file, covar_file, work)
        for tag, f in (("default", d), ("expr_conditioned", c)):
            tests.append(read_gsa(f).assign(trait=trait, model=tag))
        mg = pd.read_csv(ROOT / "results" / "07b_magma" / f"{trait}_genes.tsv", sep="\t")
        mg = mg.dropna(subset=["SYMBOL"]).drop_duplicates("SYMBOL").set_index("SYMBOL")
        zs[trait] = mg
    tests = pd.concat(tests).rename(columns={"VARIABLE": "gene_set", "NGENES": "n_genes", "BETA": "beta",
                                             "BETA_STD": "beta_std", "SE": "se", "P": "p_one_sided"})
    tests = tests[["trait", "model", "gene_set", "n_genes", "beta", "beta_std", "se", "p_one_sided"]]
    tests.to_csv(out / "gene_set_tests.tsv", sep="\t", index=False)
    print(tests.round(4).to_string(index=False), flush=True)

    # 经验零分布：统一基因宇宙 = 全部性状均有 MAGMA Z 且在 KPMP 单核中被测到的基因（符号层面）
    common = expr.index.intersection(set.intersection(*[set(m.index) for m in zs.values()]))
    z = pd.DataFrame({t: zs[t].loc[common, "ZSTAT"] for t in pg["traits"]})
    ref = zs[pg["traits"][0]].loc[common]
    cov = pd.DataFrame({"expr_bin": pd.qcut(expr[common].rank(method="first"), pg["n_bins_expr"], labels=False),
                        "len_bin": pd.qcut((ref.STOP - ref.START).rank(method="first"), pg["n_bins_length"],
                                           labels=False)}, index=common)
    cov["bin"] = cov.expr_bin * 100 + cov.len_bin
    null_rows = []
    for name, genes in sets.items():
        obs, null = matched_null(z, cov, genes, pg, rng)
        for j, t in enumerate(pg["traits"]):
            nl = null[:, j]
            null_rows.append({"trait": t, "gene_set": name, "n_genes": int(z.index.isin(genes).sum()),
                              "mean_z": float(obs[j]), "null_mean": float(nl.mean()), "null_sd": float(nl.std(ddof=1)),
                              "z_vs_null": float((obs[j] - nl.mean()) / nl.std(ddof=1)),
                              "p_empirical_greater": float((1 + (nl >= obs[j]).sum()) / (1 + len(nl))),
                              "n_null": len(nl), "universe": len(common)})
    nul = pd.DataFrame(null_rows)
    nul.to_csv(out / "empirical_null.tsv", sep="\t", index=False)
    print(nul.round(4).to_string(index=False), flush=True)

    sup = supported_genes(sets, pg, ent)
    sup.to_csv(out / "supported_programme_genes.tsv", sep="\t", index=False)
    print(sup[["trait", "symbol", "programme", "n_lines", "b_smr", "higher_expression_effect",
               "direction_concordant_with_injury"]].to_string(index=False), flush=True)

    inputs = [ROOT / "results/15_key_genes" / f"{pg['tal_state']}_key_genes.tsv",
              ROOT / "results/15_key_genes" / f"{pg['pt_state']}_key_genes.tsv",
              ROOT / "results/13_gwas_cell_enrichment/specific_genes.tsv",
              ROOT / "results/07f_genetic_convergence/genetic_evidence_by_gene.tsv",
              ROOT / "data/tools/magma/NCBI37.3.gene.loc", ROOT / "data/raw/annot/hgnc_complete_set.txt"]
    inputs += [MAGMA_WORK / f"{t}.genes.raw" for t in pg["traits"]]
    inputs += [ROOT / "results/07b_magma" / f"{t}_genes.tsv" for t in pg["traits"]]
    write_provenance(STAGE, inputs, [out / f for f in ("gene_sets.tsv", "gene_set_tests.tsv", "empirical_null.tsv",
                                                       "supported_programme_genes.tsv")],
                     seed=cfg["seed"], extra={"magma_version": "v1.10", "magma_set_file": str(set_file.relative_to(ROOT)),
                                              "null_matching": "KPMP snRNA mean CP10k decile x gene length decile"})


if __name__ == "__main__":
    main()
