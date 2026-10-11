"""Stage 18b：肾功能遗传信号在健康细胞谱系之间的分布（stage 18 匹配零分布的谱系对照）。

每个谱系 = 其健康亚群在 stage 13 中的特异基因并集；与 stage 18 相同的基因宇宙、
表达十分位 × 基因长度十分位匹配抽样与 1000 次随机集。
输出：results/18b_lineage_genetics/lineage_null.tsv
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

STAGE = "18b_lineage_genetics"
_spec = importlib.util.spec_from_file_location("s18", Path(__file__).with_name("18_programme_genetics.py"))
s18 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s18)


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    pg, lg = cfg["programme_genetics"], cfg["lineage_genetics"]
    rng = np.random.default_rng(cfg["seed"])
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    spec = pd.read_csv(ROOT / "results/13_gwas_cell_enrichment/specific_genes.tsv", sep="\t")
    sets = {k: set(spec.gene[spec.subclass.isin(v)]) for k, v in lg["lineages"].items()}

    expr = s18.kidney_expression(cfg, pg["min_cells_total"])
    zs = {}
    for t in pg["traits"]:
        mg = pd.read_csv(ROOT / "results/07b_magma" / f"{t}_genes.tsv", sep="\t")
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
            nl = null[:, j]
            rows.append({"trait": t, "lineage": name, "n_genes": int(z.index.isin(genes).sum()),
                         "mean_z": float(obs[j]), "null_mean": float(nl.mean()), "null_sd": float(nl.std(ddof=1)),
                         "z_vs_null": float((obs[j] - nl.mean()) / nl.std(ddof=1)),
                         "p_empirical_greater": float((1 + (nl >= obs[j]).sum()) / (1 + len(nl))),
                         "n_null": len(nl), "universe": len(common)})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "lineage_null.tsv", sep="\t", index=False)
    print(tab.pivot(index="lineage", columns="trait", values="z_vs_null").round(2).to_string())
    print(tab.pivot(index="lineage", columns="trait", values="p_empirical_greater").round(3).to_string())
    write_provenance(STAGE, [ROOT / "results/13_gwas_cell_enrichment/specific_genes.tsv"]
                     + [ROOT / "results/07b_magma" / f"{t}_genes.tsv" for t in pg["traits"]],
                     [out / "lineage_null.tsv"], seed=cfg["seed"])


if __name__ == "__main__":
    main()
