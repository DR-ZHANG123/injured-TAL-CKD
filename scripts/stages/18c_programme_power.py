"""Stage 18c：匹配零分布检验对损伤程序大小基因集的检出力（阳性对照）。

从健康 TAL 身份基因（stage 18 的 tal_healthy_specific）中无放回抽取与损伤 TAL 上调程序（536 个 MAGMA 基因中
落在基因宇宙内的个数）同样大小的子集，重复 n_draws 次；每个子集走与 stage 18 相同的表达 × 长度匹配零分布，
记录经验 P。检出力 = 经验 P < 0.05 的子集比例。同时报告损伤程序本身的观测值作对照。
输出：results/18c_programme_power/{power.tsv, subset_draws.tsv}
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

STAGE = "18c_programme_power"
_spec = importlib.util.spec_from_file_location("s18", Path(__file__).with_name("18_programme_genetics.py"))
s18 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s18)


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    pg, pw = cfg["programme_genetics"], cfg["programme_power"]
    rng = np.random.default_rng(cfg["seed"])
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    gs = pd.read_csv(ROOT / "results/18_programme_genetics/gene_sets.tsv", sep="\t")
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
    pool = sorted(set(gs.symbol[gs.gene_set == pw["source_set"]]) & set(common))
    size = len(set(gs.symbol[gs.gene_set == pw["size_of"]]) & set(common))
    sub_pg = dict(pg, n_null=pw["n_null"])
    rows = []
    for d in range(pw["n_draws"]):
        genes = set(rng.choice(pool, size=size, replace=False))
        obs, null = s18.matched_null(z, cov, genes, sub_pg, rng)
        for j, t in enumerate(pg["traits"]):
            nl = null[:, j]
            rows.append({"draw": d, "trait": t, "n_genes": size,
                         "z_vs_null": float((obs[j] - nl.mean()) / nl.std(ddof=1)),
                         "p_empirical_greater": float((1 + (nl >= obs[j]).sum()) / (1 + len(nl)))})
    draws = pd.DataFrame(rows)
    draws.to_csv(out / "subset_draws.tsv", sep="\t", index=False)
    pwr = draws.groupby("trait").agg(n_draws=("draw", "size"), n_genes=("n_genes", "first"),
                                     median_z=("z_vs_null", "median"),
                                     power_p05=("p_empirical_greater", lambda p: float((p < 0.05).mean())))
    emp = pd.read_csv(ROOT / "results/18_programme_genetics/empirical_null.tsv", sep="\t")
    emp = emp[emp.gene_set == pw["size_of"]].set_index("trait")[["z_vs_null", "p_empirical_greater"]]
    pwr = pwr.join(emp.add_prefix("injured_programme_")).reset_index()
    pwr.to_csv(out / "power.tsv", sep="\t", index=False)
    print(pwr.round(3).to_string(index=False))
    write_provenance(STAGE, inputs=[ROOT / "results/18_programme_genetics/gene_sets.tsv"]
                     + [ROOT / "results/07b_magma" / f"{t}_genes.tsv" for t in pg["traits"]],
                     outputs=[out / "power.tsv", out / "subset_draws.tsv"], seed=cfg["seed"])


if __name__ == "__main__":
    main()
