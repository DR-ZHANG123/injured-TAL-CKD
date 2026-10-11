"""Stage 13：肾功能 GWAS 信号在 KPMP 细胞状态特异基因上的富集（MAGMA 基因得分的基因属性回归）。

细胞状态特异性：KPMP 单核 v2 全部供体按 SubclassLevel2 汇总 → 每个基因在各亚群 CP10k 中所占比例；
每个亚群取特异性前 10% 的基因（且该亚群内表达达到阈值）。
回归：MAGMA ZSTAT ~ 是否特异基因 + log(基因长度) + log(SNP 数) + log(参数数) + log(肾内平均表达)，
只在肾内有表达的基因中进行，单侧检验系数 > 0；全部 亚群 × 性状 做 FDR。
区室对比：肾小球-血管亚群集合 vs 小管修复亚群集合的特异基因，在 eGFR 与 UACR 间的差异（性状 × 区室交互，基因块 bootstrap）。
输出：results/13_gwas_cell_enrichment/{specific_genes.tsv, enrichment.tsv}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

STAGE = "13_gwas_cell_enrichment"


def specificity(cfg: dict, min_cells_total: int) -> pd.DataFrame:
    pb = pd.read_parquet(ROOT / cfg["paths"]["processed"] / "kpmp" / "snRNA_pseudobulk.parquet")
    cnt = pd.read_csv(ROOT / "results" / "03_kpmp_pseudobulk" / "snRNA_cell_counts.tsv", sep="\t", index_col=0)
    keep_sub = cnt.columns[cnt.sum() >= min_cells_total]
    sub = pb.index.str.split("|").str[1]
    agg = pb.groupby(sub).sum()
    agg = agg.loc[agg.index.isin(keep_sub)]
    cp10k = agg.div(agg.sum(1), axis=0) * 1e4
    return cp10k, cp10k.div(cp10k.sum(0).replace(0, np.nan), axis=1)


def compartment_contrast(cfg: dict, ge: dict, cp10k: pd.DataFrame, sets: dict) -> list[dict]:
    """每个性状：Z ~ 协变量 + 肾小球-血管特异 + 小管修复特异；比较两系数之差在 eGFR 与 UACR 间的差（基因 bootstrap）。"""
    rng = np.random.default_rng(cfg["seed"])
    glom = set().union(*[sets[s] for s in ge["glomerular_vascular"] if s in sets])
    tub = set().union(*[sets[s] for s in ge["tubular_repair"] if s in sets])
    mean_expr = cp10k.mean(0)
    data = {}
    for trait in ge["traits"]:
        mg = pd.read_csv(ROOT / "results" / "07b_magma" / f"{trait}_genes.tsv", sep="\t")
        mg = mg.dropna(subset=["SYMBOL"]).drop_duplicates("SYMBOL").set_index("SYMBOL")
        mg = mg[mg.index.isin(mean_expr.index[mean_expr >= ge["min_cp10k"]])]
        X = pd.DataFrame({"log_len": np.log(mg.STOP - mg.START + 1), "log_nsnp": np.log(mg.NSNPS),
                          "log_npar": np.log(mg.NPARAM), "log_expr": np.log(mean_expr[mg.index] + 1e-3),
                          "glom": mg.index.isin(glom).astype(float), "tub": mg.index.isin(tub).astype(float)})
        data[trait] = (mg.ZSTAT, sm.add_constant(X))

    def diff(y, X, idx=None):
        if idx is not None:
            y, X = y.iloc[idx], X.iloc[idx]
        p = np.linalg.lstsq(X.values, y.values, rcond=None)[0]
        return p[list(X.columns).index("glom")] - p[list(X.columns).index("tub")]

    rows = []
    for trait, (y, X) in data.items():
        f = sm.OLS(y, X).fit()
        rows.append({"trait": trait, "beta_glomerular_vascular": float(f.params["glom"]), "p_glom": float(f.pvalues["glom"]),
                     "beta_tubular_repair": float(f.params["tub"]), "p_tub": float(f.pvalues["tub"]),
                     "glom_minus_tub": float(diff(y, X))})
    for a, b in ge["trait_pairs"]:
        ya, Xa = data[a]
        yb, Xb = data[b]
        common = ya.index.intersection(yb.index)
        ya, Xa, yb, Xb = ya[common], Xa.loc[common], yb[common], Xb.loc[common]
        obs = diff(ya, Xa) - diff(yb, Xb)
        boots = [diff(ya, Xa, i) - diff(yb, Xb, i) for i in
                 (rng.integers(0, len(common), len(common)) for _ in range(ge["n_boot"]))]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"trait": f"{a} minus {b}", "glom_minus_tub": float(obs), "ci_low": float(lo), "ci_high": float(hi),
                     "p_two_sided": float(2 * min(np.mean(np.array(boots) <= 0), np.mean(np.array(boots) >= 0)))})
    return rows


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    ge = cfg["gwas_enrichment"]
    cp10k, spec = specificity(cfg, ge["min_cells_total"])
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)

    sets = {}
    for s in spec.index:
        expressed = cp10k.columns[cp10k.loc[s] >= ge["min_cp10k"]]
        sc = spec.loc[s, expressed].dropna()
        sets[s] = set(sc.nlargest(int(np.ceil(len(spec.columns) * ge["top_fraction"]))).index)
    pd.DataFrame([(s, g) for s, gs in sets.items() for g in sorted(gs)], columns=["subclass", "gene"]) \
        .to_csv(out / "specific_genes.tsv", sep="\t", index=False)

    rows = []
    for trait in ge["traits"]:
        mg = pd.read_csv(ROOT / "results" / "07b_magma" / f"{trait}_genes.tsv", sep="\t")
        mg = mg.dropna(subset=["SYMBOL"]).drop_duplicates("SYMBOL").set_index("SYMBOL")
        mean_expr = cp10k.mean(0)
        mg = mg[mg.index.isin(mean_expr.index[mean_expr >= ge["min_cp10k"]])]
        cov = pd.DataFrame({"log_len": np.log(mg.STOP - mg.START + 1), "log_nsnp": np.log(mg.NSNPS),
                            "log_npar": np.log(mg.NPARAM), "log_expr": np.log(mean_expr[mg.index] + 1e-3)})
        for s, gs in sets.items():
            X = sm.add_constant(cov.assign(specific=mg.index.isin(gs).astype(float)))
            f = sm.OLS(mg.ZSTAT, X).fit()
            t = f.tvalues["specific"]
            rows.append({"trait": trait, "subclass": s, "n_specific": int(X.specific.sum()),
                         "beta": float(f.params["specific"]), "t": float(t),
                         "p_one_sided": float(f.pvalues["specific"] / 2 if t > 0 else 1 - f.pvalues["specific"] / 2)})
    res = pd.DataFrame(rows)
    res["rank"] = res.groupby("trait").beta.rank(ascending=False).astype(int)
    res.to_csv(out / "enrichment.tsv", sep="\t", index=False)
    contrast_rows = compartment_contrast(cfg, ge, cp10k, sets)
    pd.DataFrame(contrast_rows).to_csv(out / "compartment_contrast.tsv", sep="\t", index=False)
    print(pd.DataFrame(contrast_rows).round(4).to_string(index=False), flush=True)
    res["fdr"] = multipletests(res.p_one_sided, method="fdr_bh")[1]
    res.to_csv(out / "enrichment.tsv", sep="\t", index=False)
    for trait in ge["traits"]:
        print(trait, res[res.trait == trait].nsmallest(8, "p_one_sided")[["subclass", "beta", "p_one_sided", "fdr"]]
              .round(4).to_string(index=False), sep="\n", flush=True)
    write_provenance(STAGE, inputs=[ROOT / "results" / "07b_magma" / f"{t}_genes.tsv" for t in ge["traits"]],
                     outputs=[out / "specific_genes.tsv", out / "enrichment.tsv"], seed=cfg["seed"])


if __name__ == "__main__":
    main()
