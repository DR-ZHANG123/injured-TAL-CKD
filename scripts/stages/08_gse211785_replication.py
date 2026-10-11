"""Stage 08：GSE211785（Susztak 队列，不含 KPMP）独立复现 DKD vs HKD 小管轴。

1. 比例：iPT /（PT_S1–S3 + iPT），供体 × 技术（SN/SC），logit ~ 诊断 + 技术。
2. 状态表达：iPT 与正常 PT 的供体平均表达（存放为归一化值），先在技术内中心化，DKD vs HKD 的 Hedges g，
   与 KPMP 单核发现效应（stage 04 effects_snRNA_PT_injured / PT_normal）比较一致性；零分布 = 供体标签置换。
条形码前缀与样本列矛盾的单核文库（HK2663/HK2770）整体排除。
输出：results/08_gse211785_replication/{abundance.tsv, concordance.tsv, donors.tsv}
"""
from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.pbstats import hedges_g  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

STAGE = "08_gse211785_replication"
PT_NORMAL, PT_INJ = ["PT_S1", "PT_S2", "PT_S3"], ["iPT"]
EXCLUDE_LIBS = {"HK2663.SN", "HK2770.SN"}


def donor_means(a, obs: pd.DataFrame, mask: np.ndarray, key: pd.Series) -> pd.DataFrame:
    idx = np.where(mask)[0]
    k = key.iloc[idx]
    groups = pd.Index(sorted(k.unique()))
    gi = groups.get_indexer(k)
    X = a.raw.X[idx] if a.raw is not None else a.X[idx]
    X = X if sp.issparse(X) else sp.csr_matrix(X)
    ind = sp.csr_matrix((np.ones(len(idx)), (gi, np.arange(len(idx)))), shape=(len(groups), len(idx)))
    n = np.asarray(ind.sum(1)).ravel()
    m = (ind @ X).toarray() / n[:, None]
    return pd.DataFrame(m, index=groups, columns=a.raw.var_names), pd.Series(n, index=groups)


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    k = cfg["kpmp"]
    rng = np.random.default_rng(cfg["seed"])
    a = ad.read_h5ad(ROOT / cfg["paths"]["raw"] / "gse211785" / "susztak_preSCVI.h5ad", backed="r")
    obs = a.obs.copy()
    obs["lib"] = obs.index.str.split("_").str[0]
    keep = obs.tech.isin(["SN_RNA", "SC_RNA"]).values & ~obs.lib.isin(EXCLUDE_LIBS).values
    key = obs["sample"].astype(str) + "|" + obs["tech"].astype(str)
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)

    donors = obs[keep].drop_duplicates(["sample", "tech"]).assign(unit=lambda d: d["sample"].astype(str) + "|" + d["tech"].astype(str))
    donors = donors.set_index("unit")[["sample", "tech", "group", "age", "sex"]]
    donors.to_csv(out / "donors.tsv", sep="\t")

    ct = obs[keep].groupby([key[keep], "Cluster_Idents"], observed=True).size().unstack(fill_value=0)
    d = donors.join(pd.DataFrame({"inj": ct.reindex(columns=PT_INJ, fill_value=0).sum(1),
                                  "pt": ct.reindex(columns=PT_NORMAL + PT_INJ, fill_value=0).sum(1)}))
    d = d[(d.pt >= k["min_cells"]) & d.group.isin(["DKD", "HKD", "Control"])]
    d["frac"] = d.inj / d.pt
    d["y"] = np.log((d.frac + 1e-3) / (1 - d.frac + 1e-3))
    rows = []
    for a_, b_ in [("DKD", "HKD"), ("DKD", "Control"), ("HKD", "Control")]:
        s = d[d.group.isin([a_, b_])].copy()
        s["is_a"] = (s.group == a_).astype(int)
        f = smf.ols("y ~ is_a + C(tech)", s).fit()
        rows.append({"measure": "iPT_of_PT", "contrast": f"{a_}_vs_{b_}", "n_a": int(s.is_a.sum()),
                     "n_b": int((1 - s.is_a).sum()), "median_a": float(s.frac[s.is_a == 1].median()),
                     "median_b": float(s.frac[s.is_a == 0].median()), "coef_logit": float(f.params["is_a"]),
                     "p": float(f.pvalues["is_a"])})
    pd.DataFrame(rows).to_csv(out / "abundance.tsv", sep="\t", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False), flush=True)

    conc = []
    for state, cl, disc in [("PT_injured", PT_INJ, "PT_injured"), ("PT_normal", PT_NORMAL, "PT_normal")]:
        mask = keep & obs.Cluster_Idents.isin(cl).values
        m, n = donor_means(a, obs, mask, key)
        m = m[n >= k["min_cells"]]
        m = m.loc[:, m.mean() > 0.1]
        tech = donors.reindex(m.index)["tech"]
        m = m.groupby(tech.values, observed=True).transform(lambda x: x - x.mean())       # 技术内中心化
        lab = donors.reindex(m.index)["group"]
        g_disc = pd.read_csv(ROOT / "results" / "04_kpmp_dkd_vs_hkd" / f"effects_snRNA_{disc}.tsv",
                             sep="\t", index_col=0)["g_DKD_vs_HKD"]
        genes = g_disc.dropna().index.intersection(m.columns)
        top = g_disc[genes].abs().sort_values(ascending=False).index[:k["top_genes"]]
        M, L = m[genes], lab.values

        def stats(labels):
            g = pd.Series(hedges_g(M.values[labels == "DKD"], M.values[labels == "HKD"]), index=genes)
            ok = g.notna()
            return (g_disc[genes][ok].rank().corr(g[ok].rank()),
                    float((np.sign(g[top]) == np.sign(g_disc[top])).mean()))
        rho, same = stats(L)
        null = np.array([stats(rng.permutation(L)) for _ in range(k["n_perm"])])
        rec = {"state": state, "n_dkd": int((L == "DKD").sum()), "n_hkd": int((L == "HKD").sum()), "n_genes": len(genes),
               "spearman": rho, "p_spearman": float((1 + (null[:, 0] >= rho).sum()) / (k["n_perm"] + 1)),
               "top_same_direction": same, "top_null_mean": float(null[:, 1].mean()),
               "p_top": float((1 + (null[:, 1] >= same).sum()) / (k["n_perm"] + 1))}
        conc.append(rec)
        print(state, {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in rec.items()}, flush=True)
    pd.DataFrame(conc).to_csv(out / "concordance.tsv", sep="\t", index=False)
    write_provenance(STAGE, inputs=[], outputs=[out / "abundance.tsv", out / "concordance.tsv", out / "donors.tsv"],
                     seed=cfg["seed"], extra={"excluded_libraries": sorted(EXCLUDE_LIBS)})


if __name__ == "__main__":
    main()
