"""Stage 04：KPMP 供体水平 DKD vs 高血压肾病（HKD）——细胞状态丰度与细胞状态内表达。

1. 丰度：损伤 PT（aPT+frPT）占 PT、损伤 TAL 占 TAL、足细胞与肾小球毛细血管内皮占肾小球细胞，DKD vs HKD / REF。
2. 状态内表达：每个细胞状态的供体 pseudobulk 上 DKD vs HKD 的逐基因 Hedges g（发现 = 单核）。
3. 复现：单细胞中未出现在单核的供体上，同一比较与发现效应的一致性（Spearman、top 基因同向比例），
   零分布 = 复现供体 DKD/HKD 标签置换。
输出：results/04_kpmp_dkd_vs_hkd/{abundance.tsv, effects_<mod>_<state>.tsv, concordance.tsv}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.pbstats import concordance, group_effect, hedges_g, state_matrix  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

STAGE = "04_kpmp_dkd_vs_hkd"


def donor_labels(donors: pd.DataFrame, groups: dict) -> pd.Series:
    lab = pd.Series(np.nan, index=donors.index, dtype=object)
    for name, rule in groups.items():
        (col, val), = rule.items()
        vals = val if isinstance(val, list) else [val]
        lab[donors[col].astype(str).isin(vals)] = name
    return lab.dropna()


def load(mod: str, cfg: dict):
    pb = pd.read_parquet(ROOT / cfg["paths"]["processed"] / "kpmp" / f"{mod}_pseudobulk.parquet")
    res = ROOT / "results" / "03_kpmp_pseudobulk"
    counts = pd.read_csv(res / f"{mod}_cell_counts.tsv", sep="\t", index_col=0)
    donors = pd.read_csv(res / f"{mod}_donors.tsv", sep="\t", index_col=0)
    counts.index, donors.index = counts.index.astype(str), donors.index.astype(str)
    return pb, counts, donors


def abundance(counts: pd.DataFrame, lab: pd.Series, k: dict, mod: str) -> pd.DataFrame:
    st = k["states"]
    extra = ["dPT", "cycPT", "dTAL", "dM-TAL", "dC-TAL", "cycTAL"]          # 两个模态的退行/增殖亚群名不同
    need = {s for v in st.values() for s in v} | set(extra) | set(k["glom_cells"])
    c = counts.reindex(columns=sorted(set(counts.columns) | need), fill_value=0)
    pt = c[st["PT_normal"] + st["PT_injured"] + ["dPT", "cycPT"]].sum(1)
    tal = c[st["TAL_normal"] + st["TAL_injured"] + ["dTAL", "dM-TAL", "dC-TAL", "cycTAL"]].sum(1)
    glom = c[k["glom_cells"]].sum(1)
    fr = pd.DataFrame({
        "PT_injured_of_PT": c[st["PT_injured"]].sum(1) / pt, "frPT_of_PT": c["frPT"] / pt,
        "TAL_injured_of_TAL": c[st["TAL_injured"]].sum(1) / tal,
        "POD_of_glom": c[["POD", "dPOD"]].sum(1) / glom, "EC_GC_of_glom": c[["EC-GC", "aEC-GC", "dEC-GC"]].sum(1) / glom,
    })
    ok = pd.DataFrame({"PT_injured_of_PT": pt, "frPT_of_PT": pt, "TAL_injured_of_TAL": tal,
                       "POD_of_glom": glom, "EC_GC_of_glom": glom}) >= k["min_cells"]
    rows = []
    for col in fr.columns:
        v = fr[col][ok[col]]
        logit = np.log((v + 1e-3) / (1 - v + 1e-3))
        for a, b in [("DKD", "HKD"), ("DKD", "REF"), ("HKD", "REF")]:
            x, y = logit[lab.reindex(v.index) == a], logit[lab.reindex(v.index) == b]
            if len(x) < 3 or len(y) < 3:
                continue
            rows.append({"modality": mod, "measure": col, "contrast": f"{a}_vs_{b}", "n_a": len(x), "n_b": len(y),
                         "median_a": float(v[x.index].median()), "median_b": float(v[y.index].median()),
                         "hedges_g_logit": float(hedges_g(x.values[:, None], y.values[:, None])[0]),
                         "p_mannwhitney": float(mannwhitneyu(x, y).pvalue)})
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    k = cfg["kpmp"]
    set_global_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)

    pb_d, cnt_d, don_d = load(k["discovery"], cfg)
    pb_r, cnt_r, don_r = load(k["replication"], cfg)
    lab_d, lab_r = donor_labels(don_d, k["groups"]), donor_labels(don_r, k["groups"])
    lab_r = lab_r[~lab_r.index.isin(don_d.index)]                 # 复现只用单核中未出现的供体
    print("discovery", lab_d.value_counts().to_dict(), "| replication", lab_r.value_counts().to_dict(), flush=True)

    ab = pd.concat([abundance(cnt_d, lab_d, k, k["discovery"]), abundance(cnt_r, lab_r, k, k["replication"])])
    ab.to_csv(out / "abundance.tsv", sep="\t", index=False)

    conc, outs = [], [out / "abundance.tsv"]
    for state, subs in k["states"].items():
        md = state_matrix(pb_d, cnt_d, subs, k["min_cells"])
        mr = state_matrix(pb_r, cnt_r, subs, k["min_cells"])
        if md.empty or mr.empty:
            continue
        expressed = md.columns[md.mean() >= k["min_mean_logcpm"]]
        md = md[expressed]
        nd = lab_d.reindex(md.index).value_counts()
        if min(nd.get("DKD", 0), nd.get("HKD", 0)) < k["min_donors_per_group"]:
            continue
        g = group_effect(md, lab_d, "DKD", "HKD").rename("g_DKD_vs_HKD")
        g_ref = group_effect(md, lab_d, "HKD", "REF").rename("g_HKD_vs_REF")
        eff = pd.concat([g, g_ref], axis=1)
        eff.to_csv(out / f"effects_{k['discovery']}_{state}.tsv", sep="\t")
        outs.append(out / f"effects_{k['discovery']}_{state}.tsv")
        nr = lab_r.reindex(mr.index).value_counts()
        if min(nr.get("DKD", 0), nr.get("HKD", 0)) < 3:
            continue
        rec = concordance(g, mr, lab_r, "DKD", "HKD", k["top_genes"], k["n_perm"], rng)
        rec.update({"state": state, "n_dkd_disc": int(nd.get("DKD", 0)), "n_hkd_disc": int(nd.get("HKD", 0))})
        conc.append(rec)
        print(state, {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in rec.items()}, flush=True)
    pd.DataFrame(conc).to_csv(out / "concordance.tsv", sep="\t", index=False)
    outs.append(out / "concordance.tsv")
    write_provenance(STAGE, inputs=[], outputs=outs, seed=cfg["seed"])


if __name__ == "__main__":
    main()
