"""Stage 04b：KPMP 单核供体水平——eGFR 校正的细胞状态比例，以及肾小球细胞状态内的标志表达。

1. 损伤 PT / frPT / 损伤 TAL 比例（logit）~ 诊断 + eGFR（分档中点），DKD vs HKD。
2. 肾小球毛细血管内皮（EC-GC）与足细胞（POD）pseudobulk 内的标志得分：DKD / HKD / REF 两两比较，及 eGFR 校正的 DKD vs HKD。
输出：results/04b_kpmp_adjusted/{abundance_adjusted.tsv, glomerular_state_markers.tsv}
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.pbstats import hedges_g, state_matrix  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

STAGE = "04b_kpmp_adjusted"
EGFR_COL = "Baseline eGFR (ml/min/1.73m2) (Binned)"
FRACTIONS = {  # 分子亚群, 分母亚群
    "PT_injured_of_PT": (["aPT", "frPT"], ["PT-S1", "PT-S2", "PT-S3", "aPT", "frPT", "dPT", "cycPT"]),
    "frPT_of_PT": (["frPT"], ["PT-S1", "PT-S2", "PT-S3", "aPT", "frPT", "dPT", "cycPT"]),
    "TAL_injured_of_TAL": (["aTAL", "frTAL"], ["C-TAL", "M-TAL", "C/M-TAL", "aTAL", "frTAL", "dM-TAL", "dC-TAL", "cycTAL"]),
}
GLOM_STATES = {"EC_GC": (["EC-GC"], "glomerular_capillary_EC"),
               "EC_GC_all": (["EC-GC", "aEC-GC", "dEC-GC"], "glomerular_capillary_EC"),
               "POD": (["POD"], "podocyte"), "POD_all": (["POD", "dPOD"], "podocyte")}


def egfr_mid(s: str) -> float:
    m = re.match(r"(\d+)-(\d+)", str(s))
    return (int(m.group(1)) + int(m.group(2))) / 2 if m else np.nan


def labels(don: pd.DataFrame) -> pd.Series:
    lab = don["AdjudicationCategory"].map({"Diabetic Kidney Disease": "DKD", "Hypertensive Kidney Disease": "HKD"})
    lab[don["ConditionCategory"].isin(["Healthy Reference", "Reference"])] = "REF"
    return lab


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    k = cfg["kpmp"]
    res3 = ROOT / "results" / "03_kpmp_pseudobulk"
    cnt = pd.read_csv(res3 / "snRNA_cell_counts.tsv", sep="\t", index_col=0)
    don = pd.read_csv(res3 / "snRNA_donors.tsv", sep="\t", index_col=0)
    cnt.index, don.index = cnt.index.astype(str), don.index.astype(str)
    lab, egfr = labels(don), don[EGFR_COL].map(egfr_mid)
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, (num, den) in FRACTIONS.items():
        c = cnt.reindex(columns=sorted(set(cnt.columns) | set(den)), fill_value=0)
        n_den = c[den].sum(1)
        frac = c[num].sum(1) / n_den
        d = pd.DataFrame({"frac": frac, "lab": lab, "egfr": egfr})[n_den >= k["min_cells"]]
        d = d[d.lab.isin(["DKD", "HKD"])].dropna()
        d["y"] = np.log((d.frac + 1e-3) / (1 - d.frac + 1e-3))
        for model in ["y ~ lab", "y ~ lab + egfr"]:
            f = smf.ols(model, d).fit()
            rows.append({"measure": name, "model": model, "n_dkd": int((d.lab == "DKD").sum()),
                         "n_hkd": int((d.lab == "HKD").sum()),
                         "median_dkd": float(d.frac[d.lab == "DKD"].median()),
                         "median_hkd": float(d.frac[d.lab == "HKD"].median()),
                         "coef_DKD_minus_HKD": float(-f.params["lab[T.HKD]"]), "p": float(f.pvalues["lab[T.HKD]"]),
                         "median_egfr_dkd": float(d.egfr[d.lab == "DKD"].median()),
                         "median_egfr_hkd": float(d.egfr[d.lab == "HKD"].median())})
    pd.DataFrame(rows).to_csv(out / "abundance_adjusted.tsv", sep="\t", index=False)

    pb = pd.read_parquet(ROOT / cfg["paths"]["processed"] / "kpmp" / "snRNA_pseudobulk.parquet")
    mk = pd.read_csv(ROOT / "results" / "05_markers" / "markers.tsv", sep="\t")
    rows = []
    for state, (subs, mset) in GLOM_STATES.items():
        m = state_matrix(pb, cnt, subs, k["min_cells"])
        genes = [g for g in mk.loc[mk.marker_set == mset, "gene"] if g in m.columns]
        score = ((m[genes] - m[genes].mean()) / m[genes].std()).mean(1)
        l = lab.reindex(score.index)
        for a, b in [("DKD", "HKD"), ("DKD", "REF"), ("HKD", "REF")]:
            x, y = score[l == a], score[l == b]
            rows.append({"state": state, "marker_set": mset, "contrast": f"{a}_vs_{b}", "model": "unadjusted",
                         "n_a": len(x), "n_b": len(y), "effect": float(hedges_g(x.values[:, None], y.values[:, None])[0]),
                         "p": float(mannwhitneyu(x, y).pvalue)})
        d = pd.DataFrame({"y": score, "lab": l, "egfr": egfr.reindex(score.index)}).dropna()
        d = d[d.lab.isin(["DKD", "HKD"])]
        f = smf.ols("y ~ lab + egfr", d).fit()
        rows.append({"state": state, "marker_set": mset, "contrast": "DKD_vs_HKD", "model": "eGFR-adjusted",
                     "n_a": int((d.lab == "DKD").sum()), "n_b": int((d.lab == "HKD").sum()),
                     "effect": float(-f.params["lab[T.HKD]"]), "p": float(f.pvalues["lab[T.HKD]"])})
    pd.DataFrame(rows).to_csv(out / "glomerular_state_markers.tsv", sep="\t", index=False)
    print(pd.read_csv(out / "abundance_adjusted.tsv", sep="\t").round(3).to_string(index=False))
    write_provenance(STAGE, inputs=[res3 / "snRNA_cell_counts.tsv", res3 / "snRNA_donors.tsv"],
                     outputs=[out / "abundance_adjusted.tsv", out / "glomerular_state_markers.tsv"], seed=cfg["seed"])


if __name__ == "__main__":
    main()
