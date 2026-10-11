"""Stage 14：疾病中细胞状态内的表达变化，与肾功能遗传决定因素的方向是否一致。

对 SMR 显著且通过 HEIDI 的基因，b_SMR 给出"表达升高 → eGFR（或 UACR）变化"的方向。
在 KPMP 单核各细胞状态的供体 pseudobulk 中计算 CKD（DKD+HKD，以及分别 DKD、HKD）相对参考供体的 Hedges g。
方向一致 = 疾病中的表达变化朝向遗传上不利的方向（eGFR：g·b<0；UACR：g·b>0）。
零分布：打乱供体疾病标签重算 g（保留基因间相关）。
输出：results/14_genetic_direction/{gene_level.tsv, direction_tests.tsv}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.pbstats import hedges_g, state_matrix  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

STAGE = "14_genetic_direction"
STATES = {"PT_injured": (["aPT", "frPT"], "susztak_tubule"), "PT_normal": (["PT-S1", "PT-S2", "PT-S3"], "susztak_tubule"),
          "TAL_injured": (["aTAL", "frTAL"], "susztak_tubule"), "TAL_normal": (["C-TAL", "M-TAL", "C/M-TAL"], "susztak_tubule"),
          "EC_GC": (["EC-GC", "aEC-GC", "dEC-GC"], "susztak_glom"), "POD": (["POD", "dPOD"], "susztak_glom")}


def smr_effects(trait: str, tissue: str) -> pd.Series:
    t = pd.read_csv(ROOT / "results" / "07e_smr" / f"{trait}_{tissue}_smr.tsv", sep="\t")
    t = t[t.smr_sig & t.heidi_pass].dropna(subset=["symbol"])
    return t.sort_values("p_smr").drop_duplicates("symbol").set_index("symbol")["b_smr"]


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    gd = cfg["genetic_direction"]
    rng = np.random.default_rng(cfg["seed"])
    res3 = ROOT / "results" / "03_kpmp_pseudobulk"
    pb = pd.read_parquet(ROOT / cfg["paths"]["processed"] / "kpmp" / "snRNA_pseudobulk.parquet")
    cnt = pd.read_csv(res3 / "snRNA_cell_counts.tsv", sep="\t", index_col=0)
    don = pd.read_csv(res3 / "snRNA_donors.tsv", sep="\t", index_col=0)
    cnt.index, don.index = cnt.index.astype(str), don.index.astype(str)
    lab = don["AdjudicationCategory"].map({"Diabetic Kidney Disease": "DKD", "Hypertensive Kidney Disease": "HKD"})
    lab[don["ConditionCategory"].isin(["Healthy Reference", "Reference"])] = "REF"
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)

    gene_rows, tests = [], []
    for state, (subs, tissue) in STATES.items():
        m = state_matrix(pb, cnt, subs, cfg["kpmp"]["min_cells"])
        l = lab.reindex(m.index)
        for trait, sign in gd["traits"].items():
            b = smr_effects(trait, tissue)
            genes = b.index.intersection(m.columns[m.mean() >= gd["min_mean_logcpm"]])
            if len(genes) < gd["min_genes"]:
                continue
            M = m[genes].values
            for case in gd["cases"]:
                is_case = l.isin(case.split("+")).values
                is_ref = (l == "REF").values
                keep = is_case | is_ref
                Mk, ck = M[keep], is_case[keep]
                g = hedges_g(Mk[ck], Mk[~ck])
                adverse = np.sign(g) * np.sign(b[genes].values) * sign > 0
                obs = float(np.nanmean(adverse))
                null = []
                for _ in range(gd["n_perm"]):
                    p = rng.permutation(ck)
                    gp = hedges_g(Mk[p], Mk[~p])
                    null.append(np.nanmean(np.sign(gp) * np.sign(b[genes].values) * sign > 0))
                null = np.array(null)
                tests.append({"state": state, "trait": trait, "eqtl": tissue, "case": case, "n_case": int(ck.sum()),
                              "n_ref": int((~ck).sum()), "n_genes": len(genes), "fraction_adverse": obs,
                              "null_mean": float(null.mean()), "p_perm": float((1 + (null >= obs).sum()) / (len(null) + 1))})
                if case == gd["cases"][0]:
                    gene_rows += [{"state": state, "trait": trait, "gene": gg, "b_smr": float(b[gg]), "g_case_vs_ref": float(x),
                                   "adverse": bool(a)} for gg, x, a in zip(genes, g, adverse)]
    pd.DataFrame(gene_rows).to_csv(out / "gene_level.tsv", sep="\t", index=False)
    t = pd.DataFrame(tests)
    t.to_csv(out / "direction_tests.tsv", sep="\t", index=False)
    print(t.round(3).to_string(index=False))
    write_provenance(STAGE, inputs=[res3 / "snRNA_cell_counts.tsv"], outputs=[out / "gene_level.tsv", out / "direction_tests.tsv"],
                     seed=cfg["seed"])


if __name__ == "__main__":
    main()
