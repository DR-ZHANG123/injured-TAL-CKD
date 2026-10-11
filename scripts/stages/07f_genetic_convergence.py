"""Stage 07f：汇总三条人类遗传学证据（MAGMA 基因水平、GTEx 肾皮质 coloc、肾组织 SMR+HEIDI），
按性状给出每个基因的证据条数，供后续与单细胞 / 空间结果对接。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

STAGE = "07f_genetic_convergence"


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    g = cfg["genetics"]
    res = ROOT / "results"
    out_dir = res / STAGE
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs, tables, summary = [], [], []
    for t in g["coloc"]["traits"]:
        mp = res / "07b_magma" / f"{t}_genes.tsv"
        cp = res / "07d_coloc" / f"{t}_coloc_gtex_kidney_cortex.tsv"
        sps = [res / "07e_smr" / f"{t}_{d}_smr.tsv" for d in g["smr"]["datasets"]]
        inputs += [mp, cp, *sps]
        mg = pd.read_csv(mp, sep="\t")
        mg = mg.dropna(subset=["SYMBOL"]).drop_duplicates("SYMBOL").set_index("SYMBOL")
        co = pd.read_csv(cp, sep="\t")
        co = co[co["symbol"].notna()].groupby("symbol")["PP.H4"].max()
        sm = pd.concat([pd.read_csv(p, sep="\t") for p in sps])
        sm_pass = sm[sm["heidi_pass"]].groupby("symbol")["eqtl"].apply(lambda x: ";".join(sorted(set(x))))
        sm_minp = sm.groupby("symbol")["p_smr"].min()
        genes = sorted(set(mg.index[mg["SIG_BONF"]]) | set(co.index[co >= g["coloc"]["pp4_flag"]]) | set(sm_pass.index))
        df = pd.DataFrame({"trait": t, "symbol": genes})
        df["magma_p"] = df["symbol"].map(mg["P"])
        df["magma_bonf"] = df["symbol"].map(mg["SIG_BONF"]).eq(True)
        df["coloc_pp4_max"] = df["symbol"].map(co)
        df["coloc_pp4_ge_flag"] = df["coloc_pp4_max"] >= g["coloc"]["pp4_flag"]
        df["smr_min_p"] = df["symbol"].map(sm_minp)
        df["smr_heidi_pass_tissues"] = df["symbol"].map(sm_pass).fillna("")
        df["smr_heidi_pass"] = df["smr_heidi_pass_tissues"] != ""
        df["n_lines"] = df[["magma_bonf", "coloc_pp4_ge_flag", "smr_heidi_pass"]].sum(axis=1)
        df = df.sort_values(["n_lines", "magma_p"], ascending=[False, True])
        tables.append(df)
        summary.append({"trait": t, "n_genes_any": len(df), "n_genes_2plus": int((df["n_lines"] >= 2).sum()),
                        "n_genes_3": int((df["n_lines"] == 3).sum()),
                        "genes_3": ";".join(df.loc[df["n_lines"] == 3, "symbol"]),
                        "genes_2plus": ";".join(df.loc[df["n_lines"] >= 2, "symbol"])})
        print({k: v for k, v in summary[-1].items() if k != "genes_2plus"}, flush=True)
    op = out_dir / "genetic_evidence_by_gene.tsv"
    pd.concat(tables).to_csv(op, sep="\t", index=False)
    sp = out_dir / "convergence_summary.tsv"
    pd.DataFrame(summary).to_csv(sp, sep="\t", index=False)
    write_provenance(STAGE, inputs, [op, sp], seed=cfg["seed"])


if __name__ == "__main__":
    main()
