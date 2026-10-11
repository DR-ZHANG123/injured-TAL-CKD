"""Stage 24：损伤 TAL 程序在四个复现集中的统一口径汇总。

口径：在 1212 个发现基因（KPMP 单核，FDR < 0.05，|log2FC| ≥ 1）上计算发现与复现 log2FC 的 Pearson r，
以及复现率（同向且 |log2FC| ≥ 0.5，分母为 1212）。复现集：KPMP 单细胞独立参与者、GSE271188（stage 15），
GSE211785（stage 19a，单核与单细胞按供体合并）、GSE195460（stage 19b，全部状态标签）。
输出：results/24_replication_summary/replication_over_discovery_genes.tsv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

STAGE = "24_replication_summary"


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    kg = cfg["key_genes"]
    res = ROOT / "results"
    disc = pd.read_csv(res / "15_key_genes" / "aTAL_frTAL_discovery.tsv", sep="\t", index_col=0)
    hits = disc[(disc.fdr < kg["fdr"]) & (disc.log2fc.abs() >= kg["min_lfc"])].index
    key = pd.read_csv(res / "15_key_genes" / "aTAL_frTAL_key_genes.tsv", sep="\t", index_col=0)
    sets = {
        "KPMP scRNA-seq (independent participants)": key["log2fc_kpmp_sc"],
        "GSE271188": key["log2fc_gse271188"],
        "GSE211785": pd.read_csv(res / "19a_gse211785_relabel" / "programme_lfc.tsv", sep="\t", index_col=0).log2fc,
        "GSE195460": pd.read_csv(res / "19b_gse195460" / "programme_lfc_all_state_labels.tsv", sep="\t", index_col=0).log2fc,
    }
    rows = []
    for name, rep in sets.items():
        g = hits.intersection(rep.dropna().index)
        same = (rep[g] * disc.loc[g, "log2fc"] > 0) & (rep[g].abs() >= kg["rep_lfc"])
        rows.append({"replication_set": name, "n_discovery_genes": len(hits), "n_measured": len(g),
                     "pearson_r_over_discovery_genes": float(disc.loc[g, "log2fc"].corr(rep[g])),
                     "n_replicated": int(same.sum()), "replication_rate": float(same.sum() / len(hits))})
    out = res / STAGE
    out.mkdir(parents=True, exist_ok=True)
    t = pd.DataFrame(rows)
    t.to_csv(out / "replication_over_discovery_genes.tsv", sep="\t", index=False)
    print(t.round(3).to_string(index=False))
    write_provenance(STAGE, inputs=[res / "15_key_genes" / "aTAL_frTAL_discovery.tsv"],
                     outputs=[out / "replication_over_discovery_genes.tsv"], seed=cfg["seed"])


if __name__ == "__main__":
    main()
