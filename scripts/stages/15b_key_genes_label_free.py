"""Stage 15b：关键基因复现的敏感性分析——剔除用于注释迁移的基因。

GSE271188 的细胞标签由 stage 09 用 KPMP 各亚群特异基因（每亚群前 25 个）迁移而来；
这些基因在复现中可能存在循环。这里把它们从复现评估中剔除后，重新计算复现率与倍数变化相关。
输出：results/15b_key_genes_label_free/sensitivity.tsv
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

STAGE = "15b_key_genes_label_free"


def label_transfer_genes(cfg: dict) -> set[str]:
    spec = importlib.util.spec_from_file_location("s09", ROOT / "scripts" / "stages" / "09_gse271188_replication.py")
    s09 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s09)
    _, genes = s09.reference(cfg, 25)
    return set(genes)


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    kg = cfg["key_genes"]
    excl = label_transfer_genes(cfg)
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for state in kg["states"]:
        f = ROOT / "results" / "15_key_genes" / f"{state}_key_genes.tsv"
        if not f.exists():
            continue
        t = pd.read_csv(f, sep="\t", index_col=0)
        for subset, mask in [("all", np.ones(len(t), bool)), ("label_transfer_genes_removed", ~t.index.isin(excl))]:
            s = t[mask]
            rows.append({"state": state, "genes": subset, "n_hits": len(s), "n_excluded": int((~mask).sum()),
                         "rep_rate_gse271188": float(s.replicated_gse271188.mean()),
                         "rep_rate_kpmp_sc": float(s.replicated_kpmp_sc.mean()),
                         "lfc_corr_gse271188": float(s.log2fc.corr(s.log2fc_gse271188)),
                         "lfc_corr_kpmp_sc": float(s.log2fc.corr(s.log2fc_kpmp_sc))})
    res = pd.DataFrame(rows)
    res.to_csv(out / "sensitivity.tsv", sep="\t", index=False)
    print(res.round(3).to_string(index=False))
    write_provenance(STAGE, inputs=[], outputs=[out / "sensitivity.tsv"], seed=cfg["seed"],
                     extra={"n_label_transfer_genes": len(excl)})


if __name__ == "__main__":
    main()
