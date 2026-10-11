"""Stage 05：从 KPMP 健康参考供体定义细胞类型标志基因（不读任何疾病标签以外的分组）。
输出：results/05_markers/markers.tsv（marker_set, gene, specificity）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

STAGE = "05_markers"


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    mk = cfg["markers"]
    pb = pd.read_parquet(ROOT / cfg["paths"]["processed"] / "kpmp" / f"{mk['source']}_pseudobulk.parquet")
    don = pd.read_csv(ROOT / "results" / "03_kpmp_pseudobulk" / f"{mk['source']}_donors.tsv", sep="\t", index_col=0)
    ref = set(don.index[don["ConditionCategory"].isin(["Healthy Reference", "Reference"])].astype(str))
    donor = pb.index.str.split("|").str[0]
    sub = pb.index.str.split("|").str[1]
    x = pb[np.isin(donor, list(ref))]
    agg = x.groupby(sub[np.isin(donor, list(ref))]).sum()
    logcpm = np.log2(agg.div(agg.sum(1), axis=0) * 1e6 + 1)
    rows = []
    for name, targets in mk["targets"].items():
        t = logcpm.loc[[s for s in targets if s in logcpm.index]].mean(0)
        drop = targets + mk["exclude_related"].get(name, [])
        others = logcpm.drop(index=[s for s in drop if s in logcpm.index])
        if name == "injured_PT":                         # 损伤 PT 只与正常 PT 比
            others = logcpm.loc[[s for s in ["PT-S1/S2", "PT-S2/S3", "PT-S3"] if s in logcpm.index]]
        keep = (t >= mk["min_logcpm"]) & ~t.index.str.match(mk["drop_gene_regex"])
        spec = (t - others.max(0))[keep].sort_values(ascending=False).head(mk["n"])
        rows += [{"marker_set": name, "gene": g, "specificity": float(v)} for g, v in spec.items()]
        print(name, list(spec.index[:10]), flush=True)
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "markers.tsv", sep="\t", index=False)
    write_provenance(STAGE, inputs=[], outputs=[out / "markers.tsv"], seed=cfg["seed"],
                     extra={"n_reference_donors": len(ref)})


if __name__ == "__main__":
    main()
