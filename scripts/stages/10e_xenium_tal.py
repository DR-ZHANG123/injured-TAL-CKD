"""Stage 10e：损伤 TAL（aTAL、frTAL）在 Xenium 切片中的比例与微环境。

每位供体：TAL 谱系（不含致密斑 MD）中损伤细胞的比例（另给共同深度区间内的比例）；每个 TAL 细胞 radius_um 内
是否有免疫细胞、成纤维细胞（含肌成纤维细胞）。损伤 TAL 与同一供体内正常 TAL 的差值作为供体级读数，
跨供体用 Wilcoxon 符号秩检验；比例的组间比较用 Mann–Whitney 检验（均以供体为单位，不用细胞级 P 值）。
输出：results/10e_xenium_tal/{donor_tal.tsv, tal_paired_tests.tsv, tal_fraction_tests.tsv}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

STAGE = "10e_xenium_tal"


def neighbour_share(xy: np.ndarray, other: np.ndarray, radius: float) -> np.ndarray:
    if not len(other):
        return np.zeros(len(xy), dtype=bool)
    return cKDTree(other).query(xy, k=1)[0] <= radius


def donor_record(cells: pd.DataFrame, tc: dict, min_tx: int) -> dict:
    tal = cells[(cells["lineage"] == "TAL") & cells["label"].isin(tc["tal_labels"])]
    inj = tal["label"].isin(tc["tal_injured"]).values
    band = ((tal["n_tx"] >= min_tx) & (tal["n_tx"] < 2 * min_tx)).values
    rec = {"n_TAL": len(tal), "n_TAL_injured": int(inj.sum()),
           "frac_injured_of_TAL": float(inj.mean()) if len(tal) else np.nan,
           "frac_injured_of_TAL_depth_band": float(inj[band].mean()) if band.sum() >= tc["min_cells"] else np.nan}
    if inj.sum() < tc["min_cells"] or (~inj).sum() < tc["min_cells"]:
        return rec
    xy = tal[["x", "y"]].values
    for name, sel in tc["neighbours"].items():
        mask = cells["lineage"].isin(sel["lineages"])
        if sel.get("labels"):
            mask &= cells["label"].isin(sel["labels"])
        other = cells.loc[mask, ["x", "y"]].values
        near = neighbour_share(xy, other, tc["radius_um"])
        rec[f"{name}_nb_injured"] = float(near[inj].mean())
        rec[f"{name}_nb_normal"] = float(near[~inj].mean())
        rec[f"{name}_nb_diff"] = rec[f"{name}_nb_injured"] - rec[f"{name}_nb_normal"]
    return rec


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    xc, tc = cfg["xenium"], cfg["xenium"]["tal"]
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    proc = ROOT / xc["processed_dir"]
    samples = pd.read_csv(ROOT / "results" / "10_xenium_annotate" / "samples.tsv", sep="\t", index_col=0)
    samples = samples[samples["duplicate_of"].isna()]
    rows = []
    for gsm, row in samples.iterrows():
        cells = pd.read_parquet(proc / f"{gsm}_cells.parquet")
        rows.append({"gsm": gsm, "diagnosis": row.diagnosis, "group": row.group,
                     "processing_batch": row.processing_batch,
                     **donor_record(cells, tc, xc["min_transcripts"])})
    don = pd.DataFrame(rows).set_index("gsm")
    don.to_csv(out / "donor_tal.tsv", sep="\t")

    paired = []
    for name in tc["neighbours"]:
        k = f"{name}_nb_diff"
        for grp in ["all"] + list(xc["groups"]):
            v = don.loc[(don["group"] == grp) | (grp == "all"), k].dropna() if k in don else pd.Series(dtype=float)
            if len(v) < xc["min_donors"]:
                continue
            paired.append({"neighbour": name, "donors": grp, "n": len(v), "median_diff": float(v.median()),
                           "n_positive": int((v > 0).sum()), "p_wilcoxon": float(wilcoxon(v).pvalue)})
    pd.DataFrame(paired).to_csv(out / "tal_paired_tests.tsv", sep="\t", index=False)

    frac = []
    for col in ["frac_injured_of_TAL", "frac_injured_of_TAL_depth_band"]:
        for a, b in xc["contrasts"]:
            x, y = don.loc[don["group"] == a, col].dropna(), don.loc[don["group"] == b, col].dropna()
            if min(len(x), len(y)) < xc["min_donors"]:
                continue
            frac.append({"measure": col, "group_a": a, "group_b": b, "n_a": len(x), "n_b": len(y),
                         "median_a": float(x.median()), "median_b": float(y.median()),
                         "p_mannwhitney": float(mannwhitneyu(x, y).pvalue)})
    pd.DataFrame(frac).to_csv(out / "tal_fraction_tests.tsv", sep="\t", index=False)
    print(don.round(3).to_string())
    print(pd.DataFrame(paired).round(4).to_string(index=False))
    print(pd.DataFrame(frac).round(4).to_string(index=False))
    write_provenance(STAGE, inputs=[ROOT / "results" / "10_xenium_annotate" / "samples.tsv"]
                     + [proc / f"{g}_cells.parquet" for g in samples.index],
                     outputs=[out / "donor_tal.tsv", out / "tal_paired_tests.tsv", out / "tal_fraction_tests.tsv"],
                     seed=cfg["seed"])


if __name__ == "__main__":
    main()
