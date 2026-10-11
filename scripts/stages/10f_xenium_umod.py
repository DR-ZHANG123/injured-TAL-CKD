"""Stage 10f：原位 UMOD——损伤 TAL 细胞是否丢失 UMOD 转录本（不让 UMOD 参与损伤标签）。

TAL 谱系细胞（stage 10a 注释，不含致密斑）在 TAL 状态参照质心之间重新注释，所用基因去掉 UMOD；
每个细胞的 UMOD 读数 = UMOD 转录本 / 非 UMOD 面板转录本 × 100（避免 UMOD 自身进入分母）。
每位供体：重新注释的损伤 TAL 与健康 TAL 的 UMOD 读数中位数及 UMOD 阳性比例；跨供体 Wilcoxon 符号秩检验。
输出：results/10f_xenium_umod/{donor_umod.tsv, umod_paired_tests.tsv}
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

_spec = importlib.util.spec_from_file_location("xlib", Path(__file__).with_name("10_xenium_lib.py"))
xlib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(xlib)

STAGE = "10f_xenium_umod"


def donor_record(cells: pd.DataFrame, m: sp.csr_matrix, panel: list[str], ref: pd.DataFrame, genes: list[str],
                 tc: dict) -> dict:
    sel = (cells["lineage"] == "TAL").values & cells["label"].isin(tc["tal_labels"]).values
    if sel.sum() == 0:
        return {"n_TAL": 0}
    mt = m[sel]
    lab, _ = xlib.correlate(mt, panel, ref, genes, classes=tc["tal_labels"])
    inj = np.isin(lab, tc["tal_injured"])
    ui = panel.index("UMOD")
    umod = np.asarray(mt[:, ui].todense()).ravel()
    other = np.asarray(mt.sum(1)).ravel() - umod
    score = umod / np.maximum(other, 1) * 100
    rec = {"n_TAL": int(sel.sum()), "n_TAL_injured": int(inj.sum()), "frac_injured_relabelled": float(inj.mean()),
           "agreement_with_original": float((inj == cells.loc[sel, "label"].isin(tc["tal_injured"]).values).mean())}
    if inj.sum() >= tc["min_cells"] and (~inj).sum() >= tc["min_cells"]:
        rec.update({"umod_injured": float(np.median(score[inj])), "umod_healthy": float(np.median(score[~inj])),
                    "umod_pos_injured": float((umod[inj] > 0).mean()), "umod_pos_healthy": float((umod[~inj] > 0).mean())})
        rec["umod_diff"] = rec["umod_injured"] - rec["umod_healthy"]
        rec["umod_pos_diff"] = rec["umod_pos_injured"] - rec["umod_pos_healthy"]
    return rec


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    xc, tc = cfg["xenium"], cfg["xenium"]["tal"]
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    proc = ROOT / xc["processed_dir"]
    panel = pd.read_csv(proc / "panel_genes.tsv", sep="\t")["gene"].tolist()
    ref, spec = xlib.kpmp_reference(cfg, panel)
    genes = [g for g in panel if spec[g] >= xc["reference"]["min_specificity"] and g != "UMOD"]
    samples = pd.read_csv(ROOT / "results" / "10_xenium_annotate" / "samples.tsv", sep="\t", index_col=0)
    samples = samples[samples["duplicate_of"].isna()]
    rows = []
    for gsm, row in samples.iterrows():
        cells = pd.read_parquet(proc / f"{gsm}_cells.parquet")
        m = sp.load_npz(proc / f"{gsm}_counts.npz").tocsr()
        rows.append({"gsm": gsm, "diagnosis": row.diagnosis, "group": row.group,
                     "processing_batch": row.processing_batch, **donor_record(cells, m, panel, ref, genes, tc)})
    don = pd.DataFrame(rows).set_index("gsm")
    don.to_csv(out / "donor_umod.tsv", sep="\t")
    tests = []
    for k in ["umod_diff", "umod_pos_diff"]:
        for grp in ["all"] + list(xc["groups"]):
            v = don.loc[(don["group"] == grp) | (grp == "all"), k].dropna()
            if len(v) < xc["min_donors"]:
                continue
            tests.append({"measure": k, "donors": grp, "n": len(v), "median_diff": float(v.median()),
                          "n_lower": int((v < 0).sum()), "p_wilcoxon": float(wilcoxon(v).pvalue)})
    pd.DataFrame(tests).to_csv(out / "umod_paired_tests.tsv", sep="\t", index=False)
    print(don.round(3).to_string())
    print(pd.DataFrame(tests).round(5).to_string(index=False))
    write_provenance(STAGE, inputs=[ROOT / "results" / "10_xenium_annotate" / "samples.tsv"]
                     + [proc / f"{g}_cells.parquet" for g in samples.index]
                     + [proc / f"{g}_counts.npz" for g in samples.index],
                     outputs=[out / "donor_umod.tsv", out / "umod_paired_tests.tsv"], seed=cfg["seed"],
                     extra={"genes_for_relabelling": genes})


if __name__ == "__main__":
    main()
