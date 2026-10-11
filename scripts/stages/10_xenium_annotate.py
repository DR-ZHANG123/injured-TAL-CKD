"""Stage 10a：GSE286184 Xenium 读入、细胞 QC 与 KPMP 参照相关注释。

1. series matrix → 每张切片的诊断；细胞矩阵 MD5 相同的存放视为同一样本，只保留一次。
2. 每张切片的处理批次指纹（Xenium Analyzer 输出字段数、qv 合格率、非基因码字比例、转录本密度）。
3. 细胞 QC（面板基因转录本数 ≥ xenium.min_transcripts）。
4. 以 KPMP 单核各 SubclassLevel2 的面板基因质心（扣除环境 RNA 底）为参照，在高特异性面板基因上取 Pearson 相关最高的亚群；
   r < xenium.min_r 记为 unassigned。
输出：data/processed/xenium/<gsm>_{counts.npz,cells.parquet}；
      results/10_xenium_annotate/{samples.tsv, reference_genes.tsv, label_distribution.tsv, lineage_distribution.tsv}
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

_spec = importlib.util.spec_from_file_location("xlib", Path(__file__).with_name("10_xenium_lib.py"))
xlib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(xlib)

STAGE = "10_xenium_annotate"


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    xc = cfg["xenium"]
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    proc = ROOT / xc["processed_dir"]
    proc.mkdir(parents=True, exist_ok=True)

    samples = xlib.sample_table(xc)
    lin = xlib.lineage_of(xc)
    ref = spec = genes = panel0 = None
    label_rows, outputs, qc = [], [], {}
    for gsm, row in samples.iterrows():
        m, panel, cells = xlib.read_cells(xc, row.stem)
        if panel0 is None:
            panel0 = panel
            ref, spec = xlib.kpmp_reference(cfg, panel)
            genes = [g for g in panel if spec[g] >= xc["reference"]["min_specificity"]]
            pd.DataFrame({"gene": spec.index, "specificity": spec.values, "used_for_annotation": spec.index.isin(genes),
                          "top_subclass": ref.idxmax().reindex(spec.index).values}).to_csv(
                out / "reference_genes.tsv", sep="\t", index=False)
        elif panel != panel0:
            raise ValueError(f"{gsm}: 面板与首个样本不一致")
        ntx = np.asarray(m.sum(1)).ravel()
        keep = ntx >= xc["min_transcripts"]
        q = xlib.run_indicators(xc, row.stem, panel)
        q.update({"n_cells_total": len(ntx), "n_cells_qc": int(keep.sum()),
                  "median_tx_per_cell_all": float(np.median(ntx)), "median_tx_per_cell_qc": float(np.median(ntx[keep]))})
        qc[gsm] = q
        if row.duplicate_of:
            print(gsm, "duplicate of", row.duplicate_of, flush=True)
            continue
        mq = m[keep].tocsr()
        lab, r = xlib.correlate(mq, panel, ref, genes)
        lab = np.where(r >= xc["min_r"], lab, "unassigned")
        cdf = pd.DataFrame({"cell_id": cells.loc[keep, "cell_id"].values,
                            "x": cells.loc[keep, "x_centroid"].values, "y": cells.loc[keep, "y_centroid"].values,
                            "n_tx": ntx[keep].astype(int), "cell_area": cells.loc[keep, "cell_area"].values,
                            "label": lab, "r": r.astype(np.float32)})
        cdf["lineage"] = cdf["label"].map(lin).fillna("unassigned")
        cdf.to_parquet(proc / f"{gsm}_cells.parquet", index=False)
        sp.save_npz(proc / f"{gsm}_counts.npz", mq.astype(np.int32))
        outputs += [proc / f"{gsm}_cells.parquet", proc / f"{gsm}_counts.npz"]
        vc = cdf["label"].value_counts()
        for l, n in vc.items():
            label_rows.append({"gsm": gsm, "diagnosis": row.diagnosis, "label": l, "lineage": lin.get(l, "unassigned"),
                               "n_cells": int(n), "fraction": n / len(cdf)})
        q.update({"frac_unassigned": float((lab == "unassigned").mean()), "median_r": float(np.median(r))})
        print(gsm, row.diagnosis, int(keep.sum()), round(float(np.median(r)), 3), flush=True)
    (pd.Series(panel0, name="gene")).to_csv(proc / "panel_genes.tsv", sep="\t", index=False)

    samples = samples.join(pd.DataFrame(qc).T)
    # 处理批次：transcripts 表字段数标记 Xenium Analyzer 输出版本；与质量指标一并写出，供核对
    samples["processing_batch"] = "run_cols" + samples["n_transcript_columns"].astype(int).astype(str)
    samples.drop(columns=["stem"]).to_csv(out / "samples.tsv", sep="\t")
    lab = pd.DataFrame(label_rows)
    lab.to_csv(out / "label_distribution.tsv", sep="\t", index=False)
    lin_tab = lab.groupby(["gsm", "diagnosis", "lineage"])["n_cells"].sum().reset_index()
    lin_tab["fraction"] = lin_tab["n_cells"] / lin_tab.groupby("gsm")["n_cells"].transform("sum")
    lin_tab.to_csv(out / "lineage_distribution.tsv", sep="\t", index=False)
    print(lin_tab.pivot(index="gsm", columns="lineage", values="fraction").round(3).to_string())
    print(samples[["diagnosis", "processing_batch", "frac_qv_pass", "frac_non_gene_codewords",
                   "median_tx_per_cell_all", "n_cells_qc", "duplicate_of"]].to_string())
    raw = ROOT / xc["raw_dir"]
    write_provenance(STAGE, inputs=[raw / f"{xc['geo']}_series_matrix.txt.gz",
                                    ROOT / cfg["paths"]["processed"] / "kpmp" / "snRNA_pseudobulk.parquet"]
                     + sorted(raw.glob("*_matrix.mtx.gz")),
                     outputs=[out / "samples.tsv", out / "reference_genes.tsv", out / "label_distribution.tsv",
                              out / "lineage_distribution.tsv"] + outputs, seed=cfg["seed"],
                     extra={"n_genes_for_annotation": len(genes), "genes_for_annotation": genes})


if __name__ == "__main__":
    main()
