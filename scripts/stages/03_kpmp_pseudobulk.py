"""Stage 03：KPMP 图谱 v2（单核、单细胞）按 供体 × 细胞亚群（SubclassLevel2）汇总原始计数。

输出（每个模态）：
  data/processed/kpmp/<mod>_pseudobulk.parquet  行 = 供体|亚群，列 = 基因（原始计数之和）
  results/03_kpmp_pseudobulk/<mod>_cell_counts.tsv  供体 × 亚群 细胞数
  results/03_kpmp_pseudobulk/<mod>_donors.tsv       供体临床/入组信息
用法：python 03_kpmp_pseudobulk.py [--mod snRNA scRNA]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

STAGE = "03_kpmp_pseudobulk"
DONOR_KEY = "donor_id"                       # 两个模态共有的供体编号
DONOR_COLS = ["EnrollmentCategory", "AdjudicationCategory", "ConditionCategory", "diabetes_history",
              "hypertension_history", "hypertension", "Age", "sex", "Race", "Tissue.Type", "TissueCollection",
              "Baseline eGFR (ml/min/1.73m2) (Binned)", "ParticipantID"]


def pseudobulk(path: Path, chunk: int = 100_000) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    a = ad.read_h5ad(path, backed="r")
    obs = a.obs
    key = obs[DONOR_KEY].astype(str) + "|" + obs["SubclassLevel2"].astype(str)
    groups = pd.Index(sorted(key.unique()))
    gidx = groups.get_indexer(key)
    raw = a.raw if a.raw is not None else a
    genes = raw.var.get("feature_name", pd.Series(raw.var_names, index=raw.var_names)).astype(str).values
    acc = np.zeros((len(groups), raw.shape[1]), dtype=np.float64)
    for start in range(0, a.shape[0], chunk):
        stop = min(start + chunk, a.shape[0])
        X = raw.X[start:stop]
        X = X if sp.issparse(X) else sp.csr_matrix(X)
        ind = sp.csr_matrix((np.ones(stop - start), (gidx[start:stop], np.arange(stop - start))),
                            shape=(len(groups), stop - start))
        acc += (ind @ X).toarray()
        print(f"  {path.name}: {stop:,}/{a.shape[0]:,}", flush=True)
    pb = pd.DataFrame(acc.astype(np.float32), index=groups, columns=genes)
    pb = pb.T.groupby(level=0).sum().T                      # 合并重名基因
    counts = pd.crosstab(obs[DONOR_KEY].astype(str), obs["SubclassLevel2"].astype(str))
    donors = obs.drop_duplicates(DONOR_KEY).set_index(DONOR_KEY)[[c for c in DONOR_COLS if c in obs.columns]]
    donors.index = donors.index.astype(str)
    return pb, counts, donors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mod", nargs="+", default=["snRNA", "scRNA"])
    args = ap.parse_args()
    cfg = load_config()
    set_global_seed(cfg["seed"])
    raw = ROOT / cfg["paths"]["raw"] / "kpmp"
    out_d = ROOT / cfg["paths"]["processed"] / "kpmp"
    out_d.mkdir(parents=True, exist_ok=True)
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    ins, outs = [], []
    for mod in args.mod:
        src = raw / f"kpmp_v2_{mod}.h5ad"
        pb, counts, donors = pseudobulk(src)
        pb.to_parquet(out_d / f"{mod}_pseudobulk.parquet")
        counts.to_csv(out / f"{mod}_cell_counts.tsv", sep="\t")
        donors.to_csv(out / f"{mod}_donors.tsv", sep="\t")
        ins.append(src)
        outs += [out_d / f"{mod}_pseudobulk.parquet", out / f"{mod}_cell_counts.tsv", out / f"{mod}_donors.tsv"]
        print(mod, pb.shape, counts.shape, flush=True)
    write_provenance(STAGE, inputs=[], outputs=outs, seed=cfg["seed"],
                     extra={"inputs_note": "h5ad MD5 in results/00_download/kpmp_md5.tsv",
                            "modalities": args.mod})


if __name__ == "__main__":
    main()
