"""Stage 10 共用函数：GSE286184 Xenium 读入、KPMP 参照质心、相关注释、供体级统计。

由 10_xenium_*.py 通过 importlib 载入（文件名以数字开头，不能直接 import）。
"""
from __future__ import annotations

import gzip
import hashlib
import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.geo import read_series_matrix  # noqa: E402
from lib.pbstats import hedges_g  # noqa: E402
from lib.repro import ROOT  # noqa: E402


# ---------------------------------------------------------------- 样本与读入
def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sample_table(xc: dict) -> pd.DataFrame:
    """series matrix → 每个 GSM 的诊断、分组、文件前缀；内容完全相同的存放（细胞矩阵 MD5 一致）标为重复。"""
    raw = ROOT / xc["raw_dir"]
    smp, _ = read_series_matrix(raw / f"{xc['geo']}_series_matrix.txt.gz")
    chars = smp.filter(like="char_").fillna("").agg(" ".join, axis=1)
    diag = chars.str.extract(r"condition: (\S+)", expand=False)
    grp_of = {d: g for g, ds in xc["groups"].items() for d in ds}
    rows = []
    for gsm in smp.index:
        stem = next(raw.glob(f"{gsm}_*_matrix.mtx.gz")).name.replace("_matrix.mtx.gz", "")
        rows.append({"gsm": gsm, "title": smp.loc[gsm, "title"], "diagnosis": diag[gsm],
                     "group": grp_of.get(diag[gsm], "other"), "stem": stem,
                     "matrix_md5": file_md5(raw / f"{stem}_matrix.mtx.gz")})
    t = pd.DataFrame(rows).set_index("gsm")
    first = t.reset_index().groupby("matrix_md5")["gsm"].first()
    t["duplicate_of"] = [first[m] if first[m] != g else "" for g, m in zip(t.index, t.matrix_md5)]
    return t


def read_cells(xc: dict, stem: str) -> tuple[sp.csr_matrix, list[str], pd.DataFrame]:
    """细胞 × 面板基因原始计数（只保留 Gene Expression 特征）+ 细胞坐标表（与 barcodes 顺序核对）。"""
    raw = ROOT / xc["raw_dir"]
    feats = pd.read_csv(raw / f"{stem}_features.tsv.gz", sep="\t", header=None)
    gmask = (feats[2] == "Gene Expression").values
    m = scipy.io.mmread(raw / f"{stem}_matrix.mtx.gz").T.tocsr()[:, gmask]
    cells = pd.read_csv(raw / f"{stem}_cells.csv.gz")
    bc = pd.read_csv(raw / f"{stem}_barcodes.tsv.gz", header=None)[0].values
    if not (bc == cells["cell_id"].values).all():
        raise ValueError(f"{stem}: barcodes 与 cells.csv 顺序不一致")
    return m, feats.loc[gmask, 1].tolist(), cells


def read_transcripts(xc: dict, stem: str, cols: list[str]) -> pd.DataFrame:
    raw = ROOT / xc["raw_dir"]
    with gzip.open(raw / f"{stem}_transcripts.parquet.gz") as fh:
        t = pd.read_parquet(io.BytesIO(fh.read()), columns=cols)
    return t


def run_indicators(xc: dict, stem: str, panel: list[str]) -> dict:
    """处理批次指纹：transcripts 表字段数（Xenium Analyzer 版本）、qv≥阈值比例、阴性对照/未分配码字比例、基因转录本密度。"""
    raw = ROOT / xc["raw_dir"]
    with gzip.open(raw / f"{stem}_transcripts.parquet.gz") as fh:
        t = pd.read_parquet(io.BytesIO(fh.read()))
    gene = t["feature_name"].isin(panel)
    g = t[gene & (t["qv"] >= xc["min_qv"])]
    area_mm2 = (g.x_location.max() - g.x_location.min()) * (g.y_location.max() - g.y_location.min()) / 1e6
    return {"n_transcript_columns": t.shape[1], "frac_qv_pass": float((t["qv"] >= xc["min_qv"]).mean()),
            "frac_non_gene_codewords": float((~gene).mean()), "n_gene_transcripts": int(len(g)),
            "gene_transcripts_per_mm2_bbox": float(len(g) / area_mm2)}


# ---------------------------------------------------------------- KPMP 参照
def kpmp_reference(cfg: dict, panel: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """KPMP 单核 pseudobulk（全部供体合计，不用疾病标签）→ 每个 SubclassLevel2 的面板基因 CPM（以全转录组为分母），
    减去跨亚群的环境 RNA 底（分位数）后 log1p。返回 (亚群 × 面板基因 log 表达, 基因特异性)。"""
    rc = cfg["xenium"]["reference"]
    pb = pd.read_parquet(ROOT / cfg["paths"]["processed"] / "kpmp" / "snRNA_pseudobulk.parquet")
    sub = pb.index.str.split("|").str[1]
    lib = pb.sum(1).groupby(sub).sum()
    cpm = pb[panel].groupby(sub).sum().div(lib, axis=0) * 1e6
    keep = [s for s in cpm.index if not re.match(rc["exclude_regex"], s) and s not in set(rc["exclude"])]
    cpm = cpm.loc[keep]
    cpm = (cpm - cpm.quantile(rc["ambient_quantile"])).clip(lower=0)
    ref = np.log1p(cpm)
    spec = ref.max() - ref.median()
    return ref, spec


def zrows(a: np.ndarray) -> np.ndarray:
    return (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-9)


def correlate(m: sp.csr_matrix, panel: list[str], ref: pd.DataFrame, genes: list[str],
              classes: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """细胞在 genes 上的 log1p(CP10k，面板内归一化) 与参照质心的 Pearson 相关；返回最佳亚群与相关系数。"""
    r = ref if classes is None else ref.loc[[c for c in classes if c in ref.index]]
    gi = [panel.index(g) for g in genes]
    x = m[:, gi].toarray().astype(np.float64)
    tot = np.asarray(m.sum(1)).ravel()[:, None]
    x = zrows(np.log1p(x / np.maximum(tot, 1) * 1e4))
    c = x @ zrows(r[genes].values).T / len(genes)
    return r.index.values[c.argmax(1)], c.max(1)


def lineage_of(xc: dict) -> dict[str, str]:
    return {s: lin for lin, subs in xc["lineages"].items() for s in subs}


# ---------------------------------------------------------------- 供体级统计
def contrast_rows(df: pd.DataFrame, measures: list[str], group: pd.Series, batch: pd.Series,
                  contrasts: list[list[str]], min_n: int) -> pd.DataFrame:
    """每个读数、每个对比：两组供体数、中位数、Hedges g、双侧 Mann–Whitney P，以及两组是否同一处理批次。"""
    rows = []
    for meas in measures:
        v = df[meas].dropna()
        g = group.reindex(v.index)
        for a, b in contrasts:
            x, y = v[g == a], v[g == b]
            if len(x) < min_n or len(y) < min_n:
                continue
            same = set(batch.reindex(x.index)) | set(batch.reindex(y.index))
            rows.append({"measure": meas, "contrast": f"{a}_vs_{b}", "n_a": len(x), "n_b": len(y),
                         "median_a": float(x.median()), "median_b": float(y.median()),
                         "hedges_g": float(hedges_g(x.values[:, None], y.values[:, None])[0]),
                         "p_mannwhitney": float(mannwhitneyu(x, y, alternative="two-sided").pvalue),
                         "same_processing_batch": len(same) == 1})
    return pd.DataFrame(rows)
