"""GWAS 汇总统计读取与统一格式化（GRCh37）。

统一列：SNP CHR BP A1 A2 FRQ BETA SE P N（A1 = 效应等位基因，大写）。
CKDGen 文件以空格分隔，部分列右对齐补空格；RSID 缺失的行只有 9 个字段（列错位），
这类行在 RSID 不以 "rs" 开头、N 为缺失两步过滤中被去掉。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from repro import ROOT, load_config

COLS = ["SNP", "CHR", "BP", "A1", "A2", "FRQ", "BETA", "SE", "P", "N"]


def _read_ckdgen(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", dtype={"RSID": str, "Chr": str, "Allele1": str, "Allele2": str},
                     na_values=["NA"], keep_default_na=False, engine="c")
    df = df.rename(columns={"Chr": "CHR", "Pos_b37": "BP", "RSID": "SNP", "Allele1": "A1",
                            "Allele2": "A2", "Freq1": "FRQ", "Effect": "BETA", "StdErr": "SE",
                            "P-value": "P", "n_total_sum": "N"})
    return df


def _read_winkler(path: Path, stratum: str) -> pd.DataFrame:
    s = stratum
    use = ["cpaid", "rsid", "Allele1", "Allele2", f"Freq1.{s}", f"Effect.{s}", f"StdErr.{s}",
           f"P.value.{s}", f"n.{s}"]
    df = pd.read_csv(path, sep="\t", usecols=use, na_values=["NA", ""], keep_default_na=False,
                     dtype={"cpaid": str, "rsid": str})
    cp = df["cpaid"].str.split(":", n=2, expand=True)
    df["CHR"], df["BP"] = cp[0], pd.to_numeric(cp[1], errors="coerce")
    df = df.rename(columns={"rsid": "SNP", "Allele1": "A1", "Allele2": "A2", f"Freq1.{s}": "FRQ",
                            f"Effect.{s}": "BETA", f"StdErr.{s}": "SE", f"P.value.{s}": "P",
                            f"n.{s}": "N"})
    return df


def load_trait(trait: str) -> pd.DataFrame:
    """读取一个性状的原始文件并清洗：去缺失、去非 rs 号、去重复 rs（保留 N 最大）。"""
    g = load_config()["genetics"]
    v = g["traits"][trait]
    path = ROOT / g["gwas_dir"] / v["file"]
    df = _read_ckdgen(path) if v["format"] == "ckdgen" else _read_winkler(path, v["stratum"])
    df = df[COLS].dropna(subset=["SNP", "P", "BETA", "SE", "N", "BP"])
    df = df[df["SNP"].astype(str).str.startswith("rs")]
    for c in ["BP", "FRQ", "BETA", "SE", "P", "N"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["BP", "FRQ", "BETA", "SE", "P", "N"])
    df["CHR"] = df["CHR"].astype(str).str.replace("chr", "", regex=False)
    df = df[df["CHR"].isin([str(i) for i in range(1, 23)])]
    df["CHR"] = df["CHR"].astype(int)
    df["BP"] = df["BP"].astype(np.int64)
    df["A1"], df["A2"] = df["A1"].str.upper(), df["A2"].str.upper()
    df = df[(df["P"] > 0) & (df["P"] <= 1) & (df["SE"] > 0)]
    df = df.sort_values("N", ascending=False).drop_duplicates("SNP").sort_values(["CHR", "BP"])
    return df.reset_index(drop=True)


def processed_path(trait: str) -> Path:
    return ROOT / "data/processed/gwas" / f"{trait}.tsv.gz"


def get_trait(trait: str) -> pd.DataFrame:
    """读取已统一格式的性状表；不存在则生成并缓存到 data/processed/gwas。"""
    p = processed_path(trait)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        load_trait(trait).to_csv(p, sep="\t", index=False)
    return pd.read_csv(p, sep="\t", dtype={"SNP": str, "A1": str, "A2": str})


def define_loci(df: pd.DataFrame, p_lead: float, clump_kb: float) -> pd.DataFrame:
    """距离贪婪聚簇：按 P 从小到大取前导 SNP，±clump_kb 内的显著 SNP 归入该位点。"""
    sig = df[df["P"] < p_lead].sort_values("P")
    leads, taken = [], {}
    w = int(clump_kb * 1000)
    for r in sig.itertuples(index=False):
        near = taken.setdefault(r.CHR, [])
        if any(abs(r.BP - b) <= w for b in near):
            continue
        near.append(r.BP)
        leads.append(r)
    out = pd.DataFrame(leads, columns=df.columns)
    out.insert(0, "locus", [f"chr{c}:{b}" for c, b in zip(out["CHR"], out["BP"])])
    return out.reset_index(drop=True)
