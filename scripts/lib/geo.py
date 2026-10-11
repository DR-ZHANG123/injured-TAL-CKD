"""GEO series matrix 解析：样本注释 + 表达矩阵（探针 × 样本）。"""
from __future__ import annotations

import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd


def read_series_matrix(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (samples, expr)。samples 以 GSM 为索引，含 title / platform / characteristics_* / data_processing。"""
    meta: dict[str, list[list[str]]] = {}
    table: list[str] = []
    in_table = False
    with gzip.open(path, "rt", errors="replace") as fh:
        for line in fh:
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                break
            if in_table:
                table.append(line)
            elif line.startswith("!Sample_"):
                key, *vals = line.rstrip("\n").split("\t")
                meta.setdefault(key[8:], []).append([v.strip('"') for v in vals])
    gsm = meta["geo_accession"][0]
    cols = {"title": meta["title"][0], "platform": meta["platform_id"][0]}
    for i, vals in enumerate(meta.get("characteristics_ch1", [])):
        cols[f"char_{i}"] = vals
    cols["data_processing"] = meta.get("data_processing", [[""] * len(gsm)])[0]
    samples = pd.DataFrame(cols, index=pd.Index(gsm, name="gsm"))
    expr = pd.DataFrame()
    if len(table) > 1:
        header = [h.strip('"') for h in table[0].rstrip("\n").split("\t")]
        rows = [r.rstrip("\n").split("\t") for r in table[1:] if r.strip()]
        if rows:
            expr = pd.DataFrame([r[1:] for r in rows], index=[r[0].strip('"') for r in rows], columns=header[1:])
            expr = expr.replace({"": np.nan, "null": np.nan, "NA": np.nan}).astype(float)
    return samples, expr


BATCH_PATTERNS = [r"\b(H\d)[-_ ]", r"\[(B\d)_", r"^(B\d)_"]


def batch_from_title(title: str) -> str | None:
    """ERCB 标题中的杂交批次 H1–H7、NEPTUNE 标题中的 B1–B4；找不到返回 None。"""
    for pat in BATCH_PATTERNS:
        m = re.search(pat, title)
        if m:
            return m.group(1)
    return None
