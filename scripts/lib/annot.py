"""探针 → Entrez → 基因符号。Brainarray ENTREZG（ID 形如 1234_at）直接取 Entrez；GPL571 / GPL17586 读 GPL 注释表。"""
from __future__ import annotations

import gzip
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


@lru_cache(maxsize=1)
def entrez_to_symbol(gene_info: str) -> dict[str, str]:
    df = pd.read_csv(gene_info, sep="\t", usecols=["GeneID", "Symbol"], dtype=str)
    return dict(zip(df.GeneID, df.Symbol))


def _gpl_table(path: Path) -> pd.DataFrame:
    with open(path, errors="replace") as fh:
        lines = [ln for ln in fh if not ln.startswith(("!", "#", "^"))]
    from io import StringIO
    return pd.read_csv(StringIO("".join(lines)), sep="\t", dtype=str, low_memory=False)


@lru_cache(maxsize=4)
def gpl_probe_entrez(path: str) -> dict[str, str]:
    t = _gpl_table(Path(path))
    if "ENTREZ_GENE_ID" in t.columns:                    # GPL571
        e = t.set_index("ID")["ENTREZ_GENE_ID"].dropna()
        return {k: v for k, v in e.items() if "///" not in v}
    if "gene_assignment" in t.columns:                   # GPL17586 (HTA 2.0)
        out = {}
        for pid, ga in zip(t["ID"], t["gene_assignment"].fillna("")):
            ids = {seg.split(" // ")[4].strip() for seg in ga.split(" /// ") if seg.count(" // ") >= 4}
            ids = {i for i in ids if i.isdigit()}
            if len(ids) == 1:
                out[pid] = ids.pop()
        return out
    raise ValueError(f"unknown GPL table layout: {path}")


def probe_to_entrez(ids: pd.Index, platform: str, annot_dir: Path) -> pd.Series:
    if platform in ("GPL571", "GPL17586"):
        m = gpl_probe_entrez(str(annot_dir / f"{platform}.txt"))
        return pd.Series([m.get(i) for i in ids], index=ids)
    pat = re.compile(r"^(\d+)(?:_at)?$")              # Brainarray ENTREZG：1234_at 或 1234
    return pd.Series([pat.match(i).group(1) if pat.match(i) else None for i in ids], index=ids)


def collapse_to_genes(expr: pd.DataFrame, platform: str, annot_dir: Path) -> pd.DataFrame:
    """每个基因保留方差最大的探针；行索引为基因符号。"""
    sym = entrez_to_symbol(str(annot_dir / "Homo_sapiens.gene_info.gz"))
    ent = probe_to_entrez(expr.index, platform, annot_dir)
    genes = ent.map(lambda e: sym.get(e) if e else None)
    x = expr.loc[genes.notna()].copy()
    x["_g"] = genes[genes.notna()].values
    x["_v"] = np.nanvar(x.drop(columns="_g").values, axis=1)
    x = x.sort_values("_v", ascending=False).drop_duplicates("_g")
    return x.set_index("_g").drop(columns="_v").rename_axis("gene")
