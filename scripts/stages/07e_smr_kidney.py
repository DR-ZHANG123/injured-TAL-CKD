"""Stage 07e：SMR + HEIDI，检验肾组织（肾小管 / 肾小球 / 全肾 meta）基因表达与肾功能性状的关联。

eQTL：Susztak 实验室人肾 eQTL（仅公开 FDR<0.01 显著 SNP–基因对，GRCh37，效应等位基因 = Alt）。
SMR 只需工具 SNP（基因最强 cis-eQTL，P < p_eqtl），HEIDI 需要区域内其他显著 eQTL SNP，
显著对子集足以完成 SMR 默认 HEIDI 设定（eQTL P < 1.57e-3）。LD 来自 1000 Genomes EUR。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from bed_reader import open_bed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from gwas_io import get_trait, processed_path  # noqa: E402
from smr_heidi import heidi_test, select_heidi_snps, smr_test  # noqa: E402

STAGE = "07e_smr"


def load_eqtl(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    if "Effect2ALT" in df.columns:          # Liu 2022 meta
        df = df.rename(columns={"GeneSymbol": "symbol", "GeneID": "gene_id", "RSID": "rsid", "REF": "ref",
                                "ALT": "alt", "Effect2ALT": "beta", "SE": "se", "PVAL": "p"})
    else:                                   # Sheng 2021 tubule / glomerulus
        df = df.rename(columns={"Gene_Symbol": "symbol", "GeneID": "gene_id", "rsID": "rsid", "Ref": "ref",
                                "Alt": "alt", "Beta": "beta", "Std": "se", "Pvalue": "p"})
    df = df[["symbol", "gene_id", "rsid", "ref", "alt", "beta", "se", "p"]]
    for c in ["beta", "se", "p"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    df = df[df["rsid"].str.startswith("rs") & (df["se"] > 0)]
    df["gene_id"] = df["gene_id"].str.split(".").str[0]
    df["ref"], df["alt"] = df["ref"].str.upper(), df["alt"].str.upper()
    return df.drop_duplicates(["gene_id", "rsid"])


class LDRef:
    """1000 Genomes EUR PLINK 文件的按需 LD 计算（带符号相关，等位基因计数 = bim A1）。"""

    def __init__(self, prefix: Path):
        bim = pd.read_csv(f"{prefix}.bim", sep="\t", header=None, usecols=[1, 4, 5], names=["rsid", "a1", "a2"],
                          dtype=str)
        bim = bim.reset_index().drop_duplicates("rsid", keep=False)
        self.idx = dict(zip(bim["rsid"], bim["index"]))
        self.a1 = dict(zip(bim["rsid"], bim["a1"]))
        self.bed = open_bed(f"{prefix}.bed", count_A1=True, num_threads=8)

    def corr(self, rsids: list[str], effect_alleles: list[str]) -> np.ndarray:
        g = self.bed.read(index=np.s_[:, [self.idx[r] for r in rsids]], dtype="float32")
        g = np.where(np.isnan(g), np.nanmean(g, axis=0), g)
        sign = np.array([1.0 if self.a1[r] == a else -1.0 for r, a in zip(rsids, effect_alleles)])
        r = np.corrcoef(g * sign, rowvar=False)
        return np.nan_to_num(np.atleast_2d(r))


def align_gwas(gw: pd.DataFrame, eq: pd.DataFrame) -> pd.DataFrame:
    m = eq.merge(gw[["SNP", "A1", "A2", "BETA", "SE", "P"]], left_on="rsid", right_on="SNP")
    same = (m["A1"] == m["alt"]) & (m["A2"] == m["ref"])
    flip = (m["A1"] == m["ref"]) & (m["A2"] == m["alt"])
    m = m[same | flip].copy()
    m.loc[flip[same | flip], "BETA"] *= -1
    return m


def run_pair(trait: str, ds: str, gw: pd.DataFrame, eq: pd.DataFrame, ld: LDRef, s: dict) -> pd.DataFrame:
    m = align_gwas(gw, eq)
    m = m[m["rsid"].isin(ld.idx.keys()) & m["alt"].isin(["A", "C", "G", "T"])]
    rows = []
    for gene, sub in m.groupby("gene_id"):
        sub = sub.sort_values("p").reset_index(drop=True)
        if sub["p"].iloc[0] >= s["p_eqtl"]:
            continue
        t = sub.iloc[0]
        r = smr_test(t["BETA"], t["SE"], t["beta"], t["se"])
        rows.append({"trait": trait, "eqtl": ds, "gene_id": gene, "symbol": t["symbol"], "top_snp": t["rsid"],
                     "eqtl_beta": t["beta"], "eqtl_se": t["se"], "eqtl_p": t["p"],
                     "gwas_beta": t["BETA"], "gwas_se": t["SE"], "gwas_p": t["P"], **r, "_sub": sub})
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    n = len(res)
    res["p_smr_bonf_threshold"] = 0.05 / n
    res["smr_sig"] = res["p_smr"] < 0.05 / n
    res["n_heidi"], res["p_heidi"] = 0, np.nan
    for i in np.where(res["smr_sig"])[0]:
        sub = res.at[i, "_sub"]
        sub = sub[sub["p"] < s["heidi_p_lower"]].head(200).reset_index(drop=True)
        if len(sub) < s["heidi_min_snps"]:
            continue
        R = ld.corr(sub["rsid"].tolist(), sub["alt"].tolist())
        ch = select_heidi_snps(sub["p"].values, R[0], R, 0, s["heidi_p_lower"], s["heidi_ld_lower"],
                               s["heidi_ld_upper"], s["heidi_max_snps"])
        if len(ch) < s["heidi_min_snps"]:
            continue
        ss = sub.iloc[ch]
        h = heidi_test(ss["BETA"].values, ss["SE"].values, ss["beta"].values, ss["se"].values,
                       R[np.ix_(ch, ch)], 0)
        res.at[i, "n_heidi"], res.at[i, "p_heidi"] = h["n_heidi"], h["p_heidi"]
    res["heidi_pass"] = res["smr_sig"] & (res["p_heidi"] >= s["heidi_flag"])
    return res.drop(columns="_sub").sort_values("p_smr")


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    g = cfg["genetics"]
    s = g["smr"]
    out_dir = ROOT / "results" / STAGE
    out_dir.mkdir(parents=True, exist_ok=True)
    ld = LDRef(ROOT / g["tools_dir"] / "g1000_eur")
    eqtls = {ds: load_eqtl(ROOT / "data/raw/eqtl" / g["eqtl"][ds]["file"]) for ds in s["datasets"]}
    outputs, summary = [], []
    for t in s["traits"]:
        gw = get_trait(t)
        for ds, eq in eqtls.items():
            res = run_pair(t, ds, gw, eq, ld, s)
            p = out_dir / f"{t}_{ds}_smr.tsv"
            res.to_csv(p, sep="\t", index=False)
            outputs.append(p)
            summary.append({"trait": t, "eqtl": ds, "n_genes_tested": len(res),
                            "n_smr_sig": int(res["smr_sig"].sum()) if len(res) else 0,
                            "n_smr_sig_heidi_pass": int(res["heidi_pass"].sum()) if len(res) else 0,
                            "genes_smr_heidi": ";".join(res.loc[res["heidi_pass"], "symbol"].astype(str)) if len(res) else ""})
            print({k: v for k, v in summary[-1].items() if k != "genes_smr_heidi"}, flush=True)
    sp = out_dir / "smr_summary.tsv"
    pd.DataFrame(summary).to_csv(sp, sep="\t", index=False)
    outputs.append(sp)
    inputs = [processed_path(t) for t in s["traits"]] + [ROOT / "data/raw/eqtl" / g["eqtl"][d]["file"] for d in s["datasets"]]
    write_provenance(STAGE, inputs, outputs, seed=cfg["seed"],
                     extra={"method": "SMR + HEIDI (Zhu et al. 2016), Python implementation; HEIDI P by Imhof",
                            "ld_reference": "1000 Genomes EUR (MAGMA g1000_eur)"})


if __name__ == "__main__":
    main()
