"""Stage 07d：GWAS 显著位点与人肾皮质 eQTL（GTEx v8，eQTL Catalogue 全量统计）的共定位（coloc.abf）。

每个性状：P < p_lead 的 SNP 按距离聚簇成位点；每个位点取前导 SNP ± region_kb，
对区域内有 cis-eQTL 信号（最小 P < min_eqtl_p）的每个基因做 coloc.abf，输出 PP.H0–H4。
GWAS 为 GRCh37、GTEx 为 GRCh38，按 rsID 匹配并按等位基因对齐（效应等位基因 = eQTL 的 alt）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from gwas_io import define_loci, get_trait, processed_path  # noqa: E402
from coloc_abf import coloc_abf, pp4_with_prior  # noqa: E402

STAGE = "07d_coloc"
GTEX_COLS = ["gene_id", "ref", "alt", "maf", "pvalue", "beta", "se", "an", "rsid"]


def locus_snps(cfg: dict) -> tuple[dict, dict, set]:
    g = cfg["genetics"]
    loci, regions, rs = {}, {}, set()
    w = int(g["loci"]["region_kb"] * 1000)
    for t in g["coloc"]["traits"]:
        gw = get_trait(t)
        lc = define_loci(gw, g["loci"]["p_lead"], g["loci"]["clump_kb"])
        loci[t] = lc
        reg = []
        for r in lc.itertuples(index=False):
            sub = gw[(gw["CHR"] == r.CHR) & (gw["BP"].between(r.BP - w, r.BP + w))]
            reg.append(sub.assign(locus=r.locus, lead_snp=r.SNP))
        regions[t] = pd.concat(reg) if reg else gw.iloc[:0].assign(locus="", lead_snp="")
        rs.update(regions[t]["SNP"])
    return loci, regions, rs


def gtex_subset(path: Path, rs: set, cache: Path) -> pd.DataFrame:
    """流式读取 GTEx 全量 cis 统计，只保留位点区域内的 rsID；结果缓存在 data/interim。"""
    if cache.exists():
        return pd.read_csv(cache, sep="\t")
    keep = []
    for ch in pd.read_csv(path, sep="\t", usecols=GTEX_COLS, chunksize=5_000_000,
                          dtype={"rsid": str, "ref": str, "alt": str}):
        keep.append(ch[ch["rsid"].isin(rs)])
        print("  gtex chunk; kept so far", sum(len(k) for k in keep), flush=True)
    df = pd.concat(keep)
    df = df.dropna(subset=["beta", "se", "maf"])
    df = df[(df["se"] > 0) & (df["maf"] > 0)]
    df.to_csv(cache, sep="\t", index=False)
    return df


def align(gw: pd.DataFrame, eq: pd.DataFrame) -> pd.DataFrame:
    m = gw.merge(eq, left_on="SNP", right_on="rsid", how="inner")
    same = (m["A1"] == m["alt"]) & (m["A2"] == m["ref"])
    flip = (m["A1"] == m["ref"]) & (m["A2"] == m["alt"])
    m = m[same | flip].copy()
    m.loc[flip[same | flip].values, "BETA"] *= -1
    m["FRQ_ALT"] = np.where(flip[same | flip].values, 1 - m["FRQ"], m["FRQ"])
    m["MAF_GWAS"] = np.minimum(m["FRQ_ALT"], 1 - m["FRQ_ALT"])
    return m.drop_duplicates("SNP")


def run_trait(trait: str, regions: pd.DataFrame, gtex: pd.DataFrame, c: dict, n_eqtl_default: int,
              symbols: dict) -> pd.DataFrame:
    rows = []
    for locus, reg in regions.groupby("locus", sort=False):
        eq = gtex[gtex["rsid"].isin(reg["SNP"])]
        gene_minp = eq.groupby("gene_id")["pvalue"].min()
        for gene in gene_minp[gene_minp < c["min_eqtl_p"]].index:
            m = align(reg, eq[eq["gene_id"] == gene])
            if len(m) < c["min_snps"]:
                continue
            n_e = (m["an"] / 2).fillna(n_eqtl_default).values
            r = coloc_abf(m["BETA"].values, m["SE"].values ** 2, m["MAF_GWAS"].values, m["N"].values,
                          m["beta"].values, m["se"].values ** 2, m["maf"].values, n_e,
                          p1=c["p1"], p2=c["p2"], p12=c["p12"], sd_prior=c["sd_prior_quant"])
            top = int(np.argmax(r["snp_pp4"]))
            rec = {"trait": trait, "locus": locus, "lead_snp": reg["lead_snp"].iloc[0], "gene_id": gene,
                   "symbol": symbols.get(gene, ""), "tissue": "kidney_cortex_GTEx", "nsnps": r["nsnps"],
                   **{k: r[k] for k in ["PP.H0", "PP.H1", "PP.H2", "PP.H3", "PP.H4"]},
                   "top_snp": m["SNP"].iloc[top], "top_snp_pp4": r["snp_pp4"][top],
                   "gwas_minp_region": float(m["P"].min()), "eqtl_minp_region": float(m["pvalue"].min()),
                   "sdY_gwas": r["sdY1"], "sdY_eqtl": r["sdY2"],
                   "eqtl_beta_top": float(m["beta"].iloc[top]), "gwas_beta_top": float(m["BETA"].iloc[top])}
            for p12 in c["p12_sensitivity"]:
                rec[f"PP.H4_p12_{p12:g}"] = pp4_with_prior(r["labf1"], r["labf2"], c["p1"], c["p2"], p12)
            rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    g = cfg["genetics"]
    c = g["coloc"]
    e = g["eqtl"]["gtex_kidney_cortex"]
    out_dir = ROOT / "results" / STAGE
    out_dir.mkdir(parents=True, exist_ok=True)
    gtex_path = ROOT / "data/raw/eqtl" / e["url"].rsplit("/", 1)[-1]
    loci, regions, rs = locus_snps(cfg)
    print({t: len(v) for t, v in loci.items()}, "region SNPs:", len(rs), flush=True)
    gtex = gtex_subset(gtex_path, rs, ROOT / "data/interim/gtex_kidney_cortex_loci.tsv.gz")
    hg = pd.read_csv(ROOT / "data/raw/annot/hgnc_complete_set.txt", sep="\t", usecols=["symbol", "ensembl_gene_id"],
                     dtype=str).dropna()
    symbols = dict(zip(hg["ensembl_gene_id"], hg["symbol"]))
    outputs, summary = [], []
    for t in c["traits"]:
        lp = out_dir / f"{t}_loci.tsv"
        loci[t].to_csv(lp, sep="\t", index=False)
        res = run_trait(t, regions[t], gtex, c, e["n"], symbols)
        rp = out_dir / f"{t}_coloc_gtex_kidney_cortex.tsv"
        if len(res):
            res = res.sort_values("PP.H4", ascending=False)
        res.to_csv(rp, sep="\t", index=False)
        outputs += [lp, rp]
        strong = res[res["PP.H4"] >= c["pp4_flag"]] if len(res) else res
        summary.append({"trait": t, "n_loci": len(loci[t]), "n_gene_locus_tests": len(res),
                        "n_pp4_ge_flag": len(strong),
                        "n_loci_with_pp4_ge_flag": strong["locus"].nunique() if len(res) else 0,
                        "genes_pp4_ge_flag": ";".join(sorted(set(strong["symbol"].replace("", np.nan).dropna()))) if len(res) else ""})
        print(summary[-1], flush=True)
    sp = out_dir / "coloc_summary.tsv"
    pd.DataFrame(summary).to_csv(sp, sep="\t", index=False)
    outputs.append(sp)
    inputs = [processed_path(t) for t in c["traits"]] + [gtex_path]
    write_provenance(STAGE, inputs, outputs, seed=cfg["seed"],
                     extra={"method": "coloc.abf (Python port of R coloc v5 formulas)",
                            "priors": {k: c[k] for k in ["p1", "p2", "p12"]}, "eqtl": e["source"]})


if __name__ == "__main__":
    main()
