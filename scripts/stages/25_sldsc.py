"""Stage 25：分层 LD 分数回归（S-LDSC）——基因集注释在 baselineLD v2.2 之上的遗传力富集。

做法同 Finucane 等 2018 的基因集分析：每个基因集取基因体 ± window_kb 的 SNP 为注释，另以全部基因宇宙
（stage 18 的基因宇宙）作对照注释；回归同时包含 baselineLD v2.2、全部基因与该基因集，
报告该基因集注释系数 tau 的 Z 与单侧 P（tau > 0）、全部基因集 × 性状检验的 BH-FDR q 值，以及富集倍数。
步骤：注释（本环境）→ LD 分数（ldsc 环境，HapMap3 SNP）→ 汇总统计整理（munge）→ 回归。
输出：results/25_sldsc/{sldsc_results.tsv, annotation_sizes.tsv}
"""
from __future__ import annotations

import gzip
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

STAGE = "25_sldsc"
CHROMS = range(1, 23)


def ldsc(args: list[str], log: Path, env: str) -> None:
    with open(log, "w") as fh:
        subprocess.run(["conda", "run", "--no-capture-output", "-n", env] + args, check=True, stdout=fh,
                       stderr=subprocess.STDOUT)


def find(ref: Path, pattern: str) -> Path:
    hits = sorted(ref.rglob(pattern))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]


def gene_windows(sets: dict[str, set], universe: set, window: int) -> pd.DataFrame:
    loc = pd.read_csv(ROOT / "data/tools/magma/NCBI37.3.gene.loc", sep="\t", header=None,
                      names=["entrez", "chr", "start", "stop", "strand", "symbol"])
    loc = loc[loc.chr.astype(str).isin([str(c) for c in CHROMS])].drop_duplicates("symbol")
    loc["chr"] = loc.chr.astype(int)
    rows = []
    for name, genes in {**sets, "all_genes": universe}.items():
        g = loc[loc.symbol.isin(genes)]
        rows.append(g.assign(annot=name, lo=(g.start - window).clip(lower=0), hi=g.stop + window))
    return pd.concat(rows)


def write_annot(win: pd.DataFrame, bim: Path, out: Path) -> None:
    b = pd.read_csv(bim, sep="\t", header=None, names=["CHR", "SNP", "CM", "BP", "A1", "A2"])
    chr_ = int(b.CHR.iloc[0])
    w = win[win.chr == chr_]
    for name, g in w.groupby("annot"):
        hit = np.zeros(len(b), dtype=np.int8)
        bp = b.BP.values
        for lo, hi in zip(g.lo.values, g.hi.values):
            i, j = np.searchsorted(bp, lo, "left"), np.searchsorted(bp, hi, "right")
            hit[i:j] = 1
        with gzip.open(out / f"{name}.{chr_}.annot.gz", "wt") as fh:
            fh.write(name + "\n")
            fh.write("\n".join(map(str, hit)) + "\n")


COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}


def munge(path: Path, hm3a: pd.DataFrame) -> pd.DataFrame:
    """munge_sumstats.py 的规则（ldsc 1.0.1 默认）：HapMap3 SNP、P 在 (0,1]、MAF > 0.01、去链模糊与非 SNP、
    去重复 rs、N >= 第 90 百分位 / 1.5；Z = sign(BETA) * |Φ^-1(P/2)|；等位基因对齐到 HapMap3 表，不符者去除。"""
    from scipy.stats import norm
    d = pd.read_csv(path, sep="\t", usecols=["SNP", "A1", "A2", "FRQ", "BETA", "P", "N"])
    d = d[d.SNP.isin(hm3a.SNP)].dropna()
    d["A1"], d["A2"] = d.A1.str.upper(), d.A2.str.upper()
    d = d[(d.P > 0) & (d.P <= 1) & (np.minimum(d.FRQ, 1 - d.FRQ) > 0.01)]
    snp = d.A1.isin(COMP) & d.A2.isin(COMP) & (d.A1 != d.A2)
    d = d[snp & (d.A1.map(COMP) != d.A2)].drop_duplicates("SNP", keep=False)
    d = d[d.N >= d.N.quantile(0.9) / 1.5]
    d["Z"] = np.sign(d.BETA) * norm.isf(d.P / 2)
    m = hm3a.merge(d[["SNP", "A1", "A2", "Z", "N"]], on="SNP", how="left", suffixes=("", "_g"))
    same = (m.A1 == m.A1_g) & (m.A2 == m.A2_g) | (m.A1 == m.A1_g.map(COMP)) & (m.A2 == m.A2_g.map(COMP))
    flip = (m.A1 == m.A2_g) & (m.A2 == m.A1_g) | (m.A1 == m.A2_g.map(COMP)) & (m.A2 == m.A1_g.map(COMP))
    m.loc[flip & ~same, "Z"] *= -1
    m.loc[~(same | flip), ["Z", "N"]] = np.nan
    return m[["SNP", "A1", "A2", "Z", "N"]]


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    sc, pg = cfg["sldsc"], cfg["programme_genetics"]
    ref = ROOT / sc["ref_dir"]
    work = ROOT / "data/interim/sldsc"
    for sub in ("annot", "logs", "sumstats", "h2"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)

    gs = pd.read_csv(ROOT / "results/18_programme_genetics/gene_sets.tsv", sep="\t")
    sets = {s: set(gs.symbol[gs.gene_set == s]) for s in sc["gene_sets"]}
    univ = set(pd.read_csv(ROOT / "results/07b_magma" / f"{pg['traits'][0]}_genes.tsv", sep="\t").SYMBOL.dropna())
    win = gene_windows(sets, univ, sc["window_kb"] * 1000)
    names = list(sets) + ["all_genes"]
    bim = {c: find(ref, f"1000G.EUR.QC.{c}.bim") for c in CHROMS}
    hm3 = find(ref, "hm3_no_MHC.list.txt")
    for c in CHROMS:
        if not (work / "annot" / f"all_genes.{c}.annot.gz").exists():
            write_annot(win, bim[c], work / "annot")

    # LD 分数：每个注释 × 染色体
    jobs = []
    for n in names:
        for c in CHROMS:
            if not (work / "annot" / f"{n}.{c}.l2.ldscore.gz").exists():
                jobs.append(["ldsc.py", "--l2", "--bfile", str(bim[c].with_suffix("")), "--ld-wind-cm", "1",
                             "--annot", str(work / "annot" / f"{n}.{c}.annot.gz"), "--thin-annot",
                             "--out", str(work / "annot" / f"{n}.{c}"), "--print-snps", str(hm3)])
    with ThreadPoolExecutor(max_workers=sc["n_jobs"]) as ex:
        list(ex.map(lambda a: ldsc(a, work / "logs" / (Path(a[a.index("--out") + 1]).name + ".l2.log"), sc["env"]),
                    jobs))

    # 汇总统计整理：等位基因核对表 = HapMap3 SNP 在 1000G EUR 中的等位基因
    alle = work / "hm3_alleles.tsv"
    if not alle.exists():
        keep = set(pd.read_csv(hm3, header=None)[0])
        b = pd.concat(pd.read_csv(bim[c], sep="\t", header=None, usecols=[1, 4, 5], names=["SNP", "A1", "A2"])
                      for c in CHROMS)
        b[b.SNP.isin(keep)].to_csv(alle, sep="\t", index=False)
    hm3a = pd.read_csv(alle, sep="\t")
    for t in sc["traits"]:
        dst = work / "sumstats" / f"{t}.sumstats.gz"
        if not dst.exists():
            munge(ROOT / "data/processed/gwas" / f"{t}.tsv.gz", hm3a).to_csv(dst, sep="\t", index=False,
                                                                           compression="gzip", na_rep="NA")

    base = str(find(ref, "baselineLD.1.l2.ldscore.gz")).replace("1.l2.ldscore.gz", "")
    wts = str(find(ref, "weights.hm3_noMHC.1.l2.ldscore.gz")).replace("1.l2.ldscore.gz", "")
    frq = str(find(ref, "1000G.EUR.QC.1.frq")).replace("1.frq", "")
    rows = []
    for t in sc["traits"]:
        for n in sets:
            pre = work / "h2" / f"{t}.{n}"
            if not Path(f"{pre}.results").exists():
                ldsc(["ldsc.py", "--h2", str(work / "sumstats" / f"{t}.sumstats.gz"),
                      "--ref-ld-chr", f"{work / 'annot'}/{n}.,{work / 'annot'}/all_genes.,{base}",
                      "--w-ld-chr", wts, "--overlap-annot", "--frqfile-chr", frq, "--print-coefficients",
                      "--out", str(pre)], work / "logs" / f"h2_{t}.{n}.log", sc["env"])
            r = pd.read_csv(f"{pre}.results", sep="\t").iloc[0]
            z = r["Coefficient_z-score"]
            from scipy.stats import norm
            rows.append({"trait": t, "gene_set": n, "prop_snps": r["Prop._SNPs"], "prop_h2": r["Prop._h2"],
                         "enrichment": r["Enrichment"], "enrichment_p": r["Enrichment_p"],
                         "tau": r["Coefficient"], "tau_se": r["Coefficient_std_error"], "tau_z": z,
                         "tau_p_one_sided": float(norm.sf(z))})
    res = pd.DataFrame(rows)
    from statsmodels.stats.multitest import multipletests
    res["tau_q_bh"] = multipletests(res.tau_p_one_sided, method="fdr_bh")[1]   # 基因集 × 性状全部检验一并校正
    res.to_csv(out / "sldsc_results.tsv", sep="\t", index=False)
    win.groupby("annot").size().rename("n_genes").to_csv(out / "annotation_sizes.tsv", sep="\t")
    print(res.round(4).to_string(index=False))
    write_provenance(STAGE, inputs=[ROOT / "results/18_programme_genetics/gene_sets.tsv", hm3]
                     + [ROOT / "data/processed/gwas" / f"{t}.tsv.gz" for t in sc["traits"]],
                     outputs=[out / "sldsc_results.tsv", out / "annotation_sizes.tsv"], seed=cfg["seed"],
                     extra={"reference": "1000G Phase 3 EUR, baselineLD v2.2, HapMap3 weights (Zenodo 10515792)",
                            "window_kb": sc["window_kb"], "ldsc": "1.0.1"})


if __name__ == "__main__":
    main()
