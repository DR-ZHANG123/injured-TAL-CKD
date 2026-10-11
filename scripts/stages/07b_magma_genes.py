"""Stage 07b：MAGMA SNP→基因注释与基因水平分析（每个肾功能性状一张基因 Z/P 表）。

注释窗口、模型、并行批次、显著性阈值均来自 config/run.yaml genetics.magma。
LD 参考：1000 Genomes 欧洲人群（MAGMA 官方 g1000_eur），基因坐标 NCBI37.3（GRCh37）。
"""
from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from gwas_io import get_trait, processed_path  # noqa: E402

STAGE = "07b_magma"


def run(cmd: list[str], log: Path) -> None:
    with open(log, "w") as fh:
        subprocess.run(cmd, check=True, stdout=fh, stderr=subprocess.STDOUT)


def annotate(tools: Path, work: Path, m: dict) -> Path:
    out = work / "annot"
    if not (work / "annot.genes.annot").exists():
        run([str(tools / "magma"), "--annotate", f"window={m['window_up_kb']},{m['window_down_kb']}",
             "--snp-loc", str(tools / "g1000_eur.bim"), "--gene-loc", str(tools / "NCBI37.3.gene.loc"),
             "--out", str(out)], work / "annot.stdout")
    return work / "annot.genes.annot"


def gene_analysis(trait: str, tools: Path, work: Path, annot: Path, m: dict) -> Path:
    gw = get_trait(trait)
    pfile = work / f"{trait}.pval.tsv"
    gw[["SNP", "P", "N"]].to_csv(pfile, sep="\t", index=False)
    prefix = work / trait
    if not Path(f"{prefix}.genes.out").exists():
        k = m["n_batches"]
        base = [str(tools / "magma"), "--bfile", str(tools / "g1000_eur"),
                "--pval", str(pfile), "use=SNP,P", "ncol=N", "--gene-annot", str(annot),
                "--gene-model", m["model"]]
        with ThreadPoolExecutor(max_workers=k) as ex:
            futs = [ex.submit(run, base + ["--batch", str(i), str(k), "--out", str(prefix)],
                              work / f"{trait}.batch{i}.stdout") for i in range(1, k + 1)]
            for f in futs:
                f.result()
        run([str(tools / "magma"), "--merge", str(prefix), "--out", str(prefix)],
            work / f"{trait}.merge.stdout")
    return Path(f"{prefix}.genes.out")


def tidy(genes_out: Path, geneloc: Path, alpha: float) -> pd.DataFrame:
    df = pd.read_csv(genes_out, sep=r"\s+")
    loc = pd.read_csv(geneloc, sep="\t", header=None, names=["GENE", "CHR", "START", "STOP", "STRAND", "SYMBOL"])
    df = df.merge(loc[["GENE", "SYMBOL"]], on="GENE", how="left")
    n = len(df)
    df["P_BONF"] = (df["P"] * n).clip(upper=1.0)
    df["SIG_BONF"] = df["P"] < alpha / n
    from statsmodels.stats.multitest import multipletests
    df["Q_FDR"] = multipletests(df["P"], method="fdr_bh")[1]
    cols = ["SYMBOL", "ENTREZ", "CHR", "START", "STOP", "NSNPS", "NPARAM", "N", "ZSTAT", "P", "P_BONF", "Q_FDR", "SIG_BONF"]
    return df.rename(columns={"GENE": "ENTREZ"})[cols].sort_values("P")


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    g = cfg["genetics"]
    m = g["magma"]
    tools = ROOT / g["tools_dir"]
    work = ROOT / "data/interim/magma"
    work.mkdir(parents=True, exist_ok=True)
    out_dir = ROOT / "results" / STAGE
    out_dir.mkdir(parents=True, exist_ok=True)
    annot = annotate(tools, work, m)
    outputs, summary = [], []
    for trait in g["traits"]:
        res = tidy(gene_analysis(trait, tools, work, annot, m), tools / "NCBI37.3.gene.loc", m["alpha"])
        p = out_dir / f"{trait}_genes.tsv"
        res.to_csv(p, sep="\t", index=False)
        outputs.append(p)
        gw = pd.read_csv(processed_path(trait), sep="\t", usecols=["P"])
        summary.append({"trait": trait, "n_snps_input": len(gw), "n_snps_gws": int((gw["P"] < 5e-8).sum()),
                        "n_genes_tested": len(res), "bonferroni_p": m["alpha"] / len(res),
                        "n_genes_bonf": int(res["SIG_BONF"].sum()), "n_genes_fdr05": int((res["Q_FDR"] < 0.05).sum())})
        print(summary[-1], flush=True)
    sp = out_dir / "magma_summary.tsv"
    pd.DataFrame(summary).to_csv(sp, sep="\t", index=False)
    outputs.append(sp)
    inputs = [ROOT / g["gwas_dir"] / v["file"] for v in g["traits"].values()]
    inputs += [tools / "NCBI37.3.gene.loc", tools / "g1000_eur.bim"]
    write_provenance(STAGE, inputs, outputs, seed=cfg["seed"],
                     extra={"magma_version": "v1.10", "window_kb": [m["window_up_kb"], m["window_down_kb"]],
                            "gene_model": m["model"], "ld_reference": "1000 Genomes EUR (MAGMA g1000_eur)"})


if __name__ == "__main__":
    main()
