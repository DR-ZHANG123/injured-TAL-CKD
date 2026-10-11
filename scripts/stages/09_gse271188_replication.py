"""Stage 09：GSE271188（肾小球疾病活检单核：DKD、IgAN、MN、MCD、对照）独立复现。

1. 细胞注释迁移：以 KPMP 单核 v2 各 SubclassLevel2 亚群的平均表达谱（全部供体，不用疾病标签）为参照，
   每个细胞核在亚群特异基因上取 Pearson 相关最高的亚群。
2. 供体 × 亚群原始计数 pseudobulk。
3. 检验：肾小球毛细血管内皮（EC-GC）与足细胞（POD）状态内的标志得分、损伤 PT 比例，DKD vs 原发肾小球疾病 vs 对照。
输出：data/processed/gse271188/pseudobulk.parquet；results/09_gse271188_replication/{labels_summary.tsv, cell_counts.tsv,
      donors.tsv, glomerular_markers.tsv, abundance.tsv}
"""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.geo import read_series_matrix  # noqa: E402
from lib.pbstats import hedges_g, state_matrix  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

STAGE = "09_gse271188_replication"
GLOMERULAR = {"IgAN", "MN", "MCD"}


def reference(cfg: dict, n_per: int) -> tuple[pd.DataFrame, list[str]]:
    pb = pd.read_parquet(ROOT / cfg["paths"]["processed"] / "kpmp" / "snRNA_pseudobulk.parquet")
    sub = pb.index.str.split("|").str[1]
    agg = pb.groupby(sub).sum()
    agg = agg[agg.sum(1) > 0]
    logcpm = np.log1p(agg.div(agg.sum(1), axis=0) * 1e4)
    spec = logcpm - logcpm.median(0)
    genes = sorted({g for s in spec.index for g in spec.loc[s].nlargest(n_per).index})
    return logcpm[genes], genes


def read_sample(raw: Path, gsm: str) -> tuple[sp.csr_matrix, pd.Index]:
    stem = next(raw.glob(f"{gsm}_*_matrix.mtx.gz")).name.replace("_matrix.mtx.gz", "")
    m = scipy.io.mmread(raw / f"{stem}_matrix.mtx.gz").T.tocsr()          # 细胞 × 基因
    feats = pd.read_csv(raw / f"{stem}_features.tsv.gz", sep="\t", header=None)[1]
    return m, pd.Index(feats.astype(str).values)


def annotate(m: sp.csr_matrix, feats: pd.Index, ref: pd.DataFrame, genes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    first = pd.Series(np.arange(len(feats)), index=feats)[~feats.duplicated()]   # 重名基因取第一个
    gi = first.reindex(genes).fillna(-1).astype(int).values
    ok = gi >= 0
    x = m[:, gi[ok]].toarray().astype(np.float32)
    lib = np.asarray(m.sum(1)).ravel()[:, None]
    x = np.log1p(x / lib * 1e4)
    r = ref.loc[:, np.array(genes)[ok]].values
    x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)
    r = (r - r.mean(1, keepdims=True)) / (r.std(1, keepdims=True) + 1e-6)
    c = x @ r.T / x.shape[1]
    return ref.index.values[c.argmax(1)], c.max(1)


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    k = cfg["kpmp"]
    raw = ROOT / cfg["paths"]["raw"] / "GSE271188"
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    proc = ROOT / cfg["paths"]["processed"] / "gse271188"
    proc.mkdir(parents=True, exist_ok=True)

    smp, _ = read_series_matrix(raw / "GSE271188_series_matrix.txt.gz")
    chars = smp.filter(like="char_").fillna("").agg(" ".join, axis=1)
    donors = pd.DataFrame({"title": smp["title"],
                           "diagnosis": chars.str.extract(r"diagnosis: (\S+)", expand=False)})
    donors.to_csv(out / "donors.tsv", sep="\t")

    ref, genes = reference(cfg, 25)
    pbs, counts, summ = [], [], []
    for gsm in donors.index:
        m, feats = read_sample(raw, gsm)
        keep = np.asarray(m.sum(1)).ravel() >= 500
        m = m[keep]
        lab, score = annotate(m, feats, ref, genes)
        lab = np.where(score >= 0.2, lab, "unassigned")
        labs = pd.Index(sorted(set(lab)))
        ind = sp.csr_matrix((np.ones(len(lab)), (labs.get_indexer(lab), np.arange(len(lab)))), shape=(len(labs), len(lab)))
        pb = pd.DataFrame((ind @ m).toarray().astype(np.float32), index=[f"{gsm}|{l}" for l in labs], columns=feats)
        pbs.append(pb.T.groupby(level=0).sum().T)
        counts.append(pd.Series(lab).value_counts().rename(gsm))
        summ.append({"gsm": gsm, "n_nuclei": int(keep.sum()), "median_score": float(np.median(score)),
                     "unassigned": float((lab == "unassigned").mean())})
        print(gsm, donors.loc[gsm, "diagnosis"], int(keep.sum()), flush=True)
    pb = pd.concat(pbs).fillna(0)
    pb.to_parquet(proc / "pseudobulk.parquet")
    cnt = pd.DataFrame(counts).fillna(0).astype(int)
    cnt.to_csv(out / "cell_counts.tsv", sep="\t")
    pd.DataFrame(summ).to_csv(out / "labels_summary.tsv", sep="\t", index=False)

    lab = donors["diagnosis"].where(~donors["diagnosis"].isin(GLOMERULAR), "GLOM_DISEASE")
    mk = pd.read_csv(ROOT / "results" / "05_markers" / "markers.tsv", sep="\t")
    rows = []
    for state, subs, mset in [("EC_GC", ["EC-GC"], "glomerular_capillary_EC"),
                              ("EC_GC_all", ["EC-GC", "aEC-GC", "dEC-GC"], "glomerular_capillary_EC"),
                              ("POD", ["POD"], "podocyte"), ("POD_all", ["POD", "dPOD"], "podocyte")]:
        m = state_matrix(pb, cnt, subs, k["min_cells"])
        if m.empty:
            continue
        g = [x for x in mk.loc[mk.marker_set == mset, "gene"] if x in m.columns]
        score = ((m[g] - m[g].mean()) / m[g].std()).mean(1)
        l = lab.reindex(score.index)
        for a, b in [("DKD", "GLOM_DISEASE"), ("DKD", "Control"), ("GLOM_DISEASE", "Control")]:
            x, y = score[l == a], score[l == b]
            if len(x) >= 2 and len(y) >= 2:
                rows.append({"state": state, "contrast": f"{a}_vs_{b}", "n_a": len(x), "n_b": len(y),
                             "hedges_g": float(hedges_g(x.values[:, None], y.values[:, None])[0]),
                             "p_mannwhitney": float(mannwhitneyu(x, y).pvalue)})
    pd.DataFrame(rows).to_csv(out / "glomerular_markers.tsv", sep="\t", index=False)

    c = cnt.reindex(columns=sorted(set(cnt.columns) | {"aPT", "frPT", "dPT", "cycPT", "PT-S1", "PT-S2", "PT-S3"}), fill_value=0)
    pt = c[["PT-S1", "PT-S2", "PT-S3", "aPT", "frPT", "dPT", "cycPT"]].sum(1)
    fr = (c[["aPT", "frPT"]].sum(1) / pt)[pt >= k["min_cells"]]
    rows = []
    for a, b in [("DKD", "GLOM_DISEASE"), ("DKD", "Control"), ("GLOM_DISEASE", "Control")]:
        x, y = fr[lab.reindex(fr.index) == a], fr[lab.reindex(fr.index) == b]
        if len(x) >= 2 and len(y) >= 2:
            rows.append({"measure": "PT_injured_of_PT", "contrast": f"{a}_vs_{b}", "n_a": len(x), "n_b": len(y),
                         "median_a": float(x.median()), "median_b": float(y.median()),
                         "p_mannwhitney": float(mannwhitneyu(x, y).pvalue)})
    pd.DataFrame(rows).to_csv(out / "abundance.tsv", sep="\t", index=False)
    print(pd.read_csv(out / "glomerular_markers.tsv", sep="\t").round(3).to_string(index=False))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    write_provenance(STAGE, inputs=[raw / "GSE271188_series_matrix.txt.gz"],
                     outputs=[proc / "pseudobulk.parquet", out / "cell_counts.tsv", out / "glomerular_markers.tsv",
                              out / "abundance.tsv"], seed=cfg["seed"])


if __name__ == "__main__":
    main()
