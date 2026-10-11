"""Stage 19c：注释迁移是否机械地制造了损伤 TAL 基因程序的复现——基因对半拆分检验与无关状态对照。

对每个复现队列（GSE211785 TAL 簇 + TAL 状态参照；GSE271188 与 GSE195460 全亚群参照，与 stage 09 / 19b 相同）：
- 完整注释（全部基因）一次；再把基因随机对半拆分 n_splits 次（固定种子），只用 A 半的基因选参照特异基因并注释细胞，
  只在 B 半的基因上评价复现（log2FC Pearson r、发现命中中同向且 |log2FC| ≥ rep_lfc 的比例）。
- 同一流程用于两对健康状态（M-TAL vs C-TAL；CNT vs DCT），其发现效应同样来自 KPMP 单核供体内配对比较（stage 15 方法），
  给出“只靠注释迁移就能得到的复现水平”作为参照。
GSE211785 同时有单核与单细胞文库的供体，先在供体内平均两种技术的配对差，一个供体一个观测。
输出：results/19c_label_transfer_circularity/{runs.tsv, summary.tsv, discovery_<pair>.tsv}
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402
from lib.tal_transfer import annotate, paired_programme, programme_concordance, pseudobulk, reference  # noqa: E402

STAGE = "19c_label_transfer_circularity"
HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"s{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gse211785(cfg, rc):
    s19a = _load("19a_gse211785_relabel")
    import anndata as ad
    g = rc["gse211785"]
    h5 = ROOT / cfg["paths"]["raw"] / "gse211785" / "susztak_preSCVI.h5ad"
    a = ad.read_h5ad(h5, backed="r")
    obs = a.obs[["sample", "tech", "Cluster_Idents"]].copy()
    feats = pd.Index(a.var_names.astype(str))
    a.file.close()
    lib = obs.index.str.split("_").str[0]
    keep = (obs.tech.isin(g["techs"]).values & ~lib.isin(g["exclude_libs"]) & obs.Cluster_Idents.isin(g["tal_clusters"]).values)
    rows = np.where(keep)[0]
    m = s19a.load_counts(h5, rows)
    obs = obs.iloc[rows]
    ok = np.asarray(m.sum(1)).ravel() >= rc["min_umi"]
    m, obs = m[ok], obs[ok]
    unit = (obs["sample"].astype(str) + "|" + obs["tech"].astype(str)).values
    donor = pd.Series(obs["sample"].astype(str).values, index=unit).groupby(level=0).first()
    return {"m": m, "feats": feats, "unit": unit, "tech": obs.tech.astype(str).values, "donor_of_unit": donor,
            "mode": "tal", "pairs": ["programme", "M-TAL_vs_C-TAL"]}


def gse271188(cfg, rc):
    s09 = _load("09_gse271188_replication")
    raw = ROOT / cfg["paths"]["raw"] / "GSE271188"
    donors = pd.read_csv(ROOT / "results" / "09_gse271188_replication" / "donors.tsv", sep="\t", index_col=0)
    ms, units, feats0 = [], [], None
    for gsm in donors.index:
        m, feats = s09.read_sample(raw, gsm)
        m = m[np.asarray(m.sum(1)).ravel() >= rc["min_umi"]]
        if feats0 is None:
            feats0 = feats
        if not feats.equals(feats0):
            raise ValueError(f"{gsm}: feature order differs")
        ms.append(m)
        units += [gsm] * m.shape[0]
    import scipy.sparse as sp
    return {"m": sp.vstack(ms).tocsr(), "feats": feats0, "unit": np.array(units), "tech": None, "donor_of_unit": None,
            "mode": "all", "pairs": ["programme", "M-TAL_vs_C-TAL", "CNT_vs_DCT"]}


def gse195460(cfg, rc):
    s19b = _load("19b_gse195460")
    import scipy.sparse as sp
    raw = ROOT / rc["gse195460"]["raw_dir"]
    mats, units = [], []
    for lib in rc["gse195460"]["libraries"]:
        m, feats, _ = s19b.read_10x_h5(next(raw.glob(f"*_{lib}_filtered_feature_bc_matrix.h5")))
        m = m[np.asarray(m.sum(1)).ravel() >= rc["min_umi"]]
        mats.append((m, feats))
        units += [lib] * m.shape[0]
    feats0 = pd.Index(sorted(set.intersection(*[set(f) for _, f in mats])))
    m = sp.vstack([x[:, pd.Series(np.arange(len(f)), index=f)[~f.duplicated()].reindex(feats0).values]
                   for x, f in mats]).tocsr()
    return {"m": m, "feats": feats0, "unit": np.array(units), "tech": None, "donor_of_unit": None,
            "mode": "all", "pairs": ["programme", "M-TAL_vs_C-TAL", "CNT_vs_DCT"]}


def label(ds, pbs, rc, genes_a):
    """genes_a = None：全部基因；否则只在 A 半基因中选参照特异基因。"""
    sel = lambda pb: pb if genes_a is None else pb[pb.columns.intersection(genes_a)]
    if ds["mode"] == "tal":
        ts = rc["tal_states"]
        lab = np.empty(ds["m"].shape[0], dtype=object)
        score = np.zeros(len(lab))
        used = set()
        for t, mod in rc["gse211785"]["reference_by_tech"].items():
            ref, genes = reference(sel(pbs[mod]), ts["n_per"], ts[mod], ts["drop_gene_regex"])
            i = np.where(ds["tech"] == t)[0]
            lab[i], score[i] = annotate(ds["m"][i], ds["feats"], ref, genes)
            used |= set(genes)
        return np.where(score >= ts["min_r"], lab, "unassigned"), used
    ref, genes = reference(sel(pbs["snRNA"]), rc["all_states"]["n_per"])
    lab, score = annotate(ds["m"], ds["feats"], ref, genes)
    return np.where(score >= rc["all_states"]["min_r"], lab, "unassigned"), set(genes)


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    rc, cc = cfg["replication2"], cfg["replication2_circularity"]
    pr = rc["programme"]
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    proc = ROOT / cfg["paths"]["processed"] / "kpmp"
    pbs = {mod: pd.read_parquet(proc / f"{mod}_pseudobulk.parquet") for mod in ("snRNA", "scRNA")}

    s15 = _load("15_key_genes")
    res3 = ROOT / "results" / "03_kpmp_pseudobulk"
    sn_cnt = pd.read_csv(res3 / "snRNA_cell_counts.tsv", sep="\t", index_col=0)
    sn_cnt.index = sn_cnt.index.astype(str)
    pairs = {"programme": (rc["tal_injured"], rc["tal_normal"])}
    disc = {"programme": pd.read_csv(ROOT / "results" / "15_key_genes" / f"{pr['state']}_discovery.tsv", sep="\t", index_col=0)}
    for name, (x, y) in cc["baseline_pairs"].items():
        pairs[name] = (x, y)
        disc[name] = s15.paired(pbs["snRNA"], sn_cnt, x, y, cfg["kpmp"]["min_cells"])
        disc[name].to_csv(out / f"discovery_{name}.tsv", sep="\t")
        print(name, "KPMP discovery hits", int(((disc[name].fdr < pr["fdr"]) & (disc[name].log2fc.abs() >= pr["min_lfc"])).sum()),
              flush=True)

    universe = pbs["snRNA"].columns
    rows = []
    for dname, loader in [("GSE211785", gse211785), ("GSE271188", gse271188), ("GSE195460", gse195460)]:
        ds = loader(cfg, rc)
        print(dname, ds["m"].shape, flush=True)
        for run in ["full"] + list(range(cc["n_splits"])):
            if run == "full":
                genes_a, excl_eval = None, None
            else:
                rng = np.random.default_rng(cfg["seed"] + run)
                perm = rng.permutation(universe)
                genes_a = pd.Index(perm[: len(perm) // 2])
                excl_eval = set(genes_a)
            lab, used = label(ds, pbs, rc, genes_a)
            pb = pseudobulk(ds["m"], ds["feats"], ds["unit"], lab)
            cnt = pd.crosstab(pd.Series(ds["unit"], name="unit"), pd.Series(lab, name="label"))
            for pname in ds["pairs"]:
                x, y = pairs[pname]
                rep = paired_programme(pb, cnt, x, y, rc["min_state_cells"], donor_of_unit=ds["donor_of_unit"])
                if rep.attrs["n_donors"] < 3:
                    continue
                r = programme_concordance(disc[pname], rep, pr["fdr"], pr["min_lfc"], pr["rep_lfc"], excl_eval)
                rows.append({"dataset": dname, "pair": pname, "run": run, "n_label_genes": len(used),
                             "n_donors": rep.attrs["n_donors"], **r})
            print(dname, run, [(r_["pair"], round(r_["lfc_pearson_r"], 3), round(r_["rep_rate"], 3))
                               for r_ in rows if r_["dataset"] == dname and r_["run"] == run], flush=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(out / "runs.tsv", sep="\t", index=False)
    split = runs[runs.run != "full"]
    summ = split.groupby(["dataset", "pair"]).agg(
        n_splits=("run", "size"), n_donors=("n_donors", "median"), n_hits_B=("n_hits", "median"),
        r_mean=("lfc_pearson_r", "mean"), r_min=("lfc_pearson_r", "min"), r_max=("lfc_pearson_r", "max"),
        rep_rate_mean=("rep_rate", "mean"), rep_rate_min=("rep_rate", "min"), rep_rate_max=("rep_rate", "max"),
        rep_rate_expressed_mean=("rep_rate_expressed", "mean"), rep_rate_up_mean=("rep_rate_up", "mean"),
        rep_rate_down_mean=("rep_rate_down", "mean"))
    full = runs[runs.run == "full"].set_index(["dataset", "pair"])[["n_hits", "lfc_pearson_r", "rep_rate"]].add_prefix("full_")
    summ = summ.join(full).reset_index()
    summ.to_csv(out / "summary.tsv", sep="\t", index=False)
    print(summ.round(3).to_string(index=False), flush=True)
    write_provenance(STAGE, inputs=[ROOT / "results" / "15_key_genes" / f"{pr['state']}_discovery.tsv"],
                     outputs=[out / "runs.tsv", out / "summary.tsv"] + [out / f"discovery_{n}.tsv" for n in cc["baseline_pairs"]],
                     seed=cfg["seed"], extra={"split_seeds": f"seed + 0..{cc['n_splits'] - 1}"})


if __name__ == "__main__":
    main()
