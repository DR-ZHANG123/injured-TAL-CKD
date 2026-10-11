"""Stage 21：损伤 TAL 结论对取材方式、年龄、性别、纤维化与组成效应的稳健性；194 个 eGFR 相关基因的维度。

1. 取材方式：参考组织内按取材方式比较损伤 TAL / PT 比例；DKD、HKD 对仅经皮穿刺参考组织（Hedges g、Mann–Whitney）。
2. logit 比例 ~ 诊断 + 年龄（十年分档中点）+ 性别（+ 参考组织取材方式）；eGFR ~ 比例 + 协变量，
   全部 / CKD（DKD+HKD）/ DKD / HKD / REF 分别拟合。
3. 白蛋白尿（KDIGO A1–A3，有序 logit）~ 比例 + 诊断 + 年龄 + 性别。
4. 基因程序得分 vs eGFR 校正损伤 TAL 比例（及 frTAL 占损伤 TAL 比例、损伤 TAL 细胞数）；
   CKD 参与者中损伤 TAL 比例 vs eGFR 校正间质纤维化。
5. 194 个 eGFR 相关程序基因在损伤 TAL pseudobulk 中跨参与者的主成分结构（平行分析 + 随机基因集对照）。
单位均为参与者（KPMP 单核供体；取材比较另报单细胞供体）。
输出：results/21_confounders/*.tsv
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import kruskal, mannwhitneyu, spearmanr
from statsmodels.miscmodels.ordinal_model import OrderedModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.agreement import agreement_rows, lineage_fractions  # noqa: E402
from lib.pbstats import hedges_g, state_matrix  # noqa: E402
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

STAGE = "21_confounders"
RES = ROOT / "results"
F_FRAC = RES / "16_fig_tables" / "kpmp_donor_fractions.tsv"
F_PROG = RES / "16_fig_tables" / "tal_programme_score.tsv"
F_SN = RES / "03_kpmp_pseudobulk" / "snRNA_donors.tsv"
F_SC = RES / "03_kpmp_pseudobulk" / "scRNA_donors.tsv"
F_CNT = RES / "03_kpmp_pseudobulk" / "snRNA_cell_counts.tsv"
F_CLIN = ROOT / "metadata" / "kpmp_participants_clinical_bins.tsv"
F_IF = RES / "11_histology" / "clinical_bins_used.tsv"
F_CNT_SC = RES / "03_kpmp_pseudobulk" / "scRNA_cell_counts.tsv"
LINEAGES = {"TAL": "tal_injured_frac", "PT": "pt_injured_frac"}


def decade_mid(s) -> float:
    m = re.match(r"(\d+)-(\d+)", str(s))
    return (int(m.group(1)) + int(m.group(2)) + 1) / 2 if m else np.nan


def logit(v: pd.Series, eps: float) -> pd.Series:
    return np.log((v + eps) / (1 - v + eps))


# ---------- 参与者表 ----------
def participants(cfg: dict, c: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    fr = pd.read_csv(F_FRAC, sep="\t", dtype={"donor_id": str})
    sn_don = pd.read_csv(F_SN, sep="\t", dtype=str).set_index("donor_id")
    sc_don = pd.read_csv(F_SC, sep="\t", dtype=str).set_index("donor_id")
    clin = pd.read_csv(F_CLIN, sep="\t", dtype=str).set_index("redcap_id")
    if_pct = pd.read_csv(F_IF, sep="\t", dtype={"participant": str}).set_index("participant").IF_pct
    prog = pd.read_csv(F_PROG, sep="\t", dtype={"donor_id": str}).set_index("donor_id").programme_score
    cnt = pd.read_csv(F_CNT, sep="\t", index_col=0, dtype={0: str})
    cnt.index = cnt.index.astype(str)
    cnt = cnt.reindex(columns=sorted(set(cnt.columns) | {"aTAL", "frTAL"}), fill_value=0)
    alb_map = cfg["histology"]["clinical"]["albuminuria_A"]

    sn = fr[fr.modality == "snRNA"].set_index("donor_id")
    sn = sn[sn.dx.isin(c["dx_order"])].copy()
    sn["age10"] = sn_don.Age.reindex(sn.index).map(decade_mid) / 10
    sn["male"] = (sn_don.sex.reindex(sn.index) == "male").astype(float)
    sn["proc"] = sn_don.TissueCollection.reindex(sn.index)
    sn["alb_A"] = clin.albuminuria.reindex(sn.index).map(alb_map)
    sn["IF_pct"] = if_pct.reindex(sn.index)
    sn["programme_score"] = prog.reindex(sn.index)
    n_inj = cnt[["aTAL", "frTAL"]].sum(1).reindex(sn.index)
    sn["frTAL_share"] = (cnt.frTAL.reindex(sn.index) / n_inj).where(n_inj >= c["min_group_n"])
    sn["log_n_injured_tal"] = np.log10(n_inj.clip(lower=1))
    sn["egfr"] = sn.egfr_mid

    sc = fr[fr.modality == "scRNA"].set_index("donor_id")
    sc = sc[sc.dx.isin(c["dx_order"])].copy()
    sc["proc"] = sc_don["Tissue.Type"].reindex(sc.index)
    sc["age10"] = sc_don.Age.reindex(sc.index).map(decade_mid) / 10
    for d in (sn, sc):
        for lin, col in LINEAGES.items():
            d[f"{lin}_logit"] = logit(d[col], c["logit_eps"])
        d["isDKD"], d["isHKD"] = (d.dx == "DKD").astype(float), (d.dx == "HKD").astype(float)
    procs = sorted(p for p in sn.proc.dropna().unique() if p != c["reference_procurement"])
    for p in procs:   # 参考组织取材方式哑变量；DKD/HKD 全为经皮穿刺 = 基线
        sn[f"proc_{p.replace(' ', '_').replace('-', '_')}"] = ((sn.proc == p) & (sn.dx == "REF")).astype(float)
    return sn, sc


def proc_cols(d: pd.DataFrame) -> list[str]:
    return [x for x in d.columns if x.startswith("proc_")]


# ---------- 1. 取材方式 ----------
def procurement(sn: pd.DataFrame, sc: pd.DataFrame, c: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups, contrasts = [], []
    perc = c["reference_procurement"]
    for mod, d in (("snRNA", sn), ("scRNA", sc)):
        d = d.assign(group=np.where(d.dx == "REF", "REF:" + d.proc.astype(str), d.dx))
        for lin, col in LINEAGES.items():
            dd = d.dropna(subset=[col])
            for g, s in dd.groupby("group"):
                groups.append({"modality": mod, "lineage": lin, "group": g, "n": len(s),
                               "median_frac": s[col].median(), "q25": s[col].quantile(.25),
                               "q75": s[col].quantile(.75), "median_age": s.age10.median() * 10})
            ref = dd[dd.dx == "REF"]
            ref_groups = [s for _, s in ref.groupby("proc") if len(s) >= c["min_group_n"]]
            if len(ref_groups) >= 2:
                kw = kruskal(*[s[col] for s in ref_groups])
                contrasts.append({"modality": mod, "lineage": lin, "contrast": "REF procurement types (Kruskal-Wallis)",
                                  "n_a": len(ref), "p": kw.pvalue, "statistic": kw.statistic})
            base_name = f"REF:{perc}" if (ref.proc == perc).sum() >= c["min_group_n"] else None
            pairs = []
            if base_name:
                pairs += [(f"REF:{p}", base_name) for p in sorted(ref.proc.unique()) if p != perc]
                pairs += [(dx, base_name) for dx in ("DKD", "HKD")]
            else:
                pairs += [(dx, f"REF:{p}") for dx in ("DKD", "HKD") for p in sorted(ref.proc.unique())]
            pairs += [(dx, "REF:all") for dx in ("DKD", "HKD")]
            for a, b in pairs:
                x = dd[dd.group == a] if a != "REF:all" else ref
                y = dd[dd.group == b] if b != "REF:all" else ref
                if len(x) < c["min_group_n"] or len(y) < c["min_group_n"]:
                    continue
                g = hedges_g(x[f"{lin}_logit"].values[:, None], y[f"{lin}_logit"].values[:, None])[0]
                contrasts.append({"modality": mod, "lineage": lin, "contrast": f"{a} vs {b}", "n_a": len(x),
                                  "n_b": len(y), "median_a": x[col].median(), "median_b": y[col].median(),
                                  "hedges_g_logit": float(g), "p": mannwhitneyu(x[col], y[col]).pvalue,
                                  "statistic": np.nan})
    return pd.DataFrame(groups), pd.DataFrame(contrasts)


# ---------- 回归工具 ----------
def _clean(d: pd.DataFrame, cols: list[str]) -> list[str]:
    return [x for x in cols if x in d.columns and d[x].std() > 0]


def ols_rows(d: pd.DataFrame, y: str, expo: list[str], covs: list[str], report: list[str], meta: dict,
             min_n: int) -> list[dict]:
    d = d.dropna(subset=[y] + expo + [x for x in covs if x in d.columns])
    covs = _clean(d, covs)
    if len(d) < min_n:
        return []
    X = sm.add_constant(d[expo + covs].astype(float), has_constant="add")
    f, fr = sm.OLS(d[y], X).fit(), sm.OLS(d[y], X).fit(cov_type="HC3")
    rows = []
    for t in report:
        if t not in X.columns and "-" not in t:
            continue
        if "-" in t:                                    # 线性对比，如 isDKD-isHKD
            a, b = t.split("-")
            if a not in X.columns or b not in X.columns:
                continue
            r = np.zeros(X.shape[1])
            r[list(X.columns).index(a)], r[list(X.columns).index(b)] = 1, -1
            tt, ttr = f.t_test(r), fr.t_test(r)
            est, (lo, hi), p, ph = float(tt.effect[0]), tt.conf_int()[0], float(tt.pvalue), float(ttr.pvalue)
        else:
            est, (lo, hi), p, ph = f.params[t], f.conf_int().loc[t], f.pvalues[t], fr.pvalues[t]
        rows.append({**meta, "term": t, "n": len(d), "covariates": "+".join(covs) or "none", "estimate": est,
                     "ci_low": lo, "ci_high": hi, "p": p, "p_hc3": ph})
    return rows


def ordinal_rows(d: pd.DataFrame, expo: str, covs: list[str], meta: dict, min_n: int) -> list[dict]:
    d = d.dropna(subset=["alb_A", expo] + [x for x in covs if x in d.columns])
    covs = _clean(d, covs)
    if len(d) < min_n or d.alb_A.nunique() < 2:
        return []
    f = OrderedModel(d.alb_A.astype(int), d[[expo] + covs].astype(float), distr="logit").fit(
        method="bfgs", disp=False, maxiter=5000)
    lo, hi = f.conf_int().loc[expo]
    return [{**meta, "term": expo, "n": len(d), "n_A1": int((d.alb_A == 1).sum()), "n_A2": int((d.alb_A == 2).sum()),
             "n_A3": int((d.alb_A == 3).sum()), "covariates": "+".join(covs) or "none",
             "odds_ratio": np.exp(f.params[expo]), "ci_low": np.exp(lo), "ci_high": np.exp(hi), "p": f.pvalues[expo]}]


def dx_covs(sub: list[str]) -> list[str]:
    return [x for x, dx in (("isDKD", "DKD"), ("isHKD", "HKD")) if dx in sub and "REF" in sub] + \
        (["isDKD"] if sub == ["DKD", "HKD"] else [])


# ---------- 2. 诊断与 eGFR ----------
def dx_models(sn: pd.DataFrame, c: dict) -> pd.DataFrame:
    rows = []
    pc = proc_cols(sn)
    for lin in LINEAGES:
        for sample, d in (("all_procurement", sn), ("percutaneous_only", sn[sn.proc == c["reference_procurement"]])):
            for adj, covs in (("none", []), ("age+sex", ["age10", "male"]),
                              ("age+sex+procurement", ["age10", "male"] + pc)):
                if sample == "percutaneous_only" and "procurement" in adj:
                    continue
                rows += ols_rows(d, f"{lin}_logit", ["isDKD", "isHKD"], covs,
                                 ["isDKD", "isHKD", "isDKD-isHKD", "age10", "male"],
                                 {"lineage": lin, "sample": sample, "adjustment": adj}, c["min_model_n"])
    out = pd.DataFrame(rows)
    out["term"] = out.term.map({"isDKD": "DKD_vs_REF", "isHKD": "HKD_vs_REF", "isDKD-isHKD": "DKD_vs_HKD",
                                "age10": "age_per_decade", "male": "male_vs_female"})
    out["odds_ratio"], out["or_ci_low"], out["or_ci_high"] = np.exp(out.estimate), np.exp(out.ci_low), np.exp(out.ci_high)
    return out


def zscore_on(d: pd.DataFrame, col: str, ref: pd.DataFrame) -> tuple[pd.Series, float]:
    s = ref[col].dropna()
    return (d[col] - s.mean()) / s.std(), float(s.std())


def egfr_models(sn: pd.DataFrame, c: dict) -> pd.DataFrame:
    rows = []
    pc = proc_cols(sn)
    for lin in LINEAGES:
        z, sd = zscore_on(sn, f"{lin}_logit", sn)
        d0 = sn.assign(expo=z)
        for sname, sub in c["egfr_subsets"].items():
            d = d0[d0.dx.isin(sub)]
            dxc = dx_covs(sub)
            px = "dx+" if dxc else ""
            sets = [("none", [])] + ([("dx", dxc)] if dxc else []) + [(f"{px}age+sex", dxc + ["age10", "male"])]
            if "REF" in sub:
                sets.append((f"{px}age+sex+procurement", dxc + ["age10", "male"] + pc))
            for adj, covs in sets:
                rows += ols_rows(d, "egfr", ["expo"], covs, ["expo"],
                                 {"exposure": f"{lin}_injured_logit_perSD", "sd_logit": sd, "subset": sname,
                                  "adjustment": adj}, c["min_model_n"])
            s = d.dropna(subset=["egfr", "expo"])
            if len(s) >= c["min_model_n"]:
                rho, p = spearmanr(s.expo, s.egfr)
                rows.append({"exposure": f"{lin}_injured_logit_perSD", "subset": sname, "adjustment": "spearman",
                             "term": "expo", "n": len(s), "estimate": rho, "p": p})
    out = pd.DataFrame(rows)
    out["estimate_per_logit_unit"] = out.estimate / out.sd_logit
    return out


# ---------- 2b. 全部参与者、完整诊断类别（敏感性） ----------
def full_dx_models(cfg: dict, c: dict, x: dict) -> pd.DataFrame:
    """主分析人群 = REF+DKD+HKD；此处纳入全部单核参与者，诊断按 AdjudicationCategory 各自成一水平（REF 为基线）。
    另按 stage 11 的三分类（DKD / HKD / 其余全部）重拟合，以说明两种人群定义的差别。"""
    fr = pd.read_csv(F_FRAC, sep="\t", dtype={"donor_id": str})
    don = pd.read_csv(F_SN, sep="\t", dtype=str).set_index("donor_id")
    clin = pd.read_csv(F_CLIN, sep="\t", dtype=str).set_index("redcap_id")
    d = fr[fr.modality == "snRNA"].set_index("donor_id")
    a, cc = don.AdjudicationCategory.reindex(d.index), don.ConditionCategory.reindex(d.index)
    full = a.map(x["full_dx"])
    und = x["undetermined_prefix"] + cc.map(x["condition_short"]).fillna("other")
    full = full.fillna(und)
    full[cc.isin(["Healthy Reference", "Reference"])] = "REF"
    vc = full.value_counts()
    full = full.where(full.map(vc) >= c["min_group_n"], "other_diagnosis")
    d = d.assign(full_dx=full, age10=don.Age.reindex(d.index).map(decade_mid) / 10,
                 male=(don.sex.reindex(d.index) == "male").astype(float), egfr=d.egfr_mid,
                 alb_A=clin.albuminuria.reindex(d.index).map(cfg["histology"]["clinical"]["albuminuria_A"]))
    d["tal"] = logit(d.tal_injured_frac, c["logit_eps"])
    main_set = d[d.full_dx.isin(c["dx_order"])]
    d["tal"] = (d.tal - main_set.tal.mean()) / main_set.tal.std()     # 与主分析相同的 SD 尺度
    levels = [l for l in sorted(d.full_dx.unique()) if l != "REF"]
    for l in levels:
        d[f"dx_{l}"] = (d.full_dx == l).astype(float)
    d["s11_DKD"], d["s11_HKD"] = (d.full_dx == "DKD").astype(float), (d.full_dx == "HKD").astype(float)
    full_c = [f"dx_{l}" for l in levels]
    rows = []
    for pop, dd, covs in (("REF+DKD+HKD (main)", d[d.full_dx.isin(c["dx_order"])], ["dx_DKD", "dx_HKD"]),
                          ("all participants, full diagnosis categories", d, full_c),
                          ("all participants, DKD/HKD/all-other (stage 11 coding)", d, ["s11_DKD", "s11_HKD"])):
        for adj, extra in (("dx", []), ("dx+age+sex", ["age10", "male"])):
            rows += [{**r, "outcome": "eGFR"} for r in ols_rows(dd, "egfr", ["tal"], covs + extra, ["tal"],
                     {"population": pop, "adjustment": adj}, c["min_model_n"])]
            for r in ordinal_rows(dd, "tal", covs + extra, {"population": pop, "adjustment": adj}, c["min_model_n"]):
                rows.append({**r, "outcome": "albuminuria_A1-A3", "estimate": r.pop("odds_ratio")})
    out = pd.DataFrame(rows)
    out["term"] = "injured_TAL_logit_perSD"
    comp = d.groupby("full_dx").agg(n=("tal", "size"), n_with_tal=("tal", "count"),
                                    n_with_egfr=("egfr", "count"), n_with_alb=("alb_A", "count"),
                                    median_tal_frac=("tal_injured_frac", "median")).reset_index()
    return pd.concat([out, comp.assign(population="composition")], ignore_index=True)


# ---------- 2c. 单核 vs 单细胞一致性：跨谱系正式比较 ----------
def sn_sc_agreement(cfg: dict, x: dict, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    mc = cfg["kpmp"]["min_cells"]
    fr = {}
    for mod, f in (("sn", F_CNT), ("sc", F_CNT_SC)):
        cnt = pd.read_csv(f, sep="\t", index_col=0, dtype={0: str})
        cnt.index = cnt.index.astype(str)
        fr[mod] = lineage_fractions(cnt, x["agreement_lineages"], mc)
    k16 = pd.read_csv(F_FRAC, sep="\t", dtype={"donor_id": str})
    lab = k16[k16.modality == "snRNA"].set_index("donor_id").dx
    for mod, m16 in (("sn", "snRNA"), ("sc", "scRNA")):     # 与 stage 16 的 TAL / PT 比例逐一核对
        ref16 = k16[k16.modality == m16].set_index("donor_id")
        for lin, col in LINEAGES.items():
            a, b = fr[mod][lin].dropna(), ref16[col].reindex(fr[mod][lin].dropna().index)
            assert np.allclose(a, b, equal_nan=False), f"{mod} {lin} fraction differs from stage 16"
    return agreement_rows(fr["sn"], fr["sc"], lab, list(x["agreement_lineages"]),
                          x["agreement_reference_lineage"], c_eps(cfg), x["n_boot"], rng)


def c_eps(cfg: dict) -> float:
    return cfg["confounders"]["logit_eps"]


# ---------- 3. 白蛋白尿 ----------
def albuminuria_models(sn: pd.DataFrame, c: dict) -> pd.DataFrame:
    rows = []
    for lin in LINEAGES:
        z, _ = zscore_on(sn, f"{lin}_logit", sn)
        d0 = sn.assign(expo=z)
        for sname, sub in c["albuminuria_subsets"].items():
            d, dxc = d0[d0.dx.isin(sub)], dx_covs(sub)
            px = "dx+" if dxc else ""
            sets = [("none", [])] + ([("dx", dxc)] if dxc else []) + [(f"{px}age+sex", dxc + ["age10", "male"]),
                                                                    (f"{px}age+sex+eGFR", dxc + ["age10", "male", "egfr"])]
            for adj, covs in sets:
                rows += ordinal_rows(d, "expo", covs, {"exposure": f"{lin}_injured_logit_perSD", "subset": sname,
                                                       "adjustment": adj}, c["min_model_n"])
    return pd.DataFrame(rows)


# ---------- 4. 组成效应与纤维化 ----------
def _lab(adj: str, dxc: list[str]) -> str:
    """单一诊断子集内没有诊断协变量：标签去掉 dx。"""
    return adj if dxc else (adj.replace("dx+", "") if adj != "dx" else "none")


def programme_and_fibrosis(sn: pd.DataFrame, c: dict) -> pd.DataFrame:
    rows = []
    zp, _ = zscore_on(sn, "programme_score", sn)
    zt, _ = zscore_on(sn, "TAL_logit", sn)
    d0 = sn.assign(prog=zp, tal=zt, if10=sn.IF_pct / 10)
    comp = ["tal", "frTAL_share", "log_n_injured_tal"]
    for sname in ("all", "CKD", "DKD", "HKD"):
        sub = c["egfr_subsets"][sname]
        d, dxc = d0[d0.dx.isin(sub)], dx_covs(sub)
        cc = d.dropna(subset=["egfr", "prog"] + comp)       # 同一批参与者上比较各层校正
        for adj, covs in (("dx", dxc), ("dx+age+sex", dxc + ["age10", "male"]),
                          ("dx+injured_TAL_fraction", dxc + ["tal"]),
                          ("dx+age+sex+injured_TAL_fraction", dxc + ["age10", "male", "tal"]),
                          ("dx+age+sex+injured_TAL_fraction+frTAL_share+n_injured_cells",
                           dxc + ["age10", "male"] + comp)):
            rows += ols_rows(cc, "egfr", ["prog"], covs, ["prog", "tal"],
                             {"analysis": "programme_score_vs_eGFR", "subset": sname, "adjustment": _lab(adj, dxc)},
                             c["min_model_n"])
        cf = d.dropna(subset=["egfr", "tal", "if10"])
        for adj, covs in (("dx", dxc), ("dx+IF", dxc + ["if10"]), ("dx+age+sex", dxc + ["age10", "male"]),
                          ("dx+age+sex+IF", dxc + ["age10", "male", "if10"])):
            rows += ols_rows(cf, "egfr", ["tal"], covs, ["tal", "if10"],
                             {"analysis": "injured_TAL_fraction_vs_eGFR_with_fibrosis", "subset": sname,
                              "adjustment": _lab(adj, dxc)}, c["min_model_n"])
        if len(cf) >= c["min_model_n"]:
            rho, p = spearmanr(cf.tal, cf.if10)
            rows.append({"analysis": "injured_TAL_fraction_vs_IF_spearman", "subset": sname, "term": "tal~IF",
                         "n": len(cf), "estimate": rho, "p": p})
    out = pd.DataFrame(rows)
    out["term"] = out.term.replace({"prog": "programme_score_perSD", "tal": "injured_TAL_logit_perSD",
                                    "if10": "IF_per10pct"})
    return out


# ---------- 5. 194 基因的维度 ----------
def pc_shares(X: np.ndarray, k: int) -> np.ndarray:
    Z = (X - X.mean(0)) / X.std(0, ddof=1)
    s = np.linalg.svd(Z, compute_uv=False) ** 2
    return (s / s.sum())[:k]


def dimensionality(cfg: dict, c: dict, sn: pd.DataFrame, rng: np.random.Generator):
    kg = pd.read_csv(RES / "15_key_genes" / f"{c['pca_programme_state']}_key_genes.tsv", sep="\t", index_col=0)
    up = kg[kg.replicated_both & (kg.log2fc > 0)]
    genes194 = up.index[up.egfr_fdr < c["pca_fdr"]]
    spec = cfg["key_genes"]["states"][c["pca_programme_state"]]
    pb = pd.read_parquet(ROOT / cfg["paths"]["processed"] / "kpmp" / "snRNA_pseudobulk.parquet")
    cnt = pd.read_csv(F_CNT, sep="\t", index_col=0, dtype={0: str})
    cnt.index = cnt.index.astype(str)
    m = state_matrix(pb, cnt, spec["injured"], cfg["kpmp"]["min_cells"])
    m.index = m.index.astype(str)
    k = c["pca_n_components"]
    var_rows, summ, scores_all, load_all = [], [], [], []
    sets = {"all_participants": m.index, "REF_DKD_HKD": m.index.intersection(sn.index),
            "CKD": m.index.intersection(sn.index[sn.dx.isin(["DKD", "HKD"])])}
    for gname, genes in (("eGFR_associated_194", genes194), ("programme_all", up.index)):
        g = [x for x in genes if x in m.columns]
        for sname, ids in sets.items():
            X = m.loc[ids, g].values
            X = X[:, X.std(0, ddof=1) > 0]
            obs = pc_shares(X, k)
            null = np.array([pc_shares(np.column_stack([rng.permutation(col) for col in X.T]), k)
                             for _ in range(c["pca_n_perm"])])
            q95 = np.quantile(null, 0.95, axis=0)
            n_axes = int(np.argmax(~(obs > q95))) if not (obs > q95).all() else k
            for i in range(k):
                var_rows.append({"gene_set": gname, "participants": sname, "component": i + 1, "n": len(ids),
                                 "n_genes": X.shape[1], "variance_share": obs[i], "perm_null_q95": q95[i],
                                 "exceeds_null": bool(obs[i] > q95[i])})
            rec = {"gene_set": gname, "participants": sname, "n": len(ids), "n_genes": X.shape[1],
                   "pc1_share": obs[0], "pc2_share": obs[1], "n_components_above_null": n_axes}
            if gname == "eGFR_associated_194":
                pool = m.columns[(m.loc[ids].mean() >= 1) & ~m.columns.isin(up.index)]
                rnd = np.array([pc_shares(m.loc[ids, rng.choice(pool, X.shape[1], replace=False)].values, 1)[0]
                                for _ in range(c["pca_n_random_sets"])])
                rec.update({"random_set_pc1_median": np.median(rnd), "random_set_pc1_q95": np.quantile(rnd, .95),
                            "pc1_percentile_vs_random": float((rnd < obs[0]).mean())})
                Z = (X - X.mean(0)) / X.std(0, ddof=1)
                U, S, Vt = np.linalg.svd(Z, full_matrices=False)
                sign = np.sign(Vt[0].sum()) or 1.0
                pcs = pd.DataFrame(U[:, :2] * S[:2] * np.array([sign, 1]), index=ids, columns=["PC1", "PC2"])
                rec["frac_genes_same_sign_pc1"] = float(((Vt[0] * sign) > 0).mean())
                aux = sn.reindex(ids)[["dx", "egfr", "tal_injured_frac", "programme_score", "frTAL_share",
                                       "log_n_injured_tal"]]
                for v in ["egfr", "tal_injured_frac", "programme_score", "frTAL_share", "log_n_injured_tal"]:
                    for pcn in ("PC1", "PC2"):
                        t = pd.concat([pcs[pcn], aux[v]], axis=1).dropna()
                        if len(t) >= c["min_model_n"]:
                            rho, p = spearmanr(t.iloc[:, 0], t.iloc[:, 1])
                            rec[f"{pcn}_rho_{v}"], rec[f"{pcn}_p_{v}"] = rho, p
                scores_all.append(pcs.join(aux).assign(participants=sname).rename_axis("donor_id").reset_index())
                g_used = np.array(g)[m.loc[ids, g].std(0, ddof=1).values > 0]
                load_all.append(pd.DataFrame({"gene": g_used, "PC1_loading": Vt[0] * sign, "PC2_loading": Vt[1],
                                              "participants": sname}))
            summ.append(rec)
    return (pd.DataFrame(var_rows), pd.DataFrame(summ), pd.concat(scores_all, ignore_index=True),
            pd.concat(load_all, ignore_index=True))


def main() -> None:
    cfg = load_config()
    c = cfg["confounders"]
    set_global_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    out = RES / STAGE
    out.mkdir(parents=True, exist_ok=True)
    sn, sc = participants(cfg, c)
    tabs = {}
    tabs["participants_snRNA.tsv"] = sn.drop(columns=proc_cols(sn)).rename_axis("donor_id").reset_index()
    tabs["procurement_groups.tsv"], tabs["procurement_contrasts.tsv"] = procurement(sn, sc, c)
    tabs["dx_models.tsv"] = dx_models(sn, c)
    tabs["egfr_models.tsv"] = egfr_models(sn, c)
    x = cfg["confounders_extra"]
    tabs["full_dx_sensitivity.tsv"] = full_dx_models(cfg, c, x)
    tabs["sn_sc_agreement_lineages.tsv"], tabs["sn_sc_rho_difference.tsv"] = sn_sc_agreement(cfg, x, rng)
    tabs["albuminuria_models.tsv"] = albuminuria_models(sn, c)
    tabs["programme_fibrosis_models.tsv"] = programme_and_fibrosis(sn, c)
    (tabs["pca_variance.tsv"], tabs["pca_summary.tsv"], tabs["pca_scores.tsv"],
     tabs["pca_loadings.tsv"]) = dimensionality(cfg, c, sn, rng)
    paths = []
    for name, df in tabs.items():
        p = out / name
        df.to_csv(p, sep="\t", index=False)
        paths.append(p)
    pd.set_option("display.width", 250)
    for n in ["procurement_contrasts.tsv", "dx_models.tsv", "egfr_models.tsv", "albuminuria_models.tsv",
              "programme_fibrosis_models.tsv", "pca_summary.tsv", "full_dx_sensitivity.tsv",
              "sn_sc_agreement_lineages.tsv", "sn_sc_rho_difference.tsv"]:
        print(f"== {n}\n", tabs[n].round(4).to_string(index=False), flush=True)
    kg = RES / "15_key_genes" / f"{c['pca_programme_state']}_key_genes.tsv"
    write_provenance(STAGE, inputs=[F_FRAC, F_PROG, F_SN, F_SC, F_CNT, F_CNT_SC, F_CLIN, F_IF, kg],
                     outputs=paths, seed=cfg["seed"], extra={"params": c, "params_extra": x})


if __name__ == "__main__":
    main()
