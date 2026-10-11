"""Stage 11b：两条分子损伤轴在 KPMP 全切片上的形态学对应，及其与白蛋白尿 / eGFR 的关系。

1. 供体级分子轴（KPMP 单核图谱）：小管修复失败轴 = 损伤 PT（aPT+frPT）占 PT 与损伤 TAL（aTAL+frTAL）
   占 TAL 两个 logit 比例的 z 平均；肾小球毛细血管内皮轴 = EC-GC pseudobulk 内 EC-GC 标志得分（越低越偏离健康）。
2. 参与者对应：图谱 donor_id ↔ 资料库 redcap_id（切片文件名与临床分档）。
3. 有图谱且有切片的参与者：Phikon-v2 切片平均嵌入 → 岭回归（内层 CV 选正则化），参与者水平重复 10 折，
   置换 y 重复整个流程得 P；再给出校正诊断 / 间质纤维化后的偏相关。
4. 无任何图谱数据的参与者（真正样本外）：用全部训练参与者拟合的模型预测两轴，与白蛋白尿（KDIGO A1–A3 有序 logit）
   和 eGFR（分档中点 OLS）关联，校正诊断，敏感性再校正间质纤维化。
输出：results/11_histology/*.tsv + PROVENANCE.json
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

_spec = importlib.util.spec_from_file_location("hl", Path(__file__).with_name("11_histology_lib.py"))
hl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hl)

STAGE = "11_histology"
DIAG = {"Diabetic Kidney Disease": "DKD", "Hypertensive Kidney Disease": "HKD"}
TARGETS_SECONDARY = ["pt_injured_logit", "tal_injured_logit", "ecgc_axis_all"]


def read_descriptors(path: Path) -> pd.DataFrame:
    """KPMP 病理描述评分 'Data Table'：表头行含 'Participant ID'。"""
    raw = pd.read_excel(path, sheet_name="Data Table", header=None)
    hdr = raw.index[raw.eq("Participant ID").any(axis=1)][0]
    df = raw.iloc[hdr + 1:].copy()
    df.columns = raw.iloc[hdr].astype(str).str.strip()
    df = df[df["Participant ID"].notna()]
    df["Participant ID"] = df["Participant ID"].astype(str).str.strip()
    return df


def participant_map(slides: set, sn_don: pd.DataFrame, sc_don: pd.DataFrame, clin: pd.DataFrame,
                    axes: pd.DataFrame, atlas_mods: list[str]) -> pd.DataFrame:
    ids = sorted(slides | set(sn_don.index))
    sc_ids = set(sc_don.index) | set(sc_don["ParticipantID"].dropna().astype(str))
    mods = clin.set_index("redcap_id")["open_modalities"].fillna("")
    m = pd.DataFrame(index=pd.Index(ids, name="participant"))
    m["has_slides"] = m.index.isin(slides)
    m["in_repository"] = m.index.isin(clin.redcap_id)
    m["in_snRNA_atlas"] = m.index.isin(sn_don.index)
    m["in_scRNA_atlas"] = m.index.isin(sc_ids)
    m["repository_lists_sc_or_sn"] = [any(x in mods.get(p, "") for x in atlas_mods) for p in m.index]
    m["has_tubular_axis"] = axes.tubular_axis.reindex(m.index).notna().values
    m["has_ecgc_axis"] = axes.ecgc_axis.reindex(m.index).notna().values
    # 样本外 = 不在本项目所用的 KPMP v2 单核 / 单细胞图谱中（分子轴未知、未参与训练）。
    # 资料库对其中部分人列出单核/单细胞文件（新版本释放），仅作描述，不用于本分析。
    any_atlas = m.in_snRNA_atlas | m.in_scRNA_atlas
    m["set"] = np.select([m.has_slides & m.in_snRNA_atlas, m.has_slides & ~any_atlas, m.has_slides],
                         ["training_atlas_and_slides", "out_of_sample_no_atlas", "slides_scRNA_only_excluded"],
                         "atlas_only_no_slides")
    # 对应核对：图谱元数据与资料库分档是否一致
    rep = clin.set_index("redcap_id")
    eg = sn_don["Baseline eGFR (ml/min/1.73m2) (Binned)"].astype(str).str.replace(" ml/min/1.73m2", "", regex=False)
    m["egfr_bin_atlas"] = eg.reindex(m.index)
    m["egfr_bin_repository"] = rep.baseline_egfr.reindex(m.index).str.replace(r" ml/min/1.73m2.*", "", regex=True)
    m["diabetes_atlas"] = sn_don.diabetes_history.reindex(m.index)
    m["diabetes_repository"] = rep.diabetes_history.reindex(m.index).str.replace(r" \(dh\)", "", regex=True)
    m["diagnosis"] = rep.primary_adjudicated_category.reindex(m.index).map(DIAG).fillna("other")
    return m


def clinical_table(clin: pd.DataFrame, cc: dict, if_pct: pd.Series) -> pd.DataFrame:
    c = clin.set_index("redcap_id")
    return pd.DataFrame({"alb_A": c.albuminuria.map(cc["albuminuria_A"]),
                         "alb_ord": c.albuminuria.map(cc["albuminuria_ord"]),
                         "egfr": c.baseline_egfr.map(hl.egfr_mid),
                         "diagnosis": c.primary_adjudicated_category.map(DIAG).fillna("other"),
                         "IF_pct": if_pct.reindex(c.index)})


def run_cv(emb_sets: dict, axes: pd.DataFrame, train: list[str], clin: pd.DataFrame, cfg: dict, seed: int):
    cv, h = cfg["cv"], cfg
    jobs = [(t, ts, "HE+PAS") for t in h["primary_targets"] + TARGETS_SECONDARY for ts in ("all", "tubule")]
    jobs += [(t, "all", s) for t in h["primary_targets"] for s in h["stains"]]
    res, preds = [], []
    for target, ts, stains in jobs:
        X = {s: emb_sets[ts][s] for s in stains.split("+") if s in emb_sets[ts]}
        cover = set().union(*[set(E.index) for E in X.values()])
        y = axes.loc[[p for p in train if p in cover], target].dropna()
        pred, r = hl.cv_with_permutation(X, y, cv, seed)
        dd = hl.diag_dummies(clin.diagnosis.reindex(y.index))
        r_d = hl.partial_corr(pred, y, dd)
        r_di = hl.partial_corr(pred, y, pd.concat([dd, clin.IF_pct.reindex(y.index)], axis=1))
        r_if = hl.partial_corr(pred, y, clin.IF_pct.reindex(y.index).to_frame())
        res.append({"target": target, "tileset": ts, "stains": stains, **r,
                    "partial_r_diagnosis": r_d[0], "partial_p_diagnosis": r_d[1],
                    "partial_r_IF": r_if[0], "partial_p_IF": r_if[1], "n_IF": r_if[2],
                    "partial_r_diagnosis_IF": r_di[0], "partial_p_diagnosis_IF": r_di[1],
                    "primary": target in h["primary_targets"] and ts == h["tileset"] and stains == "HE+PAS"})
        preds.append(pd.DataFrame({"participant": y.index, "target": target, "tileset": ts, "stains": stains,
                                   "measured": y.values, "cv_predicted": pred.values}))
        print(res[-1], flush=True)
    return pd.DataFrame(res), pd.concat(preds)


def associations(axes, mp, clin, emb, cfg, seed) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = mp.index[mp.set == "training_atlas_and_slides"].tolist()
    oos = mp.index[mp.set == "out_of_sample_no_atlas"].tolist()
    rows, oos_pred = [], []
    for target in cfg["primary_targets"]:
        y = axes.loc[train, target].dropna()
        fp = hl.fit_full_predict(emb, y, oos, cfg["cv"], seed)
        oos_pred.append(fp.assign(target=target).rename_axis("participant").reset_index())
        sets = {"slide_predicted_out_of_sample": fp["pred"],
                "measured_atlas_with_slides": y,
                "measured_all_snRNA_donors": axes[target].dropna()}
        for name, ax in sets.items():
            cl = clin.reindex(ax.index)
            for cov_name, cov in [("none", cl.iloc[:, :0]), ("diagnosis", hl.diag_dummies(cl.diagnosis)),
                                  ("diagnosis+IF", pd.concat([hl.diag_dummies(cl.diagnosis), cl.IF_pct], axis=1))]:
                for r in hl.clinical_models(ax, cl, cov):
                    if r["model"] == "spearman" and cov_name != "none":
                        continue
                    rows.append({"target": target, "set": name, "adjustment": cov_name, **r,
                                 "alpha_HE": fp.attrs.get("alpha_HE"), "alpha_PAS": fp.attrs.get("alpha_PAS")})
    return pd.DataFrame(rows), pd.concat(oos_pred)


def main() -> None:
    cfg_all = load_config()
    seed = cfg_all["seed"]
    set_global_seed(seed)
    h = cfg_all["histology"]
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    dest = ROOT / h["port"]["dest"]
    res3 = ROOT / "results" / "03_kpmp_pseudobulk"
    f_cnt, f_don, f_sc = res3 / "snRNA_cell_counts.tsv", res3 / "snRNA_donors.tsv", res3 / "scRNA_donors.tsv"
    f_pb = ROOT / cfg_all["paths"]["processed"] / "kpmp" / "snRNA_pseudobulk.parquet"
    f_mk, f_clin = ROOT / "results/05_markers/markers.tsv", ROOT / "metadata/kpmp_participants_clinical_bins.tsv"
    f_tiv = dest / Path(h["port"]["src_extra"][1]).name

    cnt = pd.read_csv(f_cnt, sep="\t", index_col=0, dtype={0: str})
    sn_don = pd.read_csv(f_don, sep="\t", index_col=0, dtype=str)
    sc_don = pd.read_csv(f_sc, sep="\t", index_col=0, dtype=str)
    cnt.index = cnt.index.astype(str)
    clin = pd.read_csv(f_clin, sep="\t", dtype=str)
    axes = hl.molecular_axes(cnt, pd.read_parquet(f_pb), pd.read_csv(f_mk, sep="\t"), h["axes"])
    axes.rename_axis("participant").to_csv(out / "molecular_axes_snRNA_donors.tsv", sep="\t")

    # 目标信度：同一供体单核与单细胞两种测量的分子轴一致性（决定切片预测的上限）
    f_sccnt, f_scpb = res3 / "scRNA_cell_counts.tsv", f_pb.with_name("scRNA_pseudobulk.parquet")
    sccnt = pd.read_csv(f_sccnt, sep="\t", index_col=0, dtype={0: str})
    sccnt.index = sccnt.index.astype(str)
    ax_sc = hl.molecular_axes(sccnt, pd.read_parquet(f_scpb), pd.read_csv(f_mk, sep="\t"), h["axes"])
    rel = []
    for t in ["tubular_axis", "pt_injured_logit", "tal_injured_logit", "ecgc_axis", "ecgc_axis_all"]:
        d = pd.concat([axes[t].rename("sn"), ax_sc[t].rename("sc")], axis=1).dropna()
        rho, p = stats.spearmanr(d.sn, d.sc)
        rel.append({"axis": t, "n_donors_both": len(d), "spearman_sn_vs_sc": rho, "p": p,
                    "pearson_sn_vs_sc": d.sn.corr(d.sc)})
    pd.DataFrame(rel).to_csv(out / "axis_reliability_sn_vs_sc.tsv", sep="\t", index=False)

    desc = read_descriptors(f_tiv).set_index("Participant ID")
    if_pct = pd.to_numeric(desc[h["clinical"]["if_column"]], errors="coerce").groupby(level=0).mean()
    cl = clinical_table(clin, h["clinical"], if_pct)
    emb_sets = {ts: hl.slide_embeddings(dest / "wsi_embeddings", ts, h["min_tiles"]) for ts in ("all", "tubule")}
    slides = set().union(*[set(E.index) for E in emb_sets["all"].values()])
    mp = participant_map(slides, sn_don, sc_don, clin, axes, h["clinical"]["atlas_modalities"])
    mp.to_csv(out / "participant_map.tsv", sep="\t")
    cl.loc[cl.index.isin(mp.index)].rename_axis("participant").to_csv(out / "clinical_bins_used.tsv", sep="\t")

    cov = (mp[mp.has_slides].assign(n=1).groupby(["set", "diagnosis"]).n.sum().unstack(fill_value=0))
    cov["total"] = cov.sum(1)
    for col, lab in [("has_tubular_axis", "with_tubular_axis"), ("has_ecgc_axis", "with_ecgc_axis")]:
        cov[lab] = mp[mp.has_slides].groupby("set")[col].sum()
    for v, lab in [("alb_A", "with_albuminuria"), ("egfr", "with_egfr"), ("IF_pct", "with_IF")]:
        cov[lab] = cl[v].reindex(mp.index).notna()[mp.has_slides].groupby(mp.set[mp.has_slides]).sum()
    cov.to_csv(out / "coverage.tsv", sep="\t")
    chk = mp[mp.in_snRNA_atlas & mp.in_repository]
    agree = pd.DataFrame([{"field": "baseline_eGFR_bin", "n_both": int((chk.egfr_bin_atlas.notna() & chk.egfr_bin_repository.notna()).sum()),
                           "n_agree": int((chk.egfr_bin_atlas == chk.egfr_bin_repository).sum())},
                          {"field": "diabetes_history", "n_both": int((chk.diabetes_atlas.notna() & chk.diabetes_repository.notna()).sum()),
                           "n_agree": int((chk.diabetes_atlas == chk.diabetes_repository).sum())},
                          {"field": "snRNA_donor_id_found_as_repository_redcap_id", "n_both": len(sn_don),
                           "n_agree": int(chk.shape[0])}])
    agree.to_csv(out / "id_mapping_check.tsv", sep="\t", index=False)
    print(cov.to_string(), "\n", agree.to_string(), flush=True)

    train = mp.index[mp.set == "training_atlas_and_slides"].tolist()
    cvres, cvpred = run_cv(emb_sets, axes, train, cl, h, seed)
    cvres.to_csv(out / "cv_prediction.tsv", sep="\t", index=False)
    cvpred.to_csv(out / "cv_predictions_by_participant.tsv", sep="\t", index=False)
    assoc, oos = associations(axes, mp, cl, emb_sets[h["tileset"]], h, seed)
    assoc.to_csv(out / "clinical_associations.tsv", sep="\t", index=False)
    oos.to_csv(out / "out_of_sample_predictions.tsv", sep="\t", index=False)
    print(assoc.round(4).to_string(index=False))

    outs = [out / f for f in ["molecular_axes_snRNA_donors.tsv", "participant_map.tsv", "clinical_bins_used.tsv",
                              "coverage.tsv", "id_mapping_check.tsv", "axis_reliability_sn_vs_sc.tsv", "cv_prediction.tsv",
                              "cv_predictions_by_participant.tsv", "clinical_associations.tsv",
                              "out_of_sample_predictions.tsv"]]
    write_provenance(STAGE, inputs=[f_cnt, f_don, f_sc, f_pb, f_mk, f_clin, f_tiv, f_sccnt, f_scpb], outputs=outs, seed=seed,
                     extra={"embeddings": str(dest / "wsi_embeddings"),
                            "embeddings_ported": "results/11a_port_wsi_embeddings/PROVENANCE.json",
                            "params": h})


if __name__ == "__main__":
    main()
