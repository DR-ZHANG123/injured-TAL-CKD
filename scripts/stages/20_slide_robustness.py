"""Stage 20：损伤 TAL 切片读出的稳健性与特异性。

1. 采集偏移：训练（有单核图谱）与样本外参与者的活检登记年份、扫描日期、扫描设置（像素分辨率 / 扫描仪）与
   切片颜色反卷积统计（tile 级统计的切片均值）比较；用颜色统计或嵌入区分两组的交叉验证 AUC。
2. 样本外 eGFR 关联：分染色；再校正登记年份、扫描设置、颜色主成分；限于训练扫描设置；
   嵌入在训练折内对颜色统计线性残差化（染料强度 / 全部颜色统计）后重训。
3. 时间外推：训练参与者中早登记者训练、晚登记者测试，并与同样大小的随机划分比较。
4. 病理描述评分基线（KPMP TIV）：嵌套 CV 比较描述评分、切片读出及两者合并；读出方差中描述评分解释的比例。
5. 诊断内表现；6. 相对于单核–单细胞一致性的信度上限；7. tile 分数尺度。
特异性：全部分子轴读出的样本外临床表；置换标签读出与随机投影读出的零分布；嵌入主成分基线；
直接以 eGFR 训练的切片模型；损伤 TAL 读出相对于描述评分、EC-GC 读出、直接 eGFR 读出的增量。
输出：results/20_slide_robustness/*.tsv + PROVENANCE.json
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


rl = _load("rl", "20_slide_robustness_lib.py")
sh = _load("sh", "20_slide_robustness_shift.py")
sp = _load("sp", "20_slide_robustness_specificity.py")
s11 = _load("s11", "11b_histology.py")
s17 = _load("s17", "17_fig45_tile_scores.py")
hl = rl.hl
STAGE = "20_slide_robustness"
TARGETS = ["tal_injured_logit", "pt_injured_logit", "tubular_axis", "ecgc_axis", "ecgc_axis_all"]


def main() -> None:
    cfg = load_config()
    seed = cfg["seed"]
    set_global_seed(seed)
    h, c = cfg["histology"], cfg["slide_robustness"]
    cv = h["cv"]
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    emb_dir = ROOT / h["port"]["dest"] / "wsi_embeddings"
    r11 = ROOT / "results/11_histology"
    f = {"hdr": ROOT / "results/20a_slide_headers/slide_headers.tsv", "map": r11 / "participant_map.tsv",
         "axes": r11 / "molecular_axes_snRNA_donors.tsv", "clin": r11 / "clinical_bins_used.tsv",
         "cvp": r11 / "cv_predictions_by_participant.tsv", "oos11": r11 / "out_of_sample_predictions.tsv",
         "tiv": ROOT / h["port"]["dest"] / Path(h["port"]["src_extra"][1]).name,
         "agree": ROOT / "results/16_fig_tables/sn_vs_sc_agreement.tsv",
         "donors": ROOT / "results/16_fig_tables/sn_vs_sc_donors.tsv",
         "sel17": ROOT / "results/17_fig45_tiles/selection.tsv",
         "tA": ROOT / "results/17_fig45_tiles/tile_scores_A_high.tsv",
         "tB": ROOT / "results/17_fig45_tiles/tile_scores_B_low.tsv"}
    rd = lambda k, **kw: pd.read_csv(f[k], sep="\t", **kw)  # noqa: E731
    mp = rd("map", index_col=0, dtype={0: str})
    axes = rd("axes", index_col=0, dtype={0: str})
    cl = rd("clin", index_col=0, dtype={0: str})
    cvp = rd("cvp", dtype={"participant": str}).drop_duplicates()
    for d in (mp, axes, cl):
        d.index = d.index.astype(str)
    sets = mp.set.map({"training_atlas_and_slides": "training", "out_of_sample_no_atlas": "out_of_sample"})
    train = mp.index[sets == "training"].tolist()
    oos = mp.index[sets == "out_of_sample"].tolist()
    emb, col, inv = rl.slide_tables(emb_dir, h["min_tiles"])
    hdr = rd("hdr", dtype={"participant": str})
    acq = rl.participant_acquisition(hdr, c["training_mpp"])
    D = rl.descriptor_table(s11.read_descriptors(f["tiv"]), c["descriptors"], c["descriptor_missing_codes"])
    feats = c["colour_features"]
    target = c["target"]
    y = axes.loc[[p for p in train if p in axes.index], target].dropna()
    tabs = {}

    # ---- 1. 采集偏移 ----
    slide_acq = hdr.merge(pd.concat([d.assign(stain=s).rename_axis("participant").reset_index() for s, d in col.items()]),
                          on=["participant", "stain"], how="left").merge(inv, on=["participant", "stain"], how="left")
    slide_acq.insert(1, "set", slide_acq.participant.map(sets).fillna("not_used"))
    tabs["slide_acquisition.tsv"] = slide_acq
    tabs["acquisition_shift_tests.tsv"] = sh.shift_tests(acq, col, inv, sets, feats)
    tabs["acquisition_by_set.tsv"] = (slide_acq[slide_acq.set != "not_used"]
                                      .groupby(["set", "stain", "scan_year", "scanner_id"], dropna=False).size()
                                      .rename("n_slides").reset_index())
    tabs["set_classifier_auc.tsv"] = pd.DataFrame(
        sh.set_classifier({s: d[feats] for s, d in col.items()}, sets, seed, "colour_statistics")
        + sh.set_classifier(emb, sets, seed, "embedding_mean"))

    # ---- 2. 样本外稳健性 ----
    fp = rl.fit_predict_tf(emb, y, oos, cv, seed)
    o11 = rd("oos11", dtype={"participant": str})
    o11 = o11[o11.target == target].set_index("participant")
    assert np.allclose(fp.pred.reindex(o11.index), o11.pred, atol=1e-6), "与 stage 11 样本外预测不一致"
    Cs = {"intensity": {s: d[c["intensity_features"]] for s, d in col.items()},
          "all_colour": {s: d[feats] for s, d in col.items()}}
    cv_rows = [rl.r_row("embedding_ridge_HE+PAS", rl.cv_predict_tf(emb, y, cv, seed), y, c["n_boot"], seed)]
    fp_norm = {}
    for name, C in Cs.items():
        cv_rows.append(rl.r_row(f"colour_residualised_{name}", rl.cv_predict_tf(emb, y, cv, seed, C), y, c["n_boot"], seed))
        fp_norm[name] = rl.fit_predict_tf(emb, y, oos, cv, seed, C)
        cv_rows.append(rl.r_row(f"colour_statistics_only_{name}", rl.cv_predict_tf(C, y, cv, seed), y, c["n_boot"], seed))
    pcs = rl.colour_pcs(col, feats, train, c["colour_pcs"])
    for pc in pcs.columns:
        cv_rows.append(rl.r_row(f"training_measured_vs_{pc}", pcs[pc].reindex(y.index), y, c["n_boot"], seed))
    tabs["cv_colour_variants.tsv"] = pd.DataFrame(cv_rows)
    tabs["oos_egfr_robustness.tsv"] = sh.oos_robustness(fp, fp_norm, cl, acq, pcs, oos, acq.alt_scanner == 0)
    tabs["stain_agreement.tsv"] = sh.stain_agreement(cvp, fp, acq, target, oos)
    # 分染色读出在训练（CV 预测）与样本外是否同样跟随病理描述评分与 eGFR（读出的形态学含义是否保持）
    cvw = cvp[(cvp.target == target) & (cvp.tileset == "all")].pivot_table(index="participant", columns="stains",
                                                                          values="cv_predicted")
    cvw = cvw.rename(columns={"HE": "pred_HE", "PAS": "pred_PAS", "HE+PAS": "pred"})
    trk = []
    for lab, P_ in [("training_cv", cvw), ("out_of_sample", fp)]:
        for col_ in ["pred", "pred_PAS", "pred_HE"]:
            for ref, v in [("IF_pct", D.IF_pct), ("TA_pct", D.TA_pct), ("egfr", cl.egfr)]:
                trk.append({"set": lab, **rl.r_row(f"{col_}_vs_{ref}", P_[col_], v.reindex(P_.index), c["n_boot"], seed)})
    tabs["stain_readout_tracking.tsv"] = pd.DataFrame(trk)

    # ---- 3. 时间外推 ----
    tabs["temporal_validation.tsv"] = sh.temporal(emb, y, acq, cl, cv, c, seed)

    # ---- 4. 病理描述评分基线 ----
    yd = y[D.reindex(y.index).IF_pct.notna()]
    P = sp.descriptor_nested_cv(emb, yd, D.reindex(yd.index), cv, seed)
    P.rename_axis("participant").assign(measured=yd).to_csv(out / "descriptor_nested_cv_predictions.tsv", sep="\t")
    rows = [rl.r_row(k, P[k], yd, c["n_boot"], seed) for k in P.columns]
    s_cv = cvp[(cvp.target == target) & (cvp.tileset == "all") & (cvp.stains == "HE+PAS")].set_index("participant").cv_predicted
    dxy = hl.diag_dummies(cl.diagnosis.reindex(y.index))
    for lab, cov in [("partial_r_slide_cv_adj_all_descriptors", D.reindex(y.index)),
                     ("partial_r_slide_cv_adj_IF_TA", D.reindex(y.index)[["IF_pct", "TA_pct"]]),
                     ("partial_r_slide_cv_adj_all_descriptors+diagnosis", pd.concat([D.reindex(y.index), dxy], axis=1))]:
        r, p, n = hl.partial_corr(s_cv.reindex(y.index), y, cov)
        rows.append({"analysis": lab, "n": n, "pearson_r": r, "p": p})
    tabs["descriptor_baseline_cv.tsv"] = pd.DataFrame(rows)
    tabs["descriptor_delta_r.tsv"] = pd.DataFrame([sp.paired_delta_r(P, yd, "descriptors+slide", "descriptors", c["n_boot"], seed),
                                                  sp.paired_delta_r(P, yd, "slide", "descriptors", c["n_boot"], seed)])
    tabs["descriptor_coverage.tsv"] = (D.reindex(train + oos).notna().groupby(sets.reindex(train + oos).values).sum()
                                       .rename_axis("set").reset_index())

    # ---- 特异性：全部读出 ----
    fps = {}
    for t in TARGETS:
        yt = axes.loc[[p for p in train if p in axes.index], t].dropna()
        fps[t] = rl.fit_predict_tf(emb, yt, oos, cv, seed)
    pd.concat([v.assign(target=k).rename_axis("participant").reset_index() for k, v in fps.items()]).to_csv(
        out / "oos_predictions_all_targets.tsv", sep="\t", index=False)
    acqcov = pd.concat([acq[["accession_time", "alt_scanner"]], pcs], axis=1)
    tabs["oos_clinical_all_targets.tsv"] = sp.oos_all_targets(fps, cl, D, acqcov, oos)
    eg, dd = cl.egfr.reindex(oos), hl.diag_dummies(cl.diagnosis.reindex(oos))
    obs = {t: rl.ols_egfr(fps[t].pred, eg, dd, "diagnosis") for t in TARGETS}
    pcE = sp.embedding_pcs(emb, train, c["embedding_pcs"])
    for k in pcE.columns:
        obs[f"embedding_{k}"] = rl.ols_egfr(pcE[k].reindex(oos), eg, dd, "diagnosis")
    # 直接以 eGFR（分档中点）训练的切片模型
    ye = cl.egfr.reindex(train).dropna()
    ye = ye[[p for p in ye.index if any(p in E.index for E in emb.values())]]
    ecv = rl.cv_predict_tf(emb, ye, cv, seed)
    fpe = rl.fit_predict_tf(emb, ye, oos, cv, seed)
    obs["direct_eGFR_model"] = rl.ols_egfr(fpe.pred, eg, dd, "diagnosis")
    direct = [rl.r_row("direct_eGFR_model_training_cv", ecv, ye, c["n_boot"], seed),
              rl.r_row("direct_eGFR_model_out_of_sample", fpe.pred.reindex(oos), eg, c["n_boot"], seed),
              rl.r_row("injured_TAL_readout_out_of_sample_vs_eGFR", fps[target].pred.reindex(oos), eg, c["n_boot"], seed),
              rl.r_row("injured_TAL_readout_vs_direct_eGFR_readout_out_of_sample", fps[target].pred.reindex(oos),
                       fpe.pred.reindex(oos), c["n_boot"], seed),
              rl.r_row("injured_TAL_readout_vs_ECGC_readout_out_of_sample", fps[target].pred.reindex(oos),
                       fps["ecgc_axis"].pred.reindex(oos), c["n_boot"], seed)]
    tabs["direct_egfr_model.tsv"] = pd.DataFrame(direct)
    null_perm = sp.permuted_label_null(emb, y, oos, eg, dd, cv, seed, c["n_null"])
    null_rp = sp.random_projection_null(emb, train, oos, eg, dd, seed, c["n_null"])
    pd.DataFrame(null_perm, columns=["egfr_per_sd", "t", "p"]).to_csv(out / "null_permuted_label_readouts.tsv", sep="\t", index=False)
    pd.DataFrame(null_rp, columns=["egfr_per_sd", "t", "p"]).to_csv(out / "null_random_projections.tsv", sep="\t", index=False)
    tabs["oos_egfr_null_comparison.tsv"] = pd.DataFrame(sp.null_summary("permuted_injured_TAL_labels", null_perm, obs)
                                                        + sp.null_summary("random_projections", null_rp, obs))
    # 增量价值（样本外 eGFR）
    tal = fps[target].pred.reindex(oos)
    Do = D.reindex(oos)
    inc = [("diagnosis", dd), ("diagnosis+all_descriptors", pd.concat([dd, Do], axis=1)),
           ("diagnosis+IF+TA", pd.concat([dd, Do[["IF_pct", "TA_pct"]]], axis=1)),
           ("diagnosis+ECGC_readout", pd.concat([dd, fps["ecgc_axis"].pred.rename("ecgc")], axis=1)),
           ("diagnosis+all_descriptors+ECGC_readout", pd.concat([dd, Do, fps["ecgc_axis"].pred.rename("ecgc")], axis=1)),
           ("diagnosis+direct_eGFR_readout", pd.concat([dd, fpe.pred.rename("direct")], axis=1)),
           ("diagnosis+embedding_PCs", pd.concat([dd, pcE.reindex(oos)], axis=1)),
           ("diagnosis+all_descriptors+ECGC+direct_eGFR+embedding_PCs",
            pd.concat([dd, Do, fps["ecgc_axis"].pred.rename("ecgc"), fpe.pred.rename("direct"), pcE.reindex(oos)], axis=1))]
    rows = [{"readout": "injured_TAL", **rl.ols_egfr(tal, eg, cov, lab)} for lab, cov in inc]
    rows += [{"readout": "ECGC", **rl.ols_egfr(fps["ecgc_axis"].pred.reindex(oos), eg, pd.concat([dd, tal.rename("tal")], axis=1),
                                               "diagnosis+injured_TAL_readout")},
             {"readout": "direct_eGFR", **rl.ols_egfr(fpe.pred.reindex(oos), eg, pd.concat([dd, tal.rename("tal")], axis=1),
                                                      "diagnosis+injured_TAL_readout")}]
    tabs["oos_incremental_value.tsv"] = pd.DataFrame(rows)
    ve = sp.variance_explained(tal, Do, cl.diagnosis.reindex(oos), "out_of_sample_readout")
    ve += sp.variance_explained(s_cv.reindex(y.index), D.reindex(y.index), cl.diagnosis.reindex(y.index), "training_cv_readout")
    ve += sp.variance_explained(y, D.reindex(y.index), cl.diagnosis.reindex(y.index), "training_measured_injured_TAL")
    tabs["readout_variance_explained_by_descriptors.tsv"] = pd.DataFrame(ve)

    # ---- 5–7 ----
    tabs["within_diagnosis_cv.tsv"] = sh.within_diagnosis(cvp, emb, y, cl, cv, c, seed, target)
    tabs["reliability_ceiling.tsv"] = sh.reliability(rd("agree"), rd("donors", dtype={"donor_id": str}),
                                                     float(s_cv.reindex(y.index).corr(y)), s_cv.reindex(y.index), y,
                                                     h["axes"]["logit_eps"], c["n_boot"], seed, train)
    ts = sh.tile_scale(emb_dir, emb["PAS"], y, cv, seed, s17.ridge_coef, feats)
    tabs["tile_scores_all_training_PAS.tsv"] = ts
    sel = rd("sel17", dtype={"participant": str})
    scale = [{"quantity": "measured_injured_TAL_logit_training", "min": y.min(), "max": y.max(), "sd": y.std()},
             {"quantity": "slide_cv_prediction_training", "min": s_cv.min(), "max": s_cv.max(), "sd": s_cv.std()},
             {"quantity": "loo_PAS_slide_prediction_training", "min": ts.loo_slide_prediction.min(),
              "max": ts.loo_slide_prediction.max(), "sd": ts.loo_slide_prediction.std()},
             {"quantity": "tile_scores_all_training_pooled", "min": ts.tile_min.min(), "max": ts.tile_max.max(),
              "sd": float(np.sqrt((ts.tile_sd ** 2).mean() + ts.tile_mean.var()))},
             {"quantity": "between_slide_sd_of_tile_means", "sd": ts.tile_mean.std()},
             {"quantity": "loo_alpha_at_grid_lower_bound_fraction", "mean": float((ts.alpha == min(cv["ridge_alphas"])).mean())},
             {"quantity": "tubule_minus_other_tile_score_median", "mean": ts.tubule_minus_other_mean_score.median(),
              "min": ts.tubule_minus_other_mean_score.min(), "max": ts.tubule_minus_other_mean_score.max(),
              "sd": float((ts.tubule_minus_other_mean_score > 0).mean())},
             {"quantity": "median_within_slide_tile_sd", "sd": ts.tile_sd.median(),
              "min": ts.tile_sd.min(), "max": ts.tile_sd.max()}]
    for lab, k in [("A_high", "tA"), ("B_low", "tB")]:
        tt = rd(k)
        a = sel.set_index("label").loc[lab]
        scale.append({"quantity": f"fig4d_tiles_{lab}", "min": tt.tile_score.min(), "max": tt.tile_score.max(),
                      "sd": tt.tile_score.std(), "mean": tt.tile_score.mean(), "loo_alpha": a.loo_alpha,
                      "cv_predicted_HE+PAS": a.cv_predicted, "loo_PAS_slide_prediction": a.loo_slide_prediction})
    tabs["tile_score_scale.tsv"] = pd.DataFrame(scale)

    outs = []
    for name, t in tabs.items():
        t.to_csv(out / name, sep="\t", index=False)
        outs.append(out / name)
        print(f"== {name}\n", t.round(4).head(40).to_string(), flush=True)
    outs += [out / x for x in ["descriptor_nested_cv_predictions.tsv", "oos_predictions_all_targets.tsv",
                               "null_permuted_label_readouts.tsv", "null_random_projections.tsv"]]
    write_provenance(STAGE, inputs=list(f.values()), outputs=outs, seed=seed,
                     extra={"embeddings": str(emb_dir), "params": c, "cv_params": cv})


if __name__ == "__main__":
    main()
