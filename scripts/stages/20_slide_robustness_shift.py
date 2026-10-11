"""Stage 20 第 1–3、5–7 部分：采集偏移、样本外稳健性、时间外推、诊断内表现、信度上限、tile 分数尺度。"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_spec = importlib.util.spec_from_file_location("rl", Path(__file__).with_name("20_slide_robustness_lib.py"))
rl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rl)
hl = rl.hl


# ---------- 1. 采集偏移 ----------
def shift_tests(acq: pd.DataFrame, col: dict, inv: pd.DataFrame, sets: pd.Series, feats: list) -> pd.DataFrame:
    rows = []
    for stain, D in col.items():
        pids = D.index[D.index.map(sets).isin(["training", "out_of_sample"])]
        g = sets.reindex(pids)
        ntl = inv[inv.stain == stain].set_index("participant")
        V = pd.concat([acq.reindex(pids)[["accession_time", "scan_time", "alt_scanner"]],
                       D.loc[pids, feats + ["tissue_frac", "nuclei_per_1e4px_tissue"]],
                       ntl.reindex(pids)[["n_tiles", "tubule_tile_frac"]]], axis=1)
        for v in V.columns:
            a, b = V.loc[g == "training", v].dropna(), V.loc[g == "out_of_sample", v].dropna()
            if v == "alt_scanner":
                tab = [[int((a == 1).sum()), int((a == 0).sum())], [int((b == 1).sum()), int((b == 0).sum())]]
                p, test = stats.fisher_exact(tab)[1], "fisher_exact"
                sa, sb = a.mean(), b.mean()
            else:
                p, test = stats.mannwhitneyu(a, b).pvalue, "mann_whitney"
                sa, sb = a.median(), b.median()
            rows.append({"stain": stain, "variable": v, "n_training": len(a), "n_out_of_sample": len(b),
                         "training_summary": sa, "out_of_sample_summary": sb,
                         "summary": "proportion" if v == "alt_scanner" else "median", "test": test, "p": p})
    return pd.DataFrame(rows)


def set_classifier(X: dict, sets: pd.Series, seed: int, name: str) -> list[dict]:
    """参与者水平 5 折：从切片特征区分训练集与样本外集合（AUC 越高，分布偏移越大）。"""
    out = []
    for s, D in X.items():
        pids = D.index[D.index.map(sets).isin(["training", "out_of_sample"])]
        y = (sets.reindex(pids) == "out_of_sample").astype(int).values
        clf = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=10, cv=5, max_iter=5000))
        p = cross_val_predict(clf, D.loc[pids].values, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                              method="predict_proba")[:, 1]
        out.append({"stain": s, "features": name, "n": len(y), "n_out_of_sample": int(y.sum()),
                    "cv_auc_training_vs_out_of_sample": roc_auc_score(y, p)})
    return out


# ---------- 2. 样本外 eGFR 关联的稳健性 ----------
def oos_robustness(fp: pd.DataFrame, fp_norm: dict, cl: pd.DataFrame, acq: pd.DataFrame, pcs: pd.DataFrame,
                   oos: list, training_mpp_mask: pd.Series) -> pd.DataFrame:
    dd = hl.diag_dummies(cl.diagnosis.reindex(oos))
    eg = cl.egfr.reindex(oos)
    A = acq.reindex(oos)
    pc_cols = {"pred": list(pcs.columns), "pred_HE": [c for c in pcs if c.startswith("HE_")],
               "pred_PAS": [c for c in pcs if c.startswith("PAS_")]}
    rows = []
    for which in ["pred", "pred_PAS", "pred_HE"]:
        x = fp[which].reindex(oos)
        P = pcs.reindex(oos)[pc_cols[which]]
        specs = [("diagnosis", dd), ("diagnosis+accession_year", pd.concat([dd, A.accession_time], axis=1)),
                 ("diagnosis+scan_year", pd.concat([dd, A.scan_time], axis=1)),
                 ("diagnosis+alt_scanner", pd.concat([dd, A.alt_scanner], axis=1)),
                 ("diagnosis+colour_PCs", pd.concat([dd, P], axis=1)),
                 ("diagnosis+accession_year+alt_scanner+colour_PCs",
                  pd.concat([dd, A[["accession_time", "alt_scanner"]], P], axis=1))]
        for lab, cov in specs:
            rows.append({"readout": which, "subset": "all_out_of_sample", **rl.ols_egfr(x, eg, cov, lab)})
        keep = [p for p in oos if training_mpp_mask.get(p, False)]
        rows.append({"readout": which, "subset": "training_scanner_only",
                     **rl.ols_egfr(x.reindex(keep), eg.reindex(keep), dd.reindex(keep), "diagnosis")})
    for name, f in fp_norm.items():
        for which in ["pred", "pred_PAS", "pred_HE"]:
            rows.append({"readout": which, "subset": f"colour_residualised_{name}",
                         **rl.ols_egfr(f[which].reindex(oos), eg, dd, "diagnosis")})
    return pd.DataFrame(rows)


def stain_agreement(cvp: pd.DataFrame, fp: pd.DataFrame, acq: pd.DataFrame, target: str, oos: list) -> pd.DataFrame:
    c = cvp[(cvp.target == target) & (cvp.tileset == "all")].pivot_table(index="participant", columns="stains",
                                                                         values="cv_predicted")
    o = fp.reindex(oos)
    alt = acq.alt_scanner.reindex(oos)
    rows = [{"set": "training_cv", "n": int(c[["HE", "PAS"]].dropna().shape[0]), "r_HE_vs_PAS": c.HE.corr(c.PAS),
             "mean_pred_HE+PAS": c["HE+PAS"].mean(), "sd_pred_HE+PAS": c["HE+PAS"].std()}]
    for lab, m in [("out_of_sample_all", alt.notna()), ("out_of_sample_training_scanner", alt == 0),
                   ("out_of_sample_other_scanner", alt == 1)]:
        s = o[m.values]
        rows.append({"set": lab, "n": int(s[["pred_HE", "pred_PAS"]].dropna().shape[0]),
                     "r_HE_vs_PAS": s.pred_HE.corr(s.pred_PAS), "mean_pred_HE+PAS": s.pred.mean(),
                     "sd_pred_HE+PAS": s.pred.std()})
    return pd.DataFrame(rows)


# ---------- 3. 时间外推（训练参与者内） ----------
def temporal(X: dict, y: pd.Series, acq: pd.DataFrame, cl: pd.DataFrame, cv: dict, c: dict, seed: int) -> pd.DataFrame:
    rows = []
    for split_var in ["accession_time", "scan_time"]:
        t = acq[split_var].reindex(y.index)
        early = y.index[t < c["temporal_split_year"] + 1].tolist()
        late = y.index[t >= c["temporal_split_year"] + 1].tolist()
        fp = rl.fit_predict_tf(X, y.loc[early], late, cv, seed)
        row = rl.r_row(f"temporal_{split_var}", fp.pred, y.loc[late], c["n_boot"], seed,
                       n_train=len(early), n_test=len(late))
        dd = hl.diag_dummies(cl.diagnosis.reindex(late))
        row["partial_r_diagnosis"], row["partial_p_diagnosis"], _ = hl.partial_corr(fp.pred, y.loc[late], dd)
        row["test_diagnoses"] = cl.diagnosis.reindex(late).value_counts().to_dict().__repr__()
        row["train_diagnoses"] = cl.diagnosis.reindex(early).value_counts().to_dict().__repr__()
        rng = np.random.default_rng(seed)
        ref = []
        for _ in range(c["random_split_reps"]):
            perm = rng.permutation(y.index.to_numpy())
            tr, te = list(perm[:len(early)]), list(perm[len(early):])
            ref.append(rl.fit_predict_tf(X, y.loc[tr], te, cv, int(rng.integers(1e9))).pred.corr(y.loc[te]))
        ref = np.array(ref)
        row.update({"random_split_median_r": float(np.median(ref)), "random_split_q025": float(np.quantile(ref, .025)),
                    "random_split_q975": float(np.quantile(ref, .975)),
                    "fraction_random_splits_below_temporal_r": float((ref < row["pearson_r"]).mean())})
        rows.append(row)
    return pd.DataFrame(rows)


# ---------- 5. 诊断内表现 ----------
def within_diagnosis(cvp: pd.DataFrame, X: dict, y: pd.Series, cl: pd.DataFrame, cv: dict, c: dict, seed: int,
                     target: str) -> pd.DataFrame:
    sub = cvp[(cvp.target == target)].drop_duplicates()
    rows = []
    main = sub[(sub.tileset == "all") & (sub.stains == "HE+PAS")].set_index("participant")
    dx = cl.diagnosis.reindex(main.index)
    for lab, m in [("all_training", dx.notna()), ("DKD+HKD", dx.isin(c["within_diagnoses"])),
                   ("DKD", dx == "DKD"), ("HKD", dx == "HKD"), ("other", dx == "other")]:
        rows.append(rl.r_row(f"pooled_model_{lab}", main.cv_predicted[m], main.measured[m], c["n_boot"], seed,
                             tileset="all", stains="HE+PAS", trained_on="all_training"))
    for ts, st in [("tubule", "HE+PAS"), ("all", "PAS"), ("all", "HE")]:
        s = sub[(sub.tileset == ts) & (sub.stains == st)].set_index("participant")
        rows.append(rl.r_row(f"pooled_model_{ts}_{st}", s.cv_predicted, s.measured, c["n_boot"], seed,
                             tileset=ts, stains=st, trained_on="all_training"))
    # 只用 DKD+HKD（及只用 DKD）参与者重新训练和交叉验证，置换 y 重做整个流程
    from joblib import Parallel, delayed
    for lab, keep in [("DKD+HKD", c["within_diagnoses"]), ("DKD", ["DKD"])]:
        yy = y[cl.diagnosis.reindex(y.index).isin(keep)]
        pred = rl.cv_predict_tf(X, yy, cv, seed)
        r = pred.corr(yy)
        rng = np.random.default_rng(seed)
        perms = [pd.Series(rng.permutation(yy.values), index=yy.index) for _ in range(c["n_perm"])]
        null = np.array(Parallel(n_jobs=cv["n_jobs"])(delayed(lambda yp: rl.cv_predict_tf(X, yp, cv, seed).corr(yp))(yp)
                                                      for yp in perms))
        row = rl.r_row(f"retrained_within_{lab}", pred, yy, c["n_boot"], seed, tileset="all", stains="HE+PAS",
                       trained_on=lab)
        row.update({"perm_p": float((1 + (null >= r).sum()) / (len(null) + 1)), "null_mean_r": float(null.mean())})
        rows.append(row)
    return pd.DataFrame(rows)


# ---------- 6. 信度上限 ----------
def reliability(agree: pd.DataFrame, donors: pd.DataFrame, cv_r: float, cv_pred: pd.Series, y: pd.Series,
                eps: float, n_boot: int, seed: int, train: list) -> pd.DataFrame:
    t = donors[donors.lineage == "TAL"].set_index("donor_id")
    lg = np.log((t[["sn", "sc"]] + eps) / (1 - t[["sn", "sc"]] + eps))
    rel = lg.sn.corr(lg.sc)
    rng = np.random.default_rng(seed)
    d = pd.concat([cv_pred.rename("p"), y.rename("y")], axis=1).dropna()
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(lg), len(lg))
        j = rng.integers(0, len(d), len(d))
        rr, cr = lg.iloc[i].sn.corr(lg.iloc[i].sc), d.iloc[j].p.corr(d.iloc[j].y)
        bs.append((rr, cr, cr / np.sqrt(rr), np.sqrt(rr)))
    bs = np.array(bs)
    tr = lg.loc[lg.index.intersection(train)]
    a = agree.set_index("lineage").loc["TAL"]
    rows = [("sn_vs_sc_pearson_logit_all_donors", rel, bs[:, 0], len(lg)),
            ("sn_vs_sc_spearman_all_donors", a.spearman, None, int(a.n_donors)),
            ("sn_vs_sc_pearson_logit_training_slide_participants", tr.sn.corr(tr.sc), None, len(tr)),
            ("slide_cv_pearson_r", cv_r, bs[:, 1], len(d)),
            ("max_attainable_r_sqrt_reliability", np.sqrt(rel), bs[:, 3], len(lg)),
            ("disattenuated_r_cv_over_sqrt_reliability", cv_r / np.sqrt(rel), bs[:, 2], len(d)),
            ("share_of_reproducible_variance_r2_over_reliability", cv_r ** 2 / rel, bs[:, 1] ** 2 / bs[:, 0], len(d))]
    return pd.DataFrame([{"quantity": q, "value": v, "n": n,
                          "ci_low": np.nanpercentile(b, 2.5) if b is not None else np.nan,
                          "ci_high": np.nanpercentile(b, 97.5) if b is not None else np.nan} for q, v, b, n in rows])


# ---------- 7. tile 分数尺度（全部训练参与者 PAS，留一参与者模型） ----------
def tile_scale(emb_dir: Path, E: pd.DataFrame, y: pd.Series, cv: dict, seed: int, ridge_coef, colour_feats: list):
    rows = []
    for pid in [p for p in y.index if p in E.index]:
        tr = [p for p in y.index if p != pid and p in E.index]
        rng = np.random.default_rng(seed)
        p_loo, alpha = hl.fit_predict(E.loc[tr].values, y.loc[tr].values, E.loc[[pid]].values,
                                      cv["ridge_alphas"], cv["inner_folds"], rng)
        mu, sd, beta, ym = ridge_coef(E.loc[tr].values.astype(np.float64), y.loc[tr].values, alpha)
        f = sorted(emb_dir.glob(f"{pid}__PAS__*.npz"))[0]
        zf = np.load(f, allow_pickle=True)
        s = ((zf["emb"].astype(np.float32).astype(np.float64) - mu) / sd) @ beta + ym
        assert abs(s.mean() - p_loo[0]) < 1e-3
        F = pd.DataFrame(zf["feats"], columns=list(zf["feat_names"]))
        tub = zf["tubule"].astype(bool)
        row = {"participant": pid, "measured": y[pid], "loo_slide_prediction": p_loo[0], "alpha": alpha,
               "n_tiles": len(s), "tile_mean": s.mean(), "tile_sd": s.std(), "tile_min": s.min(),
               "tile_max": s.max(), "tile_q05": np.quantile(s, .05), "tile_q95": np.quantile(s, .95),
               "tubule_minus_other_mean_score": s[tub].mean() - s[~tub].mean() if tub.any() and (~tub).any() else np.nan}
        for v in colour_feats + ["nuclei_per_1e4px_tissue"]:
            row[f"spearman_tile_score_vs_{v}"] = stats.spearmanr(s, F[v], nan_policy="omit")[0]
        rows.append(row)
    return pd.DataFrame(rows)
