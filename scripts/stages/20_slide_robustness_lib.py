"""Stage 20 公用函数：切片级嵌入与颜色统计、采集元数据、带折内变换的参与者水平交叉验证、自助法、eGFR 模型。"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

_spec = importlib.util.spec_from_file_location("hl", Path(__file__).with_name("11_histology_lib.py"))
hl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hl)


# ---------- 切片级表示 ----------
def slide_tables(emb_dir: Path, min_tiles: int) -> tuple[dict, dict, pd.DataFrame]:
    """返回 {染色: 参与者×1024 平均嵌入}、{染色: 参与者×颜色统计切片均值}、切片清单（含 tile 数）。
    与 11_histology_lib.slide_embeddings(tileset='all') 相同的 tile 集合与 min_tiles 过滤。"""
    emb, col, rows = {}, {}, []
    for f in sorted(emb_dir.glob("*.npz")):
        pid, stain = f.name.split("__")[:2]
        z = np.load(f, allow_pickle=True)
        if len(z["emb"]) < min_tiles:
            continue
        emb.setdefault(stain, {})[pid] = z["emb"].astype(np.float32).mean(0)
        feats = pd.DataFrame(z["feats"], columns=list(z["feat_names"]))
        col.setdefault(stain, {})[pid] = feats.mean()
        rows.append({"participant": pid, "stain": stain, "npz": f.name, "n_tiles": len(z["emb"]),
                     "tubule_tile_frac": float(z["tubule"].mean())})
    return ({s: pd.DataFrame(v).T for s, v in emb.items()}, {s: pd.DataFrame(v).T for s, v in col.items()},
            pd.DataFrame(rows))


def participant_acquisition(hdr: pd.DataFrame, training_mpp: float) -> pd.DataFrame:
    """参与者级采集变量：登记日期（年 + 月/12；缺失时用扫描日期）、扫描年份、是否非训练扫描设置。"""
    h = hdr.copy()
    h["acc_t"] = h.accession_year + (h.accession_month - 1) / 12
    h["scan_t"] = h.scan_year + (h.scan_month - 1) / 12
    h["acc_t"] = h.acc_t.fillna(h.scan_t)
    h["alt_scanner"] = (abs(h.mpp_inventory - training_mpp) > 1e-3).astype(float)
    g = h.groupby("participant")
    return pd.DataFrame({"accession_time": g.acc_t.min(), "accession_year": np.floor(g.acc_t.min()),
                         "scan_time": g.scan_t.min(), "scan_year": g.scan_year.min(),
                         "alt_scanner": g.alt_scanner.max(), "scanner_ids": g.scanner_id.agg(lambda s: ";".join(sorted(set(s))))})


# ---------- 折内变换：在训练折内用颜色统计线性回归嵌入，取残差（训练折系数用于测试折） ----------
def resid_fit(E: np.ndarray, C: np.ndarray):
    A = np.column_stack([np.ones(len(C)), C])
    B, *_ = np.linalg.lstsq(A, E, rcond=None)
    return B


def resid_apply(E: np.ndarray, C: np.ndarray, B) -> np.ndarray:
    return E - np.column_stack([np.ones(len(C)), C]) @ B


def _split_fit(E: pd.DataFrame, Cc: pd.DataFrame | None, tr: list, te: list, y: np.ndarray, cv: dict, rng):
    Xtr, Xte = E.loc[tr].values, E.loc[te].values
    if Cc is not None:
        B = resid_fit(Xtr, Cc.loc[tr].values)
        Xtr, Xte = resid_apply(Xtr, Cc.loc[tr].values, B), resid_apply(Xte, Cc.loc[te].values, B)
    return hl.fit_predict(Xtr, y, Xte, cv["ridge_alphas"], cv["inner_folds"], rng)[0]


def cv_predict_tf(X: dict, y: pd.Series, cv: dict, seed: int, C: dict | None = None,
                  repeats: int | None = None) -> pd.Series:
    """与 hl.cv_predict 相同的参与者水平重复 K 折与染色平均；可选颜色残差化（折内拟合）。"""
    pids = y.index.to_numpy()
    R = repeats or cv["repeats"]
    preds = np.zeros((R, len(pids)))
    for r in range(R):
        rng = np.random.default_rng(seed + r)
        fold = rng.permutation(np.arange(len(pids)) % cv["outer_folds"])
        acc, cnt = np.zeros(len(pids)), np.zeros(len(pids))
        for i in range(cv["outer_folds"]):
            for s, E in X.items():
                has = np.isin(pids, E.index)
                tr, te = has & (fold != i), has & (fold == i)
                if te.sum() == 0:
                    continue
                acc[te] += _split_fit(E, None if C is None else C[s], list(pids[tr]), list(pids[te]),
                                      y.values[tr], cv, rng)
                cnt[te] += 1
        preds[r] = acc / cnt
    return pd.Series(preds.mean(0), index=y.index)


def fit_predict_tf(X: dict, y: pd.Series, new: list, cv: dict, seed: int, C: dict | None = None) -> pd.DataFrame:
    """在全部训练参与者上拟合（可选颜色残差化），预测新参与者；与 hl.fit_full_predict 同一随机流。"""
    rng = np.random.default_rng(seed)
    out = pd.DataFrame(index=new)
    for s, E in X.items():
        tr = [p for p in y.index if p in E.index]
        te = [p for p in new if p in E.index]
        out.loc[te, f"pred_{s}"] = _split_fit(E, None if C is None else C[s], tr, te, y.loc[tr].values, cv, rng)
    out["pred"] = out[[c for c in out.columns if c.startswith("pred_")]].mean(1)
    return out


def colour_pcs(col: dict, feats: list, train: list, k: int) -> pd.DataFrame:
    """每种染色：颜色统计在训练参与者内 z 标准化并做 PCA，投影全部参与者；缺失染色以训练均值（0）填补。"""
    parts = []
    for s, D in col.items():
        D = D[feats]
        tr = D.loc[D.index.intersection(train)]
        Z = (D - tr.mean()) / tr.std()
        _, _, Vt = np.linalg.svd(Z.loc[tr.index].values, full_matrices=False)
        parts.append(pd.DataFrame(Z.values @ Vt[:k].T, index=D.index, columns=[f"{s}_colPC{j+1}" for j in range(k)]))
    return pd.concat(parts, axis=1).fillna(0.0)


# ---------- 统计 ----------
def boot_r(x: pd.Series, y: pd.Series, n_boot: int, seed: int) -> tuple[float, float, float, int]:
    d = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    rng = np.random.default_rng(seed)
    bs = [np.corrcoef(d.values[i].T)[0, 1] for i in (rng.integers(0, len(d), len(d)) for _ in range(n_boot))]
    return float(d.x.corr(d.y)), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5)), len(d)


def r_row(label: str, x: pd.Series, y: pd.Series, n_boot: int, seed: int, **kw) -> dict:
    r, lo, hi, n = boot_r(x, y, n_boot, seed)
    d = pd.concat([x, y], axis=1).dropna()
    p = stats.pearsonr(d.iloc[:, 0], d.iloc[:, 1])[1] if n > 2 else np.nan
    rho = stats.spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0] if n > 2 else np.nan
    return {"analysis": label, "n": n, "pearson_r": r, "ci_low": lo, "ci_high": hi, "p": p, "spearman_rho": rho, **kw}


def z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def ols_egfr(pred: pd.Series, egfr: pd.Series, cov: pd.DataFrame | None = None, label: str = "") -> dict:
    """eGFR（分档中点）~ 读出（在分析样本内 z 标准化）+ 协变量；同时给出加入读出前后的 R²。"""
    parts = [pred.rename("readout"), egfr.rename("egfr")] + ([cov] if cov is not None and cov.shape[1] else [])
    d = pd.concat(parts, axis=1).dropna()
    d["readout"] = z(d.readout)
    C = d.drop(columns=["readout", "egfr"]).astype(float)
    C = C.loc[:, C.std() > 0]
    full = sm.OLS(d.egfr, sm.add_constant(pd.concat([d.readout, C], axis=1), has_constant="add")).fit()
    base = sm.OLS(d.egfr, sm.add_constant(C, has_constant="add")).fit()
    ci = full.conf_int().loc["readout"]
    return {"model": label, "n": len(d), "covariates": "+".join(C.columns) or "none",
            "egfr_per_sd": full.params["readout"], "ci_low": ci[0], "ci_high": ci[1], "p": full.pvalues["readout"],
            "t": full.tvalues["readout"], "r2_without_readout": base.rsquared, "r2_with_readout": full.rsquared,
            "delta_r2": full.rsquared - base.rsquared, "adj_r2_with_readout": full.rsquared_adj}


def descriptor_table(raw: pd.DataFrame, cols: dict, missing: list) -> pd.DataFrame:
    """KPMP TIV 评分 → 数值；缺失代码置 NaN；同一参与者多条记录取平均。"""
    d = raw[list(cols.values())].copy()
    d.columns = list(cols)
    miss = {str(m) for m in missing}
    d = d.apply(lambda s: pd.to_numeric(s.astype(str).str.strip().where(~s.astype(str).str.strip().isin(miss))
                                        .str.replace("0-5", "2.5", regex=False), errors="coerce"))
    d.index = raw["Participant ID"].astype(str).values
    return d.groupby(level=0).mean()
