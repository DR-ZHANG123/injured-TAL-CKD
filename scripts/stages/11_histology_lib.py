"""Stage 11 公用函数：切片表示、供体级分子轴、参与者水平嵌套交叉验证岭回归、临床关联。"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.miscmodels.ordinal_model import OrderedModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.pbstats import state_matrix  # noqa: E402


# ---------- 切片表示 ----------
def slide_embeddings(emb_dir: Path, tileset: str, min_tiles: int) -> dict[str, pd.DataFrame]:
    """{染色: 参与者 × 1024 平均嵌入}。tileset = all（全部 QC tile）或 tubule（预设小管富集 tile）。"""
    out: dict[str, dict] = {}
    for f in sorted(emb_dir.glob("*.npz")):
        pid, stain = f.name.split("__")[:2]
        z = np.load(f, allow_pickle=True)
        m = np.ones(len(z["emb"]), bool) if tileset == "all" else z["tubule"].astype(bool)
        if m.sum() >= min_tiles:
            out.setdefault(stain, {})[pid] = z["emb"][m].astype(np.float32).mean(0)
    return {s: pd.DataFrame(v).T for s, v in out.items()}


# ---------- 分子轴 ----------
def _logit(p: pd.Series, eps: float) -> pd.Series:
    return np.log((p + eps) / (1 - p + eps))


def _z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def molecular_axes(cnt: pd.DataFrame, pb: pd.DataFrame, markers: pd.DataFrame, a: dict) -> pd.DataFrame:
    """供体级：损伤 PT / TAL 比例（logit）及其 z 平均 = 小管修复失败轴；EC-GC 标志得分。
    z 标准化在全部合格单核供体中进行（与切片无关）。"""
    c = cnt.reindex(columns=sorted(set(cnt.columns) | set(a["pt_den"]) | set(a["tal_den"])), fill_value=0)
    d = pd.DataFrame(index=cnt.index)
    for k in ("pt", "tal"):
        den = c[a[f"{k}_den"]].sum(1)
        d[f"n_{k}"] = den
        d[f"{k}_injured_frac"] = (c[a[f"{k}_num"]].sum(1) / den).where(den >= a["min_cells"])
        d[f"{k}_injured_logit"] = _logit(d[f"{k}_injured_frac"], a["logit_eps"])
    d["tubular_axis"] = (_z(d.pt_injured_logit) + _z(d.tal_injured_logit)) / 2
    genes = markers.loc[markers.marker_set == a["ecgc_marker_set"], "gene"].tolist()
    for name, states in [("ecgc_axis", a["ecgc_states"]), ("ecgc_axis_all", a["ecgc_states_sensitivity"])]:
        m = state_matrix(pb, cnt, states, a["min_cells"])
        g = [x for x in genes if x in m.columns]
        d[name] = ((m[g] - m[g].mean()) / m[g].std()).mean(1).reindex(d.index)
        d[f"n_{name}_cells"] = cnt[[s for s in states if s in cnt.columns]].sum(1)
        d[f"{name}_n_genes"] = len(g)
    return d


# ---------- 嵌套交叉验证岭回归（核形式，一次特征分解覆盖全部 alpha） ----------
def _std(Xtr: np.ndarray, Xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _ridge_all_alpha(Xtr, ytr, Xte, alphas) -> np.ndarray:
    """返回 len(alphas) × n_te 预测。"""
    A, B = _std(Xtr, Xte)
    ym = ytr.mean()
    w, V = np.linalg.eigh(A @ A.T)
    proj = V.T @ (ytr - ym)
    Kte = B @ A.T @ V
    return np.stack([Kte @ (proj / (w + al)) + ym for al in alphas])


def fit_predict(Xtr, ytr, Xte, alphas, inner_folds, rng) -> np.ndarray:
    """内层 K 折选 alpha（训练集内），再在整个训练集上拟合并预测 Xte。"""
    n = len(ytr)
    k = min(inner_folds, n)
    fold = rng.permutation(np.arange(n) % k)
    err = np.zeros(len(alphas))
    for i in range(k):
        tr, te = fold != i, fold == i
        err += ((_ridge_all_alpha(Xtr[tr], ytr[tr], Xtr[te], alphas) - ytr[te]) ** 2).sum(1)
    best = int(np.argmin(err))
    return _ridge_all_alpha(Xtr, ytr, Xte, [alphas[best]])[0], alphas[best]


def cv_predict(X: dict[str, pd.DataFrame], y: pd.Series, cv: dict, seed: int) -> pd.Series:
    """参与者水平重复 K 折。每个染色单独建模（只用有该染色的训练参与者），
    测试参与者的预测 = 其可用染色预测的平均；重复间再平均。"""
    pids = y.index.to_numpy()
    preds = np.zeros((cv["repeats"], len(pids)))
    for r in range(cv["repeats"]):
        rng = np.random.default_rng(seed + r)
        fold = rng.permutation(np.arange(len(pids)) % cv["outer_folds"])
        acc, cnt = np.zeros(len(pids)), np.zeros(len(pids))
        for i in range(cv["outer_folds"]):
            for E in X.values():
                has = np.isin(pids, E.index)
                tr, te = has & (fold != i), has & (fold == i)
                if te.sum() == 0:
                    continue
                p, _ = fit_predict(E.loc[pids[tr]].values, y.values[tr], E.loc[pids[te]].values,
                                   cv["ridge_alphas"], cv["inner_folds"], rng)
                acc[te] += p
                cnt[te] += 1
        preds[r] = acc / cnt
    return pd.Series(preds.mean(0), index=y.index)


def cv_with_permutation(X, y, cv, seed) -> tuple[pd.Series, dict]:
    from joblib import Parallel, delayed
    pred = cv_predict(X, y, cv, seed)
    r, rho = stats.pearsonr(pred, y)[0], stats.spearmanr(pred, y)[0]
    rng = np.random.default_rng(seed)
    perms = [pd.Series(rng.permutation(y.values), index=y.index) for _ in range(cv["n_perm"])]

    def one(yp):
        p = cv_predict(X, yp, cv, seed)
        return stats.pearsonr(p, yp)[0], stats.spearmanr(p, yp)[0]
    null = np.array(Parallel(n_jobs=cv["n_jobs"])(delayed(one)(yp) for yp in perms))
    return pred, {"n": len(y), "cv_pearson_r": r, "cv_spearman_rho": rho,
                  "perm_p_pearson": float((1 + (null[:, 0] >= r).sum()) / (len(null) + 1)),
                  "perm_p_spearman": float((1 + (null[:, 1] >= rho).sum()) / (len(null) + 1)),
                  "null_mean_r": float(null[:, 0].mean()), "n_perm": len(null)}


def fit_full_predict(X: dict[str, pd.DataFrame], y: pd.Series, new: list[str], cv: dict, seed: int) -> pd.DataFrame:
    """在全部训练参与者上拟合（内层 CV 选 alpha），预测新参与者；返回各染色预测与平均。"""
    rng = np.random.default_rng(seed)
    out = pd.DataFrame(index=new)
    for s, E in X.items():
        tr = [p for p in y.index if p in E.index]
        te = [p for p in new if p in E.index]
        p, al = fit_predict(E.loc[tr].values, y.loc[tr].values, E.loc[te].values, cv["ridge_alphas"],
                            cv["inner_folds"], rng)
        out.loc[te, f"pred_{s}"] = p
        out.attrs[f"alpha_{s}"] = al
    out["pred"] = out[[c for c in out.columns if c.startswith("pred_")]].mean(1)
    return out


# ---------- 统计 ----------
def partial_corr(x: pd.Series, y: pd.Series, covar: pd.DataFrame) -> tuple[float, float, int]:
    d = pd.concat([x.rename("x"), y.rename("y"), covar], axis=1).dropna()
    C = sm.add_constant(d.drop(columns=["x", "y"]).astype(float), has_constant="add")
    rx = sm.OLS(d.x, C).fit().resid
    ry = sm.OLS(d.y, C).fit().resid
    r, p = stats.pearsonr(rx, ry)
    k = C.shape[1] - 1
    t = r * np.sqrt((len(d) - 2 - k) / (1 - r ** 2))
    return float(r), float(2 * stats.t.sf(abs(t), len(d) - 2 - k)), len(d)


def egfr_mid(s) -> float:
    m = re.match(r"(\d+)-(\d+)", str(s))
    return (int(m.group(1)) + int(m.group(2)) + 1) / 2 if m else np.nan


def diag_dummies(diag: pd.Series) -> pd.DataFrame:
    """诊断协变量：DKD / HKD / other；参照 = 人数最多的水平；人数 < 3 的水平并入 other。"""
    vc = diag.value_counts()
    keep = [k for k in vc.index if vc[k] >= 3]
    dd = pd.get_dummies(diag.where(diag.isin(keep), "other"), dtype=float)
    return dd.drop(columns=[dd.sum().idxmax()]) if dd.shape[1] > 1 else dd.iloc[:, :0]


def clinical_models(axis: pd.Series, clin: pd.DataFrame, covars: pd.DataFrame) -> list[dict]:
    """axis（z 标准化后）与白蛋白尿（KDIGO A1–A3 有序 logit；5 档 Spearman）和 eGFR（分档中点 OLS；Spearman）。"""
    rows = []
    z = _z(axis).rename("axis")
    for outcome, kind in [("alb_A", "ordinal"), ("alb_ord", "spearman"), ("egfr", "ols"), ("egfr", "spearman")]:
        parts = [z, clin[outcome].rename("out")] + ([covars] if covars.shape[1] else [])
        d = pd.concat(parts, axis=1).dropna()
        if len(d) < 10 or d.out.nunique() < 2:
            continue
        cov = d.drop(columns=["axis", "out"])
        cov = cov.loc[:, cov.std() > 0]
        base = {"outcome": outcome, "model": kind, "n": len(d), "covariates": "+".join(cov.columns) or "none"}
        if kind == "spearman":
            rho, p = stats.spearmanr(d.axis, d.out)
            rows.append({**base, "covariates": "none", "estimate": rho, "ci_low": np.nan, "ci_high": np.nan,
                         "p": p, "estimate_type": "spearman_rho"})
        elif kind == "ols":
            f = sm.OLS(d.out, sm.add_constant(pd.concat([d.axis, cov], axis=1))).fit()
            ci = f.conf_int().loc["axis"]
            rows.append({**base, "estimate": f.params["axis"], "ci_low": ci[0], "ci_high": ci[1],
                         "p": f.pvalues["axis"], "estimate_type": "eGFR_change_per_SD"})
        else:
            f = OrderedModel(d.out.astype(int), pd.concat([d.axis, cov], axis=1), distr="logit").fit(
                method="bfgs", disp=False, maxiter=2000)
            ci = f.conf_int().loc["axis"]
            rows.append({**base, "estimate": np.exp(f.params["axis"]), "ci_low": np.exp(ci[0]),
                         "ci_high": np.exp(ci[1]), "p": f.pvalues["axis"],
                         "estimate_type": "odds_ratio_per_SD_higher_albuminuria_category"})
    return rows
