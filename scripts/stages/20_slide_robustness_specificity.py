"""Stage 20 第 4 部分及特异性检验：病理描述评分基线（嵌套 CV）、全部读出的样本外临床表、
置换标签读出的零分布、直接以 eGFR 训练的切片模型、通用"损伤"基线（嵌入主成分 / 随机投影）、增量价值。"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed

_spec = importlib.util.spec_from_file_location("rl", Path(__file__).with_name("20_slide_robustness_lib.py"))
rl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rl)
hl = rl.hl


# ---------- 4a/b 病理描述评分基线与嵌套增量 ----------
def _ols_pred(Dtr: pd.DataFrame, ytr: np.ndarray, Dte: pd.DataFrame) -> np.ndarray:
    med = Dtr.median()
    A, B = Dtr.fillna(med), Dte.fillna(med)
    keep = A.columns[A.std() > 0]
    f = sm.OLS(ytr, sm.add_constant(A[keep].astype(float), has_constant="add")).fit()
    return f.predict(sm.add_constant(B[keep].astype(float), has_constant="add")).values


def _slide_pred(X: dict, ytr: pd.Series, te: list, cv: dict, rng) -> np.ndarray:
    acc, cnt = np.zeros(len(te)), np.zeros(len(te))
    for E in X.values():
        tr = [p for p in ytr.index if p in E.index]
        m = np.array([p in E.index for p in te])
        if m.sum() == 0 or len(tr) < 5:
            continue
        acc[m] += hl.fit_predict(E.loc[tr].values, ytr.loc[tr].values, E.loc[np.array(te)[m]].values,
                                 cv["ridge_alphas"], cv["inner_folds"], rng)[0]
        cnt[m] += 1
    return acc / cnt


def descriptor_nested_cv(X: dict, y: pd.Series, D: pd.DataFrame, cv: dict, seed: int) -> pd.DataFrame:
    """同一外层折：描述评分 OLS；切片读出；描述评分 + 切片读出（切片读出在训练折内用内层折外预测，避免泄漏）。"""
    pids = y.index.to_numpy()
    out = {k: np.zeros((cv["repeats"], len(pids))) for k in ["descriptors", "slide", "descriptors+slide"]}
    for r in range(cv["repeats"]):
        rng = np.random.default_rng(seed + r)
        fold = rng.permutation(np.arange(len(pids)) % cv["outer_folds"])
        for i in range(cv["outer_folds"]):
            tr, te = list(pids[fold != i]), list(pids[fold == i])
            ytr = y.loc[tr]
            s_te = _slide_pred(X, ytr, te, cv, rng)
            inner = rng.permutation(np.arange(len(tr)) % cv["inner_folds"])
            s_tr = np.zeros(len(tr))
            for k in range(cv["inner_folds"]):
                itr = [p for p, f in zip(tr, inner) if f != k]
                ite = [p for p, f in zip(tr, inner) if f == k]
                s_tr[inner == k] = _slide_pred(X, ytr.loc[itr], ite, cv, rng)
            idx = fold == i
            out["descriptors"][r, idx] = _ols_pred(D.loc[tr], ytr.values, D.loc[te])
            out["slide"][r, idx] = s_te
            out["descriptors+slide"][r, idx] = _ols_pred(D.loc[tr].assign(slide=s_tr), ytr.values,
                                                         D.loc[te].assign(slide=s_te))
    return pd.DataFrame({k: v.mean(0) for k, v in out.items()}, index=y.index)


def paired_delta_r(P: pd.DataFrame, y: pd.Series, a: str, b: str, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    d = P[[a, b]].assign(y=y)
    bs = []
    for _ in range(n_boot):
        s = d.iloc[rng.integers(0, len(d), len(d))]
        bs.append(s[a].corr(s.y) - s[b].corr(s.y))
    return {"comparison": f"{a} minus {b}", "delta_r": d[a].corr(d.y) - d[b].corr(d.y),
            "ci_low": np.percentile(bs, 2.5), "ci_high": np.percentile(bs, 97.5),
            "p_boot_two_sided": 2 * min(np.mean(np.array(bs) <= 0), np.mean(np.array(bs) >= 0))}


def variance_explained(pred: pd.Series, D: pd.DataFrame, dx: pd.Series, label: str) -> list[dict]:
    rows = []
    for name, cols, use_dx in [("IF+TA", ["IF_pct", "TA_pct"], False), ("all_descriptors", list(D.columns), False),
                               ("diagnosis", [], True), ("IF+TA+diagnosis", ["IF_pct", "TA_pct"], True),
                               ("all_descriptors+diagnosis", list(D.columns), True)]:
        C = D[cols].copy()
        if use_dx:
            C = pd.concat([C, hl.diag_dummies(dx)], axis=1)
        d = pd.concat([pred.rename("p"), C], axis=1).dropna()
        f = sm.OLS(d.p, sm.add_constant(d.drop(columns="p").astype(float), has_constant="add")).fit()
        rows.append({"readout_set": label, "predictors": name, "n": len(d), "r2": f.rsquared, "adj_r2": f.rsquared_adj})
    return rows


# ---------- 特异性：全部读出的样本外临床表 ----------
def oos_all_targets(fps: dict, cl: pd.DataFrame, D: pd.DataFrame, acqcov: pd.DataFrame, oos: list) -> pd.DataFrame:
    c = cl.reindex(oos)
    dd = hl.diag_dummies(c.diagnosis)
    Dd = D.reindex(oos)
    specs = [("none", c.iloc[:, :0]), ("diagnosis", dd), ("diagnosis+IF", pd.concat([dd, Dd.IF_pct], axis=1)),
             ("diagnosis+IF+TA", pd.concat([dd, Dd[["IF_pct", "TA_pct"]]], axis=1)),
             ("diagnosis+all_descriptors", pd.concat([dd, Dd], axis=1)),
             ("diagnosis+acquisition", pd.concat([dd, acqcov.reindex(oos)], axis=1))]
    rows = []
    for target, fp in fps.items():
        for lab, cov in specs:
            for r in hl.clinical_models(fp.pred.reindex(oos), c, cov):
                if r["model"] == "spearman" and lab != "none":
                    continue
                rows.append({"target": target, "adjustment": lab, **r})
    return pd.DataFrame(rows)


# ---------- 特异性：置换标签零分布、随机投影、嵌入主成分、直接 eGFR 模型 ----------
def permuted_label_null(X: dict, y: pd.Series, oos: list, egfr: pd.Series, dd: pd.DataFrame, cv: dict,
                        seed: int, n: int) -> np.ndarray:
    def one(k):
        rng = np.random.default_rng(seed + 10_000 + k)
        yp = pd.Series(rng.permutation(y.values), index=y.index)
        fp = rl.fit_predict_tf(X, yp, oos, cv, seed + k)
        r = rl.ols_egfr(fp.pred, egfr, dd, "diagnosis")
        return r["egfr_per_sd"], r["t"], r["p"]
    return np.array(Parallel(n_jobs=cv["n_jobs"])(delayed(one)(k) for k in range(n)))


def _std_pool(X: dict, train: list) -> dict:
    """每种染色：用训练参与者的均值 / 标准差标准化嵌入。"""
    out = {}
    for s, E in X.items():
        tr = E.loc[E.index.intersection(train)]
        sd = tr.std().replace(0, 1)
        out[s] = (E - tr.mean()) / sd
    return out


def embedding_pcs(X: dict, train: list, k: int) -> pd.DataFrame:
    """训练参与者内 PCA（每种染色），PC 符号定为与训练集 tile 数无关的固定方向（载荷和为正），染色间平均。"""
    Z = _std_pool(X, train)
    parts = []
    for s, E in Z.items():
        tr = E.loc[E.index.intersection(train)].values
        _, _, Vt = np.linalg.svd(tr - tr.mean(0), full_matrices=False)
        Vt = Vt[:k] * np.sign(Vt[:k].sum(1, keepdims=True))
        parts.append(pd.DataFrame(E.values @ Vt.T, index=E.index, columns=[f"PC{j+1}" for j in range(k)]).assign(stain=s))
    return pd.concat(parts).groupby(level=0).mean(numeric_only=True)


def random_projection_null(X: dict, train: list, oos: list, egfr: pd.Series, dd: pd.DataFrame, seed: int,
                           n: int) -> np.ndarray:
    Z = _std_pool(X, train)
    dim = next(iter(Z.values())).shape[1]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        w = rng.standard_normal(dim)
        sc = pd.concat([(E.reindex(oos).dropna() @ w).rename(s) for s, E in Z.items()], axis=1).mean(1)
        r = rl.ols_egfr(sc, egfr, dd, "diagnosis")
        out.append((r["egfr_per_sd"], r["t"], r["p"]))
    return np.array(out)


def null_summary(name: str, null: np.ndarray, obs: dict) -> list[dict]:
    rows = []
    for lab, o in obs.items():
        rows.append({"null": name, "n_null": len(null), "readout": lab, "observed_egfr_per_sd": o["egfr_per_sd"],
                     "observed_t": o["t"], "null_median_abs_t": float(np.median(abs(null[:, 1]))),
                     "null_q95_abs_t": float(np.quantile(abs(null[:, 1]), .95)),
                     "p_abs_t_ge_observed": float((1 + (abs(null[:, 1]) >= abs(o["t"])).sum()) / (len(null) + 1)),
                     "p_signed_t_le_observed": float((1 + (null[:, 1] <= o["t"]).sum()) / (len(null) + 1)),
                     "null_fraction_p_lt_0.05": float((null[:, 2] < 0.05).mean())})
    return rows
