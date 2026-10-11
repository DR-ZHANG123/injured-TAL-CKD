"""SMR 与 HEIDI（Zhu et al. 2016 Nat Genet）的实现。

SMR：以基因的最强 cis-eQTL 为工具变量，T = z_gwas^2 z_eqtl^2 / (z_gwas^2 + z_eqtl^2) ~ chi2(1)，
b_xy = b_gwas / b_eqtl。HEIDI：区域内其他 cis-eQTL SNP 的 b_xy 与工具 SNP 的差值 d_i，
用 delta 法给出 d 的协方差（依赖 SNP 间 LD），统计量 sum(d_i^2 / var(d_i)) 服从加权 chi2，
P 值用 Imhof 数值积分（远尾用鞍点近似）。两组样本视为不重叠。
"""
from __future__ import annotations

import numpy as np
import warnings

from scipy import integrate, optimize, stats


def smr_test(b_zy: float, se_zy: float, b_zx: float, se_zx: float) -> dict:
    z_zy, z_zx = b_zy / se_zy, b_zx / se_zx
    t = z_zy ** 2 * z_zx ** 2 / (z_zy ** 2 + z_zx ** 2)
    b_xy = b_zy / b_zx
    se_xy = abs(b_xy) / np.sqrt(t) if t > 0 else np.nan
    return {"b_smr": b_xy, "se_smr": se_xy, "p_smr": float(stats.chi2.sf(t, 1))}


def saddlepoint_sf(q: float, lam: np.ndarray) -> float:
    """加权 chi2 的上尾概率，Lugannani–Rice 鞍点近似（Kuonen 1999），用于远尾。"""
    lam = lam[lam > 1e-10]
    k = lambda t: -0.5 * np.sum(np.log(1 - 2 * lam * t))
    k1 = lambda t: np.sum(lam / (1 - 2 * lam * t))
    k2 = lambda t: np.sum(2 * lam ** 2 / (1 - 2 * lam * t) ** 2)
    hi = 0.5 / lam.max() - 1e-12
    if q <= np.sum(lam):
        return np.nan
    t = optimize.brentq(lambda x: k1(x) - q, 0.0, hi * (1 - 1e-12), xtol=1e-14)
    w = np.sign(t) * np.sqrt(2 * (t * q - k(t)))
    v = t * np.sqrt(k2(t))
    return float(stats.norm.sf(w) + stats.norm.pdf(w) * (1 / v - 1 / w))


def weighted_chi2_sf(q: float, lam: np.ndarray) -> float:
    """Imhof 积分；积分不收敛或结果 < 1e-4 时改用鞍点近似。"""
    with warnings.catch_warnings():
        warnings.simplefilter("error", integrate.IntegrationWarning)
        try:
            p = imhof_sf(q, lam)
            ok = True
        except integrate.IntegrationWarning:
            ok = False
    if not ok or p < 1e-4:
        sp = saddlepoint_sf(q, lam)
        if np.isfinite(sp):
            return sp
    return p


def imhof_sf(q: float, lam: np.ndarray) -> float:
    """P(sum lam_i chi2_1 > q)，Imhof (1961) 数值积分。"""
    lam = lam[lam > 1e-10]
    if q <= 0:
        return 1.0

    def f(u):
        theta = 0.5 * np.sum(np.arctan(lam * u)) - 0.5 * q * u
        rho = np.prod((1.0 + (lam * u) ** 2) ** 0.25)
        return np.sin(theta) / (u * rho)

    val, _ = integrate.quad(f, 0, np.inf, limit=500)
    return float(min(max(0.5 + val / np.pi, 0.0), 1.0))


def heidi_test(b_zy, se_zy, b_zx, se_zx, ld, top: int) -> dict:
    """b_*/se_* 为同一组 SNP（已对齐到同一等位基因），ld 为带符号的相关矩阵，top 为工具 SNP 下标。"""
    b_zy, se_zy, b_zx, se_zx = map(np.asarray, (b_zy, se_zy, b_zx, se_zx))
    bxy = b_zy / b_zx
    # delta 法：cov(bxy_i, bxy_j)
    a = se_zy[:, None] * se_zy[None, :] / (b_zx[:, None] * b_zx[None, :])
    c = se_zx[:, None] * se_zx[None, :] * b_zy[:, None] * b_zy[None, :] / (b_zx[:, None] ** 2 * b_zx[None, :] ** 2)
    cov = ld * (a + c)
    idx = np.array([i for i in range(len(bxy)) if i != top])
    d = bxy[idx] - bxy[top]
    cd = cov[np.ix_(idx, idx)] - cov[idx, top][:, None] - cov[top, idx][None, :] + cov[top, top]
    vd = np.diag(cd)
    keep = vd > 0
    d, cd, vd = d[keep], cd[np.ix_(keep, keep)], vd[keep]
    if len(d) < 1:
        return {"n_heidi": 0, "p_heidi": np.nan}
    tstat = float(np.sum(d ** 2 / vd))
    corr = cd / np.sqrt(np.outer(vd, vd))
    lam = np.linalg.eigvalsh(corr)
    return {"n_heidi": int(len(d)), "p_heidi": weighted_chi2_sf(tstat, lam)}


def select_heidi_snps(p_eqtl: np.ndarray, r_with_top: np.ndarray, ld: np.ndarray, top: int,
                      p_lower: float, ld_lower: float, ld_upper: float, max_snps: int) -> list[int]:
    """按 SMR 默认：eQTL P < p_lower，与工具 SNP 的 r^2 介于 (ld_lower, ld_upper)，
    彼此 r^2 < ld_upper 贪婪去冗余（按 eQTL P 排序），最多 max_snps 个（含工具 SNP）。"""
    r2t = r_with_top ** 2
    cand = [i for i in np.argsort(p_eqtl) if i != top and p_eqtl[i] < p_lower and ld_lower < r2t[i] < ld_upper]
    chosen = [top]
    for i in cand:
        if all(ld[i, j] ** 2 < ld_upper for j in chosen):
            chosen.append(i)
        if len(chosen) >= max_snps:
            break
    return chosen
