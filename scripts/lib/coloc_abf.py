"""coloc.abf（Giambartolomei et al. 2014）的 Python 实现，公式与 R 包 coloc v5 的 coloc.abf 一致。

输入两组同一批 SNP 的 beta、varbeta（已对齐到同一效应等位基因）、MAF、N。
定量性状 W = sd_prior * sdY；sdY 未知时按 coloc::sdY.est 由 varbeta、MAF、N 估计。
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def sdy_est(varbeta: np.ndarray, maf: np.ndarray, n: np.ndarray) -> float:
    """coloc::sdY.est：对 2N·MAF·(1-MAF) ~ 1/varbeta 做过原点回归，sdY = sqrt(斜率)。"""
    oneover = 1.0 / varbeta
    nvx = 2.0 * n * maf * (1.0 - maf)
    coef = np.sum(oneover * nvx) / np.sum(oneover * oneover)
    return float(np.sqrt(coef)) if coef > 0 else 1.0


def labf(beta: np.ndarray, varbeta: np.ndarray, sd_prior: float) -> np.ndarray:
    """Wakefield 近似贝叶斯因子（对数）。"""
    z2 = beta ** 2 / varbeta
    r = sd_prior ** 2 / (sd_prior ** 2 + varbeta)
    return 0.5 * np.log(1.0 - r) + 0.5 * r * z2


def _logdiff(a: float, b: float) -> float:
    m = max(a, b)
    return m + np.log(np.exp(a - m) - np.exp(b - m))


def combine_abf(l1: np.ndarray, l2: np.ndarray, p1: float, p2: float, p12: float) -> np.ndarray:
    """返回 PP.H0..PP.H4（与 coloc:::combine.abf 相同）。"""
    lsum = l1 + l2
    lH0 = 0.0
    lH1 = np.log(p1) + logsumexp(l1)
    lH2 = np.log(p2) + logsumexp(l2)
    lH3 = np.log(p1) + np.log(p2) + _logdiff(logsumexp(l1) + logsumexp(l2), logsumexp(lsum))
    lH4 = np.log(p12) + logsumexp(lsum)
    v = np.array([lH0, lH1, lH2, lH3, lH4])
    return np.exp(v - logsumexp(v))


def coloc_abf(b1, v1, maf1, n1, b2, v2, maf2, n2, p1=1e-4, p2=1e-4, p12=1e-5, sd_prior=0.15,
              sdy1=None, sdy2=None) -> dict:
    """两个定量性状的 coloc.abf；maf/N 分别取自各自数据集。"""
    b1, v1, maf1, n1 = map(np.asarray, (b1, v1, maf1, n1))
    b2, v2, maf2, n2 = map(np.asarray, (b2, v2, maf2, n2))
    sdy1 = sdy_est(v1, maf1, n1) if sdy1 is None else sdy1
    sdy2 = sdy_est(v2, maf2, n2) if sdy2 is None else sdy2
    l1 = labf(b1, v1, sd_prior * sdy1)
    l2 = labf(b2, v2, sd_prior * sdy2)
    pp = combine_abf(l1, l2, p1, p2, p12)
    lsum = l1 + l2
    snp_pp4 = np.exp(lsum - logsumexp(lsum))
    return {"nsnps": len(b1), "PP.H0": pp[0], "PP.H1": pp[1], "PP.H2": pp[2], "PP.H3": pp[3],
            "PP.H4": pp[4], "sdY1": sdy1, "sdY2": sdy2, "snp_pp4": snp_pp4, "labf1": l1, "labf2": l2}


def pp4_with_prior(l1: np.ndarray, l2: np.ndarray, p1: float, p2: float, p12: float) -> float:
    return float(combine_abf(l1, l2, p1, p2, p12)[4])
