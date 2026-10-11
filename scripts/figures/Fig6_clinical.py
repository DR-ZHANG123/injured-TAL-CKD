"""Fig 6：损伤 TAL 负担伴随较低 eGFR 与较高白蛋白尿。

只读 results/21_confounders/ 与 results/20_slide_robustness/ 的表；每个 panel 的绘图输入写到
figures/source_data/Fig6_*.tsv。参与者编号不进入图与 source data。
输出 figures/Fig6.pdf（矢量）与 figures/Fig6.png（300 dpi）；之后运行 finalize_sizes.py。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from FigS_common import LEGEND_KW, RES, SRC, fmt_p, panel_letters, save  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Liberation Sans", "Arial", "DejaVu Sans"]
R20, R21 = RES / "20_slide_robustness", RES / "21_confounders"
DX = ["REF", "DKD", "HKD"]
DX_COL = {"REF": "#8C8C8C", "DKD": "#C44E52", "HKD": "#4C72B0"}
DX_LAB = {"REF": "Reference tissue", "DKD": "DKD", "HKD": "HKD"}
INK = "#333333"
MINUS = "−"
EXPO = "TAL_injured_logit_perSD"


def pfmt(p: float) -> str:
    """P 值：< 0.001 用科学计数；< 0.1 保留三位小数；其余两位。"""
    if p < 1e-3:
        return fmt_p(p)
    return f"P = {p:.3f}" if p < 0.1 else f"P = {p:.2f}"


def num(v: float, nd: int = 1) -> str:
    return f"{v:.{nd}f}".replace("-", MINUS)


def ci(r, nd: int = 1) -> str:
    return f"{num(r.estimate, nd)} ({num(r.ci_low, nd)} to {num(r.ci_high, nd)})"


def load():
    ps = pd.read_csv(R21 / "participants_snRNA.tsv", sep="\t", dtype={"donor_id": str})
    ps = ps[ps.dx.isin(DX)]
    eg = pd.read_csv(R21 / "egfr_models.tsv", sep="\t")
    eg = eg[(eg.exposure == EXPO) & (eg.term == "expo")]
    fib = pd.read_csv(R21 / "programme_fibrosis_models.tsv", sep="\t")
    alb = pd.read_csv(R21 / "albuminuria_models.tsv", sep="\t")
    alb = alb[(alb.exposure == EXPO) & (alb.term == "expo")]
    inc = pd.read_csv(R20 / "oos_incremental_value.tsv", sep="\t")
    null = pd.read_csv(R20 / "oos_egfr_null_comparison.tsv", sep="\t")
    return ps, eg, fib, alb, inc, null


def pick(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k] == v).values
    out = df[m]
    assert len(out) == 1, (kw, len(out))
    return out.iloc[0]


def panel_a(ax, ps, eg):
    d = ps.dropna(subset=["TAL_logit", "egfr"]).copy()
    for dx in DX:
        s = d[d.dx == dx]
        ax.scatter(s.TAL_logit, s.egfr, s=13, c=DX_COL[dx], edgecolor="white", lw=0.3, label=DX_LAB[dx],
                   alpha=0.9, zorder=3)
    adj = pick(eg, subset="all", adjustment="dx+age+sex")
    ckd = pick(eg, subset="CKD", adjustment="dx+age+sex")
    assert int(adj.n) == len(d), (adj.n, len(d))
    txt = (f"eGFR per SD of injured TAL\nDiagnosis, age, sex (n = {int(adj.n)}):\n  {ci(adj)}, {pfmt(adj.p)}\n"
           f"CKD only, same (n = {int(ckd.n)}):\n  {ci(ckd)}, {pfmt(ckd.p)}")
    ax.set_ylim(0, d.egfr.max() * 1.25)
    ax.set_yticks([0, 30, 60, 90, 120, 150])
    ax.set_xlabel("Measured injured TAL\n(single-nucleus, logit fraction)", fontweight="bold")
    ax.set_ylabel("eGFR (ml/min/1.73 m$^2$)", fontweight="bold")
    ax.legend(loc="upper left", fontsize=5.8, handletextpad=0.1, borderaxespad=0.3, **LEGEND_KW)
    d = d.rename(columns={"dx": "group", "TAL_logit": "injured_tal_logit", "egfr": "egfr_bin_midpoint"})
    d["participant"] = [f"P{i:03d}" for i in range(1, len(d) + 1)]
    d[["participant", "group", "injured_tal_logit", "egfr_bin_midpoint", "age10", "male"]].to_csv(
        SRC / "Fig6_a_tal_vs_egfr.tsv", sep="\t", index=False)
    pd.DataFrame([adj, ckd]).to_csv(SRC / "Fig6_a_models.tsv", sep="\t", index=False)


def forest_rows(eg, fib, inc):
    def row(group, label, r, src):
        return {"group": group, "label": label, "n": int(r.n), "estimate": r.estimate if "estimate" in r else
                r.egfr_per_sd, "ci_low": r.ci_low, "ci_high": r.ci_high, "p": r.p, "source": src}
    fr = fib[fib.analysis.str.startswith("injured_TAL_fraction")]
    rows = [
        row("measured", "Diagnosis", pick(eg, subset="all", adjustment="dx"), "21 egfr_models all/dx"),
        row("measured", "Diagnosis, age, sex", pick(eg, subset="all", adjustment="dx+age+sex"),
            "21 egfr_models all/dx+age+sex"),
        row("measured", "CKD only: diagnosis,\nage, sex", pick(eg, subset="CKD", adjustment="dx+age+sex"),
            "21 egfr_models CKD/dx+age+sex"),
        row("measured", "CKD only: diagnosis,\nage, sex, fibrosis",
            pick(fr, subset="CKD", adjustment="dx+age+sex+IF", term="injured_TAL_logit_perSD"),
            "21 programme_fibrosis_models CKD/dx+age+sex+IF"),
    ]
    for lab, mdl in [("Diagnosis", "diagnosis"), ("Diagnosis, all\npathologist scores", "diagnosis+all_descriptors")]:
        r = pick(inc, readout="injured_TAL", model=mdl)
        rows.append({"group": "slide", "label": lab, "n": int(r.n), "estimate": r.egfr_per_sd, "ci_low": r.ci_low,
                     "ci_high": r.ci_high, "p": r.p, "source": f"20 oos_incremental_value injured_TAL/{mdl}"})
    return pd.DataFrame(rows)


def panel_b(ax, d):
    heads = {"measured": "Measured injured TAL", "slide": "Slide-estimated injured TAL"}
    ys, y, hy = [], 0.0, {}
    for g in ["measured", "slide"]:
        hy[g] = y
        y -= 1.1 if g == "measured" else 1.4
        for _ in range((d.group == g).sum()):
            ys.append(y)
            y -= 1.0
        y -= 0.5
    d = d.assign(y=ys)
    for g in ["measured", "slide"]:
        yy = d[d.group == g].y
        ax.plot([0, 0], [yy.min() - 0.5, yy.max() + 0.5], color="#999999", lw=0.6, ls="--", zorder=1)
    for _, r in d.iterrows():
        col = INK if r.group == "measured" else DX_COL["DKD"]
        sig = r.p < 0.05
        ax.plot([r.ci_low, r.ci_high], [r.y, r.y], color=col, lw=1.0, zorder=2)
        ax.scatter(r.estimate, r.y, s=18, marker="s", facecolor=col if sig else "white", edgecolor=col, lw=0.8,
                   zorder=3)
        ax.text(-0.04, r.y, f"{r.label}\n(n = {r.n})", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=5.8, linespacing=1.0)
        ax.text(1.05, r.y, pfmt(r.p),
                transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=5.8, linespacing=1.0)
    for g, yh in hy.items():
        ax.text(-1.06, yh, heads[g], transform=ax.get_yaxis_transform(), ha="left", va="top", fontsize=6.2,
                fontweight="bold", linespacing=1.05)
    yt = 0.0
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-24, 6)
    ax.set_ylim(min(ys) - 0.9, yt + 0.2)
    ax.set_xlabel("eGFR difference per SD\nof injured TAL\n(ml/min/1.73 m$^2$, 95% CI)", fontweight="bold")
    d.to_csv(SRC / "Fig6_b_forest_egfr.tsv", sep="\t", index=False)


def panel_c(ax, ps, alb):
    d = ps.dropna(subset=["TAL_logit", "alb_A"]).copy()
    cats = [1, 2, 3]
    data = [d.loc[d.alb_A == c, "TAL_logit"].values for c in cats]
    bp = ax.boxplot(data, positions=cats, widths=0.5, showfliers=False, patch_artist=True,
                    medianprops=dict(color="black", lw=0.8), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6),
                    boxprops=dict(lw=0.6))
    for p in bp["boxes"]:
        p.set_facecolor("#EEEEEE")
    rng = np.random.default_rng(0)
    d["x_jitter"] = d.alb_A + rng.uniform(-0.15, 0.15, len(d))
    for dx in DX:
        s = d[d.dx == dx]
        ax.scatter(s.x_jitter, s.TAL_logit, s=10, c=DX_COL[dx], edgecolor="white", lw=0.25, zorder=3)
    adj = pick(alb, subset="all", adjustment="dx+age+sex")
    adje = pick(alb, subset="all", adjustment="dx+age+sex+eGFR")
    assert int(adj.n) == len(d), (adj.n, len(d))
    ax.set_xticks(cats)
    ax.set_xticklabels([f"A{c}\n(n = {len(v)})" for c, v in zip(cats, data)])
    ax.set_xlabel("Albuminuria category", fontweight="bold")
    ax.set_ylabel("Measured injured TAL\n(logit fraction)", fontweight="bold")
    lo, hi = d.TAL_logit.min(), d.TAL_logit.max()
    ax.set_ylim(lo - 0.3, hi + (hi - lo) * 0.08)
    txt = (f"Odds ratio per SD (ordinal)\nDiagnosis, age, sex:\n  {adj.odds_ratio:.1f} ({adj.ci_low:.1f}–"
           f"{adj.ci_high:.1f}), {pfmt(adj.p)}\n+ eGFR:\n  {adje.odds_ratio:.1f} ({adje.ci_low:.1f}–"
           f"{adje.ci_high:.1f}), {pfmt(adje.p)}")
    d = d.rename(columns={"dx": "group", "TAL_logit": "injured_tal_logit"})
    d[["group", "alb_A", "injured_tal_logit", "x_jitter"]].to_csv(SRC / "Fig6_c_albuminuria.tsv", sep="\t",
                                                                   index=False)
    alb[alb.subset == "all"].to_csv(SRC / "Fig6_c_ordinal_models.tsv", sep="\t", index=False)


def main():
    SRC.mkdir(parents=True, exist_ok=True)
    ps, eg, fib, alb, inc, null = load()
    fig = plt.figure(figsize=(6.6, 5.6))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1.45], left=0.1, right=0.97, top=0.95, bottom=0.09,
                  wspace=0.3, hspace=0.45)
    ax_a, ax_c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    sub = gs[:, 1].subgridspec(1, 3, width_ratios=[0.33, 0.30, 0.37], wspace=0)
    ax_b = fig.add_subplot(sub[1])
    panel_a(ax_a, ps, eg)
    panel_b(ax_b, forest_rows(eg, fib, inc))
    panel_c(ax_c, ps, alb)
    # 切片读出的置换背景（同等强度的关联也见于其他细胞状态与直接 eGFR 读出）：只写入 source data
    null[null.null == "permuted_injured_TAL_labels"].to_csv(SRC / "Fig6_b_slide_permutation_context.tsv", sep="\t",
                                                            index=False)
    panel_letters(fig, [[("a", ax_a)], [("c", ax_c)]])
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    rend = fig.canvas.get_renderer()
    ya = inv.transform((0, ax_a.get_tightbbox(rend).y1))[1]
    xb = inv.transform((ax_b.get_tightbbox(rend).x0, 0))[0]
    fig.text(xb, ya + 0.008, "b", fontsize=11, fontweight="bold", ha="left", va="bottom")
    save(fig, "Fig6")


if __name__ == "__main__":
    main()
