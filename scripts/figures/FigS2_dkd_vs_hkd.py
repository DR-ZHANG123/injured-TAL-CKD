"""Supplementary Fig 5：DKD 与高血压肾病（HKD）——损伤小管状态比例（含 eGFR 校正）与细胞状态内表达差异的一致性。

只读 results/04_kpmp_dkd_vs_hkd/{abundance,concordance}.tsv、results/04b_kpmp_adjusted/abundance_adjusted.tsv、
results/08_gse211785_replication/{abundance,concordance}.tsv；绘图输入写到 figures/source_data/FigS2_*.tsv。
输出 figures/FigS2.pdf/.png。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

from FigS_common import GROUP_COL, LEGEND_KW, RES, SRC, fmt_p, panel_letters, save

MEAS = {"PT_injured_of_PT": "Injured PT", "frPT_of_PT": "Failed-repair PT", "TAL_injured_of_TAL": "Injured TAL"}
CONTR = [("DKD_vs_REF", "DKD vs reference", GROUP_COL["DKD"]), ("HKD_vs_REF", "HKD vs reference", GROUP_COL["HKD"]),
         ("DKD_vs_HKD", "DKD vs HKD", "#555555")]
MOD_MK = {"snRNA": "o", "scRNA": "^"}
STATE = {"PT_normal": "Normal PT", "PT_injured": "Injured PT", "aPT": "Adaptive PT", "frPT": "Failed-repair PT",
         "TAL_normal": "Normal TAL", "TAL_injured": "Injured TAL"}


def panel_a(ax) -> None:
    a = pd.read_csv(RES / "04_kpmp_dkd_vs_hkd" / "abundance.tsv", sep="\t")
    a = a[a.measure.isin(MEAS)]
    a.to_csv(SRC / "FigS2_a_state_fractions.tsv", sep="\t", index=False)
    meas = list(MEAS)
    for i, m in enumerate(meas):
        for k, (cn, lab, col) in enumerate(CONTR):
            for mod, dx in (("snRNA", -0.09), ("scRNA", 0.09)):
                r = a[(a.measure == m) & (a.contrast == cn) & (a.modality == mod)]
                if r.empty:
                    continue
                r = r.iloc[0]
                y = i + (k - 1) * 0.27 + dx
                ax.scatter(r.hedges_g_logit, y, s=15, marker=MOD_MK[mod], color=col, edgecolor="white",
                           linewidth=0.3, zorder=3, alpha=1.0 if r.p_mannwhitney < 0.05 else 0.45)
        if i:
            ax.axhline(i - 0.5, color="#EEEEEE", lw=0.6, zorder=0)
    ax.axvline(0, color="#555555", lw=0.6, ls="--")
    ax.set_yticks(range(len(meas)))
    ax.set_yticklabels([MEAS[m] for m in meas])
    ax.set_ylim(len(meas) - 0.5, -0.5)
    ax.set_xlabel("Hedges' g of state fraction (logit)", fontweight="bold")
    h = [Line2D([], [], marker="s", ls="", color=c, ms=4, label=l) for _, l, c in CONTR]
    h += [Line2D([], [], marker=mk, ls="", color="#888888", ms=4, label=f"KPMP {m}") for m, mk in MOD_MK.items()]
    ax.legend(handles=h, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=6, handletextpad=0.2, **LEGEND_KW)


def panel_b(ax) -> None:
    d = pd.read_csv(RES / "04b_kpmp_adjusted" / "abundance_adjusted.tsv", sep="\t")
    # OLS 系数的 95% CI 由系数与 P 值反推（t 分布，df = n − 参数数）
    k = np.where(d.model.str.contains("egfr"), 3, 2)
    df_ = d.n_dkd + d.n_hkd - k
    t = stats.t.isf(d.p / 2, df_)
    se = d.coef_DKD_minus_HKD.abs() / t
    q = stats.t.ppf(0.975, df_)
    d["ci_low"], d["ci_high"] = d.coef_DKD_minus_HKD - q * se, d.coef_DKD_minus_HKD + q * se
    d.to_csv(SRC / "FigS2_b_egfr_adjusted.tsv", sep="\t", index=False)
    meas = list(dict.fromkeys(d.measure))
    for i, m in enumerate(meas):
        for j, (model, col, lab) in enumerate((("y ~ lab", "#8C8C8C", "Diagnosis only"),
                                                 ("y ~ lab + egfr", "#C44E52", "Adjusted for eGFR"))):
            r = d[(d.measure == m) & (d.model == model)].iloc[0]
            y = i + (j - 0.5) * 0.32
            ax.errorbar(r.coef_DKD_minus_HKD, y, xerr=[[r.coef_DKD_minus_HKD - r.ci_low], [r.ci_high - r.coef_DKD_minus_HKD]],
                        fmt="o", ms=3.8, color=col, mec="white", mew=0.3, elinewidth=0.8, capsize=0,
                        label=lab if i == 0 else None)
            ax.text(1.02, y, f"P = {r.p:.3f}", transform=ax.get_yaxis_transform(), va="center", fontsize=5.8)
    r0 = d.iloc[0]
    ax.axvline(0, color="#555555", lw=0.6, ls="--")
    ax.set_yticks(range(len(meas)))
    ax.set_yticklabels([MEAS[m] for m in meas])
    ax.set_ylim(len(meas) + 0.15, -0.75)
    ax.set_xlabel("DKD − HKD difference\n(logit state fraction, KPMP snRNA)", fontweight="bold")
    ax.legend(loc="lower right", ncol=2, fontsize=6, handletextpad=0.2, columnspacing=0.8, **LEGEND_KW)
    ax.set_title(f"DKD n={int(r0.n_dkd)}, HKD n={int(r0.n_hkd)}; median eGFR {r0.median_egfr_dkd:g} vs "
                 f"{r0.median_egfr_hkd:g}", fontsize=6.2, pad=4)


def panel_c(ax) -> None:
    a = pd.read_csv(RES / "08_gse211785_replication" / "abundance.tsv", sep="\t")
    a.to_csv(SRC / "FigS2_c_gse211785_fraction.tsv", sep="\t", index=False)
    lab = {"DKD_vs_HKD": ("DKD vs HKD", "#555555"), "DKD_vs_Control": ("DKD vs control", GROUP_COL["DKD"]),
           "HKD_vs_Control": ("HKD vs control", GROUP_COL["HKD"])}
    for i, (_, r) in enumerate(a.iterrows()):
        name, col = lab[r.contrast]
        ax.barh(i, r.coef_logit, height=0.6, color=col, edgecolor="white", linewidth=0.3)
        ax.text(r.coef_logit + 0.06, i, f"{int(r.n_a)} vs {int(r.n_b)}, {fmt_p(r.p)}", va="center", fontsize=5.8)
    ax.axvline(0, color="#555555", lw=0.6)
    ax.set_yticks(range(len(a)))
    ax.set_yticklabels([lab[c][0] for c in a.contrast])
    ax.set_ylim(len(a) - 0.5, -0.5)
    ax.set_xlim(0, 3.6)
    ax.set_xlabel("Difference in injured PT fraction\n(logit, GSE211785)", fontweight="bold")


def concordance() -> pd.DataFrame:
    k = pd.read_csv(RES / "04_kpmp_dkd_vs_hkd" / "concordance.tsv", sep="\t").assign(dataset="KPMP scRNA")
    g = pd.read_csv(RES / "08_gse211785_replication" / "concordance.tsv", sep="\t").assign(dataset="GSE211785")
    return pd.concat([k, g], ignore_index=True)


def panel_d(ax) -> None:
    c = concordance()
    c.to_csv(SRC / "FigS2_d_expression_concordance.tsv", sep="\t", index=False)
    states = list(STATE)
    col = {"KPMP scRNA": "#4C72B0", "GSE211785": "#55A868"}
    for j, (ds, dx) in enumerate((("KPMP scRNA", -0.15), ("GSE211785", 0.15))):
        q = c[c.dataset == ds].set_index("state")
        for i, s in enumerate(states):
            if s not in q.index:
                continue
            r = q.loc[s]
            ax.scatter(r.spearman, i + dx, s=18, color=col[ds], edgecolor="white", linewidth=0.3, zorder=3,
                       label=ds if i == 0 else None)
            null = r.get("spearman_null_mean")
            ax.scatter(0 if pd.isna(null) else null, i + dx, s=10, marker="|", color="#777777", zorder=2)
            if r.p_spearman < 0.05:
                ax.text(r.spearman + 0.012, i + dx, f"P = {r.p_spearman:.3f}", va="center", fontsize=5.6)
    ax.axvline(0, color="#555555", lw=0.6, ls="--")
    ax.set_yticks(range(len(states)))
    ax.set_yticklabels([STATE[s] for s in states])
    ax.set_ylim(len(states) - 0.5, -0.5)
    ax.set_xlim(-0.1, 0.3)
    ax.set_xlabel("Spearman ρ of DKD − HKD gene effects\n(KPMP snRNA vs replication)", fontweight="bold")
    h = [Line2D([], [], marker="o", ls="", color=v, ms=4, label=k) for k, v in col.items()]
    h.append(Line2D([], [], marker="|", ls="", color="#777777", ms=6, label="Permutation mean"))
    ax.legend(handles=h, loc="center right", fontsize=6, handletextpad=0.2, **LEGEND_KW)


def panel_e(ax) -> None:
    c = concordance()
    states = list(STATE)
    col = {"KPMP scRNA": "#4C72B0", "GSE211785": "#55A868"}
    for ds, dx in (("KPMP scRNA", -0.15), ("GSE211785", 0.15)):
        q = c[c.dataset == ds].set_index("state")
        for i, s in enumerate(states):
            if s not in q.index:
                continue
            r = q.loc[s]
            ax.scatter(r.top_same_direction, i + dx, s=18, color=col[ds], edgecolor="white", linewidth=0.3, zorder=3)
            ax.scatter(r.top_null_mean, i + dx, s=10, marker="|", color="#777777", zorder=2)
            if r.p_top < 0.05:
                ax.text(r.top_same_direction + 0.015, i + dx, f"P = {r.p_top:.2f}", va="center", fontsize=5.6)
    ax.axvline(0.5, color="#555555", lw=0.6, ls="--")
    ax.set_yticks(range(len(states)))
    ax.set_yticklabels([])
    ax.set_ylim(len(states) - 0.5, -0.5)
    ax.set_xlim(0.35, 0.85)
    ax.set_xlabel("Top-200 genes with the same\nDKD − HKD direction (fraction)", fontweight="bold")


def main() -> None:
    SRC.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6.6, 6.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.55, left=0.13, right=0.97, top=0.94,
                          bottom=0.09)
    top = gs[0].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.95)
    ax_a, ax_b = fig.add_subplot(top[0]), fig.add_subplot(top[1])
    bot = gs[1].subgridspec(1, 3, width_ratios=[0.9, 1.0, 0.85], wspace=0.75)
    ax_c, ax_d, ax_e = (fig.add_subplot(bot[i]) for i in range(3))
    panel_a(ax_a)
    panel_b(ax_b)
    panel_c(ax_c)
    panel_d(ax_d)
    panel_e(ax_e)
    panel_letters(fig, [[("a", ax_a), ("b", ax_b)], [("c", ax_c), ("d", ax_d), ("e", ax_e)]])
    save(fig, "FigS2")


if __name__ == "__main__":
    main()
