"""Supplementary Fig. S4：GSE286184 Xenium 原位——损伤 TAL 细胞的 UMOD、微环境与比例。

只读 results/10f_xenium_umod/{donor_umod,umod_paired_tests}.tsv 与
results/10e_xenium_tal/{donor_tal,tal_paired_tests,tal_fraction_tests}.tsv；
绘图输入写到 figures/source_data/FigS4_*.tsv。输出 figures/FigS4.pdf/.png。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from FigS_common import GROUP_COL, LEGEND_KW, RES, SRC, fmt_p, panel_letters, save

GROUPS = ["CTRL", "DKD", "PGD"]
GNAME = {"CTRL": "Control", "DKD": "DKD", "PGD": "Primary GN"}
BATCH_MK = {"run_cols11": "o", "run_cols13": "s"}
FRAC = "frac_injured_of_TAL_depth_band"


def donors() -> pd.DataFrame:
    return pd.read_csv(RES / "10e_xenium_tal" / "donor_tal.tsv", sep="\t")


def panel_fraction(ax) -> None:
    d = donors().dropna(subset=[FRAC])
    t = pd.read_csv(RES / "10e_xenium_tal" / "tal_fraction_tests.tsv", sep="\t")
    t = t[t.measure == FRAC]
    rng = np.random.default_rng(3)
    rows = []
    for i, g in enumerate(GROUPS):
        q = d[d.group == g]
        jit = rng.uniform(-0.17, 0.17, len(q))
        for b, mk in BATCH_MK.items():
            sel = (q.processing_batch == b).values
            ax.scatter(i + jit[sel], q.loc[sel, FRAC], s=16, marker=mk, color=GROUP_COL[g],
                       edgecolor="white", linewidth=0.3, zorder=3)
        ax.hlines(q[FRAC].median(), i - 0.3, i + 0.3, color="black", lw=1.0, zorder=4)
        rows += [{"gsm": s, "group": g, "processing_batch": b, FRAC: v, "x": i + j}
                 for s, b, v, j in zip(q.gsm, q.processing_batch, q[FRAC], jit)]
    pd.DataFrame(rows).to_csv(SRC / "FigS4_d_injured_tal_fraction.tsv", sep="\t", index=False)
    ymax, span = d[FRAC].max(), d[FRAC].max() - d[FRAC].min()
    for k, (a, b, i, j) in enumerate((("DKD", "CTRL", 0, 1), ("DKD", "PGD", 1, 2))):
        r = t[(t.group_a == a) & (t.group_b == b)].iloc[0]
        y = ymax + span * (0.10 + 0.20 * k)
        # 跨处理批次的比较用虚线括号
        ax.plot([i, i, j, j], [y - span * 0.03, y, y, y - span * 0.03], color="#333333", lw=0.6,
                ls="-" if b == "CTRL" else (0, (2, 1.5)))
        ax.text((i + j) / 2, y + span * 0.02, fmt_p(r.p_mannwhitney), ha="center", va="bottom", fontsize=5.8)
    ax.set_ylim(top=ymax + span * 0.55)
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels([GNAME[g] for g in GROUPS], rotation=30, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.6, len(GROUPS) - 0.4)
    ax.set_ylabel("Injured TAL cells\n(fraction of TAL)", fontweight="bold")


def paired(ax, name: str, ylabel: str, tag: str) -> None:
    d = donors()
    inj, nor = f"{name}_nb_injured", f"{name}_nb_normal"
    d = d.dropna(subset=[inj, nor])
    t = pd.read_csv(RES / "10e_xenium_tal" / "tal_paired_tests.tsv", sep="\t")
    r = t[(t.neighbour == name) & (t.donors == "all")].iloc[0]
    d[["gsm", "group", "processing_batch", nor, inj]].to_csv(SRC / f"FigS4_{tag}.tsv", sep="\t", index=False)
    for _, row in d.iterrows():
        ax.plot([0, 1], [row[nor], row[inj]], color=GROUP_COL[row.group], lw=0.7, alpha=0.8,
                marker=BATCH_MK[row.processing_batch], ms=3.2, mec="white", mew=0.3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Healthy TAL", "Injured TAL"])
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.text(0.5, 1.02, f"{int(r.n_positive)}/{int(r.n)} donors higher, {fmt_p(r.p_wilcoxon)}",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=6)


def panel_umod(ax) -> None:
    d = pd.read_csv(RES / "10f_xenium_umod" / "donor_umod.tsv", sep="\t").dropna(subset=["umod_injured"])
    t = pd.read_csv(RES / "10f_xenium_umod" / "umod_paired_tests.tsv", sep="\t")
    r = t[(t.measure == "umod_diff") & (t.donors == "all")].iloc[0]
    d[["gsm", "group", "processing_batch", "umod_healthy", "umod_injured"]].to_csv(
        SRC / "FigS4_a_umod.tsv", sep="\t", index=False)
    for _, row in d.iterrows():
        ax.plot([0, 1], [row.umod_healthy, row.umod_injured], color=GROUP_COL[row.group], lw=0.7, alpha=0.8,
                marker=BATCH_MK[row.processing_batch], ms=3.2, mec="white", mew=0.3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Healthy TAL", "Injured TAL"])
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel("UMOD transcripts per 100\nother panel transcripts (median)", fontweight="bold")
    ax.text(0.5, 1.02, f"{int(r.n_lower)}/{int(r.n)} donors lower, {fmt_p(r.p_wilcoxon)}",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=6)


def main() -> None:
    SRC.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 4, figsize=(6.6, 2.9), gridspec_kw=dict(wspace=0.95, left=0.08, right=0.99,
                                                                      top=0.86, bottom=0.30))
    panel_umod(axs[0])
    paired(axs[1], "fibroblast", "TAL cells with a fibroblast\nwithin 50 µm (fraction)", "b_fibroblast_within_50um")
    paired(axs[2], "immune", "TAL cells with an immune cell\nwithin 50 µm (fraction)", "c_immune_within_50um")
    panel_fraction(axs[3])
    handles = [Line2D([], [], marker="o", ls="", color="#666666", ms=4, label="Processing run 1 (control, DKD)"),
               Line2D([], [], marker="s", ls="", color="#666666", ms=4, label="Processing run 2 (primary GN)")]
    handles += [Patch(color=GROUP_COL[g], label=GNAME[g]) for g in GROUPS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=6,
               handlelength=1.2, columnspacing=1.2, **LEGEND_KW)
    panel_letters(fig, [list(zip("abcd", axs))])
    save(fig, "FigS4")


if __name__ == "__main__":
    main()
