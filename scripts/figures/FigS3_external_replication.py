"""Supplementary Fig 8：损伤 TAL 基因程序的外部复现与注释迁移检验——KPMP 发现 vs GSE211785、GSE195460 的
损伤 TAL 对正常 TAL log2FC；基因对半拆分（注释与评价用互不重叠的基因）后的复现，与健康状态对（M-TAL vs C-TAL、
CNT vs DCT）的参照水平比较。

只读 results/15_key_genes/aTAL_frTAL_discovery.tsv、results/19a_gse211785_relabel/{programme_lfc,programme_concordance}.tsv、
results/19b_gse195460/{programme_lfc_all_state_labels,programme_concordance}.tsv、
results/19c_label_transfer_circularity/{runs,summary}.tsv；绘图输入写到 figures/source_data/FigS3_*.tsv。
输出 figures/FigS3.pdf/.png。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from FigS_common import RES, SRC, panel_letters, save

plt.rcParams.update({"font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"], "font.size": 6.5,
                     "axes.labelsize": 6.5, "axes.titlesize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "legend.fontsize": 5.8, "mathtext.default": "regular"})
SMALL = 5.6
FDR, LFC = 0.05, 1.0          # 与 stage 15/19 的命中定义一致（FDR < 0.05、|log2FC| ≥ 1）
KEY = {"EGF": (-1.0, 0.9), "UMOD": (0.25, -1.1), "SLC12A1": (0.3, -0.9), "MMP7": (-1.3, 0.8), "HAVCR1": (0.4, -1.0),
       "CCL2": (-1.6, 0.6), "CD44": (0.3, 0.8)}
UP, DOWN, BG = "#8C2D04", "#2C7FB8", "#D5D5D5"
PAIR_LAB = {"programme": "Injured TAL programme", "M-TAL_vs_C-TAL": "M-TAL vs C-TAL", "CNT_vs_DCT": "CNT vs DCT"}


def scatter_panel(ax, rep_file: str, conc_file: str, conc_sel: dict, title: str, tag: str) -> None:
    disc = pd.read_csv(RES / "15_key_genes/aTAL_frTAL_discovery.tsv", sep="\t", index_col=0)
    rep = pd.read_csv(RES / rep_file, sep="\t", index_col=0)
    conc = pd.read_csv(RES / conc_file, sep="\t")
    for k, v in conc_sel.items():
        conc = conc[conc[k] == v]
    c = conc.iloc[0]
    j = disc[["log2fc", "fdr"]].join(rep[["log2fc"]].rename(columns={"log2fc": "log2fc_rep"}), how="inner")
    j = j.rename(columns={"log2fc": "log2fc_discovery", "fdr": "fdr_discovery"})
    j["hit"] = np.where((j.fdr_discovery < FDR) & (j.log2fc_discovery >= LFC), "up",
                        np.where((j.fdr_discovery < FDR) & (j.log2fc_discovery <= -LFC), "down", "none"))
    j.to_csv(SRC / f"FigS3_{tag}_log2fc.tsv", sep="\t", index_label="gene")
    conc.to_csv(SRC / f"FigS3_{tag}_concordance.tsv", sep="\t", index=False)
    bg = j[j.hit == "none"]
    ax.scatter(bg.log2fc_discovery, bg.log2fc_rep, s=1.2, color=BG, linewidth=0, rasterized=True, zorder=1)
    for h, col in (("up", UP), ("down", DOWN)):
        s = j[j.hit == h]
        ax.scatter(s.log2fc_discovery, s.log2fc_rep, s=2.2, color=col, linewidth=0, alpha=0.8, rasterized=True,
                   zorder=2)
    lim = (-6, 6)
    ax.plot(lim, lim, color="#888888", lw=0.5, ls="--", zorder=0)
    ax.axhline(0, color="#BBBBBB", lw=0.4, zorder=0)
    ax.axvline(0, color="#BBBBBB", lw=0.4, zorder=0)
    for g, (dx, dy) in KEY.items():
        if g in j.index:
            x, y = j.loc[g, "log2fc_discovery"], j.loc[g, "log2fc_rep"]
            ax.scatter(x, y, s=9, facecolor="none", edgecolor="black", linewidth=0.6, zorder=3)
            ax.annotate(g, (x, y), xytext=(x + dx, y + dy), fontsize=SMALL, fontstyle="italic", zorder=4,
                        arrowprops=dict(arrowstyle="-", lw=0.4, color="black", shrinkA=0, shrinkB=2))
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("KPMP discovery log2FC (injured vs healthy TAL)")
    ax.set_ylabel(f"{title} log2FC")
    ax.text(0.03, 0.97, f"r = {c.lfc_pearson_r:.2f} ({int(c.n_genes_common):,} genes)\n"
            f"Replicated: {c.rep_rate:.0%} of {int(c.n_hits):,} key genes\n{int(c.n_donors)} donors",
            transform=ax.transAxes, ha="left", va="top", fontsize=SMALL, linespacing=1.15)
    ax.set_title(title, pad=4)


def panel_c(ax_r, ax_q) -> None:
    runs = pd.read_csv(RES / "19c_label_transfer_circularity/runs.tsv", sep="\t")
    summ = pd.read_csv(RES / "19c_label_transfer_circularity/summary.tsv", sep="\t")
    runs.to_csv(SRC / "FigS3_c_split_half_runs.tsv", sep="\t", index=False)
    summ.to_csv(SRC / "FigS3_c_split_half_summary.tsv", sep="\t", index=False)
    order = [("GSE211785", "programme"), ("GSE211785", "M-TAL_vs_C-TAL"),
             ("GSE195460", "programme"), ("GSE195460", "CNT_vs_DCT"),
             ("GSE271188", "programme"), ("GSE271188", "M-TAL_vs_C-TAL"), ("GSE271188", "CNT_vs_DCT")]
    ys, y, prev = [], 0.0, None
    for ds, _ in order:
        if prev is not None and ds != prev:
            y += 0.6
        ys.append(y)
        prev = ds
        y += 1
    rng = np.random.default_rng(20261013)
    for ax, col, full, xl in [(ax_r, "lfc_pearson_r", "full_lfc_pearson_r", "log2FC correlation with discovery, r"),
                              (ax_q, "rep_rate", "full_rep_rate", "Replication rate of key genes")]:
        for yy, (ds, pr) in zip(ys, order):
            sp = runs[(runs.dataset == ds) & (runs.pair == pr) & (runs.run != "full")][col].values
            prog = pr == "programme"
            c = "black" if prog else "#8C8C8C"
            ax.scatter(sp, yy + rng.uniform(-0.18, 0.18, len(sp)), s=4, color=c, alpha=0.6, linewidth=0, zorder=2)
            f = summ[(summ.dataset == ds) & (summ.pair == pr)][full].iloc[0]
            ax.scatter(f, yy, s=26, marker="|", color="#C44E52" if prog else "#555555", linewidth=1.3, zorder=3)
        ax.set_yticks(ys)
        ax.set_xlabel(xl)
        ax.set_ylim(ys[-1] + 0.6, -0.6)
        ax.grid(axis="x", color="#EEEEEE", lw=0.5)
        ax.set_axisbelow(True)
    labs = []
    for ds, pr in order:
        n = int(summ[(summ.dataset == ds) & (summ.pair == pr)].n_donors.iloc[0])
        labs.append(f"{ds}: {PAIR_LAB[pr]} ({n})")
    ax_r.set_yticklabels(labs, fontsize=SMALL)
    for t, (_, pr) in zip(ax_r.get_yticklabels(), order):
        if pr == "programme":
            t.set_fontweight("bold")
    ax_q.set_yticklabels([])
    ax_r.set_xlim(0.4, 1.0)
    ax_r.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax_q.set_xlim(0.55, 0.9)
    ax_r.set_title("Split-half genes: labels and evaluation use disjoint genes", pad=4, loc="left")
    ax_q.legend(handles=[Line2D([], [], marker="o", ls="", color="black", ms=2.5, alpha=0.6, label="20 gene splits"),
                         Line2D([], [], marker="|", ls="", color="#C44E52", ms=6, mew=1.3, label="All genes")],
                loc="upper left", frameon=True, fancybox=False, edgecolor="#CCCCCC", fontsize=SMALL,
                handletextpad=0.3, borderpad=0.3, labelspacing=0.2)


def main() -> None:
    fig = plt.figure(figsize=(6.6, 6.0))
    g1 = fig.add_gridspec(1, 2, left=0.09, right=0.97, top=0.95, bottom=0.52, wspace=0.3)
    a, b = fig.add_subplot(g1[0]), fig.add_subplot(g1[1])
    g2 = fig.add_gridspec(1, 2, left=0.36, right=0.97, top=0.37, bottom=0.07, wspace=0.1)
    c1, c2 = fig.add_subplot(g2[0]), fig.add_subplot(g2[1])
    scatter_panel(a, "19a_gse211785_relabel/programme_lfc.tsv", "19a_gse211785_relabel/programme_concordance.tsv",
                  {"technology": "both", "gene_set": "all_genes"}, "GSE211785", "a")
    scatter_panel(b, "19b_gse195460/programme_lfc_all_state_labels.tsv", "19b_gse195460/programme_concordance.tsv",
                  {"labels": "all_state_labels", "gene_set": "all_genes"}, "GSE195460", "b")
    b.legend(handles=[Line2D([], [], marker="o", ls="", color=UP, ms=3, label="Key gene, up"),
                      Line2D([], [], marker="o", ls="", color=DOWN, ms=3, label="Key gene, down"),
                      Line2D([], [], marker="o", ls="", color=BG, ms=3, label="Other genes")],
             loc="lower right", frameon=True, fancybox=False, edgecolor="#CCCCCC", fontsize=SMALL,
             handletextpad=0.1, borderpad=0.3, labelspacing=0.2)
    panel_c(c1, c2)
    panel_letters(fig, [[("a", a), ("b", b)], [("c", c1)]])
    save(fig, "FigS3")


if __name__ == "__main__":
    main()
