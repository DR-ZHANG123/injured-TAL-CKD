"""Fig 4：肾功能的遗传信号落在健康 TAL 功能基因上，而不在损伤程序上。

只读 results/18_programme_genetics、results/25_sldsc 与 results/23_fig_tables_v2 的表；绘图输入写到 figures/source_data/Fig4_*.tsv。
输出 figures/Fig4.pdf（矢量）与 figures/Fig4.png（300 dpi）。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
FIG = ROOT / "figures"
SRC = FIG / "source_data"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
    "mathtext.it": "Liberation Sans:italic", "mathtext.default": "regular",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})
LEG = dict(frameon=True, fancybox=False, edgecolor="#CCCCCC", framealpha=1)
SETS = ["tal_healthy_specific", "tal_injured_down", "tal_injured_up", "pt_injured_up"]
SET_LAB = {"tal_healthy_specific": "Healthy TAL identity genes", "tal_healthy": "Healthy TAL genes", "tal_injured_down": "Lost in injured TAL",
           "tal_injured_up": "Gained in injured TAL", "pt_injured_up": "Gained in injured PT"}
SET_COL = {"tal_healthy_specific": "#2E7D4F", "tal_healthy": "#2E7D4F", "tal_injured_down": "#8CC79E", "tal_injured_up": "#C44E52",
           "pt_injured_up": "#8C8C8C"}
LDSC_TRAITS = ["egfr_EA", "egfr_noDM", "egfr_DM", "bun_EA", "uacr_EA", "ckd_EA"]
TRAITS = ["egfr_EA", "egfr_noDM", "egfr_DM", "bun_EA", "uacr_EA", "uacr_DM", "ckd_EA"]
TRAIT_LAB = {"egfr_EA": "eGFR\nall", "egfr_noDM": "eGFR\nno diabetes", "egfr_DM": "eGFR\ndiabetes",
             "uacr_EA": "UACR\nall", "uacr_DM": "UACR\ndiabetes", "bun_EA": "BUN\nall", "ckd_EA": "CKD\nall"}
EGFR = ["egfr_EA", "egfr_noDM", "egfr_DM"]
HIGHLIGHT = ["UMOD", "KNG1", "EPB41L3", "SLC47A1", "CTSS", "A4GALT", "XYLB", "DACH1", "SLC15A2", "TFDP2"]


def fmt_p(p: float) -> str:
    if p >= 0.01:
        return f"P = {p:.2f}"
    if p >= 0.001:
        return f"P = {p:.3f}"
    m, e = f"{p:.1e}".split("e")
    return rf"P = {m}$\times$10$^{{{int(e)}}}$"


def panel_a(ax) -> None:
    t = pd.read_csv(RES / "18_programme_genetics/gene_set_tests.tsv", sep="\t")
    t = t[t.gene_set.isin(SETS) & t.trait.isin(TRAITS)].copy()
    t["neglog10_p"] = -np.log10(t.p_one_sided)
    t["trait_label"] = t.trait.map(lambda k: TRAIT_LAB[k].replace("\n", ", "))
    t["set_label"] = t.gene_set.map(SET_LAB)
    t.to_csv(SRC / "Fig4_a_magma.tsv", sep="\t", index=False)
    off = dict(zip(SETS, [-0.3, -0.1, 0.1, 0.3]))
    for i, tr in enumerate(TRAITS):
        for s in SETS:
            r = t[(t.trait == tr) & (t.gene_set == s)].set_index("model")
            x0, x1 = i + off[s] - 0.035, i + off[s] + 0.035
            y0, y1 = r.loc["default", "neglog10_p"], r.loc["expr_conditioned", "neglog10_p"]
            ax.plot([x0, x1], [y0, y1], color=SET_COL[s], lw=0.6, zorder=1)
            ax.scatter([x0], [y0], s=14, color=SET_COL[s], edgecolor="black", linewidth=0.3, zorder=3)
            ax.scatter([x1], [y1], s=14, facecolor="white", edgecolor=SET_COL[s], linewidth=0.9, zorder=3)
        if i:
            ax.axvline(i - 0.5, color="#E3E3E3", lw=0.5, zorder=0)
    ax.set_yscale("function", functions=(np.sqrt, np.square))
    ax.set_ylim(0, 19)
    ax.set_yticks([0, 1, 2, 4, 8, 16])
    ax.axhline(-np.log10(0.05), color="#555555", lw=0.5, ls="--", zorder=0)
    ax.text(len(TRAITS) - 0.52, -np.log10(0.05), "P = 0.05", ha="right", va="bottom", fontsize=5.5, color="#555555")
    ax.set_xticks(range(len(TRAITS)))
    ax.set_xticklabels([TRAIT_LAB[k] for k in TRAITS], fontsize=6.5)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, len(TRAITS) - 0.5)
    ax.set_ylabel("Gene-set enrichment, −log$_{10}$ P\n(square-root scale)")
    n = t[(t.trait == "egfr_EA") & (t.model == "default")].set_index("gene_set").n_genes
    h1 = [Line2D([], [], marker="o", ls="", color=SET_COL[s], markeredgecolor="black", markeredgewidth=0.3, ms=4,
                 label=f"{SET_LAB[s]} ({int(n[s]):,})") for s in SETS]
    h2 = [Line2D([], [], marker="o", ls="", color="#555555", markeredgecolor="black", markeredgewidth=0.3, ms=4,
                 label="Standard model"),
          Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="#555555",
                 markeredgewidth=0.9, ms=4, label="Adjusted for kidney expression")]
    leg = ax.legend(handles=h1, loc="upper right", fontsize=6, handletextpad=0.2, borderpad=0.35,
                    labelspacing=0.25, **LEG)
    ax.add_artist(leg)
    ax.legend(handles=h2, loc="upper center", bbox_to_anchor=(0.47, 1.0), fontsize=6, handletextpad=0.2,
              borderpad=0.35, labelspacing=0.25, **LEG)


def panel_sldsc(ax) -> None:
    t = pd.read_csv(RES / "25_sldsc/sldsc_results.tsv", sep="\t")
    t = t[t.trait.isin(LDSC_TRAITS) & t.gene_set.isin(SETS)].copy()
    t["trait_label"] = t.trait.map(lambda k: TRAIT_LAB[k].replace("\n", ", "))
    t["set_label"] = t.gene_set.map(SET_LAB)
    t.to_csv(SRC / "Fig4_b_sldsc.tsv", sep="\t", index=False)
    off = dict(zip(SETS, [-0.3, -0.1, 0.1, 0.3]))
    for i, tr in enumerate(LDSC_TRAITS):
        for s in SETS:
            r = t[(t.trait == tr) & (t.gene_set == s)].iloc[0]
            sig = r.tau_q_bh < 0.05
            ax.bar(i + off[s], r.tau_z, width=0.18, color=SET_COL[s] if sig else "white", edgecolor=SET_COL[s],
                   linewidth=0.8, zorder=2)
            if sig:
                ax.text(i + off[s], r.tau_z + 0.12, "*", ha="center", va="bottom", fontsize=7, zorder=3)
        if i:
            ax.axvline(i - 0.5, color="#E3E3E3", lw=0.5, zorder=0)
    ax.axhline(0, color="#999999", lw=0.4, zorder=1)
    ax.axhline(1.645, color="#555555", lw=0.5, ls="--", zorder=1)
    ax.text(len(LDSC_TRAITS) - 0.52, 1.645, "P = 0.05", ha="right", va="bottom", fontsize=5.5, color="#555555")
    ax.set_xticks(range(len(LDSC_TRAITS)))
    ax.set_xticklabels([TRAIT_LAB[k] for k in LDSC_TRAITS], fontsize=6.5)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, len(LDSC_TRAITS) - 0.5)
    ax.set_ylim(-3.6, 4.6)
    ax.set_yticks([-3, -2, -1, 0, 1, 2, 3, 4])
    ax.set_ylabel("Heritability coefficient z\n(conditional on baseline-LD)")


def panel_b(axs) -> None:
    nd = pd.read_csv(RES / "23_fig_tables_v2/programme_null_draws.tsv", sep="\t")
    st = pd.read_csv(RES / "18_programme_genetics/empirical_null.tsv", sep="\t")
    nd = nd[nd.trait.isin(EGFR) & nd.gene_set.isin(SETS)]
    st = st[st.trait.isin(EGFR) & st.gene_set.isin(SETS)]
    nd.to_csv(SRC / "Fig4_c_null_draws.tsv", sep="\t", index=False)
    st.to_csv(SRC / "Fig4_c_stats.tsv", sep="\t", index=False)
    for ax, tr in zip(axs, EGFR):
        for k, s in enumerate(SETS):
            v = nd[(nd.trait == tr) & (nd.gene_set == s)].null_mean_z.values
            r = st[(st.trait == tr) & (st.gene_set == s)].iloc[0]
            vp = ax.violinplot([v], positions=[k], vert=False, widths=0.62, showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor("#D9D9D9")
                b.set_edgecolor("#8C8C8C")
                b.set_linewidth(0.4)
                b.set_alpha(1)
            ax.scatter([r.mean_z], [k], marker="D", s=16, color=SET_COL[s], edgecolor="black", linewidth=0.4,
                       zorder=3)
            p = r.p_empirical_greater
            lab = "P ≤ 0.001" if p <= 1 / (r.n_null + 1) + 1e-12 else f"P = {p:.2f}"
            ax.text(0.98, k - 0.40, lab, transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=5.5, fontweight="bold" if p < 0.05 else "normal")
        ax.set_ylim(len(SETS) - 0.45, -0.75)
        ax.set_title(TRAIT_LAB[tr].replace("\n", ", "), pad=3)
        ax.set_yticks(range(len(SETS)))
        ax.set_yticklabels([SET_LAB[s] for s in SETS] if ax is axs[0] else [], fontsize=6)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        lo = min(nd[nd.trait == tr].null_mean_z.min(), st[st.trait == tr].mean_z.min())
        hi = max(nd[nd.trait == tr].null_mean_z.max(), st[st.trait == tr].mean_z.max())
        pad = 0.12 * (hi - lo)
        ax.set_xlim(lo - pad, hi + 0.9 * pad)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    axs[1].set_xlabel("Mean gene Z")


def panel_c(ax) -> None:
    d = pd.read_csv(RES / "23_fig_tables_v2/genetic_support_genes.tsv", sep="\t")
    d["highlight"] = d.symbol.isin(HIGHLIGHT)
    d.to_csv(SRC / "Fig4_d_smr_genes.tsv", sep="\t", index=False)
    for s in ["tal_healthy", "tal_injured_down", "tal_injured_up"]:
        for hp in (True, False):
            q = d[(d.gene_set == s) & (d.heidi_pass == hp)]
            ax.scatter(q.log2fc_injured_vs_normal, q.z_protective, s=14 if s != "tal_healthy" else 9,
                       facecolor=SET_COL[s] if hp else "white", edgecolor=SET_COL[s] if not hp else "black",
                       linewidth=0.8 if not hp else 0.3, zorder=3 if s != "tal_healthy" else 2)
    ax.axhline(0, color="#999999", lw=0.4)
    ax.axvline(0, color="#999999", lw=0.4)
    pos = {"UMOD": (-2.35, -3.4), "KNG1": (-2.55, 7.7), "EPB41L3": (-1.6, 8.4), "SLC47A1": (-1.9, 3.2),
           "CTSS": (1.45, -7.4), "A4GALT": (1.75, 7.6), "XYLB": (-0.75, -8.3), "DACH1": (-0.45, 8.7),
           "SLC15A2": (-1.0, 9.6), "TFDP2": (0.4, 9.6)}
    for _, r in d[d.highlight].iterrows():
        ax.annotate(r.symbol, (r.log2fc_injured_vs_normal, r.z_protective), xytext=pos[r.symbol], fontsize=5.5, fontstyle="italic",
                    ha="center", va="center", zorder=4,
                    arrowprops=dict(arrowstyle="-", lw=0.35, color="#333333", shrinkA=1, shrinkB=2))
    ax.set_xlim(-3.1, 2.7)
    ax.set_ylim(-9.5, 10.5)
    ax.set_yticks([-5, 0, 5, 10])
    ax.set_xlabel("log$_2$FC, injured vs healthy TAL")
    ax.set_ylabel("SMR z in kidney eQTL\n(+ = higher expression, better function)")
    h = [Line2D([], [], marker="o", ls="", color=SET_COL[s], markeredgecolor="black", markeredgewidth=0.3, ms=3.5,
                label=SET_LAB[s]) for s in ["tal_healthy", "tal_injured_down", "tal_injured_up"]]
    h.append(Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="#555555",
                    markeredgewidth=0.8, ms=3.5, label="Fails single-variant test"))
    ax.legend(handles=h, loc="center left", bbox_to_anchor=(0.0, 0.47), fontsize=5.5, handletextpad=0.1,
              borderpad=0.3, labelspacing=0.2, **LEG)


def main() -> None:
    SRC.mkdir(parents=True, exist_ok=True)
    for f in SRC.glob("Fig4_*"):
        if f.name.startswith(("Fig4_a_magma", "Fig4_b_sldsc", "Fig4_c_null", "Fig4_c_stats", "Fig4_d_smr", "Fig4_b_null", "Fig4_b_stats", "Fig4_c_smr")):
            f.unlink()
    fig = plt.figure(figsize=(6.6, 7.6))
    gs = fig.add_gridspec(3, 1, left=0.135, right=0.99, top=0.96, bottom=0.06, height_ratios=[1, 0.72, 1.15],
                          hspace=0.42)
    a = fig.add_subplot(gs[0])
    b = fig.add_subplot(gs[1])
    bot = gs[2].subgridspec(1, 2, width_ratios=[1.45, 1], wspace=0.42)
    csub = bot[0].subgridspec(1, 3, wspace=0.12)
    cs = [fig.add_subplot(csub[i]) for i in range(3)]
    d = fig.add_subplot(bot[1])
    panel_a(a)
    panel_sldsc(b)
    panel_b(cs)
    panel_c(d)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    bb = lambda ax: inv.transform(ax.get_tightbbox(rend))  # noqa: E731
    x_left = min(bb(ax)[0, 0] for ax in (a, b, cs[0]))
    bot_y = max(bb(ax)[1, 1] for ax in cs + [d])
    for lab, x, y in [("a", x_left, bb(a)[1, 1]), ("b", x_left, bb(b)[1, 1]), ("c", x_left, bot_y),
                      ("d", bb(d)[0, 0], bot_y)]:
        fig.text(x, y + 0.006, lab, fontsize=11, fontweight="bold", ha="left", va="bottom")
    fig.savefig(FIG / "Fig4.pdf", dpi=300)
    fig.savefig(FIG / "Fig4.png", dpi=300)
    print("wrote", FIG / "Fig4.pdf")


if __name__ == "__main__":
    main()
