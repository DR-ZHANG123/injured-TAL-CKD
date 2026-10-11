"""Fig 3：损伤 TAL 细胞丢失分化功能并获得尿液损伤标志物；EGF 与 eGFR 的关联由损伤 TAL 负担解释。

只读 results/15_key_genes、16_fig_tables、22_tal_states、23_fig_tables_v2 的表；绘图输入写到 figures/source_data/Fig3_*.tsv。
输出 figures/Fig3.pdf（矢量）与 figures/Fig3.png（300 dpi）。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

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
    "mathtext.it": "Liberation Sans:italic", "mathtext.bf": "Liberation Sans:bold", "mathtext.default": "regular",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})
COL = {"REF": "#8C8C8C", "DKD": "#C44E52", "HKD": "#4C72B0"}
C_REP, C_NOT = "#C44E52", "#BBBBBB"
LEG = dict(frameon=True, fancybox=False, edgecolor="#CCCCCC", framealpha=1)
ALIAS = {"UMOD": "uromodulin", "SLC12A1": "NKCC2", "KCNJ1": "ROMK", "PTGER3": "EP3", "CASR": "CaSR",
         "LCN2": "NGAL", "HAVCR1": "KIM-1", "CCL2": "MCP-1", "SPP1": "osteopontin"}
FUNCTION = ["EGF", "UMOD", "SLC12A1", "KCNJ1", "CLDN16", "PTGER3", "CASR", "CLDN10"]
INJURY = ["LCN2", "HAVCR1", "MMP7", "CCL2", "SPP1", "CD44", "VCAM1", "TIMP2"]
SETS = [("KPMP_snRNA_discovery", "KPMP\nsnRNA"), ("KPMP_scRNA_independent", "KPMP\nscRNA"),
        ("GSE271188", "GSE\n271188")]


def fmt_p(p: float) -> str:
    if p >= 0.01:
        return f"P = {p:.2f}"
    if p >= 0.001:
        return f"P = {p:.3f}"
    m, e = f"{p:.1e}".split("e")
    return rf"P = {m}$\times$10$^{{{int(e)}}}$"


def panel_a(axs) -> None:
    kg = pd.read_csv(RES / "15_key_genes/aTAL_frTAL_key_genes.tsv", sep="\t", index_col=0).rename_axis("gene")
    st = pd.read_csv(RES / "16_fig_tables/key_gene_replication.tsv", sep="\t").set_index("replication_set")
    kg[["log2fc", "log2fc_kpmp_sc", "log2fc_gse271188", "replicated_kpmp_sc", "replicated_gse271188",
        "replicated_both"]].to_csv(SRC / "Fig3_a_log2fc.tsv", sep="\t")
    st.to_csv(SRC / "Fig3_a_stats.tsv", sep="\t")
    for ax, col, rcol, lab, key in [(axs[0], "log2fc_kpmp_sc", "replicated_kpmp_sc", "KPMP scRNA", "KPMP_scRNA"),
                                    (axs[1], "log2fc_gse271188", "replicated_gse271188", "GSE271188", "GSE271188")]:
        d = kg[["log2fc", col, rcol]].dropna(subset=[col])
        for ok, c, z in [(False, C_NOT, 1), (True, C_REP, 2)]:
            s = d[d[rcol] == ok]
            ax.scatter(s.log2fc, s[col], s=3, color=c, alpha=0.6, linewidth=0, zorder=z, rasterized=True)
        ax.axhline(0, color="#999999", lw=0.4)
        ax.axvline(0, color="#999999", lw=0.4)
        lim = (-6.5, 6.5)
        ax.plot(lim, lim, color="#555555", lw=0.5, ls="--", zorder=0)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xticks([-5, 0, 5])
        ax.set_yticks([-5, 0, 5])
        ax.set_xlabel("Discovery log$_2$FC (KPMP snRNA)")
        ax.set_ylabel(f"Replication log$_2$FC ({lab})")
        r = st.loc[key]
        ax.set_title(lab, pad=4)
    axs[1].legend(handles=[Line2D([], [], marker="o", ls="", color=C_REP, ms=3, label="Replicated"),
                           Line2D([], [], marker="o", ls="", color=C_NOT, ms=3, label="Not replicated")],
                  loc="lower right", fontsize=5.5, handletextpad=0.1, borderpad=0.3, labelspacing=0.2, **LEG)


def panel_b(ax, cax, lax) -> None:
    t = pd.read_csv(RES / "22_tal_states/tal_gene_lookup.tsv", sep="\t")
    t = t[(t.state == "aTAL_frTAL") & t.gene.isin(FUNCTION + INJURY)].set_index("gene")
    fn = t.loc[FUNCTION].sort_values(f"log2fc_{SETS[0][0]}").index.tolist()
    inj = t.loc[INJURY].sort_values(f"log2fc_{SETS[0][0]}", ascending=False).index.tolist()
    ypos = {g: i + 1 for i, g in enumerate(fn)} | {g: i + 2 + len(fn) for i, g in enumerate(inj)}
    rows = []
    for g in fn + inj:
        for j, (k, _) in enumerate(SETS):
            rows.append({"gene": g, "group": t.loc[g, "category"], "dataset": k, "n_participants": t.loc[g, f"n_{k}"],
                         "log2fc": t.loc[g, f"log2fc_{k}"], "fdr": t.loc[g, f"fdr_{k}"], "x": j, "y": ypos[g]})
    d = pd.DataFrame(rows)
    d["neglog10_fdr"] = -np.log10(d.fdr)
    d.to_csv(SRC / "Fig3_b_genes.tsv", sep="\t", index=False)
    cap, vmax = 20.0, 4.0
    size = lambda v: 3 + 42 * np.minimum(v, cap) / cap  # noqa: E731
    sc = ax.scatter(d.x, d.y, s=size(d.neglog10_fdr), c=d.log2fc.clip(-vmax, vmax), cmap="RdBu_r", vmin=-vmax,
                    vmax=vmax, edgecolor="#555555", linewidth=0.3)
    ax.set_yticks([ypos[g] for g in fn + inj])
    ax.set_yticklabels(fn + inj, fontsize=6, fontstyle="italic")
    ax.set_ylim(len(fn) + len(inj) + 1.6, -0.6)
    ax.set_xlim(-0.45, 2.55)
    ax.set_xticks(range(3))
    n = t.iloc[0]
    ax.set_xticklabels([f"{lab}\n(n = {int(n[f'n_{k}'])})" for k, lab in SETS], fontsize=5.5)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=2)
    for s in ax.spines.values():
        s.set_visible(False)
    for y0, lab in [(0, "TAL function"), (len(fn) + 1, "Urinary injury markers")]:
        ax.text(-0.45, y0, lab, ha="left", va="center", fontsize=6, fontweight="bold")
    ax.plot([-0.45, 3.3], [len(fn) + 0.75] * 2, color="#CCCCCC", lw=0.5)
    cb = plt.colorbar(sc, cax=cax, orientation="horizontal", extend="both")
    cb.set_label("log$_2$FC, injured vs healthy TAL", fontsize=6)
    cb.set_ticks([-4, -2, 0, 2, 4])
    cb.ax.tick_params(labelsize=5.5, length=2)
    cb.outline.set_linewidth(0.4)
    lax.axis("off")
    hs = [Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="#555555", markeredgewidth=0.3,
                 ms=np.sqrt(size(v)), label=lab) for v, lab in [(2, "2"), (10, "10"), (20, "≥20")]]
    lax.legend(handles=hs, title="−log$_{10}$ FDR", ncol=3, loc="center", fontsize=5.5, title_fontsize=5.5,
               handletextpad=0.1, columnspacing=0.8, borderpad=0.3, **LEG)


def panel_c(ax) -> None:
    g = pd.read_csv(RES / "22_tal_states/participant_tal_genes.tsv", sep="\t")
    a = pd.read_csv(RES / "22_tal_states/tal_gene_participant_assoc.tsv", sep="\t")
    d = g[g.dx.isin(COL)].dropna(subset=["all_TAL_EGF", "injured_frac"])[
        ["participant", "dx", "injured_frac", "all_TAL_EGF"]]
    r = a[(a.pool == "all_TAL") & (a.gene == "EGF")]
    d.to_csv(SRC / "Fig3_c_participants.tsv", sep="\t", index=False)
    r.to_csv(SRC / "Fig3_c_stats.tsv", sep="\t", index=False)
    r = r.iloc[0]
    for dx in ["REF", "HKD", "DKD"]:
        s = d[d.dx == dx]
        ax.scatter(s.injured_frac, s.all_TAL_EGF, s=8, color=COL[dx], alpha=0.85, edgecolor="white", linewidth=0.25)
    ax.set_xscale("logit")
    ticks = [0.002, 0.01, 0.05, 0.2, 0.5]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:g}%"))
    ax.set_xlim(0.0012, 0.85)
    lo, hi = d.all_TAL_EGF.min(), d.all_TAL_EGF.max()
    ax.set_ylim(lo - 0.6, hi + 0.08 * (hi - lo))
    ax.set_xlabel("Injured TAL fraction (logit scale)")
    ax.set_ylabel(r"$\it{EGF}$ in TAL (log$_2$ CPM)")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=COL[k], ms=3.5, label=l)
                       for k, l in [("REF", "Reference"), ("DKD", "DKD"), ("HKD", "HKD")]],
              loc="lower left", fontsize=6, handletextpad=0.2, borderpad=0.3, labelspacing=0.25, **LEG)


def panel_d(ax) -> None:
    m = pd.read_csv(RES / "23_fig_tables_v2/egf_egfr_standardized.tsv", sep="\t")
    m = m[m.model.isin(["none", "dx", "dx+injured_frac"])]
    m.to_csv(SRC / "Fig3_d_models.tsv", sep="\t", index=False)
    lab = {"none": "Unadjusted", "dx": "Diagnosis", "dx+injured_frac": "Diagnosis + injured TAL"}
    col = {"none": "#555555", "dx": "#555555", "dx+injured_frac": "#C44E52"}
    yt, yl, bold = [], [], []
    y = 0
    for pool, title in [("all_TAL", "All TAL"), ("normal_TAL", "Healthy TAL")]:
        s = m[m.pool == pool].set_index("model")
        yt.append(y)
        yl.append(f"{title} (n = {int(s.n.iloc[0])})")
        bold.append(True)
        y += 1
        for k in ["none", "dx", "dx+injured_frac"]:
            r = s.loc[k]
            ax.plot([r.ci_low, r.ci_high], [y, y], color=col[k], lw=0.9, solid_capstyle="butt")
            ax.scatter([r.beta_std], [y], s=14, marker="s", color=col[k], zorder=3)
            ax.text(0.57, y, fmt_p(r.p), ha="left", va="center", fontsize=5.5)
            yt.append(y)
            yl.append(lab[k])
            bold.append(False)
            y += 1
        y += 0.3
    ax.axvline(0, color="#999999", lw=0.5, ls="--")
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=6)
    for tl, b in zip(ax.get_yticklabels(), bold):
        if b:
            tl.set_fontweight("bold")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(y - 0.5, -0.6)
    ax.set_xlim(-0.32, 0.92)
    ax.set_xticks([-0.2, 0, 0.2, 0.4])
    ax.set_xticklabels(["−0.2", "0", "0.2", "0.4"])
    ax.set_xlabel(r"Standardized $\beta$ per SD eGFR")


def main() -> None:
    SRC.mkdir(parents=True, exist_ok=True)
    for f in SRC.glob("Fig3_*"):
        f.unlink()
    fig = plt.figure(figsize=(6.6, 6.2))
    gs = fig.add_gridspec(1, 2, left=0.08, right=0.99, top=0.93, bottom=0.08, width_ratios=[2.0, 0.82], wspace=0.22)
    left = gs[0, 0].subgridspec(2, 1, hspace=0.42)
    top = left[0].subgridspec(1, 2, wspace=0.42)
    bot = left[1].subgridspec(1, 2, width_ratios=[1, 0.78], wspace=1.05)
    a1, a2 = fig.add_subplot(top[0]), fig.add_subplot(top[1])
    c, dax = fig.add_subplot(bot[0]), fig.add_subplot(bot[1])
    sub = gs[0, 1].subgridspec(4, 1, height_ratios=[1, 0.016, 0.06, 0.07], hspace=0.1)
    b, cax, lax = fig.add_subplot(sub[0]), fig.add_subplot(sub[1]), fig.add_subplot(sub[3])
    panel_a([a1, a2])
    panel_b(b, cax, lax)
    panel_c(c)
    panel_d(dax)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    bb = lambda ax: inv.transform(ax.get_tightbbox(rend))  # noqa: E731
    top_y = max(bb(ax)[1, 1] for ax in (a1, a2, b))
    bot_y = max(bb(ax)[1, 1] for ax in (c, dax))
    x_left = min(bb(ax)[0, 0] for ax in (a1, c))
    for lab, x, y in [("a", x_left, top_y), ("b", bb(b)[0, 0], top_y), ("c", x_left, bot_y),
                      ("d", bb(dax)[0, 0], bot_y)]:
        fig.text(x, y + 0.008, lab, fontsize=11, fontweight="bold", ha="left", va="bottom")
    fig.savefig(FIG / "Fig3.pdf", dpi=300)
    fig.savefig(FIG / "Fig3.png", dpi=300)
    print("wrote", FIG / "Fig3.pdf")


if __name__ == "__main__":
    main()
