"""Fig 2：损伤 TAL 细胞在 DKD 与高血压肾病中扩张，并且是两种测序平台间测量最一致的损伤上皮状态。

只读 results/ 下的表：
  a  22_tal_states/participant_fractions.tsv（参与者比例、取材方式）、21_confounders/{procurement_contrasts,dx_models}.tsv、
     22_tal_states/group_contrasts.tsv（AKI、无肾病糖尿病）
  b  19a_gse211785_relabel/{unit_fractions,abundance}.tsv（单核，一个供体一个点）、19b_gse195460/{donor_fractions,
     abundance_clinical}.tsv、16_fig_tables/{gse271188_donor_fractions,gse271188_abundance}.tsv
  c  21_confounders/{sn_sc_agreement_lineages,sn_sc_rho_difference}.tsv + 22_tal_states/participant_fractions.tsv（散点）
  d  22_tal_states/{sn_vs_sc_agreement,sn_sc_state_profile_concordance}.tsv + participant_fractions.tsv
每个 panel 的绘图输入写到 figures/source_data/Fig2_*.tsv；输出 figures/Fig2.pdf（矢量）与 Fig2.png（300 dpi）。
比例轴用平方根刻度（可容纳 0 值，低比例区间仍可分辨）。
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
    "font.size": 6.5, "axes.labelsize": 6.5, "axes.titlesize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 5.8, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.default": "regular",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})
COL = {"REF": "#8C8C8C", "Control": "#8C8C8C", "DKD": "#C44E52", "HKD": "#4C72B0", "AKI": "#DD8452",
       "DM_resilient": "#55A868", "GLOM_DISEASE": "#8172B3", "other": "#D9D9D9"}
DXLAB = {"REF": "Reference", "DKD": "DKD", "HKD": "HKD", "AKI": "AKI", "DM_resilient": "Diabetes without CKD",
         "other": "Other"}
SMALL = 5.5
RNG = np.random.default_rng(20261010)
SQ_TICKS = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]


def fmt_p(p: float) -> str:
    if p >= 0.1:
        return f"P = {p:.2f}"
    if p >= 0.01:
        return f"P = {p:.3f}"
    if p >= 0.001:
        return f"P = {p:.3f}"
    m, e = f"{p:.1e}".split("e")
    return rf"P = {m}$\times$10$^{{{int(e)}}}$"


def sqrt_axis(ax, top: float = 1.0, ylabel: str | None = None, ticks=SQ_TICKS) -> None:
    ticks = [t for t in ticks if t <= top + 1e-9]
    ax.set_yticks(np.sqrt(ticks))
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.spines["left"].set_bounds(0, np.sqrt(top))
    if ylabel:
        ax.set_ylabel(ylabel)


def strip(ax, x: float, v: np.ndarray, color: str, marker: str = "o", s: float = 7, width: float = 0.34,
          edge: str = "white", median: bool = True) -> None:
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    jit = RNG.uniform(-width / 2, width / 2, len(v))
    ax.scatter(x + jit, np.sqrt(v), s=s, color=color, alpha=0.85, edgecolor=edge, linewidth=0.3, marker=marker,
               zorder=2, clip_on=False)
    if median and len(v):
        ax.hlines(np.sqrt(np.median(v)), x - width * 0.75, x + width * 0.75, color="black", lw=1.0, zorder=3)


def bracket(ax, x1: float, x2: float, y: float, text: str, h: float = 0.025) -> None:
    ax.plot([x1, x1, x2, x2], [y - h, y, y, y - h], color="black", lw=0.5, clip_on=False)
    ax.text((x1 + x2) / 2, y + 0.008, text, ha="center", va="bottom", fontsize=SMALL, clip_on=False)


# ---------------------------------------------------------------- a
def panel_a(ax) -> None:
    pf = pd.read_csv(RES / "22_tal_states/participant_fractions.tsv", sep="\t", dtype={"participant": str})
    pc = pd.read_csv(RES / "21_confounders/procurement_contrasts.tsv", sep="\t")
    dm = pd.read_csv(RES / "21_confounders/dx_models.tsv", sep="\t")
    gc = pd.read_csv(RES / "22_tal_states/group_contrasts.tsv", sep="\t")
    order = ["REF", "DKD", "HKD", "AKI", "DM_resilient"]
    d = pf[(pf.modality == "snRNA") & pf.dx.isin(order)].dropna(subset=["injured_frac"]).copy()
    d["percutaneous"] = (d.dx == "REF") & (d.tissue == "Percutaneous Needle Biopsy")
    d[["participant", "dx", "tissue", "percutaneous", "Age", "sex", "n_tal", "injured_frac"]].to_csv(
        SRC / "Fig2_a_participants.tsv", sep="\t", index=False)

    # 统计：DKD/HKD 对经皮参考（21）；校正年龄性别取材（21 dx_models）；AKI/DM 对全部参考（22，校正年龄性别）
    pcs = pc[(pc.modality == "snRNA") & (pc.lineage == "TAL")].set_index("contrast")
    dms = dm[(dm.lineage == "TAL") & (dm["sample"] == "all_procurement") & (dm.adjustment == "age+sex+procurement")]
    dms = dms.set_index("term")
    gcs = gc[(gc.modality == "snRNA") & (gc.measure == "injured_frac")].set_index("contrast")
    rows = []
    for g in ["DKD", "HKD"]:
        r = pcs.loc[f"{g} vs REF:Percutaneous Needle Biopsy"]
        a = dms.loc[f"{g}_vs_REF"]
        rows.append({"group": g, "comparator": "percutaneous reference", "n_group": r.n_a, "n_comparator": r.n_b,
                     "hedges_g_logit": r.hedges_g_logit, "p_mannwhitney": r.p,
                     "adjusted_for": "age+sex+procurement (all reference)", "beta_logit_adj": a.estimate,
                     "ci_low_adj": a.ci_low, "ci_high_adj": a.ci_high, "p_adj": a.p})
    for g in ["AKI", "DM_resilient"]:
        r = gcs.loc[f"{g}_vs_REF"]
        rows.append({"group": g, "comparator": "all reference", "n_group": r.n_a, "n_comparator": r.n_b,
                     "hedges_g_logit": r.hedges_g_logit, "p_mannwhitney": r.p_mannwhitney,
                     "adjusted_for": "age+sex", "beta_logit_adj": r.beta_logit_adj_age_sex, "ci_low_adj": np.nan,
                     "ci_high_adj": np.nan, "p_adj": r.p_adj_age_sex})
    st = pd.DataFrame(rows)
    kw = pcs.loc["REF procurement types (Kruskal-Wallis)"]
    st = pd.concat([st, pd.DataFrame([{"group": "REF", "comparator": "procurement types (Kruskal-Wallis)",
                                        "n_group": kw.n_a, "p_mannwhitney": kw.p}])], ignore_index=True)
    st.to_csv(SRC / "Fig2_a_stats.tsv", sep="\t", index=False)

    for i, g in enumerate(order):
        v = d[d.dx == g]
        if g == "REF":
            o, pcu = v[~v.percutaneous], v[v.percutaneous]
            strip(ax, i, o.injured_frac, COL[g], s=7)
            strip(ax, i, pcu.injured_frac, "white", marker="D", s=9, edge="black", median=False)
            ax.hlines(np.sqrt(pcu.injured_frac.median()), i - 0.3, i + 0.3, color="black", lw=0.8, ls=(0, (2, 1.2)),
                      zorder=3)
        else:
            strip(ax, i, v.injured_frac, COL[g], s=7)
    stx = st.set_index("group")
    n = d.groupby("dx").size()
    labs = {"REF": "Reference", "DKD": "DKD", "HKD": "HKD", "AKI": "AKI", "DM_resilient": "Diabetes\nwithout CKD"}
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{labs[g]}\nn = {n[g]}" for g in order], fontsize=SMALL + 0.3)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(0, 1.2)
    sqrt_axis(ax, ylabel="Injured TAL fraction (aTAL + frTAL of TAL)")
    ax.legend(handles=[Line2D([], [], marker="D", ls="", mfc="white", mec="black", mew=0.5, ms=3,
                              label="Reference, percutaneous biopsy"),
                       Line2D([], [], marker="o", ls="", color=COL["REF"], ms=3, label="Reference, other procurement")],
              loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=True, fancybox=False,
              edgecolor="#CCCCCC", fontsize=SMALL, handletextpad=0.1, borderpad=0.3, columnspacing=0.8)
    ax.set_title("KPMP snRNA-seq", pad=14)


# ---------------------------------------------------------------- b
def cohort_axis(ax, title: str, groups: list[str], labels: list[str], d: pd.DataFrame, val: str,
                brackets: list[tuple], ylabel: str | None) -> None:
    for i, g in enumerate(groups):
        strip(ax, i, d.loc[d.group == g, val], COL[g], s=8, width=0.36)
    n = d.groupby("group").size()
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{lab}\nn = {n.get(g, 0)}" for g, lab in zip(groups, labels)], fontsize=SMALL + 0.3)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(0, 1.2)
    sqrt_axis(ax, ylabel=ylabel)
    if ylabel is None:
        ax.set_yticklabels([])
    for x1, x2, y, txt in brackets:
        bracket(ax, x1, x2, y, txt)
    ax.set_title(title, pad=14)


def panel_b(axs) -> None:
    # GSE211785：单核文库，一个供体一个点（19a，TAL 簇 + TAL 参照）
    u = pd.read_csv(RES / "19a_gse211785_relabel/unit_fractions.tsv", sep="\t")
    u = u[u.tech == "SN_RNA"].rename(columns={"sample": "donor"})
    ab = pd.read_csv(RES / "19a_gse211785_relabel/abundance.tsv", sep="\t")
    ab = ab[ab.analysis == "tal_clusters_tal_reference|prefer_SN"].set_index("contrast")
    # GSE195460（19b，全亚群参照）
    b = pd.read_csv(RES / "19b_gse195460/donor_fractions.tsv", sep="\t")
    bc = pd.read_csv(RES / "19b_gse195460/abundance_clinical.tsv", sep="\t")
    bc = bc[(bc.analysis == "all_state_labels") & (bc.test == "DKD_vs_Control")].iloc[0]
    # GSE271188（09 → 16）
    g = pd.read_csv(RES / "16_fig_tables/gse271188_donor_fractions.tsv", sep="\t").rename(
        columns={"gsm": "donor", "tal_injured_frac": "frac"})
    gs = pd.read_csv(RES / "16_fig_tables/gse271188_abundance.tsv", sep="\t")
    gs = gs[gs.measure == "tal_injured_frac"].set_index("contrast")

    pd.concat([u.assign(cohort="GSE211785 snRNA")[["cohort", "donor", "group", "n_tal", "frac"]],
               b.assign(cohort="GSE195460")[["cohort", "donor", "group", "n_tal", "frac"]],
               g.assign(cohort="GSE271188")[
                   ["cohort", "donor", "group", "n_tal", "frac"]]], ignore_index=True).to_csv(
        SRC / "Fig2_b_donors.tsv", sep="\t", index=False)
    stats = pd.DataFrame([
        {"cohort": "GSE211785 snRNA", "contrast": c, "p_mannwhitney": ab.loc[c, "p_mw_SN_RNA"],
         "n": ab.loc[c, "n_SN_RNA"]} for c in ["DKD_vs_Control", "HKD_vs_Control", "DKD_vs_HKD"]]
        + [{"cohort": "GSE195460", "contrast": "DKD_vs_Control", "p_mannwhitney": bc.p_mannwhitney, "n": bc.n,
            "hedges_g_logit": bc.hedges_g_logit, "p_adj_chemistry": bc.p_adj_chem}]
        + [{"cohort": "GSE271188", "contrast": c, "p_mannwhitney": gs.loc[c, "p_mannwhitney"],
            "n": f"{gs.loc[c, 'n_a']}v{gs.loc[c, 'n_b']}", "hedges_g_logit": gs.loc[c, "hedges_g_logit"]}
           for c in ["DKD_vs_Control", "GLOM_DISEASE_vs_Control", "DKD_vs_GLOM_DISEASE"]])
    stats.to_csv(SRC / "Fig2_b_stats.tsv", sep="\t", index=False)

    cohort_axis(axs[0], "GSE211785 (snRNA)", ["Control", "DKD", "HKD"], ["Ctrl", "DKD", "HKD"], u, "frac",
                [(0, 1, 1.0, fmt_p(ab.loc["DKD_vs_Control", "p_mw_SN_RNA"])),
                 (0, 2, 1.1, fmt_p(ab.loc["HKD_vs_Control", "p_mw_SN_RNA"]))],
                "Injured TAL fraction")
    cohort_axis(axs[1], "GSE195460", ["Control", "DKD"], ["Ctrl", "DKD"], b, "frac",
                [(0, 1, 1.0, fmt_p(bc.p_mannwhitney))], None)
    cohort_axis(axs[2], "GSE271188", ["Control", "GLOM_DISEASE", "DKD"], ["Ctrl", "PGD", "DKD"], g, "frac",
                [(0, 1, 1.0, fmt_p(gs.loc["GLOM_DISEASE_vs_Control", "p_mannwhitney"])),
                 (0, 2, 1.1, fmt_p(gs.loc["DKD_vs_Control", "p_mannwhitney"]))], None)


# ---------------------------------------------------------------- c
def paired(pf: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    sn = pf[pf.modality == "snRNA"].set_index("participant")
    sc = pf[pf.modality == "scRNA"].set_index("participant")
    j = sn[["dx"] + cols].join(sc[cols], lsuffix="_sn", rsuffix="_sc", how="inner")
    return j.reset_index()


def dx_scatter(ax, j: pd.DataFrame, x: str, y: str) -> None:
    for g in ["other", "REF", "DM_resilient", "AKI", "HKD", "DKD"]:
        s = j[j.dx == g]
        ax.scatter(s[x], s[y], s=9, color=COL[g], edgecolor="#777777" if g == "other" else "white", linewidth=0.3,
                   alpha=0.9, zorder=2)


def panel_c(ax1, ax2, ax3) -> None:
    pf = pd.read_csv(RES / "22_tal_states/participant_fractions.tsv", sep="\t", dtype={"participant": str})
    ag = pd.read_csv(RES / "21_confounders/sn_sc_agreement_lineages.tsv", sep="\t")
    ag = ag[(ag.subset == "all") & (ag.participants == "own_available")].set_index("lineage")
    rd = pd.read_csv(RES / "21_confounders/sn_sc_rho_difference.tsv", sep="\t")
    rd = rd[rd.subset == "all"].set_index("other_lineage")
    j = paired(pf, ["injured_frac"]).dropna(subset=["injured_frac_sn", "injured_frac_sc"])
    j.to_csv(SRC / "Fig2_c_tal_sn_vs_sc.tsv", sep="\t", index=False)
    ag.to_csv(SRC / "Fig2_c_agreement.tsv", sep="\t")
    rd.to_csv(SRC / "Fig2_c_rho_difference.tsv", sep="\t")

    # c1：TAL 同一参与者散点（平方根刻度）
    jj = j.assign(xs=np.sqrt(j.injured_frac_sn), ys=np.sqrt(j.injured_frac_sc))
    dx_scatter(ax1, jj, "xs", "ys")
    ax1.plot([0, 1], [0, 1], color="#999999", lw=0.5, ls="--", zorder=1)
    tk = [0, 0.05, 0.25, 0.5, 1.0]
    for f in (ax1.set_xticks, ax1.set_yticks):
        f(np.sqrt(tk))
    ax1.set_xticklabels([f"{t:g}" for t in tk])
    ax1.set_yticklabels([f"{t:g}" for t in tk])
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect("equal")
    ax1.set_xlabel("snRNA injured TAL fraction")
    ax1.set_ylabel("scRNA injured TAL fraction")
    r = ag.loc["TAL"]
    # c2：五种损伤上皮状态的 ICC 与 Spearman
    lins = ["TAL", "PT", "DCT", "CNT", "DTL"]
    names = {"TAL": "TAL", "PT": "PT", "DCT": "DCT", "CNT": "CNT", "DTL": "DTL"}
    for i, lin in enumerate(lins):
        r = ag.loc[lin]
        bold = lin == "TAL"
        ax2.plot([r.icc_absolute_logit, r.spearman], [i, i], color="#BBBBBB", lw=0.8, zorder=1)
        ax2.scatter(r.icc_absolute_logit, i, s=18, marker="s", color="black" if bold else "#666666", zorder=3)
        ax2.scatter(r.spearman, i, s=18, marker="o", facecolor="white", edgecolor="black" if bold else "#666666",
                    linewidth=0.8, zorder=3)
    ax2.axvline(0, color="#999999", lw=0.5, ls="--")
    ax2.set_yticks(range(len(lins)))
    ax2.set_yticklabels([f"{names[l]} ({int(ag.loc[l, 'n'])})" for l in lins])
    ax2.get_yticklabels()[0].set_fontweight("bold")
    ax2.set_ylim(len(lins) - 0.4, -0.6)
    ax2.set_xlim(-0.7, 1.0)
    ax2.set_xticks([-0.5, 0, 0.5, 1])
    ax2.set_xlabel("snRNA–scRNA agreement")
    ax2.legend(handles=[Line2D([], [], marker="s", ls="", color="black", ms=3.5, label="ICC (absolute)"),
                        Line2D([], [], marker="o", ls="", mfc="white", mec="black", ms=3.5, label=r"Spearman $\rho$")],
               loc="upper left", frameon=True, fancybox=False, edgecolor="#CCCCCC", fontsize=SMALL,
               handletextpad=0.2, borderpad=0.3, labelspacing=0.2)

    # c3：ρ 差（TAL − 其他）参与者 bootstrap 95% CI
    others = ["PT", "DCT", "CNT", "DTL"]
    for i, lin in enumerate(others):
        r = rd.loc[lin]
        ax3.plot([r.ci_low, r.ci_high], [i, i], color="black", lw=0.9)
        ax3.scatter(r.rho_diff, i, s=16, color="black", zorder=3)
        pb = r.p_boot_two_sided
        ptxt = "P < 0.001" if pb < 0.001 else f"P = {pb:.3f}"
        ax3.text(2.12, i, ptxt, va="center", ha="left", fontsize=SMALL, clip_on=False)
    ax3.axvline(0, color="#999999", lw=0.5, ls="--")
    ax3.set_yticks(range(len(others)))
    ax3.set_yticklabels([f"vs {l} ({int(rd.loc[l, 'n'])})" for l in others])
    ax3.set_ylim(len(others) - 0.4, -0.6)
    ax3.set_xlim(-0.2, 2.05)
    ax3.set_xticks([0, 0.5, 1, 1.5, 2])
    ax3.set_xticklabels(["0", "0.5", "1.0", "1.5", "2.0"])
    ax3.set_xlabel(r"$\rho$(TAL) $-$ $\rho$(other), 95% CI")


# ---------------------------------------------------------------- d
def panel_d(ax1, ax2, ax3) -> None:
    pf = pd.read_csv(RES / "22_tal_states/participant_fractions.tsv", sep="\t", dtype={"participant": str})
    sa = pd.read_csv(RES / "22_tal_states/sn_vs_sc_agreement.tsv", sep="\t").set_index("measure")
    pc = pd.read_csv(RES / "22_tal_states/sn_sc_state_profile_concordance.tsv", sep="\t")
    j = paired(pf, ["frTAL_share_of_injured"]).dropna(
        subset=["frTAL_share_of_injured_sn", "frTAL_share_of_injured_sc"])
    j.to_csv(SRC / "Fig2_d_frtal_share_sn_vs_sc.tsv", sep="\t", index=False)
    sa.to_csv(SRC / "Fig2_d_measure_agreement.tsv", sep="\t")
    pc.to_csv(SRC / "Fig2_d_profile_concordance.tsv", sep="\t", index=False)

    dx_scatter(ax1, j, "frTAL_share_of_injured_sn", "frTAL_share_of_injured_sc")
    ax1.plot([0, 1], [0, 1], color="#999999", lw=0.5, ls="--", zorder=1)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect("equal")
    ax1.set_xticks([0, 0.5, 1])
    ax1.set_yticks([0, 0.5, 1])
    ax1.set_xlabel("snRNA frTAL share of injured TAL")
    ax1.set_ylabel("scRNA frTAL share")
    r = sa.loc["frTAL_share_of_injured"]
    # d2：各指标同一参与者 Spearman
    meas = [("injured_frac", "aTAL + frTAL"), ("aTAL_frac", "aTAL"), ("frTAL_frac", "frTAL"),
            ("frTAL_share_of_injured", "frTAL share")]
    for i, (m, lab) in enumerate(meas):
        r = sa.loc[m]
        col = "black" if m == "injured_frac" else "#9A9A9A"
        ax2.barh(i, r.spearman, height=0.62, color=col, edgecolor="white", linewidth=0.3)
        neg = r.spearman < 0
        ax2.text(r.spearman + (0.04 if not neg else 0.05), i, f"{r.spearman:.2f}".replace("-", "−"), va="center",
                 ha="left", fontsize=SMALL, color="white" if neg else "black")
    ax2.axvline(0, color="black", lw=0.5)
    ax2.set_yticks(range(len(meas)))
    ax2.set_yticklabels([lab for _, lab in meas])
    ax2.get_yticklabels()[0].set_fontweight("bold")
    ax2.set_ylim(len(meas) - 0.5, -0.5)
    ax2.set_xlim(-1.15, 1.15)
    ax2.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax2.set_xticklabels(["−1", "−0.5", "0", "0.5", "1"])
    ax2.set_xlabel(r"snRNA vs scRNA, Spearman $\rho$")

    # d3：基因谱层面（单核状态 vs 单细胞状态，相对正常 TAL 的 log2FC 谱 Pearson r）
    sel = [("aTAL_vs_normal_sn", "aTAL_vs_normal_sc", "sn aTAL ~ sc aTAL"),
           ("aTAL_vs_normal_sn", "frTAL_vs_normal_sc", "sn aTAL ~ sc frTAL"),
           ("frTAL_vs_aTAL_sn", "frTAL_vs_aTAL_sc", "sn (frTAL − aTAL) ~\nsc (frTAL − aTAL)")]
    pci = pc.set_index(["snRNA_profile", "scRNA_profile"])
    for i, (a, b, lab) in enumerate(sel):
        v = pci.loc[(a, b), "pearson"]
        ax3.barh(i, v, height=0.62, color="#9A9A9A" if i != 1 else "black", edgecolor="white", linewidth=0.3)
        neg = v < 0
        ax3.text(v + (0.04 if not neg else 0.03), i, f"{v:.2f}".replace("-", "−"), va="center", ha="left",
                 fontsize=SMALL, color="white" if neg else "black")
    ax3.axvline(0, color="black", lw=0.5)
    ax3.set_yticks(range(len(sel)))
    ax3.set_yticklabels([s[2] for s in sel], fontsize=SMALL + 0.3)
    ax3.set_ylim(len(sel) - 0.5, -0.5)
    ax3.set_xlim(-0.5, 1.15)
    ax3.set_xticks([-0.5, 0, 0.5, 1])
    ax3.set_xticklabels(["−0.5", "0", "0.5", "1"])
    ax3.set_xlabel("Expression profile correlation, r")


def letters(fig, rows) -> None:
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    xs = {}
    for row in rows:
        ytop = max(inv.transform((0, ax.get_tightbbox(rend).y1))[1] for _, grp in row for ax in grp)
        for lt, grp in row:
            x = min(inv.transform((ax.get_tightbbox(rend).x0, 0))[0] for ax in grp)
            xs[lt] = (x, ytop)
    col0 = min(v[0] for k, v in xs.items() if k in "acd")
    for lt, (x, y) in xs.items():
        fig.text(col0 if lt in "acd" else x, y + 0.006, lt, fontsize=11, fontweight="bold", ha="left", va="bottom")


def main() -> None:
    SRC.mkdir(parents=True, exist_ok=True)
    for f in SRC.glob("Fig2_*"):
        f.unlink()
    fig = plt.figure(figsize=(6.6, 5.7))
    top = fig.add_gridspec(1, 4, left=0.085, right=0.995, top=0.915, bottom=0.66, width_ratios=[2.35, 1.12, 0.78, 1.12],
                           wspace=0.16)
    a = fig.add_subplot(top[0])
    b = [fig.add_subplot(top[i]) for i in (1, 2, 3)]
    mid = fig.add_gridspec(1, 3, left=0.085, right=0.92, top=0.53, bottom=0.315, width_ratios=[1, 1.05, 1.15],
                           wspace=0.62)
    c1, c2, c3 = (fig.add_subplot(mid[i]) for i in range(3))
    bot = fig.add_gridspec(1, 3, left=0.085, right=0.995, top=0.215, bottom=0.045, width_ratios=[1, 0.95, 1.05],
                           wspace=1.0)
    d1, d2, d3 = (fig.add_subplot(bot[i]) for i in range(3))
    panel_a(a)
    panel_b(b)
    panel_c(c1, c2, c3)
    panel_d(d1, d2, d3)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=COL[k], mec="#777777" if k == "other" else COL[k],
                               mew=0.3, ms=3.5, label=DXLAB[k])
                        for k in ["REF", "DKD", "HKD", "AKI", "DM_resilient", "other"]],
               ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=SMALL + 0.3,
               handletextpad=0.1, columnspacing=1.2, alignment="center")
    letters(fig, [[("a", [a]), ("b", b)], [("c", [c1, c2, c3])], [("d", [d1, d2, d3])]])
    fig.savefig(FIG / "Fig2.pdf")
    fig.savefig(FIG / "Fig2.png", dpi=300)
    print("wrote", FIG / "Fig2.pdf")


if __name__ == "__main__":
    main()
