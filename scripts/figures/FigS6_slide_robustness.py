"""Supplementary Fig 7：切片读出的稳健性——训练集与独立切片之间的采集差异（扫描年份、扫描仪、颜色）、
独立参与者 eGFR 关联按染色与采集变量校正、打乱标签读出的零分布（标出损伤 TAL、损伤 PT 与直接 eGFR 读出）。

只读 results/20_slide_robustness/{acquisition_by_set,acquisition_shift_tests,slide_acquisition,set_classifier_auc,
oos_egfr_robustness,null_permuted_label_readouts,null_random_projections,oos_egfr_null_comparison}.tsv；
绘图输入写到 figures/source_data/FigS6_*.tsv。输出 figures/FigS6.pdf/.png。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from FigS_common import RES, SRC, fmt_p, panel_letters, save

plt.rcParams.update({"font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"], "font.size": 6.5,
                     "axes.labelsize": 6.5, "axes.titlesize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "legend.fontsize": 5.8, "mathtext.default": "regular"})
R = RES / "20_slide_robustness"
SET_COL = {"training": "#64B5CD", "out_of_sample": "#CCB974"}
SET_LAB = {"training": "Training", "out_of_sample": "Independent"}
STAIN_COL = {"pred": "black", "pred_HE": "#DA8BC3", "pred_PAS": "#937860"}
STAIN_LAB = {"pred": "PAS + H&E (average)", "pred_HE": "H&E model", "pred_PAS": "PAS model"}
SMALL = 5.6
RNG = np.random.default_rng(20261012)


def p3(p: float) -> str:
    return fmt_p(p) if p < 0.001 else (f"P = {p:.3f}" if p < 0.1 else f"P = {p:.2f}")


def panel_a(ax) -> None:
    a = pd.read_csv(R / "acquisition_by_set.tsv", sep="\t")
    a = a[a.stain == "PAS"]
    a.to_csv(SRC / "FigS6_a_scan_year_scanner.tsv", sep="\t", index=False)
    t = pd.read_csv(R / "acquisition_shift_tests.tsv", sep="\t")
    t = t[t.variable.isin(["scan_time", "accession_time", "alt_scanner"])]
    t.to_csv(SRC / "FigS6_a_tests.tsv", sep="\t", index=False)
    years = list(range(int(a.scan_year.min()), int(a.scan_year.max()) + 1))
    for k, st in enumerate(["training", "out_of_sample"]):
        s = a[a.set == st]
        main = s[s.scanner_id == "SS7032"].groupby("scan_year").n_slides.sum().reindex(years, fill_value=0)
        alt = s[s.scanner_id != "SS7032"].groupby("scan_year").n_slides.sum().reindex(years, fill_value=0)
        x = np.arange(len(years)) + (k - 0.5) * 0.38
        ax.bar(x, main.values, width=0.36, color=SET_COL[st], edgecolor="white", linewidth=0.3)
        ax.bar(x, alt.values, bottom=main.values, width=0.36, color="white", edgecolor=SET_COL[st], hatch="/////",
               linewidth=0.6)
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([str(y) for y in years], rotation=45, ha="right")
    ax.set_xlabel("Scan year (PAS slides)")
    ax.set_ylabel("Slides")
    ts = t[t.stain == "PAS"].set_index("variable")
    ax.text(0.02, 0.98, f"Scan date {p3(ts.loc['scan_time', 'p'])}\nOther scanners {p3(ts.loc['alt_scanner', 'p'])}",
            transform=ax.transAxes, ha="left", va="top", fontsize=SMALL)
    ax.legend(handles=[Patch(color=SET_COL["training"], label="Training"),
                       Patch(color=SET_COL["out_of_sample"], label="Independent"),
                       Patch(facecolor="white", edgecolor="#555555", hatch="/////", label="Other scanner")],
              loc="upper left", bbox_to_anchor=(0.0, 0.80), frameon=True, fancybox=False, edgecolor="#CCCCCC",
              fontsize=SMALL, handlelength=1.2, borderpad=0.3, labelspacing=0.2)
    ax.set_title("Acquisition date and scanner", pad=4)


def panel_b(ax) -> None:
    s = pd.read_csv(R / "slide_acquisition.tsv", sep="\t", dtype={"participant": str})
    s = s[s.set.isin(["training", "out_of_sample"])][["participant", "set", "stain", "scanner_id", "scan_year",
                                                       "eosin_mean", "hema_mean"]]
    s.to_csv(SRC / "FigS6_b_slide_colour.tsv", sep="\t", index=False)
    t = pd.read_csv(R / "acquisition_shift_tests.tsv", sep="\t").set_index(["stain", "variable"])
    for i, st in enumerate(["HE", "PAS"]):
        for k, ss in enumerate(["training", "out_of_sample"]):
            v = s[(s.stain == st) & (s.set == ss)].eosin_mean.dropna().values * 100
            x = i * 2.6 + k
            ax.scatter(x + RNG.uniform(-0.18, 0.18, len(v)), v, s=5, color=SET_COL[ss], edgecolor="white",
                       linewidth=0.2, zorder=2)
            ax.hlines(np.median(v), x - 0.27, x + 0.27, color="black", lw=1.0, zorder=3)
        ax.text(i * 2.6 + 0.5, 0.99, p3(t.loc[(st, "eosin_mean"), "p"]), transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=SMALL)
    ax.set_xticks([0, 1, 2.6, 3.6])
    ax.set_xticklabels(["Train", "Indep.", "Train", "Indep."])
    for i, st in enumerate(["H&E", "PAS"]):
        ax.text(i * 2.6 + 0.5, -0.11, st, transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=6.3, fontweight="bold")
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, 4.2)
    ax.set_ylabel("Eosin channel intensity (a.u. × 100)")
    ax.set_ylim(0.2, 3.6)
    ax.set_title("Stain colour", pad=4)


def panel_c(ax) -> None:
    a = pd.read_csv(R / "set_classifier_auc.tsv", sep="\t")
    a.to_csv(SRC / "FigS6_c_set_classifier_auc.tsv", sep="\t", index=False)
    feats = [("colour_statistics", "Colour statistics"), ("embedding_mean", "Slide embedding")]
    for i, (f, lab) in enumerate(feats):
        for k, st in enumerate(["HE", "PAS"]):
            v = a[(a.features == f) & (a.stain == st)].cv_auc_training_vs_out_of_sample.iloc[0]
            x = i + (k - 0.5) * 0.36
            ax.bar(x, v, width=0.34, color=STAIN_COL["pred_HE" if st == "HE" else "pred_PAS"], edgecolor="white")
            ax.text(x, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=SMALL)
    ax.axhline(0.5, color="#999999", lw=0.5, ls="--")
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels([l for _, l in feats])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("Cross-validated AUC,\ntraining vs other slides")
    ax.legend(handles=[Patch(color=STAIN_COL["pred_HE"], label="H&E"), Patch(color=STAIN_COL["pred_PAS"], label="PAS")],
              loc="upper left", frameon=True, fancybox=False, edgecolor="#CCCCCC", fontsize=SMALL, handlelength=1,
              borderpad=0.3, labelspacing=0.2)
    ax.set_title("Separability of slide sets", pad=4)


def panel_d(ax) -> None:
    o = pd.read_csv(R / "oos_egfr_robustness.tsv", sep="\t")
    o.to_csv(SRC / "FigS6_d_oos_egfr_robustness.tsv", sep="\t", index=False)
    spec = [("all_out_of_sample", "diagnosis", "Diagnosis"),
            ("all_out_of_sample", "diagnosis+accession_year", "+ biopsy year"),
            ("all_out_of_sample", "diagnosis+scan_year", "+ scan year"),
            ("all_out_of_sample", "diagnosis+alt_scanner", "+ scanner"),
            ("all_out_of_sample", "diagnosis+colour_PCs", "+ colour components"),
            ("all_out_of_sample", "diagnosis+accession_year+alt_scanner+colour_PCs", "+ all acquisition variables"),
            ("training_scanner_only", "diagnosis", "Training scanner only"),
            ("colour_residualised_intensity", "diagnosis", "Stain intensity removed from features"),
            ("colour_residualised_all_colour", "diagnosis", "All colour removed from features")]
    off = {"pred": -0.24, "pred_HE": 0.0, "pred_PAS": 0.24}
    for i, (sub, mod, _) in enumerate(spec):
        for rd in ["pred", "pred_HE", "pred_PAS"]:
            r = o[(o.readout == rd) & (o.subset == sub) & (o.model == mod)].iloc[0]
            y = i + off[rd]
            ax.plot([r.ci_low, r.ci_high], [y, y], color=STAIN_COL[rd], lw=0.8)
            ax.scatter(r.egfr_per_sd, y, s=9, color=STAIN_COL[rd], zorder=3,
                       edgecolor="black" if rd == "pred_PAS" else "none", linewidth=0.2)
        if i:
            ax.axhline(i - 0.5, color="#EEEEEE", lw=0.5, zorder=0)
        n = int(o[(o.readout == "pred") & (o.subset == sub) & (o.model == mod)].n.iloc[0])
        pr = o[(o.readout == "pred") & (o.subset == sub) & (o.model == mod)].p.iloc[0]
        ax.text(9.0, i - 0.24, f"{p3(pr)} (n = {n})", va="center", ha="left", fontsize=SMALL - 0.2)
    ax.axvline(0, color="#999999", lw=0.5, ls="--")
    ax.set_yticks(range(len(spec)))
    ax.set_yticklabels([s[2] for s in spec], fontsize=SMALL)
    ax.set_ylim(len(spec) - 0.5, -1.1)
    ax.set_xlim(-24, 22)
    ax.spines["bottom"].set_bounds(-20, 10)
    ax.set_xticks([-20, -10, 0, 10])
    ax.set_xlabel("eGFR difference per SD of slide read-out\n(ml/min/1.73 m², 95% CI)")
    ax.legend(handles=[Line2D([], [], marker="o", color=STAIN_COL[k], ms=3, lw=0.8, label=STAIN_LAB[k])
                       for k in ["pred", "pred_HE", "pred_PAS"]],
              loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, frameon=False, fontsize=SMALL,
              handlelength=1.5, columnspacing=1.0)
    ax.set_title("Participants without snRNA-seq or scRNA-seq", pad=4)
    ax.text(9.0, -0.72, "Average read-out", ha="left", va="bottom", fontsize=SMALL - 0.2, style="italic")


def panel_e(ax) -> None:
    nl = pd.read_csv(R / "null_permuted_label_readouts.tsv", sep="\t")
    rp = pd.read_csv(R / "null_random_projections.tsv", sep="\t")
    cmp = pd.read_csv(R / "oos_egfr_null_comparison.tsv", sep="\t")
    cmp = cmp[~cmp.readout.str.startswith("ecgc")]
    pd.concat([nl.assign(null="permuted_injured_TAL_labels"), rp.assign(null="random_projections")]).to_csv(
        SRC / "FigS6_e_null_draws.tsv", sep="\t", index=False)
    cmp.to_csv(SRC / "FigS6_e_observed_vs_null.tsv", sep="\t", index=False)
    bins = np.linspace(-4.6, 4.6, 47)
    ax.hist(nl.t, bins=bins, color="#C8C8C8", edgecolor="white", linewidth=0.3, label="Permuted-label read-outs")
    ax.hist(rp.t, bins=bins, histtype="step", color="#555555", lw=0.7, label="Random projections")
    c = cmp[cmp.null == "permuted_injured_TAL_labels"].set_index("readout")
    q = c.null_q95_abs_t.iloc[0]
    for s in (-q, q):
        ax.axvline(s, color="#888888", lw=0.5, ls=":")
    marks = [("tal_injured_logit", "Injured TAL", "#C44E52", 0.97, 0.1), ("pt_injured_logit", "Injured PT", "#555555", 0.62, -0.1),
             ("direct_eGFR_model", "Direct\neGFR\nmodel", "black", 0.97, 0.1)]
    for key, lab, col, yf, dx in marks:
        r = c.loc[key]
        ax.axvline(r.observed_t, color=col, lw=1.0)
        ax.text(r.observed_t + dx, yf, f"{lab}\nP = {r.p_abs_t_ge_observed:.3f}", transform=ax.get_xaxis_transform(),
                ha="left" if dx > 0 else "right", va="top", fontsize=SMALL, color=col, linespacing=1.1,
                bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none", alpha=0.85))
    ax.set_xlim(-5.9, 4.6)
    ax.set_xlabel("t statistic, eGFR on read-out (adjusted for diagnosis)")
    ax.set_ylabel("Read-outs")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False, fontsize=SMALL,
              handlelength=1.5)
    ax.set_title("Read-outs trained on permuted labels", pad=4)


def main() -> None:
    fig = plt.figure(figsize=(6.6, 5.6))
    g1 = fig.add_gridspec(1, 3, left=0.08, right=0.98, top=0.93, bottom=0.62, width_ratios=[1.2, 1, 0.9], wspace=0.5)
    a, b, c = (fig.add_subplot(g1[i]) for i in range(3))
    g2 = fig.add_gridspec(1, 2, left=0.27, right=0.98, top=0.47, bottom=0.14, width_ratios=[1, 1.05], wspace=0.42)
    d, e = fig.add_subplot(g2[0]), fig.add_subplot(g2[1])
    panel_a(a)
    panel_b(b)
    panel_c(c)
    panel_d(d)
    panel_e(e)
    panel_letters(fig, [[("a", a), ("b", b), ("c", c)], [("d", d), ("e", e)]])
    save(fig, "FigS6")


if __name__ == "__main__":
    main()
