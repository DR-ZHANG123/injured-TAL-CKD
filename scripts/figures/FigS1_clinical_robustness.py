"""Supplementary Fig 6：损伤 TAL 比例的临床稳健性——参考组织取材方式、年龄/性别/取材校正的诊断效应、
CKD 内 eGFR 关联（按 DKD/HKD 分开）、只用皮质 TAL 分母的一致性。

只读 results/21_confounders/{participants_snRNA,procurement_groups,procurement_contrasts,dx_models,egfr_models}.tsv、
results/22_tal_states/{participant_fractions,cortical_vs_main,group_contrasts,egfr_associations}.tsv；
绘图输入写到 figures/source_data/FigS1_*.tsv。输出 figures/FigS1.pdf/.png。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from FigS_common import LEGEND_KW, RES, SRC, fmt_p, panel_letters, save

plt.rcParams.update({"font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"], "font.size": 6.5,
                     "axes.labelsize": 6.5, "axes.titlesize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "legend.fontsize": 5.8, "mathtext.default": "regular"})
COL = {"REF": "#8C8C8C", "DKD": "#C44E52", "HKD": "#4C72B0", "AKI": "#DD8452", "DM_resilient": "#55A868",
       "other": "#D9D9D9", "CKD": "#555555", "all": "black"}
SMALL = 5.6
RNG = np.random.default_rng(20261011)
PROC = ["Percutaneous Needle Biopsy", "Transplant Pre-perfusion Biopsy", "Nephrectomy",
        "Deceased Donor Nephrectomy", "Intra-operative Biopsy"]
PROC_LAB = {"Percutaneous Needle Biopsy": "Percutaneous\nbiopsy", "Transplant Pre-perfusion Biopsy": "Pre-perfusion\ntransplant",
            "Nephrectomy": "Tumour\nnephrectomy", "Deceased Donor Nephrectomy": "Deceased-\ndonor\nnephrectomy",
            "Intra-operative Biopsy": "Intra-\noperative\nbiopsy"}


def sqrt_y(ax, ticks=(0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75)) -> None:
    ax.set_yticks(np.sqrt(ticks))
    ax.set_yticklabels([f"{t:g}" for t in ticks])


def p3(p: float) -> str:
    return fmt_p(p).replace("P = 0.0", "P = 0.0") if p < 0.001 else (f"P = {p:.3f}" if p < 0.1 else f"P = {p:.2f}")


def panel_a(ax) -> None:
    p = pd.read_csv(RES / "21_confounders/participants_snRNA.tsv", sep="\t", dtype={"donor_id": str})
    pc = pd.read_csv(RES / "21_confounders/procurement_contrasts.tsv", sep="\t")
    pc = pc[(pc.modality == "snRNA") & (pc.lineage == "TAL")]
    d = p.dropna(subset=["tal_injured_frac"])[["donor_id", "dx", "proc", "age10", "male", "tal_injured_frac"]]
    d.to_csv(SRC / "FigS1_a_participants.tsv", sep="\t", index=False)
    pc.to_csv(SRC / "FigS1_a_contrasts.tsv", sep="\t", index=False)
    pcs = pc.set_index("contrast")
    cats = [("REF", pr) for pr in PROC] + [("DKD", None), ("HKD", None)]
    ns = []
    for i, (dx, pr) in enumerate(cats):
        v = d[(d.dx == dx) & ((d.proc == pr) if pr else True)].tal_injured_frac.values
        x = i + (0.4 if dx != "REF" else 0)
        mk = "D" if pr == "Percutaneous Needle Biopsy" else "o"
        ax.scatter(x + RNG.uniform(-0.17, 0.17, len(v)), np.sqrt(v), s=8, marker=mk,
                   color="white" if mk == "D" else COL[dx], edgecolor="black" if mk == "D" else "white",
                   linewidth=0.4, zorder=2)
        ax.hlines(np.sqrt(np.median(v)), x - 0.26, x + 0.26, color="black", lw=1.0, zorder=3)
        key = (f"REF:{pr} vs REF:Percutaneous Needle Biopsy" if dx == "REF"
               else f"{dx} vs REF:Percutaneous Needle Biopsy")
        if key in pcs.index:
            r = pcs.loc[key]
            ax.text(x, 0.92, f"g = {r.hedges_g_logit:.2f}".replace("-", "−") + f"\n{p3(r.p)}", ha="center",
                    va="bottom", fontsize=SMALL, linespacing=1.1)
        ns.append(len(v))
    xs = [i + (0.4 if dx != "REF" else 0) for i, (dx, _) in enumerate(cats)]
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{PROC_LAB[pr] if pr else dx}\nn = {n}" for (dx, pr), n in zip(cats, ns)], fontsize=SMALL)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_xlim(-0.6, xs[-1] + 0.6)
    ax.set_ylim(0, 1.12)
    ax.spines["left"].set_bounds(0, np.sqrt(0.75))
    sqrt_y(ax)
    ax.set_ylabel("Injured TAL fraction (snRNA)")
    ax.plot([-0.4, 4.4], [-0.33, -0.33], transform=ax.get_xaxis_transform(), color="black", lw=0.5, clip_on=False)
    kw = pcs.loc["REF procurement types (Kruskal-Wallis)"]
    ax.text(2, -0.35, f"Reference tissue by procurement (Kruskal–Wallis {p3(kw.p)})", ha="center", va="top",
            fontsize=SMALL, transform=ax.get_xaxis_transform())
    ax.text(0.0, 1.0, "g and P vs percutaneous reference", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=SMALL, style="italic")


def panel_b(ax) -> None:
    dm = pd.read_csv(RES / "21_confounders/dx_models.tsv", sep="\t")
    dm = dm[(dm.lineage == "TAL") & dm.term.isin(["DKD_vs_REF", "HKD_vs_REF", "DKD_vs_HKD"])]
    models = [("all_procurement", "none", "Unadjusted"), ("all_procurement", "age+sex", "Age, sex"),
              ("all_procurement", "age+sex+procurement", "Age, sex, procurement"),
              ("percutaneous_only", "age+sex", "Percutaneous only; age, sex")]
    terms = [("DKD_vs_REF", "DKD vs reference", COL["DKD"]), ("HKD_vs_REF", "HKD vs reference", COL["HKD"]),
             ("DKD_vs_HKD", "DKD vs HKD", "#555555")]
    rows, y, ticks, labs = [], 0.75, [], []
    for t, tlab, col in terms:
        for smp, adj, mlab in models:
            r = dm[(dm["sample"] == smp) & (dm.adjustment == adj) & (dm.term == t)].iloc[0]
            ax.plot([r.ci_low, r.ci_high], [y, y], color=col, lw=0.9)
            ax.scatter(r.estimate, y, s=14, color=col, zorder=3)
            ax.text(3.05, y, p3(r.p), va="center", ha="left", fontsize=SMALL)
            rows.append(r.to_dict() | {"model": mlab.replace("\n", " ")})
            ticks.append(y)
            labs.append(mlab)
            y += 1
        ax.text(-0.02, y - 4.75, tlab, ha="right", va="center", fontsize=SMALL + 0.3, fontweight="bold", color=col,
                transform=ax.get_yaxis_transform())
        y += 1.0
    pd.DataFrame(rows).to_csv(SRC / "FigS1_b_dx_models.tsv", sep="\t", index=False)
    ax.axvline(0, color="#999999", lw=0.5, ls="--")
    ax.set_yticks(ticks)
    ax.set_yticklabels(labs, fontsize=SMALL)
    ax.set_ylim(y - 0.7, -0.5)
    ax.set_xlim(-0.95, 3.0)
    ax.set_xlabel("Difference in injured TAL (logit), 95% CI")
    ax.set_title("Diagnosis effects after adjustment (n = 142; 68)", pad=4)


def panel_c(ax) -> None:
    p = pd.read_csv(RES / "21_confounders/participants_snRNA.tsv", sep="\t", dtype={"donor_id": str})
    em = pd.read_csv(RES / "21_confounders/egfr_models.tsv", sep="\t")
    d = p[p.dx.isin(["DKD", "HKD"])].dropna(subset=["tal_injured_frac", "egfr"])[
        ["donor_id", "dx", "egfr", "tal_injured_frac", "age10", "male"]]
    d.to_csv(SRC / "FigS1_c_ckd_participants.tsv", sep="\t", index=False)
    for g in ["HKD", "DKD"]:
        s = d[d.dx == g]
        ax.scatter(s.egfr + RNG.uniform(-2.5, 2.5, len(s)), np.sqrt(s.tal_injured_frac), s=10, color=COL[g],
                   edgecolor="white", linewidth=0.3, zorder=2, alpha=0.9)
    em = em[em.exposure == "TAL_injured_logit_perSD"]
    lines = []
    for g in ["DKD", "HKD"]:
        r = em[(em.subset == g) & (em.adjustment == "none")].iloc[0]
        lines.append((g, f"{g}: {r.estimate:.1f} per SD, {p3(r.p)} (n = {int(r.n)})".replace("-", "−")))
    for k, (g, t) in enumerate(lines):
        ax.text(0.98, 0.99 - k * 0.075, t, transform=ax.transAxes, ha="right", va="top", fontsize=SMALL,
                color=COL[g])
    ax.set_xlabel("eGFR (ml/min/1.73 m², band midpoint)")
    ax.set_ylabel("Injured TAL fraction (snRNA)")
    ax.set_ylim(0, 1.06)
    ax.spines["left"].set_bounds(0, np.sqrt(0.75))
    sqrt_y(ax)
    ax.set_xlim(15, 125)
    ax.set_title("Participants with DKD or HKD", pad=4)


def panel_d(ax) -> None:
    em = pd.read_csv(RES / "21_confounders/egfr_models.tsv", sep="\t")
    em = em[(em.exposure == "TAL_injured_logit_perSD") & (em.adjustment != "spearman")]
    spec = [("all", "dx", "Ref + DKD + HKD", "diagnosis"), ("all", "dx+age+sex", None, "+ age, sex"),
            ("all", "dx+age+sex+procurement", None, "+ age, sex, procurement"),
            ("CKD", "dx", "DKD + HKD", "diagnosis"), ("CKD", "dx+age+sex", None, "+ age, sex"),
            ("DKD", "none", "DKD", "unadjusted"), ("DKD", "age+sex", None, "age, sex"),
            ("HKD", "none", "HKD", "unadjusted"), ("HKD", "age+sex", None, "age, sex"),
            ("REF", "none", "Reference", "unadjusted"), ("REF", "age+sex", None, "age, sex"),
            ("REF", "age+sex+procurement", None, "age, sex, procurement")]
    colmap = {"all": "black", "CKD": "#555555", "DKD": COL["DKD"], "HKD": COL["HKD"], "REF": COL["REF"]}
    rows, y, ticks, labs = [], 0, [], []
    for sub, adj, head, lab in spec:
        if head is not None and y > 0:
            y += 0.6
        r = em[(em.subset == sub) & (em.adjustment == adj)].iloc[0]
        c = colmap[sub]
        ax.plot([r.ci_low, r.ci_high], [y, y], color=c, lw=0.9)
        ax.scatter(r.estimate, y, s=13, color=c, zorder=3)
        ax.text(16.5, y, p3(r.p), va="center", ha="left", fontsize=SMALL)
        ticks.append(y)
        labs.append(f"{head} (n = {int(r.n)}): {lab}" if head else lab)
        rows.append(r.to_dict() | {"label": labs[-1]})
        y += 1
    pd.DataFrame(rows).to_csv(SRC / "FigS1_d_egfr_models.tsv", sep="\t", index=False)
    ax.axvline(0, color="#999999", lw=0.5, ls="--")
    ax.set_yticks(ticks)
    ax.set_yticklabels(labs, fontsize=SMALL)
    for t, (sub, _, head, _) in zip(ax.get_yticklabels(), spec):
        t.set_color(colmap[sub])
    ax.set_ylim(y - 0.4, -0.6)
    ax.set_xlim(-30, 16)
    ax.set_xlabel("eGFR difference per SD injured TAL\n(ml/min/1.73 m², 95% CI)")
    ax.set_title("eGFR association by population", pad=4)


def panel_e(ax) -> None:
    pf = pd.read_csv(RES / "22_tal_states/participant_fractions.tsv", sep="\t", dtype={"participant": str})
    cv = pd.read_csv(RES / "22_tal_states/cortical_vs_main.tsv", sep="\t")
    gc = pd.read_csv(RES / "22_tal_states/group_contrasts.tsv", sep="\t")
    ea = pd.read_csv(RES / "22_tal_states/egfr_associations.tsv", sep="\t")
    d = pf[pf.modality == "snRNA"].dropna(subset=["injured_frac", "injured_frac_cortical"])
    d[["participant", "dx", "n_tal", "n_tal_cortical", "injured_frac", "injured_frac_cortical"]].to_csv(
        SRC / "FigS1_e_cortical_vs_main.tsv", sep="\t", index=False)
    gcs = gc[(gc.modality == "snRNA") & (gc.measure == "injured_frac_cortical")]
    eas = ea[(ea.cohort == "snRNA") & (ea.measure == "injured_frac_cortical") & (ea.subset == "all")]
    pd.concat([cv.assign(table="cortical_vs_main"), gcs.assign(table="group_contrasts"),
               eas.assign(table="egfr_associations")], ignore_index=True).to_csv(
        SRC / "FigS1_e_stats.tsv", sep="\t", index=False)
    for g in ["other", "REF", "DM_resilient", "AKI", "HKD", "DKD"]:
        s = d[d.dx == g]
        ax.scatter(np.sqrt(s.injured_frac), np.sqrt(s.injured_frac_cortical), s=7, color=COL[g],
                   edgecolor="#777777" if g == "other" else "white", linewidth=0.3, zorder=2, alpha=0.9)
    ax.plot([0, 1], [0, 1], color="#999999", lw=0.5, ls="--", zorder=1)
    tk = [0, 0.05, 0.25, 0.5, 1.0]
    ax.set_xticks(np.sqrt(tk))
    ax.set_yticks(np.sqrt(tk))
    ax.set_xticklabels([f"{t:g}" for t in tk])
    ax.set_yticklabels([f"{t:g}" for t in tk])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Injured fraction of all TAL")
    ax.set_ylabel("Injured fraction of cortical TAL")
    a = cv[(cv.modality == "snRNA") & (cv.subset == "all")].iloc[0]
    g = gcs.set_index("contrast")
    e = eas.iloc[0]
    txt = (rf"$\rho$ = {a.spearman:.2f} (n = {int(a.n)})" + "\n"
           + f"DKD vs Ref g = {g.loc['DKD_vs_REF', 'hedges_g_logit']:.2f}\n"
           + f"HKD vs Ref g = {g.loc['HKD_vs_REF', 'hedges_g_logit']:.2f}\n"
           + f"eGFR {p3(e.p)}")
    ax.text(0.98, 0.03, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=SMALL, linespacing=1.15)
    ax.set_title("Cortical-only denominator", pad=4)


def main() -> None:
    fig = plt.figure(figsize=(6.6, 7.7))
    g1 = fig.add_gridspec(1, 1, left=0.08, right=0.99, top=0.92, bottom=0.73)
    a = fig.add_subplot(g1[0])
    g2 = fig.add_gridspec(1, 2, left=0.25, right=0.97, top=0.585, bottom=0.355, width_ratios=[1.25, 1], wspace=0.55)
    b, c = fig.add_subplot(g2[0]), fig.add_subplot(g2[1])
    g3 = fig.add_gridspec(1, 2, left=0.30, right=0.97, top=0.245, bottom=0.06, width_ratios=[1.35, 0.8], wspace=0.55)
    d, e = fig.add_subplot(g3[0]), fig.add_subplot(g3[1])
    panel_a(a)
    panel_b(b)
    panel_c(c)
    panel_d(d)
    panel_e(e)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=COL[k], mec="#777777" if k == "other" else COL[k],
                               mew=0.3, ms=3.5, label=l)
                        for k, l in [("REF", "Reference"), ("DKD", "DKD"), ("HKD", "HKD"), ("AKI", "AKI"),
                                     ("DM_resilient", "Diabetes without CKD"), ("other", "Other")]],
               ncol=6, loc="upper center", bbox_to_anchor=(0.5, 0.005), frameon=False, fontsize=5.9,
               handletextpad=0.1, columnspacing=1.2)
    panel_letters(fig, [[("a", a)], [("b", b), ("c", c)], [("d", d), ("e", e)]])
    save(fig, "FigS1")


if __name__ == "__main__":
    main()
