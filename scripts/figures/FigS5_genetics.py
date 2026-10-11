"""Supplementary Fig 4：肾功能 GWAS 的基因水平证据（MAGMA / 共定位 / SMR+HEIDI）、KPMP 细胞状态富集与方向检验。

只读 results/07f_genetic_convergence/genetic_evidence_by_gene.tsv、results/13_gwas_cell_enrichment/enrichment.tsv、
results/14_genetic_direction/direction_tests.tsv；绘图输入写到 figures/source_data/FigS5_*.tsv。
输出 figures/FigS5.pdf/.png。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from FigS_common import LEGEND_KW, RES, SRC, panel_letters, save

TRAIT = {"egfr_EA": "eGFR", "uacr_EA": "UACR", "egfr_DM": "eGFR, diabetes", "uacr_DM": "UACR, diabetes",
         "egfr_noDM": "eGFR, no diabetes"}
LINES = [("magma_bonf", "MAGMA gene test", "#4C72B0"), ("coloc_pp4_ge_flag", "Colocalization (PP4 ≥ 0.8)", "#DD8452"),
         ("smr_heidi_pass", "SMR with HEIDI P ≥ 0.01", "#55A868"), ("two_plus", "≥ 2 lines of evidence", "#C44E52")]
SMR_TISSUE = [("susztak_meta", "SMR whole kidney"), ("susztak_tubule", "SMR tubule"),
              ("susztak_glom", "SMR glomerulus")]


def evidence() -> pd.DataFrame:
    e = pd.read_csv(RES / "07f_genetic_convergence" / "genetic_evidence_by_gene.tsv", sep="\t")
    e["smr_heidi_pass_tissues"] = e["smr_heidi_pass_tissues"].fillna("")
    return e


def panel_a(ax) -> None:
    e = evidence()
    e["two_plus"] = e["n_lines"] >= 2
    traits = ["egfr_EA", "uacr_EA", "egfr_DM", "uacr_DM"]
    cnt = e.groupby("trait")[[c for c, _, _ in LINES]].sum().reindex(traits)
    cnt.reset_index().to_csv(SRC / "FigS5_a_gene_counts.tsv", sep="\t", index=False)
    h = 0.19
    for k, (c, lab, col) in enumerate(LINES):
        y = np.arange(len(traits)) + (k - 1.5) * h
        v = cnt[c].values
        ax.barh(y, np.where(v > 0, v, np.nan), height=h * 0.95, color=col, edgecolor="white", linewidth=0.3,
                label=lab)
        for yi, vi in zip(y, v):
            ax.text(max(vi, 0.8) * 1.15, yi, str(int(vi)), va="center", fontsize=5.5)
    ax.set_xscale("log")
    ax.set_xlim(0.8, 6000)
    ax.set_yticks(range(len(traits)))
    ax.set_yticklabels([TRAIT[t] for t in traits])
    ax.set_ylim(len(traits) - 0.5, -0.5)
    ax.set_xlabel("Genes supported (Bonferroni or threshold)", fontweight="bold")
    ax.legend(loc="lower right", fontsize=5.6, handlelength=1.0, labelspacing=0.3, **LEGEND_KW)


def select_genes(e: pd.DataFrame) -> list[str]:
    keep = set(e.loc[(e.trait.isin(["uacr_EA", "egfr_DM"])) & (e.n_lines >= 2), "symbol"])
    eg = e[(e.trait == "egfr_EA") & (e.n_lines >= 2)].sort_values(["n_lines", "magma_p"], ascending=[False, True])
    keep |= set(eg.symbol.head(18))
    keep |= set(e.loc[e.coloc_pp4_ge_flag & ~e.symbol.str.startswith(("LINC", "ENSG")), "symbol"])
    return sorted(keep)


def panel_b(ax) -> None:
    e = evidence()
    genes = select_genes(e)
    traits = ["egfr_EA", "uacr_EA", "egfr_DM"]
    cols = []
    for t in traits:
        cols += [(t, "magma", "MAGMA"), (t, "coloc", "Coloc.")] + [(t, s, lab) for s, lab in SMR_TISSUE]
    rows = []
    for g in genes:
        for j, (t, kind, _) in enumerate(cols):
            r = e[(e.trait == t) & (e.symbol == g)]
            if r.empty:
                hit = False
            elif kind == "magma":
                hit = bool(r.magma_bonf.iloc[0])
            elif kind == "coloc":
                hit = bool(r.coloc_pp4_ge_flag.iloc[0])
            else:
                hit = kind in r.smr_heidi_pass_tissues.iloc[0].split(";")
            rows.append({"gene": g, "trait": t, "evidence": kind, "col": j, "hit": hit})
    m = pd.DataFrame(rows)
    score = m.groupby("gene")["hit"].sum().sort_values(ascending=False)
    order = sorted(genes, key=lambda g: (-score[g], g))
    m["row"] = m["gene"].map(order.index)
    m.to_csv(SRC / "FigS5_b_gene_evidence_matrix.tsv", sep="\t", index=False)
    colour = {"magma": "#4C72B0", "coloc": "#DD8452"}
    for _, r in m.iterrows():
        c = colour.get(r.evidence, "#55A868")
        if r.hit:
            ax.scatter(r.col, r.row, s=16, marker="s", color=c, edgecolor="none")
        else:
            ax.scatter(r.col, r.row, s=3, marker="s", color="#E3E3E3", edgecolor="none")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=5.6, fontstyle="italic")
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[2] for c in cols], rotation=90, fontsize=5.6)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for k, t in enumerate(traits):
        ax.text(k * 5 + 2, -0.004, TRAIT[t], transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=6.3, fontweight="bold")
        if k:
            ax.axvline(k * 5 - 0.5, color="#AAAAAA", lw=0.6)
    ax.set_xlim(-0.6, len(cols) - 0.4)


def panel_c(ax, cax) -> None:
    en = pd.read_csv(RES / "13_gwas_cell_enrichment" / "enrichment.tsv", sep="\t")
    traits = ["egfr_EA", "egfr_noDM", "egfr_DM", "uacr_EA", "uacr_DM"]
    key = ["PT-S1", "PT-S2", "aPT", "frPT", "C-TAL", "M-TAL", "aTAL", "frTAL", "dC-TAL", "aCNT", "dCNT",
           "POD", "EC-GC", "MC"]
    top = set(en.sort_values("rank").groupby("trait").head(2).subclass)
    subs = key + sorted(top - set(key))
    q = en[en.subclass.isin(subs) & en.trait.isin(traits)].copy()
    q.to_csv(SRC / "FigS5_c_cell_enrichment.tsv", sep="\t", index=False)
    mat = q.pivot(index="subclass", columns="trait", values="p_one_sided").reindex(index=subs, columns=traits)
    rk = q.pivot(index="subclass", columns="trait", values="rank").reindex(index=subs, columns=traits)
    im = ax.imshow(-np.log10(mat.values), cmap="Purples", vmin=0, vmax=3.5, aspect="auto")
    for i in range(len(subs)):
        for j in range(len(traits)):
            v = -np.log10(mat.values[i, j])
            ax.text(j, i, int(rk.values[i, j]), ha="center", va="center", fontsize=5.3,
                    color="white" if v > 2.0 else "#222222")
    ax.set_xticks(range(len(traits)))
    ax.set_xticklabels([TRAIT[t] for t in traits], rotation=40, ha="right", rotation_mode="anchor", fontsize=6)
    ax.set_yticks(range(len(subs)))
    ax.set_yticklabels(subs, fontsize=6)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label("−log$_{10}$ P", fontsize=6)
    cb.ax.tick_params(labelsize=5.5, width=0.5, length=2)
    cb.outline.set_linewidth(0.5)


def panel_d(ax) -> None:
    d = pd.read_csv(RES / "14_genetic_direction" / "direction_tests.tsv", sep="\t")
    d.to_csv(SRC / "FigS5_d_direction_tests.tsv", sep="\t", index=False)
    pairs = list(dict.fromkeys(zip(d.state, d.trait)))
    sname = {"PT_injured": "Injured PT", "PT_normal": "Normal PT", "TAL_injured": "Injured TAL",
             "TAL_normal": "Normal TAL", "EC_GC": "EC-GC", "POD": "Podocyte"}
    ccol = {"DKD+HKD": "#555555", "DKD": "#C44E52", "HKD": "#4C72B0"}
    for i, (s, t) in enumerate(pairs):
        for k, (case, col) in enumerate(ccol.items()):
            r = d[(d.state == s) & (d.trait == t) & (d.case == case)].iloc[0]
            y = i + (k - 1) * 0.22
            ax.scatter(r.fraction_adverse, y, s=14, color=col, edgecolor="white", linewidth=0.3, zorder=3,
                       label=case.replace("+", " + ") if i == 0 else None)
            if r.p_perm < 0.05:
                ax.text(r.fraction_adverse + 0.02, y, f"P = {r.p_perm:.2f}", va="center", fontsize=5.3)
    ax.axvline(0.5, color="#555555", lw=0.6, ls="--")
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([f"{sname[s]}, {TRAIT[t]}" for s, t in pairs], fontsize=6)
    ax.set_ylim(len(pairs) - 0.5, -0.5)
    ax.set_xlim(0.2, 0.85)
    ax.set_xlabel("Fraction of SMR genes shifted\nin the risk direction", fontweight="bold")
    ax.legend(loc="lower right", fontsize=5.8, handletextpad=0.1, labelspacing=0.3, **LEGEND_KW)


def main() -> None:
    SRC.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6.6, 7.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[0.6, 1.0], hspace=0.22, wspace=0.55,
                          left=0.15, right=0.97, top=0.96, bottom=0.06)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[:, 1])
    low = gs[1, 0].subgridspec(2, 2, height_ratios=[1.3, 1.0], width_ratios=[1, 0.05], hspace=0.62, wspace=0.05)
    ax_c, cax = fig.add_subplot(low[0, 0]), fig.add_subplot(low[0, 1])
    ax_d = fig.add_subplot(low[1, 0])
    panel_a(ax_a)
    panel_b(ax_b)
    panel_c(ax_c, cax)
    panel_d(ax_d)
    handles = [Line2D([], [], marker="s", ls="", color=c, ms=4, label=l)
               for c, l in (("#4C72B0", "MAGMA, Bonferroni"), ("#DD8452", "Colocalization, PP4 ≥ 0.8"),
                            ("#55A868", "SMR, HEIDI P ≥ 0.01"))]
    ax_b.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=1, fontsize=5.8,
                **LEGEND_KW)
    panel_letters(fig, [[("a", ax_a), ("b", ax_b)], [("c", ax_c)], [("d", ax_d)]])
    save(fig, "FigS5")


if __name__ == "__main__":
    main()
