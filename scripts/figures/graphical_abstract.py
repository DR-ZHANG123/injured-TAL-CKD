"""Graphical abstract（Human Genomics 投稿必需：横版，约 920 × 300 像素，PNG/JPEG/SVG）。

复用 Fig 1 的插图（figures/source_data/Fig1_icon_*.png），只读结果表中的数字：
results/16_fig_tables/sn_vs_sc_agreement.tsv、results/15_key_genes/aTAL_frTAL_key_genes.tsv、results/11_histology/{cv_prediction,clinical_associations}.tsv。
输出：figures/graphical_abstract.{png,svg,pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES, SRC, FIG = ROOT / "results", ROOT / "figures" / "source_data", ROOT / "figures"
plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"], "pdf.fonttype": 42,
                     "svg.fonttype": "none"})
RED, BLUE, GREY, INK = "#C44E52", "#4C72B0", "#8C8C8C", "#222222"


def numbers() -> dict:
    fr = pd.read_csv(SRC / "Fig2_a_participants.tsv", sep="\t")
    med = lambda m: float(fr.loc[m, "injured_frac"].median()) * 100
    perf = pd.read_csv(SRC / "Fig5_b_performance.tsv", sep="\t").set_index("setting")
    mg = pd.read_csv(SRC / "Fig4_a_magma.tsv", sep="\t")
    healthy = mg[(mg.trait == "egfr_EA") & (mg.model == "default") & (mg.gene_set == "tal_healthy_specific")].iloc[0]
    genes = pd.read_csv(SRC / "Fig3_b_genes.tsv", sep="\t")
    lfc = genes[genes.dataset == "KPMP_snRNA_discovery"].set_index("gene").log2fc
    ag = pd.read_csv(RES / "21_confounders" / "sn_sc_agreement_lineages.tsv", sep="\t")
    icc = float(ag[(ag.subset == "all") & (ag.lineage == "TAL")].icc_absolute_logit.iloc[0])
    return {"dkd": med((fr.dx == "DKD")), "hkd": med((fr.dx == "HKD")), "ref": med((fr.dx == "REF") & fr.percutaneous),
            "r": float(perf.loc["All participants", "r"]), "r_path": float(perf.loc["Pathologist scores only", "r"]),
            "p_gen": float(10 ** -healthy.neglog10_p), "lfc": lfc, "icc": icc}


def icon(fig, name: str, box: tuple) -> None:
    ax = fig.add_axes(box)
    ax.imshow(Image.open(SRC / f"Fig1_icon_{name}.png"))
    ax.axis("off")


def card(fig, x: float, w: float, title: str, color: str) -> None:
    fig.patches.append(FancyBboxPatch((x, 0.06), w, 0.86, boxstyle="round,pad=0,rounding_size=0.025",
                                      transform=fig.transFigure, fc="white", ec="#C8C8CF", lw=0.8, zorder=-10))
    fig.text(x + w / 2, 0.86, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color=color)


def arrow(fig, x0: float, x1: float) -> None:
    fig.patches.append(FancyArrowPatch((x0, 0.5), (x1, 0.5), transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=12, lw=1.4, color="#555555"))


def main() -> None:
    n = numbers()
    fig = plt.figure(figsize=(9.205, 3.0), dpi=100)
    W, gap = 0.225, 0.028
    xs = [0.012 + i * (W + gap) for i in range(4)]
    card(fig, xs[0], W, "Kidney biopsies", INK)
    icon(fig, "kidney", (xs[0] + 0.01, 0.26, 0.10, 0.52))
    icon(fig, "nephron", (xs[0] + 0.115, 0.26, 0.10, 0.52))
    fig.text(xs[0] + W / 2, 0.17, "Diabetic and hypertensive\nkidney disease", ha="center", va="center",
             fontsize=8, color=INK)

    card(fig, xs[1], W, "Injured TAL cells expand", RED)
    icon(fig, "singlecell", (xs[1] + 0.015, 0.47, W - 0.03, 0.30))
    ax = fig.add_axes((xs[1] + 0.05, 0.20, W - 0.10, 0.22))
    ax.bar([0, 1, 2], [n["ref"], n["dkd"], n["hkd"]], color=[GREY, RED, BLUE], width=0.6)
    for x, v in enumerate([n["ref"], n["dkd"], n["hkd"]]):
        ax.text(x, v + 1, f"{v:.0f}%", ha="center", va="bottom", fontsize=6.6)
    ax.set_xticks([0, 1, 2], ["Ref", "DKD", "HKD"], fontsize=6.6)
    ax.set_yticks([])
    ax.set_ylim(0, max(n["dkd"], n["hkd"]) * 1.35)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    fig.text(xs[1] + W / 2, 0.10, f"Highest cross-method agreement (ICC {n['icc']:.2f})", ha="center", va="center",
             fontsize=6.4, color=RED, fontweight="bold")

    card(fig, xs[2], W, "TAL function is lost", INK)
    ax = fig.add_axes((xs[2] + 0.07, 0.40, W - 0.10, 0.36))
    gl = ["EGF", "UMOD", "SLC12A1", "CD44", "MMP7", "HAVCR1"]
    vals = [float(n["lfc"][g]) for g in gl]
    ax.barh(range(len(gl))[::-1], vals, color=[BLUE if v < 0 else RED for v in vals], height=0.65)
    ax.set_yticks(range(len(gl))[::-1], gl, fontsize=6.4, style="italic")
    ax.axvline(0, color="#555555", lw=0.6)
    ax.set_xticks([-3, 0, 3], ["−3", "0", "3"], fontsize=6.2)
    ax.set_xlabel("log2FC, injured vs healthy", fontsize=6.2, labelpad=1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    pg = f"{n['p_gen']:.1e}".split("e")
    fig.text(xs[2] + W / 2, 0.13, "Kidney-function genetics concentrates\nin differentiated TAL identity",
             ha="center", va="center", fontsize=6.4, color="#2E7D4F", fontweight="bold")

    card(fig, xs[3], W, "Read from routine slides", BLUE)
    icon(fig, "slide", (xs[3] + 0.01, 0.42, 0.105, 0.34))
    icon(fig, "model", (xs[3] + 0.115, 0.42, 0.10, 0.34))
    fig.text(xs[3] + W / 2, 0.27, "Foundation model on PAS and H&E\nslides approximates injured TAL", ha="center",
             va="center", fontsize=7.2, color=INK)
    fig.text(xs[3] + W / 2, 0.12, f"r = {n['r']:.2f} (pathologist scores {n['r_path']:.2f})", ha="center",
             va="center", fontsize=7.2, color=BLUE, fontweight="bold")
    for i in range(3):
        arrow(fig, xs[i] + W + 0.002, xs[i + 1] - 0.002)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(FIG / f"graphical_abstract.{ext}", dpi=100)
    plt.close(fig)
    print("wrote graphical_abstract (920 x 300 px)")


if __name__ == "__main__":
    main()
