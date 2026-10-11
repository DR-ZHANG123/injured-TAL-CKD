"""补充图共用：字体与样式、配色、panel 字母对齐、保存。只做绘图，不做计算。"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
FIG = ROOT / "figures"
SRC = FIG / "source_data"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

GROUP_COL = {"DKD": "#C44E52", "HKD": "#4C72B0", "REF": "#8C8C8C", "CTRL": "#8C8C8C", "PGD": "#8172B3",
             "Control": "#8C8C8C", "OTHER_CKD": "#8C8C8C"}
LEGEND_KW = dict(frameon=True, fancybox=False, edgecolor="#CCCCCC", framealpha=1, borderpad=0.4)


def fmt_p(p: float) -> str:
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return f"P = {m}×10$^{{{int(e)}}}$"
    return f"P = {p:.3f}" if p < 0.01 else f"P = {p:.2f}"


def panel_letters(fig, rows: list[list], dy: float = 0.008) -> None:
    """rows：每行一个 (letter, ax) 列表。同一行字母顶端对齐；字母放在各 ax 含刻度标签的左外沿。
    跨行同一列（左缘相近，<0.03）的字母取最左值，使列内左缘对齐。"""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    items = []
    for row in rows:
        y1 = max(inv.transform((0, ax.get_tightbbox(rend).y1))[1] for _, ax in row)
        for letter, ax in row:
            items.append([letter, inv.transform((ax.get_tightbbox(rend).x0, 0))[0], y1])
    for it in items:   # 列对齐
        near = [j[1] for j in items if abs(j[1] - it[1]) < 0.03]
        it[1] = min(near)
    for letter, x, y in items:
        fig.text(max(x, 0.0), y + dy, letter, fontsize=11, fontweight="bold", ha="left", va="bottom")


def save(fig, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", FIG / f"{name}.pdf")
