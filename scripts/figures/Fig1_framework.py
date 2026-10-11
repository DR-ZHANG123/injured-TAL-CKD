"""Fig. 1 研究框架图（矢量 SVG → PDF/PNG）。

输入（只读）：figures/source_data/Fig1_icon_*.png（fig1_icons.py 生成的插图）；
  迷你示意图用到的数值来自其他主图的绘图输入：Fig2_a_participants.tsv（损伤 TAL 比例中位数，参考为经皮穿刺）、
  Fig3_b_genes.tsv（TAL 功能基因与损伤标志物 log2FC）、Fig4_c_stats.tsv（eGFR 匹配随机基因集 Z）、
  Fig5_a_cv_tal.tsv（CV 估计 vs 实测）、Fig6_a_tal_vs_egfr.tsv（损伤 TAL vs eGFR）、Fig6_c_albuminuria.tsv（白蛋白尿分级）。
输出：figures/Fig1.svg / Fig1.pdf（矢量，文字为可编辑 <text>，Liberation Sans）/ Fig1.png（300 dpi），
  figures/source_data/Fig1_layout.json（全部元素的位置与文字）。
修改文字或位置：编辑 Fig1_layout.json 后运行 `python scripts/figures/Fig1_framework.py --from-layout`，不需要重新生成图标。
坐标单位为 pt（宽 170 mm = 481.9 pt）。
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import cairosvg
import numpy as np
import pandas as pd
from scipy import ndimage
from PIL import Image, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig1_layout import build_layout  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures"
SRC = FIG / "source_data"
LAYOUT = SRC / "Fig1_layout.json"
FONT = "Liberation Sans"
FONT_FILES = {"normal": "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
              "bold": "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
              "italic": "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf"}
_FONTS: dict = {}


def text_width(s: str, size: float, weight: str = "normal", style: str = "normal") -> float:
    key = "bold" if weight == "bold" else ("italic" if style == "italic" else "normal")
    if key not in _FONTS:
        _FONTS[key] = ImageFont.truetype(FONT_FILES[key], 200)
    return _FONTS[key].getlength(s) * size / 200


def text_bbox(t: dict) -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1)；y 为基线，字高取 ascent 0.73、descent 0.21 em。旋转 -90 的文字按竖排计算。"""
    w = text_width(t["s"], t["size"], t.get("weight", "normal"), t.get("style", "normal"))
    if t.get("sup"):
        w += text_width(t["sup"], 0.8 * t["size"], t.get("weight", "normal"), t.get("style", "normal"))
    a, d = 0.73 * t["size"], 0.21 * t["size"]
    anchor = t.get("anchor", "start")
    off = {"start": 0, "middle": -w / 2, "end": -w}[anchor]
    if t.get("rotate") == -90:
        return t["x"] - a, t["y"] - off - w, t["x"] + d, t["y"] - off
    return t["x"] + off, t["y"] - a, t["x"] + off + w, t["y"] + d


# ---------------------------------------------------------------- icons
def icon_data(name: str, max_px: int = 700) -> tuple[str, float]:
    """裁掉白边、透明化近白背景（只透明化与边缘相连的白色），缩到 max_px，返回 base64 PNG 与宽高比。"""
    im = Image.open(SRC / f"Fig1_icon_{name}.png").convert("RGB")
    a = np.asarray(im).astype(int)
    fg = (a.min(axis=2) < 235)
    ys, xs = np.where(fg)
    pad = 6
    box = (max(xs.min() - pad, 0), max(ys.min() - pad, 0), min(xs.max() + pad, a.shape[1]), min(ys.max() + pad, a.shape[0]))
    im = im.crop(box)
    im = _transparent_bg(im)
    im.thumbnail((max_px, max_px), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode(), im.width / im.height


def _transparent_bg(im: Image.Image) -> Image.Image:
    """与图边相连的近白、低饱和像素设为透明（内部白色如载玻片保持不透明），边缘 1 px 羽化。"""
    a = np.asarray(im).astype(int)
    light = (a.min(axis=2) >= 236) & (a.max(axis=2) - a.min(axis=2) <= 10)
    lab, _ = ndimage.label(light)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    bg = np.isin(lab, edge[edge > 0])
    alpha = np.where(bg, 0, 255).astype(np.uint8)
    alpha = ndimage.uniform_filter(alpha.astype(float), 3).astype(np.uint8)
    return Image.fromarray(np.dstack([a.astype(np.uint8), alpha]), "RGBA")


def fit_image(el: dict, aspect: float) -> dict:
    """在目标框 (x, y, w, h) 内保持宽高比居中，写回实际位置。"""
    bw, bh = el["box"][2], el["box"][3]
    w, h = (bw, bw / aspect) if bw / aspect <= bh else (bh * aspect, bh)
    el.update(x=el["box"][0] + (bw - w) / 2, y=el["box"][1] + (bh - h) / 2, w=w, h=h)
    return el


# ---------------------------------------------------------------- data glyphs
def load_glyph_data() -> dict:
    """迷你示意图的数值全部取自其他主图的绘图输入（figures/source_data/），不重新计算模型。"""
    p = pd.read_csv(SRC / "Fig2_a_participants.tsv", sep="\t")
    ref = p[(p.dx == "REF") & (p.percutaneous.astype(str) == "True")]
    med = {"REF": float(ref.injured_frac.median()),
           "DKD": float(p[p.dx == "DKD"].injured_frac.median()),
           "HKD": float(p[p.dx == "HKD"].injured_frac.median())}
    assert [round(100 * med[k]) for k in ("REF", "DKD", "HKD")] == [4, 24, 14], med
    gb = pd.read_csv(SRC / "Fig3_b_genes.tsv", sep="\t")
    genes = ["EGF", "UMOD", "SLC12A1", "KCNJ1", "CD44", "MMP7", "HAVCR1", "LCN2"]
    sets = ["KPMP_snRNA_discovery", "KPMP_scRNA_independent", "GSE271188"]
    heat = gb.pivot(index="gene", columns="dataset", values="log2fc").loc[genes, sets]
    mg = pd.read_csv(SRC / "Fig4_c_stats.tsv", sep="\t")
    mg = mg[mg.trait == "egfr_EA"].set_index("gene_set")
    assert mg.loc["tal_healthy_specific", "p_empirical_greater"] <= 0.001 and round(mg.loc["tal_injured_up", "p_empirical_greater"], 2) == 0.50
    cv = pd.read_csv(SRC / "Fig5_a_cv_tal.tsv", sep="\t")
    eg = pd.read_csv(SRC / "Fig6_a_tal_vs_egfr.tsv", sep="\t")
    al = pd.read_csv(SRC / "Fig6_c_albuminuria.tsv", sep="\t")
    return {"tal_median": med,
            "heat": {g: [float(v) for v in heat.loc[g]] for g in genes},
            "magma": {"healthy": float(mg.loc["tal_healthy_specific", "z_vs_null"]),
                      "injury": float(mg.loc["tal_injured_up", "z_vs_null"])},
            "cv": {"x": cv.measured.tolist(), "y": cv.cv_predicted.tolist(), "dx": cv.diagnosis.tolist()},
            "egfr": {"x": eg.injured_tal_logit.tolist(), "y": eg.egfr_bin_midpoint.tolist(), "dx": eg.group.tolist()},
            "alb": {"x": al.x_jitter.tolist(), "y": al.injured_tal_logit.tolist(), "cat": al.alb_A.astype(int).tolist(),
                    "dx": al.group.tolist()}}


# ---------------------------------------------------------------- SVG
def svg_el(e: dict, icons: dict) -> str:
    k = e["type"]
    if k == "rect":
        dash = f' stroke-dasharray="{e["dash"]}"' if e.get("dash") else ""
        return (f'<rect x="{e["x"]:.2f}" y="{e["y"]:.2f}" width="{e["w"]:.2f}" height="{e["h"]:.2f}" rx="{e.get("rx", 0)}" '
                f'fill="{e.get("fill", "none")}" stroke="{e.get("stroke", "none")}" stroke-width="{e.get("sw", 0.6)}"{dash}/>')
    if k == "circle":
        return (f'<circle cx="{e["x"]:.2f}" cy="{e["y"]:.2f}" r="{e["r"]:.2f}" fill="{e.get("fill", "none")}" '
                f'stroke="{e.get("stroke", "none")}" stroke-width="{e.get("sw", 0.4)}" fill-opacity="{e.get("alpha", 1)}"/>')
    if k == "line":
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in e["pts"])
        dash = f' stroke-dasharray="{e["dash"]}"' if e.get("dash") else ""
        out = (f'<polyline points="{pts}" fill="none" stroke="{e.get("stroke", "#555")}" stroke-width="{e.get("sw", 0.8)}" '
               f'stroke-linecap="round" stroke-linejoin="round"{dash}/>')
        if e.get("head"):
            (x0, y0), (x1, y1) = e["pts"][-2], e["pts"][-1]
            ang = np.arctan2(y1 - y0, x1 - x0)
            L, W = e.get("hl", 4.2), e.get("hw", 2.0)
            p = [(x1, y1), (x1 - L * np.cos(ang) + W * np.sin(ang), y1 - L * np.sin(ang) - W * np.cos(ang)),
                 (x1 - L * np.cos(ang) - W * np.sin(ang), y1 - L * np.sin(ang) + W * np.cos(ang))]
            out += f'<polygon points="{" ".join(f"{a:.2f},{b:.2f}" for a, b in p)}" fill="{e.get("stroke", "#555")}"/>'
        return out
    if k == "path":
        return (f'<path d="{e["d"]}" fill="{e.get("fill", "none")}" stroke="{e.get("stroke", "none")}" '
                f'stroke-width="{e.get("sw", 0.6)}" stroke-linecap="round" stroke-linejoin="round"/>')
    if k == "image":
        return (f'<image x="{e["x"]:.2f}" y="{e["y"]:.2f}" width="{e["w"]:.2f}" height="{e["h"]:.2f}" '
                f'preserveAspectRatio="xMidYMid meet" href="data:image/png;base64,{icons[e["icon"]]}"/>')
    if k == "text":
        rot = f' transform="rotate(-90 {e["x"]:.2f} {e["y"]:.2f})"' if e.get("rotate") == -90 else ""
        sup = (f'<tspan font-size="{0.8 * e["size"]:.2f}" dy="{-0.38 * e["size"]:.2f}">{escape(e["sup"])}</tspan>'
               if e.get("sup") else "")
        return (f'<text x="{e["x"]:.2f}" y="{e["y"]:.2f}" font-family="{FONT}" font-size="{e["size"]}" '
                f'font-weight="{e.get("weight", "normal")}" font-style="{e.get("style", "normal")}" '
                f'text-anchor="{e.get("anchor", "start")}" fill="{e.get("fill", "#222222")}"{rot}>{escape(e["s"])}{sup}</text>')
    raise ValueError(k)


def to_svg(layout: dict) -> str:
    icons = {}
    for e in layout["elements"]:
        if e["type"] == "image" and e["icon"] not in icons:
            icons[e["icon"]] = icon_data(e["icon"])[0]
    W, H = layout["width_pt"], layout["height_pt"]
    body = "\n".join(svg_el(e, icons) for e in layout["elements"])
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.2f}pt" height="{H:.2f}pt" viewBox="0 0 {W:.2f} {H:.2f}">\n'
            f'<rect x="0" y="0" width="{W:.2f}" height="{H:.2f}" fill="#FFFFFF"/>\n{body}\n</svg>\n')


# ---------------------------------------------------------------- checks
def check_layout(layout: dict) -> list[str]:
    """文字不得互相重叠、不得压在图标上、必须完整落在所属框（container）内。"""
    els = layout["elements"]
    boxes = {e["id"]: (e["x"], e["y"], e["x"] + e["w"], e["y"] + e["h"]) for e in els if e["type"] == "rect" and "id" in e}
    texts = [(e, text_bbox(e)) for e in els if e["type"] == "text"]
    imgs = [(e["icon"], (e["x"], e["y"], e["x"] + e["w"], e["y"] + e["h"])) for e in els if e["type"] == "image"]
    inter = lambda a, b, m=0.3: a[0] < b[2] - m and b[0] < a[2] - m and a[1] < b[3] - m and b[1] < a[3] - m  # noqa: E731
    bad = []
    for i, (t, b) in enumerate(texts):
        c = t.get("container")
        if c and not (b[0] >= boxes[c][0] + 1 and b[2] <= boxes[c][2] - 1 and b[1] >= boxes[c][1] + 1 and b[3] <= boxes[c][3] - 1):
            bad.append(f"text outside {c}: {t['s']!r} {tuple(round(v, 1) for v in b)} vs {boxes[c]}")
        for t2, b2 in texts[i + 1:]:
            if inter(b, b2):
                bad.append(f"text overlap: {t['s']!r} / {t2['s']!r}")
        for name, ib in imgs:
            if inter(b, ib):
                bad.append(f"text on icon {name}: {t['s']!r}")
    if any(b[2] > layout["width_pt"] or b[0] < 0 for _, b in texts):
        bad.append("text beyond figure width")
    return bad


def export(layout: dict) -> None:
    s = to_svg(layout)
    (FIG / "Fig1.svg").write_text(s)
    cairosvg.svg2pdf(bytestring=s.encode(), write_to=str(FIG / "Fig1.pdf"))
    px = round(layout["width_pt"] / 72 * 300)
    cairosvg.svg2png(bytestring=s.encode(), write_to=str(FIG / "Fig1.png"), output_width=px,
                     output_height=round(px * layout["height_pt"] / layout["width_pt"]))
    im = Image.open(FIG / "Fig1.png").convert("RGB")
    im.save(FIG / "Fig1.png", dpi=(300, 300))
    print(f"wrote Fig1: {layout['width_pt'] * 25.4 / 72:.0f} x {layout['height_pt'] * 25.4 / 72:.0f} mm (svg/pdf/png)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-layout", action="store_true", help="从已编辑的 Fig1_layout.json 渲染")
    a = ap.parse_args()
    if a.from_layout:
        layout = json.loads(LAYOUT.read_text())
    else:
        aspects = {n: icon_data(n, 64)[1] for n in ("kidney", "nephron", "singlecell", "slide", "model", "spatial")}
        layout = build_layout(load_glyph_data(), text_width)
        for e in layout["elements"]:
            if e["type"] == "image":
                fit_image(e, aspects[e["icon"]])
        LAYOUT.write_text(json.dumps(layout, indent=1, ensure_ascii=False))
    bad = check_layout(layout)
    for b in bad:
        print("LAYOUT:", b)
    export(layout)
    if bad:
        raise SystemExit(f"{len(bad)} layout problems")


if __name__ == "__main__":
    main()
