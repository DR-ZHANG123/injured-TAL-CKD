"""Fig. 1 版面：生成全部元素（框、文字、箭头、图标位置、迷你示意图）。坐标单位 pt，宽 481.9 pt = 170 mm。
只排版，不做分析；数值来自 Fig1_framework.load_glyph_data()（读取其他主图的绘图输入表）。"""
from __future__ import annotations

import numpy as np
from matplotlib import colormaps


DX = {"DKD": "#C44E52", "HKD": "#4C72B0", "REF": "#8C8C8C", "other": "#8C8C8C"}
INK, MUTED, FRAME = "#222222", "#555555", "#B8B8B8"
HEAD_BG = "#F3F4F6"
TAL_HEALTHY, TAL_ADAPT, TAL_FR = "#D9D9D9", "#E8A3A5", "#C44E52"
F_LET, F_HEAD, F_SUB, F_BODY, F_SMALL = 11, 7.3, 6.5, 6.3, 6.0


class L:
    def __init__(self, tw):
        self.el, self.tw = [], tw

    def rect(self, x, y, w, h, **k):
        self.el.append(dict(type="rect", x=x, y=y, w=w, h=h, **k))

    def text(self, s, x, y, size=F_BODY, **k):
        self.el.append(dict(type="text", s=s, x=x, y=y, size=size, **k))

    def wrap(self, s, x, y, width, size=F_BODY, lh=1.22, **k):
        """按宽度折行，返回最后一行基线之后的 y。"""
        lines, cur = [], ""
        for w in s.split(" "):
            t = (cur + " " + w).strip()
            if cur and self.tw(t, size, k.get("weight", "normal"), k.get("style", "normal")) > width:
                lines.append(cur)
                cur = w
            else:
                cur = t
        lines.append(cur)
        for i, ln in enumerate(lines):
            self.text(ln, x, y + i * size * lh, size, **k)
        return y + len(lines) * size * lh

    def line(self, pts, **k):
        self.el.append(dict(type="line", pts=[list(p) for p in pts], **k))

    def arrow(self, x0, y0, x1, y1, **k):
        self.line([(x0, y0), (x1, y1)], head=True, stroke=k.pop("stroke", "#6B6B6B"), sw=k.pop("sw", 1.0), **k)

    def circle(self, x, y, r, **k):
        self.el.append(dict(type="circle", x=x, y=y, r=r, **k))

    def image(self, icon, x, y, w, h):
        self.el.append(dict(type="image", icon=icon, box=[x, y, w, h]))

    def panel(self, pid, letter, title, x, y, w, h):
        self.rect(x, y, w, h, rx=4, fill="#FFFFFF", stroke=FRAME, sw=0.6, id=pid)
        self.rect(x + 0.3, y + 0.3, w - 0.6, 15.5, rx=3.7, fill=HEAD_BG, stroke="none", id=pid + "_head")
        self.line([(x + 0.3, y + 15.8), (x + w - 0.3, y + 15.8)], stroke="#DDDDDD", sw=0.4)
        self.text(letter, x - 10, y + 11.3, F_LET, weight="bold", fill="#000000")
        self.text(title, x + 6, y + 10.8, F_HEAD, weight="bold", fill=INK, container=pid + "_head")

    def card(self, cid, x, y, w, h, dashed=False):
        self.rect(x, y, w, h, rx=3, fill="#FFFFFF" if not dashed else "#FAFAFA", stroke="#CFCFCF", sw=0.5,
                  dash="2,1.5" if dashed else None, id=cid)

    def scatter(self, x, y, w, h, xs, ys, cols, xlab, ylab, cid, r=1.0):
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        xr, yr = (xs.min(), xs.max()), (ys.min(), ys.max())
        px = lambda v: x + 3 + (v - xr[0]) / (xr[1] - xr[0]) * (w - 6)  # noqa: E731
        py = lambda v: y + h - 3 - (v - yr[0]) / (yr[1] - yr[0]) * (h - 6)  # noqa: E731
        self.line([(x, y), (x, y + h), (x + w, y + h)], stroke="#444444", sw=0.5)
        for a, b, c in zip(xs, ys, cols):
            self.circle(px(a), py(b), r, fill=DX.get(c, "#8C8C8C"), stroke="#FFFFFF", sw=0.2, alpha=0.85)
        k, b0 = np.polyfit(xs, ys, 1)
        self.line([(px(xr[0]), py(k * xr[0] + b0)), (px(xr[1]), py(k * xr[1] + b0))], stroke="#222222", sw=0.7)
        self.text(xlab, x + w / 2, y + h + 7.5, F_SMALL, anchor="middle", fill=MUTED, container=cid)
        self.text(ylab, x - 3, y + h / 2, F_SMALL, anchor="middle", fill=MUTED, rotate=-90, container=cid)


def cell(g: L, x: float, y: float, col: str) -> None:
    g.el.append(dict(type="path", d=f"M{x - 11:.2f},{y - 7:.2f} h22 a3,3 0 0 1 3,3 v8 a3,3 0 0 1 -3,3 h-22 "
                                     f"a3,3 0 0 1 -3,-3 v-8 a3,3 0 0 1 3,-3 z", fill=col, stroke="#7A7A7A", sw=0.5))
    g.circle(x, y, 3.2, fill="#FFFFFF", stroke="#7A7A7A", sw=0.4, alpha=0.9)


def clinical_glyph(g: L, x: float, y: float) -> None:
    """eGFR 示意：坐标轴 + 下降折线。"""
    g.line([(x - 18, y - 14), (x - 18, y + 13), (x + 20, y + 13)], stroke="#444444", sw=0.6)
    pts = [(x - 14, y - 9), (x - 5, y - 6), (x + 3, y - 1), (x + 9, y + 3), (x + 17, y + 8)]
    g.line(pts, stroke=TAL_FR, sw=1.0)
    for px, py in pts:
        g.circle(px, py, 1.3, fill=TAL_FR)


def manhattan_glyph(g: L, x: float, y: float) -> None:
    """GWAS 示意：交替两色的 Manhattan 点列 + 显著性虚线（形状为固定示意，不代表数据）。"""
    g.line([(x - 24, y - 14), (x - 24, y + 13), (x + 24, y + 13)], stroke="#444444", sw=0.6)
    g.line([(x - 22, y - 6), (x + 23, y - 6)], stroke="#C44E52", sw=0.5, dash="1.5,1.2")
    rng = np.random.default_rng(7)
    for k in range(6):
        cx0 = x - 20 + k * 7.4
        col = "#7F7F7F" if k % 2 == 0 else "#B5B5B5"
        hs = rng.uniform(0, 9, 7)
        if k == 2:
            hs = np.append(hs, [14, 17, 21])
        for j, hh in enumerate(hs):
            g.circle(cx0 + (j % 4) * 1.4, y + 11 - hh, 0.75, fill=col)

W = 170 / 25.4 * 72          # 481.89 pt
HEAT_VMAX = 4.0              # 与 Fig3b 一致（RdBu_r，±4）


def build_layout(d: dict, tw) -> dict:
    g = L(tw)
    X0, X1 = 14.0, W - 3.0
    _panel_a(g, d, X0, X1, 4.0, 120.0)
    yb, hb, wb = 134.0, 158.0, 200.0
    _panel_b(g, d, X0, yb, wb, hb)
    _panel_c(g, d, X0 + wb + 12, yb, X1 - X0 - wb - 12, hb)
    yd, hd = yb + hb + 10, 122.0
    _panel_d(g, d, X0, X1, yd, hd)
    H = yd + hd + 4
    return {"width_pt": W, "height_pt": H, "units": "pt (1 pt = 0.3528 mm)", "font": "Liberation Sans",
            "palette": {"DKD": DX["DKD"], "HKD": DX["HKD"], "Reference": DX["REF"]},
            "elements": g.el}


# ------------------------------------------------ a: cohorts and data layers
def _panel_a(g: L, d: dict, X0: float, X1: float, ya: float, ha: float) -> None:
    g.panel("a", "a", "Kidney biopsy cohorts and data layers", X0, ya, X1 - X0, ha)
    g.image("kidney", X0 + 6, ya + 22, 56, 54)
    cx = X0 + 34
    g.text("KPMP participants", cx, ya + 87, F_SUB, weight="bold", anchor="middle", container="a")
    for i, (lab, key) in enumerate([("Reference", "REF"), ("DKD", "DKD"), ("HKD", "HKD")]):
        yy = ya + 97 + i * 8.4
        g.circle(cx - 17, yy - 2.1, 2.2, fill=DX[key])
        g.text(lab, cx - 12, yy, F_BODY, container="a")
    g.arrow(X0 + 68, ya + 50, X0 + 78, ya + 50)

    cards = [("a1", "Single-cell transcriptomes", 117, "singlecell", False),
             ("a2", "Whole-slide images", 64, "slide", False),
             ("a3", "Clinical measures", 60, None, False),
             ("a4", "Genetic association", 72, None, False),
             ("a5", "In situ", 54, "spatial", True)]
    gap, cxs = 3.0, X0 + 81
    yc, hc = ya + 21, ha - 25
    for cid, title, w, icon, dashed in cards:
        g.card(cid, cxs, yc, w, hc, dashed)
        g.text(title, cxs + w / 2, yc + 10, F_SUB, weight="bold", anchor="middle", container=cid,
               fill=MUTED if dashed else INK)
        if icon:
            g.image(icon, cxs + 5, yc + 15, w - 10, 32)
        tx, tw_, ty, lh = cxs + 5, w - 10, yc + 58, 8.2
        if cid == "a1":
            for ds, role in [("KPMP snRNA-seq", "discovery"), ("KPMP scRNA-seq", "replication")]:
                g.text(ds, tx, ty, F_BODY, weight="bold", container=cid)
                g.text(role, tx + 60, ty, F_BODY, fill=MUTED, container=cid)
                ty += lh
            g.text("Independent cohorts", tx, ty, F_BODY, weight="bold", container=cid)
            g.wrap("GSE211785 (sn + sc), GSE195460, GSE271188", tx, ty + lh, tw_, F_BODY, lh=1.3, fill=MUTED, container=cid)
        elif cid == "a2":
            g.text("KPMP biopsies", tx, ty, F_BODY, weight="bold", container=cid)
            g.text("PAS and H&E", tx, ty + lh, F_BODY, fill=MUTED, container=cid)
        elif cid == "a3":
            clinical_glyph(g, cxs + w / 2, yc + 31)
            for s in ["eGFR", "Albuminuria"]:
                g.text(s, tx, ty, F_BODY, container=cid)
                ty += lh
            g.wrap("Pathologist scores", tx, ty, tw_, F_BODY, lh=1.3, container=cid)
        elif cid == "a4":
            manhattan_glyph(g, cxs + w / 2, yc + 31)
            g.text("CKDGen GWAS", tx, ty, F_BODY, weight="bold", container=cid)
            g.text("eGFR, BUN,", tx, ty + lh, F_BODY, fill=MUTED, container=cid)
            g.text("UACR, CKD", tx, ty + 2 * lh, F_BODY, fill=MUTED, container=cid)
            g.text("Kidney eQTL", tx, ty + 3 * lh, F_BODY, weight="bold", container=cid)
        else:
            g.text("Xenium", tx, ty, F_BODY, weight="bold", container=cid)
            g.wrap("Injured TAL neighbours", tx, ty + lh, tw_, F_BODY, lh=1.3, fill=MUTED, container=cid)
        cxs += w + gap
    assert cxs - gap <= X1 - 3, cxs


# ------------------------------------------------ b: injured TAL state
def _panel_b(g: L, d: dict, X0: float, yb: float, wb: float, hb: float) -> None:
    g.panel("b", "b", "Injured TAL cells expand in DKD and HKD", X0, yb, wb, hb)
    g.image("nephron", X0 + 4, yb + 22, 52, 84)
    g.text("Thick ascending limb", X0 + 31, yb + 118, F_BODY, anchor="middle", style="italic", container="b")
    g.text("(TAL)", X0 + 31, yb + 126, F_BODY, anchor="middle", fill=MUTED, container="b")
    sx, sy = [X0 + 84, X0 + 126, X0 + 168], yb + 46
    for x, c in zip(sx, [TAL_HEALTHY, TAL_ADAPT, TAL_FR]):
        cell(g, x, sy, c)
    for a, b in zip(sx[:-1], sx[1:]):
        g.arrow(a + 15, sy, b - 15, sy, sw=0.8)
    for x, s in zip(sx, ["Healthy", "Adaptive", "Failed-repair"]):
        g.text(s, x, sy + 19, F_BODY, anchor="middle", container="b")
    g.line([(sx[1] - 14, yb + 34), (sx[1] - 14, yb + 31), (sx[2] + 14, yb + 31), (sx[2] + 14, yb + 34)],
           stroke=TAL_FR, sw=0.7)
    g.text("Injured TAL", (sx[1] + sx[2]) / 2, yb + 27.5, F_SUB, weight="bold", anchor="middle", fill=TAL_FR,
           container="b")
    # injured fraction of TAL: KPMP snRNA-seq medians (reference = percutaneous biopsies)
    bx0, base, hmax = X0 + 68, yb + 146, 36.0
    vmax = max(d["tal_median"].values())
    g.text("Injured fraction of TAL", bx0 + 29, yb + 87, F_BODY, anchor="middle", weight="bold", container="b")
    for i, (key, lab) in enumerate([("REF", "Ref"), ("DKD", "DKD"), ("HKD", "HKD")]):
        h = hmax * d["tal_median"][key] / vmax
        x = bx0 + 4 + i * 19
        g.rect(x, base - h, 13, h, fill=DX[key], stroke="none")
        g.text(f"{round(100 * d['tal_median'][key])}%", x + 6.5, base - h - 2.2, F_SMALL, anchor="middle",
               fill=INK, container="b")
        g.text(lab, x + 6.5, base + 7.5, F_SMALL, anchor="middle", container="b")
    g.line([(bx0, base), (bx0 + 59, base)], stroke="#444444", sw=0.5)
    # replication and agreement between sequencing methods
    rx, rw = X0 + 140, 54
    g.text("Replicated in", rx + rw / 2, yb + 87, F_BODY, anchor="middle", weight="bold", container="b")
    for i, s in enumerate(["GSE211785", "GSE195460", "GSE271188"]):
        y = yb + 92 + i * 12
        g.rect(rx, y, rw, 10, rx=5, fill="#FBEFEF", stroke="#E3B5B6", sw=0.5, id=f"chip{i}")
        g.el.append(dict(type="path", d=f"M{rx + 5:.2f},{y + 5.2:.2f} l1.8,2 l3.2,-4.2", stroke=TAL_FR, sw=0.8))
        g.text(s, rx + 13, y + 7.2, F_SMALL, container=f"chip{i}")
    g.text("Most consistent", rx + rw / 2, yb + 136, F_SMALL, anchor="middle", container="b")
    g.text("across methods", rx + rw / 2, yb + 143.5, F_SMALL, anchor="middle", container="b")
    g.text("ICC 0.85", rx + rw / 2, yb + 151.5, F_BODY, anchor="middle", weight="bold", fill=TAL_FR,
           container="b")


# ------------------------------------------------ c: function loss and genetics
def _panel_c(g: L, d: dict, xc: float, yb: float, wc: float, hb: float) -> None:
    g.panel("c", "c", "Injury erodes TAL identity enriched for kidney-function genetics", xc, yb, wc, hb)
    genes = list(d["heat"])
    hx, hy, cw, rh = xc + 38, yb + 72, 13.0, 8.2
    g.text("Injured vs healthy TAL", hx + 1.5 * cw + 8, yb + 25.5, F_BODY, weight="bold", anchor="middle",
           container="c")
    for j, s in enumerate(["Discovery", "KPMP scRNA", "GSE271188"]):
        g.text(s, hx + j * cw + cw / 2 + 2.0, hy - 3, F_SMALL, rotate=-90, container="c", fill=MUTED)
    for i, gname in enumerate(genes):
        g.text(gname, hx - 3, hy + i * rh + 6.2, F_SMALL, anchor="end", style="italic", container="c")
        for j, v in enumerate(d["heat"][gname]):
            g.rect(hx + j * cw, hy + i * rh, cw - 0.8, rh - 0.8, fill=_rdbu(v), stroke="none")
    # group brackets: TAL function genes (down) and injury markers (up)
    bxr = hx + 3 * cw + 2
    for k, (l1, l2, col) in enumerate([("TAL", "function", DX["HKD"]), ("Injury", "markers", TAL_FR)]):
        y0, y1 = hy + k * 4 * rh + 0.5, hy + (k + 1) * 4 * rh - 1.3
        g.line([(bxr, y0), (bxr + 2, y0), (bxr + 2, y1), (bxr, y1)], stroke=col, sw=0.6)
        ym = (y0 + y1) / 2
        g.text(l1, bxr + 5, ym - 0.8, F_SMALL, fill=col, weight="bold", container="c")
        g.text(l2, bxr + 5, ym + 6.2, F_SMALL, fill=col, weight="bold", container="c")
    yk, kw = hy + len(genes) * rh + 4, 3 * cw - 0.8
    for k in range(24):
        g.rect(hx + k * kw / 24, yk, kw / 24 + 0.05, 4, fill=_rdbu(-HEAT_VMAX + k * 2 * HEAT_VMAX / 23), stroke="none")
    for v, xx in [("−4", hx), ("0", hx + kw / 2), ("4", hx + kw)]:
        g.text(v, xx, yk + 11, F_SMALL, anchor="middle", fill=MUTED, container="c")
    g.text("log2FC", hx - 3, yk + 4.2, F_SMALL, anchor="end", fill=MUTED, container="c")
    # genetic signal for eGFR (CKDGen eGFR; Z of mean gene association against 1000 matched random gene sets)
    xg = xc + 132
    g.line([(xg - 10, yb + 24), (xg - 10, yb + hb - 8)], stroke="#E2E2E2", sw=0.5)
    g.text("Genetic signal for eGFR", xg + 54, yb + 26.5, F_BODY, weight="bold", anchor="middle", container="c")
    m, lmax = d["magma"], 100.0
    full = max(m.values())
    rows = [("Healthy TAL genes", "healthy", "#7F7F7F", yb + 44), ("Gained in injured TAL", "injury", TAL_ADAPT, yb + 84)]
    for lab, key, col, y in rows:
        g.text(lab, xg + 2, y, F_BODY, container="c")
        g.rect(xg, y + 3, max(lmax * m[key] / full, 1.5), 10, fill=col, stroke="none")
    g.text("P ≤ 0.001", xg + 2, yb + 69, F_BODY, weight="bold", fill=INK, container="c")
    g.text("P = 0.50", xg + 5, yb + 94.5, F_BODY, fill=MUTED, container="c")
    g.line([(xg, yb + 36), (xg, yb + 102)], stroke="#444444", sw=0.5)
    g.text("Z vs matched random genes", xg + lmax / 2, yb + 109, F_SMALL, anchor="middle", fill=MUTED, container="c")
    g.text("Kidney-function genetics concentrates", xg + 54, yb + 133, F_BODY, anchor="middle", container="c")
    g.text("in differentiated TAL identity", xg + 54, yb + 141.5, F_BODY, anchor="middle", container="c")


def _rdbu(v: float) -> str:
    r, gg, b, _ = colormaps["RdBu_r"](float(np.clip((v + HEAT_VMAX) / (2 * HEAT_VMAX), 0, 1)))
    return "#%02X%02X%02X" % (round(r * 255), round(gg * 255), round(b * 255))


# ------------------------------------------------ d: slide estimate and clinical association
def _panel_d(g: L, d: dict, X0: float, X1: float, yd: float, hd: float) -> None:
    g.panel("d", "d", "Routine biopsy slides estimate injured TAL, whose burden tracks kidney function",
            X0, yd, X1 - X0, hd)
    yi = yd + 24
    mid, ytx = yi + 24, yi + 72
    c = [X0 + 32, X0 + 104, X0 + 174, X0 + 246, X0 + 318, X0 + 410]

    def caption(cx, lines):
        y = ytx
        for s, kw in lines:
            g.text(s, cx, y, F_BODY, anchor="middle", container="d", **kw)
            y += 8.2

    g.image("slide", c[0] - 25, yi + 2, 50, 44)
    caption(c[0], [("PAS / H&E", dict(weight="bold")), ("whole-slide image", {})])
    g.image("model", c[1] - 33, yi + 2, 66, 44)
    caption(c[1], [("Phikon-v2", dict(weight="bold")), ("tile embeddings,", dict(fill=MUTED)),
                   ("mean per participant", dict(fill=MUTED))])
    g.rect(c[2] - 24, mid - 14, 48, 28, rx=4, fill="#EEF2F8", stroke="#9DB0CF", sw=0.6, id="ridge")
    g.text("Ridge", c[2], mid - 2, F_BODY, weight="bold", anchor="middle", container="ridge")
    g.text("regression", c[2], mid + 6, F_BODY, weight="bold", anchor="middle", container="ridge")
    caption(c[2], [("Participant-level", dict(fill=MUTED)), ("cross-validation", dict(fill=MUTED))])
    sw_, sh_ = 50, 46
    g.scatter(c[3] - sw_ / 2 + 4, yi, sw_, sh_, d["cv"]["x"], d["cv"]["y"], d["cv"]["dx"], "Measured",
              "Estimate", "d")
    caption(c[3], [("Estimate vs measured", dict(weight="bold")), ("r = 0.73", dict(weight="bold", fill=TAL_FR))])
    # correlation of slide estimate and of pathologist scores with the measured fraction
    bb, bh = yi + sh_, 40.0
    for i, (lab, r, col) in enumerate([("Slide", 0.73, TAL_FR), ("Pathologist", 0.38, "#A6A6A6")]):
        x = c[4] - 19 + i * 26
        g.rect(x, bb - bh * r, 12, bh * r, fill=col, stroke="none")
        g.text(f"{r:.2f}", x + 6, bb - bh * r - 2.2, F_SMALL, anchor="middle", container="d")
        g.text(lab, x + 6, bb + 7.5, F_SMALL, anchor="middle", container="d")
    g.line([(c[4] - 23, bb), (c[4] + 23, bb)], stroke="#444444", sw=0.5)
    g.text("r", c[4] - 26, bb - 14, F_SMALL, anchor="middle", fill=MUTED, style="italic", container="d")
    caption(c[4], [("Beyond pathologist", dict(weight="bold")), ("scores", dict(weight="bold"))])
    for a0, a1 in [(c[0] + 27, c[1] - 35), (c[1] + 35, c[2] - 26), (c[2] + 26, c[3] - 30)]:
        g.arrow(a0, mid, a1, mid)
    xdiv = c[5] - 58
    g.line([(xdiv, yi - 2), (xdiv, yd + hd - 6)], stroke="#E2E2E2", sw=0.5)
    pw, ph = 40, 46
    g.scatter(c[5] - 45, yi, pw, ph, d["egfr"]["x"], d["egfr"]["y"], d["egfr"]["dx"], "Injured TAL", "eGFR", "d",
              r=0.9)
    _albuminuria(g, d["alb"], c[5] + 14, yi, 34, ph)
    caption(c[5], [("Higher injured TAL burden:", dict(weight="bold")),
                   ("lower eGFR,", {}), ("higher albuminuria", {})])


def _albuminuria(g: L, a: dict, x: float, y: float, w: float, h: float) -> None:
    xs, ys, cat = np.asarray(a["x"]), np.asarray(a["y"]), np.asarray(a["cat"])
    yr = (ys.min(), ys.max())
    px = lambda v: x + (v - 0.5) / 3 * w  # noqa: E731
    py = lambda v: y + h - 3 - (v - yr[0]) / (yr[1] - yr[0]) * (h - 6)  # noqa: E731
    g.line([(x, y), (x, y + h), (x + w, y + h)], stroke="#444444", sw=0.5)
    for xv, yv, col in zip(xs, ys, a["dx"]):
        g.circle(px(xv), py(yv), 0.9, fill=DX.get(col, "#8C8C8C"), stroke="#FFFFFF", sw=0.2, alpha=0.85)
    for k in (1, 2, 3):
        md = float(np.median(ys[cat == k]))
        g.line([(px(k) - 5, py(md)), (px(k) + 5, py(md))], stroke="#222222", sw=0.9)
        g.text(f"A{k}", px(k), y + h + 7.5, F_SMALL, anchor="middle", fill=MUTED, container="d")
    g.text("Injured TAL", x - 3, y + h / 2, F_SMALL, anchor="middle", fill=MUTED, rotate=-90, container="d")
    g.text("Albuminuria", x + w / 2, y + h + 15, F_SMALL, anchor="middle", fill=MUTED, container="d")
