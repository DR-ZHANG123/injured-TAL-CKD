"""把 figures/ 下各图的 PDF 按矢量等比缩放到期刊尺寸上限，并由缩放后的 PDF 重新渲染 300 dpi PNG。

Human Genomics：整页宽图最宽 170 mm；图加图注最高 225 mm。主图高度上限取 175 mm（为图注留出空间），
补充图上限取 205 mm（补充材料 Word 中图注较短）。只缩小、不放大；缩放后最小字号写入 figures/size_report.tsv。
在全部出图脚本之后运行：python scripts/figures/finalize_sizes.py
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
from pypdf import PdfReader, PdfWriter, Transformation

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures"
MM = 72 / 25.4
LIMITS = {"main": (170.0, 175.0), "supp": (170.0, 205.0)}
MIN_FONT_PT = {"Fig1": 6.0, "default": 5.3}   # 各出图脚本中使用的最小字号（Fig1 由布局 JSON 给出）


def fit(pdf: Path, max_w: float, max_h: float) -> dict:
    reader = PdfReader(str(pdf))
    page = reader.pages[0]
    w, h = float(page.mediabox.width) / MM, float(page.mediabox.height) / MM
    s = min(1.0, max_w / w, max_h / h)
    if s < 1.0:
        page.add_transformation(Transformation().scale(s, s))
        page.mediabox.upper_right = (float(page.mediabox.width) * s, float(page.mediabox.height) * s)
        page.cropbox = page.mediabox
        writer = PdfWriter()
        writer.add_page(page)
        with open(pdf, "wb") as fh:
            writer.write(fh)
    png = pdf.with_suffix(".png")
    subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile", str(pdf), str(png.with_suffix(""))], check=True)
    base = MIN_FONT_PT.get(pdf.stem, MIN_FONT_PT["default"])
    return {"figure": pdf.stem, "orig_w_mm": round(w, 1), "orig_h_mm": round(h, 1), "scale": round(s, 3),
            "final_w_mm": round(w * s, 1), "final_h_mm": round(h * s, 1), "min_font_pt_after": round(base * s, 2)}


def main() -> None:
    rows = []
    for pdf in sorted(FIG.glob("Fig*.pdf")):
        kind = "supp" if pdf.stem.startswith("FigS") else "main"
        rows.append(fit(pdf, *LIMITS[kind]))
    rep = pd.DataFrame(rows)
    rep.to_csv(FIG / "size_report.tsv", sep="\t", index=False)
    print(rep.to_string(index=False))
    bad = rep[rep.min_font_pt_after < 5.0]
    if len(bad):
        raise SystemExit(f"字号低于 5 pt：{', '.join(bad.figure)}")


if __name__ == "__main__":
    main()
