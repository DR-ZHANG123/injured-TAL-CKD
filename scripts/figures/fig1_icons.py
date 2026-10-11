"""Fig. 1 插图图标：调用 OpenAI 兼容图像接口生成扁平科学插图图标（无文字），存为 figures/source_data/Fig1_icon_*.png。

图标是 Fig1_framework.py 的输入；已存在的图标不重新生成（--force 覆盖）。
API key 只从环境变量 GPTIMAGE_API_KEY 读取，不写入任何文件。
用法：GPTIMAGE_API_KEY=... python scripts/figures/fig1_icons.py [--only kidney,tal] [--force]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures" / "source_data"
BASE_URL = os.environ.get("GPTIMAGE_BASE_URL", "https://cpa.onelyi.com/v1")
MODEL = os.environ.get("GPTIMAGE_MODEL", "gpt-image-2")

STYLE = ("Flat vector-style scientific illustration icon in the style of BioRender, for a Nature journal figure. "
         "Clean shapes, soft muted colours (dusty rose, slate blue, warm grey, light pink tissue tones), thin dark-grey "
         "outlines, subtle shading, centred, generous margin, plain pure white background. "
         "Absolutely no text, no letters, no numbers, no labels, no arrows, no watermark.")

ICONS = {
    "kidney": "A human kidney in cross-section showing cortex and medulla, with a thin cylindrical needle biopsy core "
              "lying next to it.",
    "nephron": "A single kidney nephron (glomerulus, proximal tubule, loop of Henle, distal tubule, collecting duct) "
               "drawn in neutral grey, with only the thick ascending limb of the loop of Henle highlighted in deep red.",
    "singlecell": "Single-cell / single-nucleus droplet sequencing: a microfluidic chip channel with small oil droplets, "
                  "each droplet containing one cell nucleus and a bead.",
    "slide": "A glass microscope slide with a PAS-stained kidney biopsy tissue section (magenta-pink tissue), overlaid with "
             "a fine square grid of tiles on the tissue.",
    "model": "A foundation model glyph: a stylised deep neural network of interconnected nodes in layers, receiving small "
             "square histology image tiles on the left and producing a short vector of coloured cells on the right.",
    "spatial": "A thin kidney tissue section seen from above, covered with many small coloured dots marking cell "
               "positions as in spatial transcriptomics.",
}


def generate(name: str, force: bool) -> str:
    dst = OUT / f"Fig1_icon_{name}.png"
    if dst.exists() and not force:
        return f"{name}: exists"
    key = os.environ["GPTIMAGE_API_KEY"]
    body = json.dumps({"model": MODEL, "prompt": f"{ICONS[name]} {STYLE}", "size": "1024x1024", "n": 1}).encode()
    req = urllib.request.Request(f"{BASE_URL}/images/generations", data=body,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())["data"][0]
    if d.get("b64_json"):
        dst.write_bytes(base64.b64decode(d["b64_json"]))
    else:
        with urllib.request.urlopen(d["url"], timeout=300) as r:
            dst.write_bytes(r.read())
    return f"{name}: wrote {dst.name}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    names = [n for n in a.only.split(",") if n] or list(ICONS)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "Fig1_icon_prompts.json").write_text(json.dumps({"model": MODEL, "style": STYLE, "icons": ICONS}, indent=1))
    with ThreadPoolExecutor(len(names)) as ex:
        for msg in ex.map(lambda n: generate(n, a.force), names):
            print(msg)


if __name__ == "__main__":
    sys.exit(main())
