"""Stage 20a：读取全部已嵌入切片 .svs 文件头中的采集元数据（扫描日期、扫描仪编号、像素分辨率、物镜倍数、
曝光时间），并从文件名解析病理登记号的年月（S-YYMM-xxxxxx，即活检登记时间）。只读文件头，不读像素。

运行环境：openslide 只在 tal-ckd-slides 环境中可用（config slide_robustness.reader_env）：
    conda run -n tal-ckd-slides python scripts/stages/20a_slide_headers.py
输出：results/20a_slide_headers/slide_headers.tsv + PROVENANCE.json
"""
from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

STAGE = "20a_slide_headers"
KEYS = {"scan_date": "aperio.Date", "scan_time": "aperio.Time", "scanner_id": "aperio.ScanScope ID",
        "app_mag": "aperio.AppMag", "exposure_time": "aperio.Exposure Time", "mpp_header": "aperio.MPP",
        "parmset": "aperio.Parmset", "dsr_id": "aperio.DSR ID"}


def accession_yymm(fname: str) -> tuple[float, float]:
    m = re.search(r"_S-(\d{2})(\d{2})-\d+_", fname)
    return (2000 + int(m.group(1)), int(m.group(2))) if m else (float("nan"), float("nan"))


def main() -> None:
    import openslide
    cfg = load_config()
    set_global_seed(cfg["seed"])
    c = cfg["slide_robustness"]
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    f_inv = ROOT / "results/11a_port_wsi_embeddings/slide_inventory.tsv"
    inv = pd.read_csv(f_inv, sep="\t", dtype=str)
    rows = []
    for _, r in inv.iterrows():
        p = Path(c["slide_dir"]) / r.participant / r.file
        rec = {"participant": r.participant, "stain": r.stain, "file": r.file, "mpp_inventory": float(r.mpp)}
        rec["accession_year"], rec["accession_month"] = accession_yymm(r.file)
        try:
            props = openslide.OpenSlide(str(p)).properties
            rec.update({k: props.get(v) for k, v in KEYS.items()})
            rec["header_read"] = True
        except Exception as e:  # noqa: BLE001  文件缺失或不可读时如实记录
            rec["header_read"] = False
            rec["header_error"] = type(e).__name__
        if rec.get("scan_date"):
            fmt = "%m/%d/%Y" if len(rec["scan_date"].split("/")[-1]) == 4 else "%m/%d/%y"
            d = datetime.strptime(rec["scan_date"], fmt)
            rec["scan_year"], rec["scan_month"] = d.year, d.month
            rec["scan_date_iso"] = d.date().isoformat()
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(out / "slide_headers.tsv", sep="\t", index=False)
    print(df.header_read.value_counts(), df.scanner_id.value_counts(dropna=False), sep="\n")
    write_provenance(STAGE, inputs=[f_inv], outputs=[out / "slide_headers.tsv"], seed=cfg["seed"],
                     extra={"slide_dir": c["slide_dir"], "header_keys": KEYS})


if __name__ == "__main__":
    main()
