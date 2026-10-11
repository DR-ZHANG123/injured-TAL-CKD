"""Stage 11a：移植 KPMP 全切片 Phikon-v2 tile 嵌入（每张切片一个 npz），逐文件核对 MD5。

来源与参数见 config/run.yaml `histology.port`。嵌入由 owkin/phikon-v2（固定 revision，冻结 fp16）在
0.5 µm/px、224 px（112 µm）tile 上计算；每张切片最多 2,000 个 QC tile。生成代码原样存档到
data/processed/histology/source_code/，不在本项目重跑 GPU 步骤。

核对：
  1. 源日志 slides_shard*.jsonl 与切片清单的 MD5 与源 PROVENANCE.json 中记录值一致；
  2. 每个 npz 复制前后 MD5 相同，并能被读取（emb 形状 n_tiles × 1024）；
  3. 日志中 status=ok 的切片与 npz 文件一一对应。
输出：
  data/processed/histology/wsi_embeddings/*.npz（+ 清单、病理描述评分、源代码存档）
  results/11a_port_wsi_embeddings/{ported_md5.tsv, slide_inventory.tsv, coverage.tsv, PROVENANCE.json}
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import ROOT, load_config, md5, set_global_seed, write_provenance  # noqa: E402

STAGE = "11a_port_wsi_embeddings"


def copy_verified(src: Path, dst: Path) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    m_src = md5(src)
    if not (dst.exists() and md5(dst) == m_src):
        shutil.copy2(src, dst)
    m_dst = md5(dst)
    if m_dst != m_src:
        raise RuntimeError(f"MD5 mismatch after copy: {src}")
    return {"source": str(src), "dest": str(dst.relative_to(ROOT)), "bytes": src.stat().st_size,
            "md5_source": m_src, "md5_dest": m_dst, "match": True}


def recorded_md5(prov_path: Path) -> dict[str, tuple[str, int]]:
    """源 PROVENANCE 中 inputs/outputs 的 {文件名: (md5, 字节数)}。"""
    rec = json.loads(prov_path.read_text())
    out = {}
    for key in ("inputs", "outputs"):
        for item in rec.get(key, []):
            out[Path(item["path"]).name] = (item["md5"], int(item["bytes"]))
    return out


def check_recorded(f: Path, md5_now: str, rec: tuple[str, int] | None) -> str:
    """与源记录比较。分片日志在源 PROVENANCE 写出后仍可能追加记录（另一分片晚结束），
    此时要求记录时的字节前缀 MD5 不变（只追加、未改写）。"""
    if rec is None:
        return "not_recorded"
    if md5_now == rec[0]:
        return "match"
    with open(f, "rb") as fh:
        prefix = hashlib.md5(fh.read(rec[1])).hexdigest()
    if prefix == rec[0]:
        return "recorded_prefix_match_appended_after"
    raise RuntimeError(f"source file changed since it was recorded: {f}")


def slide_log(log_dir: Path) -> pd.DataFrame:
    """合并分片日志；同一文件在重复轮次中可能记录多次，保留最后一条。"""
    rows = [json.loads(l) for f in sorted(log_dir.glob("slides_shard*.jsonl")) for l in f.read_text().splitlines()]
    return pd.DataFrame(rows).drop_duplicates("file", keep="last").reset_index(drop=True)


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg["seed"])
    p = cfg["histology"]["port"]
    dest = ROOT / p["dest"]
    out = ROOT / "results" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    rec = recorded_md5(Path(p["src_provenance"]))
    rows = []

    # 1. 日志与清单：与源 PROVENANCE 记录值核对
    for f in sorted(Path(p["src_logs"]).glob("slides_shard*.jsonl")) + [Path(x) for x in p["src_extra"]]:
        r = copy_verified(f, dest / f.name)
        r["kind"] = "log" if f.suffix == ".jsonl" else "metadata"
        r["md5_recorded"] = rec[f.name][0] if f.name in rec else ""
        r["recorded_check"] = check_recorded(f, r["md5_source"], rec.get(f.name))
        rows.append(r)
    for f in map(Path, p["src_code"]):
        r = copy_verified(f, dest / "source_code" / f.name)
        r.update(kind="code", md5_recorded="", recorded_check="not_recorded")
        rows.append(r)

    # 2. 嵌入
    log = slide_log(Path(p["src_logs"]))
    npz = sorted(Path(p["src_embeddings"]).glob("*.npz"))
    inv = []
    for f in npz:
        r = copy_verified(f, dest / "wsi_embeddings" / f.name)
        r.update(kind="embedding", md5_recorded="", recorded_check="not_recorded")
        rows.append(r)
        z = np.load(dest / "wsi_embeddings" / f.name, allow_pickle=True)
        pid, stain, stem = f.stem.split("__", 2)
        inv.append({"participant": pid, "stain": stain, "file": stem + ".svs", "npz": f.name,
                    "n_tiles": int(z["emb"].shape[0]), "emb_dim": int(z["emb"].shape[1]),
                    "n_tubule_tiles": int(z["tubule"].sum())})
    md5_tab = pd.DataFrame(rows)[["kind", "source", "dest", "bytes", "md5_source", "md5_recorded", "recorded_check",
                                "md5_dest", "match"]]
    md5_tab.to_csv(out / "ported_md5.tsv", sep="\t", index=False)

    # 3. 日志 ok 切片 ↔ npz 一一对应；附切片清单中的参与者临床分档
    inv = pd.DataFrame(inv)
    ok = log[log.status == "ok"]
    missing = set(ok.file) - set(inv.file)
    extra = set(inv.file) - set(ok.file)
    if missing or extra:
        raise RuntimeError(f"log/npz mismatch: missing {len(missing)}, extra {len(extra)}")
    inv = inv.merge(ok[["file", "n_grid", "mpp"]], on="file", how="left")
    man = pd.read_csv(dest / "download_manifest_wsi_priority.tsv", sep="\t", dtype=str)
    man = man.drop_duplicates("file_name").set_index("file_name")
    for c in ["primary_adjudicated_category", "enrollment_category", "baseline_egfr", "albuminuria"]:
        inv[c] = inv.file.map(man[c])
    inv.to_csv(out / "slide_inventory.tsv", sep="\t", index=False)

    failed = log[log.status != "ok"]
    part = inv.groupby("participant").agg(stains=("stain", lambda s: "+".join(sorted(s))),
                                          category=("primary_adjudicated_category", "first"))
    cov = [{"item": "slides_logged", "value": len(log)}, {"item": "slides_embedded", "value": len(inv)},
           {"item": "slides_failed", "value": len(failed)},
           {"item": "failed_files", "value": ";".join(failed.file)},
           {"item": "tiles_total", "value": int(inv.n_tiles.sum())},
           {"item": "participants", "value": part.shape[0]}]
    cov += [{"item": f"slides_{s}", "value": int((inv.stain == s).sum())} for s in sorted(inv.stain.unique())]
    cov += [{"item": f"participants_stains_{k}", "value": int(v)} for k, v in part.stains.value_counts().items()]
    cov += [{"item": f"participants_{k}", "value": int(v)}
            for k, v in part.category.fillna("not adjudicated").value_counts().items()]
    pd.DataFrame(cov).to_csv(out / "coverage.tsv", sep="\t", index=False)
    print(pd.DataFrame(cov).to_string(index=False))

    write_provenance(
        STAGE, inputs=[], outputs=[out / "ported_md5.tsv", out / "slide_inventory.tsv", out / "coverage.tsv"],
        seed=cfg["seed"],
        extra={"ported_from": {"embeddings": p["src_embeddings"], "logs": p["src_logs"],
                               "provenance": p["src_provenance"], "extra": p["src_extra"], "code": p["src_code"]},
               "ported_md5_table": "results/11a_port_wsi_embeddings/ported_md5.tsv",
               "n_files_verified": int(md5_tab.match.sum()),
               "encoder": p["encoder"], "encoder_revision": p["encoder_revision"],
               "source_params": json.loads(Path(p["src_provenance"]).read_text()).get("extra", {}).get("params")})


if __name__ == "__main__":
    main()
