"""11a0_wsi_embed: tile KPMP whole-slide images (H&E, PAS) and embed with the frozen encoder.

Per slide (resumable; one npz per slide in data/processed/wsi_embeddings/):
  tissue mask -> tile grid at 0.5 um/px (224 px = 112 um) -> random (seeded) order -> tile QC
  (focus, ink) -> up to max_tiles tiles -> interpretable features (histo.he_features) and
  phikon-v2 CLS embeddings.
Slide summaries: mean embedding and mean features over (i) all QC tiles and (ii) the
pre-specified tubule-rich subset = tiles with white (lumen) fraction within tubule_lumen_range
and eosin fraction >= the slide median.
Only files whose size matches the repository manifest are processed (downloads are ongoing).
Usage: python 11a0_wsi_embed.py [--shard i/n] [--watch]
"""
from __future__ import annotations
import json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import openslide

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import ROOT, load_config, set_global_seed, write_provenance
from lib.histo import he_features, Encoder
from lib.wsi import tissue_mask, tile_grid, read_tile, tile_qc


def stage_dir(stage: str) -> Path:
    d = ROOT / "results" / stage
    d.mkdir(parents=True, exist_ok=True)
    return d

STAGE = "11a0_wsi_embed"
RAW = ROOT / "data/raw/kpmp/Light_Microscopic_Whole_Slide_Images"
EMB = ROOT / "data/processed/wsi_embeddings"
STAIN = {"H&E stain": "HE", "PAS stain": "PAS"}


def manifest() -> pd.DataFrame:
    m = pd.read_csv(ROOT / "data/raw/kpmp/download_manifest_wsi_priority.tsv", sep="\t")
    m["path"] = [RAW / r.redcap_id / r.file_name for r in m.itertuples()]
    m["complete"] = [p.exists() and p.stat().st_size == int(float(s)) for p, s in zip(m.path, m.file_size)]
    m["stain"] = m.workflow_type.map(STAIN)
    return m


def process_slide(row, enc: Encoder, cfg: dict, hcfg: dict, seed: int) -> dict:
    s = openslide.OpenSlide(str(row.path))
    mask, scale = tissue_mask(s, cfg)
    grid = tile_grid(s, mask, scale, cfg)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(grid))
    tiles, feats, coords = [], [], []
    with ThreadPoolExecutor(8) as ex:
        for start in range(0, len(order), 256):
            idx = order[start:start + 256]
            batch = list(ex.map(lambda i: read_tile(s, *grid[i], cfg["tile_px"]), idx))
            for i, t in zip(idx, batch):
                if tile_qc(t, cfg):
                    tiles.append(t); coords.append(grid[i][:2]); feats.append(he_features(t, hcfg))
            if len(tiles) >= cfg["max_tiles"]:
                break
    tiles, coords = tiles[:cfg["max_tiles"]], coords[:cfg["max_tiles"]]
    F = pd.DataFrame(feats[:len(tiles)])
    info = {"participant": row.redcap_id, "stain": row.stain, "file": row.file_name,
            "n_grid": len(grid), "n_tiles": len(tiles), "mpp": float(s.properties.get("openslide.mpp-x", "nan"))}
    if len(tiles) < cfg["min_tiles_slide"]:
        info["status"] = "too_few_tiles"; return info
    E = enc(np.stack(tiles))
    lo, hi = cfg["tubule_lumen_range"]
    tub = ((F.white_frac >= lo) & (F.white_frac <= hi) & (F.eosin_frac >= F.eosin_frac.median())).values
    np.savez_compressed(EMB / f"{row.redcap_id}__{row.stain}__{Path(row.file_name).stem}.npz",
                        emb=E.astype(np.float16), coords=np.asarray(coords), tubule=tub,
                        feats=F.values.astype(np.float32), feat_names=np.array(F.columns))
    info.update({"status": "ok", "n_tubule": int(tub.sum())})
    return info


def run_once(cfg, hcfg, seed, shard, enc) -> tuple[int, float]:
    m = manifest()
    todo = m[m.complete & m.stain.notna()].reset_index(drop=True)
    todo = todo[todo.index % shard[1] == shard[0]]
    log = stage_dir(STAGE) / f"slides_shard{shard[0]}.jsonl"
    done = {json.loads(l)["file"] for l in log.read_text().splitlines()} if log.exists() else set()
    n_new = 0
    for r in todo.itertuples():
        if r.file_name in done:
            continue
        try:
            info = process_slide(r, enc, cfg, hcfg, seed)
        except (openslide.OpenSlideError, OSError, ValueError) as e:
            info = {"participant": r.redcap_id, "stain": r.stain, "file": r.file_name, "status": f"error: {e}"[:200]}
        with open(log, "a") as fh:
            fh.write(json.dumps(info) + "\n")
        n_new += 1
        print(info, flush=True)
    return n_new, float(m.complete.mean())


def main():
    cfg_all = load_config(); cfg, hcfg = cfg_all["wsi_embed"]["tiles"], cfg_all["wsi_embed"]["features"]
    seed = cfg_all["seed"]; set_global_seed(seed)
    shard = (0, 1)
    if "--shard" in sys.argv:
        i, n = sys.argv[sys.argv.index("--shard") + 1].split("/"); shard = (int(i), int(n))
    EMB.mkdir(parents=True, exist_ok=True)
    enc = Encoder(hcfg["encoder"], hcfg["encoder_revision"])
    t0 = time.time()
    while True:
        n_new, frac = run_once(cfg, hcfg, seed, shard, enc)
        print(f"pass done: {n_new} new slides, manifest complete fraction {frac:.2f}", flush=True)
        if "--watch" not in sys.argv or frac >= 1.0 or (frac >= cfg["poll_target_frac"] and n_new == 0) \
                or time.time() - t0 > cfg["poll_max_hours"] * 3600:
            break
        time.sleep(cfg["poll_minutes"] * 60)
    if shard[0] == 0:
        logs = sorted(stage_dir(STAGE).glob("slides_shard*.jsonl"))
        write_provenance(STAGE, [ROOT / "data/raw/kpmp/download_manifest_wsi_priority.tsv"], logs, seed,
                         extra={"encoder": hcfg["encoder"], "revision": hcfg["encoder_revision"], "params": cfg})


if __name__ == "__main__":
    main()
