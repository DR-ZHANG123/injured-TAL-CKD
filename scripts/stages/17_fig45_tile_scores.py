"""Stage 17：Fig 4d 的切片 tile 级损伤 TAL 读出。

对训练参与者中交叉验证预测损伤 TAL 最高与最低的各一名，用其余训练参与者拟合与 stage 11 相同的岭回归
（参与者平均嵌入、训练集内标准化、内层 CV 选正则化），即被展示者不参与拟合。模型为线性，故把每个 tile 的
嵌入代入同一模型即得 tile 得分，其平均等于该切片的预测值。输出 tile 坐标与得分、组织缩略图、得分最高 / 最低的
原始 tile 图像（从全切片读取）。不输出参与者编号以外的任何个人信息；图中只用 A/B 标注。

运行环境：读取 .svs 需要 openslide（config fig45.reader_env）。
输出：results/17_fig45_tiles/*.tsv、thumb_*.png、tiles/*.png + PROVENANCE.json
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.repro import ROOT, load_config, set_global_seed, write_provenance  # noqa: E402

_spec = importlib.util.spec_from_file_location("hl", Path(__file__).with_name("11_histology_lib.py"))
hl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hl)

STAGE = "17_fig45_tiles"


def ridge_coef(Xtr: np.ndarray, ytr: np.ndarray, alpha: float):
    """与 hl._ridge_all_alpha 相同的核形式岭回归，返回标准化参数与系数：pred = ((x-mu)/sd)·beta + ym。"""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1
    A = (Xtr - mu) / sd
    ym = ytr.mean()
    w, V = np.linalg.eigh(A @ A.T)
    beta = A.T @ (V @ ((V.T @ (ytr - ym)) / (w + alpha)))
    return mu, sd, beta, ym


def slide_file(emb_dir: Path, pid: str, stain: str) -> Path:
    f = sorted(emb_dir.glob(f"{pid}__{stain}__*.npz"))
    if len(f) != 1:
        raise RuntimeError(f"{pid} {stain}: {len(f)} embedding files")
    return f[0]


def main() -> None:
    cfg_all = load_config()
    seed = cfg_all["seed"]
    set_global_seed(seed)
    h, c = cfg_all["histology"], cfg_all["fig45"]
    out = ROOT / "results" / STAGE
    (out / "tiles").mkdir(parents=True, exist_ok=True)
    emb_dir = ROOT / h["port"]["dest"] / "wsi_embeddings"
    r11 = ROOT / "results" / "11_histology"
    f_ax, f_map, f_cv = (r11 / "molecular_axes_snRNA_donors.tsv", r11 / "participant_map.tsv",
                         r11 / "cv_predictions_by_participant.tsv")
    axes = pd.read_csv(f_ax, sep="\t", index_col=0, dtype={0: str})
    mp = pd.read_csv(f_map, sep="\t", index_col=0, dtype={0: str})
    cvp = pd.read_csv(f_cv, sep="\t", dtype={"participant": str}).drop_duplicates()

    emb = hl.slide_embeddings(emb_dir, h["tileset"], h["min_tiles"])[c["stain"]]
    train = [p for p in mp.index[mp.set == "training_atlas_and_slides"] if p in emb.index]
    y = axes.loc[train, c["target"]].dropna()
    cv = cvp[(cvp.target == c["target"]) & (cvp.tileset == h["tileset"]) & (cvp.stains == "HE+PAS")]
    cv = cv[cv.participant.isin(y.index)].set_index("participant")[c["select_by"]]
    picks = {"A_high": cv.idxmax(), "B_low": cv.idxmin()}

    import openslide
    sel_rows, outs = [], []
    for lab, pid in picks.items():
        tr = [p for p in y.index if p != pid]
        rng = np.random.default_rng(seed)
        p_loo, alpha = hl.fit_predict(emb.loc[tr].values, y.loc[tr].values, emb.loc[[pid]].values,
                                      h["cv"]["ridge_alphas"], h["cv"]["inner_folds"], rng)
        mu, sd, beta, ym = ridge_coef(emb.loc[tr].values.astype(np.float64), y.loc[tr].values, alpha)
        f = slide_file(emb_dir, pid, c["stain"])
        z = np.load(f, allow_pickle=True)
        E = z["emb"].astype(np.float32).astype(np.float64)
        score = ((E - mu) / sd) @ beta + ym
        assert abs(score.mean() - p_loo[0]) < 1e-3, (score.mean(), p_loo[0])
        coords = z["coords"]
        ts = pd.DataFrame({"x0": coords[:, 0], "y0": coords[:, 1], "tile_score": score,
                           "tubule_rich": z["tubule"].astype(bool)})

        slide = openslide.OpenSlide(str(Path(c["slide_dir"]) / pid / (f.stem.split("__", 2)[2] + ".svs")))
        mpp = hl_mpp(slide)
        size0 = int(round(c["tile_px"] * c["target_um_per_px"] / mpp))
        ts["size0"] = size0
        ts.to_csv(out / f"tile_scores_{lab}.tsv", sep="\t", index=False)
        # 组织缩略图：只截取 tile 覆盖的组织范围
        x0, y0 = int(ts.x0.min()), int(ts.y0.min())
        x1, y1 = int(ts.x0.max()) + size0, int(ts.y0.max()) + size0
        ds = max(x1 - x0, y1 - y0) / c["thumb_max_px"]
        lvl = slide.get_best_level_for_downsample(ds)
        lds = slide.level_downsamples[lvl]
        reg = slide.read_region((x0, y0), lvl, (int((x1 - x0) / lds), int((y1 - y0) / lds))).convert("RGB")
        reg = reg.resize((int((x1 - x0) / ds), int((y1 - y0) / ds)), Image.LANCZOS)
        reg.save(out / f"thumb_{lab}.png")
        # 示例 tile：高者取得分最高、低者取得分最低
        order = ts.tile_score.sort_values(ascending=(lab == "B_low")).index[:c["n_example_tiles"]]
        for k, i in enumerate(order, 1):
            t = slide.read_region((int(ts.x0[i]), int(ts.y0[i])), 0, (size0, size0)).convert("RGB")
            t.resize((c["tile_px"], c["tile_px"]), Image.LANCZOS).save(out / "tiles" / f"{lab}_{k}.png")
            outs.append(out / "tiles" / f"{lab}_{k}.png")
        sel_rows.append({"label": lab, "participant": pid, "diagnosis": mp.diagnosis.get(pid),
                         "measured": y[pid], "cv_predicted": cv[pid], "loo_slide_prediction": p_loo[0],
                         "measured_injured_tal_frac": axes.loc[pid, "tal_injured_frac"],
                         "loo_alpha": alpha, "n_tiles": len(ts), "mpp": mpp, "size0": size0,
                         "thumb_origin_x0": x0, "thumb_origin_y0": y0, "thumb_downsample": ds,
                         "example_tiles": ";".join(f"{lab}_{k}" for k in range(1, len(order) + 1)),
                         "example_tile_scores": ";".join(f"{ts.tile_score[i]:.3f}" for i in order),
                         "example_tile_x0y0": ";".join(f"{ts.x0[i]},{ts.y0[i]}" for i in order),
                         "n_train": len(tr)})
        outs += [out / f"tile_scores_{lab}.tsv", out / f"thumb_{lab}.png"]
        print(sel_rows[-1], flush=True)
    pd.DataFrame(sel_rows).to_csv(out / "selection.tsv", sep="\t", index=False)
    outs.append(out / "selection.tsv")
    write_provenance(STAGE, inputs=[f_ax, f_map, f_cv], outputs=outs, seed=seed,
                     extra={"embeddings": str(emb_dir), "slides": c["slide_dir"], "params": c})


def hl_mpp(slide) -> float:
    v = slide.properties.get("openslide.mpp-x") or slide.properties.get("aperio.MPP")
    return float(v) if v else 0.25


if __name__ == "__main__":
    main()
