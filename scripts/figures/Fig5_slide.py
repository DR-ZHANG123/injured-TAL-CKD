"""Fig 5：常规活检切片估计损伤 TAL 负担，且超出病理医生评分。

只读 results/11_histology/、results/20_slide_robustness/ 的表与 results/17_fig45_tiles/ 的 tile 图像；
每个 panel 的绘图输入写到 figures/source_data/Fig5_*.tsv。参与者编号不进入图与 source data。
输出 figures/Fig5.pdf（矢量）与 figures/Fig5.png（300 dpi）；之后运行 finalize_sizes.py。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from FigS_common import LEGEND_KW, RES, SRC, panel_letters, save  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Liberation Sans", "Arial", "DejaVu Sans"]
R11, R17, R20 = RES / "11_histology", RES / "17_fig45_tiles", RES / "20_slide_robustness"
DX_COL = {"DKD": "#C44E52", "HKD": "#4C72B0", "other": "#8C8C8C"}
DX_LAB = {"DKD": "DKD", "HKD": "HKD", "other": "Reference or other"}
INK, OPEN = "#333333", "#8C8C8C"
MINUS = "−"


def r2(v: float) -> str:
    return f"{v:.2f}".replace("-", MINUS)


def panel_a(ax, cvp, cvr, mp, sel):
    d = cvp[(cvp.target == "tal_injured_logit") & (cvp.tileset == "all") & (cvp.stains == "HE+PAS")].copy()
    d = d.drop_duplicates(subset=["participant"])
    d["diagnosis"] = d.participant.map(mp.diagnosis).fillna("other")
    lab = {v: k[0] for k, v in sel.participant.items()}
    d["highlight"] = d.participant.map(lab).fillna("")
    r = cvr[(cvr.target == "tal_injured_logit") & (cvr.tileset == "all") & (cvr.stains == "HE+PAS")].iloc[0]
    for dx in ["DKD", "HKD", "other"]:
        s = d[d.diagnosis == dx]
        ax.scatter(s.measured, s.cv_predicted, s=14, c=DX_COL[dx], edgecolor="white", lw=0.3, label=DX_LAB[dx],
                   zorder=3)
    b = np.polyfit(d.measured, d.cv_predicted, 1)
    xs = np.linspace(d.measured.min(), d.measured.max(), 50)
    ax.plot(xs, np.polyval(b, xs), color=INK, lw=0.8, zorder=2)
    for _, row in d[d.highlight != ""].iterrows():
        ax.scatter(row.measured, row.cv_predicted, s=32, facecolor="none", edgecolor="black", lw=0.7, zorder=4)
        ax.annotate({"A": "1", "B": "2"}[row.highlight], (row.measured, row.cv_predicted),
                    xytext=(-9, -1) if row.highlight == "A" else (5, 3), textcoords="offset points",
                    fontsize=6.5, fontweight="bold")
    p_lab = "P ≤ 0.001" if r.perm_p_pearson <= 0.0011 else f"P = {r.perm_p_pearson:.3f}"
    lo, hi = d.cv_predicted.min(), d.cv_predicted.max()
    ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.30 * (hi - lo))
    ax.set_xlabel("Measured injured TAL\n(single-nucleus, logit fraction)", fontweight="bold")
    ax.set_ylabel("Slide-estimated injured TAL\n(cross-validated, logit)", fontweight="bold")
    ax.legend(loc="upper right", fontsize=5.8, handletextpad=0.1, borderaxespad=0.3, borderpad=0.3, **{
        k: v for k, v in LEGEND_KW.items() if k != "borderpad"})
    d["participant"] = [f"P{i:02d}" for i in range(1, len(d) + 1)]
    d["highlight"] = d.highlight.map({"A": "Participant 1", "B": "Participant 2"}).fillna("")
    d[["participant", "diagnosis", "measured", "cv_predicted", "highlight"]].to_csv(
        SRC / "Fig5_a_cv_tal.tsv", sep="\t", index=False)
    pd.DataFrame([{"n": int(r.n), "cv_pearson_r": r.cv_pearson_r, "perm_p_pearson": r.perm_p_pearson,
                   "n_perm": int(r.n_perm), "label": p_lab}]).to_csv(SRC / "Fig5_a_stats.tsv", sep="\t", index=False)


def perf_rows():
    w = pd.read_csv(R20 / "within_diagnosis_cv.tsv", sep="\t").set_index("analysis")
    t = pd.read_csv(R20 / "temporal_validation.tsv", sep="\t").set_index("analysis")
    db = pd.read_csv(R20 / "descriptor_baseline_cv.tsv", sep="\t").set_index("analysis")
    rows = [("All participants", w.loc["pooled_model_all_training"], "slide"),
            ("Within DKD", w.loc["pooled_model_DKD"], "slide"),
            ("Within DKD + HKD", w.loc["pooled_model_DKD+HKD"], "slide"),
            ("Later biopsies (temporal)", t.loc["temporal_accession_time"], "slide"),
            ("Pathologist scores only", db.loc["descriptors"], "descriptors")]
    src = {"All participants": "within_diagnosis_cv:pooled_model_all_training",
           "Within DKD": "within_diagnosis_cv:pooled_model_DKD",
           "Within DKD + HKD": "within_diagnosis_cv:pooled_model_DKD+HKD",
           "Later biopsies (temporal)": "temporal_validation:temporal_accession_time",
           "Tubule-enriched tiles": "within_diagnosis_cv:pooled_model_tubule_HE+PAS",
           "Pathologist scores only": "descriptor_baseline_cv:descriptors"}
    out = pd.DataFrame([{"setting": lab, "model": kind, "n": int(s.n), "r": s.pearson_r, "ci_low": s.ci_low,
                         "ci_high": s.ci_high, "p": s.p, "source": src[lab]} for lab, s, kind in rows])
    dr = pd.read_csv(R20 / "descriptor_delta_r.tsv", sep="\t").set_index("comparison")
    return out, dr.loc["descriptors+slide minus descriptors"]


def panel_b(ax, out, dr):
    y = np.array([6.0, 5.0, 4.0, 3.0, 1.6])
    ax.axvline(0, color="#BBBBBB", lw=0.5, zorder=1)
    for yi, (_, r) in zip(y, out.iterrows()):
        filled = r.model == "slide"
        ax.plot([r.ci_low, r.ci_high], [yi, yi], color=INK, lw=0.9, zorder=2)
        ax.scatter(r.r, yi, s=20, marker="o", facecolor=INK if filled else "white", edgecolor=INK, lw=0.8, zorder=3)
        ax.text(1.0, yi, f"{r.r:.2f}", ha="left", va="center", fontsize=6.0)
    ax.axhline(2.3, color="#CCCCCC", lw=0.5, ls=":")
    ax.set_yticks(y)
    short = {"All participants": "All participants", "Within DKD": "Within DKD", "Within DKD + HKD": "Within DKD + HKD",
             "Later biopsies (temporal)": "Later biopsies", "Tubule-enriched tiles": "Tubule-rich tiles",
             "Pathologist scores only": "Pathologist scores"}
    ax.set_yticklabels([f"{short[s]}\n(n = {n})" for s, n in zip(out.setting, out.n)], fontsize=6.0, linespacing=0.95)
    ax.text(1.0, 6.75, "r", ha="left", va="center", fontsize=6.0, fontweight="bold")
    # 增量：在描述评分之上加入切片读出的 Δr
    x0, x1 = out.r.iloc[-1], out.r.iloc[0]
    yb = 0.0
    ax.set_xlim(0, 1.1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylim(0.9, 7.1)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Correlation with measured\ninjured TAL (r, 95% CI)", fontweight="bold")
    o = out.copy()
    o.loc[len(o)] = {"setting": "Increment: slide added to pathologist scores (delta r)", "model": "delta_r",
                     "n": int(out.n.iloc[0]), "r": dr.delta_r, "ci_low": dr.ci_low, "ci_high": dr.ci_high,
                     "p": dr.p_boot_two_sided, "source": "descriptor_delta_r:descriptors+slide minus descriptors"}
    o.to_csv(SRC / "Fig5_b_performance.tsv", sep="\t", index=False)


def panel_c(ax, cvr):
    r11 = cvr[(cvr.target == "tal_injured_logit") & (cvr.tileset == "all") & (cvr.stains == "HE+PAS")].iloc[0]
    db = pd.read_csv(R20 / "descriptor_baseline_cv.tsv", sep="\t").set_index("analysis")
    alld = db.loc["partial_r_slide_cv_adj_all_descriptors+diagnosis"]
    d = pd.DataFrame([
        {"adjustment": "None (cross-validated r)", "tick": "None", "n": int(r11.n), "r": r11.cv_pearson_r,
         "p": r11.perm_p_pearson},
        {"adjustment": "Diagnosis", "tick": "Diagnosis", "n": int(r11.n), "r": r11.partial_r_diagnosis,
         "p": r11.partial_p_diagnosis},
        {"adjustment": "Diagnosis + interstitial fibrosis", "tick": "Diagnosis +\nfibrosis", "n": int(r11.n_IF),
         "r": r11.partial_r_diagnosis_IF, "p": r11.partial_p_diagnosis_IF},
        {"adjustment": "Diagnosis + all pathologist scores", "tick": "Diagnosis +\nall scores", "n": int(alld.n),
         "r": alld.pearson_r, "p": alld.p}])
    y = np.arange(len(d))
    cols = [INK, "#6B6B6B", "#6B6B6B", "#6B6B6B"]
    ax.barh(y, d.r, 0.62, color=cols, edgecolor="white", lw=0.3, zorder=2)
    for yi, v in zip(y, d.r):
        ax.text(v + 0.02, yi, f"{v:.2f}", ha="left", va="center", fontsize=6.0)
    ax.set_yticks(y)
    ax.set_yticklabels(d.tick, fontsize=6.0, linespacing=1.0)
    ax.set_ylim(len(d) - 0.45, -0.55)
    ax.set_xlim(0, 0.9)
    ax.set_xticks([0, 0.25, 0.5, 0.75])
    ax.tick_params(axis="y", length=0)
    ax.set_ylabel("Covariates adjusted for", fontweight="bold")
    ax.set_xlabel("Correlation with measured\ninjured TAL (partial r)", fontweight="bold")
    d.drop(columns="tick").to_csv(SRC / "Fig5_c_partial_r.tsv", sep="\t", index=False)


# ---------------------------------------------------------------- panel d：tile 分数图（自 stage 17 表与图像）
def first_section(ts: pd.DataFrame) -> pd.DataFrame:
    """切片上常有两张连续切面；只展示 x 方向最左侧的一张（按排序后 x 坐标最大间隙切分）。"""
    xs = np.sort(ts.x0.unique())
    gaps = np.diff(xs)
    cut = xs[np.argmax(gaps)] if gaps.max() > 5 * ts.size0.iloc[0] else xs.max()
    return ts[ts.x0 <= cut]


def score_raster(ts: pd.DataFrame):
    s0 = int(ts.size0.iloc[0])
    gx, gy = ((ts.x0 - ts.x0.min()) // s0).astype(int), ((ts.y0 - ts.y0.min()) // s0).astype(int)
    R = np.full((gy.max() + 1, gx.max() + 1), np.nan)
    R[gy, gx] = ts.tile_score
    return R, (ts.x0.min(), ts.x0.max() + s0, ts.y0.max() + s0, ts.y0.min())


def section_view(lab, s):
    ts = pd.read_csv(R17 / f"tile_scores_{lab}.tsv", sep="\t")
    sec = first_section(ts)
    th = Image.open(R17 / f"thumb_{lab}.png")
    ds, ox, oy = s.thumb_downsample, s.thumb_origin_x0, s.thumb_origin_y0
    x0, x1 = sec.x0.min(), sec.x0.max() + s.size0
    y0, y1 = sec.y0.min(), sec.y0.max() + s.size0
    crop = th.crop((int((x0 - ox) / ds), int((y0 - oy) / ds), int(np.ceil((x1 - ox) / ds)),
                    int(np.ceil((y1 - oy) / ds))))
    return sec, crop, (x0, x1, y0, y1)


def _clean(a):
    a.set_xticks([])
    a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(False)


def panel_d(fig, sel, box):
    """box = (left, bottom, width, height)（图坐标）。两名参与者并排；切面按同一物理比例尺绘制、顶端对齐。"""
    allsc = pd.concat([pd.read_csv(R17 / f"tile_scores_{l}.tsv", sep="\t").tile_score for l in ["A_high", "B_low"]])
    vlim = (float(np.floor(np.percentile(allsc, 2))), float(np.ceil(np.percentile(allsc, 98))))
    L, B, W, H = box
    fw, fh = fig.get_size_inches()
    views = {lab: section_view(lab, sel.loc[lab]) for lab in ["A_high", "B_low"]}
    um = {lab: max(v[2][3] - v[2][2], 1) * sel.loc[lab].mpp for lab, v in views.items()}
    head_h = 0.075
    in_per_um = (H - head_h) * fh / max(um.values())
    tile_in, gap_in = 0.66, 0.05
    first_axes, src, im = [], [], None
    xcur = L
    for k, lab in enumerate(["A_high", "B_low"]):
        s = sel.loc[lab]
        sec, crop, (x0, x1, y0, y1) = views[lab]
        w_in, h_in = (x1 - x0) * s.mpp * in_per_um, (y1 - y0) * s.mpp * in_per_um
        top = B + H - head_h
        ax_t = fig.add_axes([xcur, top - h_in / fh, w_in / fw, h_in / fh])
        ax_m = fig.add_axes([xcur + (w_in + 0.06) / fw, top - h_in / fh, w_in / fw, h_in / fh])
        ax_t.imshow(crop, extent=(x0, x1, y1, y0), interpolation="lanczos", aspect="auto")
        R, ext = score_raster(sec)
        ax_m.set_facecolor("#F2F2F2")
        im = ax_m.imshow(R, extent=ext, cmap="magma_r", vmin=vlim[0], vmax=vlim[1], interpolation="nearest",
                         aspect="auto")
        for a in (ax_t, ax_m):
            a.set_xlim(x0, x1)
            a.set_ylim(y1, y0)
            _clean(a)
        Lpx = 1000 / s.mpp
        xs0 = x0 + 0.06 * (x1 - x0)
        ax_t.plot([xs0, xs0 + Lpx], [y1 + 0.03 * (y1 - y0)] * 2, color="black", lw=1.2, clip_on=False)
        ax_t.text(xs0 + Lpx / 2, y1 + 0.05 * (y1 - y0), "1 mm", ha="center", va="top", fontsize=5.8)
        ax_t.set_title("PAS", fontsize=6.3, pad=2)
        ax_m.set_title("Tile score", fontsize=6.3, pad=2)
        scores = [float(v) for v in s.example_tile_scores.split(";")]
        tx = xcur + (2 * w_in + 0.06 + 0.12) / fw
        for j in range(4):
            r, c = divmod(j, 2)
            a = fig.add_axes([tx + c * (tile_in + gap_in) / fw, top - ((r + 1) * tile_in + r * gap_in) / fh,
                              tile_in / fw, tile_in / fh])
            a.imshow(Image.open(R17 / "tiles" / f"{lab}_{j + 1}.png"), interpolation="lanczos")
            _clean(a)
            a.text(0.03, 0.03, f"{scores[j]:.1f}".replace("-", MINUS), transform=a.transAxes, fontsize=5.8,
                   color="white", ha="left", va="bottom",
                   bbox=dict(boxstyle="square,pad=0.15", fc="black", ec="none", alpha=0.55))
        cap = "Highest-scoring tiles" if k == 0 else "Lowest-scoring tiles"
        fig.text(tx, top + 0.004, cap, fontsize=6.3, ha="left", va="bottom")
        fig.text(xcur, B + H - 0.012, f"Participant {k + 1}", fontsize=6.8, fontweight="bold", ha="left", va="top")
        first_axes.append((ax_t, tx + (2 * tile_in + gap_in) / fw, top))
        src.append(sec.assign(participant=f"Participant {k + 1}").drop(columns=["size0"]))
        pd.DataFrame({"participant": f"Participant {k + 1}", "tile": [f"{lab}_{j + 1}.png" for j in range(4)],
                      "tile_score": scores}).to_csv(SRC / f"Fig5_d_example_tiles_participant{k + 1}.tsv",
                                                    sep="\t", index=False)
        xcur = L + W * 0.5
    pd.concat(src).to_csv(SRC / "Fig5_d_tile_scores.tsv", sep="\t", index=False)
    pd.DataFrame({"participant": ["Participant 1", "Participant 2"], "diagnosis": sel.diagnosis.values,
                  "measured_logit": sel.measured.values,
                  "measured_injured_tal_frac": sel.measured_injured_tal_frac.values,
                  "cv_slide_estimate_HE+PAS": sel.cv_predicted.values,
                  "tile_mean_PAS_leave_one_out": sel.loo_slide_prediction.values,
                  "colour_limits": [f"{vlim[0]}..{vlim[1]}"] * 2,
                  "tile_images": ["results/17_fig45_tiles/tiles/A_high_1-4.png",
                                  "results/17_fig45_tiles/tiles/B_low_1-4.png"]}).to_csv(
        SRC / "Fig5_d_participants.tsv", sep="\t", index=False)
    return first_axes, im


def main():
    SRC.mkdir(parents=True, exist_ok=True)
    cvp = pd.read_csv(R11 / "cv_predictions_by_participant.tsv", sep="\t", dtype={"participant": str}).drop_duplicates()
    cvr = pd.read_csv(R11 / "cv_prediction.tsv", sep="\t").drop_duplicates(subset=["target", "tileset", "stains"])
    mp = pd.read_csv(R11 / "participant_map.tsv", sep="\t", index_col=0, dtype={0: str})
    sel = pd.read_csv(R17 / "selection.tsv", sep="\t", dtype={"participant": str}).set_index("label")
    fig = plt.figure(figsize=(6.6, 6.3))
    top = GridSpec(1, 3, figure=fig, width_ratios=[1, 0.8, 0.95], wspace=0.75, left=0.1, right=0.985,
                   top=0.955, bottom=0.575)
    ax_a, ax_b, ax_c = (fig.add_subplot(top[i]) for i in range(3))
    panel_a(ax_a, cvp, cvr, mp, sel)
    out, dr = perf_rows()
    panel_b(ax_b, out, dr)
    panel_c(ax_c, cvr)
    d_box = (0.03, 0.02, 0.95, 0.435)
    d_axes, im = panel_d(fig, sel, d_box)
    axB, xr, topB = d_axes[1]
    pb = axB.get_position()
    # 色条挂在参与者 2 的切面与示例图块下方，与该参与者整块等宽
    tiles_bottom = topB - (2 * 0.66 + 0.05) / fig.get_size_inches()[1]
    cax = fig.add_axes([pb.x0, min(pb.y0 - 0.075, tiles_bottom - 0.05), xr - pb.x0, 0.013])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Tile score", fontsize=6)
    cb.ax.tick_params(labelsize=5.8, width=0.5)
    cb.outline.set_linewidth(0.5)
    panel_letters(fig, [[("a", ax_a), ("b", ax_b), ("c", ax_c)]])
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    xa = inv.transform((ax_a.get_tightbbox(fig.canvas.get_renderer()).x0, 0))[0]
    fig.text(min(xa, d_box[0]), d_box[1] + d_box[3] + 0.012, "d", fontsize=11, fontweight="bold", ha="left",
             va="bottom")
    save(fig, "Fig5")


if __name__ == "__main__":
    main()
