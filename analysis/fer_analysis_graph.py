from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import japanize_matplotlib  # noqa

# ========= 見た目の基本設定 =========
plt.rcParams.update({
    "font.family": "Meiryo",
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 0.8,
    "figure.dpi": 120,
})

# パレット（adgreen）
PALETTE = {
    "adgreen":["#01916D","#34A78A","#67BDA7","#99D3C5","#CCE9E2","#E6F4F0","#014937"],
    "gray":["#333333","#5C5C5C","#858585","#ADADAD","#D6D6D6","#EBEBEB"],
}

def _nice_hist_bins(x: pd.Series, target_bins: int = 35) -> int:
    """外れ値耐性を少し上げたヒストビン数（単純だけど実務で安定）"""
    x = x.dropna().values
    if len(x) < 30:
        return max(10, int(np.sqrt(max(len(x), 1))))
    q1, q3 = np.percentile(x, [25, 75])
    iqr = max(q3 - q1, 1e-12)
    bw = 2 * iqr / (len(x) ** (1/3))  # Freedman–Diaconis
    if bw <= 0:
        return target_bins
    bins = int(np.ceil((x.max() - x.min()) / bw))
    return int(np.clip(bins, 20, 60))

def _beautify_axes(ax):
    """スパイン/グリッドを整えて、資料向けの見た目にする"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", axis="both", alpha=0.25)
    ax.tick_params(direction="out", length=3, width=0.8)

def plot_width_profile_with_hist(
    df: pd.DataFrame,
    title: str = "",
    save_path: Path | None = None,
    show: bool = True,
):
    """
    df columns: ["Transport", "WidthPos", "Xe", "Rd"]
    横軸: WidthPos, 縦軸: Xe/Rd
    幅手位置ごとに、Transport順（=データ順）で線結合
    右に縦方向ヒストグラムを添える（sharey）
    """
    c = PALETTE["adgreen"]
    line_c = c[0]
    point_c = c[2]
    hist_c = c[4]
    edge_c = c[0]

    # 幅手位置を昇順に（離散15点を想定）
    df = df.copy()
    df["WidthPos"] = pd.to_numeric(df["WidthPos"], errors="coerce")
    df["Transport"] = pd.to_numeric(df["Transport"], errors="coerce")

    # 図レイアウト
    fig = plt.figure(figsize=(13.8, 5.4))
    gs = GridSpec(1, 4, width_ratios=[4.3, 1.2, 4.3, 1.2], wspace=0.30)

    # ================== Xe ==================
    ax_xe = fig.add_subplot(gs[0])
    # 幅手位置ごとにMD順で線結合（xは同値なので“縦の折れ線”になる）
    for w, g in df.groupby("WidthPos", sort=True):
        g = g.sort_values("Transport")
        ax_xe.plot(g["WidthPos"], g["Xe"], color=line_c, lw=0.9, alpha=0.85, zorder=2)
        ax_xe.scatter(g["WidthPos"], g["Xe"], s=8, color=point_c, alpha=0.75, zorder=3)

    ax_xe.set_xlabel("幅手位置 [mm]")
    ax_xe.set_ylabel("Xe")
    ax_xe.set_title(f"{title}  Xe" if title else "Xe")
    _beautify_axes(ax_xe)

    ax_xe_h = fig.add_subplot(gs[1], sharey=ax_xe)
    bins_xe = _nice_hist_bins(df["Xe"])
    ax_xe_h.hist(
        df["Xe"].dropna(),
        bins=bins_xe,
        orientation="horizontal",
        color=hist_c,
        edgecolor=edge_c,
        linewidth=0.6,
    )
    ax_xe_h.set_xlabel("Count")
    ax_xe_h.grid(True, axis="x", alpha=0.25)
    ax_xe_h.spines["top"].set_visible(False)
    ax_xe_h.spines["right"].set_visible(False)
    ax_xe_h.tick_params(direction="out", length=3, width=0.8)
    plt.setp(ax_xe_h.get_yticklabels(), visible=False)

    # ================== Rd ==================
    ax_rd = fig.add_subplot(gs[2])
    for w, g in df.groupby("WidthPos", sort=True):
        g = g.sort_values("Transport")
        ax_rd.plot(g["WidthPos"], g["Rd"], color=line_c, lw=0.9, alpha=0.85, zorder=2)
        ax_rd.scatter(g["WidthPos"], g["Rd"], s=8, color=point_c, alpha=0.75, zorder=3)

    ax_rd.set_xlabel("幅手位置 [mm]")
    ax_rd.set_ylabel("Rd")
    ax_rd.set_title(f"{title}  Rd" if title else "Rd")
    _beautify_axes(ax_rd)

    ax_rd_h = fig.add_subplot(gs[3], sharey=ax_rd)
    bins_rd = _nice_hist_bins(df["Rd"])
    ax_rd_h.hist(
        df["Rd"].dropna(),
        bins=bins_rd,
        orientation="horizontal",
        color=hist_c,
        edgecolor=edge_c,
        linewidth=0.6,
    )
    ax_rd_h.set_xlabel("Count")
    ax_rd_h.grid(True, axis="x", alpha=0.25)
    ax_rd_h.spines["top"].set_visible(False)
    ax_rd_h.spines["right"].set_visible(False)
    ax_rd_h.tick_params(direction="out", length=3, width=0.8)
    plt.setp(ax_rd_h.get_yticklabels(), visible=False)

    # x軸に少し余白（端が詰まらないように）
    for ax in (ax_xe, ax_rd):
        xmin, xmax = ax.get_xlim()
        pad = (xmax - xmin) * 0.04
        ax.set_xlim(xmin - pad, xmax + pad)

    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
