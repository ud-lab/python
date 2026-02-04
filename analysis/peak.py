import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from pathlib import Path


# -----------------------------
# 1) CSVローダ（ヘッダ有無どちらもなるべく吸収）
# -----------------------------
def load_2d_csv(csv_path: str) -> np.ndarray:
    """
    CSVを2次元の数値配列として読み込む。
    ヘッダや非数値列が混ざっていても、数値として解釈できる列だけ拾う方針。
    """
    df = pd.read_csv(csv_path, header=None)
    # 数値化（失敗はNaN）
    df_num = df.apply(pd.to_numeric, errors="coerce")
    # 全部NaNの列/行は落とす（ヘッダ文字列などを想定）
    df_num = df_num.dropna(axis=0, how="all").dropna(axis=1, how="all")
    # まだNaNが混じる場合は前後から補間/穴埋めするより、ここでは落とすのが安全
    # → 行/列まるごとNaNは落としたので、残りのNaNは0埋め等もありだが、まずは除外的に補完
    arr = df_num.to_numpy(dtype=float)
    if np.isnan(arr).any():
        # どうしても混ざる場合の保険：NaNを近傍で埋める（簡易に0）
        arr = np.nan_to_num(arr, nan=0.0)
    if arr.ndim != 2 or min(arr.shape) < 2:
        raise ValueError(f"2次元データとして読み込めませんでした: shape={arr.shape}")
    return arr


# -----------------------------
# 2) 1本の横プロファイルから山・谷を抽出
# -----------------------------
def extract_peaks_troughs_1d(
    y: np.ndarray,
    *,
    distance: int = 5,
    prominence: float | None = None,
    height: float | None = None,
):
    """
    y: 1Dプロファイル
    distance: 山同士の最小距離(サンプル数)
    prominence: ピークの顕著性(ノイズ除去に効く)。Noneなら未指定。
    height: ピーク高さの下限。Noneなら未指定。

    return:
      peaks_idx, peaks_val, troughs_idx, troughs_val
      troughs_val は実値（通常は負側）を返す。分布化時に abs() を取る。
    """
    y = np.asarray(y, dtype=float)

    peaks_idx, peaks_prop = find_peaks(
        y,
        distance=distance,
        prominence=prominence,
        height=height,
    )
    troughs_idx, troughs_prop = find_peaks(
        -y,
        distance=distance,
        prominence=prominence,
        height=None if height is None else -height,  # 厳密には同条件にしたいなら調整
    )

    peaks_val = y[peaks_idx]
    troughs_val = y[troughs_idx]
    return peaks_idx, peaks_val, troughs_idx, troughs_val


# -----------------------------
# 3) 中心1ラインと中心±band_rowsの帯領域を解析
# -----------------------------
def analyze_center_and_band(
    I: np.ndarray,
    *,
    band_rows: int = 100,     # ±100行（=±5cm）
    distance: int = 5,
    prominence: float | None = None,
    height: float | None = None,
):
    nrows, ncols = I.shape
    r0 = nrows // 2

    # --- 中心ライン ---
    y_center = I[r0, :]
    p_i, p_v, t_i, t_v = extract_peaks_troughs_1d(
        y_center, distance=distance, prominence=prominence, height=height
    )

    center_result = {
        "row_index": r0,
        "peaks_idx": p_i,
        "peaks_val": p_v,
        "troughs_idx": t_i,
        "troughs_val": t_v,
    }

    # --- 帯領域（中心±band_rows） ---
    r_start = max(0, r0 - band_rows)
    r_end   = min(nrows - 1, r0 + band_rows)

    all_peaks = []
    all_troughs = []
    per_row_counts = []

    for r in range(r_start, r_end + 1):
        y = I[r, :]
        p_i, p_v, t_i, t_v = extract_peaks_troughs_1d(
            y, distance=distance, prominence=prominence, height=height
        )
        all_peaks.append(p_v)
        all_troughs.append(t_v)
        per_row_counts.append((r, len(p_v), len(t_v)))

    band_peaks = np.concatenate(all_peaks) if len(all_peaks) else np.array([])
    band_troughs = np.concatenate(all_troughs) if len(all_troughs) else np.array([])

    band_result = {
        "row_range": (r_start, r_end),
        "peaks_val": band_peaks,
        "troughs_val": band_troughs,
        "per_row_counts": np.array(per_row_counts, dtype=int),
    }

    return center_result, band_result


# -----------------------------
# 4) 分布作成（ヒストグラム）+ 可視化
# -----------------------------
def plot_profile_with_marks(y, peaks_idx, troughs_idx, title="profile"):
    x = np.arange(len(y))
    plt.figure()
    plt.plot(x, y)
    if len(peaks_idx):
        plt.plot(peaks_idx, y[peaks_idx], "o", label="peaks")
    if len(troughs_idx):
        plt.plot(troughs_idx, y[troughs_idx], "x", label="troughs")
    plt.title(title)
    plt.xlabel("column index")
    plt.ylabel("intensity")
    plt.legend()
    plt.tight_layout()


def plot_distributions(peaks_val, troughs_val, title_prefix="dist", bins=50):
    peaks_val = np.asarray(peaks_val, dtype=float)
    troughs_abs = np.abs(np.asarray(troughs_val, dtype=float))  # 谷は絶対値で分布化

    plt.figure()
    plt.hist(peaks_val, bins=bins)
    plt.title(f"{title_prefix}: peak heights")
    plt.xlabel("peak intensity")
    plt.ylabel("count")
    plt.tight_layout()

    plt.figure()
    plt.hist(troughs_abs, bins=bins)
    plt.title(f"{title_prefix}: |trough| heights")
    plt.xlabel("|trough intensity|")
    plt.ylabel("count")
    plt.tight_layout()


def save_results(out_dir: str, center_result, band_result):
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    # 中心ラインのピーク/谷
    pd.DataFrame({
        "peaks_idx": center_result["peaks_idx"],
        "peaks_val": center_result["peaks_val"],
    }).to_csv(outp / "center_peaks.csv", index=False)

    pd.DataFrame({
        "troughs_idx": center_result["troughs_idx"],
        "troughs_val": center_result["troughs_val"],
        "troughs_abs": np.abs(center_result["troughs_val"]),
    }).to_csv(outp / "center_troughs.csv", index=False)

    # 帯領域（集約）
    pd.DataFrame({
        "peaks_val": band_result["peaks_val"],
    }).to_csv(outp / "band_peaks.csv", index=False)

    pd.DataFrame({
        "troughs_val": band_result["troughs_val"],
        "troughs_abs": np.abs(band_result["troughs_val"]),
    }).to_csv(outp / "band_troughs.csv", index=False)

    # 行ごとの検出数（品質チェック用）
    prc = band_result["per_row_counts"]
    pd.DataFrame(prc, columns=["row", "n_peaks", "n_troughs"]).to_csv(outp / "band_counts_per_row.csv", index=False)


# -----------------------------
# 5) 使い方（JupyterでもOK）
# -----------------------------
def run(csv_path: str, out_dir: str = "out", band_rows: int = 100,
        distance: int = 5, prominence: float | None = None, height: float | None = None,
        bins: int = 50):
    I = load_2d_csv(csv_path)

    center_result, band_result = analyze_center_and_band(
        I,
        band_rows=band_rows,
        distance=distance,
        prominence=prominence,
        height=height,
    )

    # 中心ラインの可視化
    y_center = I[center_result["row_index"], :]
    plot_profile_with_marks(
        y_center,
        center_result["peaks_idx"],
        center_result["troughs_idx"],
        title=f"Center row profile (row={center_result['row_index']})"
    )
    plot_distributions(
        center_result["peaks_val"],
        center_result["troughs_val"],
        title_prefix="Center row",
        bins=bins
    )

    # 帯領域の分布
    plot_distributions(
        band_result["peaks_val"],
        band_result["troughs_val"],
        title_prefix=f"Band rows {band_result['row_range'][0]}..{band_result['row_range'][1]}",
        bins=bins
    )

    save_results(out_dir, center_result, band_result)

    # ざっくり統計も返す
    summary = {
        "center": {
            "n_peaks": int(len(center_result["peaks_val"])),
            "n_troughs": int(len(center_result["troughs_val"])),
            "peak_mean": float(np.mean(center_result["peaks_val"])) if len(center_result["peaks_val"]) else np.nan,
            "trough_abs_mean": float(np.mean(np.abs(center_result["troughs_val"]))) if len(center_result["troughs_val"]) else np.nan,
        },
        "band": {
            "row_range": band_result["row_range"],
            "n_peaks": int(len(band_result["peaks_val"])),
            "n_troughs": int(len(band_result["troughs_val"])),
            "peak_mean": float(np.mean(band_result["peaks_val"])) if len(band_result["peaks_val"]) else np.nan,
            "trough_abs_mean": float(np.mean(np.abs(band_result["troughs_val"]))) if len(band_result["troughs_val"]) else np.nan,
        }
    }
    return summary


# 実行例（Jupyterならセルで）
# summary = run("your_data.csv", out_dir="out", band_rows=100, distance=8, prominence=2.0, bins=60)
# print(summary)
# plt.show()
