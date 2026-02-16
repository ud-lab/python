import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, label
from scipy.stats import median_abs_deviation


def read_csv_2d(csv_path: str, header=None, skiprows: int = 0) -> np.ndarray:
    df = pd.read_csv(csv_path, header=header, skiprows=skiprows)
    return df.to_numpy(dtype=float)


def detect_top_vertical_mura_with_mask(
    X: np.ndarray,
    pixel_mm: float = 0.5,
    center_width_mm: float = 120.0,
    short_mm: float = 8.0,
    long_mm: float = 100.0,
    z_thresh: float = 4.2,
    min_height_mm: float = 10.0,
    target_width_mm: float = 3.0,
    width_tol_mm: float = 1.5,
    top_n: int = 5
):
    """
    戻り値:
      topN: 上位N件（score降順）
      mask_full: 元画像座標の2値マスク（中央領域のみ有効）
      x_range_global: (x0, x1) 中央領域
    """
    Y, W = X.shape

    # --- 中央領域 ---
    half = int(round((center_width_mm / pixel_mm) / 2))
    xc = W // 2
    x0 = max(0, xc - half)
    x1 = min(W, xc + half)
    Xc = X[:, x0:x1]

    # --- 背景除去（y方向） ---
    long_sigma = (long_mm / pixel_mm) / 2.0
    bg = gaussian_filter1d(Xc, sigma=long_sigma, axis=0)
    R = Xc - bg

    # --- y方向バンドパス ---
    short_sigma = (short_mm / pixel_mm) / 2.0
    smooth_short = gaussian_filter1d(R, sigma=short_sigma, axis=0)
    smooth_long  = gaussian_filter1d(R, sigma=long_sigma, axis=0)
    band = smooth_short - smooth_long

    # --- ロバストz ---
    med = np.median(band)
    mad = median_abs_deviation(band)
    z = (band - med) / (mad + 1e-9)

    mask_c = np.abs(z) > z_thresh

    # --- 元画像座標へ戻したマスク（中央領域のみ） ---
    mask_full = np.zeros((Y, W), dtype=bool)
    mask_full[:, x0:x1] = mask_c

    # --- 連結成分解析（中央領域側でbboxを取る） ---
    labeled_c, num = label(mask_c)

    min_height_px = (min_height_mm / pixel_mm)
    target_width_px = (target_width_mm / pixel_mm)
    width_tol_px = (width_tol_mm / pixel_mm)

    comps = []
    for comp_id in range(1, num + 1):
        ys, xs = np.where(labeled_c == comp_id)
        if ys.size == 0:
            continue

        y0p, y1p = int(ys.min()), int(ys.max())  # inclusive
        x0c, x1c = int(xs.min()), int(xs.max())  # inclusive

        height_px = y1p - y0p
        width_px  = x1c - x0c

        if height_px < min_height_px:
            continue
        if not (target_width_px - width_tol_px <= width_px <= target_width_px + width_tol_px):
            continue
        if height_px / (width_px + 1e-6) < 2.0:
            continue

        score = float(np.max(np.abs(z[y0p:y1p+1, x0c:x1c+1])))

        # 元座標
        gx0 = x0 + x0c
        gx1 = x0 + x1c

        comps.append({
            "score": score,
            "y_start_px": y0p, "y_end_px": y1p,
            "x_start_px": gx0, "x_end_px": gx1,
            "y_start_mm": y0p * pixel_mm, "y_end_mm": y1p * pixel_mm,
            "x_start_mm": gx0 * pixel_mm, "x_end_mm": gx1 * pixel_mm,
        })

    comps.sort(key=lambda d: d["score"], reverse=True)
    topN = comps[:top_n]
    return topN, mask_full, (x0, x1)


def plot_original(X: np.ndarray, pixel_mm: float = 0.5, title: str = "Original map"):
    Y, W = X.shape
    extent = [0, W * pixel_mm, Y * pixel_mm, 0]
    plt.figure(figsize=(9, 6))
    plt.imshow(X, aspect="auto", extent=extent)
    plt.colorbar(label="value")
    plt.title(title)
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.tight_layout()
    plt.show()


def plot_highlight_with_contour(
    X: np.ndarray,
    topN,
    mask_full: np.ndarray,
    pixel_mm: float = 0.5,
    title: str = "Top 5 mura contours"
):
    """
    2個目の図：上位N本の「輪郭」を強調（contour）
    """
    Y, W = X.shape
    extent = [0, W * pixel_mm, Y * pixel_mm, 0]

    # 上位Nのbbox部分だけマスクを残す（輪郭用）
    selected = np.zeros_like(mask_full, dtype=bool)
    for r in topN:
        ys, ye = r["y_start_px"], r["y_end_px"]
        xs, xe = r["x_start_px"], r["x_end_px"]
        selected[ys:ye+1, xs:xe+1] |= mask_full[ys:ye+1, xs:xe+1]

    plt.figure(figsize=(9, 6))
    plt.imshow(X, aspect="auto", extent=extent)
    plt.colorbar(label="value")

    # 輪郭を重ねる（0/1マスクの0.5等値線）
    # extent付きimshowに合わせて、contour側もextentを渡す
    x = np.linspace(0, W * pixel_mm, W)
    y = np.linspace(0, Y * pixel_mm, Y)
    plt.contour(x, y, selected.astype(float), levels=[0.5], linewidths=2)

    plt.title(title)
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.tight_layout()
    plt.show()


# ====== 使い方 ======
if __name__ == "__main__":
    csv_path = "Book.csv"
    pixel_mm = 0.5

    X = read_csv_2d(csv_path, header=None, skiprows=0)

    top5, mask_full, xr = detect_top_vertical_mura_with_mask(
        X,
        pixel_mm=pixel_mm,
        center_width_mm=120.0,
        short_mm=8.0,
        long_mm=100.0,
        z_thresh=4.2,
        min_height_mm=10.0,
        target_width_mm=3.0,
        width_tol_mm=1.5,
        top_n=5
    )

    print("=== Top 5 ===")
    for i, r in enumerate(top5, 1):
        print(i, r)

    # 1) 元画像だけ
    plot_original(X, pixel_mm=pixel_mm, title="Original map (mm axes)")

    # 2) 上位5本の輪郭を強調した図
    plot_highlight_with_contour(
        X, top5, mask_full, pixel_mm=pixel_mm,
        title="Original map + Top 5 mura contours (mm axes)"
    )
