from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =======================
# ユーザー設定（ここだけ触ればOK）
# =======================
TARGET_FOLDER = r"C:\data\excel_files"   # ★解析したいフォルダ
RECURSIVE = True                         # サブフォルダも含める
START_ROW = 46                           # データ開始行（1始まり）
USE_SHEET = None                         # Noneで先頭シート / "Sheet1" のように指定も可

# 列：E=搬送長, F=目的値, G=測定位置(mm)
USECOLS = "E:G"

# position(mm) の丸め刻み（Noneなら自動推定）
# 例：0.5, 0.1, 0.01 など
POS_ROUND_MM = None

# ヒートマップの横軸を「等間隔格子」に揃える（欠測が多いと見やすい）
# True推奨：位置が数値軸なら等間隔が自然
MAKE_UNIFORM_POSITION_GRID = True

# 等間隔格子の刻み（Noneなら丸め刻みと同じ or 自動推定）
POS_GRID_STEP_MM = None

# 分布図
HIST_BINS = 50

# ヒートマップの集約方法（同じ length×position に複数点がある場合）
HEATMAP_AGG = "mean"  # mean / median / max / min など

# 出力先（NoneならTARGET_FOLDER配下）
OUT_FOLDER = None
# =======================


def safe_sheet_name(s: str) -> str:
    s = str(s)
    s = re.sub(r"[\\/*\[\]:\?]", "_", s).strip()
    return s[:31] if s else "UNKNOWN"


def pos_to_sheet_name(pos_mm: float, decimals: int = 3) -> str:
    """
    Excelシート名は記号に弱いので、mm位置を安全な名前へ変換
    例: -2.5 -> pos_m2p500mm, 0 -> pos_0p000mm
    """
    txt = f"{pos_mm:.{decimals}f}"
    txt = txt.replace("-", "m").replace(".", "p")
    return safe_sheet_name(f"pos_{txt}mm")


def estimate_step(values: np.ndarray) -> float | None:
    """
    値の刻み（最頻差分）をざっくり推定
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return None

    v = np.unique(np.sort(v))
    if v.size < 3:
        return None

    diffs = np.diff(v)
    diffs = diffs[(diffs > 0) & np.isfinite(diffs)]
    if diffs.size == 0:
        return None

    # 数値誤差が乗るので、差分を丸めて最頻値を見る
    # 桁はデータのレンジから適当に決める
    rng = np.nanmax(v) - np.nanmin(v)
    decimals = 4 if rng < 1000 else 3
    diffs_r = np.round(diffs, decimals)

    uniq, cnt = np.unique(diffs_r, return_counts=True)
    step = float(uniq[np.argmax(cnt)])
    if step <= 0:
        return None
    return step


def round_to_step(x: pd.Series, step: float) -> pd.Series:
    return (x / step).round() * step


def read_excel_data(path: Path) -> pd.DataFrame:
    df = pd.read_excel(
        path,
        engine="openpyxl",
        sheet_name=USE_SHEET,
        skiprows=START_ROW - 1,
        usecols=USECOLS,
        header=None,
    )
    df.columns = ["length", "value", "position"]

    # 空行削除
    df = df.dropna(how="all")

    # 型変換
    df["length"] = pd.to_numeric(df["length"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["position"] = pd.to_numeric(df["position"], errors="coerce")

    # 必須列欠損は落とす
    df = df.dropna(subset=["value", "position"])

    # 搬送長が無い点はヒートマップに使えないが、分布には使える
    # ここでは残しておき、ヒートマップ作成時に落とす
    return df


def make_position_processing(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    position(mm) を丸めて安定化し、丸め刻み(step)を返す
    """
    pos = df["position"].to_numpy(dtype=float)
    step = POS_ROUND_MM

    if step is None:
        step = estimate_step(pos) or 0.01  # 推定不能なら保険で0.01mm

    df = df.copy()
    df["position"] = round_to_step(df["position"], step)
    return df, float(step)


def export_grouped_excel(df: pd.DataFrame, out_xlsx: Path):
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    df_sorted = df.sort_values(["position", "length"], kind="mergesort")

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df_sorted.to_excel(writer, sheet_name="ALL", index=False)

        for pos, g in df_sorted.groupby("position", sort=True):
            sheet = pos_to_sheet_name(float(pos), decimals=3)
            g2 = g.sort_values("length", kind="mergesort")
            g2.to_excel(writer, sheet_name=sheet, index=False)


def plot_histograms(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 全体
    plt.figure()
    plt.hist(df["value"].values, bins=HIST_BINS)
    plt.xlabel("Value (F)")
    plt.ylabel("Count")
    plt.title("Overall distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "hist_overall.png", dpi=200)
    plt.close()

    # 位置別（数が多いと画像が大量になるので、必要ならここで絞ってもOK）
    for pos, g in df.groupby("position", sort=True):
        plt.figure()
        plt.hist(g["value"].values, bins=HIST_BINS)
        plt.xlabel("Value (F)")
        plt.ylabel("Count")
        plt.title(f"Distribution @ position(mm) = {pos:g}")
        plt.tight_layout()
        plt.savefig(out_dir / f"hist_pos_{pos_to_sheet_name(float(pos))}.png", dpi=200)
        plt.close()


def build_heatmap_pivot(df: pd.DataFrame, step_pos: float) -> pd.DataFrame:
    """
    index=length, columns=position(mm), values=value のpivotを作る
    """
    d2 = df.dropna(subset=["length"]).copy()
    d2 = d2.sort_values("length", kind="mergesort")

    pivot = pd.pivot_table(
        d2,
        index="length",
        columns="position",
        values="value",
        aggfunc=HEATMAP_AGG,
    )

    # 位置は数値順
    pivot = pivot.sort_index(axis=1)

    # 位置を等間隔格子に揃える（欠測位置があると列が飛び飛びになるのを防ぐ）
    if MAKE_UNIFORM_POSITION_GRID:
        grid_step = POS_GRID_STEP_MM if POS_GRID_STEP_MM is not None else step_pos
        if grid_step is None or grid_step <= 0:
            grid_step = 0.01

        cols = pivot.columns.to_numpy(dtype=float)
        if cols.size >= 2:
            cmin, cmax = float(np.nanmin(cols)), float(np.nanmax(cols))
            # 端を丸めてグリッド作成
            gmin = np.floor(cmin / grid_step) * grid_step
            gmax = np.ceil(cmax / grid_step) * grid_step
            grid = np.round(np.arange(gmin, gmax + grid_step * 0.5, grid_step), 10)
            pivot = pivot.reindex(columns=grid)

    return pivot


def plot_heatmap(pivot: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    mat = pivot.values

    plt.figure(figsize=(12, 6))
    im = plt.imshow(mat, aspect="auto", origin="upper")
    plt.colorbar(im, label="Value (F)")
    plt.xlabel("Position (mm)")
    plt.ylabel("Transport length (E)")
    plt.title("Heatmap: value by (length x position)")

    # x ticks（位置mm）
    x = pivot.columns.to_numpy(dtype=float)
    if x.size <= 25:
        plt.xticks(np.arange(x.size), [f"{v:g}" for v in x], rotation=90)
    else:
        step = max(1, x.size // 25)
        xs = np.arange(0, x.size, step)
        plt.xticks(xs, [f"{x[i]:g}" for i in xs], rotation=90)

    # y ticks（搬送長）
    y = pivot.index.to_numpy(dtype=float)
    if y.size <= 30:
        plt.yticks(np.arange(y.size), [f"{v:g}" for v in y])
    else:
        step = max(1, y.size // 30)
        ys = np.arange(0, y.size, step)
        plt.yticks(ys, [f"{y[i]:g}" for i in ys])

    plt.tight_layout()
    plt.savefig(out_dir / "heatmap_length_position.png", dpi=200)
    plt.close()


def export_position_summary(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = (
        df.groupby("position")["value"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .reset_index()
        .sort_values("position")
    )
    summary.to_csv(out_dir / "summary_by_position.csv", index=False, encoding="utf-8-sig")


def find_excel_files(folder: Path) -> list[Path]:
    patterns = folder.rglob("*") if RECURSIVE else folder.glob("*")
    files = []
    for p in patterns:
        if not p.is_file():
            continue
        if p.name.startswith("~$"):
            continue
        if p.suffix.lower() not in {".xlsx", ".xlsm"}:
            continue
        # 出力物を誤って再処理しない
        if p.name.startswith("Result_"):
            continue
        if "plots" in p.parts:
            continue
        files.append(p)
    return sorted(files)


def analyze_one_file(input_path: Path, out_base: Path):
    df_raw = read_excel_data(input_path)
    df, step_pos = make_position_processing(df_raw)

    # 出力先
    out_xlsx = out_base / f"Result_{input_path.stem}.xlsx"
    plot_dir = out_base / "plots" / input_path.stem
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 位置別Excel
    export_grouped_excel(df, out_xlsx)

    # 分布
    plot_histograms(df, plot_dir)

    # サマリー
    export_position_summary(df, plot_dir)

    # ヒートマップ
    pivot = build_heatmap_pivot(df, step_pos=step_pos)
    pivot.to_csv(plot_dir / "heatmap_pivot.csv", encoding="utf-8-sig")
    plot_heatmap(pivot, plot_dir)

    return {
        "file": str(input_path),
        "rows": int(len(df)),
        "pos_step_mm_used": float(step_pos),
        "out_excel": str(out_xlsx),
        "plot_dir": str(plot_dir),
    }


def main():
    base = Path(TARGET_FOLDER)
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"TARGET_FOLDER not found: {base}")

    out_base = Path(OUT_FOLDER) if OUT_FOLDER else base
    out_base.mkdir(parents=True, exist_ok=True)

    files = find_excel_files(base)
    print(f"Found {len(files)} target Excel files (.xlsx/.xlsm) in: {base}")

    results = []
    ok = 0
    ng = 0

    for f in files:
        try:
            r = analyze_one_file(f, out_base)
            results.append(r)
            ok += 1
            print(f"[OK] {f.name} rows={r['rows']} pos_step={r['pos_step_mm_used']}mm")
        except Exception as e:
            ng += 1
            print(f"[NG] {f.name} : {type(e).__name__} - {e}")

    # 全ファイルのまとめ
    if results:
        df_res = pd.DataFrame(results)
        df_res.to_csv(out_base / "batch_results.csv", index=False, encoding="utf-8-sig")

    print("\nDone.")
    print(f"Success: {ok}, Failed: {ng}")
    print(f"Output base: {out_base}")


if __name__ == "__main__":
    main()
