"""
generate_checkerboard.py — Generate a printable checkerboard for camera calibration.

Usage:
  python generate_checkerboard.py                     # 9x6 inner corners, 25 mm squares, 300 DPI
  python generate_checkerboard.py 9 6 25
  python generate_checkerboard.py 9 6 25 --dpi 600
  python generate_checkerboard.py 9 6 25 --margin 10

Convention: 'cols' and 'rows' are *inner corner* counts — the same numbers
calibrate.py and cv2.findChessboardCorners expect.  A 9x6 inner-corner
board is physically a 10x7 grid of squares.

Output: checkerboard_<cols>x<rows>_<square_mm>mm.png in this directory.

IMPORTANT: print at 100% / actual size — never "fit to page".  After
printing, measure a square with a ruler.  If it isn't exactly the
requested mm, pass the *measured* value as the 3rd arg to calibrate.py.
"""

import sys
from pathlib import Path

import cv2
import numpy as np


def generate(
    inner_cols: int,
    inner_rows: int,
    square_mm: float,
    dpi: int = 300,
    margin_mm: float = 10.0,
) -> str:
    n_sq_x = inner_cols + 1
    n_sq_y = inner_rows + 1
    px_per_mm = dpi / 25.4
    sq_px = int(round(square_mm * px_per_mm))
    margin_px = int(round(margin_mm * px_per_mm))

    board_w = n_sq_x * sq_px
    board_h = n_sq_y * sq_px

    img = np.full((board_h, board_w), 255, dtype=np.uint8)
    for j in range(n_sq_y):
        for i in range(n_sq_x):
            if (i + j) & 1:
                y0, x0 = j * sq_px, i * sq_px
                img[y0:y0 + sq_px, x0:x0 + sq_px] = 0

    bordered = cv2.copyMakeBorder(
        img, margin_px, margin_px, margin_px, margin_px,
        cv2.BORDER_CONSTANT, value=255,
    )

    strip_h = max(60, int(round(10 * px_per_mm)))
    canvas = np.full(
        (bordered.shape[0] + strip_h, bordered.shape[1]), 255, dtype=np.uint8,
    )
    canvas[:bordered.shape[0], :] = bordered

    label = (
        f"{inner_cols}x{inner_rows} inner corners ({n_sq_x}x{n_sq_y} squares) | "
        f"square = {square_mm} mm | DPI {dpi} | print at 100%, no scaling"
    )
    font_scale = max(0.7, dpi / 400.0)
    thickness = max(1, int(round(dpi / 300.0)))
    cv2.putText(
        canvas, label,
        (margin_px, bordered.shape[0] + int(strip_h * 0.7)),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, 0, thickness, cv2.LINE_AA,
    )

    filename = f"checkerboard_{inner_cols}x{inner_rows}_{int(square_mm)}mm.png"
    out_path = Path(__file__).parent / filename
    cv2.imwrite(str(out_path), canvas)
    return str(out_path)


def main() -> None:
    args = sys.argv[1:]

    dpi = 300
    margin_mm = 10.0

    if "--dpi" in args:
        idx = args.index("--dpi")
        dpi = int(args[idx + 1])
        args = args[:idx] + args[idx + 2:]
    if "--margin" in args:
        idx = args.index("--margin")
        margin_mm = float(args[idx + 1])
        args = args[:idx] + args[idx + 2:]

    inner_cols = int(args[0]) if len(args) > 0 else 9
    inner_rows = int(args[1]) if len(args) > 1 else 6
    square_mm = float(args[2]) if len(args) > 2 else 25.0

    path = generate(inner_cols, inner_rows, square_mm, dpi=dpi, margin_mm=margin_mm)

    n_sq_x = inner_cols + 1
    n_sq_y = inner_rows + 1
    total_w_mm = n_sq_x * square_mm + 2 * margin_mm
    total_h_mm = n_sq_y * square_mm + 2 * margin_mm
    print(f"[generate] saved {path}")
    print(
        f"[generate] {inner_cols}x{inner_rows} inner corners "
        f"({n_sq_x}x{n_sq_y} squares)"
    )
    print(
        f"[generate] square = {square_mm} mm  "
        f"total = {total_w_mm:.0f} x {total_h_mm:.0f} mm "
        f"(fits A4 if total <= ~190 x 277)"
    )
    print("[generate] print at 100% / no fit-to-page; verify with a ruler afterwards")


if __name__ == "__main__":
    main()
