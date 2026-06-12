"""
calibrate.py — Camera calibration using a printed checkerboard pattern.

Usage:
  1. Print a checkerboard (default 9x6 inner corners, ~25mm squares).
  2. Run:  python calibrate.py
       or:  python calibrate.py 9 6 25
       or:  python calibrate.py --config ../experiments/iphone16.yml
     The camera is opened from the resolved config (camera.index), honouring
     defaults.yml -> --config <experiment.yml> -> local.yml -> --set, exactly
     like main.py.  The positional args are the *board* geometry only.
  3. Hold the checkerboard in front of the camera at various angles.
     Press SPACE to capture a frame when the board is detected (green overlay).
     Capture 15-20 images from different positions and tilts.
  4. Press 'c' to run calibration and save the result.
  5. Point camera.calibration_file at the output .npz (in defaults.yml,
     local.yml, or your experiment yaml) so main.py loads it.

Press 'q' / Esc to quit without saving.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from config.loader import resolve_config_dict
from config.schema import AppConfig

OUTPUT_FILE = Path(__file__).parent / "calibration.npz"
MIN_CAPTURES = 12          # below this, views rarely have enough pose variety
RECOMMENDED_CAPTURES = 18  # 15-20 varied tilts/distances is the sweet spot

# Sanity bounds for a *real* overhead/phone camera.  A calibration that lands
# outside these is almost always degenerate (too-uniform capture poses), not a
# genuinely exotic lens — saving it silently is what wrecked tracking before.
FOV_DEG_RANGE = (20.0, 170.0)   # implied horizontal FOV
MAX_ABS_DISTORTION = 2.0        # |k1|,|k2|,|k3| for any sane lens are well under 1
ASPECT_RATIO_RANGE = (0.8, 1.25)  # fy/fx; pixels are ~square
REPROJ_WARN_PX = 1.0            # high-ish but tolerable; above this, recapture


def validate_calibration(
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    reproj_err: float,
    img_shape: tuple[int, int],
) -> list[str]:
    """Return a list of human-readable problems, empty if the calibration looks sane.

    Catches the classic degenerate solve (huge focal length + exploding
    distortion) that ``cv2.calibrateCamera`` produces from near-coplanar /
    low-variety capture frames.
    """
    import math
    problems: list[str] = []
    w, _h = img_shape
    fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
    dc = np.asarray(dist_coeffs).ravel()

    hfov = 2.0 * math.degrees(math.atan((w / 2.0) / fx))
    if not (FOV_DEG_RANGE[0] <= hfov <= FOV_DEG_RANGE[1]):
        problems.append(
            f"implied HFOV {hfov:.1f}° outside {FOV_DEG_RANGE[0]:.0f}-{FOV_DEG_RANGE[1]:.0f}° "
            f"(fx={fx:.0f}) — capture poses were too uniform; vary tilt and distance more"
        )

    aspect = fy / fx if fx else 0.0
    if not (ASPECT_RATIO_RANGE[0] <= aspect <= ASPECT_RATIO_RANGE[1]):
        problems.append(f"fy/fx aspect {aspect:.2f} is implausible (expect ~1.0)")

    for name, idx in (("k1", 0), ("k2", 1), ("k3", 4)):
        if idx < dc.size and abs(dc[idx]) > MAX_ABS_DISTORTION:
            problems.append(
                f"distortion {name}={dc[idx]:.2f} exceeds |{MAX_ABS_DISTORTION}| "
                f"— polynomial is overfitting a bad capture set"
            )

    return problems


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="calibrate.py",
        description="Camera intrinsic calibration from a printed checkerboard.",
    )
    # Board geometry — inner corner counts (one less than the squares per axis).
    p.add_argument("cols", nargs="?", type=int, default=9,
                   help="inner corners across (default 9)")
    p.add_argument("rows", nargs="?", type=int, default=6,
                   help="inner corners down (default 6)")
    p.add_argument("square_mm", nargs="?", type=float, default=25.0,
                   help="printed square size in mm — pass the measured value (default 25.0)")
    # Config surface — same as the loader's, so camera.index can come from an
    # experiment yaml or a one-off override.
    p.add_argument("--config", type=Path, default=None, metavar="PATH",
                   help="experiment yaml (for camera.index etc.)")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE",
                   help="override a config field (repeatable), e.g. --set camera.index=1")
    p.add_argument("--force", action="store_true",
                   help="save even if the calibration fails the sanity check (NOT recommended)")
    return p.parse_args(argv)


def main() -> None:
    args = _parse_args(sys.argv[1:])
    cfg = AppConfig.model_validate(resolve_config_dict(args.config, args.overrides))

    board_cols, board_rows, square_mm = args.cols, args.rows, args.square_mm
    board_size = (board_cols, board_rows)

    # 3D object points for one board pose (z=0 plane)
    objp = np.zeros((board_cols * board_rows, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
    objp *= square_mm

    obj_points: list[np.ndarray] = []  # 3D points per capture
    img_points: list[np.ndarray] = []  # 2D points per capture

    cap = cv2.VideoCapture(cfg.camera.index)
    if not cap.isOpened():
        print(f"[cal] cannot open camera {cfg.camera.index}")
        sys.exit(1)

    print(f"[cal] camera={cfg.camera.index}")
    print(f"[cal] board={board_cols}x{board_rows}  square={square_mm}mm")
    print("[cal] SPACE=capture  C=calibrate  Q/Esc=quit")

    # Resizable display window.  A default WINDOW_AUTOSIZE window opens at the
    # camera's native resolution and can't be resized, so a 4K feed won't fit on
    # screen; WINDOW_NORMAL + an initial resizeWindow fixes both.  Corner
    # detection still runs on the full-resolution frame — only the view scales.
    WINDOW = "Calibration"
    DISPLAY_MAX_PX = 1280   # longer side of the initial window, in pixels
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    window_sized = False

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    img_shape: tuple[int, int] | None = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[cal] lost camera feed")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]  # (w, h)
        fh, fw = frame.shape[:2]

        # Fit the window to the screen on the first frame (full-res feeds like
        # 4K otherwise open larger than the display).  WINDOW_NORMAL keeps it
        # freely resizable afterwards.
        if not window_sized:
            scale = min(1.0, DISPLAY_MAX_PX / max(fw, fh))
            cv2.resizeWindow(WINDOW, max(1, int(fw * scale)), max(1, int(fh * scale)))
            window_sized = True

        found, corners = cv2.findChessboardCorners(gray, board_size, None)

        display = frame.copy()
        if found:
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display, board_size, corners_refined, found)

        # Status bar — scale text with resolution so it stays legible at 4K.
        font_scale = max(0.6, fh / 1080.0 * 0.6)
        thickness = max(2, round(fh / 1080.0 * 2))
        status = f"Captures: {len(obj_points)}  |  "
        status += "Board DETECTED (SPACE to capture)" if found else "No board found"
        cv2.putText(display, status, (10, int(40 * font_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 255, 0) if found else (0, 0, 255), thickness, cv2.LINE_AA)

        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break

        elif key == ord(" ") and found:
            obj_points.append(objp)
            img_points.append(corners_refined)
            print(f"[cal] captured frame {len(obj_points)}")

        elif key == ord("c"):
            if len(obj_points) < MIN_CAPTURES:
                print(f"[cal] need at least {MIN_CAPTURES} captures (have {len(obj_points)})")
                continue

            if len(obj_points) < RECOMMENDED_CAPTURES:
                print(f"[cal] note: {len(obj_points)} frames; {RECOMMENDED_CAPTURES}+ "
                      f"with varied tilt/distance give a more robust solve")

            print(f"[cal] calibrating with {len(obj_points)} frames ...")
            ret_val, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                obj_points, img_points, img_shape, None, None,
            )
            print(f"[cal] reprojection error: {ret_val:.4f} px")
            print(f"[cal] camera_matrix:\n{camera_matrix}")
            print(f"[cal] dist_coeffs: {dist_coeffs.ravel()}")

            # --- Sanity check before we let this overwrite a working config ---
            problems = validate_calibration(camera_matrix, dist_coeffs, ret_val, img_shape)
            if ret_val > REPROJ_WARN_PX:
                print(f"[cal] WARNING: reprojection error {ret_val:.2f}px > {REPROJ_WARN_PX}px")
            if problems:
                print("[cal] ✗ calibration looks DEGENERATE — refusing to save:")
                for prob in problems:
                    print(f"[cal]     - {prob}")
                if not args.force:
                    print("[cal] capture more frames with varied tilt and distance, then press "
                          "'c' again  (or re-run with --force to save anyway)")
                    continue
                print("[cal] --force given: saving the degenerate calibration anyway")
            else:
                print("[cal] ✓ calibration passed sanity check")

            # img_shape is (w, h); store it so camera.py can rescale the matrix
            # if the live feed runs at a different resolution.
            np.savez(str(OUTPUT_FILE),
                     camera_matrix=camera_matrix,
                     dist_coeffs=dist_coeffs,
                     image_size=np.asarray(img_shape, dtype=np.int32))
            print(f"[cal] saved to {OUTPUT_FILE}")
            print(f'[cal] set camera.calibration_file="{OUTPUT_FILE.name}" in your config')
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
