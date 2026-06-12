"""
camera.py — Camera intrinsic matrix.

Either loads a calibration file produced by cv2.calibrateCamera()  (more
accurate) or estimates reasonable defaults from the frame resolution (good
enough for a PoC without a checkerboard calibration).
"""

import numpy as np
import cv2


def get_camera_matrix(
    frame_shape: tuple[int, int, int],
    calibration_file: str | None,
    hfov_deg: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (camera_matrix 3x3, dist_coeffs 4x1).

    Three quality tiers:
      1. calibration_file → load from .npz (best)
      2. hfov_deg → compute from horizontal field-of-view in degrees (good)
      3. Neither → rough estimate fx ≈ image_width (fallback)
    """
    h, w = frame_shape[:2]
    cx, cy = w / 2.0, h / 2.0

    if calibration_file:
        import math
        data = np.load(calibration_file)
        camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
        dist_coeffs = data["dist_coeffs"]
        print(f"[camera] loaded calibration from {calibration_file}")

        # Rescale intrinsics if the live feed differs from the calibration
        # resolution.  fx/fy/cx/cy scale linearly with pixels; distortion is
        # resolution-independent.  Without this, a 4K calibration used on a
        # 720p stream (or vice versa) silently mis-scales every pose.
        if "image_size" in data:
            cal_w, cal_h = (int(v) for v in np.asarray(data["image_size"]).ravel()[:2])
            if (cal_w, cal_h) != (w, h):
                sx, sy = w / cal_w, h / cal_h
                camera_matrix[0, :] *= sx   # fx, skew, cx
                camera_matrix[1, :] *= sy   # fy, cy
                print(f"[camera] rescaled intrinsics {cal_w}x{cal_h} -> {w}x{h} "
                      f"(sx={sx:.3f}, sy={sy:.3f})")
        else:
            print("[camera] WARNING: calibration has no image_size; assuming it "
                  f"matches the live feed ({w}x{h}). Re-run calibrate.py to embed it.")

        # Loud guard against a degenerate calibration (e.g. the old 9°-FOV file).
        fx = float(camera_matrix[0, 0])
        hfov = 2.0 * math.degrees(math.atan((w / 2.0) / fx)) if fx else 0.0
        k = np.asarray(dist_coeffs).ravel()
        if not (15.0 <= hfov <= 175.0) or (k.size and np.max(np.abs(k)) > 2.0):
            print(f"[camera] WARNING: calibration looks DEGENERATE "
                  f"(implied HFOV={hfov:.1f}°, max|dist|={np.max(np.abs(k)):.1f}). "
                  f"Tracking will be unreliable — recalibrate or set calibration_file: null.")
        return camera_matrix, dist_coeffs

    if hfov_deg is not None:
        import math
        fx = fy = (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        print(f"[camera] intrinsics from HFOV={hfov_deg}° (fx={fx:.0f}, cx={cx:.0f}, cy={cy:.0f})")
    else:
        fx = fy = float(w)
        print(f"[camera] rough estimated intrinsics (fx={fx:.0f}, cx={cx:.0f}, cy={cy:.0f})")

    camera_matrix = np.array(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    return camera_matrix, dist_coeffs
