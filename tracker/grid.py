"""
grid.py — Grid world from real 3D marker poses.

Uses estimatePoseSingleMarkers on the 4 corner markers to get their real 3D
positions in camera space (cm).  Builds a floor plane from those positions.

The robot marker's 3D position is projected orthogonally onto the floor plane
along the floor normal, then converted to grid coordinates.

The 2D homography (H_inv) is kept purely for drawing grid lines on the image.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from math import atan2, degrees, floor

import cv2
import numpy as np

# Set QRT_DEBUG_POSE=1 to log the robot marker's two PnP solutions per frame.
_DEBUG_POSE = bool(os.environ.get("QRT_DEBUG_POSE"))
_dbg_frame = 0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GridState:
    """Locked grid geometry."""
    # 2D — for overlay drawing only
    H_inv: np.ndarray       # (3,3) grid → image
    cols: int
    rows: int
    src_pts: np.ndarray     # (4,2) marker centers in image space (TL,TR,BR,BL)

    # 3D floor frame (camera-space, cm)
    floor_origin: np.ndarray  # (3,) TL marker position
    floor_R: np.ndarray       # (3,3) columns = [x_hat, y_hat, z_hat]
    width_cm: float           # physical TL→TR distance
    height_cm: float          # physical TL→BL distance (along y_hat)


@dataclass
class RobotPose:
    """Robot's position for a single frame."""
    grid_x: float
    grid_y: float
    cell_col: int
    cell_row: int
    heading_deg: float
    image_center: np.ndarray  # (2,) pixel coords for overlay anchoring
    in_bounds: bool
    height_cm: float          # height above floor (cm)


# ---------------------------------------------------------------------------
# Helpers for overlay drawing (2D only)
# ---------------------------------------------------------------------------

def marker_center(corners: np.ndarray) -> np.ndarray:
    """Mean of 4 marker corners → (2,) center point."""
    return corners.reshape(4, 2).mean(axis=0)


def grid_points_to_image(pts: np.ndarray, H_inv: np.ndarray) -> np.ndarray:
    """Batch transform (N,2) grid points → (N,2) image points."""
    pts_f = pts.astype(np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts_f, H_inv)
    return out.reshape(-1, 2)


def grid_to_image(pt: np.ndarray, H_inv: np.ndarray) -> np.ndarray:
    """Single grid point → image point."""
    p = np.array([pt[0], pt[1], 1.0], dtype=np.float64)
    q = H_inv @ p
    return (q[:2] / q[2]).astype(np.float64)


def image_to_grid(pt: np.ndarray, H_inv: np.ndarray) -> np.ndarray:
    """Single image point → grid point (inverse of grid_to_image)."""
    H_forward = np.linalg.inv(H_inv)
    p = np.array([pt[0], pt[1], 1.0], dtype=np.float64)
    q = H_forward @ p
    return (q[:2] / q[2]).astype(np.float64)


# ---------------------------------------------------------------------------
# Grid calibration (3D)
# ---------------------------------------------------------------------------

def compute_grid(
    corner_markers: dict[int, np.ndarray],
    corner_ids: list[int],
    cols: int,
    rows: int,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    corner_marker_size_cm: float,
) -> GridState | None:
    """
    Build a GridState from detected corner markers.

    Uses estimatePoseSingleMarkers to get real 3D positions (cm) of the
    4 corner markers, then constructs an orthogonal floor coordinate frame.

    Returns None if any corner is missing.
    """
    for cid in corner_ids:
        if cid not in corner_markers:
            return None

    # ---- 2D homography (for drawing grid lines on the image) ----
    src_pts = np.array(
        [marker_center(corner_markers[cid]) for cid in corner_ids],
        dtype=np.float32,
    )
    dst_pts = np.array(
        [[0, 0], [cols, 0], [cols, rows], [0, rows]],
        dtype=np.float32,
    )
    H_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)

    # ---- 3D pose estimation of corner markers ----
    corners_list = [corner_markers[cid] for cid in corner_ids]
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners_list, corner_marker_size_cm, camera_matrix, dist_coeffs,
    )
    # tvecs[i] shape (1,3) — squeeze to (3,)
    p_tl = tvecs[0].ravel()
    p_tr = tvecs[1].ravel()
    p_bl = tvecs[3].ravel()  # index 3 = BL (TL, TR, BR, BL order)

    # ---- Build orthogonal floor frame ----
    x_vec = p_tr - p_tl                                       # TL → TR
    y_vec = p_bl - p_tl                                       # TL → BL

    x_hat = x_vec / np.linalg.norm(x_vec)
    y_proj = y_vec - np.dot(y_vec, x_hat) * x_hat             # orthogonalise
    y_hat = y_proj / np.linalg.norm(y_proj)
    z_hat = np.cross(x_hat, y_hat)                             # floor normal (up)

    floor_R = np.column_stack([x_hat, y_hat, z_hat])           # floor→camera

    width_cm = float(np.linalg.norm(x_vec))
    height_cm = float(np.dot(y_vec, y_hat))

    if width_cm < 1e-3 or height_cm < 1e-3:
        return None

    return GridState(
        H_inv=H_inv, cols=cols, rows=rows,
        src_pts=src_pts,
        floor_origin=p_tl, floor_R=floor_R,
        width_cm=width_cm, height_cm=height_cm,
    )


# ---------------------------------------------------------------------------
# Robot localization (3D → grid)
# ---------------------------------------------------------------------------

def estimate_marker_pose_on_floor(
    corners: np.ndarray,
    marker_size_cm: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    floor_normal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Estimate a planar marker's pose, resolving the IPPE flip ambiguity.

    A flat square has *two* PnP solutions that reproject almost equally well;
    they share the in-plane rotation but flip the out-of-plane tilt, which
    swings the depth.  ``cv2.aruco.estimatePoseSingleMarkers`` returns only one
    and picks arbitrarily per frame, so on a small marker viewed obliquely the
    pose toggles between the two — jittering the projected floor position and
    height frame to frame.

    Because the robot marker lies flat — parallel to the grid floor — the
    *correct* solution is the one whose marker normal is most aligned with the
    locked grid's floor normal.  We compute both ``SOLVEPNP_IPPE_SQUARE``
    solutions and keep that one.

    Returns ``(rvec (3,), tvec (3,), min_reproj_px)`` — pose in camera space
    (cm) plus the best (minimum) reprojection error across the two solutions,
    which is a reliable detection-quality signal: a distorted marker quad on a
    compressed stream fits a square poorly (high error) and yields a bogus
    depth, so callers can gate on it.
    """
    s = marker_size_cm / 2.0
    # Object points in the marker's own frame, in the corner order ArUco emits
    # (TL, TR, BR, BL) — the ordering SOLVEPNP_IPPE_SQUARE expects.
    objp = np.array(
        [[-s,  s, 0.0],
         [ s,  s, 0.0],
         [ s, -s, 0.0],
         [-s, -s, 0.0]],
        dtype=np.float32,
    )
    img_pts = corners.reshape(4, 2).astype(np.float32)

    n_sol, rvecs, tvecs, errs = cv2.solvePnPGeneric(
        objp, img_pts, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if n_sol == 0:
        raise ValueError("solvePnP returned no pose for the robot marker")

    # Pick the solution whose marker normal (R[:,2]) is most parallel to the
    # floor normal.  abs() absorbs the sign of floor_normal (which way is "up").
    aligns: list[float] = []
    best_i, best_align = 0, -1.0
    for i in range(n_sol):
        R, _ = cv2.Rodrigues(rvecs[i])
        align = abs(float(np.dot(R[:, 2], floor_normal)))
        aligns.append(align)
        if align > best_align:
            best_align, best_i = align, i

    min_reproj = min(float(np.asarray(errs[i]).ravel()[0]) for i in range(n_sol))

    if _DEBUG_POSE:
        global _dbg_frame
        _dbg_frame += 1
        if _dbg_frame % 6 == 0:   # throttle: ~every 6th detection
            parts = " | ".join(
                f"sol{i} align={aligns[i]:.3f} z={float(tvecs[i].ravel()[2]):7.1f}cm "
                f"reproj={float(np.asarray(errs[i]).ravel()[0]):.3f}"
                for i in range(n_sol)
            )
            print(f"[pose] n={n_sol} picked=sol{best_i}  {parts}")

    return rvecs[best_i].ravel(), tvecs[best_i].ravel(), min_reproj


def localize_robot(
    robot_tvec: np.ndarray,
    robot_rvec: np.ndarray,
    grid: GridState,
    robot_corners_2d: np.ndarray,
) -> RobotPose:
    """
    Project robot's 3D position onto the floor plane and convert to grid coords.

    robot_tvec:      (3,) robot centre in camera space (cm)
    robot_rvec:      (3,) Rodrigues rotation vector
    robot_corners_2d: (1,4,2) image corners — only for overlay anchor
    """
    origin = grid.floor_origin
    z_hat = grid.floor_R[:, 2]

    # ---- Orthogonal projection onto floor ----
    height_cm = float(np.dot(robot_tvec - origin, z_hat))
    robot_on_floor = robot_tvec - height_cm * z_hat

    # ---- Convert to floor-local coordinates (cm) ----
    p_local = grid.floor_R.T @ (robot_on_floor - origin)

    # ---- Convert cm → grid units ----
    gx = float(p_local[0]) / grid.width_cm * grid.cols
    gy = float(p_local[1]) / grid.height_cm * grid.rows

    in_bounds = 0 <= gx <= grid.cols and 0 <= gy <= grid.rows
    col = max(0, min(grid.cols - 1, int(floor(gx))))
    row = max(0, min(grid.rows - 1, int(floor(gy))))

    # ---- Heading: robot forward direction projected onto floor ----
    heading = _compute_heading(robot_rvec, grid.floor_R, z_hat)

    img_center = marker_center(robot_corners_2d)

    return RobotPose(
        grid_x=gx, grid_y=gy,
        cell_col=col, cell_row=row,
        heading_deg=heading,
        image_center=img_center,
        in_bounds=in_bounds,
        height_cm=height_cm,
    )


def localize_robot_stable(
    robot_tvec: np.ndarray,
    robot_rvec: np.ndarray,
    robot_corners_2d: np.ndarray,
    grid: GridState,
    camera_matrix: np.ndarray,
    marker_height_cm: float = 0.0,
) -> RobotPose:
    """Localize the robot with a position that does NOT depend on marker depth.

    The PnP depth (and thus :func:`localize_robot`'s position) is derived from
    the marker's apparent *size*, which jitters badly for a small marker on a
    compressed stream — and the jitter turns into floor-position noise scaled by
    ``sin(theta)`` of the viewing ray.  The marker's *center*, by contrast, is
    robust (the 4-corner average cancels per-corner jitter).

    So position comes from mapping the marker center through the locked grid
    homography (camera ray ∩ floor plane), independent of depth.  Heading and
    height still come from the 3D pose (heading is EMA-smoothed downstream;
    height is the marker's ~constant elevation, shown as ALT).

    ``marker_height_cm`` parallax-corrects the ray∩floor point: an elevated
    marker's ray pierces the floor *outside* its true footprint.  With camera
    height ``H`` above the floor, nadir ``N`` and ray∩floor point ``P``, the
    footprint is ``F = N + (P - N) * (H - h) / H``.  Leave at 0 to skip the
    correction (negligible near the nadir; a small outward bias toward edges).
    """
    origin = grid.floor_origin
    z_hat = grid.floor_R[:, 2]

    height_cm = float(np.dot(robot_tvec - origin, z_hat))
    heading = _compute_heading(robot_rvec, grid.floor_R, z_hat)

    # ---- Depth-independent position: marker center → floor via homography ----
    center = marker_center(robot_corners_2d)
    p_floor = image_to_grid(center, grid.H_inv)          # ray ∩ floor (grid units)

    cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])
    cam_height = abs(float(np.dot(origin, z_hat)))        # camera height above floor (cm)
    if marker_height_cm and cam_height > 1e-6:
        nadir = image_to_grid(np.array([cx, cy], dtype=np.float64), grid.H_inv)
        gxy = nadir + (p_floor - nadir) * ((cam_height - marker_height_cm) / cam_height)
    else:
        gxy = p_floor
    gx, gy = float(gxy[0]), float(gxy[1])

    in_bounds = 0 <= gx <= grid.cols and 0 <= gy <= grid.rows
    col = max(0, min(grid.cols - 1, int(floor(gx))))
    row = max(0, min(grid.rows - 1, int(floor(gy))))

    return RobotPose(
        grid_x=gx, grid_y=gy,
        cell_col=col, cell_row=row,
        heading_deg=heading,
        image_center=center,
        in_bounds=in_bounds,
        height_cm=height_cm,
    )


def _compute_heading(
    rvec: np.ndarray,
    floor_R: np.ndarray,
    z_hat: np.ndarray,
) -> float:
    """
    Heading in grid space (degrees). 0° = grid +X (rightward), 90° = grid +Y.

    Uses the marker's local X-axis (top edge) as forward direction,
    projected onto the floor plane.
    """
    R_robot, _ = cv2.Rodrigues(rvec)
    forward_cam = R_robot[:, 0]  # marker X-axis in camera space

    # Remove vertical component (project onto floor)
    forward_flat = forward_cam - np.dot(forward_cam, z_hat) * z_hat
    norm = np.linalg.norm(forward_flat)
    if norm < 1e-10:
        return 0.0
    forward_flat /= norm

    # Express in floor frame
    f_local = floor_R.T @ forward_flat
    return degrees(atan2(f_local[1], f_local[0]))
