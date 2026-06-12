"""
perspective.py — Perspective warp operations for the grid capture pipeline.

Transforms the perspective-distorted grid region in a camera frame into a
square image using the 4 corner marker positions.
"""

from __future__ import annotations

import cv2
import numpy as np


def warp_to_square(
    frame: np.ndarray,
    src_pts: np.ndarray,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Perspective-warp the grid region to a square image.

    Parameters
    ----------
    frame : np.ndarray
        Source camera frame (BGR).
    src_pts : np.ndarray
        (4,2) float32 — marker centers in image space, ordered TL, TR, BR, BL.
    size : int
        Edge length of the output square in pixels.

    Returns
    -------
    warped : np.ndarray
        (size, size, 3) BGR image of the perspective-corrected grid region.
    M : np.ndarray
        (3,3) perspective matrix used for the warp (caller can invert for
        mapping back to image space).
    """
    dst_pts = np.array(
        [[0, 0], [size, 0], [size, size], [0, size]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src_pts.astype(np.float32), dst_pts)
    warped = cv2.warpPerspective(frame, M, (size, size))
    return warped, M
