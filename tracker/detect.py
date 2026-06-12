"""
detect.py — ArUco marker detection from a single OpenCV frame.

Uses OpenCV's cv2.aruco module which is purpose-built for planar marker
pose estimation.  Much more robust than QR/pyzbar at steep angles and
small marker sizes.

Returns a list of DetectedMarker (one per visible marker), or an empty list.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import cv2


# Dictionary lookup — maps config string to OpenCV constant
_DICT_MAP = {
    "DICT_4X4_50":   cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100":  cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250":  cv2.aruco.DICT_4X4_250,
    "DICT_5X5_50":   cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100":  cv2.aruco.DICT_5X5_100,
    "DICT_6X6_50":   cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100":  cv2.aruco.DICT_6X6_100,
}


@dataclass
class DetectedMarker:
    # 4 corner points, shape (1, 4, 2) as returned by cv2.aruco
    corners: np.ndarray
    # Integer ID encoded in the ArUco marker
    marker_id: int


def create_detector(aruco_dict_name: str) -> cv2.aruco.ArucoDetector:
    """Build an ArucoDetector from a dictionary name string (e.g. "DICT_4X4_50")."""
    dict_id = _DICT_MAP.get(aruco_dict_name)
    if dict_id is None:
        raise ValueError(
            f"Unknown ArUco dictionary '{aruco_dict_name}'. "
            f"Valid: {list(_DICT_MAP.keys())}"
        )
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    params = cv2.aruco.DetectorParameters()
    # Sub-pixel corner refinement.  Without it (the default CORNER_REFINE_NONE)
    # corners snap coarsely to the detection grid; on a small marker over a
    # compressed stream the detected quad scale jitters ~half a module frame to
    # frame, which makes PnP depth (and thus position/ALT) toggle.  Refinement
    # locks corners onto the true black/white edge with sub-pixel accuracy.
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 30
    params.cornerRefinementMinAccuracy = 0.01
    return cv2.aruco.ArucoDetector(aruco_dict, params)


def detect_markers(frame: np.ndarray, detector: cv2.aruco.ArucoDetector) -> list[DetectedMarker]:
    """
    Detect all ArUco markers in *frame*.

    Pure producer — does not mutate the frame. Use `overlay.draw_markers`
    for rendering.
    """
    corners, ids, _rejected = detector.detectMarkers(frame)

    if ids is None:
        return []

    results: list[DetectedMarker] = []
    for i, marker_id in enumerate(ids.ravel()):
        results.append(
            DetectedMarker(
                corners=corners[i],      # shape (1, 4, 2)
                marker_id=int(marker_id),
            )
        )

    return results
