"""
coord_transform.py — Coordinate transforms between tracker and ROS2 conventions.

Tracker (camera/grid):  origin top-left, X right, Y down, heading CW in degrees.
ROS2 (grid_nav_node):   origin bottom-left, X right (East), Y up (North),
                         heading CCW in radians, 0 = East.

Every cross-frame conversion used by the planner / bridge pipeline lives here
so callers never hand-roll a np.flipud or an ad-hoc `rows - 1 - row` flip.
"""

from math import degrees, radians

import numpy as np


def tracker_to_ros2(
    grid_x: float,
    grid_y: float,
    heading_deg: float,
    grid_rows: int,
) -> tuple[float, float, float]:
    """Convert tracker coordinates to ROS2 convention.

    Returns (ros2_x, ros2_y, ros2_heading_rad).
    """
    ros2_x = grid_x
    ros2_y = grid_rows - grid_y
    ros2_heading_rad = -radians(heading_deg)
    return ros2_x, ros2_y, ros2_heading_rad


def ros2_to_tracker(
    ros2_x: float,
    ros2_y: float,
    ros2_heading_rad: float,
    grid_rows: int,
) -> tuple[float, float, float]:
    """Convert ROS2 coordinates back to tracker convention.

    Returns (grid_x, grid_y, heading_deg).
    """
    grid_x = ros2_x
    grid_y = grid_rows - ros2_y
    heading_deg = -degrees(ros2_heading_rad)
    return grid_x, grid_y, heading_deg


# ---------------------------------------------------------------------------
# Map / cell / direction conversions for the planner pipeline
# ---------------------------------------------------------------------------

def tracker_map_to_bridge(arr: np.ndarray) -> np.ndarray:
    """Convert a (rows, cols) tracker-frame map to a bridge-frame [x][y] map.

    Tracker arrays are indexed ``[row][col]`` with row 0 at the top.  The
    bridge / planner indexes maps as ``[x][y]`` with X=East and Y=North
    (origin bottom-left).  The transform is therefore a Y-flip followed by
    a row/col->x/y transpose so that ``out[x][y]`` reads the cell whose
    bridge coordinates are (x, y).  Returns a contiguous copy.
    """
    return np.ascontiguousarray(np.flipud(arr).T)


def tracker_cell_to_bridge(row: int, col: int, grid_rows: int) -> tuple[int, int]:
    """Convert a tracker (row, col) cell index to bridge-frame (x, y)."""
    return int(col), int(grid_rows - 1 - row)


def bridge_direction_to_tracker_delta(dx: int, dy: int) -> tuple[int, int]:
    """Convert a bridge-frame direction step (dx, dy) to tracker (drow, dcol).

    Bridge +Y is North (bottom-up); tracker +row is Down (top-down).  So a
    bridge step of dy=+1 (north) corresponds to drow=-1 (one row up in the
    tracker array).  dx maps straight through to dcol.
    """
    return -int(dy), int(dx)
