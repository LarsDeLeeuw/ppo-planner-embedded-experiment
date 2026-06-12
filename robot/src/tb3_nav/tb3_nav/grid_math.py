"""Pure math functions for grid-based navigation.

No ROS dependencies -- all functions are stateless and unit-testable.
"""

import math


def normalize_angle(angle: float) -> float:
    """Wrap an angle to the range [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def angle_to_target(from_x: float, from_y: float,
                    to_x: float, to_y: float) -> float:
    """Compute heading angle from one point to another (radians).

    Uses world coordinates (meters). Returns angle in [-pi, pi],
    ROS convention (0 = +X, CCW positive).
    """
    return math.atan2(to_y - from_y, to_x - from_x)


def distance_between(from_x: float, from_y: float,
                     to_x: float, to_y: float) -> float:
    """Euclidean distance between two points in world coordinates (meters)."""
    return math.hypot(to_x - from_x, to_y - from_y)


def grid_to_world(grid_x: float, grid_y: float, cell_size: float) -> tuple[float, float]:
    """Convert grid coordinates to world coordinates (meters).

    Grid origin (0, 0) maps to world origin (0.0, 0.0).
    Each grid step is cell_size meters.
    """
    return (grid_x * cell_size, grid_y * cell_size)


def world_to_grid(world_x: float, world_y: float, cell_size: float) -> tuple[float, float]:
    """Convert world coordinates to grid coordinates.

    Returns continuous grid coordinates (sub-cell precision).
    """
    return (world_x / cell_size, world_y / cell_size)


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Extract yaw (rotation around Z axis) from a quaternion.

    Returns angle in [-pi, pi].
    """
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)
