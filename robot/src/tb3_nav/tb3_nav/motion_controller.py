"""Proportional controllers for heading and distance.

No ROS dependencies -- pure computation, unit-testable.

Handles the TurtleBot3 Burger's motor deadzone: commands are either
zero (within tolerance) or at least min_speed to avoid stalling.
"""

import math
from dataclasses import dataclass


@dataclass
class ControlParams:
    """Parameters for the motion controllers."""
    kp_angular: float = 2.0
    kp_linear: float = 1.0
    kp_heading_correction: float = 0.5
    max_angular_speed: float = 0.8
    min_angular_speed: float = 0.15
    max_linear_speed: float = 0.15
    min_linear_speed: float = 0.05
    heading_tolerance: float = 0.05
    distance_tolerance: float = 0.02


def compute_rotation_cmd(heading_error: float, params: ControlParams) -> float:
    """Compute angular velocity command for in-place rotation.

    Args:
        heading_error: signed heading error in radians (target - current).
        params: controller parameters.

    Returns:
        Angular velocity (rad/s). Positive = CCW.
    """
    if abs(heading_error) < params.heading_tolerance:
        return 0.0

    raw = params.kp_angular * heading_error
    speed = max(abs(raw), params.min_angular_speed)
    speed = min(speed, params.max_angular_speed)
    return math.copysign(speed, heading_error)


def compute_drive_cmd(distance_remaining: float, heading_error: float,
                      params: ControlParams) -> tuple[float, float]:
    """Compute linear + angular velocity for straight-line driving.

    Args:
        distance_remaining: distance to target in meters (positive).
        heading_error: signed heading deviation during drive (radians).
        params: controller parameters.

    Returns:
        (linear_x, angular_z) velocity commands.
    """
    if distance_remaining < params.distance_tolerance:
        return (0.0, 0.0)

    # Linear speed proportional to remaining distance
    raw_linear = params.kp_linear * distance_remaining
    linear_x = max(raw_linear, params.min_linear_speed)
    linear_x = min(linear_x, params.max_linear_speed)

    # Small heading correction to stay on course
    angular_z = params.kp_heading_correction * heading_error
    angular_z = max(-params.max_angular_speed, min(angular_z, params.max_angular_speed))

    return (linear_x, angular_z)
