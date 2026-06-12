"""Sensor fusion and position tracking for grid navigation.

Subscribes to odometry and IMU topics, maintains correction offsets
for hybrid position tracking, and hosts the SetGridPose service for
external localization corrections.
"""

import threading

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from tb3_interfaces.srv import SetGridPose

from .grid_math import (
    yaw_from_quaternion,
    normalize_angle,
    grid_to_world,
    world_to_grid,
)


class StateTracker:
    """Tracks the robot's position and heading via odom + IMU fusion.

    Not a ROS2 node itself -- attaches subscriptions and a service
    to the parent node passed at construction.

    Thread-safe: subscription callbacks may fire from different threads
    than the action execute callback that reads the pose.
    """

    def __init__(self, node: Node, *,
                 use_imu_heading: bool = True,
                 initial_x: float = 0.0,
                 initial_y: float = 0.0,
                 initial_heading: float = 0.0,
                 cell_size: float = 0.33):
        self._node = node
        self._use_imu_heading = use_imu_heading
        self._cell_size = cell_size
        self._lock = threading.Lock()

        # Raw sensor values
        self._odom_x = 0.0
        self._odom_y = 0.0
        self._odom_yaw = 0.0
        self._imu_yaw = 0.0
        self._odom_received = False
        self._imu_received = False

        # Correction offsets (additive)
        # Initialized so that get_continuous_pose() returns (initial_x, initial_y, initial_heading)
        # before any odom/imu data arrives -- offsets are recalculated on first sensor callback
        initial_world_x, initial_world_y = grid_to_world(
            initial_x, initial_y, cell_size)
        self._offset_x = initial_world_x
        self._offset_y = initial_world_y
        self._offset_yaw = normalize_angle(initial_heading)

        # Subscriptions (relative topic names for namespace remapping)
        sensor_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._odom_sub = node.create_subscription(
            Odometry, 'odom', self._odom_cb, sensor_qos)
        self._imu_sub = node.create_subscription(
            Imu, 'imu', self._imu_cb, sensor_qos)

        # SetGridPose service for external localization corrections
        self._set_pose_srv = node.create_service(
            SetGridPose, '~/set_grid_pose', self._set_grid_pose_cb)

    @property
    def has_data(self) -> bool:
        """True once at least one odom and one IMU message have been received."""
        with self._lock:
            if self._use_imu_heading:
                return self._odom_received and self._imu_received
            return self._odom_received

    def get_continuous_pose(self) -> tuple[float, float, float]:
        """Get the corrected world-frame pose (x, y, yaw).

        Returns:
            (x_meters, y_meters, yaw_radians) with correction offsets applied.
        """
        with self._lock:
            heading = self._imu_yaw if self._use_imu_heading else self._odom_yaw
            return (
                self._odom_x + self._offset_x,
                self._odom_y + self._offset_y,
                normalize_angle(heading + self._offset_yaw),
            )

    def get_grid_pose(self) -> tuple[float, float, float]:
        """Get the current pose as grid coordinates + heading.

        Returns:
            (grid_x, grid_y, yaw_radians).
        """
        wx, wy, yaw = self.get_continuous_pose()
        gx, gy = world_to_grid(wx, wy, self._cell_size)
        return (gx, gy, yaw)

    def apply_override(self, grid_x: float, grid_y: float,
                       heading: float) -> None:
        """Reset correction offsets so the pose matches the given grid position.

        Called when an external localization system provides a corrected pose.
        Does not reset odometry -- only adjusts the additive offsets.
        """
        world_x, world_y = grid_to_world(grid_x, grid_y, self._cell_size)
        with self._lock:
            raw_heading = self._imu_yaw if self._use_imu_heading else self._odom_yaw
            self._offset_x = world_x - self._odom_x
            self._offset_y = world_y - self._odom_y
            self._offset_yaw = normalize_angle(heading - raw_heading)

    def update_cell_size(self, cell_size: float) -> None:
        """Update the cell size (e.g. if the parameter changes at runtime)."""
        self._cell_size = cell_size

    # -- ROS callbacks (run from subscription executor threads) --

    def _odom_cb(self, msg: Odometry) -> None:
        with self._lock:
            self._odom_x = msg.pose.pose.position.x
            self._odom_y = msg.pose.pose.position.y
            q = msg.pose.pose.orientation
            self._odom_yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
            if not self._odom_received:
                self._odom_received = True

    def _imu_cb(self, msg: Imu) -> None:
        with self._lock:
            q = msg.orientation
            self._imu_yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
            if not self._imu_received:
                self._imu_received = True

    def _set_grid_pose_cb(self, request, response) -> SetGridPose.Response:
        pose = request.pose
        self._node.get_logger().info(
            f"SetGridPose: correcting to grid ({pose.x}, {pose.y}), "
            f"heading={pose.heading:.3f} rad"
        )
        self.apply_override(pose.x, pose.y, pose.heading)
        response.success = True
        response.message = f"Pose set to ({pose.x}, {pose.y}, {pose.heading:.3f})"
        return response
