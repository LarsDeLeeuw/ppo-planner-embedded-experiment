"""Grid navigation action server for TurtleBot3.

Receives MoveToGrid goals, executes closed-loop rotate-then-drive
maneuvers, and publishes the robot's estimated grid pose.
"""

import collections
import math
import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle

from geometry_msgs.msg import Twist

from tb3_interfaces.action import MoveToGrid
from tb3_interfaces.msg import ControlLoopStats, GridPose, MoveToGridResult

from .grid_math import (
    angle_to_target,
    distance_between,
    grid_to_world,
    world_to_grid,
    normalize_angle,
)
from .state_tracker import StateTracker
from .motion_controller import ControlParams, compute_rotation_cmd, compute_drive_cmd
from .movement_phases import Phase, MovementStateMachine


class GridNavNode(Node):

    def __init__(self):
        super().__init__('grid_nav_node')

        # -- Declare parameters --
        self.declare_parameter('cell_size', 0.33)
        self.declare_parameter('max_move_distance', 3.0)
        self.declare_parameter('max_linear_speed', 0.15)
        self.declare_parameter('min_linear_speed', 0.05)
        self.declare_parameter('max_angular_speed', 0.8)
        self.declare_parameter('min_angular_speed', 0.15)
        self.declare_parameter('kp_linear', 1.0)
        self.declare_parameter('kp_angular', 2.0)
        self.declare_parameter('kp_heading_correction', 0.5)
        self.declare_parameter('heading_tolerance', 0.05)
        self.declare_parameter('distance_tolerance', 0.02)
        self.declare_parameter('settling_time', 0.3)
        self.declare_parameter('move_timeout', 30.0)
        self.declare_parameter('control_rate', 20.0)
        self.declare_parameter('use_imu_heading', True)
        self.declare_parameter('linear_calibration', 1.0)
        self.declare_parameter('angular_calibration', 1.0)
        self.declare_parameter('initial_grid_x', 0)
        self.declare_parameter('initial_grid_y', 0)
        self.declare_parameter('initial_heading', 0.0)

        # -- Read parameters --
        cell_size = self._p('cell_size')
        self._control_params = ControlParams(
            kp_angular=self._p('kp_angular'),
            kp_linear=self._p('kp_linear'),
            kp_heading_correction=self._p('kp_heading_correction'),
            max_angular_speed=self._p('max_angular_speed'),
            min_angular_speed=self._p('min_angular_speed'),
            max_linear_speed=self._p('max_linear_speed'),
            min_linear_speed=self._p('min_linear_speed'),
            heading_tolerance=self._p('heading_tolerance'),
            distance_tolerance=self._p('distance_tolerance'),
        )

        # -- State tracker (odom/IMU fusion + SetGridPose service) --
        self._state_tracker = StateTracker(
            self,
            use_imu_heading=self._p('use_imu_heading'),
            initial_x=float(self._p('initial_grid_x')),
            initial_y=float(self._p('initial_grid_y')),
            initial_heading=self._p('initial_heading'),
            cell_size=cell_size,
        )

        # -- Movement state machine --
        self._phase_machine = MovementStateMachine(
            heading_tolerance=self._p('heading_tolerance'),
            distance_tolerance=self._p('distance_tolerance'),
        )
        self._settling_time = self._p('settling_time')

        # -- Publishers --
        self._cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 1)
        self._grid_pose_pub = self.create_publisher(GridPose, '~/grid_pose', 10)
        # Instrumentation for the power experiment:
        #   ~/loop_stats   — control-loop period health (detects slip under load)
        #   ~/last_result  — action result on a normal topic (ros2 bag can't
        #                    record action services; this is the ground-truth
        #                    goal-success signal for analysis)
        self._loop_stats_pub = self.create_publisher(ControlLoopStats, '~/loop_stats', 10)
        self._last_result_pub = self.create_publisher(MoveToGridResult, '~/last_result', 10)

        # -- Action server (reentrant so callbacks run in parallel with subs) --
        self._action_cb_group = ReentrantCallbackGroup()
        self._action_server = ActionServer(
            self,
            MoveToGrid,
            'move_to_grid',
            execute_callback=self._execute_cb,
            goal_callback=self._goal_cb,
            cancel_callback=self._cancel_cb,
            callback_group=self._action_cb_group,
        )

        # -- Goal management (preempt-and-replace) --
        self._goal_lock = threading.Lock()
        self._current_goal_handle: ServerGoalHandle | None = None

        # -- Periodic grid_pose publisher (even when idle) --
        pose_period = 1.0 / self._p('control_rate')
        self._pose_timer = self.create_timer(pose_period, self._publish_grid_pose)

        # -- Control-loop period tracking --
        # Ring buffer of (monotonic_time, period_ms) appended each control
        # iteration; a 1 Hz timer summarizes the last `_loop_window_s` seconds.
        self._nominal_period_ms = pose_period * 1000.0
        self._loop_samples: collections.deque = collections.deque(maxlen=200)
        self._loop_window_s = 2.0
        self._last_tick: float | None = None
        self._loop_stats_timer = self.create_timer(1.0, self._publish_loop_stats)

        self.get_logger().info(
            f"GridNavNode started (cell_size={cell_size}m, "
            f"control_rate={self._p('control_rate')}Hz, "
            f"linear_calibration={self._p('linear_calibration')}, "
            f"angular_calibration={self._p('angular_calibration')})"
        )

    def _p(self, name: str):
        """Shorthand to read a parameter value."""
        return self.get_parameter(name).value

    # -- Action callbacks --

    def _goal_cb(self, goal_request) -> GoalResponse:
        """Accept all goals (preempt-and-replace policy)."""
        self.get_logger().info(
            f"Goal received: target=({goal_request.target_x}, {goal_request.target_y})"
        )
        return GoalResponse.ACCEPT

    def _cancel_cb(self, goal_handle) -> CancelResponse:
        """Always accept cancellation requests."""
        self.get_logger().info("Cancel requested")
        return CancelResponse.ACCEPT

    def _execute_cb(self, goal_handle: ServerGoalHandle) -> MoveToGrid.Result:
        """Execute a MoveToGrid goal.

        Thin wrapper around _do_execute that guarantees the robot is stopped
        when the callback exits — on success, abort, preemption, or any
        uncaught exception in the control loop.
        """
        result = None
        try:
            result = self._do_execute(goal_handle)
        except Exception as e:
            self.get_logger().error(f"Move execution failed: {e!r}", exc_info=True)
            result = MoveToGrid.Result()
            result.success = False
            result.message = f"Exception: {e}"
            if goal_handle.is_active:
                goal_handle.abort()
        finally:
            self._publish_stop()
            self._phase_machine.reset()
        # Republish the result on a normal topic so it lands in the bag
        # (ros2 bag does not record action result services).
        self._publish_last_result(result)
        return result

    def _do_execute(self, goal_handle: ServerGoalHandle) -> MoveToGrid.Result:
        """Inner move execution. Wrapped by _execute_cb for safety."""

        # -- Preempt any active goal --
        with self._goal_lock:
            if self._current_goal_handle is not None and self._current_goal_handle.is_active:
                self.get_logger().info("Preempting active goal")
                self._current_goal_handle.abort()
            self._current_goal_handle = goal_handle

        result = MoveToGrid.Result()
        cell_size = self._p('cell_size')
        max_move_dist = self._p('max_move_distance')
        control_period = 1.0 / self._p('control_rate')
        timeout = self._p('move_timeout')
        linear_cal = self._p('linear_calibration')
        angular_cal = self._p('angular_calibration')
        use_imu = self._p('use_imu_heading')
        # Sanity: zero/negative calibration would divide by zero or invert motion.
        if linear_cal <= 0.0:
            linear_cal = 1.0
        if angular_cal <= 0.0:
            angular_cal = 1.0

        # -- Wait for sensor data --
        if not self._state_tracker.has_data:
            self.get_logger().warn("Waiting for odom/IMU data...")
            wait_start = time.monotonic()
            while not self._state_tracker.has_data:
                if time.monotonic() - wait_start > 10.0:
                    self._publish_stop()
                    result.success = False
                    result.message = "Timeout waiting for sensor data"
                    goal_handle.abort()
                    return result
                time.sleep(0.1)

        # -- Compute move --
        cx, cy, cyaw = self._state_tracker.get_continuous_pose()
        target_x = goal_handle.request.target_x
        target_y = goal_handle.request.target_y
        tx, ty = grid_to_world(target_x, target_y, cell_size)

        target_heading = angle_to_target(cx, cy, tx, ty)
        target_distance = distance_between(cx, cy, tx, ty)
        # Compensate odom-yaw slip when not relying on IMU. IMU is a gyro and
        # is unaffected by wheel slip, so the rotation target is left untouched
        # in that mode.
        if not use_imu:
            rot = normalize_angle(target_heading - cyaw)
            target_heading = normalize_angle(cyaw + rot / angular_cal)
        total_rotation = 0.0

        # -- Safety check --
        if target_distance > max_move_dist:
            self._publish_stop()
            result.success = False
            result.message = (
                f"Distance {target_distance:.2f}m exceeds max_move_distance "
                f"{max_move_dist:.2f}m"
            )
            goal_handle.abort()
            self.get_logger().warn(result.message)
            return result

        self.get_logger().info(
            f"Moving: ({cx:.3f},{cy:.3f}) -> ({tx:.3f},{ty:.3f}), "
            f"heading={target_heading:.3f} rad, dist={target_distance:.3f} m"
        )

        # -- Start state machine --
        self._phase_machine.reset()
        self._phase_machine.start_move(target_heading, target_distance)

        start_time = time.monotonic()
        move_start_x, move_start_y = cx, cy    # for total distance calculation
        drive_start_x, drive_start_y = cx, cy  # updated when entering DRIVING
        settling_start = 0.0                    # set when entering SETTLING
        prev_yaw = cyaw
        heading_error = normalize_angle(target_heading - cyaw)
        distance_remaining = target_distance
        self._last_tick = None                  # reset loop-period tracking for this move

        # -- Control loop --
        while rclpy.ok():
            # Record actual control-loop period (sleep + work). Slip above the
            # nominal period under load is the signal we want to capture.
            now_mono = time.monotonic()
            if self._last_tick is not None:
                self._loop_samples.append((now_mono, (now_mono - self._last_tick) * 1000.0))
            self._last_tick = now_mono

            # Check if this goal was preempted
            if not goal_handle.is_active:
                self._publish_stop()
                self._phase_machine.reset()
                result.success = False
                result.message = "Preempted"
                return result

            # Check cancellation
            if goal_handle.is_cancel_requested:
                self._publish_stop()
                self._phase_machine.reset()
                goal_handle.canceled()
                result.success = False
                result.message = "Canceled"
                return result

            # Check timeout
            elapsed = time.monotonic() - start_time
            if elapsed > timeout:
                self._publish_stop()
                self._phase_machine.abort()
                goal_handle.abort()
                result.success = False
                result.message = f"Timeout after {elapsed:.1f}s"
                self.get_logger().warn(result.message)
                break

            # Read current state
            cx, cy, cyaw = self._state_tracker.get_continuous_pose()

            # Track total rotation
            yaw_delta = abs(normalize_angle(cyaw - prev_yaw))
            total_rotation += yaw_delta
            prev_yaw = cyaw

            phase = self._phase_machine.phase

            if phase == Phase.ROTATING:
                heading_error = normalize_angle(target_heading - cyaw)
                distance_remaining = distance_between(cx, cy, tx, ty)

                self._phase_machine.update(heading_error, distance_remaining)

                if self._phase_machine.phase == Phase.SETTLING:
                    self._publish_stop()
                    settling_start = time.monotonic()
                else:
                    cmd = Twist()
                    cmd.angular.z = compute_rotation_cmd(
                        heading_error, self._control_params)
                    self._cmd_vel_pub.publish(cmd)

            elif phase == Phase.SETTLING:
                self._publish_stop()

                if time.monotonic() - settling_start >= self._settling_time:
                    # Recalculate navigation from actual post-rotation pose
                    cx, cy, cyaw = self._state_tracker.get_continuous_pose()
                    target_heading = angle_to_target(cx, cy, tx, ty)
                    target_distance = distance_between(cx, cy, tx, ty)
                    if not use_imu:
                        rot = normalize_angle(target_heading - cyaw)
                        target_heading = normalize_angle(cyaw + rot / angular_cal)
                    heading_error = normalize_angle(target_heading - cyaw)

                    if abs(heading_error) < self._control_params.heading_tolerance:
                        # Heading is good, proceed to driving
                        drive_start_x, drive_start_y = cx, cy
                        distance_remaining = target_distance
                        self._phase_machine.settle_complete()
                        self.get_logger().info(
                            f"Settling complete. Recalculated: "
                            f"heading={target_heading:.3f}, "
                            f"dist={target_distance:.3f}"
                        )
                    else:
                        # Heading drifted too much, re-rotate to new bearing
                        self._phase_machine.back_to_rotating()
                        self.get_logger().info(
                            f"Post-settle heading error {heading_error:.3f} rad "
                            f"exceeds tolerance, re-rotating"
                        )

            elif phase == Phase.DRIVING:
                heading_error = normalize_angle(target_heading - cyaw)
                # Odometry over- or under-counts on slippery surfaces; convert
                # the integrated odom distance into estimated physical distance
                # so the stop criterion matches the user-requested target.
                distance_traveled = (
                    distance_between(drive_start_x, drive_start_y, cx, cy)
                    * linear_cal
                )
                distance_remaining = max(0.0, target_distance - distance_traveled)

                self._phase_machine.update(heading_error, distance_remaining)

                if self._phase_machine.phase != Phase.DONE:
                    cmd = Twist()
                    cmd.linear.x, cmd.angular.z = compute_drive_cmd(
                        distance_remaining, heading_error, self._control_params)
                    self._cmd_vel_pub.publish(cmd)

            elif phase in (Phase.DONE, Phase.ABORTED):
                break

            # Publish feedback
            feedback = MoveToGrid.Feedback()
            feedback.phase = self._phase_machine.phase.name.lower()
            gx, gy, gyaw = self._state_tracker.get_grid_pose()
            feedback.current_pose = GridPose(x=gx, y=gy, heading=gyaw)
            feedback.distance_remaining_m = (
                distance_remaining if phase in (Phase.ROTATING, Phase.SETTLING, Phase.DRIVING)
                else 0.0
            )
            feedback.heading_error_rad = (
                heading_error if phase in (Phase.ROTATING, Phase.SETTLING, Phase.DRIVING)
                else 0.0
            )
            feedback.elapsed_time_s = elapsed
            goal_handle.publish_feedback(feedback)

            time.sleep(control_period)

        # -- Finalize --
        self._publish_stop()
        cx, cy, cyaw = self._state_tracker.get_continuous_pose()
        gx, gy = world_to_grid(cx, cy, cell_size)

        result.final_pose = GridPose(x=gx, y=gy, heading=cyaw)
        result.total_distance_m = distance_between(
            move_start_x, move_start_y, cx, cy)
        result.total_rotation_rad = total_rotation

        if self._phase_machine.phase == Phase.DONE:
            result.success = True
            result.message = f"Reached ({gx}, {gy})"
            goal_handle.succeed()
            self.get_logger().info(result.message)
        else:
            if result.success is False and not result.message:
                result.message = "Move did not complete"
            if goal_handle.is_active:
                goal_handle.abort()

        self._phase_machine.reset()
        return result

    # -- Helpers --

    def _publish_stop(self) -> None:
        """Publish zero velocity to stop the robot."""
        self._cmd_vel_pub.publish(Twist())

    def _publish_grid_pose(self) -> None:
        """Publish current estimated grid pose (runs on timer)."""
        gx, gy, gyaw = self._state_tracker.get_grid_pose()
        msg = GridPose(x=gx, y=gy, heading=gyaw)
        self._grid_pose_pub.publish(msg)

    def _publish_loop_stats(self) -> None:
        """Summarize control-loop period over the recent window (1 Hz timer)."""
        now = time.monotonic()
        samples = [d for (t, d) in list(self._loop_samples) if now - t <= self._loop_window_s]

        msg = ControlLoopStats()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.nominal_period_ms = float(self._nominal_period_ms)
        if samples:
            ordered = sorted(samples)
            n = len(ordered)
            msg.mean_period_ms = float(sum(ordered) / n)
            msg.p99_period_ms = float(ordered[min(n - 1, round(0.99 * (n - 1)))])
            msg.max_period_ms = float(ordered[-1])
            msg.ticks = n
            msg.overruns = sum(1 for d in ordered if d > 2.0 * self._nominal_period_ms)
        else:
            # Idle (no active move in the window): ticks=0 marks "alive but idle".
            msg.mean_period_ms = 0.0
            msg.p99_period_ms = 0.0
            msg.max_period_ms = 0.0
            msg.ticks = 0
            msg.overruns = 0
        self._loop_stats_pub.publish(msg)

    def _publish_last_result(self, result: MoveToGrid.Result) -> None:
        """Republish a completed action result on ~/last_result for the bag."""
        msg = MoveToGridResult()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.run_id_hint = ""
        msg.success = bool(result.success)
        msg.message = result.message
        msg.final_pose = result.final_pose
        msg.total_distance_m = result.total_distance_m
        msg.total_rotation_rad = result.total_rotation_rad
        msg.termination_reason = self._termination_reason(result)
        self._last_result_pub.publish(msg)

    @staticmethod
    def _termination_reason(result: MoveToGrid.Result) -> str:
        if result.success:
            return "goal_reached"
        text = (result.message or "").lower()
        if "timeout" in text:
            return "timeout"
        if "preempt" in text or "cancel" in text:
            return "preempted"
        return "error"


def main(args=None):
    rclpy.init(args=args)
    node = GridNavNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_stop()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
