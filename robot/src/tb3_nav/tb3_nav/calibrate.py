"""Surface-calibration tool for grid_nav_node.

Drives a known sequence of test maneuvers, prompts the user for
tape-measured ground truth, and prints calibration values to paste
into the bringup experiment.yaml. While each maneuver runs, raw IMU
yaw, raw odom (x, y, yaw) and the node's world-frame estimate are
snapshotted at every phase transition so rotation-coupled drift can
be diagnosed.

Usage:
    # In one terminal:
    ros2 launch tb3_bringup bringup.launch.py use_bridge:=false planner:=none

    # In another:
    ros2 run tb3_nav calibrate

The calibrate node reads the running grid_nav_node's `cell_size` and
`use_imu_heading` via the parameter service, so it does not need to
be re-configured when those change.

Tests:
    T1  3x  no-rotation forward drive of 1.0 m   -> linear_calibration
    T2  2x  ~90 deg rotation + tiny drive        -> rotation diagnostics
    T3  1x  90 deg rotation + 1.0 m drive        -> rotation+drive cross-check
"""

from __future__ import annotations

import math
import sys
import threading
import time
from dataclasses import dataclass

import rclpy
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from rcl_interfaces.srv import GetParameters

from tb3_interfaces.action import MoveToGrid
from tb3_interfaces.msg import GridPose
from tb3_interfaces.srv import SetGridPose

from .grid_math import normalize_angle, yaw_from_quaternion


# Names match what bringup launches with default node names.
NAV_NODE = '/grid_nav_node'
MOVE_ACTION = 'move_to_grid'
SET_POSE_SERVICE = f'{NAV_NODE}/set_grid_pose'
GET_PARAMS_SERVICE = f'{NAV_NODE}/get_parameters'

DEFAULT_CELL_SIZE = 0.25
DEFAULT_TEST_DISTANCE_M = 1.0  # T1 + T3 forward distance
T2_DRIVE_DISTANCE_M = 0.05     # tiny drive to satisfy distance tolerance


@dataclass
class PhaseSnapshot:
    """Pose state at a single phase transition during a move."""
    phase: str
    elapsed_s: float
    grid_x: float
    grid_y: float
    world_heading: float
    imu_yaw_raw: float
    odom_x: float
    odom_y: float
    odom_yaw_raw: float

    def world_x(self, cell_size: float) -> float:
        return self.grid_x * cell_size

    def world_y(self, cell_size: float) -> float:
        return self.grid_y * cell_size


class CalibrateNode(Node):
    """ROS plumbing for the calibration script.

    Owns the action/service clients, the IMU/odom subscriptions, and a
    snapshot list filled in from action feedback. Designed to be spun
    on a background thread while the main thread runs the interactive
    procedure.
    """

    def __init__(self) -> None:
        super().__init__('calibrate')

        # Latest sensor values (guarded by _lock)
        self._lock = threading.Lock()
        self._imu_yaw_raw: float = 0.0
        self._imu_received = False
        self._odom_x: float = 0.0
        self._odom_y: float = 0.0
        self._odom_yaw_raw: float = 0.0
        self._odom_received = False

        # Per-move state
        self._snapshots: list[PhaseSnapshot] = []
        self._last_phase: str | None = None
        self._goal_handle: ClientGoalHandle | None = None
        self._goal_done = threading.Event()
        self._goal_result: MoveToGrid.Result | None = None

        # Subscriptions — match the QoS used by the nav node (best-effort).
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        sensor_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Imu, '/imu', self._imu_cb, sensor_qos)
        self.create_subscription(Odometry, '/odom', self._odom_cb, sensor_qos)

        # Clients
        self._goal_client = ActionClient(self, MoveToGrid, MOVE_ACTION)
        self._pose_client = self.create_client(SetGridPose, SET_POSE_SERVICE)
        self._param_client = self.create_client(GetParameters, GET_PARAMS_SERVICE)

    # -- Public API used by the procedure --

    def wait_for_servers(self, timeout: float = 15.0) -> None:
        deadline = time.monotonic() + timeout
        if not self._goal_client.wait_for_server(timeout_sec=timeout):
            raise RuntimeError(f'MoveToGrid action server not available at {MOVE_ACTION!r}')
        if not self._pose_client.wait_for_service(timeout_sec=max(0.1, deadline - time.monotonic())):
            raise RuntimeError(f'SetGridPose service not available at {SET_POSE_SERVICE!r}')
        # Param service is optional — fall back to defaults if it never appears.
        self._param_client.wait_for_service(timeout_sec=2.0)
        # Wait for at least one IMU + odom message.
        while time.monotonic() < deadline:
            with self._lock:
                if self._imu_received and self._odom_received:
                    return
            time.sleep(0.1)
        raise RuntimeError('Timed out waiting for /imu and /odom messages')

    def fetch_nav_params(self) -> tuple[float, bool]:
        """Read cell_size and use_imu_heading from the running nav node.

        Falls back to (DEFAULT_CELL_SIZE, True) if the parameter service
        isn't reachable (e.g. nav node renamed).
        """
        if not self._param_client.service_is_ready():
            self.get_logger().warning(
                'GetParameters service not ready; assuming cell_size=%.3f, use_imu_heading=True'
                % DEFAULT_CELL_SIZE
            )
            return DEFAULT_CELL_SIZE, True

        req = GetParameters.Request()
        req.names = ['cell_size', 'use_imu_heading']
        future = self._param_client.call_async(req)
        # Spin (in another thread) handles the future; wait here.
        if not _wait_future(future, timeout=3.0):
            self.get_logger().warning('GetParameters call timed out; using defaults')
            return DEFAULT_CELL_SIZE, True

        try:
            values = future.result().values
            cell_size = float(values[0].double_value)
            use_imu = bool(values[1].bool_value)
            return cell_size, use_imu
        except Exception as exc:
            self.get_logger().warning(f'GetParameters parse failed ({exc}); using defaults')
            return DEFAULT_CELL_SIZE, True

    def set_pose(self, grid_x: float, grid_y: float, heading: float,
                 timeout: float = 3.0) -> None:
        req = SetGridPose.Request()
        req.pose = GridPose(x=float(grid_x), y=float(grid_y), heading=float(heading))
        future = self._pose_client.call_async(req)
        if not _wait_future(future, timeout=timeout):
            raise RuntimeError('SetGridPose call timed out')
        # The service always returns success=True today, but check anyway.
        resp = future.result()
        if not resp.success:
            raise RuntimeError(f'SetGridPose failed: {resp.message}')

    def run_move(self, target_x: float, target_y: float,
                 timeout: float = 60.0) -> tuple[MoveToGrid.Result, list[PhaseSnapshot]]:
        # Reset per-move state
        self._snapshots = []
        self._last_phase = None
        self._goal_handle = None
        self._goal_result = None
        self._goal_done.clear()

        goal = MoveToGrid.Goal()
        goal.target_x = float(target_x)
        goal.target_y = float(target_y)

        send_future = self._goal_client.send_goal_async(
            goal, feedback_callback=self._feedback_cb)
        send_future.add_done_callback(self._goal_response_cb)

        if not self._goal_done.wait(timeout=timeout):
            if self._goal_handle is not None:
                self._goal_handle.cancel_goal_async()
            raise RuntimeError(f'Move did not complete within {timeout}s')

        if self._goal_result is None:
            raise RuntimeError('Goal was rejected by the nav node')
        return self._goal_result, list(self._snapshots)

    def cancel_active_goal(self) -> None:
        if self._goal_handle is not None and not self._goal_done.is_set():
            self._goal_handle.cancel_goal_async()

    # -- Callbacks --

    def _imu_cb(self, msg: Imu) -> None:
        q = msg.orientation
        yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        with self._lock:
            self._imu_yaw_raw = yaw
            self._imu_received = True

    def _odom_cb(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        with self._lock:
            self._odom_x = msg.pose.pose.position.x
            self._odom_y = msg.pose.pose.position.y
            self._odom_yaw_raw = yaw
            self._odom_received = True

    def _feedback_cb(self, fb_msg) -> None:
        fb = fb_msg.feedback
        phase = fb.phase
        if phase == self._last_phase:
            return
        with self._lock:
            snap = PhaseSnapshot(
                phase=phase,
                elapsed_s=float(fb.elapsed_time_s),
                grid_x=float(fb.current_pose.x),
                grid_y=float(fb.current_pose.y),
                world_heading=float(fb.current_pose.heading),
                imu_yaw_raw=self._imu_yaw_raw,
                odom_x=self._odom_x,
                odom_y=self._odom_y,
                odom_yaw_raw=self._odom_yaw_raw,
            )
        self._snapshots.append(snap)
        self._last_phase = phase

    def _goal_response_cb(self, fut) -> None:
        handle = fut.result()
        if not handle.accepted:
            self._goal_result = None
            self._goal_done.set()
            return
        self._goal_handle = handle
        result_fut = handle.get_result_async()
        result_fut.add_done_callback(self._result_cb)

    def _result_cb(self, fut) -> None:
        self._goal_result = fut.result().result
        self._goal_done.set()


# -- Helpers --

def _wait_future(future, timeout: float = 5.0) -> bool:
    """Wait for an rclpy future to complete on the spinning executor."""
    deadline = time.monotonic() + timeout
    while not future.done():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.02)
    return True


def _read_float(prompt: str, allow_negative: bool = True) -> float:
    while True:
        raw = input(prompt).strip()
        try:
            v = float(raw)
        except ValueError:
            print('  Not a number. Try again.')
            continue
        if not allow_negative and v < 0:
            print('  Must be >= 0. Try again.')
            continue
        return v


def _ensure_enter(prompt: str) -> None:
    input(prompt)


def _hr() -> None:
    print('-' * 60)


def _phase_table(snapshots: list[PhaseSnapshot]) -> str:
    """Format snapshots as a compact table for the report."""
    if not snapshots:
        return '  (no phase samples captured)'
    rows = ['  phase       t(s)   gx     gy     world_h   imu_h    odom(x,y,h)']
    for s in snapshots:
        rows.append(
            f'  {s.phase:<10} {s.elapsed_s:5.2f}  '
            f'{s.grid_x:5.2f}  {s.grid_y:5.2f}  '
            f'{s.world_heading:+7.3f}  {s.imu_yaw_raw:+7.3f}  '
            f'({s.odom_x:+5.2f}, {s.odom_y:+5.2f}, {s.odom_yaw_raw:+6.3f})'
        )
    return '\n'.join(rows)


def _pick(snapshots: list[PhaseSnapshot], phase: str) -> PhaseSnapshot | None:
    for s in snapshots:
        if s.phase == phase:
            return s
    return None


# -- Test maneuvers --

def t1_linear(node: CalibrateNode, cell_size: float) -> list[tuple[float, list[PhaseSnapshot]]]:
    """Three trials: drive 1.0 m forward with no pre-rotation."""
    target_x_cells = DEFAULT_TEST_DISTANCE_M / cell_size  # 4.0 at cell_size=0.25
    print()
    _hr()
    print('TEST 1 / 3 - Straight-drive linear calibration')
    print(f'  Robot will drive {DEFAULT_TEST_DISTANCE_M:.2f} m forward (no pre-rotation).')
    print('  Repeat 3x; report the actual distance with a tape measure.')
    _hr()

    results: list[tuple[float, list[PhaseSnapshot]]] = []
    for trial in range(3):
        print(f'\n[T1.{trial + 1}] Place the robot facing along your reference line.')
        print('       Mark its current footprint on the floor (front wheel center is fine).')
        _ensure_enter('       Press Enter when ready: ')
        node.set_pose(0.0, 0.0, 0.0)
        time.sleep(0.3)  # let the override propagate
        result, snaps = node.run_move(target_x_cells, 0.0)
        if not result.success:
            print(f'       Move failed: {result.message}. Skipping trial.')
            continue
        measured = _read_float(
            f'       Measured forward distance from start (m): ',
            allow_negative=False,
        )
        results.append((measured, snaps))
    return results


def t2_rotation(node: CalibrateNode, cell_size: float) -> list[tuple[float, list[PhaseSnapshot]]]:
    """Two trials: rotate ~90 deg in place (tiny forward drive after)."""
    # Target due North at 0.05 m: forces +pi/2 rotation, then ~5 cm drive.
    target_y_cells = T2_DRIVE_DISTANCE_M / cell_size  # 0.2 at cell_size=0.25
    print()
    _hr()
    print('TEST 2 / 3 - Rotation calibration')
    print('  Robot will rotate ~90 deg CCW, then drive ~5 cm.')
    print('  After it stops, measure the actual rotation against a printed 90 deg template')
    print('  or two reference lines you mark on the floor before the rotation.')
    _hr()

    results: list[tuple[float, list[PhaseSnapshot]]] = []
    for trial in range(2):
        print(f'\n[T2.{trial + 1}] Place the robot facing along your reference line.')
        print('       Mark a second line at +90 deg from it (printed template helps).')
        _ensure_enter('       Press Enter when ready: ')
        node.set_pose(0.0, 0.0, 0.0)
        time.sleep(0.3)
        result, snaps = node.run_move(0.0, target_y_cells)
        if not result.success:
            print(f'       Move failed: {result.message}. Skipping trial.')
            continue
        measured_deg = _read_float(
            '       Measured rotation in degrees (positive = CCW): ',
        )
        results.append((math.radians(measured_deg), snaps))
    return results


def t3_rotation_drive(node: CalibrateNode, cell_size: float) -> tuple[
        tuple[float, float, float] | None, list[PhaseSnapshot]]:
    """One trial: rotate ~90 deg then drive 1.0 m. Cross-check."""
    target_y_cells = DEFAULT_TEST_DISTANCE_M / cell_size  # 4.0
    print()
    _hr()
    print('TEST 3 / 3 - Rotation + drive cross-check')
    print(f'  Robot will rotate ~90 deg CCW then drive {DEFAULT_TEST_DISTANCE_M:.2f} m.')
    print('  Measure the final position and final heading relative to the start.')
    _hr()

    print('\n[T3]   Place the robot facing along your reference line.')
    print('       Mark its current footprint clearly (you will measure two displacements).')
    _ensure_enter('       Press Enter when ready: ')
    node.set_pose(0.0, 0.0, 0.0)
    time.sleep(0.3)
    result, snaps = node.run_move(0.0, target_y_cells)
    if not result.success:
        print(f'       Move failed: {result.message}.')
        return None, snaps

    print()
    print('       Measure the final position relative to the start point.')
    print('       +X is the direction the robot was facing initially.')
    print('       +Y is 90 deg CCW from that (i.e. left of the start heading).')
    dx = _read_float('       Final dX (m, +forward): ')
    dy = _read_float('       Final dY (m, +left):    ')
    dh_deg = _read_float('       Final heading change (deg, +CCW from start): ')
    return (dx, dy, math.radians(dh_deg)), snaps


# -- Reporting --

def _summarize_t1(t1: list[tuple[float, list[PhaseSnapshot]]],
                  cell_size: float) -> tuple[float | None, list[str]]:
    lines: list[str] = []
    if not t1:
        lines.append('  T1: no successful trials.')
        return None, lines
    cmd = DEFAULT_TEST_DISTANCE_M
    ratios: list[float] = []
    for i, (measured, snaps) in enumerate(t1, 1):
        ratio = measured / cmd
        ratios.append(ratio)
        lines.append(
            f'  T1.{i}: commanded {cmd:.3f} m, measured {measured:.3f} m, ratio {ratio:.3f}'
        )
    avg = sum(ratios) / len(ratios)
    spread = max(ratios) - min(ratios)
    lines.append(f'  T1 mean ratio = {avg:.3f}  (spread {spread:.3f})')
    return avg, lines


def _summarize_t2(t2: list[tuple[float, list[PhaseSnapshot]]]) -> tuple[
        float | None, float | None, list[str]]:
    """Returns (mean_imu_rot_rad, mean_physical_rot_rad, log_lines)."""
    lines: list[str] = []
    if not t2:
        lines.append('  T2: no successful trials.')
        return None, None, lines
    imu_rots: list[float] = []
    phys_rots: list[float] = []
    for i, (measured_rad, snaps) in enumerate(t2, 1):
        rot_start = _pick(snaps, 'rotating')
        rot_end = _pick(snaps, 'settling') or _pick(snaps, 'driving')
        if rot_start is None or rot_end is None:
            lines.append(f'  T2.{i}: missing phase samples (skipped).')
            continue
        imu_rot = normalize_angle(rot_end.imu_yaw_raw - rot_start.imu_yaw_raw)
        odom_rot = normalize_angle(rot_end.odom_yaw_raw - rot_start.odom_yaw_raw)
        imu_rots.append(imu_rot)
        phys_rots.append(measured_rad)
        lines.append(
            f'  T2.{i}: physical {math.degrees(measured_rad):+6.1f} deg | '
            f'IMU {math.degrees(imu_rot):+6.1f} deg | '
            f'odom {math.degrees(odom_rot):+6.1f} deg'
        )
    if not imu_rots:
        return None, None, lines
    return (sum(imu_rots) / len(imu_rots),
            sum(phys_rots) / len(phys_rots),
            lines)


def _summarize_t3(t3: tuple[tuple[float, float, float] | None, list[PhaseSnapshot]],
                  cell_size: float) -> tuple[dict | None, list[str]]:
    measured, snaps = t3
    lines: list[str] = []
    if measured is None:
        lines.append('  T3: failed.')
        return None, lines
    dx, dy, dh = measured
    expected_dist = DEFAULT_TEST_DISTANCE_M  # ideal physical drive after 90 deg rotation
    actual_dist = math.hypot(dx, dy)
    actual_dir = math.atan2(dy, dx)  # in robot's start frame
    expected_dir = math.pi / 2  # commanded to drive due-left from start heading
    rot_start = _pick(snaps, 'rotating')
    drive_start = _pick(snaps, 'driving')
    drive_end = snaps[-1] if snaps else None

    imu_rot = (
        normalize_angle(drive_start.imu_yaw_raw - rot_start.imu_yaw_raw)
        if (rot_start and drive_start) else None
    )
    lines.append(f'  T3 commanded: rotate +90 deg, drive {expected_dist:.3f} m')
    lines.append(
        f'  T3 measured: pos = ({dx:+.3f}, {dy:+.3f}) m  '
        f'(|d|={actual_dist:.3f} m, dir={math.degrees(actual_dir):+.1f} deg)  '
        f'final heading change = {math.degrees(dh):+.1f} deg'
    )
    if imu_rot is not None:
        lines.append(f'  T3 IMU rotation during ROTATING phase = {math.degrees(imu_rot):+.1f} deg')
    return {
        'dx': dx, 'dy': dy, 'dh': dh,
        'actual_dist': actual_dist, 'actual_dir': actual_dir,
        'expected_dir': expected_dir, 'imu_rot': imu_rot,
    }, lines


def _verdict(t1_ratio: float | None,
             t2_imu_rot: float | None, t2_phys_rot: float | None,
             t3: dict | None) -> list[str]:
    """Plain-English diagnosis. Each string is a bullet."""
    out: list[str] = []
    if t1_ratio is None:
        out.append('Linear: no data.')
    elif 0.95 <= t1_ratio <= 1.05:
        out.append(f'Linear: no significant slip ({t1_ratio:.2f}). Leave linear_calibration at 1.0.')
    else:
        out.append(
            f'Linear: robot {"under" if t1_ratio < 1 else "over"}-moves by '
            f'~{abs(1 - t1_ratio) * 100:.0f}%. '
            f'Suggested linear_calibration = {t1_ratio:.3f}.'
        )

    if t2_imu_rot is None or t2_phys_rot is None:
        out.append('Rotation: no data.')
    else:
        imu_err = abs(normalize_angle(t2_imu_rot - math.pi / 2))
        phys_err = abs(normalize_angle(t2_phys_rot - math.pi / 2))
        if imu_err < math.radians(5) and phys_err < math.radians(5):
            out.append('Rotation: IMU and physical both ~90 deg -> rotation is reliable.')
        elif imu_err < math.radians(5) and phys_err >= math.radians(5):
            out.append(
                f'Rotation: IMU says 90 deg but physical was '
                f'{math.degrees(t2_phys_rot):.1f} deg -> IMU yaw is biased '
                'or the user reference template was wrong.'
            )
        elif imu_err >= math.radians(5):
            out.append(
                f'Rotation: IMU only reached {math.degrees(t2_imu_rot):.1f} deg '
                f'(commanded 90 deg). Controller stop condition is not converging - '
                'check heading_tolerance and kp_angular.'
            )

    if t3 is not None:
        # Compare physical motion direction to IMU heading at drive start.
        # If they differ, recorded x/y is in a frame rotated from the IMU/world.
        if t3['imu_rot'] is not None:
            cmd_dir = math.pi / 2
            phys_dir = t3['actual_dir']
            phys_err = abs(normalize_angle(phys_dir - cmd_dir))
            if phys_err < math.radians(10):
                out.append(
                    'T3 cross-check: motion direction matches commanded heading. '
                    'No rotation-coupled drift detected.'
                )
            else:
                out.append(
                    f'T3 cross-check: motion direction off by '
                    f'{math.degrees(phys_err):.0f} deg from commanded. Likely '
                    'rotation-coupled drift -- raise settling_time and/or '
                    'lower max_angular_speed before trusting calibration.'
                )
    return out


def _yaml_snippet(linear_cal: float | None,
                  angular_cal: float | None,
                  use_imu: bool) -> str:
    lines = ['grid_nav_node:', '  ros__parameters:']
    if linear_cal is None:
        lines.append('    linear_calibration: 1.0   # T1 had no data; left at 1.0')
    else:
        lines.append(f'    linear_calibration: {linear_cal:.3f}')
    if not use_imu and angular_cal is not None:
        lines.append(f'    angular_calibration: {angular_cal:.3f}')
    else:
        lines.append('    # angular_calibration: 1.0  # IMU mode in use; not applied')
    return '\n'.join(lines)


def _full_report(t1, t2, t3, cell_size: float, use_imu: bool) -> None:
    print()
    print('=' * 60)
    print('CALIBRATION REPORT')
    print('=' * 60)
    print(f'cell_size in use: {cell_size:.3f} m')
    print(f'use_imu_heading:  {use_imu}')

    print('\n[T1] Linear (no rotation):')
    t1_ratio, t1_lines = _summarize_t1(t1, cell_size)
    for line in t1_lines:
        print(line)
    for trial_idx, (_, snaps) in enumerate(t1, 1):
        print(f'  T1.{trial_idx} phase samples:')
        print(_phase_table(snaps))

    print('\n[T2] Rotation:')
    t2_imu, t2_phys, t2_lines = _summarize_t2(t2)
    for line in t2_lines:
        print(line)
    for trial_idx, (_, snaps) in enumerate(t2, 1):
        print(f'  T2.{trial_idx} phase samples:')
        print(_phase_table(snaps))

    print('\n[T3] Rotation + drive:')
    t3_summary, t3_lines = _summarize_t3(t3, cell_size)
    for line in t3_lines:
        print(line)
    print('  T3 phase samples:')
    print(_phase_table(t3[1]))

    print('\nVerdict:')
    for bullet in _verdict(t1_ratio, t2_imu, t2_phys, t3_summary):
        print(f'  - {bullet}')

    # angular_cal only makes sense from T2 in use_imu=False mode.
    angular_cal = None
    if not use_imu and t2_phys is not None and t2_imu is not None and abs(t2_imu) > 1e-6:
        angular_cal = t2_phys / t2_imu

    print('\nPaste into src/tb3_bringup/config/experiment.yaml:')
    print('-' * 60)
    print(_yaml_snippet(t1_ratio, angular_cal, use_imu))
    print('-' * 60)


# -- Entry point --

def main(args=None) -> int:
    rclpy.init(args=args)
    node = CalibrateNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    exit_code = 0
    try:
        print('Waiting for grid_nav_node and sensor data...')
        node.wait_for_servers()
        cell_size, use_imu = node.fetch_nav_params()
        print(f'Connected. cell_size={cell_size:.3f} m, use_imu_heading={use_imu}')

        t1 = t1_linear(node, cell_size)
        t2 = t2_rotation(node, cell_size)
        t3 = t3_rotation_drive(node, cell_size)

        _full_report(t1, t2, t3, cell_size, use_imu)
    except KeyboardInterrupt:
        print('\nAborted by user.')
        node.cancel_active_goal()
        exit_code = 130
    except Exception as exc:
        print(f'\nCalibration failed: {exc}', file=sys.stderr)
        node.cancel_active_goal()
        exit_code = 1
    finally:
        executor.shutdown()
        rclpy.try_shutdown()
        spin_thread.join(timeout=1.0)
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
