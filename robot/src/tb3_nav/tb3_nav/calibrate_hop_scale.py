"""Hop-scale linear calibration for grid_nav_node.

Why this exists separately from ``calibrate``: T1 in the main tool measures
``linear_calibration`` across a 1.0 m sustained-cruise drive, where the
trajectory is ~93% saturated cruise at max_linear_speed. The experiment's
operating point is the opposite — ~20 cm goals between standstills, where
the trajectory is dominated by acceleration transients, the P-controller
ramp-down, the min_linear_speed floor, and brake overshoot. The slip
dynamics at those operating points differ, so this tool measures the
calibration at the experiment's actual scale.

What it does:
    1. Resets the state tracker to (0, 0, 0) and drives the robot 0.20 m
       forward.
    2. Repeats N times (default 5). Between hops the pose is reset so each
       hop is independent in the state tracker's reference frame — but
       the robot physically keeps moving forward, so the tape-measured
       cumulative distance is the source of truth.
    3. Prompts for the cumulative tape-measured distance (or per-hop with
       ``--per-hop``).
    4. Computes a hop-scale ``linear_calibration`` correction that
       accounts for the currently-active calibration on the running nav
       node, and prints a paste-ready YAML snippet.

Math (since we run with a non-1.0 calibration already active):
    The controller stops when ``odom_integrated * linear_cal_current >=
    target_D``. The physical distance covered is ``odom_integrated *
    true_slip_ratio = target_D * (true_slip_ratio / linear_cal_current)``.
    So ``measured / target = true_slip / linear_cal_current`` and the new
    calibration that gives correct future moves is
    ``new_linear_cal = (measured / target) * linear_cal_current``.

    If you run T1 of ``calibrate`` with linear_cal_current=1.0 (the
    package default), this collapses to the textbook ``new = measured /
    commanded``. If you run it with a non-1.0 value already in place
    (e.g. the previous T1 result), this corrects for that — pasting the
    output is always idempotent toward the true slip ratio.

Usage:
    # In one terminal:
    ros2 launch tb3_bringup bringup.launch.py use_bridge:=false planner:=none

    # In another:
    ros2 run tb3_nav calibrate_hop                   # 5 hops x 0.20 m
    ros2 run tb3_nav calibrate_hop --hops 10         # 10 hops
    ros2 run tb3_nav calibrate_hop --distance 0.15   # different hop length
    ros2 run tb3_nav calibrate_hop --per-hop         # measure every hop (gives sigma)
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor

from rcl_interfaces.srv import GetParameters

from .calibrate import (
    CalibrateNode,
    _ensure_enter,
    _hr,
    _read_float,
    _wait_future,
)


DEFAULT_HOPS = 5
DEFAULT_HOP_DISTANCE_M = 0.20
# Pause between hops so the robot is fully at rest before we reset pose.
# (run_move returns when the action declares DONE; the motors take a few
# hundred ms to physically settle from the brake.)
INTER_HOP_REST_S = 0.5


def fetch_linear_calibration(node: CalibrateNode) -> float:
    """Read the running grid_nav_node's current linear_calibration param.

    Returns 1.0 if the param service is unreachable or the call fails —
    same defensive fallback as ``CalibrateNode.fetch_nav_params``.
    """
    # `_param_client` is private to CalibrateNode but we're in the same
    # package and use the same patterns it does. Acceptable coupling.
    client = node._param_client  # noqa: SLF001
    if not client.service_is_ready():
        print('  WARN: param service not ready; assuming linear_calibration=1.0')
        return 1.0
    req = GetParameters.Request()
    req.names = ['linear_calibration']
    future = client.call_async(req)
    if not _wait_future(future, timeout=3.0):
        print('  WARN: param fetch timed out; assuming linear_calibration=1.0')
        return 1.0
    try:
        return float(future.result().values[0].double_value)
    except Exception as exc:  # noqa: BLE001 — log + fall back
        print(f'  WARN: param parse failed ({exc}); assuming linear_calibration=1.0')
        return 1.0


def run_hop_scale(node: CalibrateNode, hops: int, hop_distance: float,
                  cell_size: float, per_hop: bool) -> None:
    target_x_cells = hop_distance / cell_size
    current_linear_cal = fetch_linear_calibration(node)
    expected_total = hops * hop_distance

    print()
    _hr()
    print('HOP-SCALE LINEAR CALIBRATION')
    _hr()
    print(f'  hops:                 {hops}')
    print(f'  distance per hop:     {hop_distance:.3f} m  '
          f'(= {target_x_cells:.3f} cells at cell_size {cell_size:.3f})')
    print(f'  expected total:       {expected_total:.3f} m')
    print(f'  current linear_calibration on grid_nav_node: {current_linear_cal:.4f}')
    print(f'  per-hop measurement:  {"enabled (gives sigma)" if per_hop else "off (cumulative only)"}')
    _hr()

    print()
    print('SETUP:')
    print('  - Place the robot facing along your reference line.')
    print('  - Mark its starting position with tape — front-wheel center is a')
    print('    good landmark and matches how T1 measures.')
    print(f'  - Ensure ~{expected_total + 0.5:.1f} m of clear floor ahead of the robot.')
    _ensure_enter('  Press Enter to start: ')

    per_hop_measured: list[float] = []
    per_hop_action: list[float] = []

    for i in range(hops):
        print(f'\n[hop {i + 1}/{hops}] reset pose to origin; drive {hop_distance:.3f} m forward')
        node.set_pose(0.0, 0.0, 0.0)
        # Let the override propagate to the state tracker before we issue
        # the goal, otherwise the action server may snapshot pre-reset state.
        time.sleep(0.3)
        try:
            result, _snaps = node.run_move(target_x_cells, 0.0)
        except RuntimeError as exc:
            print(f'  ERROR: {exc}. Aborting test.')
            return
        if not result.success:
            print(f'  Move failed: {result.message}. Aborting test.')
            return

        action_distance = float(result.total_distance_m)
        per_hop_action.append(action_distance)
        bias_action = action_distance - hop_distance
        print(f'  action.total_distance_m = {action_distance:.4f} m '
              f'(target {hop_distance:.4f}, bias {bias_action:+.4f})')

        if per_hop:
            print('  Mark the robot\'s current position on the floor.')
            measured = _read_float(
                '  Measured physical distance THIS hop only (m): ',
                allow_negative=False,
            )
            per_hop_measured.append(measured)

        # Let the robot physically settle before we reset for the next hop.
        time.sleep(INTER_HOP_REST_S)

    # --- final cumulative measurement (always required) ------------------
    print()
    _hr()
    print('FINAL MEASUREMENT')
    _hr()
    print(f'  Measure cumulative physical distance from your START tape mark')
    print(f'  to the robot\'s current position (same landmark as the start mark).')
    cumulative = _read_float('  Cumulative measured distance (m): ',
                             allow_negative=False)

    # --- analysis --------------------------------------------------------
    per_hop_mean_meas = cumulative / hops
    per_hop_bias = per_hop_mean_meas - hop_distance
    bias_pct = (per_hop_bias / hop_distance) * 100.0
    # See module docstring for the derivation.
    new_linear_cal = (per_hop_mean_meas / hop_distance) * current_linear_cal
    delta = new_linear_cal - current_linear_cal

    print()
    _hr()
    print('RESULTS')
    _hr()
    print(f'  hops driven:           {hops}')
    print(f'  per-hop commanded:     {hop_distance:.4f} m')
    print(f'  cumulative commanded:  {expected_total:.4f} m')
    print(f'  cumulative measured:   {cumulative:.4f} m')
    print(f'  per-hop mean measured: {per_hop_mean_meas:.4f} m')
    print(f'  per-hop bias:          {per_hop_bias:+.4f} m  '
          f'({bias_pct:+.1f} % of commanded)')

    if per_hop and len(per_hop_measured) >= 2:
        n = len(per_hop_measured)
        mean = sum(per_hop_measured) / n
        var = sum((x - mean) ** 2 for x in per_hop_measured) / (n - 1)
        sigma = var ** 0.5
        spread = max(per_hop_measured) - min(per_hop_measured)
        print()
        print(f'  per-hop measurements:  {[f"{m:.4f}" for m in per_hop_measured]}')
        print(f'  per-hop mean:          {mean:.4f} m')
        print(f'  per-hop sigma:         {sigma:.4f} m  (random per-hop error)')
        print(f'  per-hop spread:        {spread:.4f} m  (max - min)')
        if sigma > 0.015:
            print('  NOTE: sigma > 15 mm — high random per-hop variance. Likely causes:')
            print('    - inconsistent floor surface (dust, varying friction)')
            print('    - battery voltage decay over the test (run again with fresh charge)')
            print('    - tape-measurement technique (mark exactly the same wheel point each time)')

    print()
    print(f'  CURRENT linear_calibration:  {current_linear_cal:.4f}')
    print(f'  HOP-SCALE recommendation:    {new_linear_cal:.4f}')
    print(f'  delta:                       {delta:+.4f}')
    if abs(delta) < 0.02:
        print('  -> Within +/-0.02 of current value. T1\'s sustained-cruise ratio')
        print('     happens to transfer well to short hops here; no change needed.')
    elif abs(delta) < 0.05:
        print('  -> Modest correction. Pasting the hop-scale value will tighten')
        print('     per-hop accuracy by a few millimetres.')
    else:
        print('  -> Significant correction. T1\'s value is mismatched for the')
        print('     experiment\'s 20 cm operating point — paste the hop-scale value.')

    print()
    print('  Action-reported per-hop distances (state tracker frame, NOT physical):')
    for i, d in enumerate(per_hop_action):
        print(f'    hop {i + 1}: {d:.4f} m')
    print(f'  Each should land near {hop_distance:.4f} m within distance_tolerance.')
    print('  Drift here would mean the controller is stopping for a non-tolerance')
    print('  reason (timeout, preempt). All-near-target is the healthy signal.')

    print()
    _hr()
    print('PASTE INTO src/tb3_bringup/config/experiment.yaml')
    _hr()
    print()
    print('grid_nav_node:')
    print('  ros__parameters:')
    print(f'    linear_calibration: {new_linear_cal:.4f}  '
          f'# hop-scale ({hops} x {hop_distance:.2f} m, '
          f'previous {current_linear_cal:.4f})')
    print()


def main(args=None) -> int:
    parser = argparse.ArgumentParser(
        description='Hop-scale linear calibration for grid_nav_node — '
                    'measures linear_calibration over short start-stop hops '
                    'that match the experiment\'s operating point.',
    )
    parser.add_argument('--hops', type=int, default=DEFAULT_HOPS,
                        help='number of consecutive hops (default: %(default)s)')
    parser.add_argument('--distance', type=float, default=DEFAULT_HOP_DISTANCE_M,
                        help='distance per hop in meters (default: %(default)s)')
    parser.add_argument('--per-hop', action='store_true',
                        help='prompt for a measurement after each hop; gives sigma '
                             '(default: cumulative measurement only)')
    cli_args = parser.parse_args(args)

    if cli_args.hops < 1:
        print('ERROR: --hops must be >= 1', file=sys.stderr)
        return 1
    if cli_args.distance <= 0.0:
        print('ERROR: --distance must be > 0', file=sys.stderr)
        return 1

    rclpy.init()
    node = CalibrateNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    rc = 0
    try:
        node.wait_for_servers()
        cell_size, _use_imu = node.fetch_nav_params()
        run_hop_scale(node, cli_args.hops, cli_args.distance, cell_size,
                      cli_args.per_hop)
    except KeyboardInterrupt:
        print('\nAborted.')
        node.cancel_active_goal()
        rc = 130
    except RuntimeError as exc:
        print(f'\nERROR: {exc}', file=sys.stderr)
        rc = 1
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
        spin_thread.join(timeout=1.0)

    return rc


if __name__ == '__main__':
    sys.exit(main())
