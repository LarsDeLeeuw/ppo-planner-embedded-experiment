#!/usr/bin/env python3
"""Smoke-drive the TCP bridge end-to-end without the orchestrator.

Sends a sequence of bridge JSON messages that exercises the full
experiment-side wiring:
  - experiment_event(run_start)   -> /experiment/events
  - pose                          -> /grid_nav_node/set_grid_pose
  - predict (sequence=1)          -> planner /predict_action service
                                  -> /planner/metrics + /bridge/predict_timing
  - goal                          -> move_to_grid action
                                  -> /grid_nav_node/last_result (on result)
  - experiment_event(run_end)

Each step prints a one-line status. Use --no-goal to skip the physical
motion (useful for benchtop testing before the robot is on the ground).

Run this on any host that can reach the bridge TCP port (9090). It does
NOT need a ROS install.

Usage:
  scripts/smoke/04_drive_bridge.py --robot-ip 192.0.2.10
  scripts/smoke/04_drive_bridge.py --robot-ip turtlebot3.local --no-goal
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from contextlib import contextmanager


GRID_SIZE = 10  # PPO model expects 10x10; A* doesn't care.
DEFAULT_PORT = 9090
RUN_ID = f"smoke-{int(time.time())}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Drive the bridge for a smoke test.")
    p.add_argument("--robot-ip", required=True, help="Bridge host (hostname or IP).")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--no-goal", action="store_true",
                   help="Skip the move_to_grid goal (no physical motion).")
    p.add_argument("--goal-cell", type=int, nargs=2, default=[1, 0],
                   metavar=("X", "Y"), help="Goal grid cell for predict + goal.")
    p.add_argument("--cell-size", type=float, default=0.20,
                   help="cell_size in metres; used to convert goal cell -> metres.")
    p.add_argument("--goal-timeout-s", type=float, default=60.0)
    return p.parse_args()


@contextmanager
def bridge_socket(host: str, port: int):
    sock = socket.create_connection((host, port), timeout=10.0)
    sock.settimeout(2.0)
    try:
        yield sock
    finally:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        sock.close()


class BridgeClient:
    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._buf = b""

    def send(self, payload: dict) -> None:
        self._sock.sendall((json.dumps(payload) + "\n").encode())

    # The bridge forwards every /grid_nav_node/grid_pose tick (10 Hz) back as
    # nav_pose. Useful for an orchestrator with a live visualizer, but pure
    # noise during a smoke test. Drop them from the interim print stream.
    # goal_feedback is kept because the phase transitions (rotating ->
    # driving -> finalizing) are useful operator signal.
    _NOISY_INTERIM_TYPES = {"nav_pose"}

    def recv_until(self, msg_type: str, timeout_s: float) -> dict | None:
        """Read newline-delimited JSON; return the first message of msg_type
        seen, or None on timeout. Other messages are echoed to stderr,
        except those in _NOISY_INTERIM_TYPES which are silently skipped."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                chunk = self._sock.recv(8192)
            except socket.timeout:
                continue
            if not chunk:
                return None
            self._buf += chunk
            while b"\n" in self._buf:
                line, self._buf = self._buf.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"  [warn] non-JSON line: {exc}", file=sys.stderr)
                    continue
                if msg.get("type") == msg_type:
                    return msg
                if msg.get("type") in self._NOISY_INTERIM_TYPES:
                    continue
                # surface interim messages so the operator can see progress
                print(f"  [...] {msg.get('type')}: "
                      f"{ {k: v for k, v in msg.items() if k != 'type'} }",
                      file=sys.stderr)
        return None


def zeros(n: int) -> list[list[float]]:
    return [[0.0] * n for _ in range(n)]


def main() -> int:
    args = parse_args()
    print(f"[04] run_id={RUN_ID}  target={args.robot_ip}:{args.port}")
    failed = 0

    with bridge_socket(args.robot_ip, args.port) as sock:
        c = BridgeClient(sock)

        # 1. ping/pong handshake
        c.send({"type": "ping"})
        pong = c.recv_until("pong", timeout_s=3.0)
        print(f"  [{'PASS' if pong else 'FAIL'}] ping -> pong")
        if not pong:
            return 1

        # 2. run_start event (lands on /experiment/events)
        c.send({"type": "experiment_event", "event_type": "run_start",
                "run_id": RUN_ID,
                "payload": {"mode": "smoke", "planner": "astar",
                            "map_id": "smoke", "goal_cell": args.goal_cell,
                            "start_cell": [0, 0], "surface": "lab"}})
        print("  [SENT] experiment_event(run_start)")

        # 3. pose-before-predict invariant: SetGridPose, then PredictAction
        c.send({"type": "pose", "x": 0.0, "y": 0.0, "heading": 0.0})
        c.send({"type": "predict", "sequence": 1,
                "obstacle_map": zeros(GRID_SIZE),
                "energy_map":   zeros(GRID_SIZE),
                "robot_pos":    [0, 0],
                "goal_pos":     list(args.goal_cell)})
        result = c.recv_until("predict_result", timeout_s=10.0)
        if result and result.get("sequence") == 1:
            print(f"  [PASS] predict_result seq=1 action={result.get('action')} "
                  f"direction={result.get('direction')}")
        else:
            print(f"  [FAIL] predict_result not received "
                  f"(got: {result})", file=sys.stderr)
            failed += 1

        # 4. goal (skippable)
        if not args.no_goal:
            target_x_m = float(args.goal_cell[0]) * args.cell_size
            target_y_m = float(args.goal_cell[1]) * args.cell_size
            print(f"  [SENT] goal -> ({target_x_m:.2f}, {target_y_m:.2f}) m "
                  f"(cell {args.goal_cell}, cell_size={args.cell_size})")
            c.send({"type": "goal",
                    "target_x": target_x_m,
                    "target_y": target_y_m})
            gr = c.recv_until("goal_result", timeout_s=args.goal_timeout_s)
            if gr and gr.get("success"):
                print(f"  [PASS] goal_result success=True message={gr.get('message')!r}")
            else:
                print(f"  [FAIL] goal_result success=False (got: {gr})",
                      file=sys.stderr)
                failed += 1
        else:
            print("  [skip] goal (--no-goal)")

        # 5. run_end event
        outcome = "success" if failed == 0 else "failure"
        c.send({"type": "experiment_event", "event_type": "run_end",
                "run_id": RUN_ID, "payload": {"outcome": outcome}})
        print(f"  [SENT] experiment_event(run_end, outcome={outcome})")

    print()
    print(f"[04] {'all checks PASSED' if failed == 0 else f'{failed} check(s) FAILED'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
