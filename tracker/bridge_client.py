"""
bridge_client.py — Non-blocking TCP client for the ROS2 bridge.

Sends pose corrections and navigation goals as newline-delimited JSON.
Receives feedback/results from the bridge node via optional callbacks.
All coordinates are transformed to ROS2 convention before sending.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from coord_transform import tracker_to_ros2
from robot import RobotState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BridgeConfig:
    host: str
    port: int
    grid_rows: int


class BridgeClient:
    """TCP client that sends pose/goal messages to the ROS2 bridge node.

    All public methods are non-blocking — messages are queued and sent
    by a background thread.  If the connection drops, the sender thread
    reconnects automatically with exponential backoff.
    """

    def __init__(self, config: BridgeConfig) -> None:
        self._cfg = config

        # Outbound: latest pose (overwritten) + goal queue
        self._lock = threading.Lock()
        self._pending_pose: dict | None = None
        self._pending_messages: list[dict] = []
        self._has_work = threading.Event()
        self._activated = threading.Event()  # gates connect until first send

        # Inbound callbacks: type_str → list of callable(msg_dict).
        # Multiple subscribers per type are supported; all are invoked in
        # registration order and an exception in one does not silence others.
        self._callbacks: dict[str, list[Callable[[dict], None]]] = {}

        # Predict round-trip timing.  t_tcp_send_ns is captured at the socket
        # write; t_tcp_recv_ns when the matching predict_result is parsed.
        # Both use time.monotonic_ns() on this (laptop) host so the delta is a
        # single-host quantity (see PredictTcpTiming.msg).  Correlation prefers
        # an echoed `sequence` on the response; until the bridge echoes it, we
        # rely on the state machine gating exactly one predict in flight.
        self._timing_lock = threading.Lock()
        self._outstanding_predict: tuple[int, int] | None = None  # (sequence, send_ns)
        self._timing_cbs: list[Callable[[int, int, int], None]] = []

        # Connection state
        self._sock: socket.socket | None = None
        self._running = True

        self._send_thread = threading.Thread(
            target=self._send_loop, name="bridge-send", daemon=True,
        )
        self._recv_thread: threading.Thread | None = None
        self._send_thread.start()

    # -- public API (called from main thread) --------------------------------

    def _build_pose_msg(self, robot: RobotState) -> dict:
        x, y, h = tracker_to_ros2(
            robot.grid_x,
            robot.grid_y,
            robot.heading_deg,
            self._cfg.grid_rows,
        )
        return {"type": "pose", "x": round(x, 4), "y": round(y, 4),
                "heading": round(h, 4)}

    def send_pose(self, robot: RobotState) -> None:
        """Queue a pose correction with the robot's current continuous pose.

        The latest queued pose overwrites any earlier un-sent pose; the
        send loop always emits it before any pending goal/predict so the
        bridge sees a fresh pose before acting on a request.
        """
        pose_msg = self._build_pose_msg(robot)
        with self._lock:
            self._pending_pose = pose_msg
        self._has_work.set()
        self._activated.set()

    def send_goal(
        self, target_col: float, target_row: float, robot: RobotState,
    ) -> None:
        """Queue a pose correction followed by a navigation goal.

        The pose is sent first so the nav node knows the robot's current
        position before it starts planning towards the target cell.
        """
        pose_msg = self._build_pose_msg(robot)

        # Flip to ROS2 Y-up convention (col stays the same)
        ros2_col = target_col
        ros2_row = self._cfg.grid_rows - target_row
        goal_msg = {"type": "goal", "target_x": float(ros2_col), "target_y": float(ros2_row)}

        with self._lock:
            self._pending_pose = pose_msg
            self._pending_messages.append(goal_msg)
        self._has_work.set()
        self._activated.set()

    def send_cancel(self) -> None:
        """Queue a goal cancellation."""
        with self._lock:
            self._pending_messages.append({"type": "cancel_goal"})
        self._has_work.set()
        self._activated.set()

    def send_experiment_event(
        self, event_type: str, run_id: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Queue an `experiment_event` for the bridge to forward onto
        /experiment/events (handover §7.2). The orchestrator uses this to mark
        run_start / run_end / marker / bag_capped in the ROS-side bag stream.
        """
        msg: dict[str, Any] = {
            "type": "experiment_event",
            "event_type": str(event_type),
            "run_id": str(run_id),
            "payload": payload or {},
        }
        with self._lock:
            self._pending_messages.append(msg)
        self._has_work.set()
        self._activated.set()

    def send_predict(
        self,
        obstacle_map: list[list[float]],
        energy_map: list[list[float]],
        robot_pos: tuple[int, int],
        goal_pos: tuple[int, int],
        sequence: int | None = None,
    ) -> None:
        """Queue a one-shot predict request.

        Coordinates and maps are expected already in bridge frame (origin
        bottom-left, +Y=North).  No flipping is done here.

        `sequence` (optional) is an additive per-run monotonic id stamped onto
        the outbound JSON.  The bridge ignores unknown fields today, so this is
        forward-compatible; once the bridge echoes it through to PlannerMetrics
        the latency decomposition can join on it.  When provided, it also keys
        the PredictTcpTiming round-trip capture (see on_predict_timing).
        """
        msg = {
            "type": "predict",
            "obstacle_map": obstacle_map,
            "energy_map": energy_map,
            "robot_pos": [int(robot_pos[0]), int(robot_pos[1])],
            "goal_pos":  [int(goal_pos[0]),  int(goal_pos[1])],
        }
        if sequence is not None:
            msg["sequence"] = int(sequence) & 0xFFFFFFFF
        with self._lock:
            self._pending_messages.append(msg)
        self._has_work.set()
        self._activated.set()

    def on_predict_timing(
        self, callback: Callable[[int, int, int], None],
    ) -> None:
        """Register a callback fired once per completed predict round-trip.

        Called as ``callback(sequence, t_tcp_send_ns, t_tcp_recv_ns)`` when a
        predict that carried a `sequence` is answered by a `predict_result`.
        Both timestamps are time.monotonic_ns() on this host.  A predict that
        ends in `error` (no result) fires no timing callback.
        """
        with self._timing_lock:
            self._timing_cbs.append(callback)

    def on(self, msg_type: str, callback: Callable[[dict], None]) -> None:
        """Register a callback for incoming messages of *msg_type*.

        Multiple subscribers may register for the same type; all are called
        in registration order when a matching message arrives.
        """
        self._callbacks.setdefault(msg_type, []).append(callback)

    def close(self) -> None:
        """Shut down sender/receiver threads and close the socket."""
        self._running = False
        self._activated.set()  # unblock sender if waiting for first message
        self._has_work.set()   # unblock sender if waiting for work
        self._send_thread.join(timeout=2)
        if self._recv_thread is not None:
            self._recv_thread.join(timeout=2)
        self._disconnect()

    # -- background threads ---------------------------------------------------

    def _send_loop(self) -> None:
        """Drain queued messages and write them to the socket."""
        # Wait until the first message is queued before connecting.
        # This avoids DNS/connection attempts that can disrupt the
        # IP camera HTTP stream on Windows (LLMNR/NetBIOS broadcasts).
        self._activated.wait()
        if not self._running:
            return

        backoff = 1.0
        while self._running:
            # Ensure connected
            if self._sock is None:
                if self._connect():
                    backoff = 1.0
                else:
                    time.sleep(min(backoff, 10.0))
                    backoff = min(backoff * 2, 10.0)
                    continue

            # Wait for work
            self._has_work.wait(timeout=1.0)
            self._has_work.clear()

            # Snapshot and reset pending work
            with self._lock:
                pose = self._pending_pose
                outbound = self._pending_messages[:]
                self._pending_pose = None
                self._pending_messages.clear()

            # Send pose first (freshest position before any goal / predict)
            msgs: list[dict] = []
            if pose is not None:
                msgs.append(pose)
            msgs.extend(outbound)

            for msg in msgs:
                if not self._send_msg(msg):
                    # Connection lost — re-queue non-pose msgs (pose is stale, drop it)
                    with self._lock:
                        self._pending_messages = outbound + self._pending_messages
                    break

    def _recv_loop(self) -> None:
        """Read newline-delimited JSON from the server."""
        buf = ""
        while self._running and self._sock is not None:
            try:
                data = self._sock.recv(4096)
                if not data:
                    logger.warning("[bridge] server closed connection")
                    self._disconnect()
                    return
                buf += data.decode("utf-8", errors="replace")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    self._handle_incoming(line)
            except OSError:
                if self._running:
                    logger.warning("[bridge] recv error, disconnecting")
                    self._disconnect()
                return

    def _handle_incoming(self, line: str) -> None:
        recv_ns = time.monotonic_ns()
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("[bridge] malformed JSON: %s", line[:120])
            return
        msg_type = msg.get("type", "")

        # Close out predict round-trip timing.  A predict_result completes the
        # round trip; an error abandons it (no timing emitted).
        if msg_type == "predict_result":
            self._complete_predict_timing(msg.get("sequence"), recv_ns)
        elif msg_type == "error":
            with self._timing_lock:
                self._outstanding_predict = None

        callbacks = self._callbacks.get(msg_type)
        if not callbacks:
            logger.debug("[bridge] unhandled message type: %s", msg_type)
            return
        for cb in callbacks:
            try:
                cb(msg)
            except Exception:
                logger.exception("[bridge] callback error for %s", msg_type)

    # -- connection helpers ---------------------------------------------------

    def _connect(self) -> bool:
        try:
            sock = socket.create_connection(
                (self._cfg.host, self._cfg.port), timeout=3.0,
            )
            sock.settimeout(2.0)
            self._sock = sock
            logger.info("[bridge] connected to %s:%d",
                        self._cfg.host, self._cfg.port)
            # Ensure previous recv thread has exited before starting a new one
            if self._recv_thread is not None:
                self._recv_thread.join(timeout=1.0)
            self._recv_thread = threading.Thread(
                target=self._recv_loop, name="bridge-recv", daemon=True,
            )
            self._recv_thread.start()
            return True
        except OSError as e:
            logger.debug("[bridge] connect failed: %s", e)
            return False

    def _disconnect(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        # Drop any predict that was awaiting a reply on the lost connection; a
        # stray predict_result later (e.g. delayed on a reconnect) would otherwise
        # fire a timing callback against the previous send_ns and produce a
        # huge artifactual RTT — invariant: _outstanding_predict is only set
        # while a predict is awaiting a reply on the CURRENT connection.
        with self._timing_lock:
            self._outstanding_predict = None

    def _send_msg(self, msg: dict) -> bool:
        sock = self._sock
        if sock is None:
            return False
        try:
            payload = json.dumps(msg, separators=(",", ":")) + "\n"
            logger.debug("[bridge] >> %s", payload.rstrip())
            send_ns = time.monotonic_ns()
            sock.sendall(payload.encode("utf-8"))
            # Record send time for predict round-trip timing, captured as close
            # to the wire as possible.  Only predicts carrying a sequence are
            # tracked.  One predict is in flight at a time (state machine gated)
            # so a new predict overwrites any stale outstanding entry.
            if msg.get("type") == "predict" and "sequence" in msg:
                with self._timing_lock:
                    self._outstanding_predict = (int(msg["sequence"]), send_ns)
            return True
        except OSError as e:
            logger.warning("[bridge] send failed: %s", e)
            self._disconnect()
            return False

    def _complete_predict_timing(
        self, echoed_sequence: int | None, recv_ns: int,
    ) -> None:
        """Match a predict_result to its outstanding predict and fire callbacks.

        Prefers the response's echoed `sequence` (once the bridge forwards it);
        otherwise uses the single outstanding predict.  Fires no callback if
        there is nothing outstanding (e.g. a result for an untracked predict).
        """
        with self._timing_lock:
            outstanding = self._outstanding_predict
            self._outstanding_predict = None
            cbs = list(self._timing_cbs)
        if outstanding is None:
            return
        seq, send_ns = outstanding
        if echoed_sequence is not None:
            try:
                seq = int(echoed_sequence)
            except (TypeError, ValueError):
                pass
        for cb in cbs:
            try:
                cb(seq, send_ns, recv_ns)
            except Exception:
                logger.exception("[bridge] predict-timing callback error")
