"""
auto_driver.py - Autonomous planner loop state machine.

Composes injected collaborators (Planner, EnergySource, MapBuilder, LegalityRules,
AutoSessionLog).  Does not touch sockets, OpenCV, or the camera directly -
all such concerns are in collaborators.  The main loop calls tick() every
frame and forwards bridge events through on_* methods.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from auto_recorder import ActionEvent, AutoRecorder, NullAutoRecorder
from auto_session_log import AutoSessionLog
from coord_transform import (
    bridge_direction_to_tracker_delta,
    tracker_cell_to_bridge,
)
from directions import label as action_label
from energy_source import EnergySource
from grid import GridState
from legality import LegalityRule, LegalityVerdict
from map_builder import MapBuilder
from planner import Planner, PredictRequest, PredictResponse
from robot import RobotState
from scene import Scene


class AutoState(Enum):
    IDLE = "idle"
    READY_TO_PREDICT = "ready_to_predict"
    WAITING_PREDICT = "waiting_predict"
    WAITING_GOAL = "waiting_goal"
    WAITING_SETTLE = "waiting_settle"
    STOPPED_ERROR = "stopped_error"


@dataclass(frozen=True)
class AutoConfig:
    max_illegal_retries: int
    predict_cooldown_s: float
    predict_timeout_s: float
    experiment_tag: str
    settle_speed_threshold: float
    settle_angular_threshold: float
    settle_min_duration_s: float
    settle_timeout_s: float
    settle_freshness_window_s: float
    # Backstop on WAITING_GOAL — guards against nav-node hangs that swallow
    # the MoveToGrid action result. Without this the state machine wedges
    # forever; the orchestrator's wallclock guard is a second line of defense.
    goal_timeout_s: float = 60.0


@dataclass(frozen=True)
class AutoStatus:
    active: bool
    error: bool
    cycle: int
    retries: int
    goal_cell: tuple[int, int] | None
    last_label: str
    last_legal: bool | None
    state: str


LogFactory = Callable[..., AutoSessionLog]


class AutoDriver:
    """State machine that drives the robot toward the scene's goal cell."""

    def __init__(
        self,
        planner: Planner,
        energy: EnergySource,
        map_builder: MapBuilder,
        rules: list[LegalityRule],
        log_factory: LogFactory,
        cfg: AutoConfig,
        goal_layer: str = "goal",
        goal_sender: Callable[[float, float, RobotState], None] | None = None,
        cancel_sender: Callable[[], None] | None = None,
        pose_sender: Callable[[RobotState], None] | None = None,
        recorder: AutoRecorder | None = None,
    ) -> None:
        self._planner = planner
        self._energy = energy
        self._map_builder = map_builder
        self._rules = list(rules)
        self._log_factory = log_factory
        self._cfg = cfg
        self._goal_layer_name = goal_layer
        self._goal_sender = goal_sender
        self._cancel_sender = cancel_sender
        self._pose_sender = pose_sender
        # Optional hook to reset the position tracker at each trial start, so a
        # previous run's held pose (e.g. the goal cell the robot was carried
        # away from) can't bleed into the next run. Wired by main.py once the
        # RobotTracker exists; None in tests / when no tracker is in play.
        self._tracker_reset: Callable[[], None] | None = None
        self._recorder: AutoRecorder = recorder if recorder is not None else NullAutoRecorder()

        planner.on_result(self.on_predict_result)
        planner.on_error(self.on_planner_error)

        # Re-entrant lock guarding all state mutations.  Used so main-thread
        # tick()/start()/stop() and bridge-recv-thread callbacks can touch
        # _state, _retry, _cycle, _current_*, _last_robot_state, etc. without
        # racing.  RLock so private helpers can be called from within a
        # public method that already holds the lock.
        self._lock = threading.RLock()

        # Runtime state (reset by start()).
        self._state: AutoState = AutoState.IDLE
        self._log: AutoSessionLog | None = None
        self._goal_cell: tuple[int, int] | None = None
        self._cycle = 0
        self._retry = 0
        self._predict_sent_at: float = 0.0
        self._next_predict_not_before: float = 0.0
        self._last_robot_state: RobotState | None = None
        self._last_label: str = ""
        self._last_legal: bool | None = None

        # PPO "last cell" assist (set per-run by start()). When True, once the
        # robot is 8-adjacent to the goal the predict is still sent + logged
        # (so the planner's proposed action is on record) but the final step is
        # driven onto the goal cell instead of the proposed cell. Gated to PPO
        # runs by the caller. `_goal_assist_fired` latches True the first time
        # the override actually triggers in this run so the orchestrator can
        # flag the finish as assisted in metadata. Reset on every start().
        self._goal_assist: bool = False
        self._goal_assist_fired: bool = False

        # Per-run monotonic predict counter.  Stamped on every outbound predict
        # (including retries) so the JSONL record, the bridge timing capture,
        # and (once the bridge echoes it) PlannerMetrics all join on it.
        # `_last_sequence` is the id of the in-flight predict, used to tag the
        # matching predict_result log line.
        self._sequence = 0
        self._last_sequence = 0

        # Session-end subscribers (orchestrator hooks in here to advance its
        # state machine when a run finishes).  Fired AFTER stop() completes,
        # with (outcome, reason) where outcome ∈ {"success","failure","aborted"}.
        self._session_end_cbs: list[Callable[[str, str], None]] = []
        self._last_outcome: str = ""
        self._last_reason: str = ""

        # When the orchestrator hands us a log via log_factory_override the log
        # is BORROWED — the orchestrator owns its lifecycle, so stop() must NOT
        # close it (the run_end JSONL record is written by RunRecorder after
        # stop fires the session-end callback).  Default path (no override)
        # stays as-is: we created the log, we close it.
        self._log_borrowed: bool = False

        # Wall-clock when the latest goal was sent; used by the WAITING_GOAL
        # timeout to detect a hung nav node.
        self._goal_sent_at: float = 0.0

        # Settle-gate bookkeeping.  Reset on every entry to WAITING_SETTLE so
        # the first frame in the new window can't be matched against a stale
        # heading reference or frame signature from the previous cycle.
        self._settle_started_at: float = 0.0
        self._below_threshold_since: float | None = None
        self._prev_settle_heading: float | None = None
        self._prev_settle_heading_ts: float = 0.0
        self._last_frame_signature: int | None = None
        self._last_frame_change_ts: float = 0.0

        # Latest live references.  Refreshed by tick() every frame so async
        # callbacks (on_predict_result) can use current scene/grid state
        # without the main loop having to thread them through.
        self._current_scene: Scene | None = None
        self._current_grid: GridState | None = None

    # -- public control --------------------------------------------------------

    def set_tracker_reset(self, cb: Callable[[], None]) -> None:
        """Register a callback that resets the position tracker.

        Invoked by start() so each trial begins with no carried-over pose
        history. Wired late (after the RobotTracker is constructed) rather than
        via the constructor because the driver is built before the tracker.
        """
        with self._lock:
            self._tracker_reset = cb

    def start(
        self,
        scene: Scene,
        grid: GridState,
        robot_state: RobotState | None,
        corner_marker_ids: list[int] | None = None,
        *,
        start_sequence: int = 0,
        log_factory_override: LogFactory | None = None,
        goal_assist: bool = False,
    ) -> tuple[bool, str]:
        """Validate preconditions and open a new session log.

        Returns (ok, reason).  Reason is human-readable on failure; empty on
        success.  The first predict is sent on the next tick() call.

        `start_sequence` lets a wrapping orchestrator pre-issue warmup predicts
        (using sequences 1..N) and have auto_driver continue from N+1 so the
        join keys remain monotonic across warmup + run.

        `log_factory_override` lets the orchestrator point this session's log
        into runs/<run_id>/predict_log.jsonl. When None, the constructor's
        log_factory is used (the existing auto_logs/auto_<ts>.jsonl path).

        `goal_assist` enables the PPO "last cell" assist for this run (see the
        `_goal_assist` field comment). The orchestrator passes True only for
        ppo cells when orchestrator.ppo_goal_assist is set.
        """
        with self._lock:
            if self._state != AutoState.IDLE:
                return False, "auto mode already active"
            if grid is None:
                return False, "grid not locked"
            if robot_state is None:
                return False, "robot not visible"
            if self._goal_layer_name not in scene:
                return False, f"no '{self._goal_layer_name}' layer registered"
            goal_cells = list(scene.get(self._goal_layer_name).marked_cells())
            if len(goal_cells) != 1:
                return False, f"need exactly 1 goal cell, have {len(goal_cells)}"
            goal_col, goal_row = goal_cells[0]

            # Refuse if goal sits on a blocking layer.
            obs_mask = self._map_builder.build_obstacle_map(scene)
            if obs_mask[goal_row, goal_col]:
                return False, "goal cell is on an obstacle"

            # Open log.  goal_cell is stored and logged as (col, row) so it
            # matches Scene.marked_cells(), cycle_start.robot_cell, and
            # minimap_state.goal_cell — and so the downstream unpacking
            # `goal_col, goal_row = self._goal_cell` is correct.
            factory = log_factory_override or self._log_factory
            self._log = factory()
            self._log_borrowed = log_factory_override is not None
            self._log.session_start(
                experiment_tag=self._cfg.experiment_tag,
                grid_dims=(grid.rows, grid.cols),
                corner_marker_ids=list(corner_marker_ids or []),
                goal_cell=(goal_col, goal_row),
                retry_max=self._cfg.max_illegal_retries,
                planner_name=getattr(self._planner, "name", type(self._planner).__name__),
                energy_source_name=getattr(self._energy, "name", type(self._energy).__name__),
                blocking_layers=self._map_builder.blocking_layers,
            )

            # Clear any pose history carried over from a prior trial before the
            # first tick samples a new pose. Best-effort: a reset failure must
            # not block starting the run.
            if self._tracker_reset is not None:
                try:
                    self._tracker_reset()
                except Exception:
                    pass

            self._goal_cell = (goal_col, goal_row)
            self._current_scene = scene
            self._current_grid = grid
            self._last_robot_state = robot_state
            self._cycle = 1          # cycle = move number; incremented on each goal_sent
            self._retry = 0
            self._last_label = ""
            self._last_legal = None
            # Resume the predict counter past any pre-run warmups the orchestrator
            # issued through bridge.send_predict directly (sequences 1..N).
            self._sequence = int(start_sequence)
            self._last_sequence = int(start_sequence)
            self._predict_sent_at = 0.0
            self._next_predict_not_before = 0.0
            self._goal_assist = bool(goal_assist)
            self._goal_assist_fired = False
            self._state = AutoState.READY_TO_PREDICT
            self._recorder.session_start()
            self._emit_minimap_state()        # initial snapshot before any predicts
            return True, ""

    def stop(self, reason: str, is_error: bool = True) -> None:
        """Terminate the session and (if goal in flight) cancel it.

        `is_error=True` (the default, for aborts) leaves the driver in the
        STOPPED_ERROR state so the HUD can show a red banner.  Clean exits
        (user toggle, app shutdown, robot reached goal) should pass
        `is_error=False` so the driver returns to IDLE immediately.
        """
        with self._lock:
            if self._state == AutoState.IDLE:
                return
            if self._state == AutoState.WAITING_GOAL and self._cancel_sender is not None:
                try:
                    self._cancel_sender()
                except Exception:
                    pass
            if self._log is not None:
                self._log.session_stop(reason=reason, cycles=self._cycle)
                # When the log is borrowed from the orchestrator, RunRecorder
                # owns the file's lifecycle — closing it here would silently
                # drop the orchestrator's subsequent run_end record (F9).
                if not self._log_borrowed:
                    self._log.close()
                self._log = None
                self._log_borrowed = False
            # Outcome derives from the path that led here, not from the caller's
            # intent: reaching the goal (is_error=False AND reason mentions it)
            # is "success"; any error abort is "failure"; user/app stops are
            # "aborted". Orchestrator wraps this with its own outcome (e.g. it
            # may upgrade to "aborted" when SSH bag stop fails).
            if is_error:
                outcome = "failure"
            elif "goal" in reason:
                outcome = "success"
            else:
                outcome = "aborted"
            self._last_outcome = outcome
            self._last_reason = reason
            cbs = list(self._session_end_cbs)
            self._state = AutoState.STOPPED_ERROR if is_error else AutoState.IDLE
        # Fire callbacks OUTSIDE the lock — the orchestrator hook is likely to
        # re-enter auto_driver state (e.g., to read hud_status) and we don't
        # want callback work serialized under our internal mutex.
        for cb in cbs:
            try:
                cb(outcome, reason)
            except Exception:
                pass

    def on_session_end(self, cb: Callable[[str, str], None]) -> None:
        """Register a session-end subscriber. Fired as `cb(outcome, reason)`
        after stop() completes. Outcome ∈ {"success","failure","aborted"}.
        """
        with self._lock:
            self._session_end_cbs.append(cb)

    def last_outcome(self) -> tuple[str, str]:
        """Return (outcome, reason) of the most recent session, or ("","")."""
        with self._lock:
            return self._last_outcome, self._last_reason

    def goal_assist_fired(self) -> bool:
        """True if the PPO last-cell assist overrode at least one step this
        run. Latched until the next start(); read by the orchestrator at run
        end to flag the metadata as an assisted finish."""
        with self._lock:
            return self._goal_assist_fired

    def clear_error(self) -> None:
        """Move from STOPPED_ERROR back to IDLE (e.g. on user key press)."""
        with self._lock:
            if self._state == AutoState.STOPPED_ERROR:
                self._state = AutoState.IDLE

    def is_active(self) -> bool:
        with self._lock:
            return self._state not in (AutoState.IDLE, AutoState.STOPPED_ERROR)

    def hud_status(self) -> AutoStatus:
        with self._lock:
            return AutoStatus(
                active=self._state not in (AutoState.IDLE, AutoState.STOPPED_ERROR),
                error=self._state == AutoState.STOPPED_ERROR,
                cycle=self._cycle,
                retries=self._retry,
                goal_cell=self._goal_cell,
                last_label=self._last_label,
                last_legal=self._last_legal,
                state=self._state.value,
            )

    # -- per-frame tick --------------------------------------------------------

    def tick(
        self,
        frame: np.ndarray,
        robot_state: RobotState | None,
        grid: GridState | None,
        scene: Scene,
        now_ts: float | None = None,
    ) -> None:
        """Drive the state machine forward.  No-op when IDLE / STOPPED_ERROR.

        `now_ts` is optional and intended for tests that freeze time; in
        production callers can pass `time.monotonic()` or omit it.  When
        omitted, `time.monotonic()` is used.
        """
        with self._lock:
            if self._state in (AutoState.IDLE, AutoState.STOPPED_ERROR):
                return
            now_ts = time.monotonic() if now_ts is None else now_ts

            # Grid / robot disappearing mid-run is a hard stop.
            if grid is None:
                self._abort("grid unlocked")
                return
            if robot_state is None:
                # Don't stop immediately - the robot may reappear next frame.
                # But don't try to predict without a pose either.
                return

            self._last_robot_state = robot_state
            self._current_scene = scene
            self._current_grid = grid

            # Termination: robot is on the goal cell.
            if self._goal_cell is not None:
                g_col, g_row = self._goal_cell
                if (robot_state.pose.cell_col == g_col
                        and robot_state.pose.cell_row == g_row):
                    self.stop("robot reached goal", is_error=False)
                    return

            # Settle gate.  After a successful goal_result we wait until
            # the tracker observes the robot as not-moving for a continuous
            # window before sampling pose for the next predict.  Three
            # signals must all be quiet: linear speed, heading rate, and
            # frame freshness (a stalled video stream gives spurious
            # zero-motion readings, so we won't treat it as settled).
            if self._state == AutoState.WAITING_SETTLE:
                # Frame freshness: track when the signature last changed.
                sig = self._frame_signature(frame)
                if self._last_frame_signature is None or sig != self._last_frame_signature:
                    self._last_frame_signature = sig
                    self._last_frame_change_ts = now_ts
                fresh = (
                    now_ts - self._last_frame_change_ts
                    <= self._cfg.settle_freshness_window_s
                )

                # Linear speed.  None on the very first observation after
                # the marker reappears — treat as not-yet-known.
                speed = robot_state.speed
                linear_ok = speed is not None and speed < self._cfg.settle_speed_threshold

                # Angular speed via successive smoothed headings, with the
                # standard ±180° wrap-around fix.
                angular_ok = False
                cur_h = robot_state.heading_deg
                if self._prev_settle_heading is not None and self._prev_settle_heading_ts > 0.0:
                    dt = now_ts - self._prev_settle_heading_ts
                    if dt > 0:
                        delta = ((cur_h - self._prev_settle_heading + 540.0) % 360.0) - 180.0
                        ang_rate = abs(delta) / dt
                        angular_ok = ang_rate < self._cfg.settle_angular_threshold
                self._prev_settle_heading = cur_h
                self._prev_settle_heading_ts = now_ts

                if fresh and linear_ok and angular_ok:
                    if self._below_threshold_since is None:
                        self._below_threshold_since = now_ts
                    if now_ts - self._below_threshold_since >= self._cfg.settle_min_duration_s:
                        self._state = AutoState.READY_TO_PREDICT
                        self._next_predict_not_before = now_ts + self._cfg.predict_cooldown_s
                        return
                else:
                    self._below_threshold_since = None

                # Bail-out: never block the experiment forever.  Log and
                # proceed; the predict will use whatever pose we have.
                if now_ts - self._settle_started_at > self._cfg.settle_timeout_s:
                    if self._log is not None:
                        self._log.error(self._cycle, "settle timeout — proceeding")
                    self._state = AutoState.READY_TO_PREDICT
                    self._next_predict_not_before = now_ts + self._cfg.predict_cooldown_s
                return

            # Predict timeout.  Re-sending doesn't help if the service has
            # stalled, and a late response from the original predict would
            # race any re-send.  Stop cleanly; operator can press `p` to
            # retry after diagnosing.
            if self._state == AutoState.WAITING_PREDICT:
                if now_ts - self._predict_sent_at > self._cfg.predict_timeout_s:
                    if self._log is not None:
                        self._log.error(self._cycle, "predict timeout")
                    self._abort("predict timeout")
                    return

            # Goal timeout.  WAITING_GOAL has no built-in deadline; a hung nav
            # node or action server that never publishes a result would wedge
            # the run forever.  Same shape as the predict_timeout block above.
            if self._state == AutoState.WAITING_GOAL and self._goal_sent_at > 0.0:
                if now_ts - self._goal_sent_at > self._cfg.goal_timeout_s:
                    if self._log is not None:
                        self._log.error(self._cycle, "goal timeout")
                    self._abort("goal timeout")
                    return

            # Fire the next predict when the state machine is ready.
            if (self._state == AutoState.READY_TO_PREDICT
                    and now_ts >= self._next_predict_not_before):
                # Emit cycle_start exactly once per move — on the first
                # predict attempt of the current cycle (retry == 0).
                if self._retry == 0 and self._log is not None:
                    self._log.cycle_start(
                        cycle=self._cycle,
                        robot_cell=(
                            robot_state.pose.cell_col,
                            robot_state.pose.cell_row,
                        ),
                        robot_heading_deg=robot_state.heading_deg,
                    )
                self._send_predict(frame, robot_state, grid, scene, now_ts)

    # -- planner / bridge callbacks -------------------------------------------

    def on_predict_result(self, resp: PredictResponse) -> None:
        with self._lock:
            if self._state != AutoState.WAITING_PREDICT:
                # Stale response (e.g. after an abort).  Ignore.
                return
            if self._goal_cell is None or self._last_robot_state is None:
                return

            self._last_label = action_label(resp.action)
            if self._log is not None:
                self._log.predict_result(
                    cycle=self._cycle,
                    retry=self._retry,
                    action=resp.action,
                    direction=resp.direction,
                    label=self._last_label,
                    sequence=self._last_sequence,
                )

            drow, dcol = bridge_direction_to_tracker_delta(*resp.direction)
            cur_row = self._last_robot_state.pose.cell_row
            cur_col = self._last_robot_state.pose.cell_col
            next_row = cur_row + drow
            next_col = cur_col + dcol

            verdict = self._check_legality(next_row, next_col)
            self._last_legal = verdict.legal
            if self._log is not None:
                self._log.legality_check(
                    cycle=self._cycle,
                    next_cell_tracker=(next_row, next_col),
                    legal=verdict.legal,
                    reason=verdict.reason,
                )
            self._recorder.predict_result(ActionEvent(
                cycle=self._cycle,
                retry=self._retry,
                robot_col=cur_col,
                robot_row=cur_row,
                action=resp.action,
                direction=(int(resp.direction[0]), int(resp.direction[1])),
                label=self._last_label,
                legal=verdict.legal,
                reason=verdict.reason,
            ))
            self._emit_minimap_state()

            # PPO last-cell assist.  The predict above is already sent + logged
            # (so the policy's proposed action is on record); here we override
            # the *movement* when the robot is one cell from the goal, driving
            # the final step onto the goal cell regardless of what the policy
            # proposed or whether that proposal was legal.  Gated to PPO runs
            # by start(goal_assist=...).
            if self._goal_assist and self._is_adjacent_to_goal(cur_row, cur_col):
                g_col, g_row = self._goal_cell
                self._goal_assist_fired = True
                if self._log is not None:
                    self._log.marker(
                        "goal_assist",
                        note=(
                            f"cycle={self._cycle} robot=({cur_col},{cur_row}) "
                            f"proposed={self._last_label} legal={verdict.legal}; "
                            f"overriding final step to goal cell ({g_col},{g_row})"
                        ),
                    )
                self._send_goal(g_row, g_col)
                return

            if verdict.legal:
                self._send_goal(next_row, next_col)
            else:
                self._retry += 1
                if self._retry >= self._cfg.max_illegal_retries:
                    self._abort("max illegal retries reached")
                    return
                self._state = AutoState.READY_TO_PREDICT
                self._next_predict_not_before = (
                    time.monotonic() + self._cfg.predict_cooldown_s
                )

    def on_planner_error(self, message: str) -> None:
        with self._lock:
            if self._state not in (AutoState.WAITING_PREDICT, AutoState.WAITING_GOAL):
                return
            if self._log is not None:
                self._log.error(self._cycle, message)
            # Treat errors as illegal attempts for retry accounting.
            self._retry += 1
            if self._retry >= self._cfg.max_illegal_retries:
                self._abort(f"planner error: {message}")
                return
            self._state = AutoState.READY_TO_PREDICT
            self._next_predict_not_before = (
                time.monotonic() + self._cfg.predict_cooldown_s
            )

    def on_goal_feedback(self, msg: dict) -> None:
        with self._lock:
            if self._state != AutoState.WAITING_GOAL or self._log is None:
                return
            self._log.goal_feedback(
                cycle=self._cycle,
                phase=str(msg.get("phase", "")),
                distance=float(msg.get("distance", 0.0)),
                heading_error=float(msg.get("heading_error", 0.0)),
            )

    def on_goal_result(self, msg: dict) -> None:
        with self._lock:
            if self._state != AutoState.WAITING_GOAL:
                return
            success = bool(msg.get("success", False))
            message = str(msg.get("message", ""))
            if self._log is not None:
                self._log.goal_result(
                    cycle=self._cycle, success=success, message=message,
                )
            if not success:
                self._abort(f"goal rejected: {message}")
                return
            # Successful step — bump the cycle counter so the next predict
            # belongs to the next move.  Termination check runs in tick()
            # before any cycle_start log or predict fires, so reaching the
            # goal on this step doesn't emit a stray cycle_start.
            self._cycle += 1
            self._retry = 0
            # Enter the settle gate before the next predict.  The video
            # stream lags reality, so the tracker keeps reporting motion
            # for some frames after the bridge says we're done — sampling
            # pose now would feed a moving robot into the planner.
            now = time.monotonic()
            self._state = AutoState.WAITING_SETTLE
            self._settle_started_at = now
            self._below_threshold_since = None
            self._prev_settle_heading = None
            self._prev_settle_heading_ts = 0.0
            self._last_frame_signature = None
            self._last_frame_change_ts = now

    # -- internals -------------------------------------------------------------

    def _send_predict(
        self,
        frame: np.ndarray,
        robot_state: RobotState,
        grid: GridState,
        scene: Scene,
        now_ts: float,
    ) -> None:
        if self._goal_cell is None:
            return
        goal_col, goal_row = self._goal_cell

        # New id for this predict (retries included).
        self._sequence += 1
        seq = self._sequence
        self._last_sequence = seq

        payload = self._map_builder.as_predict_payload(
            scene=scene,
            energy=self._energy,
            frame=frame,
            grid=grid,
            robot_row=robot_state.pose.cell_row,
            robot_col=robot_state.pose.cell_col,
            goal_row=goal_row,
            goal_col=goal_col,
        )
        self._recorder.predict_sent(payload.energy_map_tracker)

        if self._log is not None:
            self._log.predict_sent(
                cycle=self._cycle,
                retry=self._retry,
                bridge_robot_pos=payload.robot_pos,
                bridge_goal_pos=payload.goal_pos,
                obstacle_map=payload.obstacle_map,
                energy_map=payload.energy_map,
                sequence=seq,
            )

        req: PredictRequest = {
            "obstacle_map": payload.obstacle_map,
            "energy_map":   payload.energy_map,
            "robot_pos":    payload.robot_pos,
            "goal_pos":     payload.goal_pos,
        }
        # Push a fresh pose to the bridge before the predict so the nav node's
        # self-estimate matches what we just discretized into robot_pos.
        # Mirrors the pose-then-goal pattern used by send_goal in manual mode.
        if self._pose_sender is not None:
            try:
                self._pose_sender(robot_state)
            except Exception as e:
                self._abort(f"pose_sender raised: {e}")
                return
        try:
            self._planner.request(req, sequence=seq)
        except Exception as e:
            self._abort(f"planner.request raised: {e}")
            return

        self._predict_sent_at = now_ts
        self._state = AutoState.WAITING_PREDICT

    def _send_goal(self, next_row: int, next_col: int) -> None:
        if self._goal_sender is None or self._last_robot_state is None:
            self._abort("no goal_sender configured")
            return
        # Send first; log on success.  Logging after the send avoids claiming
        # a goal was sent when the sender raised.
        # Add 0.5 to target the cell center (matches the manual click path).
        try:
            self._goal_sender(next_col + 0.5, next_row + 0.5, self._last_robot_state)
        except Exception as e:
            if self._log is not None:
                self._log.error(self._cycle, f"goal_sender raised: {e}")
            self._abort(f"goal_sender raised: {e}")
            return
        rows = self._current_grid.rows if self._current_grid is not None else 0
        target_bridge = tracker_cell_to_bridge(next_row, next_col, rows)
        if self._log is not None:
            self._log.goal_sent(
                cycle=self._cycle,
                target_tracker=(next_row, next_col),
                target_bridge=target_bridge,
            )
        self._retry = 0
        self._goal_sent_at = time.monotonic()
        self._state = AutoState.WAITING_GOAL

    def _emit_minimap_state(self) -> None:
        """Write a minimap_state log line capturing scene + history + energy.

        Called from start() (initial empty state) and after every
        predict_result is processed.  Lives here because the driver is the
        only component that knows scene, recorder, and last robot pose at
        the same time.  Best-effort: silently skip if the log isn't open
        or the scene isn't yet cached.
        """
        if self._log is None or self._current_scene is None:
            return

        scene = self._current_scene
        snap = self._recorder.snapshot()

        history: list[dict] = []
        energy_list: list[list[float]] | None = None
        if snap is not None:
            history = [
                {
                    "cycle": ev.cycle, "retry": ev.retry,
                    "robot_col": ev.robot_col, "robot_row": ev.robot_row,
                    "action": ev.action,
                    "direction": list(ev.direction),
                    "label": ev.label,
                    "legal": ev.legal,
                    "reason": ev.reason,
                }
                for ev in snap.action_history
            ]
            if snap.last_energy_map is not None:
                energy_list = snap.last_energy_map.astype(float).tolist()

        obstacles: list[tuple[int, int]] = []
        if "obstacles" in scene:
            obstacles = [(int(c), int(r)) for c, r in scene.get("obstacles").marked_cells()]

        goal_cell: tuple[int, int] | None = None
        if self._goal_layer_name in scene:
            goal_marked = list(scene.get(self._goal_layer_name).marked_cells())
            if len(goal_marked) == 1:
                gc, gr = goal_marked[0]
                goal_cell = (int(gc), int(gr))

        if self._last_robot_state is not None:
            robot_cell = (
                int(self._last_robot_state.pose.cell_col),
                int(self._last_robot_state.pose.cell_row),
            )
        else:
            robot_cell = (-1, -1)

        self._log.minimap_state(
            cycle=self._cycle,
            retry=self._retry,
            robot_cell=robot_cell,
            goal_cell=goal_cell,
            obstacle_cells=obstacles,
            action_history=history,
            energy_map=energy_list,
        )

    def _is_adjacent_to_goal(self, cur_row: int, cur_col: int) -> bool:
        """True when the robot's cell is one of the goal's 8 neighbours.

        Chebyshev distance of exactly 1 — diagonals count (the robot moves on
        an 8-connected grid).  Distance 0 (already on the goal) returns False;
        that case is handled earlier by tick()'s termination check.
        """
        if self._goal_cell is None:
            return False
        g_col, g_row = self._goal_cell
        return max(abs(cur_row - g_row), abs(cur_col - g_col)) == 1

    def _check_legality(self, next_row: int, next_col: int) -> LegalityVerdict:
        scene = self._current_scene
        grid = self._current_grid
        if scene is None or grid is None:
            return LegalityVerdict(False, "no_scene_or_grid_cached")
        for rule in self._rules:
            v = rule.check(next_row, next_col, scene, grid)
            if not v.legal:
                return v
        return LegalityVerdict(True, "")

    def _abort(self, reason: str) -> None:
        self.stop(reason)

    @staticmethod
    def _frame_signature(frame: np.ndarray | None) -> int:
        # Sparse-stride pixel sum — robust to the IP camera replaying an
        # identical buffered frame (apparent zero motion masquerading as
        # a settled robot).  Pure numpy: keeps auto_driver free of cv2.
        if frame is None or frame.size == 0:
            return 0
        return int(frame[::64, ::64].sum())
