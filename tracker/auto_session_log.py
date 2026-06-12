"""
auto_session_log.py - JSONL event logger owning the auto-session schema.

One file per auto-driver session.  Keeping the schema in its own module
means AutoDriver just calls typed methods; schema evolution doesn't ripple.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any


class AutoSessionLog:
    """Writes newline-delimited JSON events to a session-scoped file.

    Thread-safe: `_emit` holds an internal lock around the write, so the
    main thread and the bridge recv thread can log concurrently without
    interleaving JSON lines.
    """

    def __init__(self, path: Path, include_maps: bool = True) -> None:
        self._path = path
        self._include_maps = include_maps
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", encoding="utf-8")
        self._write_lock = threading.Lock()

    @classmethod
    def open(
        cls,
        log_dir: Path,
        include_maps: bool = True,
        prefix: str = "auto",
    ) -> "AutoSessionLog":
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return cls(log_dir / f"{prefix}_{ts}.jsonl", include_maps=include_maps)

    @property
    def path(self) -> Path:
        return self._path

    # -- events ----------------------------------------------------------------

    def session_start(
        self,
        experiment_tag: str,
        grid_dims: tuple[int, int],
        corner_marker_ids: list[int],
        goal_cell: tuple[int, int],
        retry_max: int,
        planner_name: str,
        energy_source_name: str,
        blocking_layers: list[str],
    ) -> None:
        self._emit(
            "session_start",
            experiment_tag=experiment_tag,
            grid={"rows": grid_dims[0], "cols": grid_dims[1]},
            corner_marker_ids=list(corner_marker_ids),
            goal_cell=list(goal_cell),
            retry_max=retry_max,
            planner=planner_name,
            energy_source=energy_source_name,
            blocking_layers=list(blocking_layers),
        )

    def cycle_start(
        self,
        cycle: int,
        robot_cell: tuple[int, int],
        robot_heading_deg: float,
    ) -> None:
        self._emit(
            "cycle_start",
            cycle=cycle,
            robot_cell=list(robot_cell),
            robot_heading_deg=round(float(robot_heading_deg), 3),
        )

    def predict_sent(
        self,
        cycle: int,
        retry: int,
        bridge_robot_pos: tuple[int, int],
        bridge_goal_pos: tuple[int, int],
        obstacle_map: list[list[float]],
        energy_map: list[list[float]],
        sequence: int | None = None,
    ) -> None:
        fields: dict[str, Any] = {
            "cycle": cycle,
            "retry": retry,
            "bridge_robot_pos": list(bridge_robot_pos),
            "bridge_goal_pos": list(bridge_goal_pos),
        }
        # `sequence` joins this record to PredictTcpTiming / BridgePredictTiming
        # / PlannerMetrics in the bags. Analysis pulls the per-call energy +
        # obstacle maps from here (they are dynamic, not snapshotted to .npy).
        if sequence is not None:
            fields["sequence"] = int(sequence)
        if self._include_maps:
            fields["obstacle_map"] = obstacle_map
            fields["energy_map"] = energy_map
        self._emit("predict_sent", **fields)

    def predict_result(
        self,
        cycle: int,
        retry: int,
        action: int,
        direction: tuple[int, int],
        label: str,
        sequence: int | None = None,
    ) -> None:
        fields: dict[str, Any] = {
            "cycle": cycle,
            "retry": retry,
            "action": action,
            "direction": list(direction),
            "label": label,
        }
        # direction_x / direction_y mirror the planner's echoed step so the
        # analysis octile path length can be derived from the JSONL without
        # PlannerMetrics (which omits the direction).
        if sequence is not None:
            fields["sequence"] = int(sequence)
            fields["direction_x"] = int(direction[0])
            fields["direction_y"] = int(direction[1])
        self._emit("predict_result", **fields)

    def legality_check(
        self,
        cycle: int,
        next_cell_tracker: tuple[int, int],
        legal: bool,
        reason: str,
    ) -> None:
        # Dispatch legal vs. illegal to distinct event names so researcher
        # log-parsers can grep `"event":"illegal_action"` directly.
        event = "legality_check" if legal else "illegal_action"
        self._emit(
            event,
            cycle=cycle,
            next_cell_tracker=list(next_cell_tracker),
            legal=legal,
            reason=reason,
        )

    def goal_sent(
        self,
        cycle: int,
        target_tracker: tuple[int, int],
        target_bridge: tuple[int, int],
    ) -> None:
        self._emit(
            "goal_sent",
            cycle=cycle,
            target_tracker=list(target_tracker),
            target_bridge=list(target_bridge),
        )

    def goal_feedback(
        self,
        cycle: int,
        phase: str,
        distance: float,
        heading_error: float,
    ) -> None:
        self._emit(
            "goal_feedback",
            cycle=cycle,
            phase=phase,
            distance=round(float(distance), 4),
            heading_error=round(float(heading_error), 4),
        )

    def goal_result(
        self,
        cycle: int,
        success: bool,
        message: str,
    ) -> None:
        self._emit("goal_result", cycle=cycle, success=bool(success), message=message)

    def error(self, cycle: int, message: str) -> None:
        self._emit("error", cycle=cycle, message=message)

    def session_stop(self, reason: str, cycles: int) -> None:
        self._emit("session_stop", reason=reason, cycles=cycles)

    def minimap_state(
        self,
        cycle: int,
        retry: int,
        robot_cell: tuple[int, int],
        goal_cell: tuple[int, int] | None,
        obstacle_cells: list[tuple[int, int]],
        action_history: list[dict],
        energy_map: list[list[float]] | None,
    ) -> None:
        """Snapshot of what the live minimap shows at this point in the run.

        Emitted at session start (initial empty state) and after every
        predict_result.  Cells are tracker-frame (col, row) tuples.  The
        verbose energy_map field is only included when AUTO_LOG_MAPS is set
        (same gate as predict_sent).
        """
        fields: dict[str, Any] = {
            "cycle": cycle,
            "retry": retry,
            "robot_cell": list(robot_cell),
            "goal_cell": list(goal_cell) if goal_cell is not None else None,
            "obstacle_cells": [list(c) for c in obstacle_cells],
            "action_history": list(action_history),
        }
        if self._include_maps and energy_map is not None:
            fields["energy_map"] = energy_map
        self._emit("minimap_state", **fields)

    # -- run-lifecycle markers -------------------------------------------------
    #
    # These mirror the bridge `experiment_event` envelope so the JSONL sidecar
    # carries the same run-window boundaries that get forwarded onto
    # /experiment/events.  Analysis prefers the bridge-host-stamped events in
    # the bag for the authoritative window; these are the orchestrator-side
    # copy for cross-checking and for runs analyzed from JSONL alone.

    def run_start(self, run_id: str, payload: dict[str, Any]) -> None:
        self._emit("run_start", run_id=run_id, payload=dict(payload))

    def run_end(self, run_id: str, outcome: str) -> None:
        self._emit("run_end", run_id=run_id, outcome=outcome)

    def marker(self, tag: str, note: str = "") -> None:
        self._emit("marker", tag=tag, note=note)

    def bag_capped(self, host: str) -> None:
        self._emit("bag_capped", host=host)

    # -- lifecycle -------------------------------------------------------------

    def close(self) -> None:
        with self._write_lock:
            if not self._fh.closed:
                self._fh.flush()
                self._fh.close()

    # -- internals -------------------------------------------------------------

    def _emit(self, event: str, **fields: Any) -> None:
        record: dict[str, Any] = {
            "t": datetime.now().isoformat(timespec="milliseconds"),
            # Monotonic clock for ordering / single-host deltas; wall-clock `t`
            # is a human-readable identifier only (laptop is not chrony-synced).
            "mono_ns": time.monotonic_ns(),
            "event": event,
        }
        record.update(fields)
        line = json.dumps(record, separators=(",", ":")) + "\n"
        with self._write_lock:
            if self._fh.closed:
                return
            self._fh.write(line)
            self._fh.flush()
