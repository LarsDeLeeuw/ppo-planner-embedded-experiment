"""orchestrator.run_recorder - Per-run sink: orchestrator.mcap + predict_log.jsonl
+ metadata.yaml + obstacle.npy snapshot, plus the bridge.on_predict_timing
wiring that routes round-trip timings into both sinks.

This is THE piece of "live wiring" the previous session deferred. It owns the
laptop-side capture for one run; the SSH bag side (robot/desktop bags) is
owned by BagSet. The orchestrator state machine instantiates one RunRecorder
per run and disposes it on stop.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import yaml

from auto_session_log import AutoSessionLog
from coord_transform import tracker_to_ros2
from orchestrator.state import RunContext
from orchestrator_bag import OrchestratorBag

logger = logging.getLogger(__name__)


class RunRecorder:
    """Per-run sink. Open at run start, close at run end. Single-use.

    Flow:
        rec = RunRecorder(ctx)
        rec.open()
        rec.snapshot_obstacle(obstacle_yx)       # numpy array, bridge frame, y-up
        # ...orchestrator wires bridge.on_predict_timing -> rec.record_predict_timing
        # ...orchestrator calls rec.mark_warmup_inference(seq, rtt_us) for each warmup
        rec.mark_run_start(payload)              # before auto.start()
        # ...auto driver runs, logging into rec.log_factory()
        rec.mark_run_end(outcome)                # after auto stops
        rec.write_metadata(extra_fields)
        rec.close()
    """

    def __init__(self, ctx: RunContext, *, include_maps: bool = True,
                 grid_rows: int = 10, cell_size_m: float = 0.30) -> None:
        self.ctx = ctx
        self._include_maps = include_maps
        # grid_rows is needed to convert tracker-frame goal_cell/start_cell to
        # bridge frame (origin lower, y bottom-up) for the on-disk metadata —
        # which is the convention obstacle.npy and /grid_nav_node/grid_pose use,
        # so summary.png trajectory markers line up.
        self._grid_rows = int(grid_rows)
        # Physical edge length of one grid cell (m). Written to metadata.yaml
        # so the analysis pipeline can scale /odom paths into grid units
        # without guessing.
        self._cell_size_m = float(cell_size_m)
        self._opened = False
        self._closed = False
        self._bag: OrchestratorBag | None = None
        self._log: AutoSessionLog | None = None
        # 30 Hz tracker-side pose log, gated by mark_run_start / mark_run_end so
        # only in-run samples land on disk. The bridge republishes
        # /grid_nav_node/grid_pose at its own rate even when the laptop sends
        # nothing new, so the bag stream is useless as a true trajectory —
        # this file is the authoritative trajectory source.
        self._pose_file: TextIO | None = None
        self._record_pose_enabled: bool = False
        self._t_run_start_ns: int | None = None       # laptop monotonic_ns
        self._t_run_end_ns: int | None = None
        self.warmup_rtt_us: list[int] = []
        self.events: list[dict[str, Any]] = []        # for cross-checking + diagnostic
        self._battery_band_violation: str | None = None
        # Wall-clock when mark_run_end actually fired (vs. the recorder's
        # construction time, which is what ctx.started_at captures).  Lets
        # pre-run-start aborts emit zero-duration sentinels instead of
        # misleading 1-2s "runs" in metadata.yaml.
        self._ended_at_wall: str | None = None

    # -- open / close --------------------------------------------------------

    def open(self) -> None:
        if self._opened:
            raise RuntimeError("RunRecorder already open")
        self.ctx.run_dir.mkdir(parents=True, exist_ok=True)
        (self.ctx.run_dir / "maps").mkdir(exist_ok=True)
        self._bag = OrchestratorBag(self.ctx.run_dir / "orchestrator.mcap",
                                    frame_id="laptop")
        if not self._bag.open():
            logger.warning("orchestrator.mcap could not be opened; "
                           "continuing with JSONL-only timing capture")
        self._log = AutoSessionLog(self.ctx.run_dir / "predict_log.jsonl",
                                   include_maps=self._include_maps)
        try:
            self._pose_file = (self.ctx.run_dir / "pose_log.jsonl").open(
                "w", encoding="utf-8")
        except OSError as e:
            logger.warning("pose_log.jsonl could not be opened: %s", e)
            self._pose_file = None
        self._opened = True

    def close(self) -> None:
        if self._closed:
            return
        if self._log is not None:
            self._log.close()
        if self._bag is not None:
            self._bag.close()
        if self._pose_file is not None:
            try:
                self._pose_file.close()
            except OSError:
                pass
            self._pose_file = None
        self._closed = True

    # -- log factory (orchestrator hands this to auto.start()) ---------------

    def log_factory(self) -> AutoSessionLog:
        """Used as `log_factory_override` on AutoDriver.start so auto_driver's
        per-session events land in our predict_log.jsonl (not auto_logs/)."""
        if self._log is None:
            raise RuntimeError("RunRecorder.open() must be called first")
        return self._log

    # -- callbacks plumbed from bridge / orchestrator ------------------------

    def record_predict_timing(self, sequence: int, send_ns: int, recv_ns: int,
                              *, kind: str = "run") -> None:
        """Called from BridgeClient's on_predict_timing for every round-trip.

        `kind` is "warmup" for pre-run_start predicts and "run" otherwise;
        it lands in the JSONL `predict_timing` event for filtering. Warmup
        RTT is also accumulated into `warmup_rtt_us` for metadata.yaml.
        """
        if self._closed:
            return
        if self._bag is not None:
            self._bag.write_predict_timing(sequence, send_ns, recv_ns)
        if self._log is not None:
            self._log._emit("predict_timing",     # noqa: SLF001 — same package
                            sequence=int(sequence),
                            t_tcp_send_ns=int(send_ns),
                            t_tcp_recv_ns=int(recv_ns),
                            rtt_us=int((recv_ns - send_ns) / 1000),
                            kind=kind)
        if kind == "warmup":
            self.warmup_rtt_us.append(int((recv_ns - send_ns) / 1000))

    def mark_run_start(self, payload: dict[str, Any]) -> None:
        """Stamp the run_start event into mcap + JSONL. Captures the laptop
        monotonic time for time_to_completion fallback if bag events are lost.
        """
        self._t_run_start_ns = time.monotonic_ns()
        payload_str = json.dumps(payload, separators=(",", ":"))
        if self._bag is not None:
            self._bag.write_experiment_event("run_start", self.ctx.run_id, payload_str)
        if self._log is not None:
            self._log.run_start(self.ctx.run_id, payload)
        self.events.append({"event_type": "run_start", "ns": self._t_run_start_ns,
                            "payload": payload})
        self._record_pose_enabled = True

    def mark_run_end(self, outcome: str) -> None:
        self._record_pose_enabled = False
        self._t_run_end_ns = time.monotonic_ns()
        self._ended_at_wall = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        payload = {"outcome": outcome}
        if self._bag is not None:
            self._bag.write_experiment_event("run_end", self.ctx.run_id,
                                             json.dumps(payload, separators=(",", ":")))
        if self._log is not None:
            self._log.run_end(self.ctx.run_id, outcome)
        self.events.append({"event_type": "run_end", "ns": self._t_run_end_ns,
                            "outcome": outcome})

    def record_pose(self, robot_state: Any) -> None:
        """Append one tracker pose sample to pose_log.jsonl. No-op unless
        gated open by mark_run_start (and re-closed by mark_run_end), so the
        on-disk window matches the bag window. Caller should pass the
        already-filtered RobotState (post outlier gate); coordinates are
        written in BRIDGE frame so analysis can imshow obstacle.npy under
        origin='lower' without further transforms.
        """
        if (not self._record_pose_enabled or self._pose_file is None
                or self._closed or robot_state is None):
            return
        try:
            bx, by, h_rad = tracker_to_ros2(
                float(robot_state.grid_x), float(robot_state.grid_y),
                float(robot_state.heading_deg), self._grid_rows,
            )
            speed = robot_state.speed
            rec = {
                "t_mono_ns": time.monotonic_ns(),
                "x": round(bx, 4),
                "y": round(by, 4),
                "heading_rad": round(h_rad, 4),
                "speed": None if speed is None else round(float(speed), 4),
            }
            self._pose_file.write(json.dumps(rec, separators=(",", ":")) + "\n")
        except (OSError, AttributeError):
            pass

    def mark_marker(self, tag: str, note: str = "") -> None:
        if self._bag is not None:
            self._bag.write_experiment_event(
                "marker", self.ctx.run_id,
                json.dumps({"tag": tag, "note": note}, separators=(",", ":")))
        if self._log is not None:
            self._log.marker(tag, note)

    def mark_bag_capped(self, host: str) -> None:
        if self._bag is not None:
            self._bag.write_experiment_event(
                "bag_capped", self.ctx.run_id,
                json.dumps({"host": host}, separators=(",", ":")))
        if self._log is not None:
            self._log.bag_capped(host)

    # -- artifacts -----------------------------------------------------------

    def snapshot_obstacle(self, grid_yx: np.ndarray) -> None:
        """Save the static obstacle layer at run start as maps/obstacle.npy.

        Layout on disk is (rows, cols) uint8 indexed [y, x] with y bottom-up
        (origin='lower' for matplotlib). NOTE: this is DIFFERENT from the
        bridge predict-payload convention, which transposes to [x][y]; the
        on-disk snapshot deliberately keeps the analysis-friendly [y, x] form
        so summarize_run can imshow(obs, origin='lower') without transposing.
        Do NOT pass tracker_map_to_bridge(...) here — that returns [x][y].
        Caller (orchestrator._scene_obstacle_yx) should hand in
        np.flipud(scene_obstacles_grid) only.
        """
        np.save(self.ctx.run_dir / "maps" / "obstacle.npy",
                grid_yx.astype(np.uint8))

    def write_metadata(self, extra: dict[str, Any] | None = None) -> Path:
        """Write metadata.yaml per analysis §3 v3 schema.

        Contract: caller MUST pass `extra["outcome"]` ∈ {success, failure,
        aborted}. We do NOT fall back to a default — silently mis-classifying
        an aborted run as success would contaminate paired comparisons.
        goal_cell / start_cell are stored in BRIDGE frame (origin lower) so
        they line up with obstacle.npy and /grid_nav_node/grid_pose in
        summary.png Panel B.
        """
        extra = dict(extra or {})
        if "outcome" not in extra:
            raise ValueError(
                "write_metadata: caller must provide extra['outcome'] "
                "in {success, failure, aborted}")
        outcome = extra["outcome"]
        if outcome not in ("success", "failure", "aborted"):
            raise ValueError(
                f"write_metadata: invalid outcome={outcome!r}; "
                "must be success | failure | aborted")
        meta: dict[str, Any] = {
            "run_id": self.ctx.run_id,
            "session_id": self.ctx.session_id,
            "mode": self.ctx.mode,
            "planner": self.ctx.planner,
            "map_id": self.ctx.map_id,
            "started_at": self.ctx.started_at,
            # ended_at is the wall clock at run_end if a real run completed;
            # for pre-run-start aborts we collapse to started_at so the YAML
            # shows a zero-duration window (operator-readable sentinel).
            # Authoritative duration is time_to_completion_s in analysis,
            # which uses the bridge-host bag clock; these wall-clock fields
            # are human-readable identifiers only.
            "ended_at": self._ended_at_wall or self.ctx.started_at,
            "goal_cell": self._to_bridge_cell(self.ctx.goal_cell),
            "start_cell": self._to_bridge_cell(self.ctx.start_cell) or [0, self._grid_rows - 1],
            "outcome": outcome,
            "dry_run": bool(self.ctx.dry_run),
            "bag_capped": bool(extra.get("bag_capped", False)),
            "warmup_inferences_us": list(self.warmup_rtt_us),
            "ambient_notes": extra.get("ambient_notes", ""),
            "cell_size_m": self._cell_size_m,
            "git_commits": {"orchestrator": _orchestrator_git_sha()},
        }
        if self._battery_band_violation:
            meta["battery_band_violation"] = self._battery_band_violation
        # Last in wins for extra fields outside the canonical schema (e.g.
        # end_reason, bag_elapsed_s).  Don't let it overwrite the
        # bridge-frame-normalized goal/start cells.
        protected = {"goal_cell", "start_cell", "ambient_notes"}
        meta.update({k: v for k, v in extra.items() if k not in protected})
        path = self.ctx.run_dir / "metadata.yaml"
        path.write_text(yaml.safe_dump(meta, sort_keys=False))
        return path

    def _to_bridge_cell(self, cell):
        """Tracker (col, row, row 0 = top)  ->  bridge (x, y, y 0 = bottom).

        Returns None for a None input (baseline runs have goal_cell=None).
        The conversion is `y = grid_rows - 1 - row`; analysis indexes the
        on-disk obstacle.npy as [y, x] under matplotlib origin='lower'.
        """
        if not cell:
            return None
        col, row = int(cell[0]), int(cell[1])
        return [col, self._grid_rows - 1 - row]

    def note_battery_band_violation(self, voltage: float,
                                    band: tuple[float, float]) -> None:
        self._battery_band_violation = (
            f"start voltage {voltage:.2f} V outside band {band}")

    # -- summarize_run spawn -------------------------------------------------

    def spawn_summarize(self, python: str | None, script: Path) -> None:
        """Fire-and-forget summarize_run subprocess. Failures are logged-only;
        analysis is post-hoc and never blocks the next run."""
        if not python:
            return
        if not script.exists():
            logger.warning("summarize_run script not found at %s", script)
            return
        try:
            subprocess.Popen(
                [python, str(script), str(self.ctx.run_dir)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                close_fds=(sys.platform != "win32"),
            )
        except OSError as e:
            logger.warning("summarize_run spawn failed: %s", e)


# -- helpers -----------------------------------------------------------------

def _orchestrator_git_sha() -> str:
    """`git rev-parse HEAD` of this repo. Empty string on failure."""
    if not shutil.which("git"):
        return ""
    try:
        r = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[2]),
             "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except (subprocess.TimeoutExpired, OSError):
        return ""
