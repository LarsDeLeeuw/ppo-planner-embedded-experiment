"""orchestrator.orchestrator - Top-level coordinator.

Owns the live state machine that walks one run from BAGS_STARTING through
WRITING_META, plus the queue of pending runs. Ticked once per frame from
main.py's capture loop; long-running steps (SSH start/stop, scp pull) block
the camera loop for a few seconds each — acceptable between runs.

Public surface:
  - Orchestrator.__init__       — constructed once at startup if enabled
  - .enqueue_*                  — operator commands from key bindings / CLI
  - .cancel_current             — soft cancel of the active run
  - .tick(frame, robot_state, grid)  — called each main-loop iteration
  - .status_summary() -> str    — for the HUD / log

Out of scope here:
  - Live mode-switch (changing which host runs the bridge mid-process).
    Operate one mode per tracker process; to switch, restart the app pointed
    at the new bridge host.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from bridge_client import BridgeClient
from orchestrator.bag_set import BagSet
from orchestrator.campaign import CampaignPlan
from orchestrator.run_recorder import RunRecorder
from orchestrator.ssh_bag import BagHost, SshBagError, ssh_available
from orchestrator.state import (
    Cell,
    MapEntry,
    OrchestratorState,
    QueueItem,
    RunContext,
    SessionContext,
)

logger = logging.getLogger(__name__)


@dataclass
class _ActiveRun:
    """Mutable bookkeeping for the one currently-active run."""
    qi: QueueItem
    ctx: RunContext
    recorder: RunRecorder
    bagset: BagSet
    obstacle_yx: np.ndarray | None = None
    warmup_sent: int = 0
    warmup_inflight: bool = False
    warmup_last_send_ns: int = 0
    auto_started: bool = False
    end_signal: tuple[str, str] | None = None    # (outcome, reason) from auto_driver
    baseline_started_at: float = 0.0
    # Wallclock at the moment we entered RUN_ACTIVE — backstop against
    # auto_driver wedging in WAITING_GOAL with no MoveToGrid result.
    # None until the run actually enters RUN_ACTIVE (0.0 is a valid time
    # under virtual clocks in tests, so we cannot use 0.0 as a sentinel).
    run_active_started_at: float | None = None
    bag_capped: bool = False
    extras: dict[str, Any] = field(default_factory=dict)


# Baseline + dry-run last this many seconds inside the run window.
BASELINE_DURATION_S = 60.0


class Orchestrator:

    def __init__(
        self,
        *,
        app_cfg,
        bridge: BridgeClient,
        auto: Any,                                      # AutoDriver (avoid import cycle)
        cb_state: dict,                                 # main.py's mouse-callback dict
        scene_loader: Callable[[Path], Any],            # returns a Scene
        corner_marker_ids: list[int],
        scene_dir: Path,
        on_log: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.app_cfg = app_cfg
        self.cfg = app_cfg.orchestrator
        self.bridge = bridge
        self.auto = auto
        self._cb_state = cb_state                       # main.py mutates _cb_state["scene"]
        self._scene_loader = scene_loader
        self._corner_marker_ids = corner_marker_ids
        self._scene_dir = scene_dir
        self._log = on_log or (lambda m: print(m))
        self._clock = clock

        self.state = OrchestratorState.IDLE
        self.queue: deque[QueueItem] = deque()
        self.sessions: dict[str, SessionContext] = {}
        self._active: _ActiveRun | None = None
        self._auto_advance = False                      # set by CLI --auto-advance
        self._campaign: CampaignPlan | None = None
        self._session_ts = datetime.now()               # session timestamp namespace

        # Hosts (one per remote).
        ssh_cfg = self.cfg.ssh
        self._pi_host = BagHost(
            name="pi", ssh_target=ssh_cfg.pi.target,
            bag_script=ssh_cfg.pi.bag_script,
            remote_tmp=ssh_cfg.pi.remote_tmp,
            connect_timeout_s=ssh_cfg.connect_timeout_s,
            max_retries=ssh_cfg.max_retries,
            mock=self.cfg.ssh_mock,
        )
        self._desktop_host = BagHost(
            name="desktop", ssh_target=ssh_cfg.desktop.target,
            bag_script=ssh_cfg.desktop.bag_script,
            remote_tmp=ssh_cfg.desktop.remote_tmp,
            connect_timeout_s=ssh_cfg.connect_timeout_s,
            max_retries=ssh_cfg.max_retries,
            mock=self.cfg.ssh_mock,
        )

        if not self.cfg.ssh_mock and not ssh_available():
            self._log("[orch] WARNING: ssh/scp not on PATH; real bag pulls will fail")

        # Dump the resolved values the operator most often gets wrong (SSH
        # targets, remote bag-script paths, experiments root) so typos
        # surface in one second on startup rather than 30+ seconds into the
        # first real run when an SSH connect times out.
        if not self.cfg.ssh_mock:
            self._log(
                f"[orch] resolved SSH:\n"
                f"          pi      target={self._pi_host.ssh_target!r}  "
                f"bag_script={self._pi_host.bag_script}\n"
                f"          desktop target={self._desktop_host.ssh_target!r}  "
                f"bag_script={self._desktop_host.bag_script}\n"
                f"          experiments_root={Path(self.cfg.experiments_root).resolve()}"
            )

        # Wire callbacks. on_predict_timing fires for EVERY round-trip; we
        # route them to the active recorder based on state. auto's session-end
        # callback flips end_signal so the next tick can advance.
        bridge.on_predict_timing(self._on_predict_timing)
        bridge.on("error", self._on_bridge_error)
        auto.on_session_end(self._on_session_end)

        # Active session is computed from queue items (we keep one session ctx
        # per (session_id) so SessionContext.completed accumulates correctly).
        self._active_session_id: str | None = None

        # Monotonic across the orchestrator's lifetime — used to disambiguate
        # run_ids when two trials of the same (cell, map) session start in
        # the same wall-clock second (10 trials at ~10s each is fine; 10
        # trials at <1s each, e.g. baselines or mock tests, would collide).
        self._trial_counter = 0

    # --- public commands ----------------------------------------------------

    def load_campaign(self, plan_path: Path) -> None:
        """Load campaign_plan.yaml and enqueue every session+trial it contains.

        Each scheduled session = one (cell, map) pair with `trials` identical
        runs back-to-back, so the operator never has to physically swap the
        map mid-session. Map-major ordering means once map M is finished
        (all cells * trials done on it) you can rearrange to the next map.
        """
        self._campaign = CampaignPlan.load(plan_path)
        warns = self._campaign.validate_scene_files(self._scene_dir)
        for w in warns:
            self._log(f"[orch] warn: {w}")
        self._campaign.write_resolved()
        for sched in self._campaign.schedule:
            sess_id = self._session_id_for(sched.cell, sched.map, idx=sched.index)
            session = self._ensure_session(sched.cell, sess_id)
            for _ in range(sched.trials):
                self.queue.append(QueueItem(
                    kind="run", cell=sched.cell, map_entry=sched.map,
                    session_id=sess_id))
                session.queue.append(sched.map)
        self._log(f"[orch] campaign loaded: {len(self.queue)} run(s) across "
                  f"{len(self._campaign.schedule)} session(s) "
                  f"({len(self._campaign.maps)} maps × {len(self._campaign.cells)} cells "
                  f"× {self._campaign.runs_per_session} trials)")

    def enqueue_run(self, cell: Cell, map_entry: MapEntry) -> None:
        sess_id = self._session_id_for(cell, map_entry)
        self._ensure_session(cell, sess_id)
        self.queue.append(QueueItem(
            kind="run", cell=cell, map_entry=map_entry, session_id=sess_id))

    def enqueue_baseline(self, cell: Cell) -> None:
        sess_id = self._session_id_for(cell)
        self._ensure_session(cell, sess_id)
        self.queue.append(QueueItem(
            kind="baseline", cell=cell, map_entry=None, session_id=sess_id))

    def enqueue_dry_run(self, cell: Cell, map_entry: MapEntry | None = None) -> None:
        sess_id = self._session_id_for(cell, map_entry)
        self._ensure_session(cell, sess_id)
        self.queue.append(QueueItem(
            kind="dry_run", cell=cell, map_entry=map_entry, session_id=sess_id))

    def set_auto_advance(self, on: bool) -> None:
        self._auto_advance = bool(on)

    def cancel_current(self, reason: str = "user cancel") -> None:
        """Cancel the active run.

        Behaviour depends on the current state:
          - pre-run_start states (BAGS_STARTING / WARMUP / RUN_ARMING): short-
            circuit immediately via _abort_before_run_start so no run_start
            event is ever emitted for a cancelled run.
          - RUN_ACTIVE with auto driving: stop auto; its on_session_end
            callback sets end_signal and the next tick walks the normal
            teardown (BAGS_STOPPING -> PULLING -> WRITING_META).
          - RUN_ACTIVE for baseline/dry_run (no auto): set end_signal so
            _tick_run_active picks it up next frame.
          - already-tearing-down (RUN_ENDING/BAGS_STOPPING/PULLING/
            WRITING_META): refuse with a visible log; the outcome is already
            committed and a late "cancel" would mislead the operator.
        """
        if self._active is None or self.state in (OrchestratorState.IDLE,
                                                  OrchestratorState.STOPPED_ERROR):
            return
        TEARDOWN = (OrchestratorState.RUN_ENDING, OrchestratorState.BAGS_STOPPING,
                    OrchestratorState.PULLING, OrchestratorState.WRITING_META)
        if self.state in TEARDOWN:
            self._log(f"[orch] cancel ignored: already tearing down "
                      f"({self.state.value})")
            return
        self._log(f"[orch] cancel: {reason}")
        PRE_RUN_START = (OrchestratorState.BAGS_STARTING,
                         OrchestratorState.WARMUP,
                         OrchestratorState.RUN_ARMING)
        if self.state in PRE_RUN_START:
            # Tear down without ever emitting run_start; bags are flushed via
            # _abort_before_run_start (best-effort stop_all + pull + cleanup).
            self._abort_before_run_start(reason)
            return
        if self.auto.is_active():
            # is_error=False: a user cancel is a clean operator abort, not an
            # auto-internal error. This returns the driver to IDLE (so the next
            # trial can start) and yields outcome "aborted" via the session_end
            # callback, which sets end_signal for the tick to consume.
            self.auto.stop(reason, is_error=False)
        else:
            self._active.end_signal = ("aborted", reason)

    def status_summary(self) -> str:
        if self._active:
            return (f"{self.state.value}  run={self._active.ctx.run_id}  "
                    f"queue={len(self.queue)}")
        return f"{self.state.value}  queue={len(self.queue)}"

    def shutdown(self, reason: str = "app exit") -> None:
        """Synchronously tear down any in-flight run. Called from main.py's
        finally block so app exit / Esc / KeyboardInterrupt never leaves bags
        running on remote hosts (handover §13 / constraint #5).

        Walks the same step sequence as the state-machine teardown (stop_all,
        pull_all_into, cleanup_remote, write metadata), but synchronously and
        with every step guarded so one failure can't block the others. Safe
        to call when no run is active — it's a no-op.
        """
        a = self._active
        if a is None:
            return
        self._log(f"[orch] shutdown: {reason}")
        # Stop auto if still running so its session-end teardown writes the
        # session_stop log line (without closing the borrowed JSONL — see F9).
        if self.auto.is_active():
            try:
                self.auto.stop(reason, is_error=True)
            except Exception:
                pass
        # If we never emitted run_start, take the pre-run abort path so we
        # don't write a misleading run_end on the bridge side.
        if self.state in (OrchestratorState.BAGS_STARTING,
                          OrchestratorState.WARMUP,
                          OrchestratorState.RUN_ARMING):
            self._abort_before_run_start(reason)
            return
        # run_start was emitted: emit run_end (best-effort) so the bridge bag
        # carries the boundary, then walk the rest of the teardown.
        a.extras.setdefault("outcome", "aborted")
        a.extras.setdefault("end_reason", reason)
        try:
            a.recorder.mark_run_end(a.extras["outcome"])
            self.bridge.send_experiment_event(
                "run_end", a.ctx.run_id,
                {"outcome": a.extras["outcome"], "reason": reason})
        except Exception:
            logger.exception("[orch] run_end emit failed during shutdown")
        try:
            elapsed = a.bagset.stop_all()
            a.extras.setdefault("bag_elapsed_s", elapsed)
        except Exception:
            pass
        try:
            pulled = a.bagset.pull_all_into(a.ctx.run_dir)
            if any(p is None for p in pulled.values()):
                a.extras["outcome"] = "aborted"
        except Exception:
            pass
        self._maybe_cleanup_remote(a)
        try:
            a.recorder.write_metadata(a.extras)
        except Exception:
            logger.exception("[orch] metadata write failed during shutdown")
        try:
            a.recorder.close()
        except Exception:
            pass
        self._record_run_outcome(a)
        self._active = None
        self.state = OrchestratorState.IDLE

    # --- main-loop tick -----------------------------------------------------

    def tick(self, frame: Any, robot_state: Any, grid: Any) -> None:
        """Called once per main-loop iteration. Advances the state machine.

        Wrapped in a top-level guard: any exception escapes us into
        STOPPED_ERROR with a best-effort teardown (stop bags + write metadata)
        so a single bad tick can never wedge the campaign queue forever.
        """
        s = self.state
        if s == OrchestratorState.STOPPED_ERROR:
            return                                    # terminal until reset
        try:
            if s == OrchestratorState.IDLE:
                self._tick_idle()
            elif s == OrchestratorState.BAGS_STARTING:
                self._tick_bags_starting()
            elif s == OrchestratorState.WARMUP:
                self._tick_warmup()
            elif s == OrchestratorState.RUN_ARMING:
                self._tick_run_arming(robot_state, grid)
            elif s == OrchestratorState.RUN_ACTIVE:
                self._tick_run_active(robot_state)
            elif s == OrchestratorState.RUN_ENDING:
                self._tick_run_ending()
            elif s == OrchestratorState.BAGS_STOPPING:
                self._tick_bags_stopping()
            elif s == OrchestratorState.PULLING:
                self._tick_pulling()
            elif s == OrchestratorState.WRITING_META:
                self._tick_writing_meta()
        except Exception as e:                        # noqa: BLE001 — top-level trap
            logger.exception("orchestrator tick raised in state=%s", s.value)
            self._enter_stopped_error(f"tick raised in {s.value}: {e!r}")

    def reset_after_error(self) -> None:
        """Move STOPPED_ERROR -> IDLE so the operator can re-queue runs without
        restarting the app. Idempotent."""
        if self.state == OrchestratorState.STOPPED_ERROR:
            self.state = OrchestratorState.IDLE
            self._log("[orch] error cleared")

    def _enter_stopped_error(self, reason: str) -> None:
        """Best-effort teardown of any in-flight run + park the state machine
        in STOPPED_ERROR. Called from tick's top-level try/except."""
        a = self._active
        self._log(f"[orch] STOPPED_ERROR: {reason}")
        if a is not None:
            try:
                a.bagset.stop_all()
            except Exception:
                pass
            try:
                a.bagset.cleanup_remote()
            except Exception:
                pass
            a.extras["outcome"] = "aborted"
            a.extras.setdefault("end_reason", reason)
            try:
                a.recorder.mark_marker("orchestrator_error", reason)
                a.recorder.write_metadata(a.extras)
            except Exception:
                pass
            try:
                a.recorder.close()
            except Exception:
                pass
            self._record_run_outcome(a)
            self._active = None
        self.state = OrchestratorState.STOPPED_ERROR

    # --- state-specific handlers --------------------------------------------

    def _tick_idle(self) -> None:
        if self._auto_advance and self.queue:
            self._begin_next()

    def _tick_bags_starting(self) -> None:
        assert self._active is not None
        if self._active.end_signal is not None:
            _, reason = self._active.end_signal
            self._abort_before_run_start(f"cancelled: {reason}")
            return
        try:
            self._active.bagset.start_all()
            self._log(f"[orch] bags recording for {self._active.ctx.run_id}")
            if self._active.qi.kind == "baseline":
                # No warmup, no auto_driver. Go straight to RUN_ARMING.
                self.state = OrchestratorState.RUN_ARMING
                return
            if self.cfg.warmup_count <= 0:
                self.state = OrchestratorState.RUN_ARMING
                return
            self.state = OrchestratorState.WARMUP
        except SshBagError as e:
            self._abort_before_run_start(f"bag start failed: {e}")

    def _tick_warmup(self) -> None:
        a = self._active
        assert a is not None
        if a.end_signal is not None:
            _, reason = a.end_signal
            self._abort_before_run_start(f"cancelled: {reason}")
            return
        rec = a.recorder
        received = len(rec.warmup_rtt_us)
        if received >= self.cfg.warmup_count:
            self.state = OrchestratorState.RUN_ARMING
            return
        # If a warmup is in flight, wait for its result (timing callback advances
        # warmup_rtt_us). Bail if it stalls past warmup_timeout_s.
        if a.warmup_inflight:
            elapsed_s = (time.monotonic_ns() - a.warmup_last_send_ns) / 1e9
            if received > a.warmup_sent - 1:
                a.warmup_inflight = False                    # this one came back
            elif elapsed_s > self.cfg.warmup_timeout_s:
                self._abort_before_run_start(
                    f"warmup #{a.warmup_sent} timed out after {elapsed_s:.1f}s")
                return
            else:
                return
        # Ready to send the next warmup.
        self._send_warmup_predict()

    def _tick_run_arming(self, robot_state, grid) -> None:
        a = self._active
        assert a is not None
        if a.end_signal is not None:
            _, reason = a.end_signal
            self._abort_before_run_start(f"cancelled: {reason}")
            return
        ctx = a.ctx
        payload = self._run_start_payload()
        # local copy via recorder + ROS-side via bridge (forwarded to /experiment/events)
        a.recorder.mark_run_start(payload)
        try:
            self.bridge.send_experiment_event("run_start", ctx.run_id, payload)
        except Exception as e:
            logger.warning("send run_start over bridge failed: %s", e)

        if a.qi.kind == "baseline" or a.qi.kind == "dry_run":
            a.baseline_started_at = self._clock()
            self.state = OrchestratorState.RUN_ACTIVE
            return

        # A prior run that ended in an auto-driver error — or a user cancel
        # routed through stop(is_error=True) — leaves the driver parked in
        # STOPPED_ERROR. start() only proceeds from IDLE, so without this every
        # subsequent trial aborts with "auto mode already active". Idempotent:
        # a no-op when the driver is already IDLE.
        self.auto.clear_error()

        # PPO last-cell assist: only for ppo cells, only when opted in. A*
        # planners are never assisted (they may step away from a diagonally
        # blocked goal on purpose).
        goal_assist = self.cfg.ppo_goal_assist and a.qi.cell.planner == "ppo"

        # Real run: kick off the auto_driver with sequence resumed past warmups.
        ok, reason = self.auto.start(
            self._cb_state["scene"], grid, robot_state,
            corner_marker_ids=self._corner_marker_ids,
            start_sequence=self.cfg.warmup_count,
            log_factory_override=a.recorder.log_factory,
            goal_assist=goal_assist,
        )
        if not ok:
            self._abort_after_run_start(f"auto.start failed: {reason}")
            return
        a.auto_started = True
        a.run_active_started_at = self._clock()
        self.state = OrchestratorState.RUN_ACTIVE

    def _tick_run_active(self, robot_state) -> None:
        a = self._active
        assert a is not None
        # 30 Hz tracker-pose log. Gated inside record_pose by run_start /
        # run_end, so calling for both baseline and real runs is safe — only
        # the in-run window lands on disk.
        if robot_state is not None:
            try:
                a.recorder.record_pose(robot_state)
            except Exception:
                pass
        if a.end_signal is not None:
            self.state = OrchestratorState.RUN_ENDING
            return
        if a.qi.kind in ("baseline", "dry_run"):
            # Stream pose to keep the nav node aware (no goal). Cheap.
            if robot_state is not None and a.qi.kind == "baseline":
                try:
                    self.bridge.send_pose(robot_state)
                except Exception:
                    pass
            if self._clock() - a.baseline_started_at >= BASELINE_DURATION_S:
                a.end_signal = ("success", "baseline duration elapsed")
                self.state = OrchestratorState.RUN_ENDING
            return
        # Real run: wallclock backstop in case auto_driver wedges in
        # WAITING_GOAL with no MoveToGrid result.  Set the run aborted; the
        # next tick will reach RUN_ENDING and walk the normal teardown.
        # The orchestrator's intent here is "aborted" (operator-equivalent),
        # not "failure" (auto.stop with is_error=True would otherwise put
        # 'failure' on end_signal via the session_end callback); force-
        # override after auto.stop so the outcome reflects orchestrator intent.
        if (a.qi.kind == "run" and a.run_active_started_at is not None
                and self._clock() - a.run_active_started_at
                    > self.cfg.run_max_duration_s):
            if self.auto.is_active():
                self.auto.stop("run wallclock timeout", is_error=True)
            a.end_signal = ("aborted", "run wallclock timeout")
            self.state = OrchestratorState.RUN_ENDING

    def _tick_run_ending(self) -> None:
        a = self._active
        assert a is not None
        outcome, reason = a.end_signal or ("aborted", "no end signal")
        # ROS-side event copy + laptop copy
        a.recorder.mark_run_end(outcome)
        try:
            self.bridge.send_experiment_event(
                "run_end", a.ctx.run_id, {"outcome": outcome, "reason": reason})
        except Exception as e:
            logger.warning("send run_end over bridge failed: %s", e)
        a.extras["outcome"] = outcome
        a.extras["end_reason"] = reason
        # Flag assisted finishes so analysis can separate them from genuine PPO
        # successes. Only meaningful for real runs (auto driver isn't active for
        # baseline/dry_run); goal_assist_fired() is reset at each run's start()
        # so it can't carry over from a prior trial.
        a.extras["goal_assisted"] = (
            a.qi.kind == "run" and self.auto.goal_assist_fired())
        self.state = OrchestratorState.BAGS_STOPPING

    def _tick_bags_stopping(self) -> None:
        a = self._active
        assert a is not None
        elapsed = a.bagset.stop_all()
        # An elapsed of -1.0 means the remote stop call failed after retries;
        # the bag may have already self-terminated via SIGINT (the 5-min cap)
        # or the connection dropped — either way we cannot confirm a clean
        # close, so treat as aborted per handover §13.
        unknown = [n for n, e in elapsed.items() if e < 0]
        if unknown:
            a.extras["outcome"] = "aborted"
            a.extras.setdefault(
                "end_reason", f"bag stop failed / unknown elapsed: {unknown}")
        if a.bagset.bag_capped(elapsed):
            a.bag_capped = True
            a.extras["outcome"] = "aborted"
            for bag_name, e in elapsed.items():
                if e >= a.bagset.max_duration_s:
                    a.recorder.mark_bag_capped(bag_name)
                    # Also forward to /experiment/events so ROS-side bags
                    # carry the event (handover §7.2 contract).
                    try:
                        self.bridge.send_experiment_event(
                            "bag_capped", a.ctx.run_id, {"host": bag_name})
                    except Exception as ex:
                        logger.warning("send bag_capped over bridge failed: %s", ex)
        a.extras["bag_capped"] = a.bag_capped
        a.extras["bag_elapsed_s"] = elapsed
        self.state = OrchestratorState.PULLING

    def _tick_pulling(self) -> None:
        a = self._active
        assert a is not None
        # Belt-and-braces: even if pull_all_into raises something the host
        # abstraction didn't anticipate (FileNotFoundError on missing scp,
        # etc.), we must still advance to WRITING_META — otherwise the next
        # tick re-enters PULLING and busy-loops on the broken connection
        # while remote bags stay orphaned (handover §13 contract).
        missing: list[str] = []
        try:
            pulled = a.bagset.pull_all_into(a.ctx.run_dir)
            missing = [n for n, p in pulled.items() if p is None]
        except Exception as e:                          # noqa: BLE001 — state-machine guard
            logger.exception("pull_all_into raised; marking run aborted")
            self._log(f"[orch] pull raised: {e}")
            a.extras["outcome"] = "aborted"
            a.extras.setdefault("end_reason", f"bag pull raised: {e!r}")
            missing = [asn.bag_name for asn in a.bagset.assignments]
        if missing and a.extras.get("outcome") != "aborted":
            a.extras["outcome"] = "aborted"
            a.extras["end_reason"] = f"bag pull failed: {missing}"
            self._log(f"[orch] pull failed: {missing}")
        self._maybe_cleanup_remote(a)
        self.state = OrchestratorState.WRITING_META

    def _tick_writing_meta(self) -> None:
        a = self._active
        assert a is not None
        # Default to aborted if upstream forgot to set outcome — write_metadata
        # itself now refuses an absent outcome, so this guard is belt-and-braces.
        if "outcome" not in a.extras:
            a.extras["outcome"] = "aborted"
            a.extras.setdefault("end_reason", "outcome not set by orchestrator")
        meta_path = a.recorder.write_metadata(a.extras)
        a.recorder.close()
        self._log(f"[orch] wrote {meta_path}")
        # summarize_run is fire-and-forget; the next run never waits on it.
        if self.cfg.summarize_after_run:
            script = (Path(__file__).resolve().parents[2]
                      / self.cfg.summarize_script.as_posix()).resolve()
            a.recorder.spawn_summarize(self.cfg.summarize_python, script)
        self._record_run_outcome(a)
        self._active = None
        self.state = OrchestratorState.IDLE

    def _record_run_outcome(self, a: _ActiveRun) -> None:
        """Append the run id to its session's completed/aborted list. Called
        from every terminal path so the in-process session ledger is accurate
        (handover §1.5 SessionContext.aborted/completed invariant)."""
        sess = self.sessions.get(a.ctx.session_id)
        if sess is None:
            return
        if a.extras.get("outcome") == "aborted":
            sess.aborted.append(a.ctx.run_id)
        else:
            sess.completed.append(a.ctx.run_id)

    # --- run setup ----------------------------------------------------------

    def _begin_next(self) -> None:
        if self.state != OrchestratorState.IDLE:
            active_id = self._active.ctx.run_id if self._active else None
            self._log(f"[orch] _begin_next ignored: state={self.state.value} "
                      f"active={active_id}")
            return
        if not self.queue:
            return
        qi = self.queue.popleft()
        ctx = self._build_run_context(qi)
        # Load the scene for this map (skipped for baseline if no map_entry)
        if qi.map_entry is not None:
            try:
                self._load_scene_for_run(qi.map_entry)
            except Exception as e:
                self._log(f"[orch] scene load failed for {qi.map_entry.map_id}: {e}")
                return
        # Snapshot the obstacle layer NOW, before any other run setup.
        obstacle_yx = self._scene_obstacle_yx()

        recorder = RunRecorder(ctx, include_maps=self.app_cfg.auto.log_maps,
                               grid_rows=self.app_cfg.grid.rows,
                               cell_size_m=self.app_cfg.grid.cell_size_m)
        try:
            recorder.open()
        except Exception as e:
            self._log(f"[orch] recorder open failed: {e}")
            return
        if obstacle_yx is not None:
            recorder.snapshot_obstacle(obstacle_yx)

        bagset = BagSet.for_mode(
            mode=qi.cell.mode, pi=self._pi_host, desktop=self._desktop_host,
            run_id=ctx.run_id, max_duration_s=self.cfg.bag_max_duration_s,
        )
        self._active = _ActiveRun(qi=qi, ctx=ctx, recorder=recorder,
                                  bagset=bagset, obstacle_yx=obstacle_yx)
        self._active_session_id = ctx.session_id
        self.state = OrchestratorState.BAGS_STARTING
        self._log(f"[orch] begin {qi.kind}: {ctx.run_id}")

    def _build_run_context(self, qi: QueueItem) -> RunContext:
        is_baseline = (qi.kind == "baseline")
        kind_tag = "" if qi.kind == "run" else qi.kind.replace("_", "")
        self._trial_counter += 1
        run_id = RunContext.make_run_id(
            qi.cell.planner,
            qi.map_entry.map_id if qi.map_entry else "idle",
            qi.cell.mode, kind=kind_tag, trial=self._trial_counter,
        )
        # session_id is stamped at enqueue time so each QueueItem carries the
        # right session it belongs to. Fall back to a per-cell id only for
        # callers that built a QueueItem directly without going through one
        # of the enqueue_* helpers (defense in depth — should not happen in
        # the regular paths).
        sess_id = qi.session_id or self._session_id_for(qi.cell, qi.map_entry)
        session = self._ensure_session(qi.cell, sess_id)
        run_dir = session.session_dir / "runs" / run_id
        return RunContext(
            run_id=run_id, session_id=sess_id, session_dir=session.session_dir,
            run_dir=run_dir, mode=qi.cell.mode, planner=qi.cell.planner,
            map_id=qi.map_entry.map_id if qi.map_entry else "idle",
            scene_name=qi.map_entry.scene_name if qi.map_entry else "",
            goal_cell=qi.map_entry.goal_cell if (qi.map_entry and not is_baseline) else None,
            start_cell=qi.map_entry.start_cell if qi.map_entry else (0, 0),
            dry_run=(qi.kind == "dry_run"),
            is_baseline=is_baseline,
        )

    def _session_id_for(self, cell: Cell, map_entry: MapEntry | None = None,
                        idx: int | None = None) -> str:
        """Build the session_id used as the runs/<id>/ directory name.

        Format: session-<YYYYMMDD>-<mode>-<planner>[-<map_id>][-<idx>].
        - map_entry is included when known (campaign-driven runs always
          have one; operator-driven baselines may not).
        - idx disambiguates re-visits of the same (cell, map) pair in
          future campaign schedules; campaign.load() supplies it.
        """
        date = self._session_ts.strftime("%Y%m%d")
        map_part = f"-{map_entry.map_id}" if map_entry is not None else ""
        suffix = f"-{idx:02d}" if idx is not None else ""
        return f"session-{date}-{cell.mode}-{cell.planner}{map_part}{suffix}"

    def _ensure_session(self, cell: Cell, sess_id: str) -> SessionContext:
        if sess_id in self.sessions:
            return self.sessions[sess_id]
        root = Path(self.cfg.experiments_root).resolve()
        session_dir = root / sess_id
        session_dir.mkdir(parents=True, exist_ok=True)
        s = SessionContext(session_id=sess_id, session_dir=session_dir,
                           cell=cell, queue=[])
        self.sessions[sess_id] = s
        return s

    def _load_scene_for_run(self, m: MapEntry) -> None:
        """Load scenes/<scene_name>.json and override the goal layer."""
        path = self._scene_dir / f"{m.scene_name}.json"
        scene = self._scene_loader(path)
        if scene is None:
            raise RuntimeError(f"scene_loader returned None for {path}")
        # Override the goal layer programmatically.
        if "goal" in scene:
            goal_layer = scene.get("goal")
            goal_layer.clear()
            goal_layer.set(int(m.goal_cell[0]), int(m.goal_cell[1]), True)
        self._cb_state["scene"] = scene

    def _scene_obstacle_yx(self) -> np.ndarray | None:
        scene = self._cb_state.get("scene")
        if scene is None or "obstacles" not in scene:
            return None
        # scene.get("obstacles").grid is (rows, cols) with row 0 at TOP (tracker frame).
        # Bridge / analysis convention is origin lower (y bottom-up). Flip vertically.
        return np.ascontiguousarray(np.flipud(scene.get("obstacles").grid)).astype(np.uint8)

    def _run_start_payload(self) -> dict:
        a = self._active
        assert a is not None
        ctx = a.ctx
        return {
            "mode": ctx.mode, "planner": ctx.planner, "map_id": ctx.map_id,
            "goal_cell": list(ctx.goal_cell) if ctx.goal_cell else None,
            "start_cell": list(ctx.start_cell),
            "surface": "lab",
            "wifi_rssi_dbm": None,                  # captured ambient on the Pi
            "kind": a.qi.kind,
        }

    # --- warmup -------------------------------------------------------------

    def _send_warmup_predict(self) -> None:
        a = self._active
        assert a is not None
        # Use a minimal map matching the configured grid size (planner doesn't
        # actually need a sensible map for warmup-side latency measurement —
        # the planner runs its full forward pass either way). Robot/goal at
        # opposite corners so A* paths exist.
        n = self.app_cfg.grid.cols
        m = self.app_cfg.grid.rows
        obstacle = [[0 for _ in range(m)] for _ in range(n)]
        energy = [[0.0 for _ in range(m)] for _ in range(n)]
        seq = a.warmup_sent + 1
        try:
            self.bridge.send_predict(
                obstacle_map=obstacle, energy_map=energy,
                robot_pos=(0, 0), goal_pos=(n - 1, m - 1),
                sequence=seq,
            )
        except Exception as e:
            self._abort_before_run_start(f"warmup send failed: {e}")
            return
        a.warmup_sent = seq
        a.warmup_inflight = True
        a.warmup_last_send_ns = time.monotonic_ns()

    # --- abort helpers ------------------------------------------------------

    def _abort_before_run_start(self, reason: str) -> None:
        """No run_start emitted yet: tear down bags + write minimal metadata.

        Goes through every teardown step the normal path does so we don't:
          - leave bags running on remote hosts (constraint #5)
          - leak /tmp/experiments/<run_id> on the Pi (handover §13)
          - corrupt the session ledger for pre-run-start aborts (F5)
          - produce zero-byte predict_log.jsonl that validate_run_folder
            flags as 'empty:' on every pre-warmup abort (F21)
        """
        a = self._active
        if a is None:
            self.state = OrchestratorState.IDLE
            return
        self._log(f"[orch] abort (pre-run_start): {reason}")
        try:
            a.bagset.stop_all()
            a.bagset.pull_all_into(a.ctx.run_dir)
        except Exception:
            pass
        self._maybe_cleanup_remote(a)
        a.extras["outcome"] = "aborted"
        a.extras["end_reason"] = reason
        # Drop at least one event into the JSONL + mcap so neither is zero-
        # byte; the validator's "empty:" check would otherwise fire on every
        # clean pre-run-start abort.
        try:
            a.recorder.mark_marker("aborted_pre_run_start", reason)
        except Exception:
            pass
        try:
            a.recorder.write_metadata(a.extras)
        except Exception:
            logger.exception("[orch] metadata write failed in pre-run-start abort")
        a.recorder.close()
        self._record_run_outcome(a)
        self._active = None
        self.state = OrchestratorState.IDLE

    def _abort_after_run_start(self, reason: str) -> None:
        """run_start was already sent; do the normal end-of-run teardown."""
        a = self._active
        if a is None:
            self.state = OrchestratorState.IDLE
            return
        self._log(f"[orch] abort (post-run_start): {reason}")
        a.end_signal = ("aborted", reason)
        self.state = OrchestratorState.RUN_ENDING

    # --- bridge / auto callbacks --------------------------------------------

    def _on_predict_timing(self, sequence: int, send_ns: int, recv_ns: int) -> None:
        # Runs on the bridge recv thread; the main thread may be mid-teardown
        # and have closed the recorder. Catch defensively so a stray
        # late-arriving predict_result can never crash the recv thread
        # (BridgeClient does not restart it).
        a = self._active
        if a is None or a.recorder is None or a.recorder._closed:    # noqa: SLF001
            return
        try:
            kind = "warmup" if self.state == OrchestratorState.WARMUP else "run"
            a.recorder.record_predict_timing(sequence, send_ns, recv_ns, kind=kind)
        except Exception:
            logger.exception("[orch] predict_timing callback failed")

    def _maybe_cleanup_remote(self, a: _ActiveRun) -> None:
        """Run `rm -rf /tmp/experiments/<run_id>` on each remote host iff the
        operator opted in via orchestrator.cleanup_remote_after_pull. Default
        is OFF — bags stay on the Pi indefinitely so the operator has a
        recovery path if scp ever fails. Cheap in storage, expensive in lost
        data, so the default favours preservation.
        """
        if not self.cfg.cleanup_remote_after_pull:
            return
        try:
            a.bagset.cleanup_remote()
        except Exception:
            pass

    def _on_session_end(self, outcome: str, reason: str) -> None:
        a = self._active
        if a is None:
            return
        a.end_signal = (outcome, reason)

    def _on_bridge_error(self, msg: dict) -> None:
        """Bridge-side error during a predict round-trip. Used to short-circuit
        warmup when the planner rejects the input (shape mismatch etc.) — would
        otherwise stall for warmup_timeout_s before aborting on its own.
        """
        a = self._active
        if a is None:
            return
        message = str(msg.get("message", "unknown bridge error"))
        if self.state == OrchestratorState.WARMUP:
            # The warmup-inflight flag would never clear via the timing path
            # for an `error`, so the existing timeout would eventually catch
            # it — but we can do better and abort with a descriptive reason.
            a.warmup_inflight = False
            self._abort_before_run_start(f"warmup planner error: {message}")
        elif self.state in (OrchestratorState.RUN_ARMING,
                            OrchestratorState.RUN_ACTIVE):
            self._abort_after_run_start(f"planner error: {message}")
