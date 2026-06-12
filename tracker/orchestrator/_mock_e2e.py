"""orchestrator._mock_e2e - End-to-end integration test for the orchestrator wiring.

Runs entirely on the laptop with no robot, no SSH, no real bridge. Drives the
state machine through:
  - a regular movement run (IDLE -> ... -> WRITING_META -> IDLE), success
  - a baseline run (no auto_driver, BASELINE_DURATION_S elapses)
  - an aborted-by-cancel run (cancel during WARMUP)

Asserts every produced runs/<run_id>/ passes validate_run_folder. Then
spot-checks key metadata.yaml fields (outcome, goal_cell in bridge frame,
warmup_inferences_us populated for the success path, etc.).

Run via:
  cd tracker
  .venv/Scripts/python.exe -m orchestrator._mock_e2e [outdir]
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                                # noqa: E402
from orchestrator.orchestrator import Orchestrator, BASELINE_DURATION_S  # noqa: E402
from orchestrator.state import Cell, MapEntry                # noqa: E402
from orchestrator.validate_run_folder import validate        # noqa: E402

GRID = 10
CELL = Cell(mode="decentralized", planner="ppo")
MAP = MapEntry(map_id="mapA", scene_name="mapA",
               goal_cell=(8, 9), start_cell=(0, 0))


# -- fakes --------------------------------------------------------------------

class FakeBridge:
    """Captures sent messages so we can assert on the experiment_event stream.

    Also drives _on_predict_timing synchronously for warmup predicts so the
    state machine advances without a real network round-trip.
    """
    def __init__(self):
        self.timing_cb = None
        self.error_cb = None
        self.events: list[tuple[str, str, dict]] = []
        self.predicts: list[int] = []
        self._next_seq_to_ack = 1

    def on_predict_timing(self, cb): self.timing_cb = cb
    def on(self, msg_type, cb):
        if msg_type == "error":
            self.error_cb = cb

    def send_experiment_event(self, event_type, run_id, payload=None):
        self.events.append((event_type, run_id, payload or {}))

    def send_predict(self, *, sequence=None, **_):
        # Synchronously ack with a fake timing — matches the bridge_client
        # "one outstanding" gate. ~1500us inference, ~18ms RTT.
        self.predicts.append(int(sequence) if sequence else 0)
        if self.timing_cb and sequence:
            send_ns = time.monotonic_ns()
            recv_ns = send_ns + 18_000_000
            self.timing_cb(int(sequence), send_ns, recv_ns)

    def send_pose(self, *_a, **_k): pass
    def send_goal(self, *_a, **_k): pass
    def send_cancel(self, *_a, **_k): pass


class FakeAuto:
    """Acts like AutoDriver but resolves on the FIRST tick after start.

    For the success path the orchestrator's _tick_run_active just polls
    is_active(); we flip to inactive once start has been called and one tick
    elapsed, simulating an instant single-step run.
    """
    def __init__(self):
        self.session_end_cbs: list = []
        self._active = False
        self._ticks_since_start = 0
        self._next_outcome: tuple[str, str] = ("success", "robot reached goal")

    def on_session_end(self, cb): self.session_end_cbs.append(cb)
    def is_active(self): return self._active

    def start(self, *_a, **_kw):
        self._active = True
        self._ticks_since_start = 0
        return True, ""

    def stop(self, reason, is_error=True):
        if not self._active:
            return
        self._active = False
        outcome = "failure" if is_error else ("success" if "goal" in reason else "aborted")
        for cb in self.session_end_cbs:
            cb(outcome, reason)

    # Advance one synthetic tick — called from the test harness, not from
    # AutoDriver itself.  When ticks_since_start reaches the trigger, fire
    # session_end as if the goal was reached.
    def harness_tick(self, fire_after: int = 2):
        if self._active and self._ticks_since_start >= fire_after:
            outcome, reason = self._next_outcome
            self._active = False
            for cb in self.session_end_cbs:
                cb(outcome, reason)
            return True
        self._ticks_since_start += 1
        return False


class FakeLayer:
    def __init__(self):
        self.grid = np.zeros((GRID, GRID), dtype=np.uint8)
    def clear(self): self.grid[:] = 0
    def set(self, col, row, value):
        self.grid[row, col] = 1 if value else 0


class FakeScene:
    cols = GRID
    rows = GRID
    def __init__(self):
        self._layers = {"obstacles": FakeLayer(), "goal": FakeLayer()}
        # Sprinkle some obstacles so the snapshot isn't all zeros.
        self._layers["obstacles"].grid[3, 6] = 1
        self._layers["obstacles"].grid[6, 3] = 1
    def __contains__(self, k): return k in self._layers
    def get(self, name): return self._layers[name]


# -- driver -------------------------------------------------------------------

def _build_orch(outdir: Path, *, summarize_python: str | None = None):
    """Construct an Orchestrator pointed at outdir with ssh_mock + fake bridge."""
    config.ORCH_CFG = config.ORCH_CFG.model_copy(update={
        "enabled": True,
        "ssh_mock": True,
        "experiments_root": str(outdir),
        "warmup_count": 3,
        "summarize_after_run": False,                 # off in tests
        "summarize_python": summarize_python,
    })
    app_cfg = config.APP_CFG.model_copy(update={"orchestrator": config.ORCH_CFG})

    bridge = FakeBridge()
    auto = FakeAuto()
    cb = {"scene": FakeScene()}
    orch = Orchestrator(
        app_cfg=app_cfg, bridge=bridge, auto=auto,
        cb_state=cb, scene_loader=lambda _p: FakeScene(),
        corner_marker_ids=[1, 2, 3, 4], scene_dir=Path("scenes"),
    )
    return orch, bridge, auto, cb


def _pump_until_idle(orch, bridge, auto, *, max_ticks=2000, baseline_speedup=True):
    """Tick the orchestrator until it returns to IDLE.

    Baseline runs use BASELINE_DURATION_S which is real seconds; if
    baseline_speedup, we monkeypatch orch._clock to advance a virtual clock
    so we don't actually sleep 60s in a test.
    """
    if baseline_speedup:
        virt = [0.0]
        def fake_clock(): return virt[0]
        orch._clock = fake_clock
    for i in range(max_ticks):
        from orchestrator.state import OrchestratorState
        # Done iff we're back in IDLE, nothing is in flight, AND the queue
        # has been fully drained. The queue check matters in auto_advance
        # campaigns: at function entry the orchestrator is IDLE with N
        # queued items waiting for the next tick to pop the first one.
        if (orch.state == OrchestratorState.IDLE
                and orch._active is None
                and not orch.queue):
            return i
        if baseline_speedup:
            virt[0] += 0.5      # 0.5s per tick — baseline finishes in ~120 ticks
        auto.harness_tick(fire_after=2)
        orch.tick(frame=None, robot_state=None, grid=None)
    raise RuntimeError(f"orchestrator did not return to IDLE after {max_ticks} ticks "
                       f"(state={orch.state.value}, active={orch._active is not None}, "
                       f"queue={len(orch.queue)})")


# -- scenarios ----------------------------------------------------------------

def scenario_success(outdir: Path) -> Path:
    """Happy-path run: bags + warmup + auto + teardown all succeed."""
    print("\n=== scenario_success ===")
    orch, bridge, auto, _cb = _build_orch(outdir)
    orch.enqueue_run(CELL, MAP)
    orch._begin_next()
    ticks = _pump_until_idle(orch, bridge, auto)
    print(f"  reached IDLE in {ticks} ticks")
    all_runs = []
    for sess in outdir.glob("session-*"):
        runs_dir = sess / "runs"
        if runs_dir.exists():
            all_runs.extend(runs_dir.iterdir())
    return max(all_runs, key=lambda p: p.stat().st_mtime)


def scenario_baseline(outdir: Path) -> Path:
    """Idle baseline — no auto_driver, just BASELINE_DURATION_S elapses."""
    print("\n=== scenario_baseline ===")
    orch, bridge, auto, _cb = _build_orch(outdir)
    orch.enqueue_baseline(CELL)
    orch._begin_next()
    ticks = _pump_until_idle(orch, bridge, auto)
    print(f"  reached IDLE in {ticks} ticks")
    # Most recent run dir
    # Sessions now split per (cell, map) AND baselines have no map_id, so
    # multiple session folders may exist. Find the most recent run dir
    # across all of them by mtime.
    all_runs = []
    for sess in outdir.glob("session-*"):
        runs_dir = sess / "runs"
        if runs_dir.exists():
            all_runs.extend(runs_dir.iterdir())
    return max(all_runs, key=lambda p: p.stat().st_mtime)


def scenario_shutdown_midrun(outdir: Path) -> Path:
    """F16: app exit while RUN_ACTIVE must call orch.shutdown() which tears
    down bags + writes metadata + sends run_end. Without the fix the finally
    block would skip the orchestrator entirely and leave bags running."""
    print("\n=== scenario_shutdown_midrun ===")
    orch, bridge, auto, _cb = _build_orch(outdir)
    orch.enqueue_run(CELL, MAP)
    orch._begin_next()
    # Pump through WARMUP into RUN_ACTIVE but NOT past it.
    from orchestrator.state import OrchestratorState
    for _ in range(20):
        orch.tick(frame=None, robot_state=None, grid=None)
        if orch.state == OrchestratorState.RUN_ACTIVE:
            break
    assert orch.state == OrchestratorState.RUN_ACTIVE, orch.state
    # Now simulate app exit.
    orch.shutdown(reason="app exit")
    assert orch.state == OrchestratorState.IDLE, orch.state
    assert orch._active is None
    # Sessions now split per (cell, map) AND baselines have no map_id, so
    # multiple session folders may exist. Find the most recent run dir
    # across all of them by mtime.
    all_runs = []
    for sess in outdir.glob("session-*"):
        runs_dir = sess / "runs"
        if runs_dir.exists():
            all_runs.extend(runs_dir.iterdir())
    return max(all_runs, key=lambda p: p.stat().st_mtime)


def scenario_wallclock_timeout(outdir: Path) -> Path:
    """F18: real run that hangs in RUN_ACTIVE longer than run_max_duration_s
    must abort cleanly with outcome=aborted (not wedge the campaign)."""
    print("\n=== scenario_wallclock_timeout ===")
    orch, bridge, auto, _cb = _build_orch(outdir)
    # Hang the run: never trigger auto session_end.
    auto.harness_tick = lambda fire_after=2: False
    orch.enqueue_run(CELL, MAP)
    orch._begin_next()
    # Virtual clock so we don't actually sleep 240s.
    virt = [0.0]
    orch._clock = lambda: virt[0]
    from orchestrator.state import OrchestratorState
    for i in range(2000):
        if orch.state == OrchestratorState.IDLE and orch._active is None:
            break
        # Reach RUN_ACTIVE first, then jump the clock past the budget.
        if orch.state == OrchestratorState.RUN_ACTIVE:
            virt[0] += 30.0      # 30s per tick — backstop fires after ~9 ticks
        orch.tick(frame=None, robot_state=None, grid=None)
    else:
        raise RuntimeError("wallclock scenario never reached IDLE")
    # Sessions now split per (cell, map) AND baselines have no map_id, so
    # multiple session folders may exist. Find the most recent run dir
    # across all of them by mtime.
    all_runs = []
    for sess in outdir.glob("session-*"):
        runs_dir = sess / "runs"
        if runs_dir.exists():
            all_runs.extend(runs_dir.iterdir())
    return max(all_runs, key=lambda p: p.stat().st_mtime)


def scenario_campaign_map_major(outdir: Path) -> list[Path]:
    """Load a small campaign plan and drain all sessions/trials through the
    orchestrator. Verifies map-major ordering and one-session-per-(cell,map).

    Tiny plan: 1 cell × 2 maps × 3 trials = 6 runs across 2 sessions.
    Expected run order on disk: map1 × 3, then map2 × 3.
    """
    print("\n=== scenario_campaign_map_major ===")
    # Isolate to its own subdir so earlier scenarios' run folders don't
    # leak into the assertions below (they use mapA/mapB too).
    outdir = outdir / "_campaign"
    outdir.mkdir(parents=True, exist_ok=True)
    plan_path = outdir / "plan.yaml"
    plan_path.write_text(
        "schema_version: 1\n"
        "random_seed: 7\n"
        "runs_per_session: 3\n"
        "session_order: as_written\n"
        "cells:\n"
        "  - {mode: decentralized, planner: ppo}\n"
        "maps:\n"
        "  - {map_id: mapA, scene_name: mapA, goal_cell: [8, 9], start_cell: [0, 0]}\n"
        "  - {map_id: mapB, scene_name: mapB, goal_cell: [4, 5], start_cell: [0, 0]}\n"
    )
    orch, bridge, auto, _cb = _build_orch(outdir)
    orch.load_campaign(plan_path)
    assert len(orch.queue) == 6, len(orch.queue)
    orch.set_auto_advance(True)              # drain all 6 unattended
    ticks = _pump_until_idle(orch, bridge, auto, max_ticks=5000)
    print(f"  drained 6-run campaign in {ticks} ticks")

    # Collect run dirs in the order they appear on disk.
    all_runs: list[Path] = []
    for sess in sorted(outdir.glob("session-*-mapA*"))  \
               + sorted(outdir.glob("session-*-mapB*")):
        runs_dir = sess / "runs"
        if runs_dir.exists():
            all_runs.extend(
                sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime))
    assert len(all_runs) == 6, len(all_runs)
    return all_runs


def scenario_cancel_warmup(outdir: Path) -> Path:
    """Cancel during WARMUP — must produce outcome=aborted via _abort_before_run_start.

    We stop the warmup-ack loop by suppressing on_predict_timing so warmups stall,
    then call cancel_current. F1 fix must short-circuit to abort even though
    auto.is_active() is False during WARMUP.
    """
    print("\n=== scenario_cancel_warmup ===")
    orch, bridge, auto, _cb = _build_orch(outdir)
    # Suppress the ack so warmup gets stuck waiting for the timing callback.
    bridge.timing_cb = None
    orch.enqueue_run(CELL, MAP)
    orch._begin_next()
    # Tick a couple of times so we pass BAGS_STARTING into WARMUP.
    for _ in range(3):
        orch.tick(frame=None, robot_state=None, grid=None)
    from orchestrator.state import OrchestratorState
    assert orch.state == OrchestratorState.WARMUP, orch.state
    # Now cancel — must short-circuit via _abort_before_run_start.
    orch.cancel_current("test cancel")
    # No run_start event should have been sent over the bridge.
    sent_run_start = [e for e in bridge.events if e[0] == "run_start"]
    assert not sent_run_start, f"run_start sent during pre-RUN_ACTIVE cancel: {sent_run_start}"
    assert orch.state == OrchestratorState.IDLE, orch.state
    # Find the most recent run dir
    # Sessions now split per (cell, map) AND baselines have no map_id, so
    # multiple session folders may exist. Find the most recent run dir
    # across all of them by mtime.
    all_runs = []
    for sess in outdir.glob("session-*"):
        runs_dir = sess / "runs"
        if runs_dir.exists():
            all_runs.extend(runs_dir.iterdir())
    return max(all_runs, key=lambda p: p.stat().st_mtime)


# -- assertions ---------------------------------------------------------------

def _assert_validates(run_dir: Path, expected_outcome: str,
                       expected_goal_cell_tracker: tuple[int, int] | None = None):
    """Assert the run folder is well-formed AND outcome matches.

    `expected_goal_cell_tracker` (col, row in tracker frame) is optional —
    when set, also verifies the tracker→bridge conversion landed correctly
    in metadata.yaml. Pass MAP's goal_cell here for scenarios that use it;
    skip for campaign scenarios that exercise multiple maps.
    """
    print(f"  validating {run_dir.name} (expecting outcome={expected_outcome})")
    ok, errors, warnings = validate(run_dir)
    for w in warnings:
        print(f"    warn: {w}")
    if not ok:
        for e in errors:
            print(f"    ERROR: {e}")
        raise AssertionError(f"validate_run_folder failed for {run_dir}")
    meta = yaml.safe_load((run_dir / "metadata.yaml").read_text())
    assert meta["outcome"] == expected_outcome, (meta["outcome"], expected_outcome)
    if expected_goal_cell_tracker is not None and meta.get("goal_cell") is not None:
        col, row = expected_goal_cell_tracker
        expected = [col, (GRID - 1) - row]
        assert meta["goal_cell"] == expected, (
            f"goal_cell={meta['goal_cell']} expected {expected} (bridge frame)")
    print(f"    -> OK; outcome={meta['outcome']}, goal_cell={meta.get('goal_cell')}, "
          f"warmups={len(meta.get('warmup_inferences_us', []))}")


def main(argv: list[str] | None = None) -> int:
    args = argv or sys.argv[1:]
    outdir = Path(args[0]) if args else Path("_mock_e2e_out")
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)

    rd1 = scenario_success(outdir)
    _assert_validates(rd1, expected_outcome="success",
                      expected_goal_cell_tracker=MAP.goal_cell)
    # Success path: warmup_inferences_us should have 3 entries (warmup_count=3).
    meta = yaml.safe_load((rd1 / "metadata.yaml").read_text())
    assert len(meta["warmup_inferences_us"]) == 3, meta["warmup_inferences_us"]
    # obstacle.npy snapshot exists + is non-trivial
    obs = np.load(rd1 / "maps" / "obstacle.npy")
    assert obs.shape == (GRID, GRID), obs.shape
    assert int(obs.sum()) >= 1, "obstacle snapshot empty"

    rd2 = scenario_baseline(outdir)
    _assert_validates(rd2, expected_outcome="success")
    meta = yaml.safe_load((rd2 / "metadata.yaml").read_text())
    assert meta["goal_cell"] is None, meta["goal_cell"]

    rd3 = scenario_cancel_warmup(outdir)
    _assert_validates(rd3, expected_outcome="aborted",
                      expected_goal_cell_tracker=MAP.goal_cell)

    rd4 = scenario_shutdown_midrun(outdir)
    _assert_validates(rd4, expected_outcome="aborted",
                      expected_goal_cell_tracker=MAP.goal_cell)

    rd5 = scenario_wallclock_timeout(outdir)
    _assert_validates(rd5, expected_outcome="aborted",
                      expected_goal_cell_tracker=MAP.goal_cell)
    meta = yaml.safe_load((rd5 / "metadata.yaml").read_text())
    assert "wallclock" in (meta.get("end_reason") or ""), meta.get("end_reason")

    rds6 = scenario_campaign_map_major(outdir)
    # Verify map-major ordering: first 3 runs on mapA, next 3 on mapB.
    map_ids = []
    for rd in rds6:
        m = yaml.safe_load((rd / "metadata.yaml").read_text())
        _assert_validates(rd, expected_outcome="success")
        map_ids.append(m["map_id"])
    assert map_ids == ["mapA", "mapA", "mapA", "mapB", "mapB", "mapB"], map_ids
    # Verify one session per (cell, map), three trials each, both with the
    # same session_id within their group.
    sess_ids_A = {yaml.safe_load((r / "metadata.yaml").read_text())["session_id"]
                  for r in rds6[:3]}
    sess_ids_B = {yaml.safe_load((r / "metadata.yaml").read_text())["session_id"]
                  for r in rds6[3:]}
    assert len(sess_ids_A) == 1 and len(sess_ids_B) == 1, (sess_ids_A, sess_ids_B)
    assert sess_ids_A != sess_ids_B, "mapA and mapB must live in different sessions"
    print(f"  mapA session_id: {next(iter(sess_ids_A))}")
    print(f"  mapB session_id: {next(iter(sess_ids_B))}")

    print("\nALL MOCK E2E SCENARIOS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
