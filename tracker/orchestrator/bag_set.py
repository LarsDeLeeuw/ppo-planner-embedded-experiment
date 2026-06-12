"""orchestrator.bag_set - The canonical per-mode topic table + multi-host BagSet.

The orchestrator records one .mcap per host involved. The topic list per
(host, mode) follows [experiment_logging_plan §3](
../../temp-context/experiment_logging_plan_2026-05-24.md) verbatim; we encode
it here as the single source of truth so any drift between this repo and the
ROS-side bag-record contract surfaces as a code review (not as silent topic
loss in a run).

Decentralized: tb3_bringup runs on the Pi -> the Pi bag carries nav data.
Centralized:   tb3_bringup runs on the desktop -> the desktop bag carries
               nav data; the Pi bag is power + thermal + diagnostics only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from orchestrator.ssh_bag import BagHost, SshBagError

logger = logging.getLogger(__name__)

# -- canonical topic table ---------------------------------------------------

POWER_THERMAL_DIAG = [
    "/power/solar", "/power/sbc", "/power/opencr",
    "/sbc/thermal",
    "/diagnostics/host",
]
NAV_TOPICS = [
    "/cmd_vel", "/odom", "/imu",
    "/grid_nav_node/grid_pose",
    "/grid_nav_node/loop_stats",
    "/grid_nav_node/last_result",
    "/move_to_grid/_action/feedback",
    "/move_to_grid/_action/status",
    "/planner/metrics",
    "/experiment/events",
    "/bridge/predict_timing",
    "/diagnostics/session",
]
DESKTOP_EXTRA = ["/desktop/rapl_energy_uj"]


def topics_for(host: str, mode: str) -> list[str]:
    """Return the topic list to record on `host` ('pi'|'desktop') under `mode`.

    Empty list means this host records nothing in this mode (e.g. desktop in
    decentralized) — the orchestrator skips it entirely.
    """
    if mode == "decentralized":
        if host == "pi":
            return POWER_THERMAL_DIAG + NAV_TOPICS
        return []                                # desktop is idle
    if mode == "centralized":
        if host == "pi":
            return POWER_THERMAL_DIAG
        if host == "desktop":
            return NAV_TOPICS + DESKTOP_EXTRA + ["/diagnostics/host"]
    raise ValueError(f"unknown mode: {mode!r}")


# -- multi-host BagSet -------------------------------------------------------

@dataclass
class BagAssignment:
    host: BagHost
    bag_name: str            # e.g. "robot.mcap"
    topics: list[str]


class BagSet:
    """Coordinates start/stop/pull across the bags relevant to one mode.

    Construction:
        BagSet.for_mode(mode, pi_host, desktop_host, run_id, max_duration_s)
    Then in order:
        bs.start_all()
        ...                  # run happens
        bs.stop_all()        # returns dict[bag_name, elapsed_s]
        bs.pull_all_into(run_dir)
        bs.cleanup_remote()
    """

    def __init__(self, run_id: str, max_duration_s: int,
                 assignments: list[BagAssignment]) -> None:
        self.run_id = run_id
        self.max_duration_s = int(max_duration_s)
        self.assignments = list(assignments)

    @classmethod
    def for_mode(cls, *, mode: str, pi: BagHost, desktop: BagHost,
                 run_id: str, max_duration_s: int) -> "BagSet":
        assignments: list[BagAssignment] = []
        if (pi_topics := topics_for("pi", mode)):
            assignments.append(BagAssignment(pi, "robot.mcap", pi_topics))
        if (desk_topics := topics_for("desktop", mode)):
            assignments.append(BagAssignment(desktop, "desktop.mcap", desk_topics))
        return cls(run_id, max_duration_s, assignments)

    # -- lifecycle -----------------------------------------------------------

    def start_all(self) -> None:
        started: list[BagAssignment] = []
        try:
            for a in self.assignments:
                a.host.start_recording(self.run_id, a.bag_name, a.topics,
                                       self.max_duration_s)
                started.append(a)
        except SshBagError:
            # Best-effort rollback so we don't leave half-started bags running
            # — they would keep recording until timeout and confuse the next run.
            for a in started:
                try:
                    a.host.stop_recording(self.run_id, a.bag_name)
                except Exception:
                    pass
            raise

    def stop_all(self) -> dict[str, float]:
        """Return {bag_name: elapsed_seconds}. Negative = stop failed/unknown."""
        elapsed: dict[str, float] = {}
        for a in self.assignments:
            try:
                elapsed[a.bag_name] = a.host.stop_recording(self.run_id, a.bag_name)
            except SshBagError as e:
                logger.error("stop %s failed: %s", a.bag_name, e)
                elapsed[a.bag_name] = -1.0
        return elapsed

    def pull_all_into(self, run_dir: Path) -> dict[str, Path | None]:
        """scp every recorded bag into run_dir. None = pull failed."""
        pulled: dict[str, Path | None] = {}
        for a in self.assignments:
            try:
                pulled[a.bag_name] = a.host.pull(self.run_id, a.bag_name, run_dir)
            except SshBagError as e:
                logger.error("pull %s failed: %s", a.bag_name, e)
                pulled[a.bag_name] = None
        return pulled

    def cleanup_remote(self) -> None:
        for a in self.assignments:
            a.host.cleanup_remote(self.run_id)

    def bag_capped(self, elapsed: dict[str, float]) -> bool:
        """True if any bag stopped at or past max_duration_s (-> outcome=aborted)."""
        return any(e >= self.max_duration_s for e in elapsed.values() if e >= 0)
