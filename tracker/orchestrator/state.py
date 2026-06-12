"""orchestrator.state - Data structures + enums for the orchestrator state machine.

Pure dataclasses + an enum. No IO, no imports beyond stdlib and the config types
the orchestrator reads. Kept in its own module so every other orchestrator
module can depend on it without circular-import gymnastics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class OrchestratorState(str, Enum):
    """High-level phases of the orchestrator. The tick() coordinator advances
    through these in order; ABORT short-circuits to teardown.

    A note on naming: BAGS_STARTING / BAGS_STOPPING / BAGS_PULLING wrap the
    SSH + scp work and are distinct from RUN_ACTIVE (where the auto_driver
    state machine is doing the cell-by-cell navigation).
    """
    IDLE = "idle"
    BAGS_STARTING = "bags_starting"
    WARMUP = "warmup"
    RUN_ARMING = "run_arming"
    RUN_ACTIVE = "run_active"
    RUN_ENDING = "run_ending"
    BAGS_STOPPING = "bags_stopping"
    PULLING = "pulling"
    WRITING_META = "writing_meta"
    SESSION_SWITCH = "session_switch"
    STOPPED_ERROR = "stopped_error"


@dataclass(frozen=True)
class RunContext:
    """Everything one run needs to be uniquely identified + reproducible.

    Built by the orchestrator at run start from session_config + the active
    map entry. Snapshot only — once a run starts, the context is frozen.
    """
    run_id: str
    session_id: str
    session_dir: Path                # ~/experiments/session-<id>/
    run_dir: Path                    # session_dir/runs/<run_id>/
    mode: str                        # "decentralized" | "centralized"
    planner: str                     # "ppo" | "astar_energy" | "astar_shortest"
    map_id: str
    scene_name: str                  # scenes/<scene_name>.json
    goal_cell: tuple[int, int] | None    # None => idle baseline
    start_cell: tuple[int, int]
    dry_run: bool = False
    is_baseline: bool = False
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
    )
    extras: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def make_run_id(planner: str, map_id: str, mode: str, *,
                    kind: str = "", session_ts: datetime | None = None,
                    trial: int | None = None) -> str:
        """Format: <YYYYMMDD-HHMMSS>-<planner>-<map>-<mode>[-<kind>][-t<NNN>].

        The optional `trial` suffix guarantees uniqueness when multiple trials
        of the same (cell, map) session start within the same wall-clock
        second — fast/dry runs and mock tests both hit that, and folders
        silently overwriting each other would be a data-loss bug.
        """
        ts = (session_ts or datetime.now()).strftime("%Y%m%d-%H%M%S")
        suffix = f"-{kind}" if kind else ""
        trial_part = f"-t{trial:03d}" if trial is not None else ""
        return f"{ts}-{planner}-{map_id}-{mode}{suffix}{trial_part}"


@dataclass(frozen=True)
class MapEntry:
    """One (map_id, goal_cell, scene_name) the campaign cycles through."""
    map_id: str
    scene_name: str
    goal_cell: tuple[int, int]
    start_cell: tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class Cell:
    """One (mode, planner) cell in the experimental design."""
    mode: str
    planner: str

    @property
    def cell_id(self) -> str:
        return f"{self.mode}-{self.planner}"


@dataclass
class SessionContext:
    """One session = one (mode, planner) cell. Holds the per-run queue."""
    session_id: str
    session_dir: Path
    cell: Cell
    queue: list[MapEntry]                            # remaining runs
    completed: list[str] = field(default_factory=list)   # run_ids
    aborted: list[str] = field(default_factory=list)


@dataclass
class QueueItem:
    """One scheduled item: either a real run or a baseline/dry-run.

    `session_id` is stamped at enqueue time (campaign path computes it from
    the resolved schedule; operator paths derive it from `_session_id_for`)
    so that 10-trials-per-session blocks all land in the same run folder
    parent rather than being scattered by mutable orchestrator state.
    """
    kind: str                        # "run" | "baseline" | "dry_run"
    cell: Cell
    map_entry: MapEntry | None       # None for baseline/dry-run that doesn't need a map
    session_id: str | None = None    # filled by enqueue_* / load_campaign
