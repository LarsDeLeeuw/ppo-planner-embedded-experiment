"""orchestrator.campaign - Load + validate campaign_plan.yaml and resolve
session order.

Schema (v1):

    schema_version: 1
    random_seed: 42
    runs_per_session: 10
    cells:
      - {mode: decentralized, planner: ppo}
      - {mode: decentralized, planner: astar_energy}
      - {mode: centralized,   planner: ppo}
      - {mode: centralized,   planner: astar_energy}
    maps:
      - {map_id: map1, scene_name: map1, goal_cell: [8, 9], start_cell: [0, 0]}
      - {map_id: map2, scene_name: map2, goal_cell: [9, 5], start_cell: [0, 0]}
    session_order: randomized        # or "as_written"

Model:
  - One session = one (cell, map) pair + `runs_per_session` identical trials.
    The map layout is fixed for the whole session so the operator only has
    to physically rearrange obstacles at session boundaries, not between
    every trial. Total runs = len(cells) * len(maps) * runs_per_session.

  - Session ORDER is MAP-MAJOR: the schedule iterates maps in plan order,
    and within each map block iterates cells (in random order when
    `session_order=randomized`, plan order when `as_written`). So the
    operator sets up map 1, finishes all the cells on it, swaps to map 2,
    etc. Maps are never revisited. The same (map_id, goal_cell) set is
    held constant across cells so [plan §12.6](
    ../../temp-context/experiment_logging_plan_2026-05-24.md) pairing
    remains valid.

The resolved schedule is written next to the plan as
`campaign_plan.resolved.yaml` so the run order is reproducible from the
artifact alone (handover_orchestrator §1.5).
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from orchestrator.state import Cell, MapEntry

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
VALID_MODES = {"decentralized", "centralized"}
VALID_PLANNERS = {"ppo", "astar_energy", "astar_shortest"}
VALID_ORDERS = {"randomized", "as_written"}


class CampaignError(ValueError):
    """campaign_plan.yaml is malformed or referenced files are missing."""


@dataclass(frozen=True)
class ScheduledSession:
    """One session in the resolved order: ONE (cell, map) pair + N trials.

    The map is fixed for the duration of the session — the operator sets up
    obstacles once at the start and only needs to reposition the robot
    between the N trials. `trials` is the run_count for THIS session.
    """
    index: int                                   # 1-based position in schedule
    cell: Cell
    map: MapEntry
    trials: int


@dataclass
class CampaignPlan:
    plan_path: Path
    schema_version: int
    random_seed: int
    runs_per_session: int
    cells: list[Cell]
    maps: list[MapEntry]
    session_order: str
    schedule: list[ScheduledSession] = field(default_factory=list)

    # -- loading -------------------------------------------------------------

    @classmethod
    def load(cls, path: Path | str) -> "CampaignPlan":
        p = Path(path)
        if not p.exists():
            raise CampaignError(f"campaign plan not found: {p}")
        try:
            doc = yaml.safe_load(p.read_text()) or {}
        except yaml.YAMLError as e:
            raise CampaignError(f"{p}: malformed yaml: {e}") from e
        if not isinstance(doc, dict):
            raise CampaignError(f"{p}: top-level must be a mapping")

        sv = doc.get("schema_version")
        if sv != SCHEMA_VERSION:
            raise CampaignError(
                f"{p}: schema_version={sv!r}, expected {SCHEMA_VERSION}")

        try:
            seed = int(doc.get("random_seed", 0))
            # New schema field. We intentionally do not accept the old
            # `runs_per_cell` silently — emit a clear migration error so
            # outdated plans don't run with the wrong session topology.
            if "runs_per_cell" in doc:
                raise CampaignError(
                    f"{p}: `runs_per_cell` was removed; use `runs_per_session` "
                    "(number of identical trials per (cell, map) session). "
                    "See orchestrator/example_campaign_plan.yaml.")
            runs_per_session = int(doc.get("runs_per_session", 1))
            if runs_per_session < 1:
                raise CampaignError("runs_per_session must be >= 1")
            cells_raw = doc["cells"]
            maps_raw = doc["maps"]
            order = doc.get("session_order", "as_written")
            if order not in VALID_ORDERS:
                raise CampaignError(f"session_order={order!r} not in {VALID_ORDERS}")
        except (KeyError, TypeError, ValueError) as e:
            raise CampaignError(f"{p}: invalid: {e}") from e

        cells: list[Cell] = []
        for i, c in enumerate(cells_raw):
            if not isinstance(c, dict) or "mode" not in c or "planner" not in c:
                raise CampaignError(f"{p}: cells[{i}] missing mode/planner")
            if c["mode"] not in VALID_MODES:
                raise CampaignError(f"{p}: cells[{i}].mode={c['mode']!r} not in {VALID_MODES}")
            if c["planner"] not in VALID_PLANNERS:
                raise CampaignError(f"{p}: cells[{i}].planner={c['planner']!r} not in {VALID_PLANNERS}")
            cells.append(Cell(mode=c["mode"], planner=c["planner"]))
        if not cells:
            raise CampaignError(f"{p}: at least one cell required")

        maps: list[MapEntry] = []
        for i, m in enumerate(maps_raw):
            if not isinstance(m, dict):
                raise CampaignError(f"{p}: maps[{i}] not a mapping")
            try:
                maps.append(MapEntry(
                    map_id=str(m["map_id"]),
                    scene_name=str(m.get("scene_name", m["map_id"])),
                    goal_cell=tuple(int(x) for x in m["goal_cell"]),
                    start_cell=tuple(int(x) for x in m.get("start_cell", [0, 0])),
                ))
            except (KeyError, TypeError, ValueError) as e:
                raise CampaignError(f"{p}: maps[{i}] invalid: {e}") from e
        if not maps:
            raise CampaignError(f"{p}: at least one map required")

        plan = cls(plan_path=p, schema_version=sv, random_seed=seed,
                   runs_per_session=runs_per_session, cells=cells, maps=maps,
                   session_order=order)
        plan.schedule = plan._build_schedule()
        return plan

    # -- scene-file existence check ------------------------------------------

    def validate_scene_files(self, scene_dir: Path) -> list[str]:
        """Return human-readable warnings about missing scene files. Empty list
        = all maps resolve to an existing scene under `scene_dir`."""
        out: list[str] = []
        for m in self.maps:
            p = scene_dir / f"{m.scene_name}.json"
            if not p.exists():
                out.append(f"map {m.map_id!r}: scene file {p} missing")
        return out

    # -- schedule build ------------------------------------------------------

    def _build_schedule(self) -> list[ScheduledSession]:
        """Schedule = product(maps, cells), map-major ordering.

        Outer iteration walks maps in plan order — never revisited — so the
        operator sets up each map's physical layout once. Inner iteration
        walks the cells, in random order per map when session_order is
        "randomized", in plan order when "as_written". Each scheduled
        session carries `runs_per_session` identical trials of its
        (cell, map) pair.
        """
        rng = random.Random(self.random_seed)
        sessions: list[ScheduledSession] = []
        for m in self.maps:                          # maps OUTER — physical setup order
            cells_block = list(self.cells)
            if self.session_order == "randomized":
                rng.shuffle(cells_block)
            for c in cells_block:
                sessions.append(ScheduledSession(
                    index=len(sessions) + 1, cell=c, map=m,
                    trials=self.runs_per_session,
                ))
        return sessions

    # -- resolved-plan emit --------------------------------------------------

    def write_resolved(self, out_path: Path | None = None) -> Path:
        """Persist the resolved session order so it's reproducible from disk."""
        out = out_path or self.plan_path.with_name("campaign_plan.resolved.yaml")
        doc = {
            "schema_version": self.schema_version,
            "random_seed": self.random_seed,
            "runs_per_session": self.runs_per_session,
            "session_order": self.session_order,
            "schedule": [
                {
                    "index": s.index,
                    "mode": s.cell.mode,
                    "planner": s.cell.planner,
                    "map_id": s.map.map_id,
                    "scene_name": s.map.scene_name,
                    "goal_cell": list(s.map.goal_cell),
                    "start_cell": list(s.map.start_cell),
                    "trials": s.trials,
                }
                for s in self.schedule
            ],
        }
        out.write_text(yaml.safe_dump(doc, sort_keys=False))
        return out


def example_plan() -> dict[str, Any]:
    """Return an example plan dict (used by docs / smoke tests)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "random_seed": 42,
        "runs_per_session": 10,
        "session_order": "randomized",
        "cells": [
            {"mode": "decentralized", "planner": "ppo"},
            {"mode": "decentralized", "planner": "astar_energy"},
            {"mode": "centralized",   "planner": "ppo"},
            {"mode": "centralized",   "planner": "astar_energy"},
        ],
        "maps": [
            {"map_id": "map1", "scene_name": "map1",
             "goal_cell": [8, 9], "start_cell": [0, 0]},
            {"map_id": "map2", "scene_name": "map2",
             "goal_cell": [9, 5], "start_cell": [0, 0]},
            {"map_id": "map3", "scene_name": "map3",
             "goal_cell": [9, 9], "start_cell": [0, 0]},
        ],
    }
