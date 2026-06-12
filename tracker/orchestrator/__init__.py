"""orchestrator - Campaign + per-run lifecycle on top of the existing auto_driver.

This package is opt-in via `orchestrator.enabled: true` in the tracker config.
It runs *inside* main.py's existing capture loop (no new threads, no headless
mode) — `Orchestrator.tick()` is called each frame, drives an explicit state
machine, and uses auto_driver for the predict→goal cycle.

Public surface (everything else is implementation detail):
  - Orchestrator           — the coordinator, ticked from main.py
  - OrchestratorState      — phases the coordinator walks through
  - RunContext / SessionContext / Cell / MapEntry / QueueItem  — value types
  - CampaignPlan           — campaign_plan.yaml loader + randomized block design
  - validate_run_folder    — standalone CLI, importable as a function
"""

from __future__ import annotations

from orchestrator.state import (
    Cell,
    MapEntry,
    OrchestratorState,
    QueueItem,
    RunContext,
    SessionContext,
)

__all__ = [
    "Cell",
    "MapEntry",
    "OrchestratorState",
    "QueueItem",
    "RunContext",
    "SessionContext",
]
