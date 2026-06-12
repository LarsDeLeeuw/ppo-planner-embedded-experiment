"""
auto_recorder.py - In-memory recorder of auto-driver events for live debug HUD.

Parallel to auto_session_log.py: AutoSessionLog persists events to disk for
post-run analysis; AutoRecorder keeps just enough state in-memory for the
minimap overlay to render the current action trail and the latest energy map.

Two implementations:
- NullAutoRecorder: zero-cost no-op, used by default.
- InMemoryAutoRecorder: thread-safe action history + last energy map snapshot.

The recorder owns its own lock independent of AutoDriver._lock.  AutoDriver
calls the recorder while holding its own lock, but the recorder never calls
back into the driver, so there is no deadlock cycle.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class ActionEvent:
    """One predict_result the auto driver received.

    `direction` is the bridge-frame (dx, dy) for the action — the same vector
    used to draw the arrow on the live minimap.  `label` is the human-readable
    compass label (e.g. "NE").  Both are derivable from `action`, but stored
    on the event so the JSONL log is self-explanatory without lookups.
    """
    cycle: int
    retry: int
    robot_col: int
    robot_row: int
    action: int
    direction: tuple[int, int]
    label: str
    legal: bool
    reason: str


@dataclass(frozen=True)
class AutoSnapshot:
    """Read-only snapshot of recorder state for one render frame."""
    action_history: list[ActionEvent]
    last_energy_map: np.ndarray | None      # tracker frame (rows, cols), [0, 1]


class AutoRecorder(Protocol):
    name: str

    def session_start(self) -> None: ...
    def predict_sent(self, energy_map_tracker: np.ndarray) -> None: ...
    def predict_result(self, ev: ActionEvent) -> None: ...
    def clear(self) -> None: ...
    def snapshot(self) -> AutoSnapshot | None: ...


class NullAutoRecorder:
    """Default recorder.  All methods are no-ops."""

    name = "null"

    def session_start(self) -> None:
        pass

    def predict_sent(self, energy_map_tracker: np.ndarray) -> None:
        pass

    def predict_result(self, ev: ActionEvent) -> None:
        pass

    def clear(self) -> None:
        pass

    def snapshot(self) -> AutoSnapshot | None:
        return None


class InMemoryAutoRecorder:
    """Keeps action history + last energy map for live HUD rendering."""

    name = "in_memory"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._history: list[ActionEvent] = []
        self._last_energy: np.ndarray | None = None

    def session_start(self) -> None:
        # Lifecycle reset.  Functionally identical to clear(); separate
        # method for caller-side readability (lifecycle vs. user-initiated).
        with self._lock:
            self._history.clear()
            self._last_energy = None

    def predict_sent(self, energy_map_tracker: np.ndarray) -> None:
        copy = np.ascontiguousarray(energy_map_tracker.copy())
        with self._lock:
            self._last_energy = copy

    def predict_result(self, ev: ActionEvent) -> None:
        with self._lock:
            self._history.append(ev)

    def clear(self) -> None:
        with self._lock:
            self._history.clear()
            self._last_energy = None

    def snapshot(self) -> AutoSnapshot | None:
        with self._lock:
            if not self._history and self._last_energy is None:
                return None
            return AutoSnapshot(
                action_history=list(self._history),
                last_energy_map=(
                    self._last_energy.copy()
                    if self._last_energy is not None else None
                ),
            )
