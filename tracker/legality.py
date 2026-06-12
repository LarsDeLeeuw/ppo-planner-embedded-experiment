"""
legality.py - Pluggable next-cell legality rules for the autonomous loop.

The AutoDriver evaluates a list of `LegalityRule`s against a proposed next
cell and refuses to send a goal if any rule rejects it.  New rules (e.g.
"not in no-go zone", "energy < threshold") land as new classes; AutoDriver
is untouched.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol

from grid import GridState
from scene import Scene


class LegalityVerdict(NamedTuple):
    legal: bool
    reason: str


class LegalityRule(Protocol):
    name: str

    def check(
        self,
        next_row: int,
        next_col: int,
        scene: Scene,
        grid: GridState,
    ) -> LegalityVerdict: ...


class InBoundsRule:
    """Rejects cells outside the locked grid."""
    name = "in_bounds"

    def check(
        self,
        next_row: int,
        next_col: int,
        scene: Scene,
        grid: GridState,
    ) -> LegalityVerdict:
        if 0 <= next_row < grid.rows and 0 <= next_col < grid.cols:
            return LegalityVerdict(True, "")
        return LegalityVerdict(False, "out_of_bounds")


class NotObstacleRule:
    """Rejects cells marked on a named scene layer (defaults to 'obstacles')."""

    def __init__(self, layer: str = "obstacles") -> None:
        self._layer = layer
        self.name = f"not_{layer}"

    def check(
        self,
        next_row: int,
        next_col: int,
        scene: Scene,
        grid: GridState,
    ) -> LegalityVerdict:
        if self._layer not in scene:
            return LegalityVerdict(True, "")
        if scene.get(self._layer).is_marked(next_col, next_row):
            return LegalityVerdict(False, f"is_{self._layer}")
        return LegalityVerdict(True, "")
