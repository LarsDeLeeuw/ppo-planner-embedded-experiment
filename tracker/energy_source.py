"""
energy_source.py - Pluggable live-loop energy-map samplers for the planner.

Mirrors the GridAnalyzer protocol in analyzer.py but is scoped to the auto-
driver use case: samples the current camera frame into a (rows, cols) float
array in [0, 1] in tracker frame (row 0 = top).  No image saving, no overlay
rendering.  Additional modalities (thermal, signal, ...) land as new classes
with no AutoDriver changes.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from grid import GridState
from luminance_analyzer import LuminanceAnalyzer
from perspective import warp_to_square


class EnergySource(Protocol):
    name: str

    def sample(
        self,
        frame: np.ndarray,
        grid: GridState,
        rows: int,
        cols: int,
    ) -> np.ndarray: ...


class LuminanceEnergySource:
    """Warp the grid region to a square and run LuminanceAnalyzer on it."""

    name = "luminance"

    def __init__(self, analyzer: LuminanceAnalyzer, warp_size: int) -> None:
        self._analyzer = analyzer
        self._warp_size = warp_size

    def sample(
        self,
        frame: np.ndarray,
        grid: GridState,
        rows: int,
        cols: int,
    ) -> np.ndarray:
        warped, _ = warp_to_square(frame, grid.src_pts, self._warp_size)
        result = self._analyzer.analyze(warped, rows, cols)
        return result.grid.astype(np.float64)
