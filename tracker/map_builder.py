"""
map_builder.py - Compose predict-ready maps from the scene + a live energy source.

Owns the tracker->bridge orientation flip for every exported map so the rest
of the pipeline never needs to know about it.  Adding another "blocking" scene
layer (no-go zones, etc.) is a one-line config change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coord_transform import tracker_cell_to_bridge, tracker_map_to_bridge
from energy_source import EnergySource
from grid import GridState
from scene import Scene


@dataclass(frozen=True)
class PredictPayload:
    """Bridge-frame maps + integer cell positions, ready to send.

    `energy_map_tracker` is the same energy values in tracker frame (row 0
    at top), kept around for in-memory consumers that want to render or log
    what we sampled — saves recomputing or un-flipping the bridge-frame
    list-of-lists.
    """
    obstacle_map: list[list[float]]
    energy_map:   list[list[float]]
    robot_pos:    tuple[int, int]
    goal_pos:     tuple[int, int]
    energy_map_tracker: np.ndarray


class MapBuilder:
    """Builds obstacle + energy maps from scene layers and an EnergySource."""

    def __init__(self, blocking_layers: list[str]) -> None:
        # Strip whitespace from env-var splits and drop empties.
        self._blocking_layers = [s.strip() for s in blocking_layers if s.strip()]

    @property
    def blocking_layers(self) -> list[str]:
        return list(self._blocking_layers)

    # -- individual builders --------------------------------------------------

    def build_obstacle_map(self, scene: Scene) -> np.ndarray:
        """(rows, cols) uint8; 1 where any blocking layer is marked."""
        mask = np.zeros((scene.rows, scene.cols), dtype=np.uint8)
        for name in self._blocking_layers:
            if name not in scene:
                continue
            mask |= (scene.get(name).grid != 0).astype(np.uint8)
        return mask

    def build_energy_map(
        self,
        source: EnergySource,
        frame: np.ndarray,
        grid: GridState,
        rows: int,
        cols: int,
    ) -> np.ndarray:
        """Sample the EnergySource and clip the result into [0, 1]."""
        arr = source.sample(frame, grid, rows, cols).astype(np.float64)
        return np.clip(arr, 0.0, 1.0)

    # -- predict payload ------------------------------------------------------

    def as_predict_payload(
        self,
        scene: Scene,
        energy: EnergySource,
        frame: np.ndarray,
        grid: GridState,
        robot_row: int,
        robot_col: int,
        goal_row: int,
        goal_col: int,
    ) -> PredictPayload:
        """Build a fully bridge-frame-converted payload in one call."""
        rows, cols = scene.rows, scene.cols

        obs_tracker = self.build_obstacle_map(scene)
        eng_tracker = self.build_energy_map(energy, frame, grid, rows, cols)

        obs_bridge = tracker_map_to_bridge(obs_tracker)
        eng_bridge = tracker_map_to_bridge(eng_tracker)

        return PredictPayload(
            obstacle_map=obs_bridge.astype(int).tolist(),
            energy_map=eng_bridge.astype(float).tolist(),
            robot_pos=tracker_cell_to_bridge(robot_row, robot_col, rows),
            goal_pos=tracker_cell_to_bridge(goal_row, goal_col, rows),
            energy_map_tracker=eng_tracker,
        )
