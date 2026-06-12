"""
navigation.py — Pure navigation state for target cell selection.

Manages the selected target cell and builds navigation packets.
No I/O, no input handling — just state and logic.
"""

from __future__ import annotations
from dataclasses import dataclass

from robot import RobotState


@dataclass(frozen=True)
class TargetCell:
    """A discrete grid cell."""
    col: int
    row: int


@dataclass(frozen=True)
class NavigationPacket:
    """Serialisable command: current robot state + target centre."""
    robot_x: float
    robot_y: float
    robot_heading_deg: float
    target_x: float
    target_y: float

    def to_dict(self) -> dict:
        return {
            "robot_x": round(self.robot_x, 4),
            "robot_y": round(self.robot_y, 4),
            "robot_heading_deg": round(self.robot_heading_deg, 2),
            "target_x": self.target_x,
            "target_y": self.target_y,
        }


class NavigationState:
    """Target cell manager, parameterised with grid dimensions."""

    def __init__(self, cols: int, rows: int) -> None:
        self._cols = cols
        self._rows = rows
        self._target: TargetCell | None = None
        self._changed = False

    # -- read-only properties ------------------------------------------------

    @property
    def target(self) -> TargetCell | None:
        return self._target

    # -- mutations -----------------------------------------------------------

    def set_target(self, col: int, row: int) -> None:
        """Set target directly (e.g. from a click after coord conversion)."""
        if 0 <= col < self._cols and 0 <= row < self._rows:
            self._target = TargetCell(col, row)
            self._changed = True

    def clear(self) -> None:
        """Reset target (e.g. on grid re-calibration)."""
        self._target = None
        self._changed = False

    # -- output --------------------------------------------------------------

    def build_packet(self, robot: RobotState) -> NavigationPacket | None:
        """Build a navigation packet from current robot state and target."""
        if self._target is None:
            return None
        return NavigationPacket(
            robot_x=robot.grid_x,
            robot_y=robot.grid_y,
            robot_heading_deg=robot.heading_deg,
            target_x=self._target.col + 0.5,
            target_y=self._target.row + 0.5,
        )

    def consume_changed(self) -> bool:
        """Return True once after target changes, then reset."""
        if self._changed:
            self._changed = False
            return True
        return False
