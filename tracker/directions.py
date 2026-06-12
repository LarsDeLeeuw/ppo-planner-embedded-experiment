"""
directions.py - Action code <-> direction vector lookup.

Mirrors src/tb3_planner_common/tb3_planner_common/directions.py on the ROS2
side so both ends agree on what action 3 means.  Pure data; no state, no I/O.

All directions are expressed in the bridge grid frame: +X = East (col),
+Y = North (row in bottom-up bridge coords).
"""

from __future__ import annotations


ACTION_TO_DELTA: dict[int, tuple[int, int]] = {
    0: (-1,  0),   # West
    1: ( 1,  0),   # East
    2: ( 0, -1),   # South
    3: ( 0,  1),   # North
    4: (-1, -1),   # SW
    5: (-1,  1),   # NW
    6: ( 1, -1),   # SE
    7: ( 1,  1),   # NE
}

ACTION_TO_LABEL: dict[int, str] = {
    0: "W", 1: "E", 2: "S", 3: "N",
    4: "SW", 5: "NW", 6: "SE", 7: "NE",
}


def delta(action: int) -> tuple[int, int]:
    """Bridge-frame (dx, dy) for an action code in 0..7."""
    return ACTION_TO_DELTA[action]


def label(action: int) -> str:
    """Human-readable compass label for an action code in 0..7."""
    return ACTION_TO_LABEL[action]
