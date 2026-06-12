"""8-action direction table shared by planner packages.

Canonical mapping between discrete grid actions (0..7) and their
(dx, dy) direction vectors. dx is the X (East-West) delta and dy the
Y (North-South) delta, matching the convention pinned in GridMap.msg
and PredictAction.srv: GridMap rows index X, cols index Y.

    0 = (-1,  0) West        1 = ( 1,  0) East
    2 = ( 0, -1) South       3 = ( 0,  1) North
    4 = (-1, -1) SW          5 = (-1,  1) NW
    6 = ( 1, -1) SE          7 = ( 1,  1) NE

This module is the single source of truth. Planner packages MUST import
from here instead of re-defining the table locally.
"""

from __future__ import annotations


DIRECTION: dict[int, tuple[int, int]] = {
    0: (-1,  0),
    1: ( 1,  0),
    2: ( 0, -1),
    3: ( 0,  1),
    4: (-1, -1),
    5: (-1,  1),
    6: ( 1, -1),
    7: ( 1,  1),
}

ACTION_BY_DIRECTION: dict[tuple[int, int], int] = {v: k for k, v in DIRECTION.items()}
