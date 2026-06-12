"""GridMap <-> numpy conversion helper shared by planner packages."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def gridmap_to_numpy(rows: int, cols: int, data: Sequence[float]) -> np.ndarray:
    """Convert a flat row-major GridMap payload into a 2D numpy array.

    Takes primitive (rows, cols, data) so this helper is independent of
    tb3_interfaces. Callers unpack the GridMap message themselves:
        gridmap_to_numpy(msg.rows, msg.cols, msg.data)
    """
    expected = int(rows) * int(cols)
    if len(data) != expected:
        raise ValueError(
            f"GridMap data length {len(data)} != "
            f"rows*cols ({rows}*{cols}={expected})"
        )
    return np.array(data, dtype=np.float64).reshape(int(rows), int(cols))
