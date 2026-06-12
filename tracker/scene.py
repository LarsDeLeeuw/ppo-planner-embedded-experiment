"""
scene.py — Semantic annotation layers on the locked grid.

A `Scene` owns the grid dimensions and a dict of named `Layer`s.  Each layer is
a 2D uint8 array with a rendering color, a keyboard selector, and a cardinality
policy (MANY: any number of cells may be marked; SINGLE: at most one cell, so
setting a new cell clears the previous).

Pure state + disk I/O, no OpenCV or camera coupling.  File I/O lives in module-
level functions so `Scene` stays side-effect-free.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Iterable, Literal

import numpy as np


# ---------------------------------------------------------------------------
# Layer
# ---------------------------------------------------------------------------

class Cardinality(str, Enum):
    MANY = "many"
    SINGLE = "single"


class Style(str, Enum):
    FILL = "fill"       # semi-transparent polygon fill across the whole cell
    CIRCLE = "circle"   # opaque filled circle at cell center (colorblind-friendly)


@dataclass
class Layer:
    """One semantic annotation layer (obstacles, goal, ...)."""
    name: str
    color: tuple[int, int, int]   # BGR
    key: int                      # keycode that selects this layer in mark mode
    cardinality: Cardinality
    grid: np.ndarray              # shape (rows, cols), dtype uint8
    style: Style = Style.FILL     # render style used by overlay.draw_scene

    # -- mutations -----------------------------------------------------------

    def toggle(self, col: int, row: int) -> None:
        if not self._in_bounds(col, row):
            return
        currently = bool(self.grid[row, col])
        if currently:
            self.grid[row, col] = 0
            return
        if self.cardinality == Cardinality.SINGLE:
            self.grid[:] = 0
        self.grid[row, col] = 1

    def set(self, col: int, row: int, value: bool) -> None:
        if not self._in_bounds(col, row):
            return
        if value:
            if self.cardinality == Cardinality.SINGLE:
                self.grid[:] = 0
            self.grid[row, col] = 1
        else:
            self.grid[row, col] = 0

    def clear(self) -> None:
        self.grid[:] = 0

    # -- queries -------------------------------------------------------------

    def is_marked(self, col: int, row: int) -> bool:
        if not self._in_bounds(col, row):
            return False
        return bool(self.grid[row, col])

    def marked_cells(self) -> Iterable[tuple[int, int]]:
        rows, cols = np.where(self.grid != 0)
        return ((int(c), int(r)) for r, c in zip(rows, cols))

    def count(self) -> int:
        return int(np.count_nonzero(self.grid))

    # -- internals -----------------------------------------------------------

    def _in_bounds(self, col: int, row: int) -> bool:
        r, c = self.grid.shape
        return 0 <= col < c and 0 <= row < r


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

@dataclass
class Scene:
    """Grid-sized container for named annotation layers."""
    cols: int
    rows: int
    _layers: dict[str, Layer] = field(default_factory=dict)

    def register(
        self,
        name: str,
        color: tuple[int, int, int],
        key: int,
        cardinality: Cardinality,
        style: Style = Style.FILL,
    ) -> Layer:
        if name in self._layers:
            raise ValueError(f"layer '{name}' already registered")
        layer = Layer(
            name=name,
            color=color,
            key=key,
            cardinality=cardinality,
            grid=np.zeros((self.rows, self.cols), dtype=np.uint8),
            style=style,
        )
        self._layers[name] = layer
        return layer

    def get(self, name: str) -> Layer:
        return self._layers[name]

    def layers(self) -> Iterable[Layer]:
        return self._layers.values()

    def layer_names(self) -> list[str]:
        return list(self._layers.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._layers

    def resize(self, cols: int, rows: int) -> None:
        """Clear all layers to a new size."""
        self.cols = cols
        self.rows = rows
        for layer in self._layers.values():
            layer.grid = np.zeros((rows, cols), dtype=np.uint8)

    def rotate90(self, k: int) -> None:
        """Rotate every layer by k * 90° CCW (np.rot90 convention).

        When k is odd, cols and rows swap.  The caller is responsible for
        ensuring the resulting shape still matches the currently locked grid.
        """
        k = k % 4
        if k == 0:
            return
        for layer in self._layers.values():
            layer.grid = np.ascontiguousarray(np.rot90(layer.grid, k))
        if k % 2 == 1:
            self.cols, self.rows = self.rows, self.cols

    def flip_horizontal(self) -> None:
        for layer in self._layers.values():
            layer.grid = np.ascontiguousarray(np.fliplr(layer.grid))


# ---------------------------------------------------------------------------
# Load result
# ---------------------------------------------------------------------------

LoadStatus = Literal[
    "ok",
    "auto_rotated",
    "dimension_mismatch",
    "unresolved_orientation",
]


@dataclass
class LoadResult:
    scene: Scene | None
    status: LoadStatus
    message: str
    applied_rotation: int = 0   # quarter-turns CW


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1


def save_scene(
    path: Path | str,
    scene: Scene,
    corner_marker_ids: list[int],
) -> None:
    """Write the scene to disk as JSON, creating parent dirs as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": SCHEMA_VERSION,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "rows": scene.rows,
        "cols": scene.cols,
        "corner_marker_ids": list(corner_marker_ids),
        "layers": {
            layer.name: {
                "cardinality": layer.cardinality.value,
                "cells": layer.grid.astype(int).tolist(),
            }
            for layer in scene.layers()
        },
    }
    p.write_text(json.dumps(payload, indent=2))


def load_scene(
    path: Path | str,
    current_cols: int,
    current_rows: int,
    current_corner_ids: list[int],
    layer_specs: list[tuple[str, tuple[int, int, int], int, Cardinality, Style]],
) -> LoadResult:
    """Load a scene from disk and align it to the current grid / corner setup.

    `layer_specs` is the registration list the caller would use for a fresh
    scene: (name, color, key, cardinality) per layer.  Saved layers matching
    a name in this list inherit the spec's color/key/cardinality so callers
    don't need to re-register after loading.  Saved layers with no matching
    spec are skipped with a warning; specs with no matching saved layer are
    registered empty.

    Returns a LoadResult.  On failure (file missing, malformed, dim mismatch),
    LoadResult.scene is None and the caller should keep its existing scene.
    """
    p = Path(path)
    if not p.exists():
        return LoadResult(None, "dimension_mismatch", f"scene file not found: {p}")

    try:
        payload = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        return LoadResult(None, "dimension_mismatch", f"malformed scene file: {e}")

    if payload.get("version") != SCHEMA_VERSION:
        return LoadResult(
            None,
            "dimension_mismatch",
            f"unsupported scene version {payload.get('version')!r}",
        )

    try:
        saved_rows = int(payload["rows"])
        saved_cols = int(payload["cols"])
        saved_corner_ids = [int(x) for x in payload["corner_marker_ids"]]
        saved_layers = payload["layers"]
    except (KeyError, TypeError, ValueError) as e:
        return LoadResult(None, "dimension_mismatch", f"invalid scene file: {e}")

    if len(saved_corner_ids) != 4 or len(current_corner_ids) != 4:
        return LoadResult(
            None,
            "dimension_mismatch",
            "corner_marker_ids must have exactly 4 entries",
        )

    # ----- Resolve orientation -----
    k = _resolve_cyclic_rotation(saved_corner_ids, current_corner_ids)

    if k is None:
        # No cyclic match.  Try to load without rotation if dims still fit.
        if saved_rows != current_rows or saved_cols != current_cols:
            return LoadResult(
                None,
                "dimension_mismatch",
                f"saved {saved_cols}x{saved_rows} does not match "
                f"current {current_cols}x{current_rows}",
            )
        scene = _scene_from_payload(
            saved_cols, saved_rows, saved_layers, layer_specs, rotation_k=0,
        )
        return LoadResult(
            scene,
            "unresolved_orientation",
            f"corner ids {saved_corner_ids} are not a cyclic permutation of "
            f"{current_corner_ids}; loaded as-is - use [/]/f to adjust",
        )

    # Expected post-rotation dims (np.rot90(arr, k) swaps axes when k is odd).
    post_rows, post_cols = (
        (saved_cols, saved_rows) if k % 2 == 1 else (saved_rows, saved_cols)
    )
    if post_rows != current_rows or post_cols != current_cols:
        return LoadResult(
            None,
            "dimension_mismatch",
            f"saved {saved_cols}x{saved_rows} (rotated k={k}) -> "
            f"{post_cols}x{post_rows}, does not match "
            f"current {current_cols}x{current_rows}",
        )

    scene = _scene_from_payload(
        current_cols, current_rows, saved_layers, layer_specs, rotation_k=k,
    )
    if k == 0:
        return LoadResult(scene, "ok", f"loaded from {p}")
    return LoadResult(
        scene,
        "auto_rotated",
        f"loaded from {p} - auto-rotated {k * 90} deg CW to match current corners",
        applied_rotation=k,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_cyclic_rotation(
    saved_ids: list[int],
    current_ids: list[int],
) -> int | None:
    """Return k in {0,1,2,3} such that np.rot90(saved_grid, k) aligns to the
    current layout; None if no cyclic match exists.

    Corner ids are stored in [TL, TR, BR, BL] order — a CW walk around the
    physical square.  If the researcher shifts the config by m slots later
    (new TL becomes the saved marker at index m), the grid rotates m × 90°
    CW physically, which the array must counter with np.rot90(saved, m) CCW.

    Worked example: saved=[1,2,3,4], current=[2,3,4,1].  Current is saved
    shifted left by 1 → np.roll(saved, -1) == current → returned k = 1.
    Applying np.rot90(saved_arr, 1) maps save's TL cell (row=0,col=0, at
    marker 1) to (row=N-1,col=0), which is the current BL cell — correct,
    since marker 1 is now at the current BL slot.
    """
    saved = np.asarray(saved_ids)
    current = list(current_ids)
    for k in range(4):
        if np.roll(saved, -k).tolist() == current:
            return k
    return None


def _scene_from_payload(
    cols: int,
    rows: int,
    saved_layers: dict,
    layer_specs: list[tuple[str, tuple[int, int, int], int, Cardinality, Style]],
    rotation_k: int,
) -> Scene:
    """Build a fresh Scene, registering all specs and populating cells from
    the saved payload (rotating each layer by np.rot90(k))."""
    scene = Scene(cols=cols, rows=rows)
    spec_by_name = {name: (color, key, card, style) for name, color, key, card, style in layer_specs}

    for name, color, key, card, style in layer_specs:
        scene.register(name, color, key, card, style)

    for saved_name, saved_body in saved_layers.items():
        if saved_name not in spec_by_name:
            print(f"[scene] warning: skipping unknown saved layer '{saved_name}'")
            continue
        try:
            cells = np.asarray(saved_body["cells"], dtype=np.uint8)
        except (KeyError, TypeError, ValueError) as e:
            print(f"[scene] warning: bad cells for layer '{saved_name}': {e}")
            continue
        if rotation_k:
            cells = np.ascontiguousarray(np.rot90(cells, rotation_k))
        target = scene.get(saved_name)
        if cells.shape != target.grid.shape:
            print(
                f"[scene] warning: layer '{saved_name}' shape {cells.shape} "
                f"!= expected {target.grid.shape}; skipping"
            )
            continue
        target.grid[:] = (cells != 0).astype(np.uint8)

    return scene
