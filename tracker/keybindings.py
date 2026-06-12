"""
Keyboard bindings for the tracker.

These are static OS-key constants (not experiment-tunable), so they live as
plain module-level data instead of going through the config system.
"""

from __future__ import annotations


# Manual navigation: key char → (delta_col, delta_row) relative to robot cell.
# WASD = 4-directional; QEZC = diagonals.
NAV_KEY_MAP: dict[int, tuple[int, int]] = {
    ord("w"): (0, -1),
    ord("s"): (0, 1),
    ord("a"): (-1, 0),
    ord("d"): (1, 0),
    ord("q"): (-1, -1),
    ord("e"): (1, -1),
    ord("z"): (-1, 1),
    ord("c"): (1, 1),
}
