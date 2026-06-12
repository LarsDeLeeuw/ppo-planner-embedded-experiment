"""
scene_input.py — Transient HUD input modes for scene save/load.

Pure state, no OpenCV or camera coupling.  Two modes:

- `NameEntry` — growing text buffer for typing a scene name before saving.
- `LoadPicker` — paginated list of available scenes for selecting one to load.

Both accept raw keycodes from cv2.waitKey and return a small result describing
what the main loop should do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# Keycodes
KEY_ESC = 27
KEY_ENTER = 13
KEY_BACKSPACE_MAIN = 8
KEY_BACKSPACE_ALT = 127     # Some platforms / layouts return DEL
KEY_LBRACKET = ord("[")
KEY_RBRACKET = ord("]")

# Characters accepted in scene names.
_ALLOWED_NAME = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")

MAX_NAME_LEN = 48


# ---------------------------------------------------------------------------
# Name entry
# ---------------------------------------------------------------------------

NameEntryResult = Literal["continue", "submit", "cancel"]


@dataclass
class NameEntry:
    """Collects a filename-safe string one keycode at a time."""
    buffer: str = ""

    def accept_key(self, key: int) -> NameEntryResult:
        if key == KEY_ESC:
            return "cancel"
        if key == KEY_ENTER:
            return "submit" if self.buffer else "cancel"
        if key in (KEY_BACKSPACE_MAIN, KEY_BACKSPACE_ALT):
            self.buffer = self.buffer[:-1]
            return "continue"
        if 0 <= key < 128:
            ch = chr(key)
            if ch in _ALLOWED_NAME and len(self.buffer) < MAX_NAME_LEN:
                self.buffer += ch
        return "continue"

    def display(self) -> str:
        return f"{self.buffer}_"  # trailing underscore as a pseudo-cursor


# ---------------------------------------------------------------------------
# Load picker
# ---------------------------------------------------------------------------

PAGE_SIZE = 9   # digits 1-9


@dataclass
class LoadPicker:
    """Paginated chooser over the *.json files found in a scene directory."""
    entries: list[Path]
    page: int = 0

    @classmethod
    def from_dir(cls, scene_dir: Path | str) -> "LoadPicker":
        return cls(entries=list_scenes(Path(scene_dir)))

    @property
    def page_count(self) -> int:
        if not self.entries:
            return 1
        return (len(self.entries) + PAGE_SIZE - 1) // PAGE_SIZE

    def visible(self) -> list[tuple[int, Path]]:
        """Return (1-indexed slot, path) for entries on the current page."""
        start = self.page * PAGE_SIZE
        return [
            (i + 1, p)
            for i, p in enumerate(self.entries[start : start + PAGE_SIZE])
        ]

    def accept_key(self, key: int) -> Path | Literal["cancel", "continue"]:
        if key == KEY_ESC:
            return "cancel"
        if key == KEY_LBRACKET:
            self.page = (self.page - 1) % self.page_count
            return "continue"
        if key == KEY_RBRACKET:
            self.page = (self.page + 1) % self.page_count
            return "continue"
        # Digits 1-9 → 0..8 index on current page
        if ord("1") <= key <= ord("9"):
            idx = key - ord("1")
            visible = self.visible()
            if idx < len(visible):
                return visible[idx][1]
        return "continue"


# ---------------------------------------------------------------------------
# Directory listing
# ---------------------------------------------------------------------------

def list_scenes(scene_dir: Path) -> list[Path]:
    """Sorted list of *.json paths in `scene_dir` (empty if dir missing)."""
    if not scene_dir.exists():
        return []
    return sorted(scene_dir.glob("*.json"))
