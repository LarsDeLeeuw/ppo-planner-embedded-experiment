"""
overlay_controller.py — runtime state for the AR overlay theme.

Owns the current `OverlayTheme`, the user scale multiplier, and the last
frame shape. Rebuilds the theme when either changes so drawing code can
pull a fresh instance each frame without recomputing constants inline.

Mirrors the state-object pattern already used by NavigationState,
RobotTracker, and CapturePipeline.
"""

from __future__ import annotations

from overlay import DEFAULT_THEME, OverlayTheme


# Clamp the user multiplier so bump() cannot drive it past sane limits.
MIN_MULTIPLIER = 0.25
MAX_MULTIPLIER = 8.0


def _resolution_scale(frame_shape: tuple, reference_height: int) -> float:
    """Auto-scale factor: 1.0 at or below reference_height, linear above."""
    return max(1.0, frame_shape[0] / reference_height)


class OverlayController:
    def __init__(
        self,
        base_theme: OverlayTheme = DEFAULT_THEME,
        reference_height: int = 1080,
        initial_multiplier: float = 1.0,
    ) -> None:
        self._base = base_theme
        self._ref_h = reference_height
        self._mult = float(initial_multiplier)
        self._last_shape: tuple | None = None
        self._theme: OverlayTheme = base_theme

    @property
    def theme(self) -> OverlayTheme:
        return self._theme

    @property
    def multiplier(self) -> float:
        return self._mult

    def refresh_for_frame(self, frame) -> None:
        """Rebuild the theme if the frame shape changed."""
        shape = frame.shape
        if shape == self._last_shape:
            return
        self._last_shape = shape
        self._rebuild()

    def bump(self, factor: float) -> float:
        """Multiply the user multiplier by factor, clamp, rebuild. Returns new multiplier."""
        new_mult = max(MIN_MULTIPLIER, min(MAX_MULTIPLIER, self._mult * factor))
        if new_mult == self._mult:
            return self._mult
        self._mult = new_mult
        self._rebuild()
        return self._mult

    def reset(self) -> float:
        """Reset user multiplier to 1.0, rebuild. Returns new multiplier."""
        if self._mult == 1.0:
            return self._mult
        self._mult = 1.0
        self._rebuild()
        return self._mult

    def set_base(self, base_theme: OverlayTheme) -> None:
        """Swap the base theme (e.g. high-contrast). Rebuilds at current scale."""
        self._base = base_theme
        self._rebuild()

    def _rebuild(self) -> None:
        if self._last_shape is None:
            self._theme = self._base.scaled(self._mult)
            return
        s = _resolution_scale(self._last_shape, self._ref_h) * self._mult
        self._theme = self._base.scaled(s)
