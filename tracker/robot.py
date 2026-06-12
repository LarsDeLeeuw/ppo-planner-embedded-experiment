"""
robot.py — Robot state tracking with speed and smoothed heading.
"""

from __future__ import annotations
from dataclasses import dataclass
from math import atan2, cos, degrees, radians, sin
from time import monotonic

import numpy as np

from grid import RobotPose


@dataclass
class RobotState:
    pose: RobotPose
    speed: float | None     # grid-units per second, None on first observation
    heading_deg: float      # smoothed heading (use for navigation / display)
    grid_x: float           # filtered x (use for bridge / navigation)
    grid_y: float           # filtered y (use for bridge / navigation)
    timestamp: float


class RobotTracker:
    """Tracks speed and applies circular EMA heading smoothing."""

    def __init__(
        self,
        heading_alpha: float = 0.15,
        max_speed: float = 2.0,
        reject_max_frames: int = 8,
    ) -> None:
        self._prev_pos: np.ndarray | None = None
        self._prev_time: float = 0.0
        self._alpha = heading_alpha
        self._max_speed = max_speed
        # Escape hatch for the outlier gate.  A single-frame spike is a bad
        # PnP/compression artefact and is rejected; but a *sustained* large
        # displacement is the robot genuinely relocating (e.g. the operator
        # carrying it to the start between trials).  Without an escape the gate
        # welds to the stale position forever — it always compares against the
        # last *accepted* (old) pos and refreshes _prev_time every frame, so dt
        # stays tiny and inst_speed stays huge.  After this many consecutive
        # rejections we accept the reading as a real teleport.  <=0 disables.
        self._reject_max = reject_max_frames
        self._reject_count = 0
        self._smooth_hx: float = 0.0
        self._smooth_hy: float = 0.0

    def update(self, pose: RobotPose) -> RobotState:
        now = monotonic()

        cur_pos = np.array([pose.grid_x, pose.grid_y])

        # --- Position outlier gate ---
        accepted_pos = cur_pos
        teleport = False
        if self._prev_pos is not None:
            dt = now - self._prev_time
            if dt > 0:
                inst_speed = float(np.linalg.norm(cur_pos - self._prev_pos) / dt)
                if inst_speed > self._max_speed:
                    self._reject_count += 1
                    if 0 < self._reject_max <= self._reject_count:
                        # Persistent disagreement → real teleport, not a spike.
                        # Accept the reading and re-seed from it.
                        teleport = True
                        self._reject_count = 0
                    else:
                        accepted_pos = self._prev_pos  # reject transient spike
                else:
                    self._reject_count = 0

        # Speed from accepted positions.  Skipped on a teleport: the jump is
        # real but not a velocity, and norm(new - old)/dt would be enormous.
        speed: float | None = None
        if self._prev_pos is not None and not teleport:
            dt = now - self._prev_time
            if dt > 0:
                speed = float(np.linalg.norm(accepted_pos - self._prev_pos) / dt)

        self._prev_pos = accepted_pos
        self._prev_time = now

        # Circular EMA: average unit vectors to avoid wrap-around artefacts
        hrad = radians(pose.heading_deg)
        hx, hy = cos(hrad), sin(hrad)
        if self._smooth_hx == 0.0 and self._smooth_hy == 0.0:
            self._smooth_hx, self._smooth_hy = hx, hy
        else:
            self._smooth_hx += self._alpha * (hx - self._smooth_hx)
            self._smooth_hy += self._alpha * (hy - self._smooth_hy)
        smoothed = degrees(atan2(self._smooth_hy, self._smooth_hx))

        # Re-derive cell indices from the accepted (filtered) position so
        # `pose.cell_col` / `pose.cell_row` stay consistent with
        # `grid_x` / `grid_y` even when the outlier gate rejected a spike.
        # Without this, send_pose (uses filtered grid_x/y) and
        # auto_driver._send_predict (uses pose.cell_*) drift apart on the
        # frame after a spike, and the orchestrator sends a goal relative
        # to the wrong cell — which the nav node rejects as too far.
        ax, ay = float(accepted_pos[0]), float(accepted_pos[1])
        filtered_pose = RobotPose(
            grid_x=ax,
            grid_y=ay,
            cell_col=int(ax),
            cell_row=int(ay),
            heading_deg=pose.heading_deg,
            image_center=pose.image_center,
            in_bounds=pose.in_bounds,
            height_cm=pose.height_cm,
        )
        return RobotState(
            pose=filtered_pose, speed=speed, heading_deg=smoothed,
            grid_x=ax, grid_y=ay,
            timestamp=now,
        )

    def reset(self) -> None:
        self._prev_pos = None
        self._prev_time = 0.0
        self._reject_count = 0
        self._smooth_hx = 0.0
        self._smooth_hy = 0.0
