"""
overlay.py — AR overlay rendering on the camera feed.

Draws marker outlines, grid lines, cell highlight, heading arrow, cell labels,
and HUD. All drawing mutates the frame in-place.

Visual parameters (colors, fonts, sizes, thicknesses, offsets, alphas) live
on `OverlayTheme`. `DEFAULT_THEME` is the 1.0x baseline — use `theme.scaled(s)`
to derive a resolution-adjusted theme. To add a new look (high-contrast,
screenshot mode, color-blind), construct another `OverlayTheme` literal.
"""

from __future__ import annotations
from dataclasses import dataclass, replace
from math import cos, radians, sin
from typing import Any

import cv2
import numpy as np

from auto_recorder import AutoSnapshot
from coord_transform import bridge_direction_to_tracker_delta
from directions import delta as action_delta
from grid import GridState, RobotPose, grid_points_to_image, grid_to_image
from robot import RobotState
from scene import Scene, Style


@dataclass(frozen=True)
class OverlayTheme:
    # Colors (BGR)
    col_grid: tuple[int, int, int]
    col_grid_border: tuple[int, int, int]
    col_cell_ok: tuple[int, int, int]
    col_cell_oob: tuple[int, int, int]
    col_arrow: tuple[int, int, int]
    col_hud_text: tuple[int, int, int]
    col_hud_bg: tuple[int, int, int]
    col_label: tuple[int, int, int]
    col_target: tuple[int, int, int]
    col_hint: tuple[int, int, int]
    col_hint_shadow: tuple[int, int, int]
    col_marker: tuple[int, int, int]
    col_marker_id: tuple[int, int, int]

    # Font
    font: int

    # Thicknesses (scale with resolution)
    grid_thick_inner: int
    grid_thick_outer: int
    arrow_thick: int
    hint_outline_thick: int
    hud_text_thick: int
    marker_thick: int

    # Font scales (scale with resolution)
    hud_scale: float
    label_scale: float
    hint_scale: float
    marker_id_scale: float

    # Layout (scale with resolution)
    hud_line_h: int
    hud_pad: int
    hud_char_w: int
    label_offset: tuple[int, int]
    hint_offset: tuple[int, int]

    # Effects / geometry (resolution-independent)
    cell_alpha: float
    hud_bg_alpha: float
    arrow_len: float
    arrow_tip: float

    # Auto-mode debug minimap (resolution-independent — fixed UI overlay).
    col_minimap_bg: tuple[int, int, int]
    col_minimap_grid: tuple[int, int, int]
    col_minimap_obstacle: tuple[int, int, int]
    col_minimap_action_legal: tuple[int, int, int]
    col_minimap_action_illegal: tuple[int, int, int]
    col_minimap_text: tuple[int, int, int]
    col_minimap_title: tuple[int, int, int]
    col_minimap_robot: tuple[int, int, int]
    col_minimap_goal: tuple[int, int, int]

    def scaled(self, s: float) -> "OverlayTheme":
        """Return a copy with size-ish fields multiplied by s."""
        def i(x: int, floor: int = 1) -> int:
            sign = -1 if x < 0 else 1
            return sign * max(floor, int(round(abs(x) * s)))

        return replace(
            self,
            grid_thick_inner=i(self.grid_thick_inner),
            grid_thick_outer=i(self.grid_thick_outer, 2),
            arrow_thick=i(self.arrow_thick, 2),
            hint_outline_thick=i(self.hint_outline_thick, 2),
            hud_text_thick=i(self.hud_text_thick),
            marker_thick=i(self.marker_thick),
            hud_scale=self.hud_scale * s,
            label_scale=self.label_scale * s,
            hint_scale=self.hint_scale * s,
            marker_id_scale=self.marker_id_scale * s,
            hud_line_h=i(self.hud_line_h),
            hud_pad=i(self.hud_pad),
            hud_char_w=i(self.hud_char_w),
            label_offset=(i(self.label_offset[0]), i(self.label_offset[1])),
            hint_offset=(i(self.hint_offset[0]), i(self.hint_offset[1])),
        )


DEFAULT_THEME = OverlayTheme(
    col_grid=(200, 200, 200),
    col_grid_border=(255, 255, 255),
    col_cell_ok=(0, 255, 128),
    col_cell_oob=(0, 0, 255),
    col_arrow=(0, 0, 255),
    col_hud_text=(255, 255, 255),
    col_hud_bg=(0, 0, 0),
    col_label=(220, 220, 220),
    col_target=(255, 180, 0),
    col_hint=(255, 255, 255),
    col_hint_shadow=(0, 0, 0),
    col_marker=(0, 255, 0),
    col_marker_id=(0, 255, 0),
    font=cv2.FONT_HERSHEY_SIMPLEX,
    grid_thick_inner=1,
    grid_thick_outer=2,
    arrow_thick=2,
    hint_outline_thick=3,
    hud_text_thick=1,
    marker_thick=2,
    hud_scale=0.55,
    label_scale=0.35,
    hint_scale=0.5,
    marker_id_scale=0.5,
    hud_line_h=24,
    hud_pad=8,
    hud_char_w=10,
    label_offset=(-10, 5),
    hint_offset=(-6, 6),
    cell_alpha=0.3,
    hud_bg_alpha=0.6,
    arrow_len=0.4,
    arrow_tip=0.3,
    col_minimap_bg=(0, 0, 0),
    col_minimap_grid=(80, 80, 80),
    col_minimap_obstacle=(60, 60, 220),       # red-ish, matches FILL palette
    col_minimap_action_legal=(220, 220, 0),   # cyan-yellow, matches AUTO HUD
    col_minimap_action_illegal=(60, 60, 220), # red — same as obstacle X
    col_minimap_text=(230, 230, 230),
    col_minimap_title=(180, 220, 220),
    col_minimap_robot=(0, 255, 128),          # bright green outline
    col_minimap_goal=(220, 0, 200),           # purple — matches goal layer
)


def draw_markers(frame: np.ndarray, markers: list, theme: OverlayTheme) -> None:
    """Outline each detected ArUco marker and label its ID."""
    for m in markers:
        pts = m.corners.reshape(-1, 2).astype(np.int32)
        cv2.polylines(frame, [pts], True, theme.col_marker, theme.marker_thick, cv2.LINE_AA)
        top_left = pts[0]
        cv2.putText(
            frame, str(m.marker_id),
            (int(top_left[0]), int(top_left[1]) - 4),
            theme.font, theme.marker_id_scale, theme.col_marker_id,
            theme.hud_text_thick, cv2.LINE_AA,
        )


def draw_grid_lines(frame: np.ndarray, grid: GridState, theme: OverlayTheme) -> None:
    """Draw NxN grid lines projected into image space."""
    for i in range(grid.cols + 1):
        pts = grid_points_to_image(
            np.array([[i, 0], [i, grid.rows]], dtype=np.float32), grid.H_inv
        )
        is_edge = (i == 0 or i == grid.cols)
        thick = theme.grid_thick_outer if is_edge else theme.grid_thick_inner
        color = theme.col_grid_border if is_edge else theme.col_grid
        cv2.line(frame, _ip(pts[0]), _ip(pts[1]), color, thick)

    for j in range(grid.rows + 1):
        pts = grid_points_to_image(
            np.array([[0, j], [grid.cols, j]], dtype=np.float32), grid.H_inv
        )
        is_edge = (j == 0 or j == grid.rows)
        thick = theme.grid_thick_outer if is_edge else theme.grid_thick_inner
        color = theme.col_grid_border if is_edge else theme.col_grid
        cv2.line(frame, _ip(pts[0]), _ip(pts[1]), color, thick)


def draw_cell_highlight(
    frame: np.ndarray, grid: GridState, col: int, row: int, in_bounds: bool,
    theme: OverlayTheme,
) -> None:
    """Semi-transparent fill on the robot's current cell."""
    corners_grid = np.array(
        [[col, row], [col + 1, row], [col + 1, row + 1], [col, row + 1]],
        dtype=np.float32,
    )
    corners_img = grid_points_to_image(corners_grid, grid.H_inv).astype(np.int32)
    color = theme.col_cell_ok if in_bounds else theme.col_cell_oob
    overlay = frame.copy()
    cv2.fillPoly(overlay, [corners_img], color)
    cv2.addWeighted(overlay, theme.cell_alpha, frame, 1 - theme.cell_alpha, 0, frame)


def draw_cell_labels(frame: np.ndarray, grid: GridState, theme: OverlayTheme) -> None:
    """Small coordinate label at each cell center."""
    for r in range(grid.rows):
        for c in range(grid.cols):
            center = grid_to_image(
                np.array([c + 0.5, r + 0.5], dtype=np.float32), grid.H_inv
            )
            label = f"{c},{r}"
            cv2.putText(
                frame, label, _ip(center, offset=theme.label_offset),
                theme.font, theme.label_scale, theme.col_label,
                theme.hud_text_thick, cv2.LINE_AA,
            )


def draw_scene(frame: np.ndarray, grid: GridState, scene: Scene, theme: OverlayTheme) -> None:
    """Render every marked cell of every layer.

    Each layer carries its own BGR color and a render style:
      - FILL: one semi-transparent polygon across the whole cell (obstacles).
      - CIRCLE: one opaque filled circle at the cell centroid (goal-type
        markers; stays visible through other layers for colorblind-friendly
        contrast).
    """
    for layer in scene.layers():
        cells = list(layer.marked_cells())
        if not cells:
            continue

        # Project cell corners once per marked cell; reused by both styles.
        per_cell_corners: list[np.ndarray] = []
        for col, row in cells:
            corners_grid = np.array(
                [[col, row], [col + 1, row], [col + 1, row + 1], [col, row + 1]],
                dtype=np.float32,
            )
            per_cell_corners.append(grid_points_to_image(corners_grid, grid.H_inv))

        if layer.style == Style.FILL:
            polys = [c.astype(np.int32) for c in per_cell_corners]
            overlay = frame.copy()
            cv2.fillPoly(overlay, polys, layer.color)
            cv2.addWeighted(overlay, theme.cell_alpha, frame, 1 - theme.cell_alpha, 0, frame)
        elif layer.style == Style.CIRCLE:
            overlay = frame.copy()
            for corners_img in per_cell_corners:
                center = corners_img.mean(axis=0)
                # Radius = ~35% of the shortest projected edge, floored at 4 px.
                edges = [
                    float(np.linalg.norm(corners_img[(i + 1) % 4] - corners_img[i]))
                    for i in range(4)
                ]
                radius = max(4, int(round(min(edges) * 0.35)))
                cx, cy = int(round(center[0])), int(round(center[1]))
                cv2.circle(overlay, (cx, cy), radius, layer.color, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, theme.cell_alpha, frame, 1 - theme.cell_alpha, 0, frame)


def draw_target_highlight(
    frame: np.ndarray, grid: GridState, col: int, row: int,
    theme: OverlayTheme,
) -> None:
    """Semi-transparent orange fill on the target cell."""
    corners_grid = np.array(
        [[col, row], [col + 1, row], [col + 1, row + 1], [col, row + 1]],
        dtype=np.float32,
    )
    corners_img = grid_points_to_image(corners_grid, grid.H_inv).astype(np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [corners_img], theme.col_target)
    cv2.addWeighted(overlay, theme.cell_alpha, frame, 1 - theme.cell_alpha, 0, frame)


def draw_nav_hints(
    frame: np.ndarray,
    grid: GridState,
    hints: list[tuple[int, int, str]],
    theme: OverlayTheme,
) -> None:
    """Key labels on adjacent cells for keyboard navigation directions."""
    for col, row, label in hints:
        center = grid_to_image(
            np.array([col + 0.5, row + 0.5], dtype=np.float32), grid.H_inv,
        )
        pt = _ip(center, offset=theme.hint_offset)
        cv2.putText(frame, label, pt, theme.font, theme.hint_scale,
                    theme.col_hint_shadow, theme.hint_outline_thick, cv2.LINE_AA)
        cv2.putText(frame, label, pt, theme.font, theme.hint_scale,
                    theme.col_hint, theme.hud_text_thick, cv2.LINE_AA)


def draw_heading_arrow(
    frame: np.ndarray, grid: GridState, pose: RobotPose,
    theme: OverlayTheme, heading_deg: float | None = None,
) -> None:
    """Arrow from robot center in heading direction."""
    hrad = radians(heading_deg if heading_deg is not None else pose.heading_deg)
    start_grid = np.array([pose.grid_x, pose.grid_y], dtype=np.float32)
    end_grid = start_grid + theme.arrow_len * np.array(
        [cos(hrad), sin(hrad)], dtype=np.float32,
    )

    start_img = grid_to_image(start_grid, grid.H_inv)
    end_img = grid_to_image(end_grid, grid.H_inv)

    cv2.arrowedLine(
        frame, _ip(start_img), _ip(end_img),
        theme.col_arrow, theme.arrow_thick, tipLength=theme.arrow_tip,
    )


def draw_hud(
    frame: np.ndarray,
    grid: GridState | None,
    robot: RobotState | None,
    visible_corner_count: int,
    theme: OverlayTheme,
    target_cell: tuple[int, int] | None = None,
    ui_multiplier: float | None = None,
    scene: Scene | None = None,
    mark_mode: bool = False,
    active_layer_name: str | None = None,
    extra_lines: list[tuple[str, tuple[int, int, int] | None]] | None = None,
    auto_status: Any = None,
) -> None:
    """Status panel in the top-left corner."""
    lines: list[tuple[str, tuple[int, int, int] | None]] = []

    def add(line: str, color: tuple[int, int, int] | None = None) -> None:
        lines.append((line, color))

    if grid is None:
        add(f"CALIBRATING ({visible_corner_count}/4 corners)")
    else:
        add(f"GRID: {grid.cols}x{grid.rows} [LOCKED]")

        if robot is None:
            add("ROBOT: NOT VISIBLE")
        else:
            p = robot.pose
            if p.in_bounds:
                add(f"CELL: ({p.cell_col}, {p.cell_row})")
            else:
                add("CELL: OUT OF BOUNDS", theme.col_cell_oob)
            add(f"POS:  ({p.grid_x:.2f}, {p.grid_y:.2f})")
            add(f"HDG:  {robot.heading_deg:.1f} deg")
            add(f"ALT:  {p.height_cm:.1f} cm")
            if robot.speed is not None:
                add(f"SPD:  {robot.speed:.2f} u/s")

        if target_cell is not None:
            add(f"TGT:  ({target_cell[0]}, {target_cell[1]})")

        if ui_multiplier is not None:
            add(f"UI:   {ui_multiplier:.2f}x")

        if scene is not None:
            if mark_mode:
                add("MARK MODE")
                for layer in scene.layers():
                    marker = ">" if layer.name == active_layer_name else " "
                    key_char = chr(layer.key).upper() if 0 <= layer.key < 128 else "?"
                    add(f"{marker} [{key_char}] {layer.name}", layer.color)
            for layer in scene.layers():
                n = layer.count()
                if n > 0:
                    add(f"{layer.name.upper()}: {n}", layer.color)

        if extra_lines:
            for text, color in extra_lines:
                add(text, color)

        if auto_status is not None and (auto_status.active or auto_status.error):
            auto_color = (0, 0, 255) if auto_status.error else (0, 220, 220)
            tag = "ERR" if auto_status.error else "ON"
            add(
                f"AUTO: {tag}   cycle={auto_status.cycle}   retries={auto_status.retries}",
                auto_color,
            )
            goal_str = (
                f"({auto_status.goal_cell[0]},{auto_status.goal_cell[1]})"
                if auto_status.goal_cell is not None else "-"
            )
            legal_str = (
                "-" if auto_status.last_legal is None
                else ("yes" if auto_status.last_legal else "no")
            )
            last = auto_status.last_label or "-"
            add(f"LAST: {last}   goal={goal_str}   legal={legal_str}", auto_color)

        if mark_mode:
            add("Esc:quit M:exit-mark [/]:rot F:flip X:clear K:save L:load")
        else:
            add("Esc:quit R:recal T:hints G:capture M:mark P:auto V:mm K:save L:load +/-:ui 0:reset")

    pad = theme.hud_pad
    w = max(len(l[0]) for l in lines) * theme.hud_char_w + pad * 2
    h = len(lines) * theme.hud_line_h + pad * 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), theme.col_hud_bg, -1)
    cv2.addWeighted(overlay, theme.hud_bg_alpha, frame, 1 - theme.hud_bg_alpha, 0, frame)

    for i, (line, color) in enumerate(lines):
        y = pad + (i + 1) * theme.hud_line_h - 4
        col = color if color is not None else theme.col_hud_text
        cv2.putText(frame, line, (pad, y), theme.font, theme.hud_scale,
                    col, theme.hud_text_thick, cv2.LINE_AA)


def draw_auto_minimaps(
    frame: np.ndarray,
    scene: Scene,
    snapshot: AutoSnapshot,
    theme: OverlayTheme,
    cell_px: int,
    robot: RobotState | None = None,
    obstacle_layer: str = "obstacles",
    goal_layer: str = "goal",
) -> None:
    """Draw the auto-mode debug minimaps (obstacles+actions, energy) top-right.

    The minimap is rendered in tracker frame (row 0 at the top), matching the
    main view.  Action arrow direction is derived via the canonical bridge→
    tracker direction conversion so the visual is guaranteed consistent with
    what the driver actually used.
    """
    h, w = frame.shape[:2]
    rows, cols = scene.rows, scene.cols
    if rows <= 0 or cols <= 0 or cell_px <= 0:
        return

    map_w = cols * cell_px
    map_h = rows * cell_px
    margin = 12
    title_h = 18
    gap = 12

    # Side-by-side layout, both anchored to the top-right corner.
    # Map 1 (obstacles+actions) on the left, Map 2 (energy) on the right.
    total_w = 2 * map_w + gap
    x0_left = w - margin - total_w
    x0_right = x0_left + map_w + gap
    if x0_left < margin:
        return  # frame too narrow for this minimap size; skip silently

    # === Map 1 — obstacles + action trail ===
    title_y = margin + title_h - 4
    cv2.putText(
        frame, "OBSTACLES + ACTIONS", (x0_left, title_y),
        theme.font, 0.5, theme.col_minimap_title, 1, cv2.LINE_AA,
    )
    grid1_top = margin + title_h
    x0 = x0_left          # local alias used by the rest of map 1's drawing
    _draw_minimap_grid(frame, x0, grid1_top, cell_px, rows, cols, theme)

    # X marks for every obstacle cell.
    if obstacle_layer in scene:
        for c, r in scene.get(obstacle_layer).marked_cells():
            x_a = x0 + c * cell_px + 4
            y_a = grid1_top + r * cell_px + 4
            x_b = x0 + (c + 1) * cell_px - 4
            y_b = grid1_top + (r + 1) * cell_px - 4
            cv2.line(frame, (x_a, y_a), (x_b, y_b),
                     theme.col_minimap_obstacle, 2, cv2.LINE_AA)
            cv2.line(frame, (x_a, y_b), (x_b, y_a),
                     theme.col_minimap_obstacle, 2, cv2.LINE_AA)

    # Action arrows in registration order — latest at a given cell wins.
    arrow_len = max(4, int(round(cell_px * 0.4)))
    for ev in snapshot.action_history:
        if not (0 <= ev.robot_col < cols and 0 <= ev.robot_row < rows):
            continue
        cx = x0 + ev.robot_col * cell_px + cell_px // 2
        cy = grid1_top + ev.robot_row * cell_px + cell_px // 2
        try:
            dx_b, dy_b = action_delta(ev.action)
        except KeyError:
            continue
        drow, dcol = bridge_direction_to_tracker_delta(dx_b, dy_b)
        # Normalize so diagonals are the same visual length as cardinals.
        norm = (drow * drow + dcol * dcol) ** 0.5 or 1.0
        ex = cx + int(round(dcol / norm * arrow_len))
        ey = cy + int(round(drow / norm * arrow_len))
        color = (theme.col_minimap_action_legal if ev.legal
                 else theme.col_minimap_action_illegal)
        cv2.arrowedLine(frame, (cx, cy), (ex, ey), color, 2, cv2.LINE_AA,
                        tipLength=0.4)

    # Goal cell — small filled circle, drawn on top of arrows so it stays
    # visible even when an arrow runs through it.
    if goal_layer in scene:
        goal_cells = list(scene.get(goal_layer).marked_cells())
        if len(goal_cells) == 1:
            gc, gr = goal_cells[0]
            if 0 <= gc < cols and 0 <= gr < rows:
                cx = x0 + gc * cell_px + cell_px // 2
                cy = grid1_top + gr * cell_px + cell_px // 2
                cv2.circle(frame, (cx, cy), max(3, cell_px // 6),
                           theme.col_minimap_goal, -1, cv2.LINE_AA)

    # Robot's current cell — outlined rectangle, drawn last so it's always on top.
    if robot is not None:
        rc, rr = robot.pose.cell_col, robot.pose.cell_row
        if 0 <= rc < cols and 0 <= rr < rows:
            x_a = x0 + rc * cell_px + 1
            y_a = grid1_top + rr * cell_px + 1
            x_b = x0 + (rc + 1) * cell_px - 1
            y_b = grid1_top + (rr + 1) * cell_px - 1
            cv2.rectangle(frame, (x_a, y_a), (x_b, y_b),
                          theme.col_minimap_robot, 2, cv2.LINE_AA)

    # === Map 2 — energy (right of map 1, same vertical position) ===
    cv2.putText(
        frame, "ENERGY", (x0_right, title_y),
        theme.font, 0.5, theme.col_minimap_title, 1, cv2.LINE_AA,
    )
    grid2_top = grid1_top      # same top as map 1 — stacked horizontally
    _draw_minimap_grid(frame, x0_right, grid2_top, cell_px, rows, cols, theme)

    eng = snapshot.last_energy_map
    if eng is not None and eng.shape == (rows, cols):
        # Pick a font scale that comfortably fits "0.42" within the cell.
        font_scale = max(0.35, min(0.6, cell_px / 80.0))
        for r in range(rows):
            for c in range(cols):
                val = float(eng[r, c])
                text = f"{val:.2f}"
                (tw, th), _ = cv2.getTextSize(text, theme.font, font_scale, 1)
                tx = x0_right + c * cell_px + (cell_px - tw) // 2
                ty = grid2_top + r * cell_px + (cell_px + th) // 2
                cv2.putText(frame, text, (tx, ty), theme.font, font_scale,
                            theme.col_minimap_text, 1, cv2.LINE_AA)


def _draw_minimap_grid(
    frame: np.ndarray,
    x0: int,
    y0: int,
    cell_px: int,
    rows: int,
    cols: int,
    theme: OverlayTheme,
) -> None:
    """Filled background + thin cell borders for one minimap."""
    map_w = cols * cell_px
    map_h = rows * cell_px
    cv2.rectangle(frame, (x0, y0), (x0 + map_w, y0 + map_h),
                  theme.col_minimap_bg, -1)
    for r in range(rows + 1):
        y = y0 + r * cell_px
        cv2.line(frame, (x0, y), (x0 + map_w, y), theme.col_minimap_grid, 1)
    for c in range(cols + 1):
        x = x0 + c * cell_px
        cv2.line(frame, (x, y0), (x, y0 + map_h), theme.col_minimap_grid, 1)


def draw_overlay(
    frame: np.ndarray,
    grid: GridState | None,
    robot: RobotState | None,
    visible_corner_count: int,
    theme: OverlayTheme,
    target_cell: tuple[int, int] | None = None,
    nav_hints: list[tuple[int, int, str]] | None = None,
    ui_multiplier: float | None = None,
    scene: Scene | None = None,
    mark_mode: bool = False,
    active_layer_name: str | None = None,
    hud_extra_lines: list[tuple[str, tuple[int, int, int] | None]] | None = None,
    auto_status: Any = None,
    auto_snapshot: AutoSnapshot | None = None,
    minimap_cell_px: int = 36,
    minimap_visible: bool = True,
) -> None:
    """Master draw function — calls sub-renderers in layer order."""
    if grid is not None:
        draw_grid_lines(frame, grid, theme)
        draw_cell_labels(frame, grid, theme)

        if scene is not None:
            draw_scene(frame, grid, scene, theme)

        if target_cell is not None:
            draw_target_highlight(frame, grid, target_cell[0], target_cell[1], theme)

        if robot is not None:
            p = robot.pose
            draw_cell_highlight(frame, grid, p.cell_col, p.cell_row, p.in_bounds, theme)
            draw_heading_arrow(frame, grid, p, theme, heading_deg=robot.heading_deg)

            if nav_hints:
                draw_nav_hints(frame, grid, nav_hints, theme)

    draw_hud(
        frame, grid, robot, visible_corner_count, theme,
        target_cell=target_cell, ui_multiplier=ui_multiplier,
        scene=scene, mark_mode=mark_mode, active_layer_name=active_layer_name,
        extra_lines=hud_extra_lines, auto_status=auto_status,
    )

    # Auto-mode debug minimaps (top-right).  Drawn after the HUD so they
    # stack above any HUD content; gated on having a snapshot to render
    # AND the runtime visibility flag.
    if (
        auto_snapshot is not None
        and minimap_visible
        and scene is not None
    ):
        draw_auto_minimaps(
            frame, scene, auto_snapshot, theme,
            cell_px=minimap_cell_px,
            robot=robot,
        )


def _ip(pt: np.ndarray, offset: tuple[int, int] = (0, 0)) -> tuple[int, int]:
    """Convert float point to integer pixel tuple."""
    return (int(round(pt[0])) + offset[0], int(round(pt[1])) + offset[1])
