"""Energy-harvesting-aware A* path planner.

Pure-Python algorithm layer — no ROS imports, unit-testable without
sourcing ROS2.

Two variants are exposed:
    - "shortest": classic 8-connected A* minimising geometric path length.
                  Octile heuristic, admissible, provably optimal.
    - "energy"  : modified A* that subtracts a weighted energy term from the
                  edge cost, making the planner prefer energy-rich cells.
                  Uses the paper's (1 - alpha)-scaled octile heuristic
                  (Eq. 35); admissible and A*-optimal when alpha in [0, 1]
                  and the energy map is normalised to [0, 1].

Returns the immediate next action (0..7), the full planned path, the
geometric distance cost, and the sum of energy along the path.
"""

from __future__ import annotations

import heapq
import math
from typing import Optional, Tuple

import numpy as np

from tb3_planner_common.directions import ACTION_BY_DIRECTION


Point = Tuple[int, int]


# 8-direction moves with corresponding step costs. Diagonal = sqrt(2).
MOVES: list[tuple[int, int, float]] = [
    (-1,  0, 1.0),
    ( 1,  0, 1.0),
    ( 0, -1, 1.0),
    ( 0,  1, 1.0),
    (-1, -1, math.sqrt(2)),
    (-1,  1, math.sqrt(2)),
    ( 1, -1, math.sqrt(2)),
    ( 1,  1, math.sqrt(2)),
]


def heuristic_octile(a: Point, b: Point) -> float:
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    F = math.sqrt(2) - 1
    return F * min(dx, dy) + max(dx, dy)


def astar_shortest(
    obstacles: list[list[int]],
    start: Point,
    goal: Point,
    max_expansions: Optional[int] = None,
) -> tuple[Optional[list[Point]], float]:
    """Classic 8-connected A* minimising path length.

    Returns (path, cost). path[0] == start, path[-1] == goal on success.
    Returns (None, inf) when no path exists or max_expansions is exceeded.
    """
    open_set = [(heuristic_octile(start, goal), 0.0, start)]
    came_from: dict[Point, Point] = {}
    g_score: dict[Point, float] = {start: 0.0}
    visited: set[Point] = set()
    rows, cols = len(obstacles), len(obstacles[0])
    expansions = 0

    while open_set:
        _f, g, curr = heapq.heappop(open_set)
        if curr == goal:
            path: list[Point] = []
            while curr in came_from:
                path.append(curr)
                curr = came_from[curr]
            return [start] + path[::-1], g
        if curr in visited:
            continue
        visited.add(curr)
        expansions += 1
        if max_expansions is not None and expansions > max_expansions:
            return None, float("inf")
        for dr, dc, cost in MOVES:
            nbr = (curr[0] + dr, curr[1] + dc)
            if 0 <= nbr[0] < rows and 0 <= nbr[1] < cols and obstacles[nbr[0]][nbr[1]] == 0:
                ng = g + cost
                if ng < g_score.get(nbr, float("inf")):
                    came_from[nbr] = curr
                    g_score[nbr] = ng
                    heapq.heappush(open_set, (ng + heuristic_octile(nbr, goal), ng, nbr))
    return None, float("inf")


def astar_energy(
    obstacles: list[list[int]],
    energy: list[list[float]],
    start: Point,
    goal: Point,
    energy_weight: float,
    max_expansions: Optional[int] = None,
) -> tuple[Optional[list[Point]], float]:
    """8-connected A* with the paper's energy-aware cost (Eq. 34, 35).

    Each transition to a destination cell costs `c_step - energy_weight * EH`,
    accumulated into a single scalar g (paper Eq. 34, with `energy_weight` =
    alpha). The octile heuristic is scaled by `(1 - energy_weight)` (Eq. 35):
    each step costs at least `1 - energy_weight` (c_step >= 1, EH <= 1) while
    octile drops by at most 1 per step, so the heuristic lower-bounds the true
    remaining cost. With `energy_weight` (alpha) in [0, 1] and energy normalised
    to [0, 1], every edge cost is non-negative and the heuristic is admissible
    and consistent → the returned path is A*-optimal for this composite cost.

    Returns (path, geometric_distance). The reported cost is the geometric path
    length (sum of step distances), NOT the energy-discounted g, so it stays
    comparable to the `shortest` variant; harvested energy is reported
    separately by the caller.
    """
    h_scale = 1.0 - energy_weight
    open_set: list[tuple[float, Point]] = [(h_scale * heuristic_octile(start, goal), start)]
    came_from: dict[Point, Point] = {}
    visited: set[Point] = set()
    g_score: dict[Point, float] = {start: 0.0}
    rows, cols = len(obstacles), len(obstacles[0])
    expansions = 0

    while open_set:
        _f, curr = heapq.heappop(open_set)
        if curr == goal:
            path: list[Point] = []
            while curr in came_from:
                path.append(curr)
                curr = came_from[curr]
            full = [start] + path[::-1]
            dist = sum(
                math.hypot(b[0] - a[0], b[1] - a[1])
                for a, b in zip(full, full[1:])
            )
            return full, dist
        if curr in visited:
            continue
        visited.add(curr)
        expansions += 1
        if max_expansions is not None and expansions > max_expansions:
            return None, float("inf")
        for dr, dc, cost in MOVES:
            nbr = (curr[0] + dr, curr[1] + dc)
            if 0 <= nbr[0] < rows and 0 <= nbr[1] < cols and obstacles[nbr[0]][nbr[1]] == 0:
                ng = g_score[curr] + cost - energy_weight * energy[nbr[0]][nbr[1]]
                if ng < g_score.get(nbr, float("inf")):
                    came_from[nbr] = curr
                    g_score[nbr] = ng
                    heapq.heappush(
                        open_set,
                        (ng + h_scale * heuristic_octile(nbr, goal), nbr),
                    )
    return None, float("inf")


class NoPathError(RuntimeError):
    """No path from start to goal (including max_expansions exceeded)."""


class AlreadyAtGoalError(RuntimeError):
    """start == goal (nothing to plan)."""


class AstarPlanner:
    """Thin, stateless wrapper over the two A* variants.

    Usage:
        planner = AstarPlanner(energy_weight=1.0)
        result  = planner.predict(obstacle_map, energy_map, (x, y), (gx, gy), variant="energy")
        # {"action": 7, "direction": [1, 1], "path": [[x0,y0], ...], "cost": 12.3, "energy": 4.5}
    """

    def __init__(self, energy_weight: float = 0.5, max_expansions: Optional[int] = None):
        self.energy_weight = float(energy_weight)
        self.max_expansions = max_expansions if (max_expansions is None or max_expansions > 0) else None

    def predict(
        self,
        obstacle_map: np.ndarray,
        energy_map: np.ndarray,
        robot_pos: Point,
        goal_pos: Point,
        variant: str = "energy",
    ) -> dict:
        if variant not in ("shortest", "energy"):
            raise ValueError(f"variant must be 'shortest' or 'energy', got {variant!r}")

        obstacles = np.asarray(obstacle_map)
        if obstacles.ndim != 2:
            raise ValueError(f"obstacle_map must be 2D, got shape {obstacles.shape}")
        energy = np.asarray(energy_map, dtype=np.float32)
        if energy.shape != obstacles.shape:
            raise ValueError(
                f"energy_map shape {energy.shape} != obstacle_map shape {obstacles.shape}"
            )

        start: Point = (int(robot_pos[0]), int(robot_pos[1]))
        goal: Point = (int(goal_pos[0]), int(goal_pos[1]))

        rows, cols = obstacles.shape
        for name, p in (("robot_pos", start), ("goal_pos", goal)):
            if not (0 <= p[0] < rows and 0 <= p[1] < cols):
                raise ValueError(f"{name} {p} outside grid {obstacles.shape}")
        if obstacles[start[0], start[1]] != 0:
            raise ValueError(f"robot_pos {start} is on an obstacle")
        if obstacles[goal[0], goal[1]] != 0:
            raise ValueError(f"goal_pos {goal} is on an obstacle")
        if start == goal:
            raise AlreadyAtGoalError(f"robot already at goal {start}")

        obs_list = (obstacles > 0).astype(int).tolist()
        # Energy is contractually normalised to [0, 1] (see SCENARIO_FORMAT.md).
        # Clip to that range so the energy variant's (1 - alpha)-scaled heuristic
        # stays admissible (paper Eq. 35 requires EH in [0, 1]).
        en_list = np.clip(energy, 0.0, 1.0).tolist()

        if variant == "shortest":
            path, cost = astar_shortest(obs_list, start, goal, max_expansions=self.max_expansions)
        else:
            path, cost = astar_energy(
                obs_list, en_list, start, goal, self.energy_weight,
                max_expansions=self.max_expansions,
            )

        if path is None or len(path) < 2:
            detail = (
                f"max_expansions={self.max_expansions} exceeded"
                if self.max_expansions is not None else "no path exists"
            )
            raise NoPathError(f"no path from {start} to {goal} (variant={variant}, {detail})")

        nxt = path[1]
        dx, dy = nxt[0] - start[0], nxt[1] - start[1]
        action = ACTION_BY_DIRECTION.get((dx, dy))
        if action is None:
            raise RuntimeError(f"non-unit step in A* output: start={start} next={nxt}")

        energy_sum = float(sum(float(energy[r, c]) for r, c in path))

        return {
            "action": int(action),
            "direction": [int(dx), int(dy)],
            "path": [[int(r), int(c)] for r, c in path],
            "cost": float(cost),
            "energy": energy_sum,
        }
