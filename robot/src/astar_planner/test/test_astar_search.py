"""Unit tests for astar_search (pure-Python, no ROS)."""

from __future__ import annotations

import numpy as np
import pytest

from astar_planner.astar_search import (
    AlreadyAtGoalError,
    AstarPlanner,
    NoPathError,
)


GRID = 10


def _empty_maps(size: int = GRID):
    return (
        np.zeros((size, size), dtype=np.int32),
        np.zeros((size, size), dtype=np.float32),
    )


def test_direction_correctness_on_open_grid():
    """A* picks the correct first-step direction and wires into the PPO-compatible
    action table. One cardinal + one diagonal is enough — symmetric cases add
    no new coverage."""
    obstacles, energy = _empty_maps()
    planner = AstarPlanner()

    diag = planner.predict(obstacles, energy, (1, 1), (8, 8), variant="shortest")
    assert diag["action"] == 7 and tuple(diag["direction"]) == (1, 1)
    assert diag["path"][0] == [1, 1] and diag["path"][-1] == [8, 8]

    card = planner.predict(obstacles, energy, (5, 2), (5, 8), variant="shortest")
    assert card["action"] == 3 and tuple(card["direction"]) == (0, 1)


def test_energy_variant_prefers_harvest_detour():
    """Key behavioral test: the energy variant detours through a high-energy
    cell that the shortest variant ignores. Also exercises the `energy` search
    path end-to-end."""
    obstacles, _ = _empty_maps()
    energy = np.zeros((GRID, GRID), dtype=np.float32)
    energy[2, 2] = 1.0

    # energy_weight (alpha) must stay in [0, 1] for the paper's admissible
    # (1 - alpha)-scaled heuristic. The NE detour (2*sqrt(2) - alpha) beats the
    # straight path (2.0) once alpha > ~0.83.
    planner = AstarPlanner(energy_weight=0.9)
    short = planner.predict(obstacles, energy, (1, 1), (3, 1), variant="shortest")
    eh = planner.predict(obstacles, energy, (1, 1), (3, 1), variant="energy")

    assert short["action"] == 1, "shortest should go straight east"
    assert eh["action"] == 7, "energy should detour NE via (2,2)"
    assert eh["energy"] > short["energy"]


def test_obstacle_detour():
    """A* routes around a wall that blocks the direct path."""
    obstacles = np.zeros((GRID, GRID), dtype=np.int32)
    obstacles[4:7, 5] = 1
    _, energy = _empty_maps()

    planner = AstarPlanner()
    res = planner.predict(obstacles, energy, (2, 5), (9, 5), variant="shortest")
    for r, c in res["path"]:
        assert obstacles[r, c] == 0, f"path visits obstacle cell ({r},{c})"
    cols_used = {c for _, c in res["path"]}
    assert cols_used != {5}, "path must leave the blocked column"


def test_start_equals_goal_raises():
    obstacles, energy = _empty_maps()
    with pytest.raises(AlreadyAtGoalError):
        AstarPlanner().predict(obstacles, energy, (3, 3), (3, 3), variant="energy")


def test_unreachable_goal_raises_no_path():
    obstacles, energy = _empty_maps()
    obstacles[0, 8] = obstacles[1, 8] = obstacles[1, 9] = 1  # wall off (0, 9)
    with pytest.raises(NoPathError):
        AstarPlanner().predict(obstacles, energy, (5, 5), (0, 9), variant="energy")


@pytest.mark.parametrize("case", ["goal_on_obstacle", "robot_on_obstacle", "shape_mismatch", "bad_variant"])
def test_input_validation_raises_value_error(case):
    """All four input-validation failures surface as ValueError."""
    obstacles, energy = _empty_maps()
    planner = AstarPlanner()

    if case == "goal_on_obstacle":
        obstacles[7, 7] = 1
        args = (obstacles, energy, (1, 1), (7, 7))
        kwargs = {"variant": "energy"}
    elif case == "robot_on_obstacle":
        obstacles[2, 2] = 1
        args = (obstacles, energy, (2, 2), (8, 8))
        kwargs = {"variant": "energy"}
    elif case == "shape_mismatch":
        args = (obstacles, np.zeros((GRID + 1, GRID), dtype=np.float32), (1, 1), (8, 8))
        kwargs = {}
    else:  # bad_variant
        args = (obstacles, energy, (1, 1), (8, 8))
        kwargs = {"variant": "nope"}

    with pytest.raises(ValueError):
        planner.predict(*args, **kwargs)


def test_max_expansions_aborts():
    """Tight cap on an open grid forces the search to give up with NoPathError."""
    obstacles, energy = _empty_maps(size=50)
    planner = AstarPlanner(max_expansions=3)
    with pytest.raises(NoPathError, match="max_expansions"):
        planner.predict(obstacles, energy, (0, 0), (49, 49), variant="shortest")
