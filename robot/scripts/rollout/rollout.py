#!/usr/bin/env python3
"""
Multi-planner rollout + route visualizer (workstation / off-robot tool).

Given a single JSON scenario bundle (obstacle map, energy map, robot start,
goal), this simulates how each of the experiment's three planners would drive a
single robot across the map, then renders the routes and reports the comparison
metrics that matter for the energy-harvesting paper replication. Use it to
sanity-check a map BEFORE running the physical experiment: does the energy-aware
planner actually detour into the bright cells? does PPO reach the goal? how much
longer is its route than the shortest path?

The three planners (select a subset with --planners):
  * astar_shortest : classic 8-connected A*, minimises geometric path length.
  * astar_energy   : energy-aware A*, trades a little distance for harvested
                     energy (energy_weight controls how much).
  * ppo            : the trained PPO policy (ONNX backend, numpy + onnxruntime).

It imports the SAME pure-Python planner layers the ROS nodes use
(astar_planner.astar_search, ppo_planner.ppo_inference), so a route here is the
exact route the deployed service would produce for the same inputs. No ROS /
colcon build is required — the src/ package dirs are added to sys.path below.

How the rollout moves
---------------------
Every step, a planner is queried for the current cell + goal and returns one
action (0..7); the robot moves one cell. This mirrors the live experiment, where
the orchestrator queries `predict` once per cell-step (closed loop).
  * A* and deterministic PPO -> a single greedy/optimal trajectory. Terminates
    on goal / blocked / loop / step-cap.
  * stochastic PPO (the default for PPO) -> samples actions like PPO was
    trained; runs --trials runs, overlays them, reports the success rate. A
    blocked move is a no-op (the robot stays), matching the training env.

INPUT CONVENTION (repo [x][y] frame — see SCENARIO_FORMAT.md and the project's
docs/coordinate-conventions.md). This is byte-identical to what the orchestrator
sends the planners over the bridge:
  * Maps are 2D arrays indexed [x][y]: FIRST index = x (East, +x = East),
    SECOND index = y (North, +y = North). Origin (0, 0) = bottom-left.
  * robot / goal are integer [x, y] cells.
  * obstacle_map cells: 0 = free, non-zero = obstacle.
  * energy_map cells: floats in [0, 1] (higher = brighter / better harvesting);
    out-of-range values are clipped.
The rendered PNGs use the same frame: x increases to the right, y upward.

POLICY CAVEAT: the PPO policy is purely local — it sees a 10x10 window centred
on the robot and clips the goal to that window's edge. On the fixed 10x10
experiment grid the window IS the whole map, so this is a non-issue; on larger
maps the policy reacts to goal *direction*, not true distance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless: we only ever save PNGs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap


# -- Make the repo's pure-Python planner layers importable without colcon ------
REPO_ROOT = Path(__file__).resolve().parents[2]
for _pkg in ("tb3_planner_common", "astar_planner", "ppo_planner"):
    _p = str(REPO_ROOT / "src" / _pkg)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tb3_planner_common.directions import DIRECTION  # noqa: E402  action -> (dx, dy)

DEFAULT_PPO_MODEL = REPO_ROOT / "src" / "ppo_planner" / "models" / "AIPPOm10EH_continued.onnx"
ALL_PLANNERS = ("astar_shortest", "astar_energy", "ppo")

# The PPO policy's local view: a WINDOW x WINDOW box centred on the robot, with
# the centre at HALF (mirrors local_obs_size / half in ppo_inference._build_obs).
WINDOW = 10
HALF = 5


# =============================================================================
# Scenario
# =============================================================================
class Scenario:
    """One map + start + goal in the repo [x][y] frame."""

    def __init__(self, data: dict, source: Path):
        self.json_name = data.get("name")
        self.name = _output_stem(source)

        for key in ("obstacle_map", "energy_map", "robot", "goal"):
            if key not in data:
                raise ValueError(f"scenario must contain '{key}'")

        self.obstacle = np.asarray(data["obstacle_map"])
        self.energy = np.asarray(data["energy_map"], dtype=np.float32)

        if self.obstacle.ndim != 2:
            raise ValueError(f"obstacle_map must be 2D, got shape {self.obstacle.shape}")
        if self.energy.ndim != 2:
            raise ValueError(f"energy_map must be 2D, got shape {self.energy.shape}")
        if self.obstacle.shape != self.energy.shape:
            raise ValueError(
                f"obstacle_map {self.obstacle.shape} and energy_map "
                f"{self.energy.shape} must have the same shape"
            )

        # shape is (X-extent, Y-extent) because arrays are indexed [x][y].
        self.nx, self.ny = self.obstacle.shape
        self.robot = self._cell("robot", data["robot"])
        self.goal = self._cell("goal", data["goal"])

        # default cap: on a static map a deterministic policy must repeat a cell
        # within nx*ny steps; loop detection usually fires far sooner.
        self.max_steps = int(data.get("max_steps", self.nx * self.ny + 10))

        self._validate()

    def _cell(self, field: str, value) -> tuple[int, int]:
        if not (isinstance(value, (list, tuple)) and len(value) == 2):
            raise ValueError(f"{field} must be a 2-element [x, y] list")
        return int(value[0]), int(value[1])

    def in_bounds(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        return 0 <= x < self.nx and 0 <= y < self.ny

    def _validate(self) -> None:
        for field, cell in (("robot", self.robot), ("goal", self.goal)):
            if not self.in_bounds(cell):
                raise ValueError(
                    f"{field} {list(cell)} is outside the {self.nx}x{self.ny} grid"
                )
        warnings = []
        if self.obstacle[self.robot] != 0:
            warnings.append(f"robot start {list(self.robot)} is on an obstacle cell")
        if self.obstacle[self.goal] != 0:
            warnings.append(f"goal {list(self.goal)} is on an obstacle cell (unreachable)")
        lo, hi = float(self.energy.min()), float(self.energy.max())
        if lo < 0.0 or hi > 1.0:
            warnings.append(
                f"energy_map range [{lo:.3f}, {hi:.3f}] falls outside [0, 1]; "
                f"values will be clipped"
            )
        self.warnings = warnings


def _output_stem(scenario_path: Path, override: str | None = None) -> str:
    if override:
        return override
    stem = scenario_path.stem
    if stem.endswith("_scenario"):
        stem = stem[: -len("_scenario")]
    return stem


# =============================================================================
# Planner adapters — uniform per-step interface: step(obstacle, energy, pos, goal) -> action
# =============================================================================
class PlannerStepError(Exception):
    """A planner could not produce a step (e.g. A* found no path)."""


class _AstarAdapter:
    def __init__(self, variant: str, energy_weight: float):
        from astar_planner.astar_search import AstarPlanner  # lazy; light deps
        self.variant = variant
        self.name = f"astar_{variant}"
        self.label = "A* energy" if variant == "energy" else "A* shortest"
        self.stochastic = False
        self._planner = AstarPlanner(energy_weight=energy_weight)

    def step(self, obstacle, energy, pos, goal) -> int:
        from astar_planner.astar_search import AlreadyAtGoalError, NoPathError
        try:
            res = self._planner.predict(obstacle, energy, pos, goal, variant=self.variant)
        except (NoPathError, AlreadyAtGoalError, ValueError) as e:
            raise PlannerStepError(str(e)) from e
        return int(res["action"])


class _PpoAdapter:
    def __init__(self, model_path: Path, grid_size, deterministic: bool, seed):
        from ppo_planner.ppo_inference import PPOPlanner  # lazy; needs onnxruntime
        self.name = "ppo"
        self.label = "PPO (argmax)" if deterministic else "PPO (sampled)"
        self.stochastic = not deterministic
        self._planner = PPOPlanner(
            str(model_path), grid_size=grid_size, deterministic=deterministic, seed=seed)

    def set_seed(self, seed) -> None:
        self._planner.set_seed(seed)

    def step(self, obstacle, energy, pos, goal) -> int:
        return int(self._planner.predict(obstacle, energy, pos, goal))


def build_adapter(name: str, args, sc: Scenario):
    if name == "astar_shortest":
        return _AstarAdapter("shortest", args.energy_weight)
    if name == "astar_energy":
        return _AstarAdapter("energy", args.energy_weight)
    if name == "ppo":
        model = args.ppo_model or DEFAULT_PPO_MODEL
        if not Path(model).exists():
            raise FileNotFoundError(f"PPO model not found: {model}")
        return _PpoAdapter(Path(model), (sc.nx, sc.ny),
                           deterministic=args.ppo_deterministic, seed=args.seed)
    raise ValueError(f"unknown planner {name!r}; expected one of {ALL_PLANNERS}")


# =============================================================================
# Metrics helpers (repo [x][y] frame)
# =============================================================================
def _path_length_geom(path: list[tuple[int, int]]) -> float:
    """Euclidean route length: 1 per cardinal step, sqrt(2) per diagonal."""
    return float(sum(math.hypot(b[0] - a[0], b[1] - a[1])
                     for a, b in zip(path, path[1:])))


def _energy_harvested(sc: Scenario, path: list[tuple[int, int]]) -> float:
    """Sum of energy_map over each cell the robot occupies (start + every landed
    cell; a revisited cell is counted each time it is occupied)."""
    en = np.clip(sc.energy, 0.0, 1.0)
    return float(sum(float(en[c]) for c in path))


def _label_dir(dx: int, dy: int) -> str:
    horz = {-1: "W", 0: "", 1: "E"}[dx]
    vert = {-1: "S", 0: "", 1: "N"}[dy]
    return (vert + horz) or "stay"


# =============================================================================
# Rollouts
# =============================================================================
def rollout_deterministic(adapter, sc: Scenario) -> dict:
    """Drive the policy from start until goal / blocked / loop / no-path / cap."""
    obstacle = (sc.obstacle > 0).astype(np.int32)
    energy = np.clip(sc.energy.astype(np.float32), 0.0, 1.0)

    pos = sc.robot
    path = [pos]
    steps: list[dict] = []
    visited = {pos}
    status = "max_steps_reached"
    message = ""

    for i in range(sc.max_steps):
        if pos == sc.goal:
            status = "reached_goal"
            break
        try:
            action = adapter.step(obstacle, energy, pos, sc.goal)
        except PlannerStepError as e:
            status = "no_path"
            message = str(e)
            break

        dx, dy = DIRECTION[action]
        nxt = (pos[0] + dx, pos[1] + dy)
        in_bounds = sc.in_bounds(nxt)
        blocked = (not in_bounds) or (in_bounds and obstacle[nxt] != 0)

        steps.append({
            "step": i, "x": pos[0], "y": pos[1],
            "action": int(action), "direction": _label_dir(dx, dy),
            "dx": dx, "dy": dy, "next_x": nxt[0], "next_y": nxt[1],
            "blocked": blocked,
        })

        if blocked:
            status = "blocked"
            break
        pos = nxt
        path.append(pos)
        if pos in visited:
            status = "loop_detected"
            break
        visited.add(pos)

    reached = status == "reached_goal"
    result = {
        "mode": "deterministic",
        "status": status,
        "message": message,
        "reached_goal": reached,
        "num_steps": len(steps),
        "path": path,
        "path_cells": len(path),
        "path_length_geom": _path_length_geom(path),
        "energy_harvested": _energy_harvested(sc, path),
        "steps": steps,
    }
    result["diagnostics"] = _diagnose(sc, status, steps, path, message)
    return result


def rollout_stochastic(adapter, sc: Scenario, n_trials: int, seed: int) -> dict:
    """Sample the policy `n_trials` times (training-env dynamics).

    A blocked move is a no-op — the robot stays and the episode continues —
    and success is landing exactly on the goal cell. Loop detection does not
    apply (revisiting a cell under sampling is normal, not a dead cycle).
    """
    obstacle = (sc.obstacle > 0).astype(np.int32)
    energy = np.clip(sc.energy.astype(np.float32), 0.0, 1.0)

    trials = []
    for t in range(n_trials):
        if seed is not None and hasattr(adapter, "set_seed"):
            adapter.set_seed(seed + t)  # reproducible but distinct per trial
        pos = sc.robot
        path = [pos]
        steps: list[dict] = []
        reached = False
        for i in range(sc.max_steps):
            if pos == sc.goal:
                reached = True
                break
            action = adapter.step(obstacle, energy, pos, sc.goal)
            dx, dy = DIRECTION[action]
            nxt = (pos[0] + dx, pos[1] + dy)
            in_bounds = sc.in_bounds(nxt)
            blocked = (not in_bounds) or (in_bounds and obstacle[nxt] != 0)
            steps.append({
                "step": i, "x": pos[0], "y": pos[1],
                "action": int(action), "direction": _label_dir(dx, dy),
                "dx": dx, "dy": dy, "next_x": nxt[0], "next_y": nxt[1],
                "blocked": blocked,
            })
            if not blocked:
                pos = nxt
                path.append(pos)
            # blocked => stay put and keep sampling (training semantics)
        dist = abs(pos[0] - sc.goal[0]) + abs(pos[1] - sc.goal[1])
        trials.append({
            "reached": reached, "moves": len(path) - 1, "path": path,
            "steps": steps, "final_dist": dist,
            "path_length_geom": _path_length_geom(path),
            "energy_harvested": _energy_harvested(sc, path),
        })

    n_reached = sum(t["reached"] for t in trials)
    moves_ok = [t["moves"] for t in trials if t["reached"]]
    # representative = shortest successful run, else the one that got closest.
    rep = (min((t for t in trials if t["reached"]), key=lambda t: t["moves"])
           if n_reached else min(trials, key=lambda t: t["final_dist"]))

    return {
        "mode": "stochastic",
        "seed": seed,
        "n_trials": n_trials,
        "n_reached": n_reached,
        "success_rate": n_reached / n_trials if n_trials else 0.0,
        "moves_min": min(moves_ok) if moves_ok else None,
        "moves_median": int(round(float(np.median(moves_ok)))) if moves_ok else None,
        "moves_max": max(moves_ok) if moves_ok else None,
        "trials": trials,
        "representative": rep,
        # fields mirrored from the representative run for the shared renderers:
        "status": "reached_goal" if rep["reached"] else "not_reached",
        "reached_goal": rep["reached"],
        "path": rep["path"],
        "steps": rep["steps"],
        "num_steps": len(rep["steps"]),
        "path_cells": len(rep["path"]),
        "path_length_geom": rep["path_length_geom"],
        "energy_harvested": rep["energy_harvested"],
        "diagnostics": {"status": "reached_goal" if rep["reached"] else "not_reached"},
    }


def _diagnose(sc: Scenario, status: str, steps: list, path: list, message: str) -> dict:
    diag = {"status": status}
    if status == "blocked" and steps:
        last = steps[-1]
        view = _view_window((last["x"], last["y"]))
        goal_in_view = (view["x0"] <= sc.goal[0] <= view["x1"] and
                        view["y0"] <= sc.goal[1] <= view["y1"])
        diag.update({
            "blocked_from": [last["x"], last["y"]],
            "blocked_cell": [last["next_x"], last["next_y"]],
            "blocked_action": last["action"],
            "view_window": view,
            "goal_in_view": bool(goal_in_view),
        })
    elif status == "loop_detected" and len(path) >= 2:
        cyc = path[-1]
        diag.update({"cycle_cell": list(cyc), "first_visited_step": path.index(cyc)})
    diag["reason"] = _failure_reason(diag, message)
    return diag


def _view_window(cell: tuple[int, int]) -> dict:
    x, y = cell
    return {"x0": x - HALF, "x1": x - HALF + WINDOW - 1,
            "y0": y - HALF, "y1": y - HALF + WINDOW - 1}


def _failure_reason(diag: dict, message: str) -> str:
    status = diag["status"]
    if status == "reached_goal":
        return "Reached the goal."
    if status == "no_path":
        return f"NO PATH: {message}"
    if status == "blocked":
        cell = diag["blocked_cell"]
        if not diag.get("goal_in_view", True):
            return (f"BLOCKED: greedy move into obstacle/edge at {cell}. The goal was "
                    f"OUTSIDE the policy's 10x10 view and got clipped to its edge, so the "
                    f"policy can't see the detour around the wall.")
        return (f"BLOCKED: greedy move into obstacle/edge at {cell}. The goal was inside the "
                f"view, but the policy steered into a local dead-end.")
    if status == "loop_detected":
        return (f"LOOP: re-entered cell {diag.get('cycle_cell')} (first seen at step "
                f"{diag.get('first_visited_step')}). A deterministic policy on a static map "
                f"cycles forever from here.")
    if status == "max_steps_reached":
        return "Hit the step cap without reaching the goal or detecting a cycle."
    return status


def shortest_reference(sc: Scenario) -> float | None:
    """Geometric length of the optimal (shortest-path A*) route start->goal, or
    None if no path / invalid. Used as the path-length-efficiency baseline."""
    try:
        from astar_planner.astar_search import AstarPlanner, NoPathError, AlreadyAtGoalError
        res = AstarPlanner().predict(sc.obstacle, sc.energy, sc.robot, sc.goal, variant="shortest")
    except AlreadyAtGoalError:
        return 0.0
    except (NoPathError, ValueError):
        return None  # genuinely no reference path / invalid endpoints
    except Exception as e:  # unexpected — surface it but don't crash the tool
        print(f"WARNING: shortest-path reference (L*) failed unexpectedly: {e!r}")
        return None
    path = [(int(x), int(y)) for x, y in res["path"]]
    return _path_length_geom(path)


# =============================================================================
# Rendering (repo [x][y] frame: x -> right, y -> up, origin bottom-left)
# =============================================================================
def _fig_for(sc: Scenario, n_panels: int = 1):
    w = max(4.5, min(13.0, sc.nx * 0.5 + 2))
    h = max(4.0, min(13.0, sc.ny * 0.45 + 2.2))
    return plt.subplots(1, n_panels, figsize=(w * n_panels, h), squeeze=False)


def _draw_base(fig, ax, sc: Scenario, colorbar: bool = True) -> None:
    """Energy heatmap + obstacles + start/goal + gridlines, in [x][y] frame."""
    extent = (-0.5, sc.nx - 0.5, -0.5, sc.ny - 0.5)
    # energy[x][y] -> transpose so imshow's row axis is y (vertical) and origin
    # is lower => y increases upward, x increases rightward.
    im = ax.imshow(np.clip(sc.energy, 0.0, 1.0).T, origin="lower", cmap="viridis",
                   vmin=0.0, vmax=1.0, interpolation="nearest", extent=extent)
    if colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("energy (0 = dark, 1 = bright)")

    obstacle = (sc.obstacle > 0)
    obs_overlay = np.ma.masked_where(~obstacle, np.ones_like(sc.obstacle, dtype=float))
    ax.imshow(obs_overlay.T, origin="lower", cmap=ListedColormap(["#202020"]),
              vmin=0, vmax=1, interpolation="nearest", extent=extent)

    ax.plot(sc.robot[0], sc.robot[1], "o", color="#2ecc40", ms=12,
            markeredgecolor="black", zorder=5, label="start")
    ax.plot(sc.goal[0], sc.goal[1], "*", color="#ff4136", ms=18,
            markeredgecolor="black", zorder=5, label="goal")

    if max(sc.nx, sc.ny) <= 30:
        ax.set_xticks(np.arange(-0.5, sc.nx, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, sc.ny, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.3, alpha=0.25)

    ax.set_xlabel("x  (East →)")
    ax.set_ylabel("y  (North ↑)")
    ax.set_xlim(-0.5, sc.nx - 0.5)
    ax.set_ylim(-0.5, sc.ny - 0.5)
    ax.set_aspect("equal")


def _draw_route(ax, path, color="white", lw=2.0, ms=3.0, alpha=1.0,
                arrows=True, label=None, zorder=3):
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    ax.plot(xs, ys, "-", color=color, lw=lw, alpha=alpha, zorder=zorder, label=label)
    ax.plot(xs, ys, "o", color=color, ms=ms, alpha=alpha, zorder=zorder)
    if arrows and len(path) > 1:
        stride = max(1, (len(path) - 1) // 25)
        for i in range(0, len(path) - 1, stride):
            (x0, y0), (x1, y1) = path[i], path[i + 1]
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2), zorder=zorder + 1)


def _draw_diagnostics(ax, sc: Scenario, result: dict) -> None:
    diag = result.get("diagnostics", {})
    if diag.get("status") == "blocked":
        vw = diag["view_window"]
        ax.add_patch(mpatches.Rectangle(
            (vw["x0"] - 0.5, vw["y0"] - 0.5), WINDOW, WINDOW,
            fill=False, edgecolor="#ff851b", lw=1.6, ls="--", zorder=4,
            label="policy 10x10 view"))
        bx, by = diag["blocked_cell"]
        ax.plot(bx, by, "x", color="#ff4136", ms=15, mew=3, zorder=6, label="blocked move")
        if not diag.get("goal_in_view", True):
            gx = int(np.clip(sc.goal[0], vw["x0"], vw["x1"]))
            gy = int(np.clip(sc.goal[1], vw["y0"], vw["y1"]))
            ax.plot(gx, gy, "*", mfc="none", mec="white", ms=16, mew=1.6,
                    zorder=6, label="goal clipped to view")
            ax.plot([gx, sc.goal[0]], [gy, sc.goal[1]], ":", color="white",
                    lw=1.0, alpha=0.7, zorder=3)
    elif diag.get("status") == "loop_detected" and "cycle_cell" in diag:
        cx, cy = diag["cycle_cell"]
        ax.plot(cx, cy, "o", mfc="none", mec="#ff851b", ms=16, mew=2.5,
                zorder=6, label="cycle re-entry")


def render_planner_png(sc: Scenario, adapter_label: str, result: dict,
                       ref_len: float | None, out_path: Path) -> None:
    """One planner, one PNG. Stochastic results overlay all trials."""
    fig, axes = _fig_for(sc, 1)
    ax = axes[0][0]
    _draw_base(fig, ax, sc)

    import textwrap
    if result["mode"] == "stochastic":
        drew_ok = drew_fail = False
        for tr in result["trials"]:
            color = "#2ecc40" if tr["reached"] else "#ff4136"
            lbl = ("reached goal" if (tr["reached"] and not drew_ok)
                   else ("did not reach" if (not tr["reached"] and not drew_fail) else None))
            _draw_route(ax, tr["path"], color=color, lw=1.3, ms=0, alpha=0.55,
                        arrows=False, label=lbl, zorder=3 if tr["reached"] else 2)
            drew_ok = drew_ok or tr["reached"]
            drew_fail = drew_fail or (not tr["reached"])
        _draw_route(ax, result["representative"]["path"], color="white", lw=2.2,
                    ms=0, alpha=0.95, arrows=True, label="representative", zorder=4)
        moves = (f"moves {result['moves_min']}/{result['moves_median']}/{result['moves_max']} "
                 f"(min/med/max)" if result["moves_min"] is not None else "no trial reached goal")
        title = (f"{sc.name} | {adapter_label} | reached {result['n_reached']}/{result['n_trials']} "
                 f"(seed {result['seed']})\nstart={list(sc.robot)} goal={list(sc.goal)} "
                 f"grid {sc.nx}x{sc.ny} | {moves}")
        good = result["n_reached"] > 0
    else:
        _draw_route(ax, result["path"], color="white", lw=2.0, ms=3.0, arrows=True)
        _draw_diagnostics(ax, sc, result)
        eff = (f" | len/shortest={result['path_length_geom'] / ref_len:.2f}"
               if (ref_len and result["reached_goal"]) else "")
        title = (f"{sc.name} | {adapter_label} | {result['status']} | "
                 f"{result['num_steps']} steps, EH={result['energy_harvested']:.2f}{eff}\n"
                 f"start={list(sc.robot)} goal={list(sc.goal)} grid {sc.nx}x{sc.ny}")
        reason = result["diagnostics"].get("reason", "")
        if reason and not result["reached_goal"]:
            title += "\n" + "\n".join(textwrap.wrap(reason, width=84))
        good = result["reached_goal"]

    ax.legend(loc="upper left", framealpha=0.85, fontsize=8)
    ax.set_title(title, fontsize=9, color=("black" if good else "#b30000"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_comparison_png(sc: Scenario, results: dict, ref_len: float | None,
                          out_path: Path) -> None:
    """All planners side by side on the shared background."""
    names = list(results.keys())
    fig, axes = _fig_for(sc, len(names))
    for ax, name in zip(axes[0], names):
        r = results[name]
        _draw_base(fig, ax, sc, colorbar=False)
        if r["mode"] == "stochastic":
            for tr in r["trials"]:
                _draw_route(ax, tr["path"], color=("#2ecc40" if tr["reached"] else "#ff4136"),
                            lw=1.0, ms=0, alpha=0.4, arrows=False, zorder=2)
            _draw_route(ax, r["representative"]["path"], color="white", lw=2.0, ms=0,
                        arrows=True, zorder=4)
            head = (f"{r['label']}: {r['n_reached']}/{r['n_trials']} reached")
        else:
            ok = r["reached_goal"]
            _draw_route(ax, r["path"], color="white", lw=2.0, ms=2.5, arrows=True)
            _draw_diagnostics(ax, sc, r)
            head = f"{r['label']}: {r['status']}"
        eff = (f", L/L*={r['path_length_geom'] / ref_len:.2f}"
               if (ref_len and r["reached_goal"]) else "")
        ax.set_title(f"{head}\nsteps={r['num_steps']}, EH={r['energy_harvested']:.2f}{eff}",
                     fontsize=9)
    fig.suptitle(f"{sc.name}  —  planner comparison  (start={list(sc.robot)} "
                 f"goal={list(sc.goal)} grid {sc.nx}x{sc.ny})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# =============================================================================
# Output writers
# =============================================================================
def write_csv(result: dict, out_path: Path) -> None:
    fields = ["step", "x", "y", "action", "direction", "dx", "dy",
              "next_x", "next_y", "blocked"]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in result["steps"]:
            w.writerow(row)


def _planner_summary(r: dict, ref_len: float | None) -> dict:
    out = {
        "label": r["label"],
        "mode": r["mode"],
        "status": r["status"],
        "reached_goal": r["reached_goal"],
        "num_steps": r["num_steps"],
        "path_cells": r["path_cells"],
        "path_length_geom": round(r["path_length_geom"], 4),
        "energy_harvested": round(r["energy_harvested"], 4),
    }
    if ref_len:
        out["shortest_path_length"] = round(ref_len, 4)
        if r["reached_goal"]:
            out["path_length_ratio"] = round(r["path_length_geom"] / ref_len, 4)
    if r["path_length_geom"] > 0:
        out["energy_per_length"] = round(r["energy_harvested"] / r["path_length_geom"], 4)
    if r["mode"] == "stochastic":
        out.update({
            "n_trials": r["n_trials"], "n_reached": r["n_reached"],
            "success_rate": round(r["success_rate"], 4),
            "moves_min": r["moves_min"], "moves_median": r["moves_median"],
            "moves_max": r["moves_max"], "seed": r["seed"],
        })
    else:
        out["reason"] = r["diagnostics"].get("reason", "")
    return out


def write_comparison_json(sc: Scenario, results: dict, ref_len: float | None,
                          args, out_path: Path) -> dict:
    summary = {
        "name": sc.name,
        "grid_shape": [sc.nx, sc.ny],
        "frame": "repo [x][y]: x=East (+x East), y=North (+y North), origin bottom-left",
        "start": list(sc.robot),
        "goal": list(sc.goal),
        "max_steps": sc.max_steps,
        "shortest_path_length": (round(ref_len, 4) if ref_len is not None else None),
        "energy_weight": args.energy_weight,
        "warnings": sc.warnings,
        "planners": {name: _planner_summary(r, ref_len) for name, r in results.items()},
        "notes": (
            "energy_harvested = sum of energy_map over each occupied cell along the route "
            "(start + every landed cell). path_length_geom counts 1 per cardinal step and "
            "sqrt(2) per diagonal. path_length_ratio = path_length_geom / shortest_path_length "
            "(>= 1; closer to 1 = more direct). PPO is local (10x10 window); on grids > 10 it "
            "reacts to goal direction, not true distance."
        ),
    }
    out_path.write_text(json.dumps(summary, indent=2))
    return summary


def _print_table(summary: dict) -> None:
    cols = ["planner", "status", "steps", "cells", "geom_len", "L/L*", "EH", "EH/len", "succ"]
    widths = [14, 16, 6, 6, 9, 6, 8, 7, 6]
    print("\n" + "  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("  ".join("-" * w for w in widths))
    for name, p in summary["planners"].items():
        succ = (f"{p['n_reached']}/{p['n_trials']}" if p["mode"] == "stochastic" else
                ("yes" if p["reached_goal"] else "no"))
        row = [
            name,
            p["status"],
            str(p["num_steps"]),
            str(p["path_cells"]),
            f"{p['path_length_geom']:.2f}",
            (f"{p['path_length_ratio']:.2f}" if "path_length_ratio" in p else "-"),
            f"{p['energy_harvested']:.2f}",
            (f"{p['energy_per_length']:.3f}" if "energy_per_length" in p else "-"),
            succ,
        ]
        print("  ".join(v.ljust(w) for v, w in zip(row, widths)))
    if summary["shortest_path_length"] is not None:
        print(f"\nshortest-path reference (L*) = {summary['shortest_path_length']:.2f} "
              f"(geometric length of optimal A* route)")


# =============================================================================
# CLI
# =============================================================================
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Roll out and compare grid planners on a scenario; plot routes + metrics.")
    p.add_argument("--scenario", type=Path, required=True, help="Scenario JSON bundle ([x][y] frame)")
    p.add_argument("--planners", default=",".join(ALL_PLANNERS),
                   help=f"Comma list from {ALL_PLANNERS} (default: all three)")
    p.add_argument("--energy-weight", type=float, default=1.0,
                   help="energy_weight for astar_energy (default 1.0; matches the experiment default)")
    p.add_argument("--ppo-model", type=Path, default=None,
                   help=f"PPO .onnx (default: {DEFAULT_PPO_MODEL})")
    p.add_argument("--ppo-deterministic", action="store_true",
                   help="Run PPO with argmax instead of sampling (single greedy trajectory; "
                        "tends to get stuck — sampling is how PPO navigates).")
    p.add_argument("--trials", type=int, default=8,
                   help="Stochastic-PPO trials to overlay (default 8; ignored for A* / argmax PPO)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for stochastic PPO (default 0)")
    p.add_argument("--max-steps", type=int, default=None, help="Override the rollout step cap")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory (default: alongside the scenario file)")
    p.add_argument("--name", default=None, help="Override the output filename stem")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    planners = [s.strip() for s in args.planners.split(",") if s.strip()]
    unknown = [p for p in planners if p not in ALL_PLANNERS]
    if unknown:
        print(f"ERROR: unknown planner(s) {unknown}; choose from {list(ALL_PLANNERS)}")
        return 1
    if args.trials < 1:
        print(f"ERROR: --trials must be >= 1, got {args.trials}")
        return 1
    if not args.scenario.exists():
        print(f"ERROR: scenario not found: {args.scenario}")
        return 1

    data = json.loads(args.scenario.read_text())
    try:
        sc = Scenario(data, args.scenario)
    except ValueError as e:
        print(f"ERROR: invalid scenario: {e}")
        return 1
    sc.name = _output_stem(args.scenario, args.name)
    if args.max_steps is not None:
        sc.max_steps = args.max_steps
    if sc.json_name and sc.json_name != sc.name:
        print(f"NOTE: JSON 'name'={sc.json_name!r} ignored for output naming; using {sc.name!r}.")
    for warn in sc.warnings:
        print(f"WARNING: {warn}")

    out_dir = args.out_dir or args.scenario.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_len = shortest_reference(sc)
    print(f"grid {sc.nx}x{sc.ny}, start={list(sc.robot)}, goal={list(sc.goal)}, "
          f"obstacles={int((sc.obstacle > 0).sum())}, shortest-path L*="
          f"{'n/a' if ref_len is None else f'{ref_len:.2f}'}")

    results: dict[str, dict] = {}
    for name in planners:
        try:
            adapter = build_adapter(name, args, sc)
        except (FileNotFoundError, ImportError, ValueError) as e:
            print(f"ERROR: could not load planner {name!r}: {e}")
            return 1

        if getattr(adapter, "stochastic", False):
            r = rollout_stochastic(adapter, sc, args.trials, args.seed)
        else:
            r = rollout_deterministic(adapter, sc)
        r["label"] = adapter.label
        results[name] = r

        if r["mode"] == "stochastic":
            print(f"  {name:14s}: reached {r['n_reached']}/{r['n_trials']} "
                  f"(rep: {r['num_steps']} steps, EH={r['energy_harvested']:.2f})")
        else:
            print(f"  {name:14s}: {r['status']:16s} "
                  f"({r['num_steps']} steps, EH={r['energy_harvested']:.2f})")

        render_planner_png(sc, adapter.label, r, ref_len,
                           out_dir / f"{sc.name}_{name}_route.png")
        write_csv(r, out_dir / f"{sc.name}_{name}_path.csv")

    render_comparison_png(sc, results, ref_len, out_dir / f"{sc.name}_comparison.png")
    summary = write_comparison_json(sc, results, ref_len, args,
                                    out_dir / f"{sc.name}_comparison.json")
    _print_table(summary)

    print(f"\nwrote: {out_dir / f'{sc.name}_comparison.png'}")
    print(f"wrote: {out_dir / f'{sc.name}_comparison.json'}")
    print(f"       + per-planner *_route.png / *_path.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
