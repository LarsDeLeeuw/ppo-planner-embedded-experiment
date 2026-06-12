"""Standalone PPO inference wrapper — ONNX backend.

Given an obstacle map, an energy map, Robot 0's current position, and a goal
position, returns a single discrete action (0..7) from the trained PPO policy.

Runs on `numpy` + `onnxruntime` only — NO torch / stable-baselines3 / gymnasium.
This is what makes the policy deployable on the TurtleBot's on-board computer:
`numpy` + `onnxruntime` is ~10-20 MB, where `torch` + `stable-baselines3` +
`gymnasium` is ~2 GB. The runtime never needs the training stack.

The ONNX graph (`models/AIPPOm10EH_continued.onnx`) wraps the SB3
`MultiInputActorCriticPolicy` actor pathway and emits BOTH:
  * `action` : (B, 10) int64 — deterministic argmax baked into the graph.
  * `logits` : (B, 10, 8) float32 — pre-softmax tensor.
The wrapper reads `action` when ``deterministic=True`` and samples Robot 0 from
``softmax(logits[robot_0])`` in pure numpy when ``deterministic=False`` — which
is how PPO was trained and what it needs to navigate reliably (under pure argmax
the policy walks into walls and oscillates near the goal). Pass ``seed=`` (or
call :meth:`set_seed`) for reproducible stochastic runs.

The observation construction (10x10 local window, ghost-robot stacking, goal
clipping) matches the training environment exactly, so the policy receives the
same tensors at inference time that SB3 constructed during training.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import onnxruntime as ort


_OBS_KEYS = ("robot_pos", "goal_pos", "obstacles", "energy_grid")


class PPOPlanner:
    """Loads an ONNX-exported PPO policy once, then produces Robot 0 actions.

    Usage:
        planner = PPOPlanner("AIPPOm10EH_continued.onnx")            # default 10x10
        planner = PPOPlanner("AIPPOm10EH_continued.onnx", grid_size=(100, 100))
        action  = planner.predict(obstacle_map, energy_map, (x0, y0), (gx, gy))

    Notes on the single-robot experiment:
        The model was trained with 10 active robots in the observation. When
        `other_positions` is None we stack robots 1..9 on Robot 0's own cell so
        they share its local-view position (5, 5) and become effectively
        invisible to the policy. This is grid-size-agnostic (works on both small
        and large grids). Override `other_positions` to place the ghosts
        somewhere specific.
    """

    def __init__(
        self,
        model_path: str,
        grid_size: Tuple[int, int] = (10, 10),
        num_robots: int = 10,
        deterministic: bool = False,
        seed: Optional[int] = None,
    ):
        self.grid_size = tuple(grid_size)
        self.num_robots = int(num_robots)
        self.deterministic = bool(deterministic)
        self.local_obs_size = 10
        self.rng = np.random.default_rng(seed)

        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

        # Sanity-check that the .onnx exposes the four obs keys we expect.
        actual_inputs = {i.name for i in self.session.get_inputs()}
        missing = set(_OBS_KEYS) - actual_inputs
        if missing:
            raise RuntimeError(
                f"ONNX model missing expected inputs: {sorted(missing)} "
                f"(has {sorted(actual_inputs)})"
            )

        # Stochastic mode needs the 'logits' output (added in the dual-output
        # export). An older .onnx without it would silently fail at predict()
        # time -- fail loudly here instead.
        actual_outputs = {o.name for o in self.session.get_outputs()}
        self._has_logits = "logits" in actual_outputs
        if not self.deterministic and not self._has_logits:
            raise RuntimeError(
                "stochastic mode requires the ONNX graph to expose a 'logits' "
                "output, but this .onnx only has {!r}. Re-export the policy "
                "with the dual-output (action + logits) graph.".format(sorted(actual_outputs)))

    def set_seed(self, seed: Optional[int]) -> None:
        """Reseed the stochastic sampler. Has no observable effect in
        deterministic mode, which never consults the RNG."""
        self.rng = np.random.default_rng(seed)

    def predict(
        self,
        obstacle_map: np.ndarray,
        energy_map: np.ndarray,
        robot0_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
        other_positions: Optional[List[Tuple[int, int]]] = None,
    ) -> int:
        if obstacle_map.shape != self.grid_size:
            raise ValueError(f"obstacle_map shape {obstacle_map.shape} != grid_size {self.grid_size}")
        if energy_map.shape != self.grid_size:
            raise ValueError(f"energy_map shape {energy_map.shape} != grid_size {self.grid_size}")

        obstacles = (np.asarray(obstacle_map) > 0).astype(np.int32)
        energy = np.clip(np.asarray(energy_map, dtype=np.float32), 0.0, 1.0)

        robots: List[Tuple[int, int]] = [tuple(robot0_pos)]
        if other_positions is None:
            robots.extend([tuple(robot0_pos)] * (self.num_robots - 1))
        else:
            if len(other_positions) != self.num_robots - 1:
                raise ValueError(
                    f"other_positions must have length {self.num_robots - 1}, got {len(other_positions)}"
                )
            robots.extend(tuple(p) for p in other_positions)

        obs = self._build_obs(obstacles, energy, robots, tuple(goal_pos))
        feed = {name: arr.astype(np.float32)[None, ...] for name, arr in obs.items()}

        if self.deterministic:
            actions = self.session.run(["action"], feed)[0]   # (1, 10), int64
            return int(actions[0, 0])

        # Stochastic: sample Robot 0 from softmax(logits[robot 0]). Equivalent
        # to torch.distributions.Categorical(logits=...).sample(), but in pure
        # numpy so the robot doesn't need torch.
        logits = self.session.run(["logits"], feed)[0]        # (1, 10, 8) float32
        z = logits[0, 0]                                       # (8,) Robot 0
        z = z - z.max()                                        # stable softmax
        probs = np.exp(z)
        probs /= probs.sum()
        return int(self.rng.choice(8, p=probs))

    def _build_obs(
        self,
        obstacles: np.ndarray,
        energy: np.ndarray,
        robots: List[Tuple[int, int]],
        goal: Tuple[int, int],
    ) -> dict:
        n = self.local_obs_size = 10
        half = n // 2
        rx, ry = int(robots[0][0]), int(robots[0][1])
        wx0, wy0 = rx - half, ry - half

        local_obs = np.zeros((n, n), dtype=np.int32)
        local_energy = np.zeros((n, n), dtype=np.float32)

        x0, x1 = wx0, wx0 + n
        y0, y1 = wy0, wy0 + n
        ax0, ax1 = max(0, x0), min(self.grid_size[0], x1)
        ay0, ay1 = max(0, y0), min(self.grid_size[1], y1)

        if ax0 < ax1 and ay0 < ay1:
            obs_slice = obstacles[ax0:ax1, ay0:ay1]
            en_slice = energy[ax0:ax1, ay0:ay1]
            ox0, oy0 = ax0 - x0, ay0 - y0
            local_obs[ox0:ox0 + obs_slice.shape[0], oy0:oy0 + obs_slice.shape[1]] = obs_slice
            local_energy[ox0:ox0 + en_slice.shape[0], oy0:oy0 + en_slice.shape[1]] = en_slice

        rel_robots = np.zeros((self.num_robots, 2), dtype=np.int32)
        for i, (px, py) in enumerate(robots):
            rel_robots[i, 0] = int(np.clip(int(px) - wx0, 0, 9))
            rel_robots[i, 1] = int(np.clip(int(py) - wy0, 0, 9))

        goal_rel = np.array([
            int(np.clip(int(goal[0]) - wx0, 0, 9)),
            int(np.clip(int(goal[1]) - wy0, 0, 9)),
        ], dtype=np.int32)

        return {
            "robot_pos":   rel_robots,
            "goal_pos":    goal_rel,
            "obstacles":   local_obs.astype(np.int8).reshape(-1),
            "energy_grid": np.clip(local_energy, 0.0, 1.0).astype(np.float32),
        }
