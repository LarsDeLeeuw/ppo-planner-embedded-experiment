"""day_aggregate.py - Load one experiment day (one map, all mode/planner cells).

Shared by analyze_day, plot_trajectories, plot_energy_dynamics and
compare_maps. Builds on aggregate.load_summaries but adds what the day-level
report needs and summary.json does not carry:

  - metadata extras merged per run: goal_assisted, end_reason, start_cell
  - genuine_success: outcome == success AND NOT goal_assisted (a goal-assisted
    PPO finish is the orchestrator driving the last step, not the planner)
  - run_dir: absolute path of the run folder (for pose_log / bag access)
  - derived efficiency columns (path efficiency, harvest rate, ...)
  - turn metrics from predict_log step directions (turn_count, n_steps,
    diagonal_step_frac) — each turn is an in-place rotation, a pure cost
  - a run inventory that accounts for tracker-app crashes (outcome=aborted,
    end_reason="app exit") so messy campaigns still report clean counts

Everything here is read-only over runs/ except the summary.json cache that
aggregate/summarize_run already maintain.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate import discover_runs, load_summaries, split_and_filter  # noqa: E402

# Display order + shared styling so every figure encodes groups identically:
# planner -> colour, mode -> hatch/marker/linestyle.
PLANNER_ORDER = ["astar_shortest", "astar_energy", "ppo"]
MODE_ORDER = ["decentralized", "centralized"]
PLANNER_COLOR = {
    "astar_shortest": "#1f77b4",
    "astar_energy": "#2ca02c",
    "ppo": "#d62728",
}
MODE_MARKER = {"decentralized": "o", "centralized": "^"}
MODE_HATCH = {"decentralized": "", "centralized": "//"}
MODE_LS = {"decentralized": "-", "centralized": "--"}

CRASH_END_REASON = "app exit"


def group_key(mode: str, planner: str) -> str:
    return f"{mode}/{planner}"


def ordered_group_keys(df: pd.DataFrame) -> list[str]:
    """(mode, planner) groups present in df, in canonical display order."""
    present = set(df["group"].dropna().unique())
    ordered = [group_key(m, p) for p in PLANNER_ORDER for m in MODE_ORDER]
    return [g for g in ordered if g in present] + sorted(present - set(ordered))


def _metadata_extras(root: Path) -> pd.DataFrame:
    """Per-run metadata fields that summary.json does not include."""
    rows = []
    for rd in discover_runs(root):
        meta = yaml.safe_load((rd / "metadata.yaml").read_text()) or {}
        rows.append({
            "run_id": meta.get("run_id", rd.name),
            "run_dir": str(rd),
            "goal_assisted": bool(meta.get("goal_assisted", False)),
            "end_reason": meta.get("end_reason"),
            "start_cell": (None if meta.get("start_cell") is None
                           else "_".join(str(c) for c in meta["start_cell"])),
            "cell_size_m": meta.get("cell_size_m"),
            "started_at": meta.get("started_at"),
        })
    return pd.DataFrame(rows)


def _octile_optimal_m(row: pd.Series) -> float | None:
    """Octile (8-connected) shortest distance start->goal in metres."""
    try:
        sx, sy = (int(v) for v in str(row["start_cell"]).split("_"))
        gx, gy = (int(v) for v in str(row["goal_cell"]).split("_"))
        cell = float(row["cell_size_m"])
    except (TypeError, ValueError, AttributeError):
        return None
    dx, dy = abs(gx - sx), abs(gy - sy)
    return round((max(dx, dy) - min(dx, dy) + min(dx, dy) * math.sqrt(2)) * cell, 4)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Success semantics + per-run efficiency metrics (vectorised, NaN-safe)."""
    df = df.copy()
    df["success"] = df["outcome"] == "success"
    df["genuine_success"] = df["success"] & ~df["goal_assisted"].fillna(False)
    df["assisted_success"] = df["success"] & df["goal_assisted"].fillna(False)

    t_s = pd.to_numeric(df.get("time_to_completion_s"), errors="coerce")
    path_m = pd.to_numeric(df.get("executed_path_length_m"), errors="coerce")
    harv = pd.to_numeric(df.get("energy_harvested_mwh"), errors="coerce")
    sbc = pd.to_numeric(df.get("energy_consumed_sbc_mwh"), errors="coerce")
    opencr = pd.to_numeric(df.get("energy_consumed_opencr_mwh"), errors="coerce")

    df["octile_optimal_m"] = df.apply(_octile_optimal_m, axis=1)
    df["path_efficiency"] = (df["octile_optimal_m"] / path_m).round(4)
    df["avg_speed_mps"] = (path_m / t_s).round(4)
    hours = t_s / 3600.0
    df["harvest_power_mw"] = (harv / hours).round(2)          # mean solar power
    df["consumed_robot_mwh"] = (sbc + opencr).round(4)
    df["net_power_mw"] = ((harv - sbc - opencr) / hours).round(2)
    df["harvest_per_m_mwh"] = (harv / path_m).round(4)
    df["consumed_per_m_mwh"] = ((sbc + opencr) / path_m).round(4)
    # zero-length/zero-time degenerate runs produce inf ratios
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    return df


def _turn_metrics(run_dir: Path) -> dict[str, float | int | None]:
    """Per-run turn statistics from the predict_log step directions.

    A "turn" is an adjacent pair of executed steps with different (dx, dy):
    the grid controller realises it as an in-place rotation (time + motor
    energy, zero progress). Computed from predict_result events (warmup
    excluded), so it covers every planner including PPO. Cheap: JSONL only,
    no bag decode.
    """
    out = {"n_steps": None, "turn_count": None, "diagonal_step_frac": None}
    pl_path = Path(run_dir) / "predict_log.jsonl"
    if not pl_path.exists():
        return out
    recs = [json.loads(line) for line in pl_path.read_text().splitlines()
            if line.strip()]
    steps = [(r.get("direction_x"), r.get("direction_y"))
             for r in sorted(recs, key=lambda r: r.get("sequence", 0))
             if r.get("event") == "predict_result" and r.get("kind") != "warmup"
             and r.get("direction_x") is not None]
    if not steps:
        return out
    out["n_steps"] = len(steps)
    out["turn_count"] = sum(1 for a, b in zip(steps, steps[1:]) if a != b)
    out["diagonal_step_frac"] = round(
        sum(1 for dx, dy in steps if dx != 0 and dy != 0) / len(steps), 3)
    return out


def add_turn_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    tm = df["run_dir"].apply(lambda rd: pd.Series(_turn_metrics(Path(rd))))
    df[tm.columns] = tm
    df["turns_per_step"] = (df["turn_count"] / df["n_steps"]).round(3)
    return df


def load_day(root: Path, map_id: str | None = None, *,
             refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load every run under `root` (optionally one map only).

    Returns (df_all, df_kept, filter_log):
      df_all  - every discovered run incl. aborted/crashed (for the inventory)
      df_kept - movement runs surviving the standard filter (dry_run, aborted,
                chrony) with metadata extras + derived columns merged in
    """
    df = load_summaries(root, refresh=refresh)
    if df.empty:
        return df, df, []
    extras = _metadata_extras(root)
    df = df.merge(extras, on="run_id", how="left")
    if map_id is not None:
        df = df[df["map_id"] == map_id].copy()
        if df.empty:
            return df, df, []
    kept, _baselines, log = split_and_filter(df)
    kept = add_derived_columns(kept)
    kept = add_turn_metrics(kept)
    return df, kept, log


def run_inventory(df_all: pd.DataFrame, df_kept: pd.DataFrame) -> pd.DataFrame:
    """Per (mode, planner): discovered / crashed / filtered / kept / successes.

    Tracker-app crashes leave behind runs with outcome=aborted and
    end_reason="app exit"; they are counted separately from intentional aborts.
    """
    rows = []
    for (mode, planner), sub in df_all.groupby(["mode", "planner"]):
        kept = df_kept[(df_kept["mode"] == mode) & (df_kept["planner"] == planner)]
        aborted = sub[sub["outcome"] == "aborted"]
        crashed = int((aborted["end_reason"] == CRASH_END_REASON).sum())
        rows.append({
            "mode": mode, "planner": planner,
            "discovered": len(sub),
            "tracker_crashes": crashed,
            "other_aborted": len(aborted) - crashed,
            "dropped_other": len(sub) - len(aborted) - len(kept),
            "kept": len(kept),
            "successes": int(kept["success"].sum()),
            "genuine_successes": int(kept["genuine_success"].sum()),
            "assisted_successes": int(kept["assisted_success"].sum()),
        })
    inv = pd.DataFrame(rows)
    order = {(m, p): i for i, (p, m) in enumerate(
        (p, m) for p in PLANNER_ORDER for m in MODE_ORDER)}
    inv["_o"] = inv.apply(lambda r: order.get((r["mode"], r["planner"]), 99), axis=1)
    return inv.sort_values("_o").drop(columns="_o").reset_index(drop=True)
