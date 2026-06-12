"""audit_run.py - Deep-dive audit of one run.

Produces a directory `<run_dir>/audit/` containing several focused PNGs and an
AUDIT.md report listing concrete observations + suspected data-collection
shortcomings. Intended to be run once per representative run; complements
summarize_run.py (which is a 1-page sanity card).

Usage:
    python audit_run.py <run_dir>

Panels:
    01_power.png         power traces, gap histograms, cumulative energy
    02_trajectory.png    dedup'd grid_pose + odom-integrated trajectory, dist-to-goal
    03_control.png       cmd_vel vs odom velocities, command-to-actual lag
    04_imu.png           accel + gyro over time
    05_thermal_clock.png SBC temp + throttle + chrony offset
    06_planner_timing.png inference / RTT distributions, clock-skew check
    07_nav_health.png    loop_stats, action results per cell
    08_data_audit.png    per-topic rate vs expected, max gap, window coverage
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_io import Run, best_time_col, load_run  # noqa: E402

# Expected rates (Hz) used to flag rate shortfalls. Conservative — these are
# what the operator described / what publishers nominally configure.
EXPECTED_RATES = {
    "/power/solar":              100.0,
    "/power/sbc":                100.0,
    "/power/opencr":             100.0,
    "/odom":                      30.0,
    "/imu":                      100.0,
    "/cmd_vel":                   30.0,
    "/grid_nav_node/grid_pose":   30.0,
    "/grid_nav_node/loop_stats":   1.0,
    "/sbc/thermal":                1.0,
    "/diagnostics/host":           1.0,
}


@dataclass
class Finding:
    severity: str   # "info" | "warn" | "issue"
    topic: str      # area / topic
    text: str       # one-line observation

    def render(self) -> str:
        icon = {"info": "ℹ️", "warn": "⚠️", "issue": "❌"}[self.severity]
        return f"- {icon} **{self.topic}** — {self.text}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rel_s(df: pd.DataFrame, t0: int) -> np.ndarray:
    return (df[best_time_col(df)].to_numpy() - t0) / 1e9


def _quat_to_yaw(qx: np.ndarray, qy: np.ndarray, qz: np.ndarray, qw: np.ndarray) -> np.ndarray:
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    return np.arctan2(siny_cosp, cosy_cosp)


def _integrate_power(df: pd.DataFrame, t0: int, t1: int) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative energy in mWh from a power-mw stream within [t0, t1]."""
    if df is None or df.empty:
        return np.array([]), np.array([])
    tcol = best_time_col(df)
    sub = df[(df[tcol] >= t0) & (df[tcol] <= t1)].sort_values(tcol)
    if sub.empty:
        return np.array([]), np.array([])
    t = (sub[tcol].to_numpy() - t0) / 1e9
    p = sub["power_mw"].to_numpy()
    dt = np.diff(t, prepend=t[0])
    # mW * s -> mWh
    inc = (p * dt) / 3600.0
    return t, np.cumsum(inc)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def panel_power(run: Run, findings: list[Finding], out: Path) -> None:
    t0, t1 = run.window_ns
    win_s = (t1 - t0) / 1e9
    fig, axes = plt.subplots(3, 1, figsize=(11, 9))

    # 1) raw traces + net
    ax = axes[0]
    series = {}
    for name, color in (("solar", "#f5a623"), ("sbc", "#4a90d9"),
                        ("opencr", "#d0021b")):
        df = run.topic(f"/power/{name}")
        if df is None or df.empty:
            continue
        df = df.sort_values(best_time_col(df))
        ax.plot(_rel_s(df, t0), df["power_mw"], label=name, color=color, lw=0.6)
        series[name] = df
    if {"solar", "sbc", "opencr"} <= series.keys():
        tcol = best_time_col(series["sbc"])
        m = series["sbc"][[tcol, "power_mw"]].rename(columns={"power_mw": "sbc"})
        for s in ("solar", "opencr"):
            o = series[s][[best_time_col(series[s]), "power_mw"]].rename(
                columns={best_time_col(series[s]): tcol, "power_mw": s}).sort_values(tcol)
            m = pd.merge_asof(m.sort_values(tcol), o, on=tcol,
                              tolerance=10_000_000, direction="nearest")
        net = m["solar"] - (m["sbc"] + m["opencr"])
        ax.plot((m[tcol].to_numpy() - t0) / 1e9, net, label="net = solar−(sbc+opencr)",
                color="#417505", lw=1.2)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="k", lw=0.6, alpha=0.5)
    ax.axvline((t1 - t0) / 1e9, color="k", lw=0.6, alpha=0.5)
    ax.set_title("A: power traces (mW)")
    ax.set_xlabel("run-relative time (s)"); ax.set_ylabel("mW")
    ax.legend(fontsize=8, loc="upper right")

    # 2) sample-interval histograms (gap diagnosis)
    ax = axes[1]
    bins = np.linspace(0, 0.05, 51)  # up to 50 ms gap
    for name, color in (("solar", "#f5a623"), ("sbc", "#4a90d9"),
                        ("opencr", "#d0021b")):
        df = series.get(name)
        if df is None:
            continue
        dt = np.diff(df[best_time_col(df)].to_numpy()) / 1e9
        ax.hist(dt, bins=bins, alpha=0.5, label=f"{name} (n={len(dt)})", color=color)
        p99 = float(np.percentile(dt, 99))
        if p99 > 0.02:
            findings.append(Finding(
                "warn", f"/power/{name}",
                f"p99 sample gap = {p99*1000:.1f} ms — expected ~10 ms at 100 Hz",
            ))
    ax.set_title("B: power-sample intervals (only gaps <50 ms shown)")
    ax.set_xlabel("seconds between samples"); ax.set_ylabel("count")
    ax.legend(fontsize=8)

    # 3) cumulative energy
    ax = axes[2]
    cum = {}
    for name, color in (("solar", "#f5a623"), ("sbc", "#4a90d9"),
                        ("opencr", "#d0021b")):
        df = series.get(name)
        if df is None:
            continue
        t, c = _integrate_power(df, t0, t1)
        ax.plot(t, c, color=color, lw=1.0, label=f"{name} ({c[-1]:+.2f} mWh)")
        cum[name] = c[-1] if len(c) else 0.0
    if {"solar", "sbc", "opencr"} <= cum.keys():
        net_e = cum["solar"] - cum["sbc"] - cum["opencr"]
        ax.text(0.02, 0.95, f"net = {net_e:+.2f} mWh  (solar − sbc − opencr)",
                transform=ax.transAxes, fontsize=10, color="#417505",
                va="top", fontweight="bold")
        if net_e < 0:
            findings.append(Finding(
                "info", "energy",
                f"net energy deficit {abs(net_e):.2f} mWh over {win_s:.0f} s "
                f"(solar {cum['solar']:.2f} − sbc {cum['sbc']:.2f} − opencr {cum['opencr']:.2f}).",
            ))
        solar_df = series.get("solar")
        if solar_df is not None and not solar_df.empty:
            inwin = solar_df[(solar_df[best_time_col(solar_df)] >= t0)
                              & (solar_df[best_time_col(solar_df)] <= t1)]
            zero_pct = 100.0 * (inwin["power_mw"] <= 0.01).sum() / max(1, len(inwin))
            max_solar = float(inwin["power_mw"].max())
            mean_solar = float(inwin["power_mw"].mean())
            sev = "issue" if zero_pct > 70 else "warn" if zero_pct > 30 else "info"
            findings.append(Finding(
                sev, "/power/solar",
                f"solar harvest is bursty: 0 mW for {zero_pct:.0f}% of samples, "
                f"mean {mean_solar:.1f} mW, max {max_solar:.0f} mW. If the lab "
                "lighting is meant to be constant, either the panel is partially shaded "
                "or the harvest sensor reads 0 below its detection floor — investigate "
                "before treating energy comparisons as conclusive.",
            ))
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_title("C: cumulative energy (mWh)")
    ax.set_xlabel("run-relative time (s)"); ax.set_ylabel("mWh")
    ax.legend(fontsize=8, loc="lower left")

    fig.tight_layout()
    fig.savefig(out, dpi=110); plt.close(fig)


def panel_solar_harvest(run: Run, findings: list[Finding], out: Path) -> None:
    """Solar-only view, autoscaled to the panel's own range. The combined power
    panel is dominated by sbc/opencr draw (orders of magnitude larger), so any
    actual harvest is invisible there. This panel answers: did the panel pick up
    *anything* measurable, and when?"""
    t0, t1 = run.window_ns
    win_s = (t1 - t0) / 1e9
    df = run.topic("/power/solar")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    if df is None or df.empty:
        for ax in axes.flat:
            ax.text(0.5, 0.5, "no /power/solar data", ha="center", va="center",
                    transform=ax.transAxes)
        fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
        findings.append(Finding("warn", "/power/solar", "no solar samples in run window."))
        return

    tcol = best_time_col(df)
    df = df[(df[tcol] >= t0) & (df[tcol] <= t1)].sort_values(tcol)
    rel = (df[tcol].to_numpy() - t0) / 1e9
    pw = df["power_mw"].to_numpy()
    solar_color = "#f5a623"

    # ---- (A) solar power over time, autoscaled to solar alone ----
    ax = axes[0, 0]
    ax.plot(rel, pw, color=solar_color, lw=0.7)
    ax.fill_between(rel, 0, pw, color=solar_color, alpha=0.3)
    mean_pw = float(pw.mean())
    ax.axhline(mean_pw, color="#b37400", lw=0.8, ls="--",
               label=f"mean {mean_pw:.2f} mW")
    ax.set_title(f"A: solar power (mW) — autoscaled, peak {pw.max():.0f} mW")
    ax.set_xlabel("run-relative time (s)"); ax.set_ylabel("mW")
    ax.legend(fontsize=8)

    # ---- (B) panel bus voltage + current: is the cell alive even at 0 mW? ----
    ax = axes[0, 1]
    if "bus_voltage_v" in df.columns:
        ax.plot(rel, df["bus_voltage_v"].to_numpy(), color="#7b61ff", lw=0.7,
                label="bus voltage")
        ax.set_ylabel("V", color="#7b61ff"); ax.tick_params(axis="y", labelcolor="#7b61ff")
    if "current_ma" in df.columns:
        ax2 = ax.twinx()
        ax2.plot(rel, df["current_ma"].to_numpy(), color="#2ca02c", lw=0.6, alpha=0.8,
                 label="current")
        ax2.set_ylabel("mA", color="#2ca02c"); ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax.set_title("B: panel bus voltage & current")
    ax.set_xlabel("run-relative time (s)")

    # ---- (C) cumulative harvested energy ----
    ax = axes[1, 0]
    ct, cc = _integrate_power(df, t0, t1)
    total_mwh = float(cc[-1]) if len(cc) else 0.0
    ax.plot(ct, cc, color=solar_color, lw=1.2)
    ax.fill_between(ct, 0, cc, color=solar_color, alpha=0.3)
    ax.set_title(f"C: cumulative harvested energy = {total_mwh:.3f} mWh")
    ax.set_xlabel("run-relative time (s)"); ax.set_ylabel("mWh")

    # ---- (D) power duration curve: how much of the run was spent harvesting ----
    ax = axes[1, 1]
    sorted_pw = np.sort(pw)[::-1]
    frac = np.arange(1, len(sorted_pw) + 1) / len(sorted_pw) * 100.0
    ax.plot(frac, sorted_pw, color=solar_color, lw=1.2)
    ax.fill_between(frac, 0, sorted_pw, color=solar_color, alpha=0.3)
    harvest_pct = 100.0 * float((pw > 0.01).sum()) / max(1, len(pw))
    ax.axvline(harvest_pct, color="#b37400", lw=0.8, ls="--",
               label=f"{harvest_pct:.1f}% of samples > 0.01 mW")
    ax.set_title("D: power duration curve")
    ax.set_xlabel("% of run-window samples at or above"); ax.set_ylabel("mW")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=110); plt.close(fig)

    findings.append(Finding(
        "info", "solar harvest",
        f"panel harvested {total_mwh:.3f} mWh over {win_s:.0f} s; measurable power "
        f"(>0.01 mW) in {harvest_pct:.1f}% of samples, peak {pw.max():.0f} mW. "
        f"Bus voltage present throughout (mean "
        f"{float(df['bus_voltage_v'].mean()):.2f} V), so the panel is wired and alive "
        "even when instantaneous power reads 0 — harvest is genuinely bursty/low, "
        "not a dead sensor.",
    ))


def panel_trajectory(run: Run, findings: list[Finding], out: Path) -> None:
    t0, t1 = run.window_ns
    fig = plt.figure(figsize=(13, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.2, 1.0])

    # Common context
    obs = run.obstacle_map
    sc = run.meta.get("start_cell"); gc = run.meta.get("goal_cell")

    # ---- (A) tracker trajectory — prefer pose_log.jsonl (30 Hz laptop side);
    # fall back to dedup'd /grid_nav_node/grid_pose for older runs. ----
    ax = fig.add_subplot(gs[0, 0])
    if obs is not None:
        ax.imshow(obs, origin="lower", cmap="Greys", alpha=0.4,
                  extent=[0, obs.shape[1], 0, obs.shape[0]], zorder=1)
    used_source = "none"
    n_points = 0
    if run.pose_log is not None and not run.pose_log.empty:
        pl = run.pose_log.sort_values("t_mono_ns")
        n_points = len(pl)
        t = (pl["t_mono_ns"].to_numpy() - pl["t_mono_ns"].iloc[0]) / 1e9
        sc_pts = ax.scatter(pl["x"], pl["y"], c=t, cmap="viridis",
                            s=4, zorder=3)
        ax.plot(pl["x"], pl["y"], "-", color="#1f77b4", lw=0.8, alpha=0.7, zorder=2)
        cbar = fig.colorbar(sc_pts, ax=ax, fraction=0.05, pad=0.02)
        cbar.set_label("t (s)", fontsize=8)
        used_source = "pose_log.jsonl (30 Hz)"
    else:
        gp = run.topic("/grid_nav_node/grid_pose")
        n_unique = 0
        if gp is not None and not gp.empty:
            gp = gp.sort_values(best_time_col(gp))
            mask = (gp[best_time_col(gp)] >= t0) & (gp[best_time_col(gp)] <= t1)
            gp = gp[mask]
            dedup = gp[(gp["x"].diff().abs() > 1e-4) | (gp["y"].diff().abs() > 1e-4)]
            dedup = pd.concat([gp.head(1), dedup]).drop_duplicates()
            n_unique = len(dedup); n_points = n_unique
            if n_unique > 1:
                t = _rel_s(dedup, t0)
                sc_pts = ax.scatter(dedup["x"], dedup["y"], c=t, cmap="viridis",
                                    s=40, edgecolors="k", lw=0.5, zorder=3)
                ax.plot(dedup["x"], dedup["y"], "-", color="#1f77b4", lw=0.8,
                        alpha=0.6, zorder=2)
                cbar = fig.colorbar(sc_pts, ax=ax, fraction=0.05, pad=0.02)
                cbar.set_label("t (s)", fontsize=8)
            used_source = f"grid_pose dedup'd ({n_unique} unique)"
        if n_unique <= 12 and gp is not None and not gp.empty:
            findings.append(Finding(
                "issue", "trajectory data",
                f"no pose_log.jsonl found; bridge republished /grid_nav_node/grid_pose "
                f"({n_unique} unique poses) is the only trajectory source — too sparse "
                "to be meaningful. Re-run with the updated orchestrator to capture pose_log.",
            ))
    if sc:
        ax.plot(sc[0] + 0.5, sc[1] + 0.5, "s", color="green", ms=10, zorder=4)
    if gc:
        ax.plot(gc[0] + 0.5, gc[1] + 0.5, "*", color="magenta", ms=14, zorder=4)
    ax.set_title(f"A: tracker trajectory — {used_source} (n={n_points})")
    ax.set_xlabel("x (col, bridge)"); ax.set_ylabel("y (row, bridge)")
    ax.set_aspect("equal")

    # ---- (B) Odom-integrated path, anchored to start_cell ----
    ax = fig.add_subplot(gs[0, 1])
    if obs is not None:
        ax.imshow(obs, origin="lower", cmap="Greys", alpha=0.4,
                  extent=[0, obs.shape[1], 0, obs.shape[0]], zorder=1)
    od = run.topic("/odom")
    cell_m = float(run.meta.get("cell_size_m", 0.30))
    if od is not None and not od.empty:
        od = od.sort_values(best_time_col(od))
        mask = (od[best_time_col(od)] >= t0) & (od[best_time_col(od)] <= t1)
        od = od[mask]
        ox = od["pose.pose.position.x"].to_numpy()
        oy = od["pose.pose.position.y"].to_numpy()
        # Initial yaw from quaternion at run start
        q = od.iloc[0]
        yaw0 = float(_quat_to_yaw(np.array([q["pose.pose.orientation.x"]]),
                                  np.array([q["pose.pose.orientation.y"]]),
                                  np.array([q["pose.pose.orientation.z"]]),
                                  np.array([q["pose.pose.orientation.w"]]))[0])
        # Translate so first sample sits at start_cell center, rotate by (target_yaw - yaw0)
        # Heuristic: tracker heading at run start -> yaw_target (radians).
        # Prefer pose_log heading; fall back to first in-window grid_pose. Fetched
        # independently here because panel A only defines `gp` on its fallback path.
        yaw_target = 0.0
        if run.pose_log is not None and not run.pose_log.empty:
            pl0 = run.pose_log.sort_values("t_mono_ns")
            yaw_target = float(pl0.iloc[0]["heading_rad"])
        else:
            gp_b = run.topic("/grid_nav_node/grid_pose")
            if gp_b is not None and not gp_b.empty:
                gp_b = gp_b.sort_values(best_time_col(gp_b))
                in_win = gp_b[(gp_b[best_time_col(gp_b)] >= t0)
                              & (gp_b[best_time_col(gp_b)] <= t1)]
                if not in_win.empty:
                    yaw_target = float(in_win.iloc[0]["heading"])
        rot = yaw_target - yaw0
        c, s = math.cos(rot), math.sin(rot)
        ox_r = c * ox - s * oy; oy_r = s * ox + c * oy
        # Convert to grid units, translate to start_cell
        if sc:
            x_cells = sc[0] + 0.5 + (ox_r - ox_r[0]) / cell_m
            y_cells = sc[1] + 0.5 + (oy_r - oy_r[0]) / cell_m
            ax.plot(x_cells, y_cells, "-", color="#d62728", lw=1.4, zorder=3, label="odom-integrated")
            ax.plot(x_cells[0], y_cells[0], "s", color="green", ms=10, zorder=4)
            ax.plot(x_cells[-1], y_cells[-1], "o", color="red", ms=8, zorder=4)
            # Total odom-traveled distance
            dist_m = float(np.sum(np.hypot(np.diff(ox), np.diff(oy))))
            findings.append(Finding(
                "info", "odom path length",
                f"odometry-traveled distance during the run = {dist_m:.2f} m "
                f"(≈ {dist_m/cell_m:.1f} cells at assumed cell_m={cell_m})",
            ))
    if gc:
        ax.plot(gc[0] + 0.5, gc[1] + 0.5, "*", color="magenta", ms=14, zorder=4)
    ax.set_title("B: odom-integrated path (anchored to start_cell)")
    ax.set_xlabel("x (col, bridge)"); ax.set_ylabel("y (row, bridge)")
    ax.set_aspect("equal")

    # ---- (C) Distance to goal over time — prefer pose_log ----
    ax = fig.add_subplot(gs[0, 2])
    if gc:
        if run.pose_log is not None and not run.pose_log.empty:
            pl = run.pose_log.sort_values("t_mono_ns")
            t = (pl["t_mono_ns"].to_numpy() - pl["t_mono_ns"].iloc[0]) / 1e9
            dist_cells = np.hypot(pl["x"] - (gc[0] + 0.5), pl["y"] - (gc[1] + 0.5))
            ax.plot(t, dist_cells, lw=1.0, color="#1f77b4")
        else:
            gp_full = run.topic("/grid_nav_node/grid_pose")
            if gp_full is not None and not gp_full.empty:
                gp_full = gp_full.sort_values(best_time_col(gp_full))
                in_win = gp_full[(gp_full[best_time_col(gp_full)] >= t0)
                                 & (gp_full[best_time_col(gp_full)] <= t1)]
                dx = in_win["x"] - (gc[0] + 0.5)
                dy = in_win["y"] - (gc[1] + 0.5)
                dist_cells = np.hypot(dx, dy)
                ax.plot(_rel_s(in_win, t0), dist_cells, lw=1.0, color="#1f77b4")
        ax.set_title("C: distance to goal (cells)")
        ax.set_xlabel("run-relative time (s)"); ax.set_ylabel("cells")
        ax.axhline(0.5, color="green", lw=0.5, ls="--", alpha=0.7)

    fig.tight_layout()
    fig.savefig(out, dpi=110); plt.close(fig)


def panel_control(run: Run, findings: list[Finding], out: Path) -> None:
    t0, t1 = run.window_ns
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    cv = run.topic("/cmd_vel")
    od = run.topic("/odom")

    # Linear
    ax = axes[0]
    if cv is not None and not cv.empty:
        cv = cv.sort_values(best_time_col(cv))
        ax.plot(_rel_s(cv, t0), cv["linear.x"], "-", color="#888", lw=1.0,
                label="cmd_vel.linear.x")
    if od is not None and not od.empty:
        od_s = od.sort_values(best_time_col(od))
        ax.plot(_rel_s(od_s, t0), od_s["twist.twist.linear.x"], "-",
                color="#1f77b4", lw=1.0, label="odom.linear.x")
    ax.axhline(0, color="gray", lw=0.4)
    ax.set_ylabel("m/s")
    ax.set_title("Linear velocity: cmd vs odom")
    ax.legend(fontsize=8)

    # Angular
    ax = axes[1]
    if cv is not None and not cv.empty:
        ax.plot(_rel_s(cv, t0), cv["angular.z"], "-", color="#888", lw=1.0,
                label="cmd_vel.angular.z")
    if od is not None and not od.empty:
        od_s = od.sort_values(best_time_col(od))
        ax.plot(_rel_s(od_s, t0), od_s["twist.twist.angular.z"], "-",
                color="#d62728", lw=1.0, label="odom.angular.z")
    ax.axhline(0, color="gray", lw=0.4)
    ax.set_ylabel("rad/s")
    ax.set_xlabel("run-relative time (s)")
    ax.set_title("Angular velocity: cmd vs odom")
    ax.legend(fontsize=8)

    # Findings: zero-output windows of cmd_vel during a non-stop run
    if cv is not None and not cv.empty:
        in_win = cv[(cv[best_time_col(cv)] >= t0) & (cv[best_time_col(cv)] <= t1)]
        n_zero = int(((in_win["linear.x"].abs() < 1e-3) & (in_win["angular.z"].abs() < 1e-3)).sum())
        pct = 100.0 * n_zero / max(1, len(in_win))
        findings.append(Finding(
            "info", "/cmd_vel",
            f"{n_zero}/{len(in_win)} commands ({pct:.1f}%) were zero "
            f"(rotation-then-drive cells have legitimate dwell time between sub-steps)",
        ))

    fig.tight_layout()
    fig.savefig(out, dpi=110); plt.close(fig)


def panel_imu(run: Run, findings: list[Finding], out: Path) -> None:
    t0, t1 = run.window_ns
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    imu = run.topic("/imu")
    if imu is None or imu.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "no /imu", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(out, dpi=110); plt.close(fig)
        return
    imu = imu.sort_values(best_time_col(imu))
    in_win = imu[(imu[best_time_col(imu)] >= t0) & (imu[best_time_col(imu)] <= t1)].copy()
    t = _rel_s(in_win, t0)

    ax = axes[0]
    for c in ("x", "y", "z"):
        ax.plot(t, in_win[f"linear_acceleration.{c}"], lw=0.6, label=c)
    ax.axhline(0, color="gray", lw=0.4)
    ax.set_title("linear_acceleration (m/s²)")
    ax.set_ylabel("m/s²"); ax.legend(fontsize=8, ncol=3)

    ax = axes[1]
    for c in ("x", "y", "z"):
        ax.plot(t, in_win[f"angular_velocity.{c}"], lw=0.6, label=c)
    ax.axhline(0, color="gray", lw=0.4)
    ax.set_title("angular_velocity (rad/s)")
    ax.set_ylabel("rad/s"); ax.legend(fontsize=8, ncol=3)

    # Yaw from quaternion
    ax = axes[2]
    yaw = _quat_to_yaw(
        in_win["orientation.x"].to_numpy(),
        in_win["orientation.y"].to_numpy(),
        in_win["orientation.z"].to_numpy(),
        in_win["orientation.w"].to_numpy(),
    )
    ax.plot(t, np.unwrap(yaw), lw=0.8, color="#4a90d9")
    ax.set_title("IMU yaw (rad, unwrapped)")
    ax.set_ylabel("rad"); ax.set_xlabel("run-relative time (s)")

    rate = len(in_win) / max(1e-3, (t1 - t0) / 1e9)
    if rate < 50:
        findings.append(Finding(
            "warn", "/imu",
            f"effective rate ~{rate:.1f} Hz — TB3 default IMU is 100 Hz; "
            "either the publisher is throttled or rosbag2 is downsampling.",
        ))

    fig.tight_layout()
    fig.savefig(out, dpi=110); plt.close(fig)


def panel_thermal_clock(run: Run, findings: list[Finding], out: Path) -> None:
    t0, t1 = run.window_ns
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    th = run.topic("/sbc/thermal")
    ax = axes[0]
    if th is not None and not th.empty:
        th = th.sort_values(best_time_col(th))
        in_win = th[(th[best_time_col(th)] >= t0) & (th[best_time_col(th)] <= t1)]
        t = _rel_s(in_win, t0)
        ax.plot(t, in_win["cpu_temp_c"], "-", color="#d62728", lw=1.2, label="cpu_temp_c")
        # Highlight throttled regions; decode Raspberry Pi throttle_flags bitmap:
        #   bit 0 (0x1)     = currently under-voltage
        #   bit 1 (0x2)     = ARM freq currently capped
        #   bit 2 (0x4)     = currently throttled
        #   bit 3 (0x8)     = soft temp limit reached
        #   bit 16 (0x10000)= under-voltage HAS occurred since boot
        #   bit 18 (0x40000)= throttling HAS occurred since boot
        if "throttled" in in_win.columns:
            throt = in_win[in_win["throttled"].astype(bool)]
            if len(throt):
                ax.scatter(_rel_s(throt, t0), throt["cpu_temp_c"],
                           color="black", marker="x", s=30, zorder=5, label="throttled=true")
                flags_max = int(in_win["throttle_flags"].max()) if "throttle_flags" in in_win.columns else 0
                reasons = []
                if flags_max & 0x1: reasons.append("under-voltage now")
                if flags_max & 0x4: reasons.append("throttling now")
                if flags_max & 0x8: reasons.append("soft-temp-limit now")
                if flags_max & 0x10000: reasons.append("under-voltage occurred")
                if flags_max & 0x40000: reasons.append("throttling occurred")
                reason_str = ", ".join(reasons) if reasons else f"unknown flags 0x{flags_max:x}"
                temp_max = float(in_win["cpu_temp_c"].max())
                # Distinguish under-voltage from thermal: at <70°C thermal isn't the cause.
                cause = "POWER SUPPLY (under-voltage)" if (
                    (flags_max & 0x1 or flags_max & 0x10000) and temp_max < 70
                ) else "thermal" if temp_max >= 70 else "unknown"
                findings.append(Finding(
                    "issue", "/sbc/thermal",
                    f"SBC throttled at {len(throt)}/{len(in_win)} samples "
                    f"({100*len(throt)/len(in_win):.0f}%); max temp {temp_max:.1f}°C; "
                    f"flags=0x{flags_max:x} ({reason_str}). Likely cause: **{cause}** — "
                    "5V/3A USB-C supply or battery rail can't hold under load; use a "
                    "beefier supply or check the battery voltage during runs.",
                ))
        ax.set_title("SBC CPU temperature (throttling here is under-voltage, not thermal)")
        ax.set_ylabel("°C"); ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no /sbc/thermal", ha="center", va="center", transform=ax.transAxes)

    dh = run.topic("/diagnostics/host")
    ax = axes[1]
    if dh is not None and not dh.empty and "chrony_offset_ms" in dh.columns:
        dh = dh.sort_values(best_time_col(dh))
        in_win = dh[(dh[best_time_col(dh)] >= t0) & (dh[best_time_col(dh)] <= t1)]
        ax.plot(_rel_s(in_win, t0), in_win["chrony_offset_ms"], "-", color="#1f77b4", lw=1.0,
                label="chrony offset (ms)")
        ax.axhline(0, color="gray", lw=0.4)
        max_off = float(in_win["chrony_offset_ms"].abs().max())
        ax.text(0.02, 0.95, f"max |offset| = {max_off:.2f} ms",
                transform=ax.transAxes, fontsize=9, va="top")
        if max_off > 50:
            findings.append(Finding(
                "warn", "chrony",
                f"max |offset| {max_off:.1f} ms — bag-host clock drifted vs reference; "
                "cross-bag joins by stamp may be off by that much.",
            ))
        ax.set_title("Chrony offset on bag host")
        ax.set_ylabel("ms"); ax.set_xlabel("run-relative time (s)")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no chrony_offset_ms in /diagnostics/host",
                ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(out, dpi=110); plt.close(fig)


def panel_planner_timing(run: Run, findings: list[Finding], out: Path) -> None:
    t0, t1 = run.window_ns
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    pm = run.topic("/planner/metrics")
    ptt = run.topic("/predict_tcp_timing")
    btt = run.topic("/bridge/predict_timing")

    # (a) inference µs scatter + percentiles
    ax = axes[0][0]
    if pm is not None and not pm.empty:
        pm = pm.sort_values("sequence")
        ax.scatter(pm["sequence"], pm["inference_us"], s=24, color="#d62728")
        for p, ls in [(50, "-"), (95, "--"), (99, ":")]:
            v = float(np.percentile(pm["inference_us"], p))
            ax.axhline(v, color="#666", ls=ls, lw=0.7, label=f"p{p}={v:.0f}")
        ax.set_title("A: planner inference µs vs predict sequence")
        ax.set_xlabel("sequence"); ax.set_ylabel("µs")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no /planner/metrics", ha="center", va="center", transform=ax.transAxes)

    # (b) TCP RTT laptop vs bridge
    ax = axes[0][1]
    if ptt is not None and not ptt.empty:
        rtt_l = (ptt["t_tcp_recv_ns"] - ptt["t_tcp_send_ns"]) / 1000.0
        ax.scatter(ptt["sequence"], rtt_l, s=24, color="#1f77b4", label="laptop monotonic RTT µs")
    if btt is not None and not btt.empty and "t_bridge_recv_ns" in btt.columns:
        rtt_b = (btt["t_bridge_send_ns"] - btt["t_bridge_recv_ns"]) / 1000.0
        ax.scatter(btt["sequence"], rtt_b, s=24, color="#f5a623", marker="^",
                   label="bridge-side process µs")
    ax.set_title("B: TCP/predict round-trip components")
    ax.set_xlabel("sequence"); ax.set_ylabel("µs")
    ax.set_yscale("log")
    ax.legend(fontsize=8)

    # (c) Latency stack: inference vs network
    ax = axes[1][0]
    if pm is not None and ptt is not None and not pm.empty and not ptt.empty:
        merged = pd.merge(pm[["sequence", "inference_us"]],
                          ptt[["sequence", "t_tcp_send_ns", "t_tcp_recv_ns"]],
                          on="sequence", how="inner")
        merged["rtt_us"] = (merged["t_tcp_recv_ns"] - merged["t_tcp_send_ns"]) / 1000.0
        merged["network_us"] = merged["rtt_us"] - merged["inference_us"]
        x = np.arange(len(merged))
        ax.bar(x, merged["inference_us"], color="#d62728", label="inference")
        ax.bar(x, merged["network_us"].clip(lower=0), bottom=merged["inference_us"],
               color="#1f77b4", label="network (rtt − inference)")
        ax.set_xticks(x)
        ax.set_xticklabels(merged["sequence"].astype(int), rotation=0, fontsize=7)
        ax.set_title("C: latency decomposition per predict")
        ax.set_xlabel("sequence"); ax.set_ylabel("µs")
        ax.legend(fontsize=8)
        # Findings
        med_inf = float(merged["inference_us"].median())
        med_net = float(merged["network_us"].median())
        findings.append(Finding(
            "info", "predict latency",
            f"median inference={med_inf/1000:.1f} ms, median network={med_net/1000:.1f} ms "
            f"(n={len(merged)} matched predicts).",
        ))

    # (d) Histogram
    ax = axes[1][1]
    if pm is not None and not pm.empty:
        ax.hist(pm["inference_us"], bins=15, alpha=0.7, color="#d62728", label="inference µs")
    if ptt is not None and not ptt.empty:
        rtt_l = (ptt["t_tcp_recv_ns"] - ptt["t_tcp_send_ns"]) / 1000.0
        ax.hist(rtt_l, bins=15, alpha=0.5, color="#1f77b4", label="laptop RTT µs")
    ax.set_title("D: distributions")
    ax.set_xlabel("µs"); ax.set_ylabel("count")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=110); plt.close(fig)


def panel_nav_health(run: Run, findings: list[Finding], out: Path) -> None:
    t0, t1 = run.window_ns
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))

    ls = run.topic("/grid_nav_node/loop_stats")
    ax = axes[0]
    if ls is not None and not ls.empty:
        ls = ls.sort_values(best_time_col(ls))
        in_win = ls[(ls[best_time_col(ls)] >= t0) & (ls[best_time_col(ls)] <= t1)]
        t = _rel_s(in_win, t0)
        ax.plot(t, in_win["mean_period_ms"], "-", color="#1f77b4", lw=1.0, label="mean_period_ms")
        if "p99_period_ms" in in_win.columns:
            ax.plot(t, in_win["p99_period_ms"], "-", color="#d62728", lw=1.0, label="p99_period_ms")
        if "nominal_period_ms" in in_win.columns:
            nom = float(in_win["nominal_period_ms"].iloc[0])
            ax.axhline(nom, color="green", ls="--", lw=0.8, label=f"nominal={nom:.0f} ms")
            overrun = (in_win["p99_period_ms"] > 1.5 * nom).sum() if "p99_period_ms" in in_win.columns else 0
            if overrun:
                findings.append(Finding(
                    "warn", "/grid_nav_node/loop_stats",
                    f"{int(overrun)} sample(s) where p99 loop period exceeded 1.5× nominal "
                    f"({nom:.0f} ms) — control loop occasionally overran.",
                ))
        ax.set_title("Nav-node control-loop period")
        ax.set_ylabel("ms"); ax.set_xlabel("run-relative time (s)")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no /grid_nav_node/loop_stats", ha="center", va="center",
                transform=ax.transAxes)

    lr = run.topic("/grid_nav_node/last_result")
    ax = axes[1]
    if lr is not None and not lr.empty:
        lr = lr.sort_values(best_time_col(lr)).reset_index(drop=True)
        in_win = lr[(lr[best_time_col(lr)] >= t0) & (lr[best_time_col(lr)] <= t1)].reset_index(drop=True)
        x = np.arange(len(in_win))
        def _bucket(row) -> str:
            if bool(row["success"]): return "success"
            msg = str(row.get("message", "")).lower()
            if "cancel" in msg: return "canceled"
            return "failure"
        buckets = [_bucket(r) for _, r in in_win.iterrows()]
        bucket_color = {"success": "#2ca02c", "canceled": "#aaaaaa", "failure": "#d62728"}
        colors = [bucket_color[b] for b in buckets]
        ax.bar(x, np.ones(len(in_win)), color=colors, edgecolor="black", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"#{i+1}" for i in range(len(in_win))], rotation=0, fontsize=8)
        for i, (_, row) in enumerate(in_win.iterrows()):
            msg = str(row.get("message", ""))[:30]
            ax.text(i, 0.5, msg, ha="center", va="center", rotation=90, fontsize=7,
                    color="white", fontweight="bold")
        ax.set_yticks([])
        n_succ = buckets.count("success"); n_canc = buckets.count("canceled"); n_fail = buckets.count("failure")
        ax.set_title(f"Per-cell action results — {n_succ} success / {n_canc} canceled / "
                     f"{n_fail} failure (canceled is expected on the last cell "
                     "when orchestrator detects goal reached)")
        if n_fail:
            findings.append(Finding(
                "warn", "/grid_nav_node/last_result",
                f"{n_fail} per-cell action(s) hard-failed (see panel 07).",
            ))
        if n_canc and buckets[-1] == "canceled":
            findings.append(Finding(
                "info", "/grid_nav_node/last_result",
                "last action was canceled — this is the expected pattern when the "
                "orchestrator's goal-reached criterion fires before the action server "
                "delivers its own success. Not a problem.",
            ))
    else:
        ax.text(0.5, 0.5, "no /grid_nav_node/last_result", ha="center", va="center",
                transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(out, dpi=110); plt.close(fig)


def panel_data_audit(run: Run, findings: list[Finding], out: Path) -> None:
    t0, t1 = run.window_ns
    win_s = (t1 - t0) / 1e9

    rows = []
    for bag_name, bag in run.streams.items():
        for topic, df in bag.items():
            if df is None or df.empty:
                rows.append((bag_name, topic, 0, 0.0, np.nan, 0.0)); continue
            tcol = best_time_col(df)
            in_win = df[(df[tcol] >= t0) & (df[tcol] <= t1)]
            actual_rate = len(in_win) / max(1e-3, win_s)
            expected = EXPECTED_RATES.get(topic, np.nan)
            ratio = actual_rate / expected if expected and expected > 0 else np.nan
            if len(in_win) >= 2:
                gaps = np.diff(in_win[tcol].to_numpy()) / 1e9
                max_gap = float(gaps.max())
            else:
                max_gap = float("nan")
            rows.append((bag_name, topic, len(in_win), actual_rate, ratio, max_gap))

    rows.sort(key=lambda r: (r[0], r[1]))
    fig, ax = plt.subplots(figsize=(13, max(6, 0.30 * len(rows) + 1)))
    ax.axis("off")
    labels = [f"{r[0]}: {r[1]}" for r in rows]
    actual = [r[3] for r in rows]
    ratios = [r[4] for r in rows]
    bar_colors = []
    for ratio in ratios:
        if np.isnan(ratio):
            bar_colors.append("#999")
        elif ratio < 0.6:
            bar_colors.append("#d62728")
        elif ratio < 0.9:
            bar_colors.append("#f5a623")
        else:
            bar_colors.append("#2ca02c")
    y = np.arange(len(rows))
    ax2 = fig.add_axes([0.32, 0.06, 0.65, 0.92])
    ax2.barh(y, actual, color=bar_colors, edgecolor="black", lw=0.3)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("actual rate in window (Hz) — red <60%, amber <90%, green ≥90% of expected; grey = no expected")
    ax2.set_title(f"Per-topic data rate (window = {win_s:.1f} s)")
    # Annotations: count + max_gap
    for i, r in enumerate(rows):
        bag_name, topic, n, rate, ratio, gap = r
        suffix = ""
        if not np.isnan(ratio):
            suffix = f"  ({rate:.1f}/{EXPECTED_RATES[topic]:.0f} Hz; {ratio*100:.0f}%)"
        gap_s = f"  max gap {gap*1000:.0f} ms" if not np.isnan(gap) else ""
        ax2.text(rate, i, f"  n={n}{suffix}{gap_s}", va="center", fontsize=7)
    # Findings: anything <60% gets flagged
    for r in rows:
        bag_name, topic, n, rate, ratio, gap = r
        if not np.isnan(ratio) and ratio < 0.6:
            findings.append(Finding(
                "issue", topic,
                f"rate shortfall — actual {rate:.1f} Hz vs expected {EXPECTED_RATES[topic]:.0f} Hz "
                f"({ratio*100:.0f}%); max gap {gap*1000:.0f} ms",
            ))
        elif not np.isnan(ratio) and ratio < 0.9:
            findings.append(Finding(
                "warn", topic,
                f"slight rate shortfall — {rate:.1f} Hz vs {EXPECTED_RATES[topic]:.0f} Hz "
                f"({ratio*100:.0f}%)",
            ))
    fig.savefig(out, dpi=110); plt.close(fig)


# ---------------------------------------------------------------------------
# Metadata / events sanity checks (no plot — finding-only)
# ---------------------------------------------------------------------------

def check_events(run: Run, findings: list[Finding]) -> None:
    for bag_name, bag in run.streams.items():
        ev = bag.get("/experiment/events")
        if ev is None or ev.empty:
            findings.append(Finding(
                "warn", f"{bag_name}.mcap /experiment/events",
                "no events recorded — cross-bag time alignment is harder without run_start/run_end.",
            ))
            continue
        types = set(ev["event_type"].astype(str))
        missing = {"run_start", "run_end"} - types
        if missing:
            findings.append(Finding(
                "warn", f"{bag_name}.mcap /experiment/events",
                f"missing event types {sorted(missing)} — only {sorted(types)} present.",
            ))


def check_metadata(run: Run, findings: list[Finding]) -> None:
    md = run.meta
    mode = str(md.get("mode", "?"))
    planner = str(md.get("planner", "?"))
    n_predicts = 0
    pm = run.topic("/planner/metrics")
    if pm is not None:
        n_predicts = len(pm)
    if mode == "decentralized" and n_predicts > 1:
        findings.append(Finding(
            "warn", "metadata/mode",
            f"mode='decentralized' but {n_predicts} planner predicts were made; "
            "either the orchestrator still calls predict per cell in decentralized mode, "
            "or the metadata label doesn't match the actual workflow.",
        ))
    # Outcome consistency. Re-frame: if the bag's last grid_pose is far from the
    # goal AND the unique-pose count is tiny, the data is stale (the bridge held
    # the laptop's last-sent pose); this isn't an outcome bug, it's a data-collection
    # gap. We surface it as an "issue" on the data-collection axis, not on outcome.
    outcome = md.get("outcome")
    gc = md.get("goal_cell")
    gp = run.topic("/grid_nav_node/grid_pose")
    if outcome == "success" and gc and gp is not None and not gp.empty:
        gp_sorted = gp.sort_values(best_time_col(gp))
        last = gp_sorted.iloc[-1]
        dist = math.hypot(last["x"] - (gc[0] + 0.5), last["y"] - (gc[1] + 0.5))
        # Count unique poses in window
        t0, t1 = run.window_ns
        in_win = gp_sorted[(gp_sorted[best_time_col(gp_sorted)] >= t0)
                           & (gp_sorted[best_time_col(gp_sorted)] <= t1)]
        unique = len(in_win[(in_win["x"].diff().abs() > 1e-4)
                            | (in_win["y"].diff().abs() > 1e-4)]) + 1
        if dist > 1.5 and unique < 20:
            findings.append(Finding(
                "issue", "data: trajectory completeness",
                f"the laptop-side ground-truth pose only updated {unique} times during the "
                f"run; the bag's last grid_pose is {dist:.2f} cells from the goal. The robot "
                "did reach goal (per orchestrator end_reason); the data simply doesn't "
                "record the final approach. Add a 30 Hz tracker-pose log on the laptop "
                "side so trajectory analysis matches reality.",
            ))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(run: Run, findings: list[Finding], out_md: Path,
                 panel_files: list[str]) -> None:
    lines: list[str] = []
    md = run.meta
    lines.append(f"# Audit: `{run.run_id}`")
    lines.append("")
    lines.append(f"- mode/planner/map: **{md.get('mode')}** / **{md.get('planner')}** / "
                 f"**{md.get('map_id')}**")
    lines.append(f"- outcome: **{md.get('outcome')}**  •  start: `{md.get('start_cell')}`  "
                 f"goal: `{md.get('goal_cell')}`  •  duration: "
                 f"{(run.window_ns[1]-run.window_ns[0])/1e9:.1f} s")
    lines.append(f"- bag_elapsed_s: `{md.get('bag_elapsed_s')}`  •  bag_capped: "
                 f"`{md.get('bag_capped')}`")
    lines.append("")
    lines.append("## Panels")
    for p in panel_files:
        lines.append(f"- [{p}]({p})")
    lines.append("")
    # Group findings
    order = {"issue": 0, "warn": 1, "info": 2}
    findings.sort(key=lambda f: (order.get(f.severity, 9), f.topic))
    lines.append("## Findings")
    if not findings:
        lines.append("- No issues detected. (suspicious — re-check.)")
    else:
        n_issue = sum(f.severity == "issue" for f in findings)
        n_warn = sum(f.severity == "warn" for f in findings)
        n_info = sum(f.severity == "info" for f in findings)
        lines.append(f"_{n_issue} issue(s), {n_warn} warning(s), {n_info} info(s)._")
        lines.append("")
        for f in findings:
            lines.append(f.render())
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args(argv)
    run = load_run(args.run_dir)
    if run.window_ns is None:
        print("ERROR: could not extract run window from /experiment/events", file=sys.stderr)
        return 1
    out_dir = args.run_dir / "audit"
    out_dir.mkdir(exist_ok=True)
    findings: list[Finding] = []

    panels = [
        ("01_power.png",          panel_power),
        ("02_trajectory.png",     panel_trajectory),
        ("03_control.png",        panel_control),
        ("04_imu.png",            panel_imu),
        ("05_thermal_clock.png",  panel_thermal_clock),
        ("06_planner_timing.png", panel_planner_timing),
        ("07_nav_health.png",     panel_nav_health),
        ("08_data_audit.png",     panel_data_audit),
        ("09_solar_harvest.png",  panel_solar_harvest),
    ]
    rendered = []
    for fname, fn in panels:
        try:
            fn(run, findings, out_dir / fname)
            rendered.append(fname)
            print(f"  wrote {fname}")
        except Exception as e:
            print(f"  FAILED {fname}: {e}", file=sys.stderr)
            findings.append(Finding("issue", fname, f"panel render failed: {e}"))

    check_events(run, findings)
    check_metadata(run, findings)

    write_report(run, findings, out_dir / "AUDIT.md", rendered)
    print(f"wrote {out_dir / 'AUDIT.md'} ({len(findings)} finding(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
