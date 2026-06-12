"""metrics.py - Per-run derivations from a loaded Run.

Pure functions over the DataFrames in run_io.Run. Produces the `summary.json`
row dict and a list of warnings. Reused by summarize_run (per run) and
analyze_session (which prefers reading the cached summary.json).

All derivations follow handover_analysis §6:
  - energy: trapezoidal, overflow excluded, gaps dropped
  - latency: single-host deltas joined by `sequence` (no cross-host subtraction)
  - run window: bridge-host-stamped run_start/run_end (run_io provides it)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from run_io import Run, best_time_col

SENSORS = ("solar", "sbc", "opencr")
GAP_MS = 50.0           # >5x nominal at 100 Hz -> drop interval
CHRONY_WARN_MS = 5.0
CHRONY_REJECT_MS = 20.0


# -- helpers ------------------------------------------------------------------

def _percentile(series: pd.Series, q: float) -> float | None:
    s = series.dropna()
    return float(np.percentile(s, q)) if len(s) else None


def _energy_mwh(df: pd.DataFrame, t0: int, t1: int, warnings: list[str],
                label: str) -> tuple[float | None, int]:
    """Trapezoidal mWh over [t0,t1]. Returns (energy, n_overflow_excluded).

    Integrates over the full stream so overflow and genuine gaps are
    disambiguated: an interval is skipped if either endpoint is overflow
    (saturated, untrusted — counted as overflow), or if there is a real gap
    (dt > 50 ms, i.e. >=5 dropped samples at 100 Hz, or a sequence skip not
    explained by the samples present). Only genuine gaps raise a warning.
    """
    if df is None or df.empty:
        return None, 0
    col = best_time_col(df)
    d = df[(df[col] >= t0) & (df[col] <= t1)].sort_values(col).reset_index(drop=True)
    n_overflow = int(d["overflow"].sum()) if "overflow" in d.columns else 0
    if len(d) < 2:
        return (0.0 if len(d) else None), n_overflow
    t_s = d[col].to_numpy() / 1e9
    p = d["power_mw"].to_numpy()
    ovf = d["overflow"].astype(bool).to_numpy() if "overflow" in d.columns else np.zeros(len(d), bool)
    seq = d["sequence"].to_numpy() if "sequence" in d.columns else None
    energy = 0.0
    n_gap = 0
    for i in range(len(d) - 1):
        if ovf[i] or ovf[i + 1]:
            continue  # saturated sample; excluded (already counted in n_overflow)
        dt = t_s[i + 1] - t_s[i]
        if dt <= 0 or dt > GAP_MS / 1000.0:
            n_gap += 1
            continue
        if seq is not None and (seq[i + 1] - seq[i]) > 1:
            n_gap += 1
            continue
        energy += 0.5 * (p[i] + p[i + 1]) * dt / 3600.0
    if n_gap:
        warnings.append(f"{label}: {n_gap} genuine sample-gap interval(s) excluded from energy")
    return energy, n_overflow


def _window_sequences(run: Run) -> set[int]:
    pm = run.topic("/planner/metrics")
    if pm is None or pm.empty or "sequence" not in pm.columns:
        return set()
    return set(int(s) for s in pm.loc[run.window_mask(pm), "sequence"])


# -- main entry ---------------------------------------------------------------

def compute_summary(run: Run) -> tuple[dict[str, Any], list[str]]:
    w: list[str] = []
    meta = run.meta
    is_baseline = meta.get("goal_cell") is None
    row: dict[str, Any] = {
        "run_id": run.run_id,
        "session_id": meta.get("session_id"),
        "mode": meta.get("mode"),
        "planner": meta.get("planner"),
        "map_id": meta.get("map_id"),
        # goal_cell is part of the paired-comparison key (map_id, goal_cell).
        # Stored as a string so it is hashable/groupable in pandas.
        "goal_cell": (None if meta.get("goal_cell") is None
                      else "_".join(str(c) for c in meta["goal_cell"])),
        "outcome": meta.get("outcome"),
        "dry_run": bool(meta.get("dry_run", False)),
        "bag_capped": bool(meta.get("bag_capped", False)),
        "is_baseline": is_baseline,
    }

    if run.window_ns is None:
        w.append("no run_start/run_end events found; cannot bound run window")
        return row, w
    t0, t1 = run.window_ns
    row["time_to_completion_s"] = round((t1 - t0) / 1e9, 3)

    # --- goal reached (ground truth: last_result success in window) ----------
    lr = run.topic("/grid_nav_node/last_result")
    goal_reached = False
    if lr is not None and not lr.empty:
        inwin = lr[run.window_mask(lr)]
        goal_reached = bool(inwin["success"].astype(bool).any()) if "success" in inwin else False
    row["goal_reached"] = goal_reached
    if (meta.get("outcome") == "success" and not goal_reached
            and not is_baseline and not meta.get("dry_run")):
        w.append("metadata outcome=success but no last_result success in window")

    # --- energy --------------------------------------------------------------
    e = {}
    for s in SENSORS:
        df = run.topic(f"/power/{s}")
        e[s], n_ovf = _energy_mwh(df, t0, t1, w, f"/power/{s}")
        row[f"{s}_overflow_samples"] = n_ovf
        if df is not None and len(df) and n_ovf / max(len(df), 1) > 0.01:
            w.append(f"/power/{s}: >1% overflow samples ({n_ovf}/{len(df)})")
    row["energy_harvested_mwh"] = None if e["solar"] is None else round(e["solar"], 4)
    row["energy_consumed_sbc_mwh"] = None if e["sbc"] is None else round(e["sbc"], 4)
    row["energy_consumed_opencr_mwh"] = None if e["opencr"] is None else round(e["opencr"], 4)
    if None not in (e["solar"], e["sbc"], e["opencr"]):
        row["energy_net_mwh"] = round(e["solar"] - (e["sbc"] + e["opencr"]), 4)
    else:
        row["energy_net_mwh"] = None

    # battery voltage from /power/sbc bus voltage at window edges
    sbc = run.topic("/power/sbc")
    if sbc is not None and not sbc.empty:
        col = best_time_col(sbc)
        d = sbc[(sbc[col] >= t0) & (sbc[col] <= t1)].sort_values(col)
        if len(d):
            row["battery_voltage_start_v"] = round(float(d.iloc[0]["bus_voltage_v"]), 3)
            row["battery_voltage_end_v"] = round(float(d.iloc[-1]["bus_voltage_v"]), 3)
    if not any(run.topic(f"/power/{s}") is not None and not run.topic(f"/power/{s}").empty
               for s in SENSORS):
        w.append("empty power bag (no /power/* samples)")

    # --- path length ---------------------------------------------------------
    if not is_baseline:
        row.update(_path_lengths(run, t0, t1))

    # --- planner inference stats (in-window only; warmup excluded by window) --
    pm = run.topic("/planner/metrics")
    if pm is not None and not pm.empty:
        inwin = pm[run.window_mask(pm)]
        if len(inwin):
            row["planner_inference_mean_us"] = int(inwin["inference_us"].mean())
            row["planner_inference_p99_us"] = int(_percentile(inwin["inference_us"], 99))
            row["planner_inference_n_calls"] = int(len(inwin))
    row["warmup_convergence_ok"] = _warmup_ok(meta.get("warmup_inferences_us"), w)

    # --- latency decomposition (single-host deltas, sequence join) -----------
    row.update(_latency_decomposition(run, w))

    # --- control loop --------------------------------------------------------
    cls = run.topic("/grid_nav_node/loop_stats")
    if cls is not None and not cls.empty:
        inwin = cls[run.window_mask(cls)]
        if len(inwin):
            row["control_loop_overruns"] = int(inwin["overruns"].sum())
            row["control_loop_mean_period_ms"] = round(float(inwin["mean_period_ms"].mean()), 2)
            row["control_loop_p99_period_ms"] = round(float(inwin["p99_period_ms"].max()), 2)
            if row["control_loop_overruns"] > 0:
                w.append(f"control loop overruns: {row['control_loop_overruns']}")

    # --- thermal -------------------------------------------------------------
    th = run.topic("/sbc/thermal")
    if th is not None and not th.empty:
        inwin = th[run.window_mask(th)]
        if len(inwin):
            row["sbc_thermal_max_c"] = round(float(inwin["cpu_temp_c"].max()), 1)
            row["sbc_throttled"] = bool(inwin["throttled"].astype(bool).any())
            if row["sbc_throttled"]:
                w.append("sbc throttled during run")

    # --- chrony + wifi (from /diagnostics/host in window) --------------------
    row.update(_diagnostics(run, t0, t1, w))

    # --- desktop RAPL energy (centralized only) ------------------------------
    row["energy_consumed_desktop_mwh"] = _rapl_energy(run, t0, t1, w)

    return row, w


def _path_lengths(run: Run, t0: int, t1: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    # meters: sum |Δposition| over /odom in window
    od = run.topic("/odom")
    if od is not None and not od.empty and "pose.pose.position.x" in od.columns:
        col = best_time_col(od)
        d = od[(od[col] >= t0) & (od[col] <= t1)].sort_values(col)
        if len(d) >= 2:
            x = d["pose.pose.position.x"].to_numpy()
            y = d["pose.pose.position.y"].to_numpy()
            out["executed_path_length_m"] = round(
                float(np.hypot(np.diff(x), np.diff(y)).sum()), 4)
    # cells: count of planner calls (one action per cell transition) in window
    seqs = _window_sequences(run)
    out["executed_path_length_cells"] = len(seqs)
    # octile: sum of step weights from predict_log directions for in-window seqs
    pl = run.predict_log
    if pl is not None and "event" in pl.columns and seqs:
        pr = pl[(pl["event"] == "predict_result") & (pl.get("sequence").isin(seqs))]
        octile = 0.0
        for _, r in pr.iterrows():
            dx, dy = r.get("direction_x"), r.get("direction_y")
            if pd.notna(dx) and pd.notna(dy):
                octile += math.sqrt(2) if (dx != 0 and dy != 0) else 1.0
        out["executed_path_length_octile"] = round(octile, 4)
    return out


def _warmup_ok(warmups: Any, w: list[str]) -> bool | None:
    if not warmups or len(warmups) < 3:
        return None
    last3 = warmups[-3:]
    mean3 = sum(last3) / 3.0
    ok = abs(warmups[-1] - mean3) <= 0.1 * mean3 if mean3 else False
    if not ok:
        w.append("warmup did not converge (last latency >10% off mean of last 3)")
    return ok


def _latency_decomposition(run: Run, w: list[str]) -> dict[str, Any]:
    pm = run.topic("/planner/metrics")
    bt = run.topic("/bridge/predict_timing")
    ptt = run.topic("/predict_tcp_timing")
    if any(x is None or x.empty for x in (pm, bt, ptt)):
        return {}
    seqs = _window_sequences(run)
    if not seqs:
        return {}
    pm = pm[pm["sequence"].isin(seqs)][["sequence", "inference_us"]]
    bt = bt[bt["sequence"].isin(seqs)][["sequence", "t_bridge_recv_ns", "t_bridge_send_ns"]]
    ptt = ptt[ptt["sequence"].isin(seqs)][["sequence", "t_tcp_send_ns", "t_tcp_recv_ns"]]
    m = pm.merge(bt, on="sequence").merge(ptt, on="sequence")
    if m.empty:
        w.append("latency: sequence join produced no rows")
        return {}
    delta_bridge = (m["t_bridge_send_ns"] - m["t_bridge_recv_ns"]) / 1000.0
    delta_orch = (m["t_tcp_recv_ns"] - m["t_tcp_send_ns"]) / 1000.0
    bridge_plumbing = delta_bridge - m["inference_us"]
    tcp = delta_orch - delta_bridge
    return {
        "tcp_rtt_mean_us": int(delta_orch.mean()),
        "tcp_rtt_p99_us": int(np.percentile(delta_orch, 99)),
        "bridge_plumbing_mean_us": int(bridge_plumbing.mean()),
        "tcp_only_mean_us": int(tcp.mean()),
    }


def _diagnostics(run: Run, t0: int, t1: int, w: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    offsets, rssi = [], None
    for bag in run.streams.values():
        hd = bag.get("/diagnostics/host")
        if hd is None or hd.empty:
            continue
        col = best_time_col(hd)
        d = hd[(hd[col] >= t0) & (hd[col] <= t1)]
        if "chrony_offset_ms" in d.columns and len(d):
            offsets.extend(d["chrony_offset_ms"].abs().tolist())
        # WiFi rssi: the Pi (nonzero) value
        if "wifi_rssi_dbm" in d.columns:
            nz = d[d["wifi_rssi_dbm"] != 0]["wifi_rssi_dbm"]
            if len(nz):
                rssi = int(nz.iloc[0])
    if offsets:
        mx = max(offsets)
        out["chrony_offset_max_ms"] = round(mx, 3)
        if mx > CHRONY_REJECT_MS:
            w.append(f"chrony offset {mx:.1f} ms > {CHRONY_REJECT_MS} ms (run should be rejected)")
        elif mx > CHRONY_WARN_MS:
            w.append(f"chrony offset {mx:.1f} ms > {CHRONY_WARN_MS} ms")
    if rssi is not None:
        out["wifi_rssi_dbm"] = rssi
    return out


def _rapl_energy(run: Run, t0: int, t1: int, w: list[str]) -> float | None:
    desktop = run.streams.get("desktop")
    if not desktop or "/desktop/rapl_energy_uj" not in desktop:
        if run.meta.get("mode") == "centralized":
            w.append("centralized run missing desktop RAPL stream")
        return None
    df = desktop["/desktop/rapl_energy_uj"]
    col = best_time_col(df)
    d = df[(df[col] >= t0) & (df[col] <= t1)].sort_values(col)
    if "passthrough_ok" in d.columns and not d["passthrough_ok"].astype(bool).all():
        w.append("RAPL passthrough not ok for entire run; desktop energy excluded")
        return None
    if len(d) < 2:
        return None
    first, last = int(d.iloc[0]["energy_pkg_uj"]), int(d.iloc[-1]["energy_pkg_uj"])
    if last < first:  # counter wrap (uint32 range of the uj counter)
        last += 2 ** 32
    return round((last - first) / 1e6 / 3.6, 4)
