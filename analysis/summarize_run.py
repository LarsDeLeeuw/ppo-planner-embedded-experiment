"""summarize_run.py - Per-run sanity check: summary.json + 4-panel summary.png.

    python summarize_run.py runs/<run_id>

Reads everything in the run folder (read-only), writes summary.json and
summary.png into it. Panels degrade gracefully: baselines (goal_cell=null) and
dry runs render only the power panel; missing streams leave their panel empty
with a note. See handover_analysis §4.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import compute_summary  # noqa: E402
from run_io import Run, best_time_col, load_run  # noqa: E402

SENSOR_COLORS = {"solar": "#f5a623", "sbc": "#4a90d9", "opencr": "#d0021b", "net": "#417505"}


def _rel_s(df: pd.DataFrame, t0: int) -> np.ndarray:
    return (df[best_time_col(df)].to_numpy() - t0) / 1e9


def _panel_power(ax, run: Run) -> None:
    t0, t1 = run.window_ns if run.window_ns else (None, None)
    base = run.topic("/power/sbc")
    if base is None or base.empty or t0 is None:
        ax.text(0.5, 0.5, "no power data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("A: power")
        return
    series = {}
    for s in ("solar", "sbc", "opencr"):
        df = run.topic(f"/power/{s}")
        if df is None or df.empty:
            continue
        df = df.sort_values(best_time_col(df))
        ax.plot(_rel_s(df, t0), df["power_mw"].to_numpy(), label=s,
                color=SENSOR_COLORS[s], lw=0.8)
        series[s] = df
    # net = solar - (sbc + opencr), aligned onto sbc timestamps
    if {"solar", "sbc", "opencr"} <= series.keys():
        tcol = best_time_col(series["sbc"])
        m = series["sbc"][[tcol, "power_mw"]].rename(columns={"power_mw": "sbc"})
        for s in ("solar", "opencr"):
            o = series[s][[best_time_col(series[s]), "power_mw"]].rename(
                columns={best_time_col(series[s]): tcol, "power_mw": s}).sort_values(tcol)
            m = pd.merge_asof(m.sort_values(tcol), o, on=tcol,
                              tolerance=10_000_000, direction="nearest")
        net = m["solar"] - (m["sbc"] + m["opencr"])
        ax.plot((m[tcol].to_numpy() - t0) / 1e9, net.to_numpy(), label="net",
                color=SENSOR_COLORS["net"], lw=1.0, ls="--")
    ax.axvline(0, color="k", lw=0.8, alpha=0.6)
    ax.axvline((t1 - t0) / 1e9, color="k", lw=0.8, alpha=0.6)
    # markers
    for bag in run.streams.values():
        ev = bag.get("/experiment/events")
        if ev is None:
            continue
        for _, r in ev.iterrows():
            if r.get("event_type") in ("marker", "bag_capped"):
                ax.axvline((r[best_time_col(ev)] - t0) / 1e9, color="purple", lw=0.6, alpha=0.5)
        break
    ax.set_xlabel("run-relative time (s)")
    ax.set_ylabel("power (mW)")
    ax.set_title("A: power traces")
    ax.legend(fontsize=7, loc="upper right")


def _panel_trajectory(ax, run: Run) -> None:
    ax.set_title("B: trajectory")
    obs = run.obstacle_map
    if obs is not None:
        ax.imshow(obs, origin="lower", cmap="Greys", alpha=0.55,
                  extent=[0, obs.shape[1], 0, obs.shape[0]], zorder=1)
    # energy overlay: first in-run energy map from JSONL (dynamic; representative)
    pl = run.predict_log
    if pl is not None and "event" in pl.columns:
        ps = pl[(pl["event"] == "predict_sent") & pl["energy_map"].notna()] \
            if "energy_map" in pl.columns else pd.DataFrame()
        if len(ps):
            em = np.array(ps.iloc[0]["energy_map"], dtype=float)  # [x][y] bridge frame
            ax.imshow(em.T, origin="lower", cmap="viridis", alpha=0.35,
                      extent=[0, em.shape[0], 0, em.shape[1]], zorder=0)
    gp = run.topic("/grid_nav_node/grid_pose")
    if gp is not None and not gp.empty:
        gp = gp.sort_values(best_time_col(gp))
        ax.plot(gp["x"], gp["y"], "-o", color="#1f77b4", ms=2, lw=1.2, zorder=3, label="path")
    sc = run.meta.get("start_cell")
    gc = run.meta.get("goal_cell")
    if sc:
        ax.plot(sc[0] + 0.5, sc[1] + 0.5, "s", color="green", ms=10, zorder=4, label="start")
    if gc:
        reached = "✓" if run.meta.get("outcome") == "success" else "✗"
        ax.plot(gc[0] + 0.5, gc[1] + 0.5, "*", color="magenta", ms=16, zorder=4,
                label=f"goal {reached}")
    ax.set_xlabel("x (col, +E)")
    ax.set_ylabel("y (row, +N)")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper left")


def _panel_timing(ax, run: Run) -> None:
    ax.set_title("C: planner timing")
    pm = run.topic("/planner/metrics")
    if pm is None or pm.empty:
        ax.text(0.5, 0.5, "no planner metrics", ha="center", va="center", transform=ax.transAxes)
        return
    pm = pm.sort_values("sequence")
    ax.scatter(pm["sequence"], pm["inference_us"], s=14, color="#d62728", label="inference µs")
    ptt = run.topic("/predict_tcp_timing")
    if ptt is not None and not ptt.empty:
        rtt = (ptt["t_tcp_recv_ns"] - ptt["t_tcp_send_ns"]) / 1000.0
        ax.scatter(ptt["sequence"], rtt, s=14, color="#1f77b4", marker="^", label="tcp rtt µs")
    # shade warmup region: sequences before the first in-window sequence
    inwin = pm[run.window_mask(pm)] if run.window_ns else pm
    if len(inwin) and len(inwin) < len(pm):
        first_run_seq = int(inwin["sequence"].min())
        ax.axvspan(pm["sequence"].min() - 0.5, first_run_seq - 0.5,
                   color="gray", alpha=0.2, label="warmup")
    ax.set_yscale("log")
    ax.set_xlabel("predict sequence")
    ax.set_ylabel("µs (log)")
    ax.legend(fontsize=7, loc="upper right")


def _panel_phases(ax, run: Run) -> None:
    ax.set_title("D: phase swimlane")
    fb = run.topic("/move_to_grid/_action/feedback")
    if fb is None or fb.empty or "feedback.phase" not in fb.columns:
        ax.text(0.5, 0.5, "no action feedback recorded\n(phase swimlane unavailable)",
                ha="center", va="center", transform=ax.transAxes, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    t0 = run.window_ns[0]
    fb = fb.sort_values(best_time_col(fb))
    phases = fb["feedback.phase"].tolist()
    times = _rel_s(fb, t0)
    lanes = {p: i for i, p in enumerate(sorted(set(phases)))}
    for i in range(len(fb) - 1):
        ax.barh(lanes[phases[i]], times[i + 1] - times[i], left=times[i], height=0.6)
    ax.set_yticks(list(lanes.values()))
    ax.set_yticklabels(list(lanes.keys()), fontsize=7)
    ax.set_xlabel("run-relative time (s)")


def render_summary_png(run: Run, summary: dict, warnings: list[str], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    _panel_power(axes[0][0], run)
    is_minimal = run.meta.get("goal_cell") is None or run.meta.get("dry_run")
    if is_minimal:
        for ax in (axes[0][1], axes[1][0], axes[1][1]):
            ax.text(0.5, 0.5, "baseline / dry-run\n(panel skipped)", ha="center",
                    va="center", transform=ax.transAxes, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    else:
        _panel_trajectory(axes[0][1], run)
        _panel_timing(axes[1][0], run)
        _panel_phases(axes[1][1], run)
    sub = (f"{summary.get('run_id')}  |  {summary.get('mode')}/{summary.get('planner')}"
           f"/{summary.get('map_id')}  |  outcome={summary.get('outcome')}  |  "
           f"chrony_max={summary.get('chrony_offset_max_ms', 'n/a')} ms")
    if warnings:
        sub += f"  |  ⚠ {len(warnings)} warning(s)"
    fig.suptitle(sub, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def summarize(run_dir: Path) -> dict:
    run = load_run(run_dir)
    summary, warnings = compute_summary(run)
    summary["warnings"] = warnings
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    render_summary_png(run, summary, warnings, run_dir / "summary.png")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args(argv)
    summary = summarize(args.run_dir)
    for wmsg in summary.get("warnings", []):
        print(f"  warning: {wmsg}", file=sys.stderr)
    print(f"wrote {args.run_dir/'summary.json'} and summary.png")
    print(f"  outcome={summary.get('outcome')} goal_reached={summary.get('goal_reached')} "
          f"path_m={summary.get('executed_path_length_m')} "
          f"t={summary.get('time_to_completion_s')}s "
          f"net_energy={summary.get('energy_net_mwh')} mWh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
