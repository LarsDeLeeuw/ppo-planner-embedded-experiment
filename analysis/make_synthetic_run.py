"""make_synthetic_run.py - Fabricate run folders for testing the analysis pipeline.

No hardware and no real experiment have run yet, so this generator emits fully
self-consistent `runs/<run_id>/` folders (robot.mcap, desktop.mcap when
centralized, orchestrator.mcap, predict_log.jsonl, metadata.yaml,
maps/obstacle.npy) that exercise every code path in summarize_run /
analyze_session / analyze_campaign.

It is ALSO the validation harness for tracker's OrchestratorBag writer: the
orchestrator.mcap here is produced by the very same rosbags path the live
orchestrator uses.

The physics are deliberately simple but internally consistent:
  - sequence ids span warmup (before run_start) + run steps (inside the window)
  - PlannerMetrics / BridgePredictTiming / PredictTcpTiming share each sequence,
    with nested single-host deltas (inference < bridge < tcp)
  - power is integrable; opencr spikes during motion; a couple of overflow
    samples test exclusion
  - the run window is bounded by run_start/run_end on /experiment/events

CLI:
  python make_synthetic_run.py run     OUTDIR [--mode ...] [--planner ...]
  python make_synthetic_run.py session OUTDIR            # one (mode,planner) cell, K runs
  python make_synthetic_run.py campaign OUTDIR           # all 4 cells x K runs
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tb3_typestore import get_tb3_typestore  # noqa: E402

CELL_M = 0.30                       # grid cell size in metres
GRID = 10                           # 10x10 grid (matches PPO model)
NOMINAL_PERIOD_MS = 50.0            # 20 Hz control loop
BASE_EPOCH = datetime(2026, 5, 27, 15, 0, 0, tzinfo=timezone.utc)


# -- bag writer ---------------------------------------------------------------

class SynthBag:
    """rosbags MCAP writer that emits a flat <name>.mcap (inner bag relocated)."""

    def __init__(self, mcap_path: Path, ts):
        from rosbags.rosbag2 import StoragePlugin, Writer
        self._path = Path(mcap_path)
        self._ts = ts
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp = Path(tempfile.mkdtemp(prefix=self._path.stem + "_",
                                          dir=self._path.parent)) / "bag"
        self._w = Writer(self._tmp, version=9, storage_plugin=StoragePlugin.MCAP)
        self._w.open()
        self._conns: dict[str, object] = {}

    def _conn(self, topic: str, typename: str):
        if topic not in self._conns:
            self._conns[topic] = self._w.add_connection(topic, typename, typestore=self._ts)
        return self._conns[topic]

    def write(self, topic: str, typename: str, msg, stamp_ns: int):
        self._w.write(self._conn(topic, typename), int(stamp_ns),
                      self._ts.serialize_cdr(msg, typename))

    def close(self):
        self._w.close()
        inner = next(self._tmp.glob("*.mcap"))
        if self._path.exists():
            self._path.unlink()
        shutil.move(str(inner), str(self._path))
        shutil.rmtree(self._tmp.parent, ignore_errors=True)


# -- message helpers ----------------------------------------------------------

class Msg:
    """Thin factory bound to a typestore for building tb3/common messages."""

    def __init__(self, ts):
        self.ts = ts
        self.T = ts.types

    def header(self, ns: int, frame: str = ""):
        Header = self.T["std_msgs/msg/Header"]
        Time = self.T["builtin_interfaces/msg/Time"]
        return Header(stamp=Time(sec=int(ns // 1_000_000_000),
                                 nanosec=int(ns % 1_000_000_000)), frame_id=frame)

    def power(self, ns, sensor_id, i2c, bus_v, current_ma, overflow=False, seq=0):
        T = self.T
        return T["tb3_interfaces/msg/PowerSample"](
            header=self.header(ns, sensor_id), sensor_id=sensor_id, i2c_address=i2c,
            shunt_resistance_ohm=0.1, bus_voltage_v=float(bus_v),
            shunt_voltage_mv=float(current_ma * 0.1), current_ma=float(current_ma),
            power_mw=float(bus_v * current_ma), overflow=bool(overflow),
            sequence=int(seq) & 0xFFFFFFFF)

    def thermal(self, ns, temp_c, throttled=False, flags=0):
        return self.T["tb3_interfaces/msg/ThermalStat"](
            header=self.header(ns, "pi"), cpu_temp_c=float(temp_c),
            throttled=bool(throttled), throttle_flags=int(flags) & 0xFFFFFFFF)

    def grid_pose(self, x, y, heading):
        return self.T["tb3_interfaces/msg/GridPose"](
            x=float(x), y=float(y), heading=float(heading))

    def planner_metrics(self, ns, seq, name, inf_us, success, nodes, path_len):
        return self.T["tb3_interfaces/msg/PlannerMetrics"](
            header=self.header(ns, "planner"), sequence=int(seq) & 0xFFFFFFFF,
            planner_name=name, inference_us=int(inf_us) & 0xFFFFFFFF,
            success=bool(success), nodes_expanded=int(nodes), path_length=int(path_len),
            grid_rows=GRID, grid_cols=GRID)

    def bridge_timing(self, ns, seq, recv_ns, send_ns):
        return self.T["tb3_interfaces/msg/BridgePredictTiming"](
            header=self.header(ns, "bridge"), sequence=int(seq) & 0xFFFFFFFF,
            t_bridge_recv_ns=int(recv_ns), t_bridge_send_ns=int(send_ns))

    def event(self, ns, event_type, run_id, payload):
        return self.T["tb3_interfaces/msg/ExperimentEvent"](
            header=self.header(ns, "bridge"), event_type=event_type, run_id=run_id,
            payload_json=json.dumps(payload, separators=(",", ":")))

    def last_result(self, ns, success, x, y, heading, dist_m, rot_rad, reason):
        return self.T["tb3_interfaces/msg/MoveToGridResult"](
            header=self.header(ns, "nav"), run_id_hint="", success=bool(success),
            message="reached" if success else "failed",
            final_pose=self.grid_pose(x, y, heading),
            total_distance_m=float(dist_m), total_rotation_rad=float(rot_rad),
            termination_reason=reason)

    def loop_stats(self, ns, mean_ms, p99_ms, max_ms, overruns):
        return self.T["tb3_interfaces/msg/ControlLoopStats"](
            header=self.header(ns, "nav"), nominal_period_ms=NOMINAL_PERIOD_MS,
            mean_period_ms=float(mean_ms), p99_period_ms=float(p99_ms),
            max_period_ms=float(max_ms), ticks=100, overruns=int(overruns) & 0xFFFFFFFF)

    def host_diag(self, ns, hostname, offset_ms, rms_ms, rssi, link_q, rapl_ok):
        return self.T["tb3_interfaces/msg/HostDiagnostics"](
            header=self.header(ns, hostname), hostname=hostname,
            chrony_offset_ms=float(offset_ms), chrony_rms_ms=float(rms_ms),
            wifi_rssi_dbm=int(rssi), wifi_link_quality=float(link_q),
            rapl_passthrough_ok=bool(rapl_ok))

    def session_meta(self, ns, hostname, git_sha, exp_yaml):
        return self.T["tb3_interfaces/msg/SessionMetadata"](
            header=self.header(ns, hostname), hostname=hostname,
            git_sha_research_project=git_sha, experiment_yaml_snapshot=exp_yaml)

    def rapl(self, ns, pkg_uj, ram_uj, ok):
        return self.T["tb3_interfaces/msg/RaplEnergy"](
            header=self.header(ns, "desktop"), energy_pkg_uj=int(pkg_uj),
            energy_ram_uj=int(ram_uj), passthrough_ok=bool(ok))

    def predict_timing(self, ns, seq, send_ns, recv_ns):
        return self.T["tb3_interfaces/msg/PredictTcpTiming"](
            header=self.header(ns, "laptop"), sequence=int(seq) & 0xFFFFFFFF,
            t_tcp_send_ns=int(send_ns), t_tcp_recv_ns=int(recv_ns))

    def odom(self, ns, x_m, y_m, vx):
        T = self.T
        Point = T["geometry_msgs/msg/Point"]
        Quat = T["geometry_msgs/msg/Quaternion"]
        Pose = T["geometry_msgs/msg/Pose"]
        PoseC = T["geometry_msgs/msg/PoseWithCovariance"]
        Vec3 = T["geometry_msgs/msg/Vector3"]
        Twist = T["geometry_msgs/msg/Twist"]
        TwistC = T["geometry_msgs/msg/TwistWithCovariance"]
        cov = np.zeros(36, dtype=np.float64)
        pose = Pose(position=Point(x=float(x_m), y=float(y_m), z=0.0),
                    orientation=Quat(x=0.0, y=0.0, z=0.0, w=1.0))
        twist = Twist(linear=Vec3(x=float(vx), y=0.0, z=0.0),
                      angular=Vec3(x=0.0, y=0.0, z=0.0))
        return T["nav_msgs/msg/Odometry"](
            header=self.header(ns, "odom"), child_frame_id="base_link",
            pose=PoseC(pose=pose, covariance=cov),
            twist=TwistC(twist=twist, covariance=cov))


# -- scenario -----------------------------------------------------------------

def _greedy_path(start, goal):
    """8-connected greedy cell path from start to goal (col, row)."""
    (cx, cy), (gx, gy) = start, goal
    path = [(cx, cy)]
    while (cx, cy) != (gx, gy):
        cx += (gx > cx) - (gx < cx)
        cy += (gy > cy) - (gy < cy)
        path.append((cx, cy))
    return path


def _ns(off_s: float) -> int:
    return int(BASE_EPOCH.timestamp() * 1e9) + int(off_s * 1e9)


def generate_run(out_dir: Path, *, mode="decentralized", planner="ppo",
                 map_id="map3", start_cell=(0, 0), goal_cell=(9, 9),
                 outcome="success", dry_run=False, seed=0,
                 throttled=False, inject_overflow=True) -> Path:
    # Seed includes the condition so noise differs across cells, and inject
    # deterministic condition effects so paired comparisons are non-degenerate:
    #   - decentralized loads the Pi (it runs inference) -> slower + more power
    #   - ppo draws a touch more SBC power than A*
    rng = random.Random((hash((mode, planner, map_id, seed)) & 0x7FFFFFFF) or 1)
    mode_load = 1.18 if mode == "decentralized" else 1.0
    planner_load = 1.06 if planner == "ppo" else 1.0
    ts = get_tb3_typestore()
    M = Msg(ts)

    is_baseline = goal_cell is None
    path = [] if (is_baseline or dry_run) else _greedy_path(start_cell, goal_cell)
    n_steps = max(0, len(path) - 1)

    kind = "-baseline" if is_baseline else ("-dry" if dry_run else "")
    run_id = (f"{BASE_EPOCH.strftime('%Y%m%d-%H%M%S')}-{planner}-{map_id}-{mode}"
              f"{kind}-{seed:03d}")
    session_id = f"session-{BASE_EPOCH.strftime('%Y%m%d')}-{mode}-{planner}"
    rd = Path(out_dir) / run_id
    if rd.exists():
        shutil.rmtree(rd)
    rd.mkdir(parents=True)

    # Timeline
    n_warm = 5
    warm_t = [0.5 + 0.4 * k for k in range(n_warm)]   # warmup predict times (s)
    t_run_start = 3.0
    step_dt = (1.25 if mode == "decentralized" else 1.1) + rng.uniform(-0.03, 0.03)
    t_run_end = t_run_start + 0.5 + max(n_steps, 1) * step_dt + 0.5
    t_bag_end = t_run_end + 0.5

    # Bridge runs on the host that runs tb3_bringup: Pi in decentralized,
    # desktop in centralized. nav/planner/events go in that host's bag.
    robot = SynthBag(rd / "robot.mcap", ts)
    desktop = SynthBag(rd / "desktop.mcap", ts) if mode == "centralized" else None
    nav_bag = desktop if mode == "centralized" else robot

    # --- always-on robot streams: power (100 Hz), thermal (1 Hz), diag (1 Hz)
    seqp = 0
    t = 0.0
    while t <= t_bag_end:
        ns = _ns(t)
        moving = any(abs(t - (t_run_start + 0.5 + k * step_dt)) < 0.6
                     for k in range(n_steps))
        solar = 40 + rng.uniform(-4, 4)
        sbc = (650 + rng.uniform(-20, 20) + (180 if moving else 0)) * planner_load * mode_load
        opencr = ((3000 if moving else 210) + rng.uniform(-30, 30)) * (mode_load if moving else 1.0)
        # battery rail bus voltage drifts down slightly over the run
        vbus = 12.4 - 0.2 * (t / max(t_bag_end, 1e-9))
        robot.write("/power/solar", "tb3_interfaces/msg/PowerSample",
                    M.power(ns, "solar", 0x40, vbus, solar / vbus, seq=seqp), ns)
        robot.write("/power/sbc", "tb3_interfaces/msg/PowerSample",
                    M.power(ns, "sbc", 0x41, vbus, sbc / vbus, seq=seqp), ns)
        ovf = bool(inject_overflow and moving and rng.random() < 0.01)
        robot.write("/power/opencr", "tb3_interfaces/msg/PowerSample",
                    M.power(ns, "opencr", 0x44, vbus, opencr / vbus, overflow=ovf, seq=seqp), ns)
        seqp += 1
        t += 0.01

    for k in range(int(t_bag_end) + 1):
        ns = _ns(k)
        temp = 48.0 + 7.0 * (k / max(t_bag_end, 1e-9))
        robot.write("/sbc/thermal", "tb3_interfaces/msg/ThermalStat",
                    M.thermal(ns, temp, throttled=(throttled and k > 2)), ns)
        robot.write("/diagnostics/host", "tb3_interfaces/msg/HostDiagnostics",
                    M.host_diag(ns, "robot-host", rng.uniform(0.1, 0.5),
                                0.1, -54, 0.72, rapl_ok=False), ns)
    if mode == "centralized":
        for k in range(int(t_bag_end) + 1):
            ns = _ns(k)
            desktop.write("/diagnostics/host", "tb3_interfaces/msg/HostDiagnostics",
                          M.host_diag(ns, "desktop-host", rng.uniform(0.1, 0.5),
                                      0.1, 0, 0.0, rapl_ok=True), ns)
            # cumulative RAPL counter, ~25 W
            desktop.write("/desktop/rapl_energy_uj", "tb3_interfaces/msg/RaplEnergy",
                          M.rapl(ns, pkg_uj=25_000_000 * k, ram_uj=4_000_000 * k, ok=True), ns)

    # --- latched session metadata (transient-local; one shot near start)
    exp_yaml = f"planner: {planner}\nmode: {mode}\ngrid_size: [{GRID}, {GRID}]\n"
    nav_bag.write("/diagnostics/session", "tb3_interfaces/msg/SessionMetadata",
                  M.session_meta(_ns(0.05), nav_bag is desktop and "desktop-host" or "robot-host",
                                 "a1b2c3d4", exp_yaml), _ns(0.05))

    # --- control loop stats (1 Hz, nav host)
    for k in range(int(t_bag_end) + 1):
        ns = _ns(k)
        overruns = 2 if (mode == "decentralized" and planner == "ppo" and k % 7 == 0) else 0
        nav_bag.write("/grid_nav_node/loop_stats", "tb3_interfaces/msg/ControlLoopStats",
                      M.loop_stats(ns, 49.8 + rng.uniform(-0.3, 0.3), 51.4, 55.0, overruns), ns)

    # --- orchestrator bag (laptop): PredictTcpTiming + ExperimentEvent copy
    orch = SynthBag(rd / "orchestrator.mcap", ts)

    # --- JSONL sidecar
    jsonl = (rd / "predict_log.jsonl").open("w", encoding="utf-8")

    def jline(rec):
        jsonl.write(json.dumps(rec, separators=(",", ":")) + "\n")

    def energy_map(jitter):
        # gradient brighter toward top-right (x,y both high), small per-call noise
        g = [[round(min(1.0, (x + y) / (2 * (GRID - 1)) + rng.uniform(-0.02, 0.02) * jitter), 4)
              for y in range(GRID)] for x in range(GRID)]
        return g

    obstacle_bridge = [[0 for _ in range(GRID)] for _ in range(GRID)]  # [x][y]
    for (ox, oy) in [(3, 6), (6, 3), (7, 7)]:
        if (ox, oy) != tuple(goal_cell or (-1, -1)):
            obstacle_bridge[ox][oy] = 1

    # nested single-host clocks (offsets arbitrary; only deltas matter)
    bridge_mono0 = 5_000_000_000
    laptop_mono0 = 9_000_000_000

    def emit_call(seq, t_call_s, inside_run, inf_override=None):
        """Emit PlannerMetrics + BridgePredictTiming + PredictTcpTiming for one predict."""
        ns = _ns(t_call_s)
        if planner == "ppo":
            inf_us = int(rng.uniform(1100, 2200))
            nodes, plen = -1, -1
        else:
            inf_us = int(rng.uniform(400, 1500))
            nodes, plen = int(rng.uniform(20, 120)), n_steps
        if inf_override is not None:
            inf_us = int(inf_override)
        bridge_plumb = int(rng.uniform(250, 450))
        tcp = int(rng.uniform(15000, 22000))  # incl. ~poll floor
        b_recv = bridge_mono0 + int(t_call_s * 1e9)
        b_send = b_recv + inf_us * 1000 + bridge_plumb * 1000
        l_send = laptop_mono0 + int(t_call_s * 1e9)
        l_recv = l_send + (b_send - b_recv) + tcp * 1000
        pname = {"ppo": "ppo", "astar_energy": "astar_energy",
                 "astar_shortest": "astar_shortest"}[planner]
        nav_bag.write("/planner/metrics", "tb3_interfaces/msg/PlannerMetrics",
                      M.planner_metrics(ns, seq, pname, inf_us, True, nodes, plen), ns)
        nav_bag.write("/bridge/predict_timing", "tb3_interfaces/msg/BridgePredictTiming",
                      M.bridge_timing(ns, seq, b_recv, b_send), ns)
        orch.write("/predict_tcp_timing", "tb3_interfaces/msg/PredictTcpTiming",
                   M.predict_timing(ns, seq, l_send, l_recv), ns)
        return inf_us

    # Warmup predicts (before run_start) — low sequence numbers, outside window.
    # Convergent profile (cold first call, then settling) so warmup_convergence_ok
    # is True by default; analysis still excludes these by the run window.
    warm_profile = [3800, 1950, 1560, 1490, 1480]
    seq = 0
    warmup_us = []
    for k in range(n_warm):
        seq += 1
        warmup_us.append(emit_call(seq, warm_t[k], inside_run=False,
                                   inf_override=warm_profile[k % len(warm_profile)]))
        jline({"t": (BASE_EPOCH).isoformat(), "mono_ns": _ns(warm_t[k]),
               "event": "warmup_predict", "sequence": seq, "inference_us": warmup_us[-1]})

    # run_start (bridge-host-stamped event + orchestrator copy + JSONL)
    payload_start = {"mode": mode, "planner": planner, "map_id": map_id,
                     "goal_cell": list(goal_cell) if goal_cell else None,
                     "start_cell": list(start_cell), "surface": "polished concrete, lab",
                     "wifi_rssi_dbm": -54}
    nav_bag.write("/experiment/events", "tb3_interfaces/msg/ExperimentEvent",
                  M.event(_ns(t_run_start), "run_start", run_id, payload_start), _ns(t_run_start))
    orch.write("/experiment/events", "tb3_interfaces/msg/ExperimentEvent",
               M.event(_ns(t_run_start), "run_start", run_id, payload_start), _ns(t_run_start))
    jline({"t": BASE_EPOCH.isoformat(), "mono_ns": _ns(t_run_start),
           "event": "run_start", "run_id": run_id, "payload": payload_start})

    # Run steps: one predict + goal + grid_pose/odom motion per cell transition
    dist_accum = 0.0
    pose_hz = 5
    for k in range(n_steps):
        seq += 1
        t_call = t_run_start + 0.5 + k * step_dt
        emit_call(seq, t_call, inside_run=True)
        cur, nxt = path[k], path[k + 1]
        dcol, drow = nxt[0] - cur[0], nxt[1] - cur[1]
        action_dir = (int(np.sign(dcol)), int(np.sign(drow)))
        jline({"t": BASE_EPOCH.isoformat(), "mono_ns": _ns(t_call),
               "event": "predict_sent", "cycle": k + 1, "retry": 0, "sequence": seq,
               "bridge_robot_pos": list(cur), "bridge_goal_pos": list(goal_cell),
               "obstacle_map": obstacle_bridge, "energy_map": energy_map(k + 1)})
        jline({"t": BASE_EPOCH.isoformat(), "mono_ns": _ns(t_call + 0.05),
               "event": "predict_result", "cycle": k + 1, "retry": 0, "sequence": seq,
               "action": 7, "direction": list(action_dir),
               "direction_x": action_dir[0], "direction_y": action_dir[1], "label": "step"})
        # motion: interpolate pose over the driving sub-interval
        seg_m = math.hypot(dcol, drow) * CELL_M
        for p in range(pose_hz):
            frac = (p + 1) / pose_hz
            x = (cur[0] + dcol * frac) + 0.5
            y = (cur[1] + drow * frac) + 0.5
            tns = _ns(t_call + 0.2 + frac * 0.8)
            nav_bag.write("/grid_nav_node/grid_pose", "tb3_interfaces/msg/GridPose",
                          M.grid_pose(x, y, math.atan2(drow, dcol)), tns)
            nav_bag.write("/odom", "nav_msgs/msg/Odometry",
                          M.odom(tns, x * CELL_M, y * CELL_M, seg_m / 0.8), tns)
        dist_accum += seg_m
        reached_final = (k == n_steps - 1)
        nav_bag.write("/grid_nav_node/last_result", "tb3_interfaces/msg/MoveToGridResult",
                      M.last_result(_ns(t_call + step_dt - 0.1),
                                    True, nxt[0] + 0.5, nxt[1] + 0.5,
                                    math.atan2(drow, dcol), seg_m, 0.0,
                                    "goal_reached"), _ns(t_call + step_dt - 0.1))

    # run_end
    payload_end = {"outcome": outcome}
    nav_bag.write("/experiment/events", "tb3_interfaces/msg/ExperimentEvent",
                  M.event(_ns(t_run_end), "run_end", run_id, payload_end), _ns(t_run_end))
    orch.write("/experiment/events", "tb3_interfaces/msg/ExperimentEvent",
               M.event(_ns(t_run_end), "run_end", run_id, payload_end), _ns(t_run_end))
    jline({"t": BASE_EPOCH.isoformat(), "mono_ns": _ns(t_run_end),
           "event": "run_end", "run_id": run_id, "outcome": outcome})

    jsonl.close()
    robot.close()
    if desktop is not None:
        desktop.close()
    orch.close()

    # maps/obstacle.npy in bridge frame, [y, x] with y bottom-up (origin lower)
    obs_yx = np.zeros((GRID, GRID), dtype=np.uint8)
    for x in range(GRID):
        for y in range(GRID):
            obs_yx[y, x] = obstacle_bridge[x][y]
    (rd / "maps").mkdir(exist_ok=True)
    np.save(rd / "maps" / "obstacle.npy", obs_yx)

    # metadata.yaml (v3 schema)
    meta = {
        "run_id": run_id, "session_id": session_id, "mode": mode, "planner": planner,
        "map_id": map_id,
        "started_at": BASE_EPOCH.isoformat(),
        "ended_at": BASE_EPOCH.isoformat(),
        "goal_cell": list(goal_cell) if goal_cell else None,
        "start_cell": list(start_cell),
        "outcome": outcome, "dry_run": bool(dry_run), "bag_capped": False,
        "warmup_inferences_us": warmup_us,
        "ambient_notes": "synthetic run", "git_commits": {"orchestrator": "deadbeef"},
    }
    (rd / "metadata.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    return rd


# -- CLI ----------------------------------------------------------------------

CELLS = [("decentralized", "ppo"), ("decentralized", "astar_energy"),
         ("centralized", "ppo"), ("centralized", "astar_energy")]
# 6 (map, goal) pairs so paired comparisons reach the Wilcoxon threshold (>=5).
MAPS_GOALS = [("map1", (8, 9)), ("map2", (9, 5)), ("map3", (9, 9)),
              ("map4", (5, 9)), ("map5", (9, 2)), ("map6", (2, 9))]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "session", "campaign"):
        s = sub.add_parser(name)
        s.add_argument("outdir")
    r = sub.choices["run"]
    r.add_argument("--mode", default="decentralized",
                   choices=["decentralized", "centralized"])
    r.add_argument("--planner", default="ppo",
                   choices=["ppo", "astar_energy", "astar_shortest"])
    r.add_argument("--map-id", default="map3")
    r.add_argument("--outcome", default="success")
    r.add_argument("--baseline", action="store_true", help="idle baseline (goal_cell=null)")
    r.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.outdir)
    if args.cmd == "run":
        goal = None if args.baseline else (9, 9)
        rd = generate_run(out, mode=args.mode, planner=args.planner, map_id=args.map_id,
                          goal_cell=goal, outcome=args.outcome, dry_run=args.dry_run)
        print(rd)
    elif args.cmd == "session":
        runs_dir = out / "session-decentralized-ppo" / "runs"
        for map_id, goal in MAPS_GOALS:
            rd = generate_run(runs_dir, mode="decentralized", planner="ppo",
                              map_id=map_id, goal_cell=goal, seed=0)
            print(rd)
    elif args.cmd == "campaign":
        for mode, planner in CELLS:
            runs_dir = out / f"session-{mode}-{planner}" / "runs"
            for map_id, goal in MAPS_GOALS:
                rd = generate_run(runs_dir, mode=mode, planner=planner,
                                  map_id=map_id, goal_cell=goal, seed=0)
                print(rd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
