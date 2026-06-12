"""
main.py — AR grid robot localization.

Capture loop:
  1. Grab frame from webcam
  2. Detect ArUco markers → classify as corner or robot
  3. If grid not locked, attempt calibration from 4 corner markers
  4. If grid locked + robot visible, localize robot in grid
  5. Draw AR overlay (grid lines, cell highlight, heading arrow, HUD)

Press Esc to quit, 'r' to re-calibrate grid, WASD/QEZC to select target cell.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import config
from keybindings import NAV_KEY_MAP
from camera import get_camera_matrix
from frame_source import ThreadedFrameSource
from detect import create_detector, detect_markers
from grid import (
    GridState, compute_grid, estimate_marker_pose_on_floor, image_to_grid,
    localize_robot_stable,
)
from robot import RobotTracker
from navigation import NavigationState
from overlay import draw_markers, draw_overlay
from overlay_controller import OverlayController
from bridge_client import BridgeClient, BridgeConfig
from capture_pipeline import CapturePipeline
from luminance_analyzer import LuminanceAnalyzer
from scene import Scene, Cardinality, Style, save_scene, load_scene
from scene_input import NameEntry, LoadPicker, list_scenes
from auto_driver import AutoDriver, AutoConfig
from auto_recorder import AutoRecorder, InMemoryAutoRecorder, NullAutoRecorder
from auto_session_log import AutoSessionLog
from energy_source import LuminanceEnergySource
from legality import InBoundsRule, NotObstacleRule
from map_builder import MapBuilder
from planner import BridgePlanner
from orchestrator import Cell, MapEntry
from orchestrator.orchestrator import Orchestrator


def _scene_path(scene_dir: str, name: str) -> str:
    return f"{scene_dir}/{name}.json"


def _scene_name_from_path(path) -> str:
    from pathlib import Path as _P
    return _P(path).stem


# Scene layers registered at startup.  Add new (name, color_bgr, key, cardinality,
# style) tuples here — everything else (rendering, save/load, rotation recovery,
# HUD) picks them up automatically.
SCENE_LAYER_SPECS: list[tuple[str, tuple[int, int, int], int, Cardinality, Style]] = [
    ("obstacles", (0, 0, 255), ord("o"), Cardinality.MANY, Style.FILL),
    ("goal",      (220, 0, 200), ord("g"), Cardinality.SINGLE, Style.CIRCLE),
]


def _build_scene(cols: int, rows: int) -> Scene:
    s = Scene(cols=cols, rows=rows)
    for name, color, key, card, style in SCENE_LAYER_SPECS:
        s.register(name, color, key, card, style)
    return s

import logging
logging.getLogger("bridge_client").setLevel(logging.DEBUG)
logging.basicConfig()


def _parse_cli(argv: list[str]) -> argparse.Namespace:
    """tracker-level CLI. Loader-level flags (--config, --set, --print-config)
    are passed through to config.load_config via `parse_known_args` upstream."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--campaign", type=Path, default=None,
                   help="path to a campaign_plan.yaml; loads + queues all runs")
    p.add_argument("--auto-advance", action="store_true",
                   help="start the next queued run as soon as the previous one ends")
    p.add_argument("--ssh-mock", action="store_true",
                   help="bypass real SSH/scp (no robot needed); useful for end-to-end test")
    return p.parse_known_args(argv)[0]


def main() -> None:
    cli_args = _parse_cli(sys.argv[1:])

    # Resolve config honouring defaults.yml -> --config <experiment.yml> ->
    # local.yml -> --set overrides.  This is the single place the experiment
    # yaml is applied; everything below reads from `cfg`, not the legacy
    # import-time `config.*` constants (which only see defaults + local.yml).
    cfg, loader_args = config.load_config(sys.argv[1:])
    if loader_args.print_config:
        print(config.dump_yaml(cfg))
        return
    if loader_args.validate_config:
        print("[config] valid")
        return

    # Type-converted conveniences (Path -> str, tuple -> list) several
    # consumers below expect, mirroring the old legacy-constant coercions.
    scene_dir = str(cfg.scene.dir)
    capture_dir = str(cfg.capture.dir)
    corner_ids = list(cfg.grid.corner_ids)
    calibration_file = (
        str(cfg.camera.calibration_file) if cfg.camera.calibration_file else None
    )

    # --- Optional Bridge ---
    bridge: BridgeClient | None = None
    auto: AutoDriver | None = None
    recorder: AutoRecorder = NullAutoRecorder()
    if cfg.bridge.enabled:
        bridge = BridgeClient(BridgeConfig(
            host=cfg.bridge.host,
            port=cfg.bridge.port,
            grid_rows=cfg.grid.rows,
        ))
        bridge.on("nav_pose", lambda _: None)

        def _print_goal_feedback(msg: dict) -> None:
            print(
                f"[bridge] feedback: {msg.get('phase')}  "
                f"dist={msg.get('distance', 0):.2f}  "
                f"hdg_err={msg.get('heading_error', 0):.2f}",
            )

        def _print_goal_result(msg: dict) -> None:
            print(
                f"[bridge] result: success={msg.get('success')}  "
                f"{msg.get('message', '')}",
            )

        bridge.on("goal_feedback", _print_goal_feedback)
        bridge.on("goal_result", _print_goal_result)

        # Build the autonomous planner loop over the bridge.
        planner = BridgePlanner(bridge)
        energy_src = LuminanceEnergySource(LuminanceAnalyzer(), cfg.auto.warp_size)
        map_builder = MapBuilder(blocking_layers=list(cfg.auto.blocking_layers))
        rules = [InBoundsRule(), NotObstacleRule(layer="obstacles")]
        recorder = (
            InMemoryAutoRecorder() if cfg.auto.minimap_enabled
            else NullAutoRecorder()
        )

        def _auto_log_factory() -> AutoSessionLog:
            return AutoSessionLog.open(
                Path(cfg.auto.log_dir), include_maps=cfg.auto.log_maps,
            )

        auto = AutoDriver(
            planner=planner,
            energy=energy_src,
            map_builder=map_builder,
            rules=rules,
            log_factory=_auto_log_factory,
            cfg=AutoConfig(
                max_illegal_retries=cfg.auto.max_illegal_retries,
                predict_cooldown_s=cfg.auto.predict_cooldown_s,
                predict_timeout_s=cfg.auto.predict_timeout_s,
                goal_timeout_s=cfg.auto.goal_timeout_s,
                experiment_tag=cfg.auto.experiment_tag,
                settle_speed_threshold=cfg.auto.settle.speed_threshold,
                settle_angular_threshold=cfg.auto.settle.angular_threshold_deg_per_s,
                settle_min_duration_s=cfg.auto.settle.min_duration_s,
                settle_timeout_s=cfg.auto.settle.timeout_s,
                settle_freshness_window_s=cfg.auto.settle.freshness_window_s,
            ),
            goal_sender=bridge.send_goal,
            cancel_sender=bridge.send_cancel,
            pose_sender=bridge.send_pose,
            recorder=recorder,
        )

        # Auto driver is a second subscriber for goal feedback / result.
        bridge.on("goal_feedback", auto.on_goal_feedback)
        bridge.on("goal_result",   auto.on_goal_result)

        print(f"[bridge] connecting to {cfg.bridge.host}:{cfg.bridge.port}")

    # --- Orchestrator (opt-in via orchestrator.enabled or --campaign / --ssh-mock) ---
    orch: Orchestrator | None = None
    orch_enabled = (
        bridge is not None
        and (cfg.orchestrator.enabled or cli_args.campaign or cli_args.ssh_mock)
    )

    # --- Camera (threaded + self-healing; see frame_source.py) ---
    cap = ThreadedFrameSource(
        cfg.camera.index,
        reconnect=cfg.camera.reconnect,
        reconnect_backoff_s=cfg.camera.reconnect_backoff_s,
        buffer_size=cfg.camera.buffer_size,
    )
    if not cap.start():
        print(f"[grid] cannot open camera index {cfg.camera.index} "
              f"({cap.last_error})")
        sys.exit(1)
    print(f"[grid] camera {cfg.camera.index} opened")

    frame = cap.wait_first_frame(timeout_s=10.0)
    if frame is None:
        print("[grid] failed to read from camera")
        cap.release()
        sys.exit(1)

    camera_matrix, dist_coeffs = get_camera_matrix(
        frame.shape, calibration_file, cfg.camera.hfov_deg,
    )
    detector = create_detector(cfg.camera.aruco_dict)
    corner_id_set = set(corner_ids)
    robot_tracker = RobotTracker(
        heading_alpha=cfg.tracking.heading_smooth_alpha,
        max_speed=cfg.tracking.position_max_speed,
        reject_max_frames=cfg.tracking.position_reject_max_frames,
    )
    # Reset the tracker at each trial start so a previous run's held position
    # can't carry over into the next trial (see RobotTracker's outlier gate).
    if auto is not None:
        auto.set_tracker_reset(robot_tracker.reset)
    last_robot_state = None   # held across frames so a gated bad detection reuses it
    nav = NavigationState(cfg.grid.cols, cfg.grid.rows)
    show_hints = True

    scene = _build_scene(cfg.grid.cols, cfg.grid.rows)
    mark_mode = False
    active_layer_name = scene.layer_names()[0] if scene.layer_names() else None

    # Transient HUD input modes.  Only one is active at a time.
    input_mode: str | None = None       # None | "naming" | "load_select"
    name_entry: NameEntry | None = None
    load_picker: LoadPicker | None = None
    last_scene_name: str = cfg.scene.default_name

    # Auto-mode debug minimap visibility.  Initial state from env var; runtime
    # toggle via `v`.  Independent of whether the recorder is constructed —
    # NullAutoRecorder always yields snapshot()=None so this flag is moot then.
    minimap_visible: bool = cfg.auto.minimap_enabled

    overlay_ctrl = OverlayController(
        reference_height=cfg.ui.reference_height,
        initial_multiplier=cfg.ui.scale_multiplier,
    )
    overlay_ctrl.refresh_for_frame(frame)

    # Capture pipeline
    pipeline = CapturePipeline(capture_dir, cfg.capture.warp_size)
    pipeline.register(LuminanceAnalyzer())
    pending_capture = False

    # Mutable state dict for the mouse callback (avoids closure rebinding issues)
    _cb_state: dict = {
        "grid": None,
        "scene": scene,
        "mark_mode": mark_mode,
        "active_layer": active_layer_name,
        "input_mode": None,
        "auto_active": False,
    }

    def _on_mouse(event: int, x: int, y: int, flags: int, param: dict) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or param["grid"] is None:
            return
        # Ignore clicks while a transient HUD mode (name entry, load picker) is open.
        if param["input_mode"] is not None:
            return
        # Ignore clicks while the autonomous loop is driving — avoids fighting
        # with its goal stream.
        if param["auto_active"]:
            return
        from math import floor as _floor
        gpt = image_to_grid(
            np.array([x, y], dtype=np.float64), param["grid"].H_inv,
        )
        col, row = int(_floor(gpt[0])), int(_floor(gpt[1]))
        if param["mark_mode"] and param["active_layer"] is not None:
            param["scene"].get(param["active_layer"]).toggle(col, row)
        else:
            nav.set_target(col, row)

    cv2.namedWindow("Grid Tracker", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Grid Tracker", _on_mouse, _cb_state)

    # --- Orchestrator wiring (sits inside the main loop; ticked each frame) --
    if orch_enabled and auto is not None and bridge is not None:
        def _orch_scene_loader(path: Path):
            result = load_scene(
                path, scene.cols, scene.rows,
                corner_ids, SCENE_LAYER_SPECS,
            )
            if result.scene is None:
                raise RuntimeError(f"load_scene failed for {path}: {result.message}")
            return result.scene

        # --ssh-mock overrides the resolved orchestrator config for this run
        # (the orchestrator reads cfg.orchestrator.ssh_mock at construction).
        if cli_args.ssh_mock:
            cfg = cfg.model_copy(update={
                "orchestrator": cfg.orchestrator.model_copy(
                    update={"ssh_mock": True}),
            })

        orch = Orchestrator(
            app_cfg=cfg,
            bridge=bridge,
            auto=auto,
            cb_state=_cb_state,
            scene_loader=_orch_scene_loader,
            corner_marker_ids=list(corner_ids),
            scene_dir=Path(scene_dir),
        )
        if cli_args.campaign:
            orch.load_campaign(cli_args.campaign)
        if cli_args.auto_advance:
            orch.set_auto_advance(True)
        print(f"[orch] enabled  ssh_mock={cfg.orchestrator.ssh_mock}  "
              f"queue={len(orch.queue)}  auto_advance={cli_args.auto_advance}")

    grid: GridState | None = None
    min_interval = 1.0 / cfg.tracking.loop_rate_hz

    print(f"[grid] grid={cfg.grid.cols}x{cfg.grid.rows}  "
          f"corners={corner_ids}  robot={cfg.grid.robot_marker_id}")
    print(f"[grid] corner_size={cfg.camera.corner_marker_size_cm}cm  "
          f"robot_size={cfg.camera.robot_marker_size_cm}cm")
    print(f"[grid] auto_lock={cfg.grid.auto_lock}")
    print("[grid] press Esc to quit, 'r' to re-calibrate, WASD/QEZC for target, +/- to scale UI, 0 to reset")
    print("[grid] 'm' to toggle mark mode (click=toggle cell), K/L save/load scene, [/]/F/X (mark mode) rotate/flip/clear")
    print("[grid] 'p' to toggle autonomous planner loop (requires bridge + goal cell)")
    print("[grid] 'v' toggles auto minimap visibility, 'V' clears the minimap trail")

    try:
        while True:
            loop_start = time.monotonic()
            # Refresh from cb_state so orchestrator-driven scene swaps (per-run
            # map loads) are visible to the local scope this iteration onwards.
            scene = _cb_state["scene"]

            ret, frame = cap.read()
            if not ret:
                # No frame yet — startup race, or the stream is down and the
                # grabber thread is reconnecting in the background.  Keep the
                # loop alive (do NOT break / exit the app) and pump the GUI so
                # the window stays responsive; press Esc to quit.
                if cv2.waitKey(30) & 0xFF == 27:
                    break
                continue

            overlay_ctrl.refresh_for_frame(frame)

            # Save a clean copy before any overlay rendering touches the frame.
            # Needed for the capture pipeline (g) and for auto mode's live
            # luminance sampling.  Explicit None init so any future caller
            # that reads raw_frame without the guard below fails loudly.
            raw_frame = None
            need_raw = pending_capture or (auto is not None and auto.is_active())
            if need_raw:
                raw_frame = frame.copy()

            markers = detect_markers(frame, detector)
            draw_markers(frame, markers, overlay_ctrl.theme)

            # Classify markers
            corner_markers: dict[int, any] = {}
            robot_marker = None
            for m in markers:
                if m.marker_id in corner_id_set:
                    corner_markers[m.marker_id] = m.corners
                elif m.marker_id == cfg.grid.robot_marker_id:
                    robot_marker = m

            # Grid calibration
            if grid is None and cfg.grid.auto_lock:
                new_grid = compute_grid(
                    corner_markers, corner_ids,
                    cfg.grid.cols, cfg.grid.rows,
                    camera_matrix, dist_coeffs,
                    cfg.camera.corner_marker_size_cm,
                )
                if new_grid is not None:
                    grid = new_grid
                    _cb_state["grid"] = grid
                    if (scene.cols, scene.rows) != (grid.cols, grid.rows):
                        scene.resize(grid.cols, grid.rows)
                    robot_tracker.reset()
                    last_robot_state = None   # old pose is in the previous floor frame
                    print("[grid] grid LOCKED")

            # Robot localization (3D pose → floor projection)
            robot_state = None
            if grid is not None and robot_marker is not None:
                # Disambiguate the planar PnP flip using the locked grid's floor
                # normal — the robot marker lies flat, so its true normal is
                # parallel to the floor.  Avoids the frame-to-frame pose toggle
                # that estimatePoseSingleMarkers exhibits on a small, obliquely
                # viewed marker.
                robot_rvec, robot_tvec, pose_reproj = estimate_marker_pose_on_floor(
                    robot_marker.corners, cfg.camera.robot_marker_size_cm,
                    camera_matrix, dist_coeffs, grid.floor_R[:, 2],
                )
                gate = cfg.tracking.max_pose_reproj_px
                if gate > 0 and pose_reproj > gate:
                    # Low-quality detection: the marker quad fits a square poorly,
                    # so the PnP depth (and thus position/ALT) is unreliable.
                    # Drop this frame and keep the last good pose.
                    if os.environ.get("QRT_DEBUG_POSE"):
                        print(f"[pose] REJECT reproj={pose_reproj:.3f} > {gate} — held last good")
                    robot_state = last_robot_state
                else:
                    # Position from the stable marker-center ∩ floor (homography),
                    # not the noisy PnP depth; heading/height from the pose.
                    pose = localize_robot_stable(
                        robot_tvec, robot_rvec, robot_marker.corners,
                        grid, camera_matrix, cfg.camera.robot_marker_height_cm,
                    )
                    robot_state = robot_tracker.update(pose)
                    last_robot_state = robot_state

            # Capture pipeline (runs on the clean frame saved before detection)
            if pending_capture and grid is not None:
                pipeline.run(raw_frame, grid)
                pending_capture = False

            # Autonomous planner tick (no-op when auto mode is not active).
            # Passes the clean pre-overlay frame so luminance sampling isn't
            # polluted by the drawn marker outlines.
            if auto is not None and auto.is_active():
                auto.tick(raw_frame, robot_state, grid, scene, time.monotonic())
            _cb_state["auto_active"] = auto is not None and auto.is_active()

            # Orchestrator state machine — only ticks when enabled; otherwise a
            # no-op so the existing GUI behavior is unaffected.
            if orch is not None:
                try:
                    orch.tick(raw_frame, robot_state, grid)
                except Exception:
                    import traceback
                    traceback.print_exc()

            # Compute nav hints from robot's current cell
            hints: list[tuple[int, int, str]] | None = None
            if show_hints and robot_state is not None and grid is not None:
                p = robot_state.pose
                hints = []
                for key_code, (dc, dr) in NAV_KEY_MAP.items():
                    nc, nr = p.cell_col + dc, p.cell_row + dr
                    if 0 <= nc < grid.cols and 0 <= nr < grid.rows:
                        hints.append((nc, nr, chr(key_code).upper()))

            # HUD extras for whichever transient input mode is active
            hud_extras: list[tuple[str, tuple[int, int, int] | None]] | None = None
            if input_mode == "naming" and name_entry is not None:
                hud_extras = [
                    (f"SAVE NAME: {name_entry.display()}", (0, 180, 255)),
                    ("(Enter: save, Esc: cancel)", None),
                ]
            elif input_mode == "load_select" and load_picker is not None:
                if not load_picker.entries:
                    hud_extras = [
                        ("LOAD: (no scenes found)", (0, 0, 255)),
                        ("(Esc to cancel)", None),
                    ]
                else:
                    lines: list[tuple[str, tuple[int, int, int] | None]] = [
                        (
                            f"LOAD: page {load_picker.page + 1}/{load_picker.page_count}",
                            (0, 180, 255),
                        )
                    ]
                    for slot, path in load_picker.visible():
                        lines.append((f"  [{slot}] {_scene_name_from_path(path)}", None))
                    lines.append(("(1-9: load, [/]: page, Esc: cancel)", None))
                    hud_extras = lines

            # Snapshot the auto recorder once per frame so render uses a
            # consistent view even if the recv thread mutates underneath.
            auto_snap = recorder.snapshot()

            # AR overlay
            target = nav.target
            draw_overlay(
                frame, grid, robot_state, len(corner_markers),
                overlay_ctrl.theme,
                target_cell=(target.col, target.row) if target else None,
                nav_hints=hints,
                ui_multiplier=overlay_ctrl.multiplier,
                scene=scene,
                mark_mode=mark_mode,
                active_layer_name=active_layer_name,
                hud_extra_lines=hud_extras,
                auto_status=auto.hud_status() if auto is not None else None,
                auto_snapshot=auto_snap,
                minimap_cell_px=cfg.auto.minimap_cell_px,
                minimap_visible=minimap_visible,
            )

            # Navigation packet (print once per target change)
            if nav.consume_changed() and robot_state is not None:
                packet = nav.build_packet(robot_state)
                if packet is not None:
                    print(f"[nav] {packet.to_dict()}")
                    if bridge is not None and nav.target is not None:
                        bridge.send_goal(nav.target.col + 0.5, nav.target.row + 0.5, robot_state)

            # Display
            cv2.imshow("Grid Tracker", frame)
            key = cv2.waitKey(1) & 0xFF

            if input_mode == "naming" and name_entry is not None:
                outcome = name_entry.accept_key(key)
                if outcome == "submit":
                    name = name_entry.buffer
                    last_scene_name = name
                    path = _scene_path(scene_dir, name)
                    try:
                        save_scene(path, scene, corner_ids)
                        counts = ", ".join(f"{l.name}={l.count()}" for l in scene.layers())
                        print(f"[scene] saved to {path}  ({counts})")
                    except OSError as e:
                        print(f"[scene] save failed: {e}")
                    input_mode = None
                    name_entry = None
                    _cb_state["input_mode"] = None
                elif outcome == "cancel":
                    print("[scene] save cancelled")
                    input_mode = None
                    name_entry = None
                    _cb_state["input_mode"] = None
            elif input_mode == "load_select" and load_picker is not None:
                outcome = load_picker.accept_key(key)
                if outcome == "cancel":
                    print("[scene] load cancelled")
                    input_mode = None
                    load_picker = None
                    _cb_state["input_mode"] = None
                elif outcome != "continue":
                    # A Path was returned — load it.
                    path = outcome
                    result = load_scene(
                        path,
                        grid.cols, grid.rows,
                        corner_ids,
                        SCENE_LAYER_SPECS,
                    )
                    print(f"[scene] load: {result.status} - {result.message}")
                    if result.scene is not None:
                        scene = result.scene
                        _cb_state["scene"] = scene
                        last_scene_name = _scene_name_from_path(path)
                        if active_layer_name not in scene:
                            active_layer_name = (
                                scene.layer_names()[0] if scene.layer_names() else None
                            )
                            _cb_state["active_layer"] = active_layer_name
                    input_mode = None
                    load_picker = None
                    _cb_state["input_mode"] = None
            elif key == 27:  # Escape (only when no input mode is active)
                break
            elif mark_mode and any(key == layer.key for layer in scene.layers()):
                # Mark-mode layer selectors win over any global keybinding that
                # uses the same letter (e.g. `g` selects goal here, not capture).
                for layer in scene.layers():
                    if key == layer.key:
                        active_layer_name = layer.name
                        _cb_state["active_layer"] = active_layer_name
                        print(f"[scene] active layer: {active_layer_name}")
                        break
            elif key == ord("r"):
                if auto is not None and auto.is_active():
                    auto.stop("grid unlocked", is_error=False)
                    _cb_state["auto_active"] = False
                    print("[auto] stopped (grid recalibrating)")
                grid = None
                _cb_state["grid"] = None
                robot_tracker.reset()
                nav.clear()
                print("[grid] re-calibrating...")
            elif key == ord("t"):
                show_hints = not show_hints
            elif key == ord("g"):
                if grid is not None:
                    pending_capture = True
                    print("[capture] will capture next frame...")
                else:
                    print("[capture] grid not locked, cannot capture")
            elif key in (ord("+"), ord("=")):
                m = overlay_ctrl.bump(1.2)
                print(f"[ui] scale x{m:.2f}")
            elif key == ord("-"):
                m = overlay_ctrl.bump(1 / 1.2)
                print(f"[ui] scale x{m:.2f}")
            elif key == ord("0"):
                m = overlay_ctrl.reset()
                print(f"[ui] scale x{m:.2f}")
            elif key == ord("m"):
                mark_mode = not mark_mode
                _cb_state["mark_mode"] = mark_mode
                print(f"[scene] mark mode: {'ON' if mark_mode else 'OFF'} "
                      f"(layer={active_layer_name})")
            elif key == ord("p"):
                if auto is None:
                    print("[auto] bridge not enabled; auto mode unavailable")
                elif auto.is_active():
                    auto.stop("user", is_error=False)
                    _cb_state["auto_active"] = False
                    print("[auto] stopped (user)")
                else:
                    auto.clear_error()
                    ok, reason = auto.start(
                        scene, grid, robot_state,
                        corner_marker_ids=list(corner_ids),
                    )
                    if ok:
                        _cb_state["auto_active"] = True
                        print(f"[auto] started (log: {auto._log.path})")  # noqa: SLF001
                    else:
                        print(f"[auto] cannot start: {reason}")
            elif key == ord("v"):
                minimap_visible = not minimap_visible
                print(f"[auto] minimap: {'ON' if minimap_visible else 'OFF'}")
            elif key == ord("V"):
                recorder.clear()
                print("[auto] minimap cleared")
            elif orch is not None and key == ord("N"):
                # Start the next queued run (campaign or hand-queued).
                # _begin_next refuses non-IDLE state internally; we just guard
                # against an empty queue here for a nicer message.
                if not orch.queue:
                    print("[orch] queue empty; nothing to start")
                else:
                    orch._begin_next()  # noqa: SLF001 — operator-driven advance
            elif orch is not None and key == ord("B"):
                # Queue an idle-baseline for the currently-active session
                # cell. If a run is in flight we ONLY enqueue (no auto-start);
                # _tick_idle / auto_advance will pick it up when the current
                # run finishes. This prevents the double-keypress hazard
                # where 'B' during BAGS_STARTING overwrites _active.
                from orchestrator.state import OrchestratorState as _OS
                cell = (orch.queue[0].cell if orch.queue
                        else Cell(mode="decentralized", planner="ppo"))
                orch.enqueue_baseline(cell)
                if orch.state == _OS.IDLE:
                    orch._begin_next()  # noqa: SLF001
                    print(f"[orch] baseline ({cell.mode}/{cell.planner}) started")
                else:
                    print(f"[orch] baseline ({cell.mode}/{cell.planner}) queued "
                          "(will start after current run)")
            elif orch is not None and key in (ord("X"), ord("c")):
                # Cancel the in-flight trial only. The campaign queue is left
                # alone so the next trial fires automatically when teardown
                # completes (requires --auto-advance, else press 'N' to start
                # the next one). Lowercase 'c' is an alias so the operator
                # doesn't have to hit Shift mid-emergency.
                orch.cancel_current("user cancel")
            elif orch is not None and key == ord("Q"):
                print(f"[orch] {orch.status_summary()}")
            elif orch is not None and key == ord("E"):
                # Clear STOPPED_ERROR back to IDLE so the operator can resume
                # the campaign without restarting the app.
                orch.reset_after_error()
            elif key == ord("k"):
                if grid is None:
                    print("[scene] grid not locked, cannot save")
                else:
                    name_entry = NameEntry(buffer=last_scene_name)
                    input_mode = "naming"
                    _cb_state["input_mode"] = input_mode
                    print(f"[scene] enter name (default '{last_scene_name}', Enter=save, Esc=cancel)")
            elif key == ord("l"):
                if grid is None:
                    print("[scene] grid not locked, cannot load")
                else:
                    load_picker = LoadPicker.from_dir(scene_dir)
                    input_mode = "load_select"
                    _cb_state["input_mode"] = input_mode
                    if not load_picker.entries:
                        print(f"[scene] no scenes found in {scene_dir}")
                    else:
                        print(f"[scene] {len(load_picker.entries)} scene(s) available")
            elif mark_mode and key in (ord("["), ord("]")):
                k_rot = 1 if key == ord("[") else 3  # CCW vs CW in np.rot90 terms
                if scene.rows != scene.cols and k_rot % 2 == 1:
                    print("[scene] cannot rotate 90 deg on non-square grid")
                else:
                    scene.rotate90(k_rot)
                    print(f"[scene] rotated {'CCW' if k_rot == 1 else 'CW'}")
            elif mark_mode and key == ord("f"):
                scene.flip_horizontal()
                print("[scene] flipped horizontally")
            elif mark_mode and key == ord("x"):
                if active_layer_name is not None:
                    scene.get(active_layer_name).clear()
                    print(f"[scene] cleared layer '{active_layer_name}'")
            elif key in NAV_KEY_MAP and robot_state is not None:
                dcol, drow = NAV_KEY_MAP[key]
                p = robot_state.pose
                nav.set_target(p.cell_col + dcol, p.cell_row + drow)

            # Rate limiting
            elapsed = time.monotonic() - loop_start
            sleep_for = min_interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    except KeyboardInterrupt:
        print("\n[grid] interrupted")
    finally:
        # Order matters: tear the orchestrator down BEFORE closing the bridge
        # (its shutdown sends run_end + scp pulls go over ssh, unrelated to
        # the bridge socket, but the bridge close is what tells the ROS side
        # we're going away — let the orchestrator emit its events first).
        if orch is not None:
            try:
                orch.shutdown(reason="app exit")
            except Exception:
                import traceback
                traceback.print_exc()
        if auto is not None and auto.is_active():
            auto.stop("app exit", is_error=True)
        cap.release()
        cv2.destroyAllWindows()
        if bridge is not None:
            bridge.close()
        print("[grid] stopped")


if __name__ == "__main__":
    main()
