"""
Pydantic schema for the tracker application config.

This is the single source of truth for what knobs exist, what types they have,
and what they default to. The loader (loader.py) populates an AppConfig from
defaults.yml → --config file → local.yml → --set CLI overrides.

All sub-configs use extra="forbid" so a typo in an experiment.yml fails fast
with a clear Pydantic error instead of silently using the default.
"""

from __future__ import annotations

from pathlib import Path

import cv2
from pydantic import BaseModel, ConfigDict, Field, model_validator


_STRICT = ConfigDict(extra="forbid")


class CameraConfig(BaseModel):
    model_config = _STRICT

    index: int | str = 0
    aruco_dict: str = "DICT_4X4_50"
    corner_marker_size_cm: float = 15.2
    robot_marker_size_cm: float = 8.0
    # Height of the robot marker's center above the floor plane (cm).  Used to
    # parallax-correct the homography-based position (the marker sits above the
    # floor, so its camera ray pierces the floor outside its true footprint).
    # 0 = no correction (fine near the camera nadir; small bias toward edges).
    robot_marker_height_cm: float = 0.0
    calibration_file: Path | None = None
    hfov_deg: float | None = 63.0
    # Threaded capture / auto-reconnect (frame_source.ThreadedFrameSource).
    # reconnect: re-open the capture on a read failure instead of crashing the
    # app (essential for network/URL streams that drop during a campaign's
    # blocking inter-trial SSH/scp steps). buffer_size sets CAP_PROP_BUFFERSIZE
    # so we get the freshest frame after a stall rather than a stale backlog.
    reconnect: bool = True
    reconnect_backoff_s: float = 1.0
    buffer_size: int = 1

    @model_validator(mode="after")
    def _check_aruco_dict(self) -> "CameraConfig":
        if not hasattr(cv2.aruco, self.aruco_dict):
            raise ValueError(
                f"camera.aruco_dict={self.aruco_dict!r} is not a valid cv2.aruco constant"
            )
        return self


class TrackingConfig(BaseModel):
    model_config = _STRICT

    loop_rate_hz: int = 30
    heading_smooth_alpha: float = 0.35
    position_max_speed: float = 2.0
    # Consecutive rejected frames before the outlier gate accepts a reading as
    # a real teleport (e.g. the operator carrying the robot to the start
    # between trials) rather than welding to the stale position. <=0 disables.
    position_reject_max_frames: int = 8
    # Reject a robot pose whose marker quad reprojects worse than this (px).
    # Bad detections on a compressed stream fit a square poorly and throw the
    # PnP depth (hence position/ALT) off; clean detections sit well under 1px.
    # Set to 0 (or negative) to disable the gate.
    max_pose_reproj_px: float = 0.9


class GridConfig(BaseModel):
    model_config = _STRICT

    corner_ids: list[int] = Field(default_factory=lambda: [1, 2, 3, 4])
    robot_marker_id: int = 0
    cols: int = 10
    rows: int = 10
    auto_lock: bool = True
    # Physical edge length of one grid cell in metres. Used by the analysis
    # pipeline (not by tracking) to scale /odom-integrated paths into bridge-
    # grid units. Measure once against your physical setup; default is a
    # placeholder that WILL be wrong for most labs.
    cell_size_m: float = 0.30

    @model_validator(mode="after")
    def _check(self) -> "GridConfig":
        if len(self.corner_ids) != 4:
            raise ValueError(
                f"grid.corner_ids must have exactly 4 entries, got {len(self.corner_ids)}"
            )
        if self.robot_marker_id in self.corner_ids:
            raise ValueError(
                f"grid.robot_marker_id={self.robot_marker_id} collides with grid.corner_ids"
            )
        if self.cols <= 0 or self.rows <= 0:
            raise ValueError("grid.cols and grid.rows must be positive")
        return self


class BridgeConfig(BaseModel):
    model_config = _STRICT

    enabled: bool = True
    host: str = "localhost"
    port: int = 9090


class SettleConfig(BaseModel):
    model_config = _STRICT

    speed_threshold: float = 0.10
    angular_threshold_deg_per_s: float = 5.0
    min_duration_s: float = 0.5
    timeout_s: float = 5.0
    freshness_window_s: float = 0.5


# auto driver WAITING_GOAL timeout — guards against nav-node hangs that swallow
# the MoveToGrid action result (handover §13). When this fires the run aborts
# with outcome=failure rather than wedging the campaign forever.
AUTO_DEFAULT_GOAL_TIMEOUT_S = 60.0


class AutoConfig(BaseModel):
    model_config = _STRICT

    log_dir: Path = Path("auto_logs")
    log_maps: bool = True
    max_illegal_retries: int = 3
    predict_cooldown_s: float = 0.25
    predict_timeout_s: float = 5.0
    goal_timeout_s: float = AUTO_DEFAULT_GOAL_TIMEOUT_S
    warp_size: int = 800
    experiment_tag: str = ""
    blocking_layers: list[str] = Field(default_factory=lambda: ["obstacles"])
    settle: SettleConfig = Field(default_factory=SettleConfig)
    minimap_enabled: bool = True
    minimap_cell_px: int = 36


class SceneConfig(BaseModel):
    model_config = _STRICT

    dir: Path = Path("scenes")
    default_name: str = "scene"


class CaptureConfig(BaseModel):
    model_config = _STRICT

    dir: Path = Path("captures")
    warp_size: int = 800


class ThemeConfig(BaseModel):
    """Mirrors `OverlayTheme` in [overlay.py:29-85].

    Defaults come from `DEFAULT_THEME` at [overlay.py:113-156]. Every field is
    tunable from yaml so HUD looks can be adjusted without touching code (e.g.
    higher contrast for report screenshots, colour-blind palettes).
    """
    model_config = _STRICT

    # Colours (BGR)
    col_grid: tuple[int, int, int] = (200, 200, 200)
    col_grid_border: tuple[int, int, int] = (255, 255, 255)
    col_cell_ok: tuple[int, int, int] = (0, 255, 128)
    col_cell_oob: tuple[int, int, int] = (0, 0, 255)
    col_arrow: tuple[int, int, int] = (0, 0, 255)
    col_hud_text: tuple[int, int, int] = (255, 255, 255)
    col_hud_bg: tuple[int, int, int] = (0, 0, 0)
    col_label: tuple[int, int, int] = (220, 220, 220)
    col_target: tuple[int, int, int] = (255, 180, 0)
    col_hint: tuple[int, int, int] = (255, 255, 255)
    col_hint_shadow: tuple[int, int, int] = (0, 0, 0)
    col_marker: tuple[int, int, int] = (0, 255, 0)
    col_marker_id: tuple[int, int, int] = (0, 255, 0)

    # cv2 font enum (FONT_HERSHEY_SIMPLEX = 0)
    font: int = 0

    # Thicknesses
    grid_thick_inner: int = 1
    grid_thick_outer: int = 2
    arrow_thick: int = 2
    hint_outline_thick: int = 3
    hud_text_thick: int = 1
    marker_thick: int = 2

    # Font scales
    hud_scale: float = 0.55
    label_scale: float = 0.35
    hint_scale: float = 0.5
    marker_id_scale: float = 0.5

    # Layout
    hud_line_h: int = 24
    hud_pad: int = 8
    hud_char_w: int = 10
    label_offset: tuple[int, int] = (-10, 5)
    hint_offset: tuple[int, int] = (-6, 6)

    # Effects / geometry
    cell_alpha: float = 0.3
    hud_bg_alpha: float = 0.6
    arrow_len: float = 0.4
    arrow_tip: float = 0.3

    # Auto-mode minimap colours
    col_minimap_bg: tuple[int, int, int] = (0, 0, 0)
    col_minimap_grid: tuple[int, int, int] = (80, 80, 80)
    col_minimap_obstacle: tuple[int, int, int] = (60, 60, 220)
    col_minimap_action_legal: tuple[int, int, int] = (220, 220, 0)
    col_minimap_action_illegal: tuple[int, int, int] = (60, 60, 220)
    col_minimap_text: tuple[int, int, int] = (230, 230, 230)
    col_minimap_title: tuple[int, int, int] = (180, 220, 220)
    col_minimap_robot: tuple[int, int, int] = (0, 255, 128)
    col_minimap_goal: tuple[int, int, int] = (220, 0, 200)


class UiConfig(BaseModel):
    model_config = _STRICT

    show_preview: bool = True
    reference_height: int = 1080
    scale_multiplier: float = 1.0
    theme: ThemeConfig = Field(default_factory=ThemeConfig)


class SshHostConfig(BaseModel):
    """Per-host SSH config. `target` is whatever your `~/.ssh/config` accepts
    (a host alias or `user@host`); `bag_script` is the absolute path to
    `tb3_bag_record.sh` on the REMOTE host. Both should be confirmed once
    against the actual deployment before the first live run — defaults assume
    user `ubuntu` with the repo cloned at `~/ppo-tb3-research-project`.
    """
    model_config = _STRICT

    target: str = "robot"            # `~/.ssh/config` alias
    bag_script: str = "/home/ubuntu/ppo-tb3-research-project/robot/scripts/tb3_bag_record.sh"
    remote_tmp: str = "/tmp/experiments"


class OrchestratorSshConfig(BaseModel):
    model_config = _STRICT

    pi: SshHostConfig = Field(default_factory=SshHostConfig)
    desktop: SshHostConfig = Field(default_factory=lambda: SshHostConfig(
        target="vm"))
    connect_timeout_s: int = 10
    max_retries: int = 3


class OrchestratorConfig(BaseModel):
    """Campaign + per-run orchestration on top of the existing auto_driver."""
    model_config = _STRICT

    enabled: bool = False
    experiments_root: Path = Path("../experiments")
    research_project_root: Path | None = None
    warmup_count: int = 5
    warmup_timeout_s: float = 10.0
    bag_max_duration_s: int = 300
    # Backstop wallclock for the in-flight run (handover §13 / WAITING_GOAL hang).
    # Set below bag_max_duration_s so we exit RUN_ACTIVE before the remote SIGINT.
    run_max_duration_s: int = 240
    # When True, the orchestrator `rm -rf`s `/tmp/experiments/<run_id>` on each
    # remote host after pull (good housekeeping for long campaigns). When
    # False, the bag stays on the Pi indefinitely so the operator can
    # manually recover if anything went sideways with the pull. Default off
    # because storage on /tmp is cheap and silent data loss is expensive.
    cleanup_remote_after_pull: bool = False
    # PPO-only "last cell" assist. When True, on a run whose cell.planner is
    # "ppo", once the robot sits 8-adjacent to the goal the orchestrator still
    # sends the predict (so the planner's proposed action is logged) but drives
    # the final step onto the goal cell itself instead of the proposed cell —
    # working around the stochastic policy stalling one cell short. Never
    # applied to A* planners (they may legitimately step away from a blocked
    # diagonal goal). Assisted finishes are flagged goal_assisted=true in
    # metadata so analysis can separate them from genuine PPO successes.
    ppo_goal_assist: bool = False
    voltage_band_v: tuple[float, float] = (11.8, 12.6)
    summarize_after_run: bool = True
    summarize_python: str | None = None        # path to analysis venv python; None=skip
    summarize_script: Path = Path("../analysis/summarize_run.py")
    ssh_mock: bool = False                     # local fake; no real SSH/scp
    ssh: OrchestratorSshConfig = Field(default_factory=OrchestratorSshConfig)


class AppConfig(BaseModel):
    """Top-level resolved config. Constructed by `config.load_config(argv)`."""
    model_config = _STRICT

    camera: CameraConfig = Field(default_factory=CameraConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    grid: GridConfig = Field(default_factory=GridConfig)
    bridge: BridgeConfig = Field(default_factory=BridgeConfig)
    scene: SceneConfig = Field(default_factory=SceneConfig)
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    auto: AutoConfig = Field(default_factory=AutoConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    ui: UiConfig = Field(default_factory=UiConfig)
