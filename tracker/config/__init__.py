"""
Legacy compatibility shim for the old tracker/config.py.

This package's preferred surface is `load_config(argv) -> AppConfig`.  Existing
code that still does `import config; config.GRID_COLS` continues to work via
the module-level constants bound below — they're populated once at import time
from defaults.yml (plus local.yml if present).  CLI overrides via --config and
--set are NOT applied at import time (sys.argv may not even belong to us);
call `load_config(sys.argv[1:])` explicitly to honour them.

main.py and calibrate.py now consume the resolved AppConfig directly
(via load_config(sys.argv[1:]) / injected sub-configs), so they honour --config
and --set.  The only remaining legacy consumer is orchestrator/_mock_e2e.py;
this shim can be removed once that migrates too.
"""

from __future__ import annotations

from .loader import (  # re-export for the new surface
    LOCAL_OVERRIDE_PATH,
    build_argparser,
    dump_yaml,
    load_config,
)
from .schema import AppConfig

# tracker/ is on sys.path (main.py runs from there), so the sibling
# module is reachable as an absolute import.  Relative `from ..keybindings`
# would require tracker/ to be a package, which it isn't.
from keybindings import NAV_KEY_MAP  # noqa: F401 (re-exported for legacy import)


# Resolve once with no CLI overrides (defaults + local.yml only).  Failures
# here propagate as ValidationError so import-time problems surface immediately.
_CFG, _ARGS = load_config(argv=[])


# --- legacy module-level names (alphabetical by section) --------------------

# Camera
CAMERA_INDEX = _CFG.camera.index
ARUCO_DICT = _CFG.camera.aruco_dict
CORNER_MARKER_SIZE_CM = _CFG.camera.corner_marker_size_cm
ROBOT_MARKER_SIZE_CM = _CFG.camera.robot_marker_size_cm
CALIBRATION_FILE = str(_CFG.camera.calibration_file) if _CFG.camera.calibration_file else None
CAMERA_HFOV = _CFG.camera.hfov_deg

# Tracking
LOOP_RATE_HZ = _CFG.tracking.loop_rate_hz
HEADING_SMOOTH_ALPHA = _CFG.tracking.heading_smooth_alpha
POSITION_MAX_SPEED = _CFG.tracking.position_max_speed

# UI
SHOW_PREVIEW = _CFG.ui.show_preview
UI_REFERENCE_HEIGHT = _CFG.ui.reference_height
UI_SCALE_MULTIPLIER = _CFG.ui.scale_multiplier

# Grid
GRID_CORNER_IDS = list(_CFG.grid.corner_ids)
ROBOT_MARKER_ID = _CFG.grid.robot_marker_id
GRID_COLS = _CFG.grid.cols
GRID_ROWS = _CFG.grid.rows
GRID_AUTO_LOCK = _CFG.grid.auto_lock

# Bridge
BRIDGE_ENABLED = _CFG.bridge.enabled
BRIDGE_HOST = _CFG.bridge.host
BRIDGE_PORT = _CFG.bridge.port

# Capture pipeline
CAPTURE_DIR = str(_CFG.capture.dir)
CAPTURE_WARP_SIZE = _CFG.capture.warp_size

# Scene
SCENE_DIR = str(_CFG.scene.dir)
SCENE_DEFAULT_NAME = _CFG.scene.default_name

# Auto driver
AUTO_LOG_DIR = str(_CFG.auto.log_dir)
AUTO_LOG_MAPS = _CFG.auto.log_maps
AUTO_MAX_ILLEGAL_RETRIES = _CFG.auto.max_illegal_retries
AUTO_PREDICT_COOLDOWN_S = _CFG.auto.predict_cooldown_s
AUTO_PREDICT_TIMEOUT_S = _CFG.auto.predict_timeout_s
AUTO_WARP_SIZE = _CFG.auto.warp_size
AUTO_EXPERIMENT_TAG = _CFG.auto.experiment_tag
AUTO_BLOCKING_LAYERS = list(_CFG.auto.blocking_layers)
AUTO_SETTLE_SPEED_THRESHOLD = _CFG.auto.settle.speed_threshold
AUTO_SETTLE_ANGULAR_THRESHOLD = _CFG.auto.settle.angular_threshold_deg_per_s
AUTO_SETTLE_MIN_DURATION_S = _CFG.auto.settle.min_duration_s
AUTO_SETTLE_TIMEOUT_S = _CFG.auto.settle.timeout_s
AUTO_SETTLE_FRESHNESS_WINDOW_S = _CFG.auto.settle.freshness_window_s
AUTO_MINIMAP_ENABLED = _CFG.auto.minimap_enabled
AUTO_MINIMAP_CELL_PX = _CFG.auto.minimap_cell_px

# Orchestrator — new code consumes the nested OrchestratorConfig directly via
# APP_CFG.orchestrator; ORCH_CFG is the convenience alias.
APP_CFG = _CFG
ORCH_CFG = _CFG.orchestrator
ORCH_ENABLED = _CFG.orchestrator.enabled


__all__ = [
    # new surface
    "AppConfig",
    "load_config",
    "build_argparser",
    "dump_yaml",
    "LOCAL_OVERRIDE_PATH",
    # legacy compatibility names
    "CAMERA_INDEX", "ARUCO_DICT", "CORNER_MARKER_SIZE_CM", "ROBOT_MARKER_SIZE_CM",
    "CALIBRATION_FILE", "CAMERA_HFOV",
    "LOOP_RATE_HZ", "HEADING_SMOOTH_ALPHA", "POSITION_MAX_SPEED",
    "SHOW_PREVIEW", "UI_REFERENCE_HEIGHT", "UI_SCALE_MULTIPLIER",
    "GRID_CORNER_IDS", "ROBOT_MARKER_ID", "GRID_COLS", "GRID_ROWS", "GRID_AUTO_LOCK",
    "BRIDGE_ENABLED", "BRIDGE_HOST", "BRIDGE_PORT",
    "CAPTURE_DIR", "CAPTURE_WARP_SIZE",
    "SCENE_DIR", "SCENE_DEFAULT_NAME",
    "AUTO_LOG_DIR", "AUTO_LOG_MAPS", "AUTO_MAX_ILLEGAL_RETRIES",
    "AUTO_PREDICT_COOLDOWN_S", "AUTO_PREDICT_TIMEOUT_S", "AUTO_WARP_SIZE",
    "AUTO_EXPERIMENT_TAG", "AUTO_BLOCKING_LAYERS",
    "AUTO_SETTLE_SPEED_THRESHOLD", "AUTO_SETTLE_ANGULAR_THRESHOLD",
    "AUTO_SETTLE_MIN_DURATION_S", "AUTO_SETTLE_TIMEOUT_S",
    "AUTO_SETTLE_FRESHNESS_WINDOW_S",
    "AUTO_MINIMAP_ENABLED", "AUTO_MINIMAP_CELL_PX",
    "NAV_KEY_MAP",
]
