#!/usr/bin/env bash
set -euo pipefail

WS="${WS:-$HOME/turtlebot3_ws}"
PKG="${PKG:-tb3_nav}"
DEV_DIR="${DEV_DIR:-$HOME/ppo-tb3-research-project/robot/src/$PKG}"
PKG_DIR="$WS/src/$PKG"
EXE="${EXE:-grid_nav_node}"

BUILD_ONLY=0
[[ "${1:-}" == "--build-only" ]] && BUILD_ONLY=1

# ---- helpers ----
die() { echo "ERROR: $*" >&2; exit 1; }

# ROS setup scripts don't like nounset sometimes
set +u
source /opt/ros/humble/setup.bash
set -u

# ---- safety checks ----
[[ -d "$WS/src" ]] || die "Workspace src/ not found: $WS/src"
[[ -d "$DEV_DIR" ]] || die "DEV_DIR not found: $DEV_DIR"

# Require DEV_DIR to look like a ROS2 package
[[ -f "$DEV_DIR/package.xml" ]] || die "DEV_DIR does not contain package.xml: $DEV_DIR"
# Optional: check it's the right package name
if ! grep -q "<name>$PKG</name>" "$DEV_DIR/package.xml"; then
  echo "WARNING: DEV_DIR package.xml name doesn't match PKG=$PKG"
  echo "DEV_DIR name is: $(grep -oP '(?<=<name>).*?(?=</name>)' "$DEV_DIR/package.xml" | head -n1)"
fi

if [[ ! -L "$PKG_DIR" ]]; then
  echo "[1/4] Linking $DEV_DIR -> $PKG_DIR"
  ln -s "$DEV_DIR" "$PKG_DIR"
else
  echo "[1/4] Symlink already exists, skipping"
fi

echo "[2/4] Building package: $PKG"
cd "$WS"
colcon build --packages-select "$PKG" --symlink-install

echo "[3/4] Build complete. To activate in your shell:"
echo "  source $WS/install/setup.bash"

echo "[4/4] Done."
if [[ $BUILD_ONLY -eq 0 && -n "$EXE" ]]; then
  set +u
  source "$WS/install/setup.bash"
  set -u
  echo "Running: ros2 run $PKG $EXE"
  ros2 run "$PKG" "$EXE"
fi
