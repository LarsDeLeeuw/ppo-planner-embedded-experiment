#!/usr/bin/env bash
# =============================================================================
# build_all.sh — Build every ROS2 package in this project.
#
# Symlinks each package into the colcon workspace and runs a single
# colcon build.  Safe to re-run: existing symlinks are left in place.
#
# Usage:
#   ./scripts/build_all.sh              # build all
#   ./scripts/build_all.sh --clean      # wipe build artifacts first
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WS="${WS:-$HOME/turtlebot3_ws}"

# ROS2 packages to build (order doesn't matter — colcon resolves deps).
PACKAGES=(tb3_interfaces tb3_planner_common tb3_nav ros2_bridge ppo_planner astar_planner tb3_power_sensor tb3_diagnostics tb3_rapl_sampler tb3_bringup)

die() { echo "ERROR: $*" >&2; exit 1; }

[[ -d "$WS/src" ]] || die "Workspace src/ not found: $WS/src"

# Handle --clean flag
if [[ "${1:-}" == "--clean" ]]; then
    echo "[clean] Removing build/ install/ log/ in $WS"
    rm -rf "$WS/build" "$WS/install" "$WS/log"
fi

# Source ROS base
set +u
source /opt/ros/humble/setup.bash
set -u

# Ensure symlinks exist
echo "[1/3] Ensuring symlinks in $WS/src/"
for pkg in "${PACKAGES[@]}"; do
    src="$PROJECT_DIR/src/$pkg"
    dst="$WS/src/$pkg"
    [[ -d "$src" ]] || die "Package not found: $src"
    if [[ ! -L "$dst" ]]; then
        ln -s "$src" "$dst"
        echo "  linked $pkg"
    fi
done

# Build
echo "[2/3] Building: ${PACKAGES[*]}"
cd "$WS"
colcon build \
    --packages-select "${PACKAGES[@]}" \
    --symlink-install

echo "[3/3] Done. Source the overlay with:"
echo "  source $WS/install/setup.bash"
