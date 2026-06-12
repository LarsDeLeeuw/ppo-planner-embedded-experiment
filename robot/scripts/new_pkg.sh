#!/usr/bin/env bash
# =============================================================================
# new_pkg.sh
#
# INTENTION:
#   Automate the boilerplate for creating a new ROS 2 Python package in the
#   turtlebot3_ws workspace. Wraps `ros2 pkg create` with the project's
#   standard conventions and eliminates the manual steps documented in
#   context/vm.md.
#
#   What it does:
#     1. Verifies TURTLEBOT3_MODEL is set (warns if not — should be 'burger')
#     2. Creates the package in WS/src/ via ros2 pkg create
#     3. Prints next-step instructions for:
#        - Adding entry points to setup.py (ros2 run targets)
#        - Making node scripts executable
#        - Running colcon build
#        - Linking with sync_build_pkg_v2.sh if developing from a remote dev dir
#
#   Run this on the VM inside the turtlebot3_ws environment.
#
# USAGE:
#   bash new_pkg.sh <package_name> [extra ros2 pkg create args...]
#
#   Examples:
#     bash new_pkg.sh tb3_mapper
#     bash new_pkg.sh tb3_planner --dependencies rclpy sensor_msgs nav_msgs
#
# CONFIGURATION (env vars):
#   WS           ROS 2 workspace root             (default: ~/turtlebot3_ws)
#   DEFAULT_DEPS Default --dependencies to pass   (default: rclpy geometry_msgs)
# =============================================================================
set -euo pipefail

WS="${WS:-$HOME/turtlebot3_ws}"
DEFAULT_DEPS="${DEFAULT_DEPS:-rclpy geometry_msgs}"

# ---- helpers ----
die() { echo "ERROR: $*" >&2; exit 1; }

[[ $# -ge 1 ]] || die "Usage: $0 <package_name> [extra ros2 pkg create args...]"
PKG="$1"
shift
EXTRA_ARGS=("$@")

# ---- sanity checks ----
[[ -d "$WS/src" ]] || die "Workspace src/ not found: $WS/src"

if [[ -z "${TURTLEBOT3_MODEL:-}" ]]; then
  echo "WARNING: TURTLEBOT3_MODEL is not set. Expected 'burger'."
  echo "  Add to ~/.bashrc: export TURTLEBOT3_MODEL=burger"
else
  echo "TURTLEBOT3_MODEL=$TURTLEBOT3_MODEL"
fi

set +u
source /opt/ros/humble/setup.bash
set -u

PKG_DIR="$WS/src/$PKG"
[[ ! -d "$PKG_DIR" ]] || die "Package already exists: $PKG_DIR"

# ---- create package ----
echo "[1/2] Creating package '$PKG' in $WS/src/"
cd "$WS/src"

# Build dependency args: use caller-supplied or fall back to defaults
if [[ ${#EXTRA_ARGS[@]} -eq 0 ]]; then
  # shellcheck disable=SC2086
  ros2 pkg create "$PKG" --build-type ament_python --dependencies $DEFAULT_DEPS
else
  ros2 pkg create "$PKG" --build-type ament_python "${EXTRA_ARGS[@]}"
fi

# ---- next steps ----
echo ""
echo "[2/2] Package created at: $PKG_DIR"
echo ""
echo "Next steps:"
echo ""
echo "  1. Add your node script(s) to: $PKG_DIR/$PKG/"
echo "     Make each executable:"
echo "       chmod +x $PKG_DIR/$PKG/<node_name>.py"
echo ""
echo "  2. Register entry points in $PKG_DIR/setup.py:"
echo "       'console_scripts': ["
echo "           '<entry_name> = $PKG.<node_name>:main',"
echo "       ],"
echo ""
echo "  3. Build the workspace:"
echo "       cd $WS && colcon build --packages-select $PKG --symlink-install"
echo "       source $WS/install/setup.bash"
echo ""
echo "  4. Run your node:"
echo "       ros2 run $PKG <entry_name>"
echo ""
echo "  5. (Optional) If developing from a remote machine, update DEV_DIR in"
echo "     sync_build_pkg_v2.sh to point at your dev copy of $PKG."
