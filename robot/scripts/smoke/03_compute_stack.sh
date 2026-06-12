#!/usr/bin/env bash
# =============================================================================
# 03_compute_stack.sh — Bring up the compute stack (nav + planner + bridge).
#
# Runs alongside 01_robot_sensors.sh on the Pi (decentralized mode), or on
# its own on a desktop VM (centralized mode). Leave it running and exercise
# the bridge with 04_drive_bridge.py from any host that can reach TCP:9090.
#
# In decentralized mode (this is on the Pi where robot_sensors.launch.py is
# already up), DO NOT pass use_diagnostics:=true — the diagnostics node is
# already owned by the always-on layer.
#
# ENV / FLAGS:
#   PLANNER          ppo | astar | astar_shortest | none   (default: ppo)
#   USE_DIAGNOSTICS  true on desktop VM in centralized mode (default: false)
#   USE_RAPL         true on desktop VM in centralized mode (default: false)
#   WS               colcon overlay                         (default: ~/turtlebot3_ws)
# =============================================================================
set -euo pipefail

PLANNER="${PLANNER:-astar}"
USE_DIAGNOSTICS="${USE_DIAGNOSTICS:-false}"
USE_RAPL="${USE_RAPL:-false}"
WS="${WS:-$HOME/turtlebot3_ws}"

set +u
source /opt/ros/humble/setup.bash
[[ -f "$WS/install/setup.bash" ]] && source "$WS/install/setup.bash"
set -u

echo "[03] launching tb3_bringup bringup.launch.py  planner=$PLANNER"
echo "     use_diagnostics=$USE_DIAGNOSTICS  use_rapl=$USE_RAPL"
echo "     Ctrl+C to stop. From another host: scripts/smoke/04_drive_bridge.py --robot-ip <ip>"
echo
exec ros2 launch tb3_bringup bringup.launch.py \
    planner:="$PLANNER" \
    use_bridge:=true \
    use_diagnostics:="$USE_DIAGNOSTICS" \
    use_rapl:="$USE_RAPL"
