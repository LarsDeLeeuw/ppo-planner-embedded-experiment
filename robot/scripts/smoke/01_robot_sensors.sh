#!/usr/bin/env bash
# =============================================================================
# 01_robot_sensors.sh — Always-on sensor layer (foreground; Ctrl+C to stop).
#
# Launches the Pi-side stack that runs in BOTH experiment modes:
#   - tb3_power_sensor: INA219 driver for solar / sbc / opencr @ 100 Hz +
#                       /sbc/thermal @ 1 Hz.
#   - tb3_diagnostics:  /diagnostics/host (chrony/WiFi/RAPL) @ 1 Hz +
#                       /diagnostics/session (latched git SHA + experiment.yaml).
#
# Run this on the robot SBC (Pi). Leave it running, then in a second SSH
# session run 02_check_sensor_topics.sh to verify rates and signs.
#
# ENV:
#   WS              colcon workspace overlay   (default: ~/turtlebot3_ws)
#   WIFI_INTERFACE  iface for WiFi RSSI probe  (default: wlan0)
# =============================================================================
set -euo pipefail

WS="${WS:-$HOME/turtlebot3_ws}"
WIFI_INTERFACE="${WIFI_INTERFACE:-wlan0}"

set +u
source /opt/ros/humble/setup.bash
[[ -f "$WS/install/setup.bash" ]] && source "$WS/install/setup.bash"
set -u

echo "[01] launching tb3_bringup robot_sensors.launch.py (wifi_interface=$WIFI_INTERFACE)"
echo "     Ctrl+C to stop. In another terminal, run scripts/smoke/02_check_sensor_topics.sh"
echo
exec ros2 launch tb3_bringup robot_sensors.launch.py \
    wifi_interface:="$WIFI_INTERFACE"
