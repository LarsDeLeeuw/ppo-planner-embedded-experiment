#!/usr/bin/env bash
# =============================================================================
# 00_preflight.sh — Verify the Pi is set up for the experiment stack.
#
# Run this on the robot SBC BEFORE the rest of the smoke scripts. It checks:
#   - ROS 2 Humble is on PATH
#   - The colcon overlay is built (every package needed by the launch files)
#   - The mcap bag storage plugin is installed (mandated by tb3_bag_record.sh)
#   - python3-smbus2 imports (INA219 driver dep)
#   - The I2C kernel module is loaded and /dev/i2c-1 is readable by this user
#   - i2cdetect sees the three INA219 sensors at 0x41 / 0x44 / 0x45
#   - chrony is reachable via `chronyc tracking` (diagnostics node)
#   - WiFi interface is present (diagnostics node, RSSI probe)
#   - /sys/class/thermal/thermal_zone0/temp readable (thermal publisher)
#   - I2C bus is pinned to 400 kHz in /boot/firmware/config.txt (WARN if not)
#
# Each line prints PASS / WARN / FAIL. Exits 0 if no FAILs (WARNs allowed),
# 1 otherwise.
#
# ENV:
#   WS                colcon overlay        (default: ~/turtlebot3_ws)
#   WIFI_INTERFACE    interface to probe    (default: wlan0)
#   I2C_BUS           bus number            (default: 1)
# =============================================================================
set -uo pipefail   # NOTE: no -e — we want to keep going past failed checks.

WS="${WS:-$HOME/turtlebot3_ws}"
WIFI_INTERFACE="${WIFI_INTERFACE:-wlan0}"
I2C_BUS="${I2C_BUS:-1}"

# Expected INA219 addresses (decimal -> hex), keep in lockstep with
# src/tb3_power_sensor/config/sensor_map.yaml.
declare -A EXPECTED_SENSORS=(
    [opencr]="41"
    [sbc]="44"
    [solar]="45"
)

FAIL=0
WARN=0

ok()   { printf "  [PASS] %s\n" "$*"; }
warn() { printf "  [WARN] %s\n" "$*"; WARN=$((WARN + 1)); }
bad()  { printf "  [FAIL] %s\n" "$*"; FAIL=$((FAIL + 1)); }

echo "=== ROS 2 base ==="
# ROS setup.bash internally references unset shell variables, which would
# trip `set -u` and kill the script before any other check runs. Toggle
# nounset off for the duration of each source.
if [[ -f /opt/ros/humble/setup.bash ]]; then
    ok "ROS 2 Humble base at /opt/ros/humble"
    set +u
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
    set -u
else
    bad "ROS 2 Humble base not found (expected /opt/ros/humble/setup.bash)"
fi

if command -v ros2 >/dev/null 2>&1; then
    ok "ros2 CLI on PATH ($(command -v ros2))"
else
    bad "ros2 CLI not on PATH after sourcing the base"
fi

echo
echo "=== workspace overlay ==="
if [[ -f "$WS/install/setup.bash" ]]; then
    ok "overlay at $WS/install/setup.bash"
    set +u
    # shellcheck disable=SC1091
    source "$WS/install/setup.bash"
    set -u
else
    bad "overlay not built — run scripts/build_all.sh"
fi

# Every package the smoke launch files invoke must be on AMENT_PREFIX_PATH.
for pkg in tb3_interfaces tb3_power_sensor tb3_diagnostics tb3_nav \
           tb3_bringup ros2_bridge ppo_planner astar_planner; do
    if ros2 pkg prefix "$pkg" >/dev/null 2>&1; then
        ok "package $pkg installed"
    else
        bad "package $pkg NOT installed — rebuild with scripts/build_all.sh"
    fi
done

echo
echo "=== bag storage ==="
# `ros2 bag record -s mcap --help` exits non-zero if the plugin is missing.
if ros2 bag record -s mcap --help >/dev/null 2>&1; then
    ok "mcap storage plugin available"
else
    bad "mcap plugin missing — apt install ros-humble-rosbag2-storage-mcap"
fi

echo
echo "=== Python deps ==="
if python3 -c "import smbus2" >/dev/null 2>&1; then
    ok "python3-smbus2 importable"
else
    bad "smbus2 missing — apt install python3-smbus2"
fi

echo
echo "=== I2C subsystem ==="
I2C_DEV="/dev/i2c-$I2C_BUS"
if [[ -e "$I2C_DEV" ]]; then
    ok "$I2C_DEV present"
    if [[ -r "$I2C_DEV" && -w "$I2C_DEV" ]]; then
        ok "$I2C_DEV readable+writable by $USER"
    else
        bad "$I2C_DEV not r/w by $USER — usermod -aG i2c $USER && log out/in"
    fi
else
    bad "$I2C_DEV missing — enable I2C via raspi-config and reboot"
fi

if command -v i2cdetect >/dev/null 2>&1; then
    ok "i2cdetect on PATH"
    SCAN=$(i2cdetect -y "$I2C_BUS" 2>/dev/null | tr -s ' ' | tr '\n' ' ')
    for sid in "${!EXPECTED_SENSORS[@]}"; do
        addr="${EXPECTED_SENSORS[$sid]}"
        if grep -q " $addr " <<< "$SCAN"; then
            ok "INA219 '$sid' at 0x$addr present on bus $I2C_BUS"
        else
            bad "INA219 '$sid' at 0x$addr NOT on bus $I2C_BUS — check wiring + A0/A1 straps"
        fi
    done
else
    bad "i2cdetect missing — apt install i2c-tools"
fi

# Warn-only: 400 kHz boot config (data still works at 100 kHz, just slower).
CONFIG_TXT=""
for candidate in /boot/firmware/config.txt /boot/config.txt; do
    [[ -f "$candidate" ]] && { CONFIG_TXT="$candidate"; break; }
done
if [[ -n "$CONFIG_TXT" ]]; then
    if grep -Eq '^dtparam=i2c_arm_baudrate=400000' "$CONFIG_TXT"; then
        ok "$CONFIG_TXT has i2c_arm_baudrate=400000"
    else
        warn "$CONFIG_TXT lacks 'dtparam=i2c_arm_baudrate=400000' — sensors will run at 100 kHz"
    fi
else
    warn "no /boot[/firmware]/config.txt found — not a Pi? skipping baud-rate check"
fi

echo
echo "=== chrony ==="
if command -v chronyc >/dev/null 2>&1; then
    ok "chronyc on PATH"
    if chronyc tracking >/dev/null 2>&1; then
        OFFSET=$(chronyc tracking 2>/dev/null | awk -F': *' '/Last offset/ { print $2 }')
        ok "chronyd reachable (Last offset: ${OFFSET:-?})"
    else
        bad "chronyc cannot reach chronyd — systemctl status chrony"
    fi
else
    bad "chronyc missing — apt install chrony"
fi

echo
echo "=== WiFi probe ==="
if command -v iwconfig >/dev/null 2>&1; then
    ok "iwconfig on PATH"
    if iwconfig "$WIFI_INTERFACE" >/dev/null 2>&1; then
        SIGNAL=$(iwconfig "$WIFI_INTERFACE" 2>/dev/null \
                 | awk -F'=' '/Signal level/ { print $NF }')
        ok "iface $WIFI_INTERFACE up (Signal: ${SIGNAL:-unknown})"
    else
        warn "iface $WIFI_INTERFACE missing — override with WIFI_INTERFACE=... if named differently"
    fi
else
    bad "iwconfig missing — apt install wireless-tools"
fi

echo
echo "=== thermal sysfs ==="
THERMAL=/sys/class/thermal/thermal_zone0/temp
if [[ -r "$THERMAL" ]]; then
    TEMP_C=$(LC_ALL=C awk -v t="$(cat "$THERMAL")" 'BEGIN{ printf "%.1f", t/1000.0 }')
    ok "$THERMAL readable (cpu_temp=${TEMP_C}°C)"
else
    bad "$THERMAL not readable — ThermalStat publisher will report NaN"
fi
# vcgencmd is Pi-specific; warn if missing (throttle_flags will be 0).
if command -v vcgencmd >/dev/null 2>&1; then
    ok "vcgencmd on PATH"
else
    warn "vcgencmd missing — ThermalStat.throttle_flags will always be 0"
fi

echo
if [[ "$FAIL" -eq 0 ]]; then
    echo "[00] preflight OK ($WARN warning(s))"
    exit 0
else
    echo "[00] preflight FAILED — $FAIL FAIL, $WARN WARN; fix the FAIL lines before continuing"
    exit 1
fi
