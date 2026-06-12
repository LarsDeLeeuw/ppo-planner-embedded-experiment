#!/usr/bin/env bash
# =============================================================================
# robot_bootstrap_check.sh — Pre-deploy sanity check for the TB3 robot SBC.
#
# Run this FROM THE BUILD VM (where this repo lives). It SSHes into the
# robot, runs a battery of PASS / WARN / FAIL checks on the apt/system-side
# prerequisites that ssh_deploy.sh + build_all.sh depend on, and prints a
# single `sudo apt install` line at the end for anything missing.
#
# Does NOT install anything. Does NOT touch source code. Read-only on the
# robot side.
#
# Use this on a fresh robot before the first ssh_deploy.sh, and any time you
# wonder "is this robot still set up correctly?" — it's idempotent.
#
# After deploy + build, scripts/smoke/00_preflight.sh (run on the robot)
# covers the same ground plus the workspace overlay.
#
# ENV:
#   ROBOT_HOST       SSH target          (default: ubuntu@turtlebot3.local)
#   WS               colcon workspace    (default: ~/turtlebot3_ws on the robot)
#   WIFI_INTERFACE   probed interface    (default: wlan0)
# =============================================================================
set -uo pipefail

ROBOT_HOST="${ROBOT_HOST:-ubuntu@turtlebot3.local}"
WS="${WS:-\$HOME/turtlebot3_ws}"
WIFI_INTERFACE="${WIFI_INTERFACE:-wlan0}"

# -- local reachability -------------------------------------------------------
echo "[bootstrap] target: $ROBOT_HOST"
if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$ROBOT_HOST" "true" >/dev/null 2>&1; then
    cat <<EOF >&2
ERROR: cannot SSH to $ROBOT_HOST.

  If the robot isn't reachable:
    - confirm the Pi is powered and on the same network
    - try \`ping turtlebot3.local\` (mDNS) or the static IP
    - override hostname/user:  ROBOT_HOST=user@host scripts/robot_bootstrap_check.sh

  If SSH key auth isn't set up yet (script needs passphrase-free auth):
    ssh-copy-id $ROBOT_HOST
EOF
    exit 1
fi

# -- remote check ------------------------------------------------------------
# Everything below runs on the robot. We ship one bash function via
# `declare -f` (same pattern as reset_robot.sh) so the checks live in one
# place and the local script stays free of quoting hell.

_remote_check() {
    set +e
    WS="${WS:-$HOME/turtlebot3_ws}"
    WIFI_INTERFACE="${WIFI_INTERFACE:-wlan0}"
    FAIL=0
    WARN=0
    MISSING_APT=()

    ok()   { printf "  [PASS] %s\n" "$*"; }
    warn() { printf "  [WARN] %s\n" "$*"; WARN=$((WARN + 1)); }
    bad()  { printf "  [FAIL] %s\n" "$*"; FAIL=$((FAIL + 1)); }
    miss() { MISSING_APT+=("$1"); }

    echo "=== host ==="
    echo "  user=$(id -un)  host=$(uname -n)  os=$(. /etc/os-release && echo "$PRETTY_NAME")  arch=$(uname -m)"

    echo
    echo "=== ROS 2 base ==="
    if [[ -f /opt/ros/humble/setup.bash ]]; then
        ok "ROS 2 Humble base at /opt/ros/humble"
        # shellcheck disable=SC1091
        source /opt/ros/humble/setup.bash >/dev/null 2>&1
    else
        bad "ROS 2 Humble base missing (expected /opt/ros/humble/setup.bash)"
        miss "ros-humble-ros-base"
    fi

    if command -v ros2 >/dev/null 2>&1; then
        ok "ros2 CLI on PATH"
    else
        bad "ros2 CLI not on PATH"
    fi

    if [[ "${TURTLEBOT3_MODEL:-}" == "burger" ]]; then
        ok "TURTLEBOT3_MODEL=burger"
    else
        warn "TURTLEBOT3_MODEL='${TURTLEBOT3_MODEL:-}' (should be 'burger' for stock TB3 bringup)"
    fi

    echo
    echo "=== build tools ==="
    if command -v colcon >/dev/null 2>&1; then
        ok "colcon on PATH"
    else
        bad "colcon missing — needed by build_all.sh"
        miss "python3-colcon-common-extensions"
    fi
    if command -v rsync >/dev/null 2>&1; then
        ok "rsync on PATH"
    else
        bad "rsync missing — needed by ssh_deploy.sh"
        miss "rsync"
    fi
    if command -v git >/dev/null 2>&1; then
        ok "git on PATH (diagnostics_node reads git SHA)"
    else
        warn "git missing — /diagnostics/session.git_sha_research_project will say 'unknown'"
        miss "git"
    fi

    echo
    echo "=== bag storage (mcap is MANDATED by tb3_bag_record.sh) ==="
    if command -v ros2 >/dev/null 2>&1 && ros2 bag record -s mcap --help >/dev/null 2>&1; then
        ok "ros2 bag mcap plugin available"
    else
        bad "ros2 bag mcap plugin missing"
        miss "ros-humble-rosbag2-storage-mcap"
    fi

    echo
    echo "=== Python deps ==="
    if python3 -c "import smbus2" >/dev/null 2>&1; then
        ok "python3-smbus2 importable"
    else
        bad "python3-smbus2 missing — INA219 driver won't load"
        miss "python3-smbus2"
    fi

    echo
    echo "=== I2C subsystem ==="
    I2C_DEV="/dev/i2c-1"
    if [[ -e "$I2C_DEV" ]]; then
        ok "$I2C_DEV present"
        if [[ -r "$I2C_DEV" && -w "$I2C_DEV" ]]; then
            ok "$I2C_DEV readable+writable by $(id -un)"
        else
            bad "$I2C_DEV not r/w by $(id -un) — sudo usermod -aG i2c $(id -un) && log out/in"
        fi
    else
        bad "$I2C_DEV missing — enable I2C: sudo raspi-config -> Interface Options -> I2C, then reboot"
    fi

    if command -v i2cdetect >/dev/null 2>&1; then
        ok "i2cdetect on PATH"
        if [[ -e "$I2C_DEV" ]]; then
            SCAN=$(i2cdetect -y 1 2>/dev/null | tr -s ' ' | tr '\n' ' ')
            # Expected addresses must match sensor_map.yaml:
            #   0x41 opencr, 0x44 sbc, 0x45 solar
            for addr in 41 44 45; do
                if grep -q " $addr " <<< "$SCAN"; then
                    ok "INA219 at 0x$addr on bus 1"
                else
                    warn "no I2C device at 0x$addr — sensor unwired or A0/A1 strap wrong (skip if not assembled yet)"
                fi
            done
        fi
    else
        bad "i2cdetect missing"
        miss "i2c-tools"
    fi

    # 400 kHz baud rate — WARN: sensors still work at 100 kHz, just slower.
    CONFIG_TXT=""
    for c in /boot/firmware/config.txt /boot/config.txt; do
        [[ -f "$c" ]] && { CONFIG_TXT="$c"; break; }
    done
    if [[ -n "$CONFIG_TXT" ]]; then
        if grep -Eq '^dtparam=i2c_arm_baudrate=400000' "$CONFIG_TXT"; then
            ok "$CONFIG_TXT has i2c_arm_baudrate=400000"
        else
            warn "$CONFIG_TXT lacks 'dtparam=i2c_arm_baudrate=400000' — sensors will run at 100 kHz"
        fi
    else
        warn "no Pi config.txt found — non-Pi host? skipping baud check"
    fi

    echo
    echo "=== chrony (per-host time sync, surfaces on /diagnostics/host) ==="
    if command -v chronyc >/dev/null 2>&1; then
        ok "chronyc on PATH"
        if chronyc tracking >/dev/null 2>&1; then
            OFFSET=$(chronyc tracking 2>/dev/null | awk -F': *' '/Last offset/ { print $2 }')
            ok "chronyd reachable (Last offset: ${OFFSET:-?})"
        else
            bad "chronyc cannot reach chronyd — sudo systemctl enable --now chrony"
        fi
    else
        bad "chronyc missing"
        miss "chrony"
    fi

    echo
    echo "=== WiFi probe (diagnostics_node reads RSSI) ==="
    if command -v iwconfig >/dev/null 2>&1; then
        ok "iwconfig on PATH"
        if iwconfig "$WIFI_INTERFACE" >/dev/null 2>&1; then
            SIG=$(iwconfig "$WIFI_INTERFACE" 2>/dev/null | awk -F'=' '/Signal level/ { print $NF }')
            ok "iface $WIFI_INTERFACE up (Signal: ${SIG:-unknown})"
        else
            warn "iface $WIFI_INTERFACE not found — override WIFI_INTERFACE=... if named differently"
        fi
    else
        bad "iwconfig missing"
        miss "wireless-tools"
    fi

    echo
    echo "=== thermal sysfs (ThermalStat publisher) ==="
    THERMAL=/sys/class/thermal/thermal_zone0/temp
    if [[ -r "$THERMAL" ]]; then
        TEMP_C=$(LC_ALL=C awk -v t="$(cat "$THERMAL")" 'BEGIN{ printf "%.1f", t/1000.0 }')
        ok "$THERMAL readable (cpu_temp=${TEMP_C}°C)"
    else
        bad "$THERMAL not readable — ThermalStat publisher will report NaN"
    fi
    if command -v vcgencmd >/dev/null 2>&1; then
        ok "vcgencmd on PATH"
    else
        warn "vcgencmd missing — ThermalStat.throttle_flags will always be 0"
    fi

    echo
    echo "=== colcon workspace ==="
    if [[ -d "$WS/src" ]]; then
        ok "$WS/src exists"
    else
        warn "$WS/src missing — run: mkdir -p $WS/src   (ssh_deploy.sh will not create the workspace root)"
    fi

    echo
    if [[ ${#MISSING_APT[@]} -gt 0 ]]; then
        echo "=== to fix missing packages ==="
        echo "  sudo apt update && sudo apt install -y ${MISSING_APT[*]}"
    fi

    echo
    if [[ "$FAIL" -eq 0 ]]; then
        echo "[bootstrap] OK ($WARN warning(s))"
        exit 0
    else
        echo "[bootstrap] FAILED — $FAIL FAIL, $WARN WARN"
        exit 1
    fi
}

# Forward WS/WIFI_INTERFACE into the remote shell, ship the function body
# via declare -f, then invoke it. `set -e` is intentionally off in the
# function so checks keep running past failures.
ssh "$ROBOT_HOST" bash -s <<EOF
WS="$WS"
WIFI_INTERFACE="$WIFI_INTERFACE"
$(declare -f _remote_check)
_remote_check
EOF
