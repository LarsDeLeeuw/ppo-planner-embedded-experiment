#!/usr/bin/env bash
# =============================================================================
# launch_session.sh — Orchestrate a centralized- or decentralized-mode session.
#
# RUNS ON: the build VM (this repo lives here).
#
# This is the single entry point for bringing up the full compute stack for an
# experiment. Bash owns SSH/lifecycle; `bringup.launch.py` owns node
# composition (Approach C in the design spec: matches the precedent set by
# ssh_deploy.sh + tb3_bag_record.sh, avoids ros2 launch's fragile remote-
# process supervision, keeps the launcher trivially auditable for the report).
#
# WHAT IT DOES
#   start: build the right packages on the right host(s), then launch
#          `ros2 launch tb3_bringup bringup.launch.py ...` on the chosen host
#          (Pi for decentralized, VM for centralized) via nohup + PID file +
#          combined log file. The always-on Pi sensor layer
#          (robot_sensors.launch.py) is assumed up in BOTH modes — centralized
#          mode still needs /scan + /odom from the Pi.
#   stop:  SIGINT the bringup PID (poll, then SIGKILL on grace exhaustion),
#          archive state, drop the active-session pointer.
#   status: report what's running where, plus the last log line as a smoke
#           signal that the stack hasn't died.
#
# DESIGN NOTES
#   - set -uo pipefail, NOT -e: keep going past partial failures so cleanup
#     paths always run and so we emit a structured state= record even on
#     failure (per spec §6).
#   - `set +u` around every `source /opt/ros/humble/setup.bash`. The ROS
#     setup scripts reference unset variables and trip -u.
#   - SSH alias `robot` (already in ~/.ssh/config). Allow ROBOT_HOST override
#     for ad-hoc retargeting (e.g. ubuntu@turtlebot3.local on a fresh image).
#   - nohup + pidfile pattern is lifted verbatim from tb3_bag_record.sh —
#     using the same convention so the bag recorder and the launcher can
#     coexist under one $RUN_ROOT/$RUN_ID/ tree.
#   - State file is flat key=value (sourceable by other bash scripts), under
#     $RUN_ROOT/active_session/state.env, with a per-run archive at
#     $RUN_ROOT/<run_id>/state.env. ros2 launch never sees this file.
#   - Remote logic is shipped via `declare -f` + heredoc (see
#     robot_bootstrap_check.sh, reset_robot.sh) so the remote bash stays
#     readable and quoting stays sane.
#
# CONFIGURATION (env vars):
#   ROBOT_HOST       SSH alias / target              (default: robot)
#   VM_WS            colcon workspace on the VM      (default: $HOME/turtlebot3_ws)
#   ROBOT_WS         colcon workspace on the robot   (default: ~/turtlebot3_ws)
#   REMOTE_DIR       repo root on the robot          (default: ~/ppo-tb3-research-project/robot)
#   RUN_ROOT         per-session artifact root       (default: /tmp/experiments)
#   BRIDGE_PORT      TCP bridge port for settle      (default: 9090)
#   SETTLE_TIMEOUT   bridge-ready timeout (s)        (default: 60)
#
# USAGE
#   start --mode {centralized|decentralized} --planner {ppo|astar|astar_shortest}
#         [--run-id ID] [--experiment-config PATH] [--skip-deploy] [--skip-build]
#         [--log-level info|debug|warn] [--settle-timeout 60] [--force]
#   stop  [--run-id ID] [--grace-s 10] [--force]
#   status [--run-id ID] [--json]
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

ROBOT_HOST="${ROBOT_HOST:-robot}"
VM_WS="${VM_WS:-$HOME/turtlebot3_ws}"
ROBOT_WS="${ROBOT_WS:-\$HOME/turtlebot3_ws}"   # remote-shell-expanded
REMOTE_DIR="${REMOTE_DIR:-\$HOME/ppo-tb3-research-project/robot}"
RUN_ROOT="${RUN_ROOT:-/tmp/experiments}"
BRIDGE_PORT="${BRIDGE_PORT:-9090}"
SETTLE_TIMEOUT="${SETTLE_TIMEOUT:-60}"

ACTIVE_DIR="$RUN_ROOT/active_session"
ACTIVE_STATE="$ACTIVE_DIR/state.env"

# Packages built on the VM (centralized mode + this launcher's build step).
# Note: the robot side is built by ssh_deploy.sh -> build_all.sh, which has
# its own canonical list. Don't duplicate the package list here for the
# robot path; defer to build_all.sh as the single source of truth.
VM_PACKAGES=(
    tb3_interfaces
    tb3_planner_common
    tb3_nav
    ros2_bridge
    ppo_planner
    astar_planner
    tb3_diagnostics
    tb3_rapl_sampler
    tb3_bringup
)

# -- helpers -----------------------------------------------------------------
die()  { echo "ERROR: $*" >&2; exit 1; }
warn() { echo "WARN:  $*" >&2; }
info() { echo "[launch] $*"; }

# All SSH calls go through _ssh so they share connect-timeout + batch-mode.
# Without ConnectTimeout, status/stop hangs indefinitely if the robot is
# unreachable; without BatchMode, a missing key falls through to an
# interactive password prompt that never gets one in an unattended context.
_ssh() {
    ssh -o BatchMode=yes -o ConnectTimeout=5 "$@"
}

# Safe loader for state.env files. Replaces `source "$state_file"` — the
# latter is arbitrary-code-execution against an untrusted file (e.g.,
# experiment_config values containing '<' would crash bash even on benign
# content; values with $(...) or ` would execute).
#
# Emits single-quote-escaped KEY='VAL' lines for whitelisted keys; caller
# uses `eval "$(_load_state ...)"` so assignments happen in the caller's
# scope (works through `local` shadowing, unlike `declare -g`). Single
# quotes inside values are escaped via the standard '\'' trick. Inside
# single-quoted strings bash does no expansion, so $(...) / backticks /
# $VAR / etc. are inert.
_VALID_STATE_KEYS=":run_id:mode:planner:host:log_level:bringup_log:bringup_pid:bringup_pidfile:started_epoch:stopped_epoch:git_sha:bridge_predict_service:bridge_endpoint:experiment_config:use_diagnostics:use_rapl:state:missing:hint:cleanup_hint:"
_load_state() {
    local file="$1"
    [[ -f "$file" ]] || return 1
    local key value escaped
    while IFS='=' read -r key value || [[ -n "$key" ]]; do
        [[ -z "$key" || "$key" =~ ^# ]] && continue
        case "$_VALID_STATE_KEYS" in
            *":$key:"*) ;;
            *)          continue ;;
        esac
        escaped="${value//\'/\'\\\'\'}"
        printf "%s='%s'\n" "$key" "$escaped"
    done < "$file"
}

# Emit a key=value pair to stdout AND append to a state file (if set).
emit_kv() {
    local key="$1" val="$2"
    printf '%s=%s\n' "$key" "$val"
    if [[ -n "${STATE_FILE:-}" ]]; then
        printf '%s=%s\n' "$key" "$val" >> "$STATE_FILE"
    fi
}

# Host the bringup runs on (NOT the same as the sensor layer host).
host_for_mode() {
    case "$1" in
        centralized)   echo vm ;;
        decentralized) echo robot ;;
        *) die "unknown mode '$1' (expected centralized|decentralized)" ;;
    esac
}

# Short single-letter codes for the auto-generated run-id suffix.
planner_short() {
    case "$1" in
        ppo)            echo p ;;
        astar)          echo a ;;
        astar_shortest) echo s ;;
        *) die "unknown planner '$1'" ;;
    esac
}

# Run a command on a given host. host=vm => local bash; host=robot => ssh.
# Used for low-level utilities (kill -0, mkdir, tail, etc.). For the bringup
# spawn we use a dedicated helper that constructs the nohup wrapper inline.
run_on_host() {
    local host="$1"; shift
    local cmd="$*"
    if [[ "$host" == "vm" ]]; then
        bash -lc "$cmd"
    else
        _ssh "$ROBOT_HOST" "$cmd"
    fi
}

# Is the named PID alive on a given host? Returns 0 (alive) or 1 (dead).
pid_alive_on() {
    local host="$1" pid="$2"
    [[ -n "$pid" && "$pid" != "0" ]] || return 1
    if [[ "$host" == "vm" ]]; then
        kill -0 "$pid" 2>/dev/null
    else
        _ssh "$ROBOT_HOST" "kill -0 $pid 2>/dev/null"
    fi
}

# SIGINT → poll up to grace_s seconds in 100ms ticks → SIGKILL if still alive.
# Same recipe as tb3_bag_record.sh stop, just parametrised by host.
# Special case: grace_s=0 means force-stop (SIGKILL immediately, no SIGINT).
# Returns 0 if the pid is confirmed dead, 1 if it's still alive after the
# kill attempts (e.g., persistent SSH failure to the host).
kill_pid_on() {
    local host="$1" pid="$2" grace_s="${3:-10}"
    [[ -n "$pid" && "$pid" != "0" ]] || return 0
    if (( grace_s == 0 )); then
        # Force mode — skip SIGINT, go straight to SIGKILL.
        if [[ "$host" == "vm" ]]; then
            kill -9 "$pid" 2>/dev/null || true
        else
            _ssh "$ROBOT_HOST" "kill -9 $pid 2>/dev/null || true"
        fi
        sleep 0.2
        pid_alive_on "$host" "$pid" && return 1
        return 0
    fi
    local ticks=$(( grace_s * 10 ))
    if [[ "$host" == "vm" ]]; then
        kill -SIGINT "$pid" 2>/dev/null || true
        for _ in $(seq 1 "$ticks"); do
            kill -0 "$pid" 2>/dev/null || return 0
            sleep 0.1
        done
        kill -9 "$pid" 2>/dev/null || true
    else
        _ssh "$ROBOT_HOST" "kill -SIGINT $pid 2>/dev/null || true"
        for _ in $(seq 1 "$ticks"); do
            _ssh "$ROBOT_HOST" "kill -0 $pid 2>/dev/null" || return 0
            sleep 0.1
        done
        _ssh "$ROBOT_HOST" "kill -9 $pid 2>/dev/null || true"
    fi
    sleep 0.2
    pid_alive_on "$host" "$pid" && return 1
    return 0
}

# Poll the bridge TCP port until it accepts a connection, up to timeout_s.
# host=vm uses bash's /dev/tcp; host=robot uses the same trick over SSH.
probe_bridge() {
    local host="$1" port="$2" timeout_s="$3"
    local deadline=$(( $(date +%s) + timeout_s ))
    local target_host="127.0.0.1"
    [[ "$host" == "robot" ]] && target_host="127.0.0.1"  # probe from the host itself
    while (( $(date +%s) < deadline )); do
        if [[ "$host" == "vm" ]]; then
            if timeout 1 bash -c "</dev/tcp/$target_host/$port" 2>/dev/null; then
                return 0
            fi
        else
            if _ssh "$ROBOT_HOST" "timeout 1 bash -c '</dev/tcp/$target_host/$port'" 2>/dev/null; then
                return 0
            fi
        fi
        sleep 1
    done
    return 1
}

# Preflight: SSH key auth must work, /scan + /odom must exist on the robot
# (so robot_sensors.launch.py is up — required in BOTH modes).
preflight_ssh() {
    _ssh "$ROBOT_HOST" "true" >/dev/null 2>&1
}

# Source ROS + workspace overlays on the robot, then list topics. Grep
# explicitly anchors to ^/odom$ / ^/imu$ to avoid matching /odom_combined etc.
# Note: /scan is NOT required — this experiment uses overhead-camera pose via
# the bridge, not a laser scanner. The LDS is allowed to be physically absent.
preflight_topics_on_robot() {
    _ssh "$ROBOT_HOST" bash <<'EOF'
set +u
source /opt/ros/humble/setup.bash >/dev/null 2>&1
[[ -f $HOME/turtlebot3_ws/install/setup.bash ]] && source $HOME/turtlebot3_ws/install/setup.bash >/dev/null 2>&1
set -u
ros2 topic list 2>/dev/null | grep -E '^(/odom|/imu)$' | sort -u
EOF
}

# -- spawn the bringup on the chosen host ------------------------------------
# Same nohup + pidfile pattern as tb3_bag_record.sh. We construct the remote
# (or local) command string and ship it; the PID file is written on the
# remote side so a flaky SSH connection doesn't lose the PID.
spawn_bringup() {
    local host="$1" run_dir="$2" launch_args="$3"
    local log_file="$run_dir/bringup.log"
    local pid_file="$run_dir/bringup.pid"

    # Inner command runs on the chosen host. Uses ROBOT_WS / VM_WS as
    # appropriate; both are sourced under `set +u` because the ROS setup
    # scripts reference unbound vars.
    local ws
    if [[ "$host" == "vm" ]]; then
        ws="$VM_WS"
    else
        # ROBOT_WS is intentionally left as a literal '$HOME/...' string so
        # the remote shell expands $HOME for that user.
        ws="$ROBOT_WS"
    fi

    # NOTE: launch_args is passed unquoted on the ros2 launch command line —
    # callers must construct it as plain `key:=value key:=value ...` tokens
    # with no shell metacharacters. All values we plug in come from validated
    # CLI args (mode/planner) or absolute paths under our control.
    local inner_cmd
    inner_cmd="set +u; \
source /opt/ros/humble/setup.bash; \
[[ -f $ws/install/setup.bash ]] && source $ws/install/setup.bash; \
set -u; \
mkdir -p $run_dir; \
nohup ros2 launch tb3_bringup bringup.launch.py $launch_args \
    > $log_file 2>&1 & \
echo \$! > $pid_file"

    if [[ "$host" == "vm" ]]; then
        bash -lc "$inner_cmd"
    else
        _ssh "$ROBOT_HOST" "bash -lc '$inner_cmd'"
    fi
}

# -- subcommand: start --------------------------------------------------------
cmd_start() {
    local mode="" planner="" run_id="" experiment_config=""
    local skip_deploy=0 skip_build=0 force=0
    local log_level="info"
    local settle_timeout="$SETTLE_TIMEOUT"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --mode)               mode="$2"; shift 2 ;;
            --planner)            planner="$2"; shift 2 ;;
            --run-id)             run_id="$2"; shift 2 ;;
            --experiment-config)  experiment_config="$2"; shift 2 ;;
            --skip-deploy)        skip_deploy=1; shift ;;
            --skip-build)         skip_build=1; shift ;;
            --log-level)          log_level="$2"; shift 2 ;;
            --settle-timeout)     settle_timeout="$2"; shift 2 ;;
            --force)              force=1; shift ;;
            *) die "unknown start arg: $1" ;;
        esac
    done

    [[ -n "$mode" ]]    || die "--mode required (centralized|decentralized)"
    [[ -n "$planner" ]] || die "--planner required (ppo|astar|astar_shortest)"
    # Validate in the main shell — host_for_mode also dies on invalid input,
    # but it's called in a subshell ($(...)) where die wouldn't propagate,
    # so we'd otherwise continue with an empty host string. Same care for
    # planner since planner_short is also called via $(...).
    case "$mode" in centralized|decentralized) ;; *) die "bad --mode '$mode' (expected centralized|decentralized)" ;; esac
    case "$planner" in ppo|astar|astar_shortest) ;; *) die "bad --planner '$planner'" ;; esac
    case "$log_level" in
        debug|info|warn|error|fatal) ;;
        *) die "bad --log-level '$log_level' (expected debug|info|warn|error|fatal)" ;;
    esac

    local host; host="$(host_for_mode "$mode")"

    # Concurrency lock — two terminals running `start` simultaneously both
    # pass the already-running guard below before either writes ACTIVE_STATE,
    # then both spawn ros2 launch and the second clobbers the state file.
    # flock makes only one cmd_start body run at a time on this VM.
    mkdir -p "$ACTIVE_DIR"
    exec 9>"$ACTIVE_DIR/.start.lock"
    if ! flock -n 9; then
        emit_kv state already_starting
        emit_kv hint  "another launch_session.sh start is in progress on this VM"
        exit 2
    fi

    # Already-running guard. If --force, stop the existing session first.
    if [[ -f "$ACTIVE_STATE" ]]; then
        local prev_pid prev_host
        prev_pid=$(awk -F= '$1=="bringup_pid"{print $2}' "$ACTIVE_STATE")
        prev_host=$(awk -F= '$1=="host"{print $2}' "$ACTIVE_STATE")
        if [[ -n "$prev_pid" ]] && pid_alive_on "$prev_host" "$prev_pid"; then
            if (( force )); then
                info "active session detected (pid $prev_pid on $prev_host); --force => stopping it first"
                cmd_stop --force || true
            else
                local prev_run_id
                prev_run_id=$(awk -F= '$1=="run_id"{print $2}' "$ACTIVE_STATE")
                emit_kv state already_running
                emit_kv run_id "$prev_run_id"
                emit_kv hint "scripts/launch_session.sh stop  (or pass --force)"
                exit 2
            fi
        else
            # Stale active dir — sweep it.
            info "stale active_session/ (pid not alive); clearing"
            rm -f "$ACTIVE_STATE"
        fi
    fi

    # Generate run-id if not supplied.
    if [[ -z "$run_id" ]]; then
        local mc="${mode:0:1}" ps; ps="$(planner_short "$planner")"
        run_id="$(date -u +%Y%m%dT%H%M%SZ)-${mc}${ps}"
    fi

    local run_dir="$RUN_ROOT/$run_id"
    mkdir -p "$run_dir" "$ACTIVE_DIR"

    # Wire emit_kv to the per-run state file from here on.
    STATE_FILE="$run_dir/state.env"
    : > "$STATE_FILE"

    emit_kv run_id           "$run_id"
    emit_kv mode             "$mode"
    emit_kv planner          "$planner"
    emit_kv host             "$host"
    emit_kv log_level        "$log_level"
    emit_kv bringup_log      "$run_dir/bringup.log"
    emit_kv bringup_pidfile  "$run_dir/bringup.pid"
    emit_kv started_epoch    "$(date +%s)"
    emit_kv git_sha          "$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"

    # Derive predict_service (mirror PLANNER_TO_PREDICT_SERVICE in bringup.launch.py).
    local predict_service
    case "$planner" in
        ppo)            predict_service='/ppo_planner_node/predict_action' ;;
        astar)          predict_service='/astar_planner_node/predict_action' ;;
        astar_shortest) predict_service='/astar_shortest_node/predict_action' ;;
    esac
    emit_kv bridge_predict_service "$predict_service"

    # use_diagnostics / use_rapl flip with mode (centralized => VM does extra
    # per-host telemetry; decentralized => the Pi's always-on diagnostics
    # already covers it, so leave them off here to avoid duplicate nodes).
    local use_diagnostics use_rapl
    if [[ "$mode" == "centralized" ]]; then
        use_diagnostics=true; use_rapl=true
    else
        use_diagnostics=false; use_rapl=false
    fi
    emit_kv use_diagnostics "$use_diagnostics"
    emit_kv use_rapl        "$use_rapl"

    # SSH preflight — needed in BOTH modes (centralized still relies on
    # /scan + /odom from the Pi's always-on sensor layer).
    info "preflight: ssh $ROBOT_HOST"
    if ! preflight_ssh; then
        emit_kv state failed_ssh
        emit_kv hint  "scripts/setup_ssh_check.sh ; ssh-copy-id $ROBOT_HOST"
        exit 4
    fi

    # Sensor-topic preflight. Must see /odom AND /imu — both come from
    # turtlebot3_node (OpenCR) and are what grid_nav_node needs.
    info "preflight: /odom + /imu on $ROBOT_HOST"
    local topics
    topics=$(preflight_topics_on_robot || true)
    if ! grep -q '^/odom$' <<<"$topics" || ! grep -q '^/imu$' <<<"$topics"; then
        emit_kv state failed_prereq
        emit_kv missing "/odom|/imu"
        emit_kv hint    "start robot_sensors.launch.py on the Pi (always-on layer)"
        exit 5
    fi

    # ---- deploy + build ----------------------------------------------------
    if (( ! skip_build )); then
        if [[ "$host" == "robot" ]]; then
            if (( skip_deploy )); then
                info "skip-deploy: rebuilding on robot via build_all.sh only"
                _ssh "$ROBOT_HOST" "bash $REMOTE_DIR/scripts/build_all.sh"
            else
                info "deploying + building on robot (ssh_deploy.sh)"
                bash "$SCRIPT_DIR/ssh_deploy.sh"
            fi
            if (( $? != 0 )); then
                emit_kv state failed_build
                emit_kv hint  "see ssh_deploy.sh / build_all.sh output above"
                exit 6
            fi
        else
            info "building on VM (colcon, packages: ${VM_PACKAGES[*]})"
            if [[ ! -d "$VM_WS/src" ]]; then
                emit_kv state failed_overlay
                emit_kv hint  "VM workspace $VM_WS/src missing"
                exit 6
            fi
            # Ensure symlinks (build_all.sh would do this, but we want to be
            # explicit about which subset we build on the VM).
            for pkg in "${VM_PACKAGES[@]}"; do
                local src="$REPO_DIR/src/$pkg"
                local dst="$VM_WS/src/$pkg"
                if [[ ! -d "$src" ]]; then
                    warn "package $pkg not found in repo ($src) — skipping symlink"
                    continue
                fi
                [[ -L "$dst" || -d "$dst" ]] || ln -s "$src" "$dst"
            done
            (
                set +u
                # shellcheck disable=SC1091
                source /opt/ros/humble/setup.bash
                set -u
                cd "$VM_WS" || exit 1
                colcon build --symlink-install \
                    --packages-select "${VM_PACKAGES[@]}"
            )
            if (( $? != 0 )); then
                emit_kv state failed_build
                emit_kv hint  "see colcon output above"
                exit 6
            fi
        fi
    else
        info "skip-build: assuming overlays on both hosts are current"
    fi

    # ---- assemble launch args ---------------------------------------------
    local launch_args="planner:=$planner use_bridge:=true"
    launch_args+=" use_diagnostics:=$use_diagnostics use_rapl:=$use_rapl"
    launch_args+=" log_level:=$log_level"
    if [[ -n "$experiment_config" ]]; then
        launch_args+=" experiment_config:=$experiment_config"
        emit_kv experiment_config "$experiment_config"
    else
        emit_kv experiment_config "<default from tb3_bringup share>"
    fi

    # bridge_endpoint convenience (consumers grep state.env for it).
    if [[ "$host" == "robot" ]]; then
        emit_kv bridge_endpoint "$ROBOT_HOST:$BRIDGE_PORT"
    else
        emit_kv bridge_endpoint "127.0.0.1:$BRIDGE_PORT"
    fi

    # ---- spawn -------------------------------------------------------------
    info "spawning bringup on $host: ros2 launch tb3_bringup bringup.launch.py $launch_args"
    if ! spawn_bringup "$host" "$run_dir" "$launch_args"; then
        emit_kv state failed_spawn
        exit 3
    fi

    # Read back the PID. The remote write happened before SSH returned, so
    # the pidfile is present — but defensive read anyway in case of races.
    local pid=""
    for _ in 1 2 3 4 5; do
        if [[ "$host" == "vm" ]]; then
            [[ -f "$run_dir/bringup.pid" ]] && pid=$(cat "$run_dir/bringup.pid")
        else
            pid=$(_ssh "$ROBOT_HOST" "cat $run_dir/bringup.pid 2>/dev/null" || true)
        fi
        [[ -n "$pid" ]] && break
        sleep 0.5
    done
    if [[ -z "$pid" ]]; then
        emit_kv state failed_partial
        emit_kv cleanup_hint "ssh $ROBOT_HOST pkill -f 'ros2 launch tb3_bringup'"
        exit 3
    fi
    emit_kv bringup_pid "$pid"

    # ---- settle (probe the bridge port) -----------------------------------
    info "waiting up to ${settle_timeout}s for bridge :$BRIDGE_PORT on $host"
    if probe_bridge "$host" "$BRIDGE_PORT" "$settle_timeout"; then
        emit_kv state running
        info "session up: run_id=$run_id pid=$pid host=$host"
    else
        warn "bridge port $BRIDGE_PORT did not open within ${settle_timeout}s — killing bringup"
        if ! kill_pid_on "$host" "$pid" 10; then
            warn "first kill attempt failed (SSH flake?); retrying after 2s"
            sleep 2
            if ! kill_pid_on "$host" "$pid" 10; then
                emit_kv state failed_settle
                emit_kv cleanup_hint "ssh $ROBOT_HOST 'pkill -f \"ros2 launch tb3_bringup\"' (pid $pid stayed alive)"
                exit 7
            fi
        fi
        emit_kv state failed_settle
        emit_kv hint  "tail $run_dir/bringup.log for the root cause"
        exit 7
    fi

    # Mirror state to the active-session pointer (last write wins; this is
    # the file `stop` and `status` consult by default).
    cp "$STATE_FILE" "$ACTIVE_STATE"
}

# -- subcommand: stop ---------------------------------------------------------
cmd_stop() {
    local run_id="" grace_s=10 force=0
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --run-id)  run_id="$2"; shift 2 ;;
            --grace-s) grace_s="$2"; shift 2 ;;
            --force)   force=1; shift ;;
            *) die "unknown stop arg: $1" ;;
        esac
    done

    # Pick the state file. Explicit --run-id wins; otherwise the active ptr.
    local state_file
    if [[ -n "$run_id" ]]; then
        state_file="$RUN_ROOT/$run_id/state.env"
        if [[ ! -f "$state_file" ]]; then
            STATE_FILE=""
            emit_kv state no_active_session
            emit_kv run_id "$run_id"
            exit 0
        fi
    else
        if [[ ! -f "$ACTIVE_STATE" ]]; then
            STATE_FILE=""
            emit_kv state no_active_session
            exit 0
        fi
        state_file="$ACTIVE_STATE"
    fi

    # Load the state file via the whitelisted parser (NOT `source` — values
    # like '<default from tb3_bringup share>' contain shell metacharacters
    # that would crash or silently execute on `source`).
    eval "$(_load_state "$state_file")"

    # From here on, append any new keys to the per-run state file (NOT the
    # active pointer, which we'll delete at the end of the happy path).
    STATE_FILE="$RUN_ROOT/$run_id/state.env"
    # If we sourced the active ptr, run_id was set from inside the file.
    [[ -n "${run_id:-}" ]] || die "state file missing run_id"

    local was_alive=0
    if pid_alive_on "${host:-vm}" "${bringup_pid:-}"; then
        was_alive=1
        # --force => grace_s=0 => SIGKILL only (skip the SIGINT grace).
        local effective_grace="$grace_s"
        (( force )) && effective_grace=0
        info "stopping bringup pid=$bringup_pid on $host (grace ${effective_grace}s$( ((force)) && echo ', --force/SIGKILL'))"
        if ! kill_pid_on "${host}" "$bringup_pid" "$effective_grace"; then
            warn "kill did not confirm pid $bringup_pid dead — see cleanup_hint"
            emit_kv cleanup_hint "ssh $ROBOT_HOST 'pkill -9 -f \"ros2 launch tb3_bringup\"'"
        fi
    else
        info "bringup pid=$bringup_pid was already dead"
    fi

    # Always clear the active pointer (so the next start isn't blocked).
    rm -f "$ACTIVE_STATE"

    if (( was_alive )); then
        emit_kv state stopped
    else
        emit_kv state stopped_was_dead
    fi
    emit_kv stopped_epoch "$(date +%s)"
}

# -- subcommand: status -------------------------------------------------------
cmd_status() {
    local run_id="" want_json=0
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --run-id) run_id="$2"; shift 2 ;;
            --json)   want_json=1; shift ;;
            *) die "unknown status arg: $1" ;;
        esac
    done

    local state_file
    if [[ -n "$run_id" ]]; then
        state_file="$RUN_ROOT/$run_id/state.env"
    else
        state_file="$ACTIVE_STATE"
    fi

    if [[ ! -f "$state_file" ]]; then
        if (( want_json )); then
            echo '{"state":"no_active_session"}'
        else
            echo "state=no_active_session"
        fi
        exit 0
    fi

    eval "$(_load_state "$state_file")"

    # Liveness check. If the recorded state was running but the PID died,
    # report state=dead (don't mutate the file — that's `stop`'s job).
    local live="false"
    if pid_alive_on "${host:-vm}" "${bringup_pid:-}"; then
        live="true"
    fi
    local reported_state="${state:-unknown}"
    if [[ "$reported_state" == "running" && "$live" == "false" ]]; then
        reported_state="dead"
    fi

    # Tail the bringup log (last line) as a smoke signal.
    local last_log=""
    if [[ -n "${bringup_log:-}" ]]; then
        if [[ "${host:-vm}" == "vm" ]]; then
            last_log=$(tail -n 1 "$bringup_log" 2>/dev/null || true)
        else
            last_log=$(_ssh "$ROBOT_HOST" "tail -n 1 $bringup_log 2>/dev/null" || true)
        fi
    fi

    if (( want_json )); then
        # Minimal hand-rolled JSON — flat key=value, no escaping needed for
        # any value we control (mode/planner/host/run_id are alnum + dashes;
        # paths have no quotes/backslashes). The log-line is the one wild
        # field; quote-escape it.
        local esc_log; esc_log="${last_log//\"/\\\"}"
        printf '{"run_id":"%s","mode":"%s","planner":"%s","host":"%s","bringup_pid":"%s","state":"%s","alive":%s,"last_log":"%s"}\n' \
            "${run_id:-}" "${mode:-}" "${planner:-}" "${host:-}" "${bringup_pid:-}" \
            "$reported_state" "$live" "$esc_log"
    else
        echo "run_id=${run_id:-}"
        echo "mode=${mode:-}"
        echo "planner=${planner:-}"
        echo "host=${host:-}"
        echo "bringup_pid=${bringup_pid:-}"
        echo "state=$reported_state"
        echo "alive=$live"
        echo "bridge_predict_service=${bridge_predict_service:-}"
        echo "bringup_log=${bringup_log:-}"
        if [[ -n "$last_log" ]]; then
            echo "last_log=$last_log"
        fi
    fi
}

# -- subcommand: tail ---------------------------------------------------------
# Convenience: tail the bringup log of the active (or named) session.
cmd_tail() {
    local target="bringup" n=200 follow=0
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -n) n="$2"; shift 2 ;;
            -f) follow=1; shift ;;
            bringup|all) target="$1"; shift ;;
            *) die "unknown tail arg: $1" ;;
        esac
    done
    [[ -f "$ACTIVE_STATE" ]] || die "no active session"
    eval "$(_load_state "$ACTIVE_STATE")"
    local flags="-n $n"
    (( follow )) && flags="$flags -f"
    if [[ "${host:-vm}" == "vm" ]]; then
        # shellcheck disable=SC2086
        tail $flags "$bringup_log"
    else
        # shellcheck disable=SC2029
        _ssh "$ROBOT_HOST" "tail $flags $bringup_log"
    fi
}

# -- subcommand: clean --------------------------------------------------------
cmd_clean() {
    local all=0
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --all) all=1; shift ;;
            *) die "unknown clean arg: $1" ;;
        esac
    done
    if [[ -f "$ACTIVE_STATE" ]]; then
        warn "active session present — refusing to clean. Run 'stop' first."
        exit 2
    fi
    if (( all )); then
        info "removing $RUN_ROOT entirely"
        rm -rf "$RUN_ROOT"
    else
        info "removing $ACTIVE_DIR"
        rm -rf "$ACTIVE_DIR"
    fi
}

# -- dispatch ----------------------------------------------------------------
CMD="${1:-}"; shift || true
case "$CMD" in
    start)  cmd_start  "$@" ;;
    stop)   cmd_stop   "$@" ;;
    status) cmd_status "$@" ;;
    tail)   cmd_tail   "$@" ;;
    clean)  cmd_clean  "$@" ;;
    ""|-h|--help|help)
        cat <<EOF
usage: $0 <subcommand> [args]

subcommands:
  start --mode {centralized|decentralized} --planner {ppo|astar|astar_shortest}
        [--run-id ID] [--experiment-config PATH] [--skip-deploy] [--skip-build]
        [--log-level info|debug|warn] [--settle-timeout 60] [--force]
  stop  [--run-id ID] [--grace-s 10] [--force]
  status [--run-id ID] [--json]
  tail   [bringup|all] [-n N] [-f]
  clean  [--all]

env vars (defaults shown):
  ROBOT_HOST=robot   VM_WS=\$HOME/turtlebot3_ws   ROBOT_WS=\$HOME/turtlebot3_ws
  REMOTE_DIR=\$HOME/ppo-tb3-research-project/robot   RUN_ROOT=/tmp/experiments
  BRIDGE_PORT=9090   SETTLE_TIMEOUT=60
EOF
        ;;
    *) die "unknown subcommand: $CMD (try '$0 help')" ;;
esac
