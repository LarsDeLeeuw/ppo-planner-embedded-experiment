# ros2_bridge Integration Guidance

Guidance for the worker implementing the external TCP client that talks to `ros2_bridge`. The bridge is a newline-delimited JSON over TCP gateway into the robot's ROS2 graph: the client sends pose corrections, navigation goals, and planner predict requests; the bridge translates them into ROS2 service calls and action goals, and streams results and feedback back over the same TCP socket.

This document is the full wire-protocol spec. A reader should be able to implement a working client from this file alone, without reading `bridge_node.py`.

## Constraints this guidance optimizes for

1. **Stable JSON wire contract.** Field names and types are load-bearing. The bridge does not tolerate renamed fields and silently drops unparseable input.
2. **Planner-agnostic client.** The `predict` request/response is identical whether PPO or A* is wired behind the bridge. The client must not try to detect which planner is running.
3. **Async-first.** Client requests may be answered in any order relative to unsolicited server pushes (`goal_feedback`, `nav_pose`). The client must be event-driven, not request-response locked.
4. **Single-client socket.** The bridge accepts one TCP client at a time. Reconnect cancels the previous client's active goal.

## Wire protocol basics

- **Transport:** TCP, default port `9090` (bridge param `tcp_port`).
- **Framing:** UTF-8 JSON objects, one per line, terminated by a single `\n`. Multiple messages may arrive in one TCP packet or be split across packets — the client must buffer on `\n` ([src/ros2_bridge/ros2_bridge/tcp_server.py:148-161](../robot/src/ros2_bridge/ros2_bridge/tcp_server.py#L148-L161)).
- **Connection model:** one client at a time. A new connection evicts the previous one ([tcp_server.py:116-122](../robot/src/ros2_bridge/ros2_bridge/tcp_server.py#L116-L122)). On disconnect — whether intentional or network loss — the bridge cancels any active navigation goal so the robot stops moving unsupervised ([bridge_node.py:203-215](../robot/src/ros2_bridge/ros2_bridge/bridge_node.py#L203-L215)).
- **No handshake, no heartbeat.** The client starts sending messages immediately after `connect()`. A `ping` → `pong` round-trip is available if the client needs liveness checks.
- **Malformed input is silent.** Non-JSON lines are dropped with a warning in bridge logs ([tcp_server.py:154-158](../robot/src/ros2_bridge/ros2_bridge/tcp_server.py#L154-L158)); unknown `type` values are dropped with a debug log ([bridge_node.py:99](../robot/src/ros2_bridge/ros2_bridge/bridge_node.py#L99)). **The client never receives an error response for bad input.** It must validate its own outgoing messages.
- **Latency floor.** Inbound messages are drained on a ROS2 timer at `poll_hz` (default 20 Hz), so per-message bridge-side latency is up to `1/poll_hz` ≈ 50 ms ([bridge_node.py:74,84-99](../robot/src/ros2_bridge/ros2_bridge/bridge_node.py#L74)). Outbound messages (sends) are emitted immediately.
- **Ordering.** Messages from one poll cycle are dispatched in receive order. Responses from ROS2 come back asynchronously and may interleave with unsolicited pushes — correlate on `type` and your own outstanding-request state.

## Coordinate and encoding conventions

All conventions below are canonical to the PredictAction service contract ([src/tb3_interfaces/srv/PredictAction.srv](../robot/src/tb3_interfaces/srv/PredictAction.srv)) and apply equally on the TCP side.

- **Grid frame:** origin `(0, 0)` is bottom-left. `+X` = East (column index). `+Y` = North (row index).
- **Heading:** radians, counter-clockwise from `+X`.
- **Obstacle map cells:** `0.0` free, `1.0` occupied. Sent over JSON as a 2D array of numbers; integers `0`/`1` are accepted and cast to `float64` server-side ([bridge_node.py:262](../robot/src/ros2_bridge/ros2_bridge/bridge_node.py#L262)).
- **Energy map cells:** floats in `[0.0, 1.0]`.
- **Grid shape:** the 2D arrays are row-major, `rows × cols`, with `obstacle_map[r][c]`. Rows and cols are derived server-side from the array dimensions ([bridge_node.py:260-261](../robot/src/ros2_bridge/ros2_bridge/bridge_node.py#L260-L261)).
- **Positions in `predict`:** integer grid cell indices.
- **Positions in `goal`:** floats; the `MoveToGrid` action accepts sub-cell targets ([src/tb3_interfaces/action/MoveToGrid.action](../robot/src/tb3_interfaces/action/MoveToGrid.action)). Integer JSON values are accepted and coerced.

### Action / direction table

Action codes `0..7` returned in `predict_result`, from [src/tb3_planner_common/tb3_planner_common/directions.py](../robot/src/tb3_planner_common/tb3_planner_common/directions.py):

```
action  (direction_x, direction_y)  label
  0     (-1,  0)                    West
  1     ( 1,  0)                    East
  2     ( 0, -1)                    South
  3     ( 0,  1)                    North
  4     (-1, -1)                    SW
  5     (-1,  1)                    NW
  6     ( 1, -1)                    SE
  7     ( 1,  1)                    NE
```

`direction_x` and `direction_y` are the components of the unit step in the grid frame: `next_cell = robot_pos + (direction_x, direction_y)`. The `predict_result` carries both `action` and `direction`; they are redundant by design so the client may use whichever is more ergonomic.

## Client → bridge messages

Each message is a JSON object with a required `type` field. Dispatch is on `type` alone; unknown types are dropped silently ([bridge_node.py:84-99](../robot/src/ros2_bridge/ros2_bridge/bridge_node.py#L84-L99)).

### `pose` — pose correction (fire-and-forget)

Inject an externally-measured pose into the nav node (e.g. after a fiducial or QR-tracker fix).

```json
{"type": "pose", "x": 3.2, "y": 1.0, "heading": 1.5708}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `type` | string | ✓ | Must be `"pose"`. |
| `x` | number | ✓ | Grid X coordinate. Sub-cell precision allowed. Missing → treated as `0.0`. |
| `y` | number | ✓ | Grid Y coordinate. Sub-cell precision allowed. Missing → treated as `0.0`. |
| `heading` | number | ✓ | Radians, CCW from `+X`. Missing → treated as `0.0`. |

- **ROS-side effect:** calls `SetGridPose` service on `pose_service` ([bridge_node.py:103-116](../robot/src/ros2_bridge/ros2_bridge/bridge_node.py#L103-L116)).
- **Response:** **none.** The call is fire-and-forget; failures are logged on the bridge side only ([bridge_node.py:118-122](../robot/src/ros2_bridge/ros2_bridge/bridge_node.py#L118-L122)).
- **If the service is unavailable:** the bridge logs a warning and skips the request without notifying the client ([bridge_node.py:105-107](../robot/src/ros2_bridge/ros2_bridge/bridge_node.py#L105-L107)).
- **Confirming it landed:** wait for the next `nav_pose` push and check that the nav node's self-estimate now reflects your correction.

### `goal` — start navigation to a grid cell

Commands the nav node to drive to a target cell. Preempts any currently active goal.

```json
{"type": "goal", "target_x": 7.0, "target_y": 4.0}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `type` | string | ✓ | Must be `"goal"`. |
| `target_x` | number | ✓ | Grid X of the target. Missing → treated as `0.0`. |
| `target_y` | number | ✓ | Grid Y of the target. Missing → treated as `0.0`. |

- **ROS-side effect:** sends a `MoveToGrid` action goal on `goal_action` ([bridge_node.py:124-153](../robot/src/ros2_bridge/ros2_bridge/bridge_node.py#L124-L153)). Any in-flight goal is cancelled first.
- **Responses:**
  - **Server not ready →** immediate `goal_result` with `success: false`, `message: "Action server not ready"` ([bridge_node.py:126-133](../robot/src/ros2_bridge/ros2_bridge/bridge_node.py#L126-L133)).
  - **Accepted by nav node →** a stream of unsolicited `goal_feedback` messages at the nav node's publish rate, terminated by a single `goal_result`.
  - **Rejected by nav node →** `goal_result` with `success: false`, `message: "Goal rejected"` ([bridge_node.py:155-164](../robot/src/ros2_bridge/ros2_bridge/bridge_node.py#L155-L164)).

### `cancel_goal` — abort the current navigation

```json
{"type": "cancel_goal"}
```

- **No fields.**
- **ROS-side effect:** cancels the active action goal handle if one exists; no-op otherwise ([bridge_node.py:193-201](../robot/src/ros2_bridge/ros2_bridge/bridge_node.py#L193-L201)).
- **Response:** no direct acknowledgement. A `goal_result` will still arrive for the in-flight goal once the nav node finishes honoring the cancel — typically with `success: false` and a message indicating cancellation.

### `predict` — planner inference

Ask the active planner (PPO or A*) for the next one-step action.

```json
{
  "type": "predict",
  "sequence": 42,
  "obstacle_map": [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
  "energy_map":   [[0.1, 0.2, 0.0], [0.3, 0.4, 0.1], [0.0, 0.5, 0.2]],
  "robot_pos": [0, 0],
  "goal_pos":  [2, 2]
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `type` | string | ✓ | Must be `"predict"`. |
| `sequence` | uint32 | – | Per-call counter for the timing decomposition. Additive: omit it (defaults to 0) and everything still works; the bridge silently ignores unknown fields, so a client can begin stamping it before the rest of the chain is wired. Echoed back on `predict_result` and stamped on the `/planner/metrics` and `/bridge/predict_timing` topics so analysis can join inference/bridge/TCP timings per call. |
| `obstacle_map` | number[][] | ✓ | 2D row-major. `0`/`1` or `0.0`/`1.0`. |
| `energy_map` | number[][] | ✓ | Same shape as `obstacle_map`. Floats in `[0.0, 1.0]`. |
| `robot_pos` | [int, int] | ✓ | `[x, y]` grid indices. |
| `goal_pos` | [int, int] | ✓ | `[x, y]` grid indices. |

- **ROS-side effect:** calls `PredictAction` service on whichever planner is wired via `predict_service`. The bridge passes both maps through as `GridMap` messages, sends `robot_x/y`, `goal_x/y` as `int32`, and forwards `sequence`. It stamps `t_bridge_recv_ns` when it begins handling the `predict` (after any preceding `pose` in the same poll batch has been dispatched), and publishes a `BridgePredictTiming` on `/bridge/predict_timing` once the response is sent.
- **Responses:**
  - **Success →** `predict_result` with the action and direction vector.
  - **Service unavailable →** `error` with `message: "PredictAction service not ready"` ([bridge_node.py:219-222](../robot/src/ros2_bridge/ros2_bridge/bridge_node.py#L219-L222)). Happens during PPO model load.
  - **Malformed request →** `error` with `message: "Bad predict request: <detail>"` ([bridge_node.py:234-236](../robot/src/ros2_bridge/ros2_bridge/bridge_node.py#L234-L236)). Missing keys, wrong types, or wrong array shapes.
  - **Planner returned success=false →** `error` with `message: <planner-provided-message>` ([bridge_node.py:250-251](../robot/src/ros2_bridge/ros2_bridge/bridge_node.py#L250-L251)). Typical causes: no path exists (A*), robot already at goal (A*), shape mismatch with PPO's trained grid size.
  - **Planner raised an exception →** `error` with the stringified exception ([bridge_node.py:252-254](../robot/src/ros2_bridge/ros2_bridge/bridge_node.py#L252-L254)).

#### Planner semantics the client must respect

- **Planner choice is a deploy-time decision.** The bringup launch arg (`planner:=ppo|astar|astar_shortest`) decides which node the bridge's `predict_service` points at. The client cannot and should not attempt to detect it. Both planners honor the same request shape and return the same response shape.
- **PPO has non-trivial startup time.** The PPO node blocks on model load at init; during that window the bridge will reply `error: PredictAction service not ready`. The client should tolerate this with retry/backoff rather than treating it as fatal.
- **A\* is instant to start.** No model to load.
- **Grid dimensions matter for PPO.** The PPO node enforces fixed `grid_size_x`/`grid_size_y` parameters (default 10×10); requests with different dimensions return `success=false`. A\* accepts any grid dimensions. If the client targets PPO, match the configured grid size; if it targets A\*, any consistent shape is fine.
- **Maps are per-request.** The planner holds no state between calls. Every `predict` must carry a fresh `obstacle_map` and `energy_map` along with current `robot_pos`.
- **Inference is synchronous from the client's perspective** (one `predict` → one `predict_result` or `error`), but the bridge calls the service async internally, so interleaving multiple `predict`s — or mixing them with `goal`s — does not deadlock.

### `ping` — liveness probe

```json
{"type": "ping"}
```

- **No fields.**
- **Response:** immediate `{"type": "pong"}` ([bridge_node.py:96-97](../robot/src/ros2_bridge/ros2_bridge/bridge_node.py#L96-L97)).
- Useful for confirming TCP is alive and the bridge's poll loop is servicing the inbox. Does not confirm any downstream ROS service is healthy.

### `experiment_event` — run-lifecycle marker (fire-and-forget)

Emitted by the orchestrator to bound and annotate an experiment run. The bridge forwards it onto the ROS topic `/experiment/events` (`tb3_interfaces/ExperimentEvent`), stamping `header.stamp` on the bridge host's (chrony-synced) clock — analysis uses `run_start`/`run_end` to bound the run window in bag time.

```json
{
  "type": "experiment_event",
  "event_type": "run_start",
  "run_id": "20260524-153012-ppo-map3-decentralized",
  "payload": {"mode": "decentralized", "planner": "ppo", "map_id": "map3", "goal_cell": [3, 4]}
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `type` | string | ✓ | Must be `"experiment_event"`. |
| `event_type` | string | ✓ | `"run_start"` \| `"run_end"` \| `"marker"` \| `"bag_capped"`. |
| `run_id` | string | ✓ | Identifies the run; written into the forwarded topic. |
| `payload` | object | – | Free-form JSON; serialized verbatim into `payload_json`. Analysis reads `event_type`/`run_id` and treats the payload as opaque. |

- **No response.** Fire-and-forget, like `pose`.
- The bridge does not interpret the payload; it only forwards. The orchestrator owns run semantics.

## Bridge → client messages

Responses to client requests AND unsolicited server pushes share the same socket. Clients must dispatch on `type`.

### `predict_result` — planner response

```json
{"type": "predict_result", "sequence": 42, "action": 3, "direction": [0, 1]}
```

| Field | Type | Notes |
|---|---|---|
| `sequence` | uint32 | Echo of the `sequence` from the originating `predict` (0 if the request omitted it). |
| `action` | int | `0..7`. See action table above. |
| `direction` | [int, int] | `[direction_x, direction_y]`, each in `{-1, 0, 1}`. |

Correlates with the originating `predict` via `sequence`. If the client stamps a unique `sequence` per request it can pair responses unambiguously even with multiple `predict`s in flight; if it leaves `sequence` at 0, responses arrive in ROS service completion order (normally FIFO but not guaranteed) — gate to one outstanding predict at a time in that case.

### `goal_feedback` — unsolicited, while a goal is active

```json
{"type": "goal_feedback", "phase": "driving", "distance": 2.41, "heading_error": 0.08}
```

| Field | Type | Notes |
|---|---|---|
| `phase` | string | Nav node's state name. See `phase` field in [MoveToGrid.action](../robot/src/tb3_interfaces/action/MoveToGrid.action). The bridge forwards it as-is; treat unknown values as informational. |
| `distance` | float | Metres to target remaining (`distance_remaining_m` from the action feedback). |
| `heading_error` | float | Radians remaining between current heading and goal heading. |

Emitted at [bridge_node.py:171-178](../robot/src/ros2_bridge/ros2_bridge/bridge_node.py#L171-L178). Frequency matches the nav node's feedback publish rate. Only fires while a goal is active — not a general heartbeat.

### `goal_result` — terminal result of a goal

```json
{"type": "goal_result", "success": true, "message": "Reached target"}
```

| Field | Type | Notes |
|---|---|---|
| `success` | bool | `true` only on a clean completion. |
| `message` | string | Human-readable; the nav node's `message` field on the action result. |

Emitted at three sites:
- Action server not ready — `{"success": false, "message": "Action server not ready"}` ([bridge_node.py:128-132](../robot/src/ros2_bridge/ros2_bridge/bridge_node.py#L128-L132)).
- Goal rejected — `{"success": false, "message": "Goal rejected"}` ([bridge_node.py:159-163](../robot/src/ros2_bridge/ros2_bridge/bridge_node.py#L159-L163)).
- Action finished — passes through the nav node's `success`/`message` ([bridge_node.py:184-188](../robot/src/ros2_bridge/ros2_bridge/bridge_node.py#L184-L188)).

Exactly one `goal_result` is emitted per `goal` request.

### `nav_pose` — unsolicited, pose stream

```json
{"type": "nav_pose", "x": 3.21, "y": 1.05, "heading": 1.57}
```

| Field | Type | Notes |
|---|---|---|
| `x` | float | Nav node's self-estimated grid X. |
| `y` | float | Nav node's self-estimated grid Y. |
| `heading` | float | Radians, CCW from `+X`. |

Emitted at [bridge_node.py:267-274](../robot/src/ros2_bridge/ros2_bridge/bridge_node.py#L267-L274). Forwarded 1:1 from the nav node's `grid_pose_topic` — whatever rate that topic publishes at.

### `error` — failure response

```json
{"type": "error", "message": "PredictAction service not ready"}
```

| Field | Type | Notes |
|---|---|---|
| `message` | string | Free-form description. |

Currently only emitted on `predict`-related failures (see `predict` section above for the five code paths). **There is no correlation id** — the client must infer context from timing and its own outstanding-request state. Other failures surface via `goal_result.success=false` or are logged silently on the bridge.

### `pong` — ping response

```json
{"type": "pong"}
```

Emitted at [bridge_node.py:97](../robot/src/ros2_bridge/ros2_bridge/bridge_node.py#L97) immediately on receipt of `ping`.

## Recommended client lifecycle

A typical session:

```
connect
  → ping                          ← pong                     (sanity check)
  → pose (initial localization)                              (fire-and-forget)
  loop:
    → predict(maps, robot_pos, goal_pos)
    ← predict_result(action, direction)                       (or error → handle)
    → goal(target = robot_pos + direction)
    ← goal_feedback ...                                       (stream)
    ← goal_result(success, message)
  on external abort:
    → cancel_goal
    ← goal_result(success=false, ...)                         (eventually)
disconnect
```

Notes on the pattern:

- The bridge does not drive motion from `predict`. A `predict_result` is an **advisory** — the robot only moves when the client issues a `goal`.
- Interleaving `predict` and `goal` is safe; they target different ROS endpoints.
- Reconnecting the TCP socket always cancels the active goal. An intentional reconnect is a valid emergency-stop if `cancel_goal` is inconvenient.
- `nav_pose` pushes will arrive concurrently with everything else. Dispatch on `type`, don't assume one message is in flight at a time.

## Failure modes and recovery

| Symptom | Likely cause | Recommended client action |
|---|---|---|
| `error: "PredictAction service not ready"` | PPO still loading, or `predict_service` param points at a nonexistent node. | Retry with backoff (e.g. 500 ms → 2 s → 5 s). After several failures, surface to operator. |
| `error: "Bad predict request: ..."` | Client-side bug: missing field, wrong type, empty grid. | Don't retry. Fix the request shape. |
| `error: <planner message>` | Planner returned `success=false` — e.g. no path (A\*), already at goal (A\*), shape mismatch (PPO). | Try a different goal / pose; don't retry the same inputs. |
| `goal_result.success=false, message="Action server not ready"` | Nav node down or not yet started. | Retry after a short delay; `ping` to confirm bridge is still reachable. |
| `goal_result.success=false, message="Goal rejected"` | Nav node refused the target (out-of-bounds, inside obstacle, etc.). | Don't retry the same target. |
| No response within N seconds | Request lost to silent drop, or downstream ROS component hung. | There is no bridge-side response timeout. The client owns timeouts; on timeout, either retry (idempotent for `predict`) or `cancel_goal` + reconnect. |
| Sudden TCP disconnect from server side | Another client connected, or bridge shut down. | Reconnect and re-establish state. The robot will have been commanded to stop. |
| No `error` at all for a malformed message | Bridge drops silently. | The client must validate its own outgoing JSON. Do not rely on server rejection. |

## Configuration reference (deployer-side, for awareness)

Bridge parameters the client author may need to discuss with whoever launches the system. Defaults from [src/ros2_bridge/config/default_params.yaml](../robot/src/ros2_bridge/config/default_params.yaml):

| Param | Default | Meaning |
|---|---|---|
| `tcp_port` | `9090` | Port the client connects to. |
| `poll_hz` | `20.0` | Inbound drain rate; sets the client-to-bridge latency floor. |
| `pose_service` | `/grid_nav_node/set_grid_pose` | Where `pose` messages are forwarded. |
| `goal_action` | `move_to_grid` | Where `goal` messages are forwarded. |
| `grid_pose_topic` | `/grid_nav_node/grid_pose` | Source of `nav_pose` stream. |
| `predict_service` | `/ppo_planner_node/predict_action` | Which planner backs `predict`. Swapping planners = changing this one string in the experiment YAML. |

The client is oblivious to all of these except by their observable effects.

## What NOT to do

- **Don't open multiple TCP connections expecting parallel throughput.** The bridge evicts the old client when a new one connects. If you need parallelism, do it client-side and serialize onto the single socket.
- **Don't treat `goal_feedback` as a heartbeat.** It only fires while a goal is active. Use `ping`/`pong` for liveness.
- **Don't rely on the bridge to validate coordinates, grid shapes, or JSON structure.** Most bad input is silently dropped. Validate outgoing messages client-side.
- **Don't couple to which planner is running.** The response shape is identical across PPO and A\*. Writing `if using_ppo` branches will rot as planners are added or swapped.
- **Don't persistently cache `predict_result` across robot motion.** The planner's inputs include the current pose; a cached action from a prior cell is wrong for the current cell.
- **Don't assume `pose` is acknowledged.** It's fire-and-forget. If you need confirmation, wait for the next `nav_pose` push and check that the nav node's self-estimate reflects the correction.
- **Don't interleave `goal`s expecting independent tracking.** A second `goal` preempts the first; you will only see a `goal_result` for the most recently accepted one (the preempted goal's result is discarded).
- **Don't block on a single outstanding `predict`.** The bridge will honor multiple in-flight requests, but since there is no correlation id, the simplest correct client gates to one predict at a time.

## Related files

- [src/ros2_bridge/ros2_bridge/bridge_node.py](../robot/src/ros2_bridge/ros2_bridge/bridge_node.py) — source of truth for every message handler and every outbound payload.
- [src/ros2_bridge/ros2_bridge/tcp_server.py](../robot/src/ros2_bridge/ros2_bridge/tcp_server.py) — framing, single-client semantics, disconnect handling.
- [src/ros2_bridge/config/default_params.yaml](../robot/src/ros2_bridge/config/default_params.yaml) — bridge-side parameter defaults.
- [src/tb3_interfaces/srv/PredictAction.srv](../robot/src/tb3_interfaces/srv/PredictAction.srv) — ROS service schema the `predict` / `predict_result` pair wraps.
- [src/tb3_interfaces/msg/GridMap.msg](../robot/src/tb3_interfaces/msg/GridMap.msg) — GridMap shape (row-major `float64[]` + dims).
- [src/tb3_interfaces/srv/SetGridPose.srv](../robot/src/tb3_interfaces/srv/SetGridPose.srv) — ROS service that `pose` messages trigger.
- [src/tb3_interfaces/action/MoveToGrid.action](../robot/src/tb3_interfaces/action/MoveToGrid.action) — ROS action that `goal` / `goal_feedback` / `goal_result` wrap.
- [src/tb3_planner_common/tb3_planner_common/directions.py](../robot/src/tb3_planner_common/tb3_planner_common/directions.py) — canonical action ↔ direction table.
- [src/tb3_bringup/launch/bringup.launch.py](../robot/src/tb3_bringup/launch/bringup.launch.py), [src/tb3_bringup/config/experiment.yaml](../robot/src/tb3_bringup/config/experiment.yaml) — where the `planner` launch arg and `predict_service` override are wired.
