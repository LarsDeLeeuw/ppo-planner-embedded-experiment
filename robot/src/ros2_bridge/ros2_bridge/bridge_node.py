"""
bridge_node.py — Single TCP gateway for the ROS2 navigation stack.

Receives pose corrections, navigation goals, and PPO predict requests
from a TCP client (e.g. qr-tracker) and forwards them as ROS2 service
calls / action goals. Sends results and feedback back over TCP.
"""

from __future__ import annotations

import functools
import json
import threading
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile

from tb3_interfaces.action import MoveToGrid
from tb3_interfaces.msg import BridgePredictTiming, ExperimentEvent, GridMap, GridPose
from tb3_interfaces.srv import PredictAction, SetGridPose

from ros2_bridge.tcp_server import TcpServer


class BridgeNode(Node):
    """Single TCP gateway translating JSON messages into ROS2 calls."""

    def __init__(self) -> None:
        super().__init__("bridge_node")

        # -- ROS2 parameters --------------------------------------------------
        self.declare_parameter("tcp_port", 9090)
        self.declare_parameter("pose_service", "/grid_nav_node/set_grid_pose")
        self.declare_parameter("goal_action", "move_to_grid")
        self.declare_parameter("grid_pose_topic", "/grid_nav_node/grid_pose")
        self.declare_parameter("predict_service", "/ppo_planner_node/predict_action")
        self.declare_parameter("poll_hz", 20.0)

        tcp_port = self.get_parameter("tcp_port").value
        pose_svc = self.get_parameter("pose_service").value
        goal_act = self.get_parameter("goal_action").value
        pose_topic = self.get_parameter("grid_pose_topic").value
        predict_svc = self.get_parameter("predict_service").value
        poll_hz = self.get_parameter("poll_hz").value

        # -- Goal state (init BEFORE TcpServer.start so the disconnect
        #    callback, which runs on a network thread, cannot observe
        #    half-initialised state) ------------------------------------------
        self._goal_lock = threading.Lock()
        self._active_goal: ClientGoalHandle | None = None

        # -- ROS2 clients -----------------------------------------------------
        self._pose_client = self.create_client(SetGridPose, pose_svc)
        self._predict_client = self.create_client(PredictAction, predict_svc)
        self._goal_client = ActionClient(self, MoveToGrid, goal_act)

        # -- TCP server -------------------------------------------------------
        # Cancel any active goal if the TCP client disconnects unexpectedly,
        # so the robot doesn't keep driving without supervision.
        self._tcp = TcpServer(
            port=tcp_port,
            on_disconnect=self._handle_client_disconnect,
        )
        self._tcp.start()

        # Subscribe to nav node's self-estimate (for optional forwarding)
        self._pose_sub = self.create_subscription(
            GridPose, pose_topic, self._on_grid_pose, 10,
        )

        # -- Experiment instrumentation publishers ----------------------------
        # /experiment/events : run-lifecycle markers forwarded from the
        #   orchestrator (run_start/run_end/marker/bag_capped) — bounds the
        #   analysis run window on the bridge-host clock.
        #
        #   TRANSIENT_LOCAL (latched), mirroring /diagnostics/session: the
        #   orchestrator emits run_start at the very start of the run, but
        #   `ros2 bag record` takes ~1.5-2 s to discover and subscribe. With the
        #   default VOLATILE QoS that early run_start is dropped (the bag isn't
        #   subscribed yet) while run_end — published seconds later — survives.
        #   Latching retains the last few events so a late-joining bag still
        #   captures run_start on subscribe. depth=10 covers a run's events
        #   (run_start + markers + run_end) with margin; because the bridge node
        #   outlives a single run, the latched buffer may carry a prior run's
        #   tail into the next bag, but every event carries run_id so analysis
        #   filters cleanly.
        # /bridge/predict_timing : per-predict bridge-side timing for the
        #   latency decomposition (single-host monotonic delta).
        events_qos = QoSProfile(
            depth=10,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self._events_pub = self.create_publisher(
            ExperimentEvent, "/experiment/events", events_qos)
        self._predict_timing_pub = self.create_publisher(
            BridgePredictTiming, "/bridge/predict_timing", 50)

        # -- Poll timer -------------------------------------------------------
        period = 1.0 / poll_hz
        self._poll_timer = self.create_timer(period, self._poll_tcp)

        self.get_logger().info(
            f"Bridge node started — TCP:{tcp_port}  "
            f"pose_svc:{pose_svc}  goal_action:{goal_act}  "
            f"predict_svc:{predict_svc}"
        )

    # -- TCP polling ----------------------------------------------------------

    def _poll_tcp(self) -> None:
        """Drain TCP inbox and dispatch messages."""
        for msg in self._tcp.poll_messages():
            msg_type = msg.get("type", "")
            if msg_type == "pose":
                self._handle_pose(msg)
            elif msg_type == "goal":
                self._handle_goal(msg)
            elif msg_type == "cancel_goal":
                self._handle_cancel()
            elif msg_type == "predict":
                self._handle_predict(msg)
            elif msg_type == "experiment_event":
                self._handle_experiment_event(msg)
            elif msg_type == "ping":
                self._tcp.send({"type": "pong"})
            else:
                self.get_logger().debug(f"Unknown message type: {msg_type}")

    # -- handlers -------------------------------------------------------------

    def _handle_pose(self, msg: dict) -> None:
        """Forward a pose correction to SetGridPose service."""
        if not self._pose_client.service_is_ready():
            self.get_logger().warning("SetGridPose service not ready, skipping")
            return

        req = SetGridPose.Request()
        req.pose = GridPose()
        req.pose.x = float(msg.get("x", 0.0))
        req.pose.y = float(msg.get("y", 0.0))
        req.pose.heading = float(msg.get("heading", 0.0))

        future = self._pose_client.call_async(req)
        future.add_done_callback(self._on_pose_response)

    def _on_pose_response(self, future) -> None:
        try:
            future.result()
        except Exception as e:
            self.get_logger().warning(f"SetGridPose call failed: {e}")

    def _handle_goal(self, msg: dict) -> None:
        """Send a MoveToGrid action goal, preempting any active goal."""
        if not self._goal_client.server_is_ready():
            self.get_logger().warning("MoveToGrid action not ready, skipping")
            self._tcp.send({
                "type": "goal_result",
                "success": False,
                "message": "Action server not ready",
            })
            return

        # Cancel active goal (nav node also preempts, but be explicit)
        with self._goal_lock:
            if self._active_goal is not None:
                self.get_logger().info("Cancelling previous goal")
                self._active_goal.cancel_goal_async()
                self._active_goal = None

        goal = MoveToGrid.Goal()
        goal.target_x = float(msg.get("target_x", 0.0))
        goal.target_y = float(msg.get("target_y", 0.0))

        self.get_logger().info(
            f"Sending goal: ({goal.target_x}, {goal.target_y})"
        )

        send_future = self._goal_client.send_goal_async(
            goal, feedback_callback=self._on_goal_feedback,
        )
        send_future.add_done_callback(self._on_goal_accepted)

    def _on_goal_accepted(self, future) -> None:
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warning("Goal rejected by nav node")
            self._tcp.send({
                "type": "goal_result",
                "success": False,
                "message": "Goal rejected",
            })
            return

        with self._goal_lock:
            self._active_goal = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_goal_result)

    def _on_goal_feedback(self, feedback_msg) -> None:
        fb = feedback_msg.feedback
        self._tcp.send({
            "type": "goal_feedback",
            "phase": fb.phase,
            "distance": float(fb.distance_remaining_m),
            "heading_error": float(fb.heading_error_rad),
        })

    def _on_goal_result(self, future) -> None:
        result = future.result().result
        with self._goal_lock:
            self._active_goal = None
        self._tcp.send({
            "type": "goal_result",
            "success": bool(result.success),
            "message": str(result.message),
        })
        self.get_logger().info(
            f"Goal result: success={result.success} — {result.message}"
        )

    def _handle_cancel(self) -> None:
        """Cancel the active goal."""
        with self._goal_lock:
            if self._active_goal is not None:
                self.get_logger().info("Cancelling goal (client request)")
                self._active_goal.cancel_goal_async()
                self._active_goal = None
            else:
                self.get_logger().debug("Cancel requested but no active goal")

    def _handle_client_disconnect(self) -> None:
        """Fired by TcpServer when the TCP client drops unexpectedly.

        Cancels any active goal so the robot stops moving when it loses
        its external supervisor. Runs on the TcpServer's network thread.
        """
        with self._goal_lock:
            if self._active_goal is not None:
                self.get_logger().warning(
                    "TCP client disconnected; cancelling active goal"
                )
                self._active_goal.cancel_goal_async()
                self._active_goal = None

    def _handle_predict(self, msg: dict) -> None:
        """Forward a predict request to the PredictAction service."""
        if not self._predict_client.service_is_ready():
            self.get_logger().warning("PredictAction service not ready, skipping")
            self._tcp.send({"type": "error", "message": "PredictAction service not ready"})
            return

        # Stamp recv at the moment we begin handling the predict. Any preceding
        # `pose` in this poll batch was already dispatched (call_async) above, so
        # its service-call cost is OUT of this bridge delta — it lands in the
        # orchestrator-side TCP slice instead (see plan §11.1).
        t_recv_ns = time.monotonic_ns()
        sequence = int(msg.get("sequence", 0))

        try:
            req = PredictAction.Request()
            req.obstacle_map = self._list_to_gridmap(msg["obstacle_map"])
            req.energy_map = self._list_to_gridmap(msg["energy_map"])
            robot_pos = msg["robot_pos"]
            goal_pos = msg["goal_pos"]
            req.robot_x = int(robot_pos[0])
            req.robot_y = int(robot_pos[1])
            req.goal_x = int(goal_pos[0])
            req.goal_y = int(goal_pos[1])
            req.sequence = sequence
        except (KeyError, TypeError, IndexError) as e:
            self._tcp.send({"type": "error", "message": f"Bad predict request: {e}"})
            return

        future = self._predict_client.call_async(req)
        future.add_done_callback(
            functools.partial(self._on_predict_response, sequence=sequence, t_recv_ns=t_recv_ns))

    def _on_predict_response(self, future, sequence: int = 0, t_recv_ns: int = 0) -> None:
        try:
            result = future.result()
        except Exception as e:
            self.get_logger().warning(f"PredictAction call failed: {e}")
            self._tcp.send({"type": "error", "message": str(e)})
            return

        # Stamp send just before writing the response to the socket.
        t_send_ns = time.monotonic_ns()
        if result.success:
            self._tcp.send({
                "type": "predict_result",
                "sequence": sequence,
                "action": int(result.action),
                "direction": [int(result.direction_x), int(result.direction_y)],
            })
        else:
            self._tcp.send({"type": "error", "message": result.message})

        # Bridge-side timing (single-host monotonic delta; joined by sequence).
        timing = BridgePredictTiming()
        timing.header.stamp = self.get_clock().now().to_msg()
        timing.sequence = sequence
        timing.t_bridge_recv_ns = t_recv_ns
        timing.t_bridge_send_ns = t_send_ns
        self._predict_timing_pub.publish(timing)

    def _handle_experiment_event(self, msg: dict) -> None:
        """Forward a run-lifecycle event from the orchestrator onto /experiment/events."""
        ev = ExperimentEvent()
        ev.header.stamp = self.get_clock().now().to_msg()
        ev.event_type = str(msg.get("event_type", ""))
        ev.run_id = str(msg.get("run_id", ""))
        ev.payload_json = json.dumps(msg.get("payload", {}))
        self._events_pub.publish(ev)
        self.get_logger().info(
            f"experiment_event: {ev.event_type} run_id={ev.run_id}"
        )

    @staticmethod
    def _list_to_gridmap(grid_2d: list) -> GridMap:
        """Convert a 2D list from JSON into a GridMap message."""
        gm = GridMap()
        gm.rows = len(grid_2d)
        gm.cols = len(grid_2d[0]) if gm.rows > 0 else 0
        gm.data = [float(v) for row in grid_2d for v in row]
        return gm

    # -- grid pose subscription -----------------------------------------------

    def _on_grid_pose(self, msg: GridPose) -> None:
        """Forward the nav node's self-estimate to the TCP client."""
        self._tcp.send({
            "type": "nav_pose",
            "x": float(msg.x),
            "y": float(msg.y),
            "heading": float(msg.heading),
        })

    # -- cleanup --------------------------------------------------------------

    def destroy_node(self) -> None:
        self._tcp.stop()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BridgeNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
