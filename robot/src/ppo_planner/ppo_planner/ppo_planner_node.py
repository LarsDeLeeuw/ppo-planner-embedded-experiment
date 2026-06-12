"""
ROS2 node that wraps the PPO planner as a native service.

Loads an ONNX-exported PPO policy once at startup (numpy + onnxruntime, no
torch), then serves PredictAction requests synchronously. Callers use the ROS2
service interface directly — no TCP in the loop.

Defaults to stochastic action sampling (deterministic:=false), which is how PPO
was trained and what it needs to navigate; set a non-negative `seed` for
reproducible runs. See README for the parameter reference.
"""

from __future__ import annotations

import os
import time

import rclpy
from rclpy.node import Node

from tb3_interfaces.msg import PlannerMetrics
from tb3_interfaces.srv import PredictAction
from tb3_planner_common.directions import DIRECTION
from tb3_planner_common.gridmap import gridmap_to_numpy

from ppo_planner.ppo_inference import PPOPlanner


class PPOPlannerNode(Node):

    def __init__(self) -> None:
        super().__init__("ppo_planner_node")

        # -- Parameters -----------------------------------------------------------
        self.declare_parameter("model_path", "AIPPOm10EH_continued.onnx")
        self.declare_parameter("grid_size_x", 10)
        self.declare_parameter("grid_size_y", 10)
        self.declare_parameter("num_robots", 10)
        self.declare_parameter("deterministic", False)
        # RNG seed for stochastic sampling. Default 0 => reproducible across
        # launches; negative => nondeterministic (fresh entropy each launch).
        # Ignored when deterministic=True.
        self.declare_parameter("seed", 0)

        model_path_param = self.get_parameter("model_path").value
        grid_size_x = self.get_parameter("grid_size_x").value
        grid_size_y = self.get_parameter("grid_size_y").value
        num_robots = self.get_parameter("num_robots").value
        deterministic = self.get_parameter("deterministic").value
        seed_param = int(self.get_parameter("seed").value)
        seed = seed_param if seed_param >= 0 else None

        # -- Resolve model path ---------------------------------------------------
        if not os.path.isabs(model_path_param):
            from ament_index_python.packages import get_package_share_directory
            share_dir = get_package_share_directory("ppo_planner")
            model_path = os.path.join(share_dir, "models", model_path_param)
        else:
            model_path = model_path_param

        # -- Load PPO model -------------------------------------------------------
        self._grid_size = (grid_size_x, grid_size_y)
        self.get_logger().info(
            f"Loading PPO model: {model_path} "
            f"(grid_size={self._grid_size}, deterministic={deterministic}, "
            f"seed={'none' if seed is None else seed})"
        )
        self._planner = PPOPlanner(
            model_path=model_path,
            grid_size=self._grid_size,
            num_robots=num_robots,
            deterministic=deterministic,
            seed=seed,
        )
        self.get_logger().info("PPO model loaded successfully (onnxruntime backend)")

        # -- Service server -------------------------------------------------------
        self._predict_srv = self.create_service(
            PredictAction, "~/predict_action", self._predict_cb,
        )
        # Per-call timing/diagnostics for the latency decomposition (absolute
        # topic so it lands on /planner/metrics regardless of node namespace).
        self._metrics_pub = self.create_publisher(PlannerMetrics, "/planner/metrics", 10)
        self.get_logger().info("PPOPlannerNode ready — service: ~/predict_action")

    # -- service callback ---------------------------------------------------------

    def _predict_cb(
        self,
        request: PredictAction.Request,
        response: PredictAction.Response,
    ) -> PredictAction.Response:
        inference_ns = 0
        success = False
        try:
            obstacle_map = gridmap_to_numpy(
                request.obstacle_map.rows,
                request.obstacle_map.cols,
                request.obstacle_map.data,
            )
            energy_map = gridmap_to_numpy(
                request.energy_map.rows,
                request.energy_map.cols,
                request.energy_map.data,
            )

            robot_pos = (request.robot_x, request.robot_y)
            goal_pos = (request.goal_x, request.goal_y)

            # Bracket ONLY the forward pass (perf_counter, single-host).
            t0 = time.perf_counter_ns()
            action = self._planner.predict(
                obstacle_map=obstacle_map,
                energy_map=energy_map,
                robot0_pos=robot_pos,
                goal_pos=goal_pos,
            )
            inference_ns = time.perf_counter_ns() - t0

            dx, dy = DIRECTION[action]
            response.success = True
            response.message = ""
            response.action = action
            response.direction_x = dx
            response.direction_y = dy
            success = True

        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")
            response.success = False
            response.message = str(e)

        finally:
            self._publish_metrics(request, inference_ns, success)

        return response

    def _publish_metrics(
        self, request: PredictAction.Request, inference_ns: int, success: bool,
    ) -> None:
        m = PlannerMetrics()
        m.header.stamp = self.get_clock().now().to_msg()
        m.sequence = request.sequence
        m.planner_name = "ppo"
        m.inference_us = max(0, inference_ns // 1000)
        m.success = success
        m.nodes_expanded = -1            # N/A for PPO
        m.path_length = -1               # PPO returns a single action, not a path
        m.grid_rows = request.obstacle_map.rows
        m.grid_cols = request.obstacle_map.cols
        self._metrics_pub.publish(m)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PPOPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
