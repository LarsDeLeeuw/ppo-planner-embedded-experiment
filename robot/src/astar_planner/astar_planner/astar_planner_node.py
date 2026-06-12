"""ROS2 node wrapping the A* planner as a native PredictAction service.

Same external contract as ppo_planner: service type, name, and response
shape are identical. Which variant (`energy` | `shortest`) runs is
controlled by the `variant` ROS parameter; the bringup launch exposes
both under distinct node names (`astar_planner_node` / `astar_shortest_node`)
so the bridge's `predict_service` selects which one is active.
"""

from __future__ import annotations

import time

import rclpy
from rclpy.node import Node

from tb3_interfaces.msg import PlannerMetrics
from tb3_interfaces.srv import PredictAction
from tb3_planner_common.gridmap import gridmap_to_numpy

from astar_planner.astar_search import (
    AlreadyAtGoalError,
    AstarPlanner,
    NoPathError,
)


class AstarPlannerNode(Node):

    def __init__(self) -> None:
        super().__init__("astar_planner_node")

        # -- Parameters -----------------------------------------------------------
        self.declare_parameter("variant", "energy")
        self.declare_parameter("energy_weight", 0.5)
        self.declare_parameter("max_expansions", 0)

        variant = self.get_parameter("variant").value
        energy_weight = self.get_parameter("energy_weight").value
        max_expansions_param = int(self.get_parameter("max_expansions").value)

        if variant not in ("energy", "shortest"):
            raise ValueError(
                f"variant must be 'energy' or 'shortest', got {variant!r}"
            )

        self._variant = variant
        self._planner = AstarPlanner(
            energy_weight=energy_weight,
            max_expansions=max_expansions_param if max_expansions_param > 0 else None,
        )

        self.get_logger().info(
            f"A* planner ready — variant='{self._variant}', "
            f"energy_weight={energy_weight}, "
            f"max_expansions={max_expansions_param if max_expansions_param > 0 else 'unlimited'}"
        )

        # -- Service server -------------------------------------------------------
        self._predict_srv = self.create_service(
            PredictAction, "~/predict_action", self._predict_cb,
        )
        # Per-call timing/diagnostics (absolute topic so it lands on
        # /planner/metrics regardless of node name/namespace).
        self._metrics_pub = self.create_publisher(PlannerMetrics, "/planner/metrics", 10)
        self.get_logger().info("AstarPlannerNode service: ~/predict_action")

    # -- service callback ---------------------------------------------------------

    def _predict_cb(
        self,
        request: PredictAction.Request,
        response: PredictAction.Response,
    ) -> PredictAction.Response:
        inference_ns = 0
        success = False
        path_length = -1
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

            # Bracket ONLY the search (perf_counter, single-host).
            t0 = time.perf_counter_ns()
            result = self._planner.predict(
                obstacle_map=obstacle_map,
                energy_map=energy_map,
                robot_pos=robot_pos,
                goal_pos=goal_pos,
                variant=self._variant,
            )
            inference_ns = time.perf_counter_ns() - t0

            response.success = True
            response.message = ""
            response.action = int(result["action"])
            response.direction_x = int(result["direction"][0])
            response.direction_y = int(result["direction"][1])
            path_length = len(result["path"])
            success = True

            self.get_logger().debug(
                f"A* {self._variant}: action={response.action} "
                f"cost={result['cost']:.2f} energy={result['energy']:.2f} "
                f"path_len={path_length}"
            )

        except (NoPathError, AlreadyAtGoalError, ValueError) as e:
            response.success = False
            response.message = str(e)
            self.get_logger().debug(f"A* {self._variant} failed: {e}")
        except Exception as e:
            response.success = False
            response.message = f"unexpected: {e}"
            self.get_logger().error(f"A* {self._variant} unexpected error: {e}")

        finally:
            self._publish_metrics(request, inference_ns, success, path_length)

        return response

    def _publish_metrics(
        self,
        request: PredictAction.Request,
        inference_ns: int,
        success: bool,
        path_length: int,
    ) -> None:
        m = PlannerMetrics()
        m.header.stamp = self.get_clock().now().to_msg()
        m.sequence = request.sequence
        m.planner_name = f"astar_{self._variant}"
        m.inference_us = max(0, inference_ns // 1000)
        m.success = success
        # True expansion count isn't exposed by astar_search (it tracks it
        # internally for max_expansions but doesn't return it); path_length is
        # the planner-effort metric used in analysis. -1 = not reported.
        m.nodes_expanded = -1
        m.path_length = path_length
        m.grid_rows = request.obstacle_map.rows
        m.grid_cols = request.obstacle_map.cols
        self._metrics_pub.publish(m)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AstarPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
