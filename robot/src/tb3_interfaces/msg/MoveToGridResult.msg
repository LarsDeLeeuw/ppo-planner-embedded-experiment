# Republication of a completed MoveToGrid action result on a normal topic.
#
# ros2 bag records topics, NOT action services, so the action's goal/result
# is invisible to bags. grid_nav_node publishes this message on
# /grid_nav_node/last_result after EVERY completed action (success, failure,
# preempt, or timeout). This is the ground-truth goal-success signal for
# analysis.
std_msgs/Header header

string run_id_hint            # may be empty; orchestrator correlates via experiment_event window
bool success
string message
tb3_interfaces/GridPose final_pose
float32 total_distance_m
float32 total_rotation_rad
string termination_reason     # "goal_reached" | "timeout" | "preempted" | "error"
