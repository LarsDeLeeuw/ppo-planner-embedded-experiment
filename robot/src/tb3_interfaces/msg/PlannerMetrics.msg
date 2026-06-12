# Per-call timing and diagnostics for a planner inference.
#
# Published by astar_planner_node and ppo_planner_node after every
# predict_action service call. inference_us brackets ONLY the algorithm
# (the A* search or the PPO forward pass) measured with perf_counter_ns() —
# not service deserialization, validation, or ROS plumbing.
std_msgs/Header header

uint32 sequence               # matches the PredictAction request sequence (set by orchestrator)
string planner_name           # "ppo" | "astar_energy" | "astar_shortest"
uint32 inference_us           # algorithm-only duration, microseconds

bool success

int32 nodes_expanded          # A* expansion count; -1 if N/A (PPO sets -1)
int32 path_length             # planner's INTERNAL planned path length in cells; -1 if N/A (PPO)

uint32 grid_rows              # so latency can be correlated with grid size
uint32 grid_cols
