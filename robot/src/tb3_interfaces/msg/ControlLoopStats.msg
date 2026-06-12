# Control-loop period statistics from grid_nav_node.
#
# Published at 1 Hz, computed over a ring buffer of the last N tick deltas.
# Used to detect loop slip under load (e.g., decentralized mode running PPO
# inference on the same Pi that runs the nav loop).
std_msgs/Header header

float32 nominal_period_ms     # configured target (e.g., 50.0 at 20 Hz)
float32 mean_period_ms        # over last `ticks` samples
float32 p99_period_ms
float32 max_period_ms
uint32 ticks                  # window size (number of deltas summarized)
uint32 overruns               # ticks with period > 2 x nominal
