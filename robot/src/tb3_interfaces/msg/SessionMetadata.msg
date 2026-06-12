# Latched-at-startup metadata for the ROS stack, published once by
# tb3_diagnostics with TRANSIENT_LOCAL QoS so `ros2 bag record` picks it up
# immediately on subscribe.
#
# This carries the git SHA and experiment.yaml snapshot that used to live in
# the orchestrator's metadata.yaml — now sourced from the bag so it reflects
# exactly what was running on the ROS host for that run.
std_msgs/Header header

string hostname
string git_sha_research_project   # `git rev-parse HEAD` of the robot code checkout at startup
string experiment_yaml_snapshot   # full contents of tb3_bringup/config/experiment.yaml at startup
