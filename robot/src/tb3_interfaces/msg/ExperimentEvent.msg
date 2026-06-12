# A run-lifecycle marker emitted by the orchestrator over the TCP bridge and
# forwarded onto /experiment/events by ros2_bridge.
#
# The bridge stamps `header.stamp` on the bridge host's (chrony-synced) clock
# when forwarding. Analysis uses run_start / run_end events to bound the run
# window in bag time — NOT the laptop wall clock.
std_msgs/Header header

string event_type             # "run_start" | "run_end" | "marker" | "bag_capped"
string run_id
string payload_json           # event-type-specific JSON; analysis reads event_type, treats payload as opaque
