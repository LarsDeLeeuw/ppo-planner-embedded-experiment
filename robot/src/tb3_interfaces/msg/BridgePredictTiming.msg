# Bridge-side timing for one predict round-trip, published by ros2_bridge.
#
# Both timestamps are taken on the BRIDGE HOST's CLOCK_MONOTONIC, so the
# local delta (t_bridge_send_ns - t_bridge_recv_ns) is meaningful in
# isolation. Never subtract these against timestamps from another host.
#
# t_bridge_recv_ns is stamped when the bridge BEGINS handling the predict
# message, AFTER the preceding `pose` has been drained from the same poll
# cycle (the orchestrator always sends pose-then-predict). This keeps the
# pose service-call cost OUT of the bridge delta; it lands in the
# orchestrator-side TCP slice instead. See plan section 11.1.
std_msgs/Header header

uint32 sequence               # matches PredictAction / PlannerMetrics / PredictTcpTiming
uint64 t_bridge_recv_ns       # CLOCK_MONOTONIC on bridge host
uint64 t_bridge_send_ns       # CLOCK_MONOTONIC on bridge host (same clock as recv)
