# Orchestrator-side (laptop) TCP round-trip timing for one predict call.
#
# Written into orchestrator.mcap by minimap-visualizer. Both timestamps are
# taken on the LAPTOP's CLOCK_MONOTONIC, so the local delta
# (t_tcp_recv_ns - t_tcp_send_ns) is meaningful in isolation.
#
# Latency decomposition (all single-host deltas, joined by `sequence`):
#   inference_us       = PlannerMetrics.inference_us
#   bridge_plumbing_us = (BridgePredictTiming delta)   - inference_us
#   tcp_us             = (PredictTcpTiming delta)       - (BridgePredictTiming delta)
std_msgs/Header header

uint32 sequence
uint64 t_tcp_send_ns          # CLOCK_MONOTONIC on laptop, at TCP send
uint64 t_tcp_recv_ns          # CLOCK_MONOTONIC on laptop, at TCP recv (same clock as send)
