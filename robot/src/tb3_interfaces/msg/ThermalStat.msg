# SBC (Pi 4B) thermal state, published at 1 Hz by the tb3_power_sensor node.
#
# Runs where `throttled` is ever true must NOT be aggregated with
# non-throttled runs: a throttled CPU runs slower AND draws less power,
# which confounds the centralized-vs-decentralized comparison.
std_msgs/Header header

float32 cpu_temp_c            # /sys/class/thermal/thermal_zone0/temp (millidegrees / 1000)
bool throttled               # bit 0 of `vcgencmd get_throttled` (currently throttled)
uint32 throttle_flags        # full raw `get_throttled` value (under-voltage, soft temp limit, etc.)
