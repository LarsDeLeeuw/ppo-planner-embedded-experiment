# Intel RAPL energy counters from the desktop VM (centralized mode only).
#
# Published at 1 Hz by tb3_rapl_sampler on the desktop. The counters are
# cumulative and wrap periodically; analysis differentiates over the run
# window and handles wrap. Only meaningful if the hypervisor exposes RAPL
# passthrough — check passthrough_ok before trusting the values.
std_msgs/Header header

uint64 energy_pkg_uj          # /sys/class/powercap/intel-rapl:0/energy_uj (cumulative)
uint64 energy_ram_uj          # cumulative; 0 if unavailable
bool   passthrough_ok         # false if the RAPL counter could not be read
