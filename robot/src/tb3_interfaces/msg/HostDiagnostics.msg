# Per-host ambient diagnostics, published at 1 Hz by tb3_diagnostics.
#
# One publisher per ROS host (Pi, desktop). Each host's bag captures only its
# own messages; the `hostname` field disambiguates if bags are later merged.
# Moving these into the bag (rather than SSH-collected metadata) means
# analysis reads chrony/WiFi/RAPL state directly from bag content within the
# run window.
std_msgs/Header header

string hostname
float32 chrony_offset_ms      # "Last offset" from `chronyc tracking`
float32 chrony_rms_ms         # "RMS offset" from `chronyc tracking`
int32   wifi_rssi_dbm         # from `iwconfig <iface>`; 0 on hosts without WiFi
float32 wifi_link_quality     # from `iwconfig`; 0 elsewhere
bool    rapl_passthrough_ok   # desktop only; false elsewhere
