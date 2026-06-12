# A single power measurement from one INA219 sensor.
#
# Three sensors are deployed on the robot's I2C bus (see tb3_power_sensor):
#   - "solar"  : between solar panel and battery (bidirectional; +current = panel charging battery)
#   - "sbc"    : between battery and Pi 4B      (+current = power into the SBC)
#   - "opencr" : between battery and OpenCR     (+current = power into motors/IMU/servo)
#
# Current and power are signed (INA219 native 2's complement). The solar
# sensor in particular relies on this to report battery-as-load (negative).
std_msgs/Header header

string sensor_id              # "solar" | "sbc" | "opencr"
uint8  i2c_address            # 0x40..0x45; reproducibility / wiring trace
float32 shunt_resistance_ohm  # snapshotted from config; default 0.1

float32 bus_voltage_v
float32 shunt_voltage_mv
float32 current_ma            # signed; CALIBRATED (gain-corrected per current_calibration below)
float32 power_mw              # signed; CALIBRATED (gain-corrected per current_calibration below)

# Per-sensor gain calibration factor measured by injecting a known reference
# current. Defined as (chip_reading_mA / actual_mA) at the calibration point:
# a value of 1.017 means the bare INA219 reads 1.7% high. The node has
# already divided current_ma and power_mw by this factor, so consumers do
# NOT re-apply it. Kept on the wire for traceability / re-derivation of the
# raw chip reading (current_chip = current_ma * current_calibration).
float32 current_calibration

bool   overflow               # true if INA219 OVF (math overflow) bit set; sample is saturated, exclude from integration
uint32 sequence               # monotonic per-sensor, for gap detection
