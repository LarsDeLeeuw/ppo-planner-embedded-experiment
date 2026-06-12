"""Publishes power telemetry from three INA219 sensors plus SBC thermal state.

Topics:
  /power/solar, /power/sbc, /power/opencr   (tb3_interfaces/PowerSample, 100 Hz)
  /sbc/thermal                              (tb3_interfaces/ThermalStat, 1 Hz)

Runs on the robot SBC (Pi 4B) in BOTH experiment modes — it is the only
always-on data source in centralized mode.

Threading: a MultiThreadedExecutor with the I2C sample timer in a
ReentrantCallbackGroup and the thermal timer in a separate
MutuallyExclusiveCallbackGroup. The thermal read is sysfs-only (no I2C
contention), so the two timers can run concurrently without starving each
other.

I2C lockup handling: if a sensor returns 3 consecutive read errors the node
logs an error and re-applies that sensor's config/calibration, then keeps
going. The per-sensor `sequence` counter only advances on successful
publishes, so a lockup shows up as a sequence gap in analysis (see analysis
handover §9). Aborting the *run* on lockup is an orchestrator-side decision
made by observing that gap — the sensor node does not own run lifecycle.
"""

from __future__ import annotations

import subprocess

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from tb3_interfaces.msg import PowerSample, ThermalStat

from tb3_power_sensor.ina219 import INA219

_THERMAL_ZONE = "/sys/class/thermal/thermal_zone0/temp"
_LOCKUP_THRESHOLD = 3  # consecutive failed reads before re-init


class _SensorEntry:
    def __init__(self, sensor_id: str, dev: INA219, publisher,
                 current_calibration: float) -> None:
        self.sensor_id = sensor_id
        self.dev = dev
        self.publisher = publisher
        # Per-sensor gain factor: chip_reading_mA / actual_mA at the calibration
        # point. Divided out below so published current_ma matches reality.
        # Stored on the message too for analysis-time traceability.
        self.current_calibration = current_calibration
        self.sequence = 0
        self.consecutive_errors = 0


class PowerSensorNode(Node):

    def __init__(self) -> None:
        super().__init__("power_sensor_node")

        # -- Parameters -------------------------------------------------------
        self.declare_parameter("i2c_bus", 1)
        self.declare_parameter("publish_rate_hz", 100.0)
        self.declare_parameter("thermal_rate_hz", 1.0)
        self.declare_parameter("adc_mode", "8avg")
        self.declare_parameter("pga", "8_320mv")
        self.declare_parameter("bus_range_v", 32)
        self.declare_parameter("sensor_ids", ["solar", "sbc", "opencr"])

        i2c_bus = self.get_parameter("i2c_bus").value
        publish_rate = self.get_parameter("publish_rate_hz").value
        thermal_rate = self.get_parameter("thermal_rate_hz").value
        adc_mode = self.get_parameter("adc_mode").value
        pga = self.get_parameter("pga").value
        bus_range_v = self.get_parameter("bus_range_v").value
        sensor_ids = self.get_parameter("sensor_ids").value

        # -- Open the I2C bus -------------------------------------------------
        # Imported here so the package builds/imports on a non-Pi dev machine;
        # only the running node on the Pi needs smbus2 + a real bus.
        import smbus2
        self._bus = smbus2.SMBus(i2c_bus)

        # -- Initialise each sensor -------------------------------------------
        self._sensors: list[_SensorEntry] = []
        for sid in sensor_ids:
            self.declare_parameter(f"{sid}.i2c_address", 0x40)
            self.declare_parameter(f"{sid}.shunt_ohm", 0.1)
            self.declare_parameter(f"{sid}.max_current_a", 3.2)
            self.declare_parameter(f"{sid}.current_calibration", 1.0)
            address = self.get_parameter(f"{sid}.i2c_address").value
            shunt_ohm = self.get_parameter(f"{sid}.shunt_ohm").value
            max_current_a = self.get_parameter(f"{sid}.max_current_a").value
            current_cal = self.get_parameter(f"{sid}.current_calibration").value
            # 0 or negative would silently zero / sign-flip every sample.
            if current_cal <= 0.0:
                self.get_logger().warning(
                    f"Sensor '{sid}' current_calibration={current_cal} is invalid; "
                    "falling back to 1.0 (no correction)."
                )
                current_cal = 1.0

            try:
                dev = INA219(
                    bus=self._bus,
                    address=address,
                    shunt_ohm=shunt_ohm,
                    max_current_a=max_current_a,
                    bus_range_v=bus_range_v,
                    pga=pga,
                    adc_mode=adc_mode,
                )
            except Exception as exc:  # noqa: BLE001 — log and skip missing sensors
                self.get_logger().error(
                    f"Sensor '{sid}' @ 0x{address:02x} failed to init: {exc}. Skipping."
                )
                continue

            publisher = self.create_publisher(PowerSample, f"power/{sid}", 50)
            self._sensors.append(_SensorEntry(sid, dev, publisher, current_cal))
            self.get_logger().info(
                f"Sensor '{sid}' @ 0x{address:02x}: shunt={shunt_ohm}Ω "
                f"current_lsb={dev.current_lsb_a * 1e3:.4f}mA cal_reg={dev.calibration} "
                f"current_calibration={current_cal:.4f}"
            )

        if not self._sensors:
            self.get_logger().fatal("No INA219 sensors initialised — nothing to publish.")

        # -- Thermal publisher ------------------------------------------------
        self._thermal_pub = self.create_publisher(ThermalStat, "sbc/thermal", 10)

        # -- Timers (separate callback groups) --------------------------------
        sample_group = ReentrantCallbackGroup()
        thermal_group = MutuallyExclusiveCallbackGroup()
        self.create_timer(1.0 / publish_rate, self._sample_cb, callback_group=sample_group)
        self.create_timer(1.0 / thermal_rate, self._thermal_cb, callback_group=thermal_group)

        self.get_logger().info(
            f"PowerSensorNode ready — {len(self._sensors)} sensor(s) @ {publish_rate:.0f} Hz, "
            f"adc_mode={adc_mode}, pga={pga}"
        )

    # -- sampling -------------------------------------------------------------

    def _sample_cb(self) -> None:
        stamp = self.get_clock().now().to_msg()
        for s in self._sensors:
            try:
                reading = s.dev.read()
            except Exception as exc:  # noqa: BLE001 — I2C read error
                s.consecutive_errors += 1
                if s.consecutive_errors == _LOCKUP_THRESHOLD:
                    self.get_logger().error(
                        f"Sensor '{s.sensor_id}' lockup ({_LOCKUP_THRESHOLD} consecutive "
                        f"errors: {exc}); re-applying config."
                    )
                    try:
                        s.dev.configure()
                    except Exception as reinit_exc:  # noqa: BLE001
                        self.get_logger().error(
                            f"Sensor '{s.sensor_id}' re-init failed: {reinit_exc}"
                        )
                    s.consecutive_errors = 0
                continue

            s.consecutive_errors = 0
            # Apply per-sensor gain correction. current_calibration is
            # (chip_reading / actual) at the calibration point — divide to
            # get back to actual current. power = V*I so the same scalar
            # applies (bus_voltage is independent of current calibration).
            cal = s.current_calibration
            msg = PowerSample()
            msg.header.stamp = stamp
            msg.header.frame_id = s.sensor_id
            msg.sensor_id = s.sensor_id
            msg.i2c_address = s.dev.address
            msg.shunt_resistance_ohm = float(s.dev.shunt_ohm)
            msg.bus_voltage_v = float(reading.bus_voltage_v)
            msg.shunt_voltage_mv = float(reading.shunt_voltage_mv)
            msg.current_ma = float(reading.current_ma / cal)
            msg.power_mw = float(reading.power_mw / cal)
            msg.current_calibration = float(cal)
            msg.overflow = reading.overflow
            msg.sequence = s.sequence
            s.publisher.publish(msg)
            s.sequence += 1

    # -- thermal --------------------------------------------------------------

    def _thermal_cb(self) -> None:
        msg = ThermalStat()
        msg.header.stamp = self.get_clock().now().to_msg()

        try:
            with open(_THERMAL_ZONE) as f:
                msg.cpu_temp_c = int(f.read().strip()) / 1000.0
        except Exception:  # noqa: BLE001
            msg.cpu_temp_c = float("nan")

        flags = self._read_throttle_flags()
        msg.throttle_flags = flags
        # bit 2 of `vcgencmd get_throttled` = currently throttled.
        msg.throttled = bool(flags & (1 << 2))
        self._thermal_pub.publish(msg)

    @staticmethod
    def _read_throttle_flags() -> int:
        try:
            out = subprocess.run(
                ["vcgencmd", "get_throttled"],
                capture_output=True, text=True, timeout=1.0,
            ).stdout.strip()
            # Format: "throttled=0x50000"
            return int(out.split("=", 1)[1], 16)
        except Exception:  # noqa: BLE001 — vcgencmd absent or failed
            return 0


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PowerSensorNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
