"""Samples Intel RAPL energy counters on the desktop VM (centralized mode).

Publishes /desktop/rapl_energy_uj (tb3_interfaces/RaplEnergy) at 1 Hz. The
counters are cumulative monotonic microjoule values that wrap periodically;
analysis differentiates over the run window and handles wrap.

This only produces meaningful data when the hypervisor exposes RAPL
passthrough to the guest. `passthrough_ok` reflects whether the package
counter was readable on each tick; if false, analysis excludes RAPL-derived
plots for that run.

Runs ONLY on the desktop VM, and only matters in centralized mode (where the
desktop runs the planner/nav/bridge whose CPU energy isn't visible to the
robot's INA219 on the SBC rail). See plan §12.5.
"""

from __future__ import annotations

import glob
import os

import rclpy
from rclpy.node import Node

from tb3_interfaces.msg import RaplEnergy

_PKG_ENERGY = "/sys/class/powercap/intel-rapl:0/energy_uj"
_PKG0_GLOB = "/sys/class/powercap/intel-rapl:0:*"


class RaplSamplerNode(Node):

    def __init__(self) -> None:
        super().__init__("rapl_sampler_node")
        self.declare_parameter("rate_hz", 1.0)
        rate = self.get_parameter("rate_hz").value

        self._dram_energy_path = self._find_dram_path()
        self._pub = self.create_publisher(RaplEnergy, "/desktop/rapl_energy_uj", 10)
        self.create_timer(1.0 / rate, self._sample_cb)

        ok = self._read_uj(_PKG_ENERGY) is not None
        self.get_logger().info(
            f"RaplSamplerNode ready (rate={rate} Hz, passthrough_ok={ok}, "
            f"dram={'yes' if self._dram_energy_path else 'no'})"
        )

    @staticmethod
    def _find_dram_path() -> str | None:
        """Locate the DRAM subdomain's energy_uj, if exposed."""
        for sub in glob.glob(_PKG0_GLOB):
            try:
                with open(os.path.join(sub, "name")) as f:
                    if f.read().strip() == "dram":
                        return os.path.join(sub, "energy_uj")
            except OSError:
                continue
        return None

    @staticmethod
    def _read_uj(path: str | None) -> int | None:
        if not path:
            return None
        try:
            with open(path) as f:
                return int(f.read().strip())
        except (OSError, ValueError):
            return None

    def _sample_cb(self) -> None:
        pkg = self._read_uj(_PKG_ENERGY)
        dram = self._read_uj(self._dram_energy_path)

        msg = RaplEnergy()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.passthrough_ok = pkg is not None
        msg.energy_pkg_uj = pkg if pkg is not None else 0
        msg.energy_ram_uj = dram if dram is not None else 0
        self._pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RaplSamplerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
