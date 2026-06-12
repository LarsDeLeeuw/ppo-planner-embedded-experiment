"""Minimal INA219 driver over I2C (smbus2).

Covers only what tb3_power_sensor needs: continuous bus + shunt monitoring
with current/power derived from a programmed calibration register.

Current and power are returned signed (the chip uses 2's complement on the
shunt and current registers). The solar sensor relies on this to report a
negative current when the battery drives current backward through the panel.

Calibration is computed from the shunt resistance and the maximum expected
current rather than hard-coded, so a hardware change (different shunt) only
needs a config edit. The chosen current_lsb and the resulting calibration are
exposed for logging.

INA219 register map and bit layout per the TI datasheet (SBOS448).
"""

from __future__ import annotations

from dataclasses import dataclass

# Register addresses.
_REG_CONFIG = 0x00
_REG_SHUNT = 0x01
_REG_BUS = 0x02
_REG_POWER = 0x03
_REG_CURRENT = 0x04
_REG_CALIBRATION = 0x05

# Config register fields.
_CONFIG_RESET = 0x8000
_BRNG = {16: 0x0000, 32: 0x2000}                       # bus voltage range
_PGA = {                                               # shunt PGA gain / range
    "1_40mv": 0x0000,
    "2_80mv": 0x0800,
    "4_160mv": 0x1000,
    "8_320mv": 0x1800,
}
# ADC resolution / averaging code (applies to BADC bits 10:7 and SADC bits 6:3).
# Averaging modes use a 12-bit base and integrate over multiple samples, which
# acts as an anti-alias low-pass — important because the OpenCR motor PWM is
# ~16 kHz and a single 12-bit conversion at 100 Hz would alias it. "8avg"
# integrates ~4.26 ms per channel (~8.5 ms shunt+bus), well matched to a 100 Hz
# read loop in continuous mode.
_ADC = {
    "9bit": 0x0, "10bit": 0x1, "11bit": 0x2, "12bit": 0x3,
    "2avg": 0x9, "4avg": 0xA, "8avg": 0xB, "16avg": 0xC,
    "32avg": 0xD, "64avg": 0xE, "128avg": 0xF,
}
_MODE_SHUNT_BUS_CONTINUOUS = 0x07

# Bus voltage register flags (low 2 bits).
_BUS_OVF = 0x0001    # math overflow: power/current readings are invalid
_BUS_CNVR = 0x0002   # conversion ready

_BUS_LSB_V = 0.004   # bus voltage LSB = 4 mV
_SHUNT_LSB_MV = 0.01  # shunt voltage LSB = 10 µV = 0.01 mV


@dataclass
class PowerReading:
    bus_voltage_v: float
    shunt_voltage_mv: float
    current_ma: float
    power_mw: float
    overflow: bool


def _to_signed(value: int) -> int:
    """Interpret a 16-bit register value as signed 2's complement."""
    return value - 0x10000 if value & 0x8000 else value


class INA219:
    """One INA219 sensor on a shared I2C bus."""

    def __init__(
        self,
        bus,                       # smbus2.SMBus instance (shared across sensors)
        address: int,
        shunt_ohm: float = 0.1,
        max_current_a: float = 3.2,
        bus_range_v: int = 32,
        pga: str = "8_320mv",
        adc_mode: str = "8avg",
    ) -> None:
        self._bus = bus
        self.address = address
        self.shunt_ohm = shunt_ohm

        if bus_range_v not in _BRNG:
            raise ValueError(f"bus_range_v must be one of {list(_BRNG)}")
        if pga not in _PGA:
            raise ValueError(f"pga must be one of {list(_PGA)}")
        if adc_mode not in _ADC:
            raise ValueError(f"adc_mode must be one of {list(_ADC)}")

        # Current_LSB chosen so full-scale ≈ max_current_a; power_lsb is fixed at
        # 20× by the chip. Calibration = trunc(0.04096 / (Current_LSB * R_shunt)).
        self.current_lsb_a = max_current_a / 32768.0
        self.power_lsb_w = 20.0 * self.current_lsb_a
        self.calibration = int(0.04096 / (self.current_lsb_a * shunt_ohm))

        adc_code = _ADC[adc_mode]
        self._config = (
            _BRNG[bus_range_v]
            | _PGA[pga]
            | (adc_code << 7)        # BADC
            | (adc_code << 3)        # SADC
            | _MODE_SHUNT_BUS_CONTINUOUS
        )
        self.configure()

    # -- register I/O (16-bit big-endian) ------------------------------------

    def _write_register(self, reg: int, value: int) -> None:
        self._bus.write_i2c_block_data(self.address, reg, [(value >> 8) & 0xFF, value & 0xFF])

    def _read_register(self, reg: int) -> int:
        msb, lsb = self._bus.read_i2c_block_data(self.address, reg, 2)
        return (msb << 8) | lsb

    # -- public API -----------------------------------------------------------

    def configure(self) -> None:
        """(Re)apply config + calibration. Order matters: calibration must be
        written after config or the current/power registers stay zero."""
        self._write_register(_REG_CONFIG, self._config)
        self._write_register(_REG_CALIBRATION, self.calibration)

    def read(self) -> PowerReading:
        bus_raw = self._read_register(_REG_BUS)
        overflow = bool(bus_raw & _BUS_OVF)
        bus_voltage_v = (bus_raw >> 3) * _BUS_LSB_V

        shunt_voltage_mv = _to_signed(self._read_register(_REG_SHUNT)) * _SHUNT_LSB_MV
        current_ma = _to_signed(self._read_register(_REG_CURRENT)) * self.current_lsb_a * 1000.0
        # Power register is unsigned per datasheet.
        power_mw = self._read_register(_REG_POWER) * self.power_lsb_w * 1000.0

        return PowerReading(
            bus_voltage_v=bus_voltage_v,
            shunt_voltage_mv=shunt_voltage_mv,
            current_ma=current_ma,
            power_mw=power_mw,
            overflow=overflow,
        )
