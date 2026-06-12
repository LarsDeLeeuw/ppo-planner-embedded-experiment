"""Unit tests for grid_math module.

Focuses on non-trivial logic: quaternion conversion, coordinate
roundtrip invariants, and rounding edge cases. One-line stdlib
wrappers (atan2, hypot, multiplication) are not tested individually.
"""

import math
import pytest
from tb3_nav.grid_math import (
    normalize_angle,
    grid_to_world,
    world_to_grid,
    yaw_from_quaternion,
)


class TestNormalizeAngle:
    """The wrapping logic has boundary behavior worth verifying."""

    def test_wraps_positive_overflow(self):
        assert normalize_angle(3 * math.pi) == pytest.approx(math.pi, abs=1e-10)

    def test_wraps_negative_overflow(self):
        assert normalize_angle(-3 * math.pi) == pytest.approx(-math.pi, abs=1e-10)

    def test_full_rotation_wraps_to_zero(self):
        assert normalize_angle(2 * math.pi) == pytest.approx(0.0, abs=1e-10)

    def test_output_always_in_range(self):
        """Property: output is always in [-pi, pi] for arbitrary input."""
        for deg in range(-720, 721, 15):
            rad = math.radians(deg)
            result = normalize_angle(rad)
            assert -math.pi <= result <= math.pi, (
                f"normalize_angle({rad}) = {result} outside [-pi, pi]"
            )


class TestGridWorldRoundtrip:
    """The real invariant: grid -> world -> grid should be identity."""

    @pytest.mark.parametrize("gx,gy", [
        (0, 0), (1, 0), (0, 1), (2, 3), (-1, -1), (5, 5),
    ])
    def test_roundtrip_is_identity(self, gx, gy):
        cell_size = 0.33
        wx, wy = grid_to_world(gx, gy, cell_size)
        rx, ry = world_to_grid(wx, wy, cell_size)
        assert rx == pytest.approx(gx, abs=1e-10)
        assert ry == pytest.approx(gy, abs=1e-10)

    @pytest.mark.parametrize("cell_size", [0.1, 0.25, 0.33, 0.5, 1.0])
    def test_roundtrip_across_cell_sizes(self, cell_size):
        gx, gy = 3, 4
        wx, wy = grid_to_world(gx, gy, cell_size)
        rx, ry = world_to_grid(wx, wy, cell_size)
        assert rx == pytest.approx(gx, abs=1e-10)
        assert ry == pytest.approx(gy, abs=1e-10)

    def test_subcell_coordinates_preserved(self):
        """Sub-cell grid coordinates survive a roundtrip."""
        cell_size = 0.33
        gx, gy = 1.5, 2.7
        wx, wy = grid_to_world(gx, gy, cell_size)
        rx, ry = world_to_grid(wx, wy, cell_size)
        assert rx == pytest.approx(gx, abs=1e-10)
        assert ry == pytest.approx(gy, abs=1e-10)

    def test_negative_coordinates(self):
        cell_size = 0.33
        rx, ry = world_to_grid(-0.33, -0.66, cell_size)
        assert rx == pytest.approx(-1.0, abs=1e-10)
        assert ry == pytest.approx(-2.0, abs=1e-10)


class TestYawFromQuaternion:
    """Non-trivial formula -- easy to swap x/y/z/w or get sign wrong."""

    def test_identity_quaternion(self):
        assert yaw_from_quaternion(0, 0, 0, 1) == pytest.approx(0.0)

    def test_90_degrees_ccw(self):
        z = math.sin(math.pi / 4)
        w = math.cos(math.pi / 4)
        assert yaw_from_quaternion(0, 0, z, w) == pytest.approx(math.pi / 2, abs=1e-6)

    def test_90_degrees_cw(self):
        z = math.sin(-math.pi / 4)
        w = math.cos(-math.pi / 4)
        assert yaw_from_quaternion(0, 0, z, w) == pytest.approx(-math.pi / 2, abs=1e-6)

    def test_180_degrees(self):
        assert yaw_from_quaternion(0, 0, 1, 0) == pytest.approx(math.pi, abs=1e-6)

    # Real TB3 Burger IMU readings captured on the physical robot.
    # These are regression tests against physical sensor data.

    def test_tb3_imu_south(self):
        yaw = yaw_from_quaternion(-0.003, -0.006, -0.042, 0.999)
        assert yaw == pytest.approx(0.0, abs=0.1)

    def test_tb3_imu_east(self):
        yaw = yaw_from_quaternion(-0.011, -0.002, 0.697, 0.717)
        assert yaw == pytest.approx(math.pi / 2, abs=0.1)

    def test_tb3_imu_north(self):
        yaw = yaw_from_quaternion(-0.006, 0.002, 0.9999, 0.009)
        assert yaw == pytest.approx(math.pi, abs=0.1)

    def test_tb3_imu_west(self):
        yaw = yaw_from_quaternion(-0.005, 0.006, 0.715, -0.699)
        assert yaw == pytest.approx(-math.pi / 2, abs=0.15)
