"""Unit tests for the motion controller.

These test the logic most likely to cause real-world misbehavior:
deadzone handling, speed clamping, direction preservation, and the
interaction between linear and angular commands during driving.
"""

import math
import pytest
from tb3_nav.motion_controller import (
    ControlParams,
    compute_rotation_cmd,
    compute_drive_cmd,
)


@pytest.fixture
def params():
    return ControlParams(
        kp_angular=2.0,
        kp_linear=1.0,
        kp_heading_correction=0.2,
        max_angular_speed=0.8,
        min_angular_speed=0.15,
        max_linear_speed=0.15,
        min_linear_speed=0.05,
        heading_tolerance=0.02,
        distance_tolerance=0.02,
    )


class TestRotationDeadzone:
    """Below tolerance -> zero. Above tolerance -> at least min_speed."""

    def test_within_tolerance_returns_zero(self, params):
        assert compute_rotation_cmd(0.01, params) == 0.0

    def test_at_tolerance_boundary_returns_zero(self, params):
        # Exactly at tolerance (abs(0.019) < 0.02) -> still zero
        assert compute_rotation_cmd(0.019, params) == 0.0

    def test_just_above_tolerance_returns_at_least_min_speed(self, params):
        result = compute_rotation_cmd(0.021, params)
        assert abs(result) >= params.min_angular_speed

    def test_small_error_above_tolerance_clamps_to_min(self, params):
        # kp=2.0 * 0.06 = 0.12, which is below min_angular_speed=0.15
        # Should be clamped up to min_angular_speed
        result = compute_rotation_cmd(0.06, params)
        assert abs(result) == pytest.approx(params.min_angular_speed)


class TestRotationClamping:
    """Large errors must not exceed max_speed."""

    def test_large_positive_error_clamped(self, params):
        result = compute_rotation_cmd(3.0, params)
        assert result == pytest.approx(params.max_angular_speed)

    def test_large_negative_error_clamped(self, params):
        result = compute_rotation_cmd(-3.0, params)
        assert result == pytest.approx(-params.max_angular_speed)


class TestRotationDirection:
    """Command direction must match error direction -- wrong sign = spin wrong way."""

    def test_positive_error_gives_positive_command(self, params):
        assert compute_rotation_cmd(0.5, params) > 0

    def test_negative_error_gives_negative_command(self, params):
        assert compute_rotation_cmd(-0.5, params) < 0

    def test_direction_preserved_near_deadzone(self, params):
        # Small error just above tolerance -- direction must still be correct
        assert compute_rotation_cmd(0.06, params) > 0
        assert compute_rotation_cmd(-0.06, params) < 0


class TestRotationProportionality:
    """Larger errors produce faster (or equal) commands, up to the clamp."""

    def test_larger_error_gives_faster_or_equal_speed(self, params):
        small = abs(compute_rotation_cmd(0.2, params))
        large = abs(compute_rotation_cmd(0.5, params))
        assert large >= small


class TestDriveDeadzone:
    """Distance within tolerance -> full stop (0, 0)."""

    def test_within_tolerance_returns_zero(self, params):
        assert compute_drive_cmd(0.01, 0.0, params) == (0.0, 0.0)

    def test_at_tolerance_boundary_returns_zero(self, params):
        assert compute_drive_cmd(0.019, 0.0, params) == (0.0, 0.0)

    def test_just_above_tolerance_returns_nonzero_linear(self, params):
        lin, _ = compute_drive_cmd(0.021, 0.0, params)
        assert lin >= params.min_linear_speed


class TestDriveLinearClamping:
    """Linear speed must stay within [min, max] when above tolerance."""

    def test_small_distance_clamps_to_min(self, params):
        # kp=1.0 * 0.03 = 0.03, below min_linear_speed=0.05
        lin, _ = compute_drive_cmd(0.03, 0.0, params)
        assert lin == pytest.approx(params.min_linear_speed)

    def test_large_distance_clamps_to_max(self, params):
        lin, _ = compute_drive_cmd(5.0, 0.0, params)
        assert lin == pytest.approx(params.max_linear_speed)

    def test_linear_is_always_positive(self, params):
        # Linear command should never be negative (robot drives forward)
        for dist in [0.03, 0.1, 0.33, 1.0, 5.0]:
            lin, _ = compute_drive_cmd(dist, 0.0, params)
            assert lin > 0


class TestDriveHeadingCorrection:
    """Small angular correction during driving to stay on course."""

    def test_zero_heading_error_gives_zero_angular(self, params):
        _, ang = compute_drive_cmd(0.2, 0.0, params)
        assert ang == pytest.approx(0.0)

    def test_positive_drift_gives_positive_correction(self, params):
        _, ang = compute_drive_cmd(0.2, 0.3, params)
        assert ang > 0

    def test_negative_drift_gives_negative_correction(self, params):
        _, ang = compute_drive_cmd(0.2, -0.3, params)
        assert ang < 0

    def test_correction_proportional_to_error(self, params):
        _, small_ang = compute_drive_cmd(0.2, 0.1, params)
        _, large_ang = compute_drive_cmd(0.2, 0.5, params)
        assert abs(large_ang) > abs(small_ang)

    def test_correction_clamped_to_max_angular(self, params):
        _, ang = compute_drive_cmd(0.2, 100.0, params)
        assert abs(ang) <= params.max_angular_speed


class TestDriveHeadingCorrectionDoesNotAffectLinear:
    """Heading drift should not change the linear speed."""

    def test_linear_same_regardless_of_heading_error(self, params):
        lin_no_drift, _ = compute_drive_cmd(0.2, 0.0, params)
        lin_with_drift, _ = compute_drive_cmd(0.2, 0.5, params)
        assert lin_no_drift == pytest.approx(lin_with_drift)
