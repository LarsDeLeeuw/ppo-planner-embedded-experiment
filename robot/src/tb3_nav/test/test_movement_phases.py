"""Unit tests for the movement state machine.

Tests focus on transition logic, boundary conditions, and lifecycle
correctness -- the things that can actually break.
"""

import pytest
from tb3_nav.movement_phases import Phase, MovementStateMachine


HEADING_TOL = 0.02   # radians
DISTANCE_TOL = 0.02  # meters


@pytest.fixture
def sm():
    return MovementStateMachine(
        heading_tolerance=HEADING_TOL,
        distance_tolerance=DISTANCE_TOL,
    )


def advance_to_driving(sm):
    """Helper: get the state machine into DRIVING phase."""
    sm.start_move(target_heading=1.0, target_distance=0.33)
    sm.update(heading_error=0.0, distance_remaining=0.33)   # -> SETTLING
    assert sm.phase == Phase.SETTLING
    sm.settle_complete()                                     # -> DRIVING
    assert sm.phase == Phase.DRIVING


def advance_to_finalizing(sm):
    """Helper: get the state machine into FINALIZING phase."""
    sm.start_move(target_heading=1.0, target_distance=0.33, final_heading=0.0)
    sm.update(heading_error=0.0, distance_remaining=0.33)   # -> SETTLING
    sm.settle_complete()                                     # -> DRIVING
    sm.update(heading_error=0.0, distance_remaining=0.001)  # -> FINALIZING
    assert sm.phase == Phase.FINALIZING


class TestStartMove:
    def test_normal_move_starts_rotating(self, sm):
        assert sm.start_move(target_heading=1.57, target_distance=0.33) == Phase.ROTATING

    def test_zero_distance_skips_to_done(self, sm):
        assert sm.start_move(target_heading=0, target_distance=0.0) == Phase.DONE

    def test_zero_distance_with_final_heading_goes_to_finalizing(self, sm):
        assert sm.start_move(target_heading=0, target_distance=0.0,
                             final_heading=1.57) == Phase.FINALIZING

    def test_rejects_start_from_non_idle(self, sm):
        sm.start_move(target_heading=0, target_distance=0.33)
        with pytest.raises(RuntimeError):
            sm.start_move(target_heading=0, target_distance=0.33)


class TestRotatingPhase:
    def test_stays_rotating_above_tolerance(self, sm):
        sm.start_move(target_heading=1.57, target_distance=0.33)
        assert sm.update(heading_error=0.5, distance_remaining=0.33) == Phase.ROTATING

    def test_transitions_to_settling_within_tolerance(self, sm):
        sm.start_move(target_heading=1.57, target_distance=0.33)
        assert sm.update(heading_error=0.01, distance_remaining=0.33) == Phase.SETTLING

    def test_error_exactly_at_tolerance_does_not_transition(self, sm):
        sm.start_move(target_heading=1.57, target_distance=0.33)
        assert sm.update(heading_error=HEADING_TOL, distance_remaining=0.33) == Phase.ROTATING

    def test_negative_heading_error_uses_absolute_value(self, sm):
        sm.start_move(target_heading=1.57, target_distance=0.33)
        assert sm.update(heading_error=-0.01, distance_remaining=0.33) == Phase.SETTLING


class TestDrivingPhase:
    def test_stays_driving_with_distance_remaining(self, sm):
        advance_to_driving(sm)
        assert sm.update(heading_error=0.0, distance_remaining=0.2) == Phase.DRIVING

    def test_transitions_to_done(self, sm):
        advance_to_driving(sm)
        assert sm.update(heading_error=0.0, distance_remaining=0.001) == Phase.DONE

    def test_transitions_to_finalizing_when_final_heading_set(self, sm):
        sm.start_move(target_heading=0, target_distance=0.33, final_heading=1.57)
        sm.update(heading_error=0.0, distance_remaining=0.33)  # -> SETTLING
        sm.settle_complete()                                    # -> DRIVING
        assert sm.update(heading_error=0.0, distance_remaining=0.001) == Phase.FINALIZING


class TestFinalizingPhase:
    def test_stays_finalizing_above_tolerance(self, sm):
        advance_to_finalizing(sm)
        assert sm.update(heading_error=0.5, distance_remaining=0.0) == Phase.FINALIZING

    def test_transitions_to_done(self, sm):
        advance_to_finalizing(sm)
        assert sm.update(heading_error=0.01, distance_remaining=0.0) == Phase.DONE


class TestTerminalStates:
    """update() in terminal/inactive states must be a no-op."""

    def test_update_in_idle_is_noop(self, sm):
        assert sm.update(heading_error=0.0, distance_remaining=0.0) == Phase.IDLE

    def test_update_in_done_is_noop(self, sm):
        sm.start_move(target_heading=0, target_distance=0.0)
        assert sm.phase == Phase.DONE
        assert sm.update(heading_error=0.0, distance_remaining=0.0) == Phase.DONE

    def test_update_in_aborted_is_noop(self, sm):
        sm.start_move(target_heading=0, target_distance=0.33)
        sm.abort()
        assert sm.update(heading_error=0.0, distance_remaining=0.0) == Phase.ABORTED


class TestLifecycle:
    def test_abort_stops_progress(self, sm):
        advance_to_driving(sm)
        sm.abort()
        assert sm.phase == Phase.ABORTED

    def test_reset_allows_new_move(self, sm):
        sm.start_move(target_heading=0, target_distance=0.33)
        sm.abort()
        sm.reset()
        assert sm.start_move(target_heading=1.0, target_distance=0.5) == Phase.ROTATING

    def test_full_sequence(self, sm):
        """Walk through the complete happy path."""
        sm.start_move(target_heading=1.57, target_distance=0.33)
        assert sm.phase == Phase.ROTATING

        sm.update(heading_error=0.8, distance_remaining=0.33)
        assert sm.phase == Phase.ROTATING

        sm.update(heading_error=0.01, distance_remaining=0.33)
        assert sm.phase == Phase.SETTLING

        sm.settle_complete()
        assert sm.phase == Phase.DRIVING

        sm.update(heading_error=0.01, distance_remaining=0.2)
        assert sm.phase == Phase.DRIVING

        sm.update(heading_error=0.01, distance_remaining=0.01)
        assert sm.phase == Phase.DONE


class TestSettlingPhase:
    def test_settle_complete_transitions_to_driving(self, sm):
        sm.start_move(target_heading=1.57, target_distance=0.33)
        sm.update(heading_error=0.0, distance_remaining=0.33)
        assert sm.phase == Phase.SETTLING
        assert sm.settle_complete() == Phase.DRIVING

    def test_settle_complete_from_wrong_phase_raises(self, sm):
        sm.start_move(target_heading=1.57, target_distance=0.33)
        assert sm.phase == Phase.ROTATING
        with pytest.raises(RuntimeError):
            sm.settle_complete()

    def test_update_in_settling_is_noop(self, sm):
        sm.start_move(target_heading=1.57, target_distance=0.33)
        sm.update(heading_error=0.0, distance_remaining=0.33)
        assert sm.phase == Phase.SETTLING
        # update() should not change phase -- settling is time-based
        sm.update(heading_error=0.0, distance_remaining=0.0)
        assert sm.phase == Phase.SETTLING

    def test_back_to_rotating_transitions(self, sm):
        sm.start_move(target_heading=1.57, target_distance=0.33)
        sm.update(heading_error=0.0, distance_remaining=0.33)
        assert sm.phase == Phase.SETTLING
        assert sm.back_to_rotating() == Phase.ROTATING

    def test_back_to_rotating_from_wrong_phase_raises(self, sm):
        sm.start_move(target_heading=1.57, target_distance=0.33)
        assert sm.phase == Phase.ROTATING
        with pytest.raises(RuntimeError):
            sm.back_to_rotating()

    def test_back_to_rotating_then_settle_then_drive(self, sm):
        """Re-rotation loop: ROTATING -> SETTLING -> ROTATING -> SETTLING -> DRIVING."""
        sm.start_move(target_heading=1.57, target_distance=0.33)
        sm.update(heading_error=0.0, distance_remaining=0.33)
        assert sm.phase == Phase.SETTLING

        sm.back_to_rotating()
        assert sm.phase == Phase.ROTATING

        sm.update(heading_error=0.0, distance_remaining=0.33)
        assert sm.phase == Phase.SETTLING

        sm.settle_complete()
        assert sm.phase == Phase.DRIVING
