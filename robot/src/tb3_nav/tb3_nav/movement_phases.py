"""State machine for grid movement execution.

No ROS dependencies -- pure Python, unit-testable.

The state machine manages the phases of a single grid move:
  IDLE -> ROTATING -> DRIVING -> DONE

An optional FINALIZING phase exists for post-drive heading adjustment.
"""

from enum import Enum, auto


class Phase(Enum):
    IDLE = auto()
    ROTATING = auto()
    SETTLING = auto()
    DRIVING = auto()
    FINALIZING = auto()
    DONE = auto()
    ABORTED = auto()


class MovementStateMachine:
    """Manages phase transitions for a rotate-then-drive maneuver.

    Usage:
        sm = MovementStateMachine(heading_tol=0.05, distance_tol=0.02)
        sm.start_move(target_heading=1.57, target_distance=0.33)
        while sm.phase not in (Phase.DONE, Phase.ABORTED):
            phase = sm.update(heading_error, distance_remaining)
            # ... act on phase ...
        sm.reset()
    """

    def __init__(self, heading_tolerance: float, distance_tolerance: float):
        self._heading_tolerance = heading_tolerance
        self._distance_tolerance = distance_tolerance
        self._phase = Phase.IDLE
        self._target_heading: float = 0.0
        self._target_distance: float = 0.0
        self._final_heading: float | None = None

    @property
    def phase(self) -> Phase:
        return self._phase

    def start_move(self, target_heading: float, target_distance: float,
                   final_heading: float | None = None) -> Phase:
        """Begin a new move. Transitions from IDLE to ROTATING.

        Args:
            target_heading: heading to face before driving (radians).
            target_distance: distance to drive (meters).
            final_heading: optional heading to face after driving.
                           If None, FINALIZING phase is skipped.

        Returns:
            The new phase (ROTATING, or DONE if distance is zero).
        """
        if self._phase != Phase.IDLE:
            raise RuntimeError(
                f"Cannot start move from phase {self._phase.name}; "
                f"must be IDLE (call reset() first)"
            )
        self._target_heading = target_heading
        self._target_distance = target_distance
        self._final_heading = final_heading

        # Skip rotation + drive if target is the current cell
        if target_distance < self._distance_tolerance:
            if final_heading is not None:
                self._phase = Phase.FINALIZING
            else:
                self._phase = Phase.DONE
        else:
            self._phase = Phase.ROTATING

        return self._phase

    def update(self, heading_error: float, distance_remaining: float) -> Phase:
        """Evaluate transition conditions and advance the phase if met.

        Args:
            heading_error: signed error from desired heading (radians).
                           During ROTATING: error from target_heading.
                           During DRIVING: error from target_heading (for correction).
                           During FINALIZING: error from final_heading.
            distance_remaining: remaining distance to target (meters).

        Returns:
            The current (possibly updated) phase.
        """
        if self._phase == Phase.ROTATING:
            if abs(heading_error) < self._heading_tolerance:
                self._phase = Phase.SETTLING

        elif self._phase == Phase.DRIVING:
            if distance_remaining < self._distance_tolerance:
                if self._final_heading is not None:
                    self._phase = Phase.FINALIZING
                else:
                    self._phase = Phase.DONE

        elif self._phase == Phase.FINALIZING:
            if abs(heading_error) < self._heading_tolerance:
                self._phase = Phase.DONE

        return self._phase

    def settle_complete(self) -> Phase:
        """Transition from SETTLING to DRIVING.

        Called by the control loop after the settling period has elapsed
        and heading error is within tolerance.
        """
        if self._phase != Phase.SETTLING:
            raise RuntimeError(
                f"settle_complete() called in phase {self._phase.name}; "
                f"must be SETTLING"
            )
        self._phase = Phase.DRIVING
        return self._phase

    def back_to_rotating(self) -> Phase:
        """Transition from SETTLING back to ROTATING.

        Called when heading error after settling is still too large and
        the robot needs to re-align before driving.
        """
        if self._phase != Phase.SETTLING:
            raise RuntimeError(
                f"back_to_rotating() called in phase {self._phase.name}; "
                f"must be SETTLING"
            )
        self._phase = Phase.ROTATING
        return self._phase

    def abort(self) -> None:
        """Abort the current move."""
        self._phase = Phase.ABORTED

    def reset(self) -> None:
        """Reset to IDLE, ready for a new move."""
        self._phase = Phase.IDLE
        self._target_heading = 0.0
        self._target_distance = 0.0
        self._final_heading = None
