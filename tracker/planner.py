"""
planner.py - Planner protocol + BridgePlanner adapter over BridgeClient.

The AutoDriver consumes a `Planner`, not a BridgeClient, so alternative
backends (local A*, mock planner for tests, offline replay) can be swapped
in without touching the state machine.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypedDict

from bridge_client import BridgeClient


class PredictRequest(TypedDict):
    obstacle_map: list[list[float]]
    energy_map:   list[list[float]]
    robot_pos:    tuple[int, int]
    goal_pos:     tuple[int, int]


@dataclass(frozen=True)
class PredictResponse:
    action: int                       # 0..7
    direction: tuple[int, int]        # bridge-frame (dx, dy)


class Planner(Protocol):
    """Fire-and-forget planner interface.

    Callers register on_result / on_error callbacks once up front and then
    call request() for each predict.  Only one request is expected in flight
    at a time (the state machine gates this).
    """
    name: str

    def request(self, req: PredictRequest, sequence: int | None = None) -> None: ...
    def on_result(self, cb: Callable[[PredictResponse], None]) -> None: ...
    def on_error(self, cb: Callable[[str], None]) -> None: ...


class BridgePlanner:
    """Planner impl that routes through the existing BridgeClient."""

    name = "BridgePlanner"

    def __init__(self, bridge: BridgeClient) -> None:
        self._bridge = bridge
        self._result_cb: Callable[[PredictResponse], None] | None = None
        self._error_cb:  Callable[[str], None] | None = None

        bridge.on("predict_result", self._handle_predict_result)
        bridge.on("error", self._handle_error)

    # -- Planner protocol ------------------------------------------------------

    def request(self, req: PredictRequest, sequence: int | None = None) -> None:
        self._bridge.send_predict(
            obstacle_map=req["obstacle_map"],
            energy_map=req["energy_map"],
            robot_pos=req["robot_pos"],
            goal_pos=req["goal_pos"],
            sequence=sequence,
        )

    def on_result(self, cb: Callable[[PredictResponse], None]) -> None:
        self._result_cb = cb

    def on_error(self, cb: Callable[[str], None]) -> None:
        self._error_cb = cb

    # -- Bridge callbacks ------------------------------------------------------

    def _handle_predict_result(self, msg: dict) -> None:
        if self._result_cb is None:
            return
        try:
            action = int(msg["action"])
            dx, dy = msg["direction"]
            resp = PredictResponse(action=action, direction=(int(dx), int(dy)))
        except (KeyError, TypeError, ValueError) as e:
            if self._error_cb is not None:
                self._error_cb(f"malformed predict_result: {e}")
            return
        self._result_cb(resp)

    def _handle_error(self, msg: dict) -> None:
        if self._error_cb is None:
            return
        self._error_cb(str(msg.get("message", "unknown bridge error")))
