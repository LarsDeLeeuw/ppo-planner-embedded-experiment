"""
frame_source.py — Threaded, self-healing camera capture.

The main capture loop in main.py is single-threaded and, during a campaign,
the orchestrator's blocking SSH/scp steps run *inline* in that loop (see
orchestrator.orchestrator's docstring).  For a local USB camera, not calling
``cap.read()`` for a few seconds just drops frames harmlessly.  For a *network*
stream (camera.index is a URL), the backend's internal buffer overruns and the
server drops the connection — the next ``cap.read()`` returns False and the old
code ``break``ed out of the loop, taking the whole app down ("lost webstream").

``ThreadedFrameSource`` decouples frame acquisition from the main loop:

  - A daemon thread reads frames as fast as the stream delivers them and keeps
    only the *latest* one.  It keeps draining the stream even while the main
    thread is blocked in ``orch.tick()``, so the connection never goes stale.
  - On a read failure it releases and reopens the ``VideoCapture`` with capped
    backoff — a transient drop self-heals instead of crashing the app.
  - ``read()`` is non-blocking and hands the consumer a private copy of the
    latest frame, so the caller can draw overlays on it without racing the
    grabber or re-drawing a frame it already annotated.

This is "Fix A": it stops the *crash*.  The main thread can still visibly
freeze for the duration of a blocking ``orch.tick()`` step (the GUI isn't
pumped) — decoupling that is "Fix B" and is intentionally out of scope here.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np

# Cap the reconnect backoff so a long outage doesn't stretch retry gaps to
# minutes — the operator wants the stream back the instant the camera returns.
_MAX_BACKOFF_S = 5.0


class ThreadedFrameSource:
    """Background-threaded wrapper around cv2.VideoCapture with auto-reconnect.

    Usage::

        src = ThreadedFrameSource(cfg.camera.index)
        if not src.start():
            sys.exit(1)
        frame = src.wait_first_frame(timeout_s=10.0)
        ...
        while True:
            ok, frame = src.read()
            if not ok:
                continue          # no frame yet; grabber is (re)connecting
            ...
        src.release()
    """

    def __init__(
        self,
        index: int | str,
        *,
        reconnect: bool = True,
        reconnect_backoff_s: float = 1.0,
        buffer_size: int = 1,
    ) -> None:
        self._index = index
        self._reconnect = reconnect
        self._backoff_s = max(0.1, float(reconnect_backoff_s))
        self._buffer_size = int(buffer_size)

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None    # latest frame (grabber-owned)
        self._frame_seq = 0                          # increments per new frame
        self._last_read_seq = 0                      # seq returned by last read()

        # Health, readable from the main thread for HUD/logging.
        self._connected = False
        self._consecutive_failures = 0
        self._last_error: str = ""

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> bool:
        """Open the capture and launch the grabber thread.

        Returns False if the device/URL can't be opened at all (so the caller
        can fail fast at startup); a later transient drop is handled by the
        thread's reconnect loop, not by returning False here.
        """
        if not self._open():
            return False
        self._thread = threading.Thread(
            target=self._run, name="frame-grabber", daemon=True)
        self._thread.start()
        return True

    def wait_first_frame(self, timeout_s: float = 10.0) -> Optional[np.ndarray]:
        """Block until the first frame arrives (for startup camera-matrix /
        overlay sizing).  Returns a copy of the frame, or None on timeout."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if self._frame is not None:
                    return self._frame.copy()
            if self._stop.is_set():
                break
            time.sleep(0.02)
        return None

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """Return ``(ok, frame)`` for the most recent frame, non-blocking.

        ``ok`` is False only when no frame has ever been received (startup
        race, or the stream is down with no prior frame).  Once frames have
        flowed, ``read()`` keeps returning the last good frame even while the
        grabber is mid-reconnect — the loop stays alive and the worst case is a
        momentarily frozen image, not a crash.  The returned array is a private
        copy, safe for the caller to mutate.
        """
        with self._lock:
            if self._frame is None:
                return False, None
            self._last_read_seq = self._frame_seq
            return True, self._frame.copy()

    def is_fresh(self) -> bool:
        """True if a new frame has arrived since the previous ``read()``."""
        with self._lock:
            return self._frame_seq != self._last_read_seq

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def last_error(self) -> str:
        return self._last_error

    def release(self) -> None:
        """Stop the grabber thread and release the capture."""
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=2.0)
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    # -- internals -----------------------------------------------------------

    def _open(self) -> bool:
        """(Re)open the VideoCapture.  Returns True on success."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        cap = cv2.VideoCapture(self._index)
        if not cap.isOpened():
            self._last_error = f"cannot open camera {self._index!r}"
            self._connected = False
            return False
        # Keep the backend's internal queue shallow so, after a stall, we get
        # the freshest frame rather than draining a backlog of stale ones.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, self._buffer_size)
        except Exception:
            pass
        self._cap = cap
        self._connected = True
        return True

    def _run(self) -> None:
        """Grabber loop: read latest frame; reconnect with backoff on failure."""
        backoff = self._backoff_s
        while not self._stop.is_set():
            cap = self._cap
            if cap is None:
                if not self._reconnect:
                    break
                self._sleep_backoff(backoff)
                backoff = min(_MAX_BACKOFF_S, backoff * 2)
                if self._open():
                    backoff = self._backoff_s
                continue

            ok, frame = cap.read()
            if ok and frame is not None:
                with self._lock:
                    self._frame = frame
                    self._frame_seq += 1
                self._consecutive_failures = 0
                self._connected = True
                backoff = self._backoff_s
                continue

            # Read failed — the stream dropped (or the device hiccuped).
            self._consecutive_failures += 1
            self._connected = False
            self._last_error = f"read() failed (x{self._consecutive_failures})"
            if not self._reconnect:
                break
            try:
                cap.release()
            except Exception:
                pass
            self._cap = None
            # Next loop iteration takes the reconnect branch above (with backoff).

    def _sleep_backoff(self, seconds: float) -> None:
        """Sleep in small slices so release() interrupts promptly."""
        end = time.monotonic() + seconds
        while time.monotonic() < end and not self._stop.is_set():
            time.sleep(0.05)
