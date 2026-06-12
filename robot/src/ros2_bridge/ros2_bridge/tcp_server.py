"""
tcp_server.py — Single-client TCP server with newline-delimited JSON.

Pure Python, no ROS2 dependency.  Designed to be driven by the bridge node:
the node calls ``poll_messages()`` on a timer to drain incoming messages,
and ``send()`` to push feedback/results back to the client.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)


class TcpServer:
    """Single-client TCP server for the bridge wire protocol."""

    def __init__(
        self,
        port: int,
        host: str = "0.0.0.0",
        on_disconnect: Callable[[], None] | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._on_disconnect = on_disconnect

        self._server_sock: socket.socket | None = None
        self._client_sock: socket.socket | None = None
        self._client_addr: tuple[str, int] | None = None

        self._lock = threading.Lock()
        self._inbox: list[dict] = []
        self._recv_buf = ""

        self._running = False
        self._accept_thread: threading.Thread | None = None
        self._recv_thread: threading.Thread | None = None

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        """Bind, listen, and start accepting connections in the background."""
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.settimeout(1.0)
        self._server_sock.bind((self._host, self._port))
        self._server_sock.listen(1)
        self._running = True
        self._accept_thread = threading.Thread(
            target=self._accept_loop, name="tcp-accept", daemon=True,
        )
        self._accept_thread.start()
        logger.info("[tcp] listening on %s:%d", self._host, self._port)

    def stop(self) -> None:
        """Shut down the server and disconnect any client."""
        self._running = False
        self._disconnect_client()
        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except OSError:
                pass
            self._server_sock = None
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=3)
        if self._recv_thread is not None:
            self._recv_thread.join(timeout=3)
        logger.info("[tcp] server stopped")

    # -- public API (called from ROS2 node) -----------------------------------

    def poll_messages(self) -> list[dict]:
        """Return and clear all messages received since the last poll."""
        with self._lock:
            msgs = self._inbox[:]
            self._inbox.clear()
        return msgs

    def send(self, msg: dict) -> bool:
        """Send a JSON message to the connected client.  Returns False on failure."""
        if self._client_sock is None:
            return False
        try:
            payload = json.dumps(msg, separators=(",", ":")) + "\n"
            self._client_sock.sendall(payload.encode("utf-8"))
            return True
        except OSError as e:
            logger.warning("[tcp] send failed: %s", e)
            self._disconnect_client()
            return False

    @property
    def connected(self) -> bool:
        return self._client_sock is not None

    # -- background threads ---------------------------------------------------

    def _accept_loop(self) -> None:
        while self._running:
            try:
                conn, addr = self._server_sock.accept()  # type: ignore[union-attr]
            except socket.timeout:
                continue
            except OSError:
                if self._running:
                    logger.exception("[tcp] accept error")
                break

            # Replace previous client
            self._disconnect_client()
            conn.settimeout(1.0)
            self._client_sock = conn
            self._client_addr = addr
            self._recv_buf = ""
            logger.info("[tcp] client connected from %s:%d", *addr)

            self._recv_thread = threading.Thread(
                target=self._recv_loop, name="tcp-recv", daemon=True,
            )
            self._recv_thread.start()

    def _recv_loop(self) -> None:
        sock = self._client_sock
        while self._running and sock is not None and sock is self._client_sock:
            try:
                data = sock.recv(4096)
                if not data:
                    logger.info("[tcp] client disconnected")
                    self._disconnect_client()
                    return
                self._recv_buf += data.decode("utf-8", errors="replace")
                self._drain_buffer()
            except socket.timeout:
                continue
            except OSError:
                if self._running:
                    logger.warning("[tcp] recv error, client disconnected")
                    self._disconnect_client()
                return

    def _drain_buffer(self) -> None:
        while "\n" in self._recv_buf:
            line, self._recv_buf = self._recv_buf.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("[tcp] malformed JSON: %s", line[:120])
                continue
            with self._lock:
                self._inbox.append(msg)

    def _disconnect_client(self) -> None:
        had_client = self._client_sock is not None
        if self._client_sock is not None:
            try:
                self._client_sock.close()
            except OSError:
                pass
            self._client_sock = None
            self._client_addr = None
        # Fire the user-supplied callback only on genuine live-client loss,
        # not during server shutdown or no-op calls.
        if had_client and self._running and self._on_disconnect is not None:
            try:
                self._on_disconnect()
            except Exception:
                logger.exception("[tcp] on_disconnect callback raised")
