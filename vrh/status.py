"""Heartbeat on stderr so vrh does not look frozen during imports / loads."""

from __future__ import annotations

import os
import sys
import threading
from typing import Optional, TextIO


def _is_tty(stream: TextIO) -> bool:
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def spinner_enabled(stream: Optional[TextIO] = None) -> bool:
    from vrh.env import first_env

    if first_env("VRH_NO_SPINNER", "VRH_DISABLE_TQDM", "REM2_NO_SPINNER", "REM2_DISABLE_TQDM"):
        return False
    return _is_tty(stream or sys.stderr)


class working:
    """Write ``message ...`` and animate the dots until the block ends."""

    def __init__(self, message: str = "working", stream: Optional[TextIO] = None):
        self.message = message
        self.stream = stream or sys.stderr
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._animate = spinner_enabled(self.stream)

    def __enter__(self) -> "working":
        if not self._animate:
            self.stream.write(f"{self.message} ...\n")
            self.stream.flush()
            return self
        self._thread = threading.Thread(target=self._spin, name="vrh-status", daemon=True)
        self._thread.start()
        return self

    def _spin(self) -> None:
        n = 0
        while not self._stop.wait(0.35):
            n = n % 3 + 1
            dots = "." * n
            pad = " " * (3 - n)
            self.stream.write(f"\r{self.message} {dots}{pad}")
            self.stream.flush()

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=1.0)
            end = "failed" if exc_type is not None else "done"
            self.stream.write(f"\r{self.message} ... {end}\n")
            self.stream.flush()
        return False
