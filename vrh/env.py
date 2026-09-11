"""Environment lookup with VRH_* first and rem2 / VenusREM2 fallbacks."""

from __future__ import annotations

import os


def first_env(*names: str) -> str:
    for name in names:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return ""
