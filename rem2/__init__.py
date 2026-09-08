"""rem2 / VenusREM2: training-free PLM calibration for variant-effect scoring."""

from __future__ import annotations

__version__ = "0.2.0"


def score(*args, **kwargs):
    from rem2.api import score as _score

    return _score(*args, **kwargs)


__all__ = ["__version__", "score"]
