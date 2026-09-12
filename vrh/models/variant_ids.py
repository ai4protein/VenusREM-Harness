"""Named ProtSSN (k, h) and CARP size keys (no torch)."""

from __future__ import annotations

import re
from typing import Optional

PROTSSN_K_VALUES = (10, 20, 30)
PROTSSN_H_VALUES = (512, 768, 1280)
PROTSSN_CONFIGS = tuple((k, h) for k in PROTSSN_K_VALUES for h in PROTSSN_H_VALUES)

CARP_VARIANTS = (
    ("carp-600k", "carp_600k", "CARP-600k"),
    ("carp-38m", "carp_38M", "CARP-38M"),
    ("carp-76m", "carp_76M", "CARP-76M"),
)

_PROTSSN_KH = re.compile(r"k(10|20|30)_h(512|768|1280)")


def protssn_variant_name(k: int, h: int) -> str:
    return f"protssn-k{k}-h{h}"


def protssn_variant_id(k: int, h: int) -> str:
    return f"protssn_k{k}_h{h}"


def parse_protssn_config(name: Optional[str]) -> Optional[tuple[int, int]]:
    raw = (name or "").lower().replace("-", "_")
    match = _PROTSSN_KH.search(raw)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))
