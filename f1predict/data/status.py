"""Classify finishing statuses as finished, DNF, DNS or DSQ.

Jolpica changed its wording in 2023. Up to 2022 a lapped car is "+1 Lap", "+2 Laps", ...
and a retirement names its cause ("Engine", "Collision", ...). From 2023 it is mostly just
"Lapped" and "Retired". Both wordings are handled here.
"""

from __future__ import annotations

import re
from enum import StrEnum

# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------


class StatusClass(StrEnum):
    """How a driver's race ended, as used for labels and DNF targets."""

    FINISHED = "finished"
    DNF = "dnf"
    DNS = "dns"
    DSQ = "dsq"


FINISHED_STATUSES = frozenset({"Finished", "Lapped"})
DNS_STATUSES = frozenset({"Did not start", "Withdrew", "Did not qualify", "Did not prequalify"})
DSQ_STATUSES = frozenset({"Disqualified", "Excluded"})
_LAPS_DOWN = re.compile(r"^\+\d+ Laps?$")

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_status(status: str) -> StatusClass:
    """Map a Jolpica status string to finished, DNF, DNS or DSQ.

    Finished means the car took the chequered flag: "Finished", "Lapped" (2023 on) or
    "+N Lap(s)" (up to 2022). DNS and DSQ are separate so they can be left out of training
    targets. Every other status counts as a DNF, including classified retirees who completed
    90% of the distance and still received an official position.
    """
    text = status.strip()
    if text in FINISHED_STATUSES or _LAPS_DOWN.match(text):
        return StatusClass.FINISHED
    if text in DNS_STATUSES:
        return StatusClass.DNS
    if text in DSQ_STATUSES:
        return StatusClass.DSQ
    return StatusClass.DNF
