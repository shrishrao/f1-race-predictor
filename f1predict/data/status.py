"""Classify finishing statuses as finished, DNF, DNS or DSQ.

Jolpica changed its wording in 2023. Up to 2022 a lapped car is "+1 Lap", "+2 Laps", ...
and a retirement names its cause ("Engine", "Collision", ...). From 2023 it is mostly just
"Lapped" and "Retired". Both wordings are handled here.

The status alone isn't always enough: Jolpica marks three cars "Withdrew" that started and
retired during the race (Norris 2019 round 18 after 48 laps, Magnussen 2020 round 14 after
55, Alonso 2023 round 19 after 47). So the number of laps completed is checked too.
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


def classify_status(status: str, laps: int | None = None) -> StatusClass:
    """Map a Jolpica status string (and the laps completed) to finished, DNF, DNS or DSQ.

    Finished means the car took the chequered flag: "Finished", "Lapped" (2023 on) or
    "+N Lap(s)" (up to 2022). DNS and DSQ are separate so they can be left out of training
    targets. A non-starter status ("Withdrew", "Did not start") with ``laps`` above 0 is a
    DNF, because the car did start. Every other status counts as a DNF, including classified
    retirees who completed 90% of the distance and still received an official position.
    """
    text = status.strip()
    if text in FINISHED_STATUSES or _LAPS_DOWN.match(text):
        return StatusClass.FINISHED
    if text in DNS_STATUSES:
        return StatusClass.DNF if laps is not None and laps > 0 else StatusClass.DNS
    if text in DSQ_STATUSES:
        return StatusClass.DSQ
    return StatusClass.DNF
