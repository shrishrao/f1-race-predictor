"""Constructor lineage: which Jolpica constructor ids are the same team under different names.

Teams rebrand without changing factory, staff or car concept (Force India became Racing
Point and then Aston Martin). Team form features follow the lineage, so a rebrand doesn't
wipe a team's recent history. Every id must be listed here: an unknown id raises instead of
quietly starting a new team with no history, so a new entrant is a deliberate decision.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# The map
# ---------------------------------------------------------------------------

LINEAGE = {
    "force_india": "aston_martin",
    "racing_point": "aston_martin",
    "aston_martin": "aston_martin",
    "toro_rosso": "rb",
    "alphatauri": "rb",
    "rb": "rb",
    "sauber": "audi",
    "alfa": "audi",
    "audi": "audi",
    "renault": "alpine",
    "alpine": "alpine",
    "ferrari": "ferrari",
    "mercedes": "mercedes",
    "red_bull": "red_bull",
    "mclaren": "mclaren",
    "williams": "williams",
    "haas": "haas",
    "cadillac": "cadillac",
}


class UnknownConstructorError(KeyError):
    """A constructor id that is not in the lineage map."""


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


def team_lineage(constructor_id: str) -> str:
    """Return the lineage id for a Jolpica constructor id (e.g. ``"alfa"`` -> ``"audi"``)."""
    try:
        return LINEAGE[constructor_id]
    except KeyError:
        raise UnknownConstructorError(
            f"Constructor {constructor_id!r} is not in the lineage map "
            "(f1predict/data/lineage.py); add it as a new team or as part of an existing one."
        ) from None
