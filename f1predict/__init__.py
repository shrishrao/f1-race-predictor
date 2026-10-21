"""f1predict: post-qualifying Formula 1 race outcome predictor.

``from f1predict import predict`` gives the package API (``f1predict.api.predict``). It is
imported the first time it's used, so ``import f1predict`` (and the command line's
``--help``) doesn't wait for pandas and LightGBM to load.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"
__all__ = ["__version__", "predict"]

if TYPE_CHECKING:
    from f1predict.api import predict


def __getattr__(name: str) -> Any:
    """Load ``predict`` from ``f1predict.api`` the first time it's asked for."""
    if name == "predict":
        from f1predict.api import predict

        return predict
    raise AttributeError(f"module 'f1predict' has no attribute {name!r}")
