"""Passenger fare computation (see ``01-07-tarif-dan-pendapatan.md``).

A passenger pays only for the **productive** segment of a trip — the distance
from pickup to destination (the ``WITH_PASSENGER`` leg). The idle->pickup leg
(``TO_PASSENGER``) and any other passenger-less driving are **unproductive**
km, recorded as a metric but never billed.

Two fare modes are configurable (``fare_mode``):

- ``PURE``: ``fare = price_per_km * productive_km``.
- ``WITH_BASE_FARE``: ``fare = base_fare + price_per_km * productive_km``.

``price_per_km`` is constant across time; ``base_fare`` applies only in
``WITH_BASE_FARE`` mode.

This module depends only on :mod:`simulation.config`. It never imports
``negotiation``.

Example
-------
>>> from simulation.config.config import FareConfig
>>> trip_fare(FareConfig(fare_mode="PURE", price_per_km=3000.0), 4.0)
12000.0
>>> trip_fare(FareConfig(fare_mode="WITH_BASE_FARE", price_per_km=3000.0,
...                      base_fare=5000.0), 4.0)
17000.0
"""

from __future__ import annotations

from simulation.config.config import FareConfig

_MODE_PURE = "PURE"
_MODE_WITH_BASE_FARE = "WITH_BASE_FARE"


def trip_fare(cfg: FareConfig, productive_km: float) -> float:
    """Compute passenger fare for a completed trip's productive distance.

    Args:
        cfg: The :class:`~simulation.config.config.FareConfig` for this run.
        productive_km: Distance driven with a passenger (pickup -> destination),
            in km (>= 0).

    Returns:
        The fare in Rupiah.

    Raises:
        ValueError: If ``cfg.fare_mode`` is not a recognized fare mode.

    Example:
        >>> from simulation.config.config import FareConfig
        >>> trip_fare(FareConfig(fare_mode="PURE", price_per_km=2500.0), 0.0)
        0.0
        >>> trip_fare(FareConfig(fare_mode="WITH_BASE_FARE", price_per_km=2500.0,
        ...                      base_fare=4000.0), 2.0)
        9000.0
    """

    variable = cfg.price_per_km * productive_km
    if cfg.fare_mode == _MODE_PURE:
        return variable
    if cfg.fare_mode == _MODE_WITH_BASE_FARE:
        return cfg.base_fare + variable
    raise ValueError(
        f"Unknown fare_mode: {cfg.fare_mode!r} (expected {_MODE_PURE!r} or {_MODE_WITH_BASE_FARE!r})"
    )
