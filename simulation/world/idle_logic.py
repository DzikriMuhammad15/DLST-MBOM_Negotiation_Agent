"""Idle decision + reachability helpers for taxis (see ``01-04``).

An idle (``IDLE_WAITING``) taxi behaves as follows after finishing work:

- If it can reach the nearest marking area and still arrive with ``SoC >=
  B_area`` (:func:`nearest_reachable_area`), it repositions there to keep
  serving orders.
- Otherwise it stays put and becomes a **charge candidate** for the next
  negotiation — *provided* it can still reach at least one port
  (:func:`can_reach_any_port`). A stuck taxi that can reach **no** port is
  handled as an emergency (``simulation.bridge.emergency``).

Candidacy is therefore reachability-based, **not** SoC-threshold-based: there is
no candidate cap and no urgency ranking. ``soc_zone`` / ``urgency_score`` survive
only as a feature signal and an order-matching eligibility test.

This module provides pure helpers used by the state machine and the bridge:
:func:`soc_zone`, :func:`urgency_score`, :func:`reachable_area_with_orders`,
:func:`nearest_reachable_area`, and :func:`can_reach_any_port`. It depends only
on :mod:`simulation.config` and :mod:`simulation.world.*`; it never imports
``negotiation``.

``cycles_waited`` semantics (Gotcha G7): this module never mutates it. It is
raised by exactly one event — a taxi joined a negotiation and won no slot —
which lives in the bridge (``handle_no_deal``), and reset to 0 on a slot grant
(``apply_deal``).

Example
-------
>>> from simulation.config.config import BatteryThresholdConfig
>>> bt = BatteryThresholdConfig(B_charge=0.3, B_area=0.5)
>>> soc_zone(0.6, bt)
'safe'
>>> soc_zone(0.4, bt)
'middle'
>>> soc_zone(0.2, bt)
'critical'
"""

from __future__ import annotations

import math
from enum import Enum
from typing import List, Optional, Tuple

from simulation.config.config import BatteryThresholdConfig
from simulation.world.entities import MarkingArea, Taxi
from simulation.world.world_state import WorldState

# Weight applied to the battery-shortfall term of the urgency score so that any
# taxi in the critical zone outranks any taxi in the middle zone regardless of
# cycles_waited (the cycles term saturates at g_max <= this offset is enforced
# by construction; see urgency_score docstring).
_CRITICAL_PRIORITY_BASE: float = 1.0


class SoCZone(str, Enum):
    """The three SoC zones (string-valued for readable comparison/logging).

    Members:
        SAFE: ``SoC >= B_area``.
        MIDDLE: ``B_charge <= SoC < B_area``.
        CRITICAL: ``SoC < B_charge``.

    Example:
        >>> SoCZone.SAFE.value
        'safe'
    """

    SAFE = "safe"
    MIDDLE = "middle"
    CRITICAL = "critical"


def soc_zone(soc: float, battery: BatteryThresholdConfig) -> str:
    """Classify a SoC into ``"safe"`` / ``"middle"`` / ``"critical"``.

    Args:
        soc: State of charge in ``[0, 1]``.
        battery: Threshold configuration (requires ``B_charge < B_area``).

    Returns:
        One of ``"safe"``, ``"middle"``, ``"critical"`` (the
        :class:`SoCZone` string values).

    Example:
        >>> from simulation.config.config import BatteryThresholdConfig
        >>> bt = BatteryThresholdConfig(B_charge=0.3, B_area=0.5)
        >>> soc_zone(0.5, bt)
        'safe'
        >>> soc_zone(0.3, bt)
        'middle'
        >>> soc_zone(0.29, bt)
        'critical'
    """

    if soc >= battery.B_area:
        return SoCZone.SAFE.value
    if soc >= battery.B_charge:
        return SoCZone.MIDDLE.value
    return SoCZone.CRITICAL.value


def urgency_score(taxi: Taxi, world: WorldState) -> float:
    """Compute a taxi's charging urgency from ``cycles_waited`` and low SoC.

    Retained as a per-taxi **feature** signal (``taxi.urgency``, see ``01-04``);
    it no longer orders or caps candidates (every stuck, port-reachable idle taxi
    joins the negotiation).

    It combines two monotone terms built from spec primitives:

    - **battery shortfall** ``s = max(0, (B_area - soc) / B_area)`` in ``[0, 1]``,
      multiplied by ``_CRITICAL_PRIORITY_BASE``. A critical-zone taxi
      (``soc < B_charge``) always has a strictly larger shortfall than any
      middle-zone taxi, so this term enforces "critical above middle".
    - **anti-starvation** ``g(c) = g_max * (1 - exp(-c / tau))`` — the same
      saturating UNASSIGNED penalty used by the fleet ufun (``04-02``), with
      ``g_max`` and ``tau`` taken from ``ufun.shape``. It is bounded by
      ``g_max`` so it breaks ties within a zone without ever flipping the
      critical/middle ordering when ``g_max <= _CRITICAL_PRIORITY_BASE`` (the
      typical configured regime).

    Args:
        taxi: The taxi to score.
        world: The world (for the battery thresholds and ufun shape config).

    Returns:
        A non-negative urgency score; higher means more urgent.

    Example:
        >>> from simulation.config.default import default_config
        >>> from simulation.world.routing import Router
        >>> from simulation.world.entities import Taxi
        >>> w = WorldState(config=default_config(), router=Router(speed_kmh=30.0))
        >>> t_low = Taxi(id="a", position=(0.0, 0.0), soc=0.1)
        >>> t_mid = Taxi(id="b", position=(0.0, 0.0), soc=0.45)
        >>> urgency_score(t_low, w) > urgency_score(t_mid, w)
        True
    """

    battery = world.config.battery
    shape = world.config.ufun.shape

    b_area = battery.B_area if battery.B_area > 0.0 else 1.0
    shortfall = max(0.0, (b_area - taxi.soc) / b_area)
    shortfall = min(1.0, shortfall)

    cycles = max(0, taxi.cycles_waited)
    tau = shape.tau if shape.tau > 0.0 else 1.0
    starvation = shape.g_max * (1.0 - math.exp(-cycles / tau))

    return _CRITICAL_PRIORITY_BASE * shortfall + starvation


def reachable_area_with_orders(
    taxi: Taxi, world: WorldState, from_soc: Optional[float] = None
) -> Optional[Tuple[MarkingArea, float]]:
    """Find the nearest marking area that has unfilled orders and is reachable.

    "Reachable" means the taxi can drive there and arrive with SoC still
    ``>= B_area`` (the worst-case threshold a taxi must keep to still be allowed
    to serve, see ``01-04`` / ``01-03``). Distances use the precise router
    (``Router.distance``); energy uses the uniform consumption spec.

    Args:
        taxi: The candidate taxi.
        world: The world (areas, router, thresholds, taxi spec).
        from_soc: SoC to evaluate reachability from; defaults to the taxi's
            current SoC. (Allows callers to test reachability from a projected
            SoC, e.g. ``B_area`` for the order-generator tendency.)

    Returns:
        ``(area, distance_km)`` for the nearest reachable area with unfilled
        orders, or ``None`` if none qualifies.

    Example:
        >>> from simulation.config.default import default_config
        >>> from simulation.world.routing import Router
        >>> from simulation.world.entities import Taxi, MarkingArea, Order
        >>> cfg = default_config()
        >>> w = WorldState(config=cfg, router=Router(speed_kmh=30.0))
        >>> area = MarkingArea(id="a0", center=(-6.2, 106.8))
        >>> area.current_orders = [Order("o0", "a0", (-6.2, 106.8),
        ...                              (-6.21, 106.81), 8)]
        >>> w.areas = {"a0": area}
        >>> t = Taxi(id="t0", position=(-6.2, 106.8), soc=0.9)
        >>> res = reachable_area_with_orders(t, w)
        >>> res[0].id
        'a0'
    """

    battery = world.config.battery
    spec = world.config.taxi_spec
    soc = taxi.soc if from_soc is None else from_soc

    best: Optional[Tuple[MarkingArea, float]] = None
    for area in world.areas.values():
        if not area.unfilled_orders():
            continue
        distance_km = world.router.distance(taxi.position, area.center)
        if spec.battery_capacity_kwh > 0.0:
            soc_drop = (distance_km * spec.consumption_kwh_per_km) / spec.battery_capacity_kwh
        else:
            soc_drop = 0.0
        arrival_soc = soc - soc_drop
        if arrival_soc < battery.B_area:
            continue
        if best is None or distance_km < best[1]:
            best = (area, distance_km)
    return best


def nearest_reachable_area(
    taxi: Taxi, world: WorldState, min_arrival_soc: Optional[float] = None
) -> Optional[Tuple[MarkingArea, float]]:
    """Return the nearest marking area reachable while keeping ``SoC >= threshold``.

    "Reachable" means the taxi can drive to the area centre and still arrive with
    ``SoC >= min_arrival_soc`` (defaults to ``B_area``). Unlike
    :func:`reachable_area_with_orders` there is **no** order requirement — this is
    the helper that drives idle repositioning and the stuck/charge-candidate
    decision (a taxi for which this returns ``None`` is "stuck" and waits for the
    next negotiation). Distances are exact (constant speed/consumption).

    Args:
        taxi: The idle taxi to evaluate.
        world: The world (areas, router, thresholds, taxi spec).
        min_arrival_soc: Minimum SoC required on arrival; defaults to ``B_area``.

    Returns:
        ``(area, distance_km)`` of the nearest qualifying area, or ``None``.

    Example:
        >>> from simulation.config.default import default_config
        >>> from simulation.world.routing import Router
        >>> from simulation.world.entities import Taxi, MarkingArea
        >>> cfg = default_config()
        >>> w = WorldState(config=cfg, router=Router(speed_kmh=30.0))
        >>> w.areas = {"a0": MarkingArea(id="a0", center=(-6.2, 106.8))}
        >>> t = Taxi(id="t0", position=(-6.2, 106.8), soc=0.9)
        >>> nearest_reachable_area(t, w)[0].id
        'a0'
    """

    battery = world.config.battery
    spec = world.config.taxi_spec
    threshold = battery.B_area if min_arrival_soc is None else float(min_arrival_soc)

    best: Optional[Tuple[MarkingArea, float]] = None
    for area in world.areas.values():
        distance_km = world.router.distance(taxi.position, area.center)
        if spec.battery_capacity_kwh > 0.0:
            soc_drop = (distance_km * spec.consumption_kwh_per_km) / spec.battery_capacity_kwh
        else:
            soc_drop = 0.0
        if taxi.soc - soc_drop < threshold:
            continue
        if best is None or distance_km < best[1]:
            best = (area, distance_km)
    return best


def can_reach_any_port(taxi: Taxi, world: WorldState) -> bool:
    """Return ``True`` if the taxi can reach at least one regular port.

    "Reach" means arrival SoC stays non-negative after driving the exact route to
    the port's station. Emergency ports are excluded (``world.all_ports`` already
    omits them). This is the authoritative "can this taxi be placed in the action
    library at all?" test; a stuck idle taxi for which this is ``False`` is handled
    as an emergency instead of becoming a candidate (see
    ``simulation.bridge.emergency``).

    Example:
        >>> from simulation.config.default import default_config
        >>> from simulation.world.routing import Router
        >>> from simulation.world.entities import Taxi
        >>> w = WorldState(config=default_config(), router=Router(speed_kmh=30.0))
        >>> t = Taxi(id="t0", position=(-6.2, 106.8), soc=0.9)
        >>> can_reach_any_port(t, w)  # no stations populated on a bare world
        False
    """

    spec = world.config.taxi_spec
    if spec.battery_capacity_kwh <= 0.0:
        return True
    for port in world.all_ports():
        station = world.station_of_port(port.id)
        if station is None:
            continue
        distance_km = world.router.distance(taxi.position, station.location)
        soc_drop = (distance_km * spec.consumption_kwh_per_km) / spec.battery_capacity_kwh
        if taxi.soc - soc_drop >= 0.0:
            return True
    return False
