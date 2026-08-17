"""Reusable world assembly helper for one simulation run (Tahap I.1).

:func:`build_world_from_config` is the single function that converts a
:class:`~simulation.config.config.SimulationConfig` into a fully populated
:class:`~simulation.world.world_state.WorldState` — taxis, stations, ports,
marking areas, initial app-user bookings, and a precomputed routing matrix.
It is called by ``main.py``, the dashboard's ``SimSession``, and any smoke
test that needs a clean world without duplicating assembly logic.

Responsibilities:

1. Build a :class:`~simulation.world.routing.Router` and register every
   fixed point (station locations, area centres, emergency station) so that
   ``precompute_matrix`` covers all pairs at start-up.
2. Instantiate :class:`~simulation.world.entities.ChargingStation` and
   :class:`~simulation.world.entities.Port` objects for every
   :class:`~simulation.config.config.ChargingStationSpec`.
3. Instantiate :class:`~simulation.world.entities.Taxi` objects from every
   :class:`~simulation.config.config.InitialTaxiState`.
4. Instantiate :class:`~simulation.world.entities.MarkingArea` objects from
   every :class:`~simulation.config.config.MarkingAreaSpec`.
5. Spawn ``initial_booking_count`` random
   :class:`~simulation.world.entities.AppUserBooking` objects across the whole
   simulation's grid timeline and lock them onto ports. Every booking gets
   exactly one fallback plan (a single concrete alternative ``(port, slot)`` of
   the same duration) with an energy-based random compensation.
6. Add the emergency station when configured (``dead_battery_mode ==
   "HARD_FLOOR"``), register its fixed point, and recompute the matrix.
7. Return a :class:`~simulation.world.world_state.WorldState` with all
   dictionaries populated and ``_port_index`` pre-built.

All randomness is driven by the supplied ``rng`` so callers can make runs
reproducible. No logic beyond assembly lives here (``main.py`` wires agents
and training; the scheduler drives the loop).

Example
-------
>>> from simulation.config.default import default_config
>>> from simulation.world.world_assembly import build_world_from_config
>>> world = build_world_from_config(default_config())
>>> len(world.taxis)
3
>>> len(world.stations)
2
>>> len(world.areas)
3
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

from simulation.config.config import SimulationConfig
from simulation.world.entities import (
    AppUserBooking,
    ChargingStation,
    FallbackPlan,
    MarkingArea,
    Port,
    Taxi,
    TaxiState,
)
from simulation.world.routing import Router
from simulation.world.world_state import WorldState

# Half-hour grid width in minutes (structural constant, see ``01-08``).
_GRID_MINUTES: int = 30
# Number of grids in one simulated day (24 h * 2 grids/h).
_GRIDS_PER_DAY: int = 48
# Minimum booking duration in grids (one half-hour slot).
_MIN_BOOKING_DURATION: int = 1
# Maximum booking duration in grids (four half-hour slots = 2 h).
_MAX_BOOKING_DURATION: int = 4
# Hours per half-hour grid (for energy-based compensation).
_GRID_HOURS: float = _GRID_MINUTES / 60.0


def _build_router(config: SimulationConfig) -> Router:
    """Instantiate a :class:`Router` and register all fixed points.

    Registers station locations, area centres, and the emergency station
    (when present), then precomputes the full pairwise distance matrix.

    Args:
        config: The immutable simulation configuration.

    Returns:
        A :class:`Router` whose fixed-point matrix is fully precomputed.

    Example:
        >>> from simulation.config.default import default_config
        >>> r = _build_router(default_config())
        >>> "spklu_0" in r._fixed_points
        True
    """

    router = Router(speed_kmh=config.taxi_spec.speed_kmh)

    # Register charging stations.
    for station_spec in config.port_station.stations:
        router.register_fixed(station_spec.station_id, station_spec.location)

    # Register marking area centres.
    for area_spec in config.marking_area_order.areas:
        router.register_fixed(area_spec.area_id, area_spec.center)

    # Register the emergency station fixed point (used by precompute_matrix).
    emerg_loc = config.emergency.emergency_station_location
    if emerg_loc != (0.0, 0.0):
        router.register_fixed("__emergency__", emerg_loc)

    router.precompute_matrix()
    return router


def _build_stations(config: SimulationConfig) -> Dict[str, ChargingStation]:
    """Build regular (non-emergency) :class:`ChargingStation` / :class:`Port` objects.

    Args:
        config: The immutable simulation configuration.

    Returns:
        A dict mapping ``station_id -> ChargingStation`` (ports already attached).

    Example:
        >>> from simulation.config.default import default_config
        >>> stations = _build_stations(default_config())
        >>> len(stations["spklu_0"].ports)
        2
    """

    stations: Dict[str, ChargingStation] = {}
    for station_spec in config.port_station.stations:
        ports = [
            Port(
                id=port_spec.port_id,
                station_id=station_spec.station_id,
                power_kw=port_spec.power_kw,
                base_price_per_kwh=port_spec.base_price_per_kwh,
            )
            for port_spec in station_spec.ports
        ]
        station = ChargingStation(
            id=station_spec.station_id,
            location=station_spec.location,
            ports=ports,
            is_emergency=False,
        )
        stations[station_spec.station_id] = station
    return stations


def _build_emergency_station(config: SimulationConfig) -> Optional[ChargingStation]:
    """Build the emergency rescue station (sikap B), or ``None`` for sikap A.

    The emergency station has an effectively unlimited number of ports (we
    model this as one synthetic port with an unlimited capacity, since the
    state machine teleports there and does not check ``is_free``). It is
    never in the action library (Gotcha G8).

    Args:
        config: The immutable simulation configuration.

    Returns:
        A :class:`ChargingStation` with ``is_emergency=True``, or ``None``
        when ``dead_battery_mode`` is ``"LET_DIE"``.

    Example:
        >>> from simulation.config.default import default_config
        >>> cfg = default_config()
        >>> _build_emergency_station(cfg) is None
        True
    """

    if config.dead_battery.dead_battery_mode != "HARD_FLOOR":
        return None
    emerg_port = Port(
        id="__emergency_port__",
        station_id="__emergency__",
        power_kw=150.0,  # high power for emergency charging
        base_price_per_kwh=config.emergency.emergency_price_per_kwh,
    )
    return ChargingStation(
        id="__emergency__",
        location=config.emergency.emergency_station_location,
        ports=[emerg_port],
        is_emergency=True,
    )


def _build_taxis(config: SimulationConfig) -> Dict[str, Taxi]:
    """Instantiate :class:`Taxi` objects from ``taxi_spec.initial_taxi_state``.

    Each taxi starts at its configured position and SoC in ``IDLE_WAITING``
    state (the scheduler's first tick will route idle taxis and spawn opening
    orders).

    Args:
        config: The immutable simulation configuration.

    Returns:
        A dict mapping ``taxi_id -> Taxi``.

    Example:
        >>> from simulation.config.default import default_config
        >>> taxis = _build_taxis(default_config())
        >>> taxis["taxi_0"].soc
        0.8
    """

    taxis: Dict[str, Taxi] = {}
    for initial_state in config.taxi_spec.initial_taxi_state:
        taxi = Taxi(
            id=initial_state.taxi_id,
            position=initial_state.position,
            soc=initial_state.soc,
            state=TaxiState.IDLE_WAITING,
        )
        taxis[initial_state.taxi_id] = taxi
    return taxis


def _build_areas(config: SimulationConfig) -> Dict[str, MarkingArea]:
    """Instantiate :class:`MarkingArea` objects from ``marking_area_order.areas``.

    Areas start with empty ``current_orders``; the first ``refresh_hour``
    call (in the scheduler's tick-0 init) populates them.

    Args:
        config: The immutable simulation configuration.

    Returns:
        A dict mapping ``area_id -> MarkingArea``.

    Example:
        >>> from simulation.config.default import default_config
        >>> areas = _build_areas(default_config())
        >>> len(areas)
        3
    """

    areas: Dict[str, MarkingArea] = {}
    for area_spec in config.marking_area_order.areas:
        areas[area_spec.area_id] = MarkingArea(
            id=area_spec.area_id,
            center=area_spec.center,
            current_orders=[],
        )
    return areas


def _all_port_objects(stations: Dict[str, ChargingStation]) -> List[Port]:
    """Return a flat list of all port objects from the regular stations dict.

    Args:
        stations: Dict of regular charging stations (non-emergency).

    Returns:
        Flat list of :class:`Port` objects.

    Example:
        >>> from simulation.world.entities import ChargingStation, Port
        >>> s = {"s0": ChargingStation("s0", (0.0, 0.0),
        ...      [Port("p0", "s0", 50.0, 2500.0)])}
        >>> len(_all_port_objects(s))
        1
    """

    ports: List[Port] = []
    for station in stations.values():
        ports.extend(station.ports)
    return ports


def _span_free_of_bookings(port: Port, slot_start: int, duration_grids: int) -> bool:
    """Return ``True`` if the span carries no existing ``"booking"`` allocation.

    Used when picking a booking's single fallback destination at assembly time
    (the only allocations present at start-up are bookings).
    """
    req_end = slot_start + duration_grids
    for start, duration, _holder, kind in port.allocations:
        if kind != "booking":
            continue
        end = start + duration
        if start < req_end and slot_start < end:
            return False
    return True


def _choose_fallback_destination(
    all_ports: List[Port],
    origin_port: Port,
    slot_start: int,
    duration_grids: int,
    total_grids: int,
    shift_range: Tuple[int, int],
    rng: random.Random,
) -> Optional[Tuple[str, int]]:
    """Pick one empty ``(to_port_id, to_slot)`` fallback of the same duration.

    The destination **always starts at or after the booking's own span ends**, so
    it can never lie in the past at the moment the shift is actually performed.

    Why that is the right rule. The fallback is chosen once at world-assembly
    time, when no grid has elapsed yet, but it is *used* much later, during
    negotiation. A taxi only ever asks for a slot at/after the current grid
    (``_start_grid_for_arrival``), and it only collides with this booking when
    that slot lies inside ``[slot_start, slot_start + duration_grids)``. Placing
    the destination at ``slot_start + duration_grids`` or later therefore puts it
    strictly ahead of the requested slot, hence ahead of the clock, for every
    reachable state. A destination chosen *before* the booking would instead be
    unusable by the time it is needed, leaving the booking effectively
    unshiftable and silently removing it from the action library.

    ``shift_range`` is read as how far **beyond the end of the booking** the
    destination may begin: offsets ``shift_min - 1 .. shift_max - 1`` grids. With
    the usual ``(1, 4)`` that is the four grids right after the booking ends,
    which reproduces the old forward behaviour for one-grid bookings and extends
    it consistently to longer ones.

    Tries the origin port first, then any other port, nearest offset first.
    Returns ``None`` if nothing fits.
    """
    shift_min, shift_max = shift_range
    shift_min = max(0, int(shift_min))
    shift_max = max(shift_min, int(shift_max))
    orig_end = slot_start + duration_grids

    offsets = list(range(max(0, shift_min - 1), max(1, shift_max)))

    # (1) Same port, starting after the booking's own span (random offset order).
    acak = list(offsets)
    rng.shuffle(acak)
    for offset in acak:
        to_slot = orig_end + offset
        if to_slot + duration_grids > total_grids:
            continue
        if _span_free_of_bookings(origin_port, to_slot, duration_grids):
            return (origin_port.id, to_slot)

    # (2) Any other port, nearest offset first.
    other_ports = [p for p in all_ports if p.id != origin_port.id]
    rng.shuffle(other_ports)
    for offset in offsets:
        to_slot = orig_end + offset
        if to_slot + duration_grids > total_grids:
            continue
        for alt in other_ports:
            if _span_free_of_bookings(alt, to_slot, duration_grids):
                return (alt.id, to_slot)
    return None


def _build_initial_bookings(
    config: SimulationConfig,
    stations: Dict[str, ChargingStation],
    rng: random.Random,
) -> Dict[str, AppUserBooking]:
    """Spawn ``initial_booking_count`` app-user bookings across the whole timeline.

    Each booking is placed on a randomly selected regular port, at a random slot
    anywhere in the simulation's grid horizon (``n_days * 48`` grids), with random
    duration in ``[_MIN_BOOKING_DURATION, _MAX_BOOKING_DURATION]``. Every booking
    gets **exactly one** fallback plan: a single empty alternative ``(port, slot)``
    of the same duration, with an **energy-based** compensation
    ``rate_per_kwh * power_kw * duration_grids * 0.5`` where ``rate_per_kwh`` is
    drawn uniformly from ``compensation_per_kwh_range``.

    A booking is skipped silently when its random original span is occupied or no
    valid fallback destination exists (the count is a *target*, not a guarantee).

    Args:
        config: The immutable simulation configuration.
        stations: Already-built regular charging stations.
        rng: Random source for reproducibility.

    Returns:
        A dict mapping ``booking_id -> AppUserBooking``.

    Example:
        >>> import random
        >>> from simulation.config.default import default_config
        >>> from simulation.world.world_assembly import (
        ...     _build_stations, _build_initial_bookings)
        >>> cfg = default_config()
        >>> stations = _build_stations(cfg)
        >>> bookings = _build_initial_bookings(cfg, stations, random.Random(0))
        >>> len(bookings) <= cfg.app_user_booking.initial_booking_count
        True
        >>> all(b.fallback_plan is not None for b in bookings.values())
        True
    """

    all_ports = _all_port_objects(stations)
    if not all_ports:
        return {}

    booking_cfg = config.app_user_booking
    fallback_cfg = booking_cfg.fallback
    total_grids = max(1, int(config.time.n_days) * _GRIDS_PER_DAY)
    comp_min, comp_max = fallback_cfg.compensation_per_kwh_range
    bookings: Dict[str, AppUserBooking] = {}

    for i in range(booking_cfg.initial_booking_count):
        port = rng.choice(all_ports)
        duration = rng.randint(_MIN_BOOKING_DURATION, _MAX_BOOKING_DURATION)
        max_start = max(0, total_grids - duration)
        slot_start = rng.randint(0, max_start)

        # Only place the booking if the port actually has a free span.
        if not port.is_free(slot_start, duration):
            continue

        destination = _choose_fallback_destination(
            all_ports, port, slot_start, duration, total_grids,
            fallback_cfg.allowed_slot_shift_range, rng,
        )
        if destination is None:
            # Every booking must have exactly one fallback; skip if none fits.
            continue
        to_port_id, to_slot = destination

        # Energy-based compensation: rate_per_kwh * power_kw * duration_grids * 0.5h.
        rate_per_kwh = rng.uniform(float(comp_min), float(comp_max))
        compensation = rate_per_kwh * port.power_kw * duration * _GRID_HOURS

        fallback_plan = FallbackPlan(
            to_port_id=to_port_id,
            to_slot_start=to_slot,
            compensation=compensation,
        )

        booking_id = f"booking_{i}"
        booking = AppUserBooking(
            id=booking_id,
            port_id=port.id,
            slot_start=slot_start,
            duration_grids=duration,
            is_shiftable=True,
            fallback_plan=fallback_plan,
            compensation=0.0,
        )
        # Lock the port allocation so it appears occupied from start.
        port.add_allocation(slot_start, duration, booking_id, "booking")
        bookings[booking_id] = booking

    return bookings


def build_world_from_config(
    config: SimulationConfig,
    rng: Optional[random.Random] = None,
) -> WorldState:
    """Build a fully populated :class:`WorldState` from a :class:`SimulationConfig`.

    This is the single assembly entry point used by ``main.py``, the dashboard's
    ``SimSession``, and smoke tests. It is pure construction — it does not start
    the scheduler, run any negotiation, or mutate config.

    Assembly order:
        1. Build the :class:`Router` and precompute the fixed-point matrix.
        2. Build regular charging stations + ports.
        3. Build the emergency station (only for ``HARD_FLOOR`` mode).
        4. Build taxis from initial states.
        5. Build marking areas.
        6. Spawn initial app-user bookings on random ports.
        7. Assemble the :class:`WorldState` and trigger a port-index build.

    Args:
        config: The immutable :class:`SimulationConfig` for this run.
        rng: Optional :class:`random.Random` for reproducible booking
            placement. A fresh instance is created if omitted.

    Returns:
        A fully populated :class:`WorldState` ready for the scheduler.

    Example:
        >>> from simulation.config.default import default_config
        >>> world = build_world_from_config(default_config())
        >>> len(world.taxis)
        3
        >>> len(world.stations)
        2
        >>> len(world.areas)
        3
        >>> world.sim_time_min
        0.0
    """

    if rng is None:
        rng = random.Random()

    router = _build_router(config)
    stations = _build_stations(config)
    emergency_station = _build_emergency_station(config)
    taxis = _build_taxis(config)
    areas = _build_areas(config)
    bookings = _build_initial_bookings(config, stations, rng)

    world = WorldState(
        config=config,
        router=router,
        taxis=taxis,
        stations=stations,
        areas=areas,
        bookings=bookings,
        emergency_station=emergency_station,
        sim_time_min=0.0,
    )

    # Pre-build the port index so the first call to get_port / station_of_port
    # is O(1) rather than triggering a lazy rebuild mid-tick.
    world._rebuild_port_index()

    return world
