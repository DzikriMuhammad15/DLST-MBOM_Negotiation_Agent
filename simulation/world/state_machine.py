"""Per-tick taxi state machine (see ``01-06``).

:func:`step_taxi` advances one taxi by ``dt_min`` simulated minutes. The full
taxi lifecycle (see ``01-06``):

* a passenger trip ``TO_PASSENGER -> WITH_PASSENGER`` then, on drop-off,
  :func:`_go_idle_and_reroute` decides the next state;
* **reposition**: if the nearest marking area is reachable with arrival
  ``SoC >= B_area`` the taxi becomes ``TO_MARKING_AREA`` and drives there,
  becoming ``IDLE_IN_AREA`` on arrival (available for new orders);
* **stuck**: otherwise it becomes ``IDLE_WAITING`` in place — the set of
  negotiation charge candidates;
* a won slot drives ``RESERVED -> TO_PORT -> CHARGING -> release``; ``RESERVED``
  waits **in place** (never repositions, never takes an order) until departure;
* on charge completion the taxi is full, so :func:`_go_idle_and_reroute` sends it
  straight to ``TO_MARKING_AREA``;
* ``EMERGENCY_CHARGING`` / ``STRANDED`` handle the dead-battery cases.

Idle (``IDLE_IN_AREA`` / ``IDLE_WAITING``) and ``RESERVED`` taxis sit still with
SoC frozen; the scheduler owns order matching and the negotiation trigger, this
module owns physical evolution and the transitions that follow from
arrival/charging completion.

Contract C5 / Gotcha G4: this module reads ``world.config``, ``world.router``,
the simulated time ``world.sim_time_min``, and calls ``world.get_port``,
``world.station_of_port``, and ``world.metrics_hook(...)``. The ``world`` object
handed in must provide all of these (``WorldState`` does).

SoC invariants (Gotcha G9): SoC drops only on the movement leg of a tick;
``IDLE_*`` / ``RESERVED`` freeze SoC; ``TO_MARKING_AREA`` / ``TO_PASSENGER`` /
``TO_PORT`` drain it; ``CHARGING`` / ``EMERGENCY_CHARGING`` raise it. Speed is
uniform.

This module never imports ``negotiation`` (layer separation).

Example
-------
>>> from simulation.config.default import default_config
>>> from simulation.world.routing import Router
>>> from simulation.world.world_state import WorldState
>>> from simulation.world.entities import Taxi, TaxiState
>>> cfg = default_config()
>>> world = WorldState(config=cfg, router=Router(speed_kmh=cfg.taxi_spec.speed_kmh))
>>> taxi = Taxi(id="t0", position=(-6.2, 106.8), soc=0.5)
>>> world.taxis = {"t0": taxi}
>>> step_taxi(taxi, world, dt_min=5.0)   # idle: no movement, SoC frozen
>>> taxi.soc
0.5
"""

from __future__ import annotations

from typing import Tuple

from simulation.world.entities import OrderState, Reservation, Taxi, TaxiState
from simulation.world.fare import trip_fare
from simulation.world.idle_logic import nearest_reachable_area
from simulation.world.world_state import WorldState

Coordinate = Tuple[float, float]

# Half-hour grid width in minutes (structural constant, see ``01-08``).
_GRID_MINUTES: int = 30
# Minutes per hour, for charge/travel conversions.
_MINUTES_PER_HOUR: float = 60.0
# Distance (km) below which a taxi is considered to have arrived at its target.
_ARRIVAL_EPSILON_KM: float = 1e-6
# Late-arrival mode (``late_arrival_mode``, ``01-08``) that lapses a late slot.
# The complementary ``STRICT_GUARD`` keeps the order gate conservative so a late
# arrival never occurs; only this mode reaches the forfeit path.
_LATE_SLOT_FORFEIT: str = "SLOT_FORFEIT"


def step_taxi(taxi: Taxi, world: WorldState, dt_min: float) -> None:
    """Advance one taxi by ``dt_min`` simulated minutes (see ``01-06``).

    Dispatches on the taxi's current state. Movement-bearing states move along
    their route and drain SoC; charging states raise SoC; idle/reserved states
    freeze SoC (RESERVED additionally checks whether it is time to depart).

    Args:
        taxi: The taxi to advance (mutated in place).
        world: The world (config, router, sim time, port/station helpers,
            metrics hook). Must satisfy Contract C5.
        dt_min: Simulated minutes to advance this tick (>= 0).

    Example:
        >>> from simulation.config.default import default_config
        >>> from simulation.world.routing import Router
        >>> from simulation.world.entities import Taxi, TaxiState
        >>> cfg = default_config()
        >>> w = WorldState(config=cfg, router=Router(speed_kmh=cfg.taxi_spec.speed_kmh))
        >>> t = Taxi(id="t0", position=(-6.2, 106.8), soc=0.5,
        ...          state=TaxiState.IDLE_WAITING)
        >>> step_taxi(t, w, 5.0)
        >>> t.soc
        0.5
    """

    state = taxi.state
    if state in (TaxiState.IDLE_IN_AREA, TaxiState.IDLE_WAITING, TaxiState.STRANDED):
        # All three stay put with SoC frozen: IDLE_IN_AREA waits in its area for an
        # order; IDLE_WAITING is stuck (a charge candidate) waiting for negotiation;
        # STRANDED is out of operation. Repositioning toward an area is a separate
        # state (TO_MARKING_AREA) — IDLE_WAITING never moves.
        return
    if state is TaxiState.TO_MARKING_AREA:
        _step_to_marking_area(taxi, world, dt_min)
        return
    if state is TaxiState.RESERVED:
        _step_reserved(taxi, world, dt_min)
        return
    if state is TaxiState.TO_PASSENGER:
        _step_to_passenger(taxi, world, dt_min)
        return
    if state is TaxiState.WITH_PASSENGER:
        _step_with_passenger(taxi, world, dt_min)
        return
    if state is TaxiState.TO_PORT:
        _step_to_port(taxi, world, dt_min)
        return
    if state is TaxiState.CHARGING:
        _step_charging(taxi, world, dt_min)
        return
    if state is TaxiState.EMERGENCY_CHARGING:
        _step_emergency_charging(taxi, world, dt_min)
        return


# ---------------------------------------------------------------------------
# Movement primitive (drains SoC ONLY when moving — Gotcha G9)
# ---------------------------------------------------------------------------
def _move_towards(
    taxi: Taxi, world: WorldState, target: Coordinate, dt_min: float, productive: bool
) -> bool:
    """Move the taxi toward ``target`` for one tick, draining SoC by distance.

    The taxi advances at the uniform speed for at most ``dt_min`` minutes (or
    less if it reaches the target first). SoC drops in proportion to the km
    actually traveled this tick (Gotcha G9); the traveled km is added to the
    productive or unproductive odometer.

    Args:
        taxi: The taxi to move (mutated in place).
        world: The world (router, taxi spec).
        target: The ``(lat, lng)`` destination of this leg.
        dt_min: Simulated minutes available this tick.
        productive: ``True`` for the WITH_PASSENGER leg (billable km); ``False``
            for any passenger-less leg (unproductive metric km).

    Returns:
        ``True`` if the taxi reached ``target`` this tick, else ``False``.

    Example:
        >>> from simulation.config.default import default_config
        >>> from simulation.world.routing import Router
        >>> from simulation.world.entities import Taxi
        >>> cfg = default_config()
        >>> w = WorldState(config=cfg, router=Router(speed_kmh=cfg.taxi_spec.speed_kmh))
        >>> t = Taxi(id="t0", position=(-6.2, 106.8), soc=1.0)
        >>> _move_towards(t, w, (-6.2, 106.8), 5.0, productive=False)
        True
    """

    spec = world.config.taxi_spec
    remaining_km = world.router.distance(taxi.position, target)
    if remaining_km <= _ARRIVAL_EPSILON_KM:
        taxi.position = target
        return True

    max_km_this_tick = spec.speed_kmh * (dt_min / _MINUTES_PER_HOUR)
    if max_km_this_tick >= remaining_km:
        traveled_km = remaining_km
        taxi.position = target
        reached = True
    else:
        fraction = max_km_this_tick / remaining_km
        lat = taxi.position[0] + (target[0] - taxi.position[0]) * fraction
        lng = taxi.position[1] + (target[1] - taxi.position[1]) * fraction
        taxi.position = (lat, lng)
        traveled_km = max_km_this_tick
        reached = False

    _drain_soc(taxi, world, traveled_km)
    if productive:
        taxi.productive_km += traveled_km
    else:
        taxi.unproductive_km += traveled_km
    # Forward the travelled km to the metrics collector (the taxi odometer above
    # is ground truth; this hook is what feeds fleet_business.unproductive_km).
    world.metrics_hook("taxi_moved", taxi_id=taxi.id, km=traveled_km, productive=productive)
    return reached


def _drain_soc(taxi: Taxi, world: WorldState, traveled_km: float) -> None:
    """Reduce SoC for ``traveled_km`` of driving (clamped at 0).

    Args:
        taxi: The taxi (mutated in place).
        world: The world (taxi spec).
        traveled_km: Distance driven this tick, in km (>= 0).

    Example:
        >>> from simulation.config.default import default_config
        >>> from simulation.world.routing import Router
        >>> from simulation.world.entities import Taxi
        >>> cfg = default_config()
        >>> w = WorldState(config=cfg, router=Router(speed_kmh=cfg.taxi_spec.speed_kmh))
        >>> t = Taxi(id="t0", position=(0.0, 0.0), soc=1.0)
        >>> _drain_soc(t, w, 0.0)
        >>> t.soc
        1.0
    """

    spec = world.config.taxi_spec
    if spec.battery_capacity_kwh <= 0.0:
        return
    delta = (traveled_km * spec.consumption_kwh_per_km) / spec.battery_capacity_kwh
    taxi.soc = max(0.0, taxi.soc - delta)


# ---------------------------------------------------------------------------
# Order-serving transitions
# ---------------------------------------------------------------------------
def _step_to_passenger(taxi: Taxi, world: WorldState, dt_min: float) -> None:
    """Drive toward the order pickup; on arrival begin the passenger trip.

    Args:
        taxi: The taxi (must have ``current_order``).
        world: The world.
        dt_min: Simulated minutes this tick.

    Example:
        >>> from simulation.config.default import default_config
        >>> from simulation.world.routing import Router
        >>> from simulation.world.entities import Taxi, Order, TaxiState
        >>> cfg = default_config()
        >>> w = WorldState(config=cfg, router=Router(speed_kmh=cfg.taxi_spec.speed_kmh))
        >>> order = Order("o0", "a0", (-6.2, 106.8), (-6.2, 106.8), 8)
        >>> t = Taxi(id="t0", position=(-6.2, 106.8), soc=1.0,
        ...          state=TaxiState.TO_PASSENGER, current_order=order)
        >>> _step_to_passenger(t, w, 5.0)
        >>> t.state is TaxiState.WITH_PASSENGER
        True
    """

    order = taxi.current_order
    if order is None:
        # No order to pursue; fall back to idle (scheduler will re-evaluate).
        taxi.state = TaxiState.IDLE_WAITING
        taxi.move_target = None
        return

    taxi.move_target = order.pickup_point
    reached = _move_towards(taxi, world, order.pickup_point, dt_min, productive=False)
    if reached:
        order.state = OrderState.IN_PROGRESS
        taxi.state = TaxiState.WITH_PASSENGER
        taxi.move_target = order.destination
        world.metrics_hook("pickup_reached", taxi_id=taxi.id, order_id=order.id)


def _step_with_passenger(taxi: Taxi, world: WorldState, dt_min: float) -> None:
    """Carry the passenger to the destination; on arrival complete the trip.

    On completion the order becomes COMPLETED, fare is recorded via the metrics
    hook, the taxi goes idle, and the idle logic re-routes it toward the nearest
    area with unfilled orders (see ``01-03`` / ``01-06``).

    Args:
        taxi: The taxi (must have ``current_order``).
        world: The world.
        dt_min: Simulated minutes this tick.

    Example:
        >>> from simulation.config.default import default_config
        >>> from simulation.world.routing import Router
        >>> from simulation.world.entities import Taxi, Order, TaxiState, OrderState
        >>> cfg = default_config()
        >>> w = WorldState(config=cfg, router=Router(speed_kmh=cfg.taxi_spec.speed_kmh))
        >>> order = Order("o0", "a0", (-6.2, 106.8), (-6.2, 106.8), 8,
        ...               state=OrderState.IN_PROGRESS)
        >>> t = Taxi(id="t0", position=(-6.2, 106.8), soc=1.0,
        ...          state=TaxiState.WITH_PASSENGER, current_order=order)
        >>> _step_with_passenger(t, w, 5.0)
        >>> order.state is OrderState.COMPLETED
        True
    """

    order = taxi.current_order
    if order is None:
        taxi.state = TaxiState.IDLE_WAITING
        taxi.move_target = None
        return

    taxi.move_target = order.destination
    reached = _move_towards(taxi, world, order.destination, dt_min, productive=True)
    if reached:
        productive_km = world.router.distance(order.pickup_point, order.destination)
        fare = trip_fare(world.config.fare, productive_km)
        order.state = OrderState.COMPLETED
        world.metrics_hook(
            "trip_completed",
            taxi_id=taxi.id,
            order_id=order.id,
            productive_km=productive_km,
            fare=fare,
        )
        taxi.current_order = None
        taxi.move_target = None
        _go_idle_and_reroute(taxi, world)


def _area_id_at(world: WorldState, position: Coordinate) -> "str | None":
    """Return the id of the marking area whose centre matches ``position`` (else None).

    Used when a ``TO_MARKING_AREA`` taxi arrives: its ``move_target`` was set to an
    area centre, so on arrival the centre equals the taxi position.
    """
    for area in world.areas.values():
        if (abs(area.center[0] - position[0]) <= _ARRIVAL_EPSILON_KM
                and abs(area.center[1] - position[1]) <= _ARRIVAL_EPSILON_KM):
            return area.id
    return None


def _step_to_marking_area(taxi: Taxi, world: WorldState, dt_min: float) -> None:
    """Drive a repositioning taxi toward its target area; on arrival go IDLE_IN_AREA.

    Movement drains SoC (unproductive km, Gotcha G9). On arrival the taxi becomes
    ``IDLE_IN_AREA`` (available for order matching in that area) with its
    ``current_area`` set. A taxi only ever enters this state when it satisfies the
    area battery constraint (arrival ``SoC >= B_area``), so the constraint is never
    violated.

    Example:
        >>> from simulation.config.default import default_config
        >>> from simulation.world.routing import Router
        >>> from simulation.world.entities import Taxi, MarkingArea, TaxiState
        >>> cfg = default_config()
        >>> w = WorldState(config=cfg, router=Router(speed_kmh=cfg.taxi_spec.speed_kmh))
        >>> w.areas = {"a0": MarkingArea(id="a0", center=(-6.2, 106.8))}
        >>> t = Taxi(id="t0", position=(-6.2, 106.8), soc=1.0,
        ...          state=TaxiState.TO_MARKING_AREA, move_target=(-6.2, 106.8))
        >>> _step_to_marking_area(t, w, 5.0)
        >>> t.state is TaxiState.IDLE_IN_AREA
        True
    """

    if taxi.move_target is None:
        taxi.state = TaxiState.IDLE_IN_AREA
        return
    reached = _move_towards(taxi, world, taxi.move_target, dt_min, productive=False)
    if reached:
        taxi.current_area = _area_id_at(world, taxi.position)
        taxi.state = TaxiState.IDLE_IN_AREA
        taxi.move_target = None


def _go_idle_and_reroute(taxi: Taxi, world: WorldState) -> None:
    """Decide a finished/charged taxi's next state: reposition to an area, or wait.

    If the taxi can reach the nearest marking area while still arriving with
    ``SoC >= B_area`` (:func:`~simulation.world.idle_logic.nearest_reachable_area`)
    it becomes ``TO_MARKING_AREA`` and heads there to keep serving orders — so it
    spends no time as a candidate. Otherwise it is **stuck**: it becomes
    ``IDLE_WAITING`` in place (``move_target = None``) and will be a charge
    candidate (or, if it can reach no port, an emergency) at the next negotiation.
    SoC is not changed here (movement happens on subsequent ticks).

    Args:
        taxi: The taxi (mutated in place).
        world: The world.

    Example:
        >>> from simulation.config.default import default_config
        >>> from simulation.world.routing import Router
        >>> from simulation.world.entities import Taxi
        >>> cfg = default_config()
        >>> w = WorldState(config=cfg, router=Router(speed_kmh=cfg.taxi_spec.speed_kmh))
        >>> t = Taxi(id="t0", position=(-6.2, 106.8), soc=1.0)
        >>> _go_idle_and_reroute(t, w)   # no areas -> stuck, waits in place
        >>> t.state is TaxiState.IDLE_WAITING
        True
    """

    taxi.current_area = None
    reachable = nearest_reachable_area(taxi, world)
    if reachable is not None:
        area, _distance = reachable
        taxi.state = TaxiState.TO_MARKING_AREA
        taxi.move_target = area.center
    else:
        # Stuck: cannot reach any area above B_area -> wait in place for the next
        # negotiation (candidate) or emergency handling.
        taxi.state = TaxiState.IDLE_WAITING
        taxi.move_target = None


# ---------------------------------------------------------------------------
# Reservation / charging transitions
# ---------------------------------------------------------------------------
def _step_reserved(taxi: Taxi, world: WorldState, dt_min: float) -> None:
    """Hold a RESERVED taxi (SoC frozen); depart for the port when due.

    When the simulated time reaches the reservation's ``departure_time_min`` the
    taxi switches to ``TO_PORT``. (The "may take an order while RESERVED" gate of
    ``01-08`` is driven by the scheduler, which can set ``current_order`` and
    flip the taxi to ``TO_PASSENGER``; this function handles the depart trigger.)

    Args:
        taxi: The reserved taxi (must have ``reservation``).
        world: The world.
        dt_min: Simulated minutes this tick (unused; SoC frozen while reserved).

    Example:
        >>> from simulation.config.default import default_config
        >>> from simulation.world.routing import Router
        >>> from simulation.world.entities import Taxi, Reservation, TaxiState
        >>> cfg = default_config()
        >>> w = WorldState(config=cfg, router=Router(speed_kmh=cfg.taxi_spec.speed_kmh))
        >>> w.sim_time_min = 100.0
        >>> res = Reservation("p", "s", 4, 1, 3000.0, departure_time_min=90.0)
        >>> t = Taxi(id="t0", position=(-6.2, 106.8), soc=0.4,
        ...          state=TaxiState.RESERVED, reservation=res)
        >>> _step_reserved(t, w, 5.0)
        >>> t.state is TaxiState.TO_PORT
        True
    """

    reservation = taxi.reservation
    if reservation is None:
        world.metrics_hook(
            "reserved_taxi_missing_reservation",
            taxi_id=taxi.id,
        )
        taxi.state = TaxiState.IDLE_WAITING
        return
    if world.sim_time_min >= reservation.departure_time_min:
        taxi.state = TaxiState.TO_PORT
        station = world.station_of_port(reservation.port_id)
        if station is not None:
            taxi.move_target = station.location
        world.metrics_hook("departure_for_port", taxi_id=taxi.id, port_id=reservation.port_id)


def _is_late_for_slot(reservation: Reservation, world: WorldState) -> bool:
    """Return whether the taxi has arrived after its slot's first grid started.

    Per ``01-08`` ("taksi wajib tiba sebelum grid pertamanya mulai") the taxi is
    late iff the current simulated time is strictly past the booked slot's first
    grid start minute (``slot_start * 30``). Used only at port arrival to decide
    whether the reserved slot is honored or forfeited.

    Args:
        reservation: The taxi's live reservation (port + slot).
        world: The world (simulated time).

    Returns:
        ``True`` if ``world.sim_time_min`` is past ``slot_start_min``.

    Example:
        >>> from simulation.config.default import default_config
        >>> from simulation.world.routing import Router
        >>> from simulation.world.entities import Reservation
        >>> w = WorldState(config=default_config(), router=Router(speed_kmh=30.0))
        >>> res = Reservation("p", "s", 4, 1, 3000.0, departure_time_min=0.0)
        >>> w.sim_time_min = 4 * 30 + 1.0
        >>> _is_late_for_slot(res, w)
        True
        >>> w.sim_time_min = 4 * 30 - 1.0
        >>> _is_late_for_slot(res, w)
        False
    """

    slot_start_min = reservation.slot_start * _GRID_MINUTES
    return world.sim_time_min > float(slot_start_min)


def _forfeit_slot(taxi: Taxi, world: WorldState, reservation: Reservation) -> None:
    """Lapse a late-arrived reserved slot: no allocation, back to idle (``01-08``).

    Under ``SLOT_FORFEIT`` a taxi that reaches its port after the slot's first
    grid has started forfeits the slot: the ``"fleet"`` allocation is **not**
    added (the port grids are freed/skipped), the reservation is cleared, the
    taxi re-enters idle/candidate logic, and the forfeit is emitted as the
    collector's ``slot_forfeit`` metric so ``slot_forfeit_count`` increments.

    Args:
        taxi: The taxi that arrived late (mutated in place).
        world: The world (metrics hook, idle re-routing).
        reservation: The forfeited reservation (read for the metric payload).

    Example:
        >>> from simulation.config.default import default_config
        >>> from simulation.world.routing import Router
        >>> from simulation.world.entities import Taxi, Reservation, TaxiState
        >>> w = WorldState(config=default_config(), router=Router(speed_kmh=30.0))
        >>> res = Reservation("p", "s", 4, 1, 3000.0, departure_time_min=0.0)
        >>> t = Taxi(id="t0", position=(-6.2, 106.8), soc=0.4,
        ...          state=TaxiState.TO_PORT, reservation=res)
        >>> _forfeit_slot(t, w, res)
        >>> t.reservation is None
        True
    """

    world.metrics_hook(
        "slot_forfeit",
        taxi_id=taxi.id,
        port_id=reservation.port_id,
        slot_start=reservation.slot_start,
        duration_grids=reservation.duration_grids,
    )
    taxi.reservation = None
    taxi.charge_until_min = None
    taxi.move_target = None
    _go_idle_and_reroute(taxi, world)


def _step_to_port(taxi: Taxi, world: WorldState, dt_min: float) -> None:
    """Drive toward the reserved port; on arrival begin CHARGING (or forfeit late).

    On arrival the taxi's reserved span is materialized as a ``"fleet"``
    allocation on the port (a hard constraint for future negotiations, see
    ``01-08``), and ``charge_until_min`` is set to the slot's end minute.

    Late-arrival handling (``01-08`` ``late_arrival_mode``): if the taxi reaches
    the port after its slot's first grid has started (:func:`_is_late_for_slot`)
    and the mode is ``SLOT_FORFEIT``, the slot lapses via :func:`_forfeit_slot`
    (no allocation, reservation cleared, back to idle, ``slot_forfeit`` emitted).
    Under ``STRICT_GUARD`` the order-acceptance gate makes a late arrival
    unreachable by design, so the taxi always charges.

    Args:
        taxi: The taxi (must have ``reservation``).
        world: The world.
        dt_min: Simulated minutes this tick.

    Example:
        >>> from simulation.config.default import default_config
        >>> from simulation.world.routing import Router
        >>> from simulation.world.entities import (Taxi, Reservation, TaxiState,
        ...                                         ChargingStation, Port)
        >>> cfg = default_config()
        >>> w = WorldState(config=cfg, router=Router(speed_kmh=cfg.taxi_spec.speed_kmh))
        >>> port = Port("p", "s", 50.0, 2500.0)
        >>> w.stations = {"s": ChargingStation("s", (-6.2, 106.8), [port])}
        >>> res = Reservation("p", "s", 4, 1, 3000.0, departure_time_min=0.0)
        >>> t = Taxi(id="t0", position=(-6.2, 106.8), soc=0.4,
        ...          state=TaxiState.TO_PORT, reservation=res)
        >>> _step_to_port(t, w, 5.0)
        >>> t.state is TaxiState.CHARGING
        True
    """

    reservation = taxi.reservation
    if reservation is None:
        taxi.state = TaxiState.IDLE_WAITING
        taxi.move_target = None
        return

    station = world.station_of_port(reservation.port_id)
    if station is None:
        taxi.state = TaxiState.IDLE_WAITING
        taxi.move_target = None
        return

    taxi.move_target = station.location
    reached = _move_towards(taxi, world, station.location, dt_min, productive=False)
    if not reached:
        return

    late_mode = world.config.future_slot.late_arrival_mode
    if late_mode == _LATE_SLOT_FORFEIT and _is_late_for_slot(reservation, world):
        _forfeit_slot(taxi, world, reservation)
        return

    port = world.get_port(reservation.port_id)
    if port is not None and port.is_free(reservation.slot_start, reservation.duration_grids):
        port.add_allocation(
            reservation.slot_start, reservation.duration_grids, taxi.id, "fleet"
        )
    taxi.state = TaxiState.CHARGING
    taxi.move_target = None
    slot_end_min = (reservation.slot_start + reservation.duration_grids) * _GRID_MINUTES
    taxi.charge_until_min = float(slot_end_min)
    world.metrics_hook("charging_started", taxi_id=taxi.id, port_id=reservation.port_id)


def _step_charging(taxi: Taxi, world: WorldState, dt_min: float) -> None:
    """Raise SoC while charging; stop at full or at slot end, then release the port.

    SoC rises by ``power_kw * dt_h / capacity`` (Gotcha G9: charging is the only
    way SoC rises). Charging ends when SoC reaches ``SoC_target`` or when the
    simulated time passes the slot end; on end the taxi goes idle and re-routes.

    Args:
        taxi: The charging taxi (should have ``reservation``).
        world: The world.
        dt_min: Simulated minutes this tick.

    Example:
        >>> from simulation.config.default import default_config
        >>> from simulation.world.routing import Router
        >>> from simulation.world.entities import (Taxi, Reservation, TaxiState,
        ...                                         ChargingStation, Port)
        >>> cfg = default_config()
        >>> w = WorldState(config=cfg, router=Router(speed_kmh=cfg.taxi_spec.speed_kmh))
        >>> port = Port("p", "s", 50.0, 2500.0)
        >>> w.stations = {"s": ChargingStation("s", (-6.2, 106.8), [port])}
        >>> res = Reservation("p", "s", 0, 2, 3000.0, departure_time_min=0.0)
        >>> t = Taxi(id="t0", position=(-6.2, 106.8), soc=0.5,
        ...          state=TaxiState.CHARGING, reservation=res, charge_until_min=60.0)
        >>> _step_charging(t, w, 5.0)
        >>> t.soc > 0.5
        True
    """

    reservation = taxi.reservation
    power_kw = 0.0
    if reservation is not None:
        port = world.get_port(reservation.port_id)
        if port is not None:
            power_kw = port.power_kw

    _raise_soc(taxi, world, power_kw, dt_min)

    full = taxi.soc >= world.config.battery.SoC_target
    slot_over = taxi.charge_until_min is not None and world.sim_time_min >= taxi.charge_until_min
    if full or slot_over:
        _finish_charging(taxi, world, energy_holder=taxi.id, kind="fleet")


def _step_emergency_charging(taxi: Taxi, world: WorldState, dt_min: float) -> None:
    """Raise SoC at the emergency station until full, then go idle (sikap B).

    Emergency charging has no slot bound — it always runs to ``SoC_target``
    (see ``05-05``). The teleport and the premium-price metric are owned by
    ``handle_emergency_taxis`` in the bridge; this function only advances the
    physical charge.

    Args:
        taxi: The emergency-charging taxi.
        world: The world (uses the emergency station's port power).
        dt_min: Simulated minutes this tick.

    Example:
        >>> from simulation.config.default import default_config
        >>> from simulation.world.routing import Router
        >>> from simulation.world.entities import (Taxi, TaxiState,
        ...                                         ChargingStation, Port)
        >>> cfg = default_config()
        >>> w = WorldState(config=cfg, router=Router(speed_kmh=cfg.taxi_spec.speed_kmh))
        >>> port = Port("ep", "emg", 50.0, 0.0)
        >>> w.emergency_station = ChargingStation("emg", (-6.2, 106.8), [port], True)
        >>> t = Taxi(id="t0", position=(-6.2, 106.8), soc=0.05,
        ...          state=TaxiState.EMERGENCY_CHARGING)
        >>> _step_emergency_charging(t, w, 5.0)
        >>> t.soc > 0.05
        True
    """

    power_kw = 0.0
    if world.emergency_station is not None and world.emergency_station.ports:
        power_kw = world.emergency_station.ports[0].power_kw

    _raise_soc(taxi, world, power_kw, dt_min)

    if taxi.soc >= world.config.battery.SoC_target:
        taxi.soc = min(taxi.soc, world.config.battery.SoC_target)
        taxi.charge_until_min = None
        world.metrics_hook("emergency_charging_finished", taxi_id=taxi.id)
        _go_idle_and_reroute(taxi, world)


def _raise_soc(taxi: Taxi, world: WorldState, power_kw: float, dt_min: float) -> None:
    """Increase SoC for ``dt_min`` minutes of charging at ``power_kw`` (clamped).

    Args:
        taxi: The taxi (mutated in place).
        world: The world (battery capacity and SoC_target).
        power_kw: Charging power, in kW (>= 0).
        dt_min: Simulated minutes of charging this tick (>= 0).

    Example:
        >>> from simulation.config.default import default_config
        >>> from simulation.world.routing import Router
        >>> from simulation.world.entities import Taxi
        >>> cfg = default_config()
        >>> w = WorldState(config=cfg, router=Router(speed_kmh=cfg.taxi_spec.speed_kmh))
        >>> t = Taxi(id="t0", position=(0.0, 0.0), soc=0.5)
        >>> _raise_soc(t, w, 0.0, 5.0)
        >>> t.soc
        0.5
    """

    spec = world.config.taxi_spec
    if spec.battery_capacity_kwh <= 0.0:
        return
    energy_kwh = power_kw * (dt_min / _MINUTES_PER_HOUR)
    delta = energy_kwh / spec.battery_capacity_kwh
    taxi.soc = min(world.config.battery.SoC_target, taxi.soc + delta)


def _finish_charging(taxi: Taxi, world: WorldState, energy_holder: str, kind: str) -> None:
    """End a regular charging session: release the slot, clear it, go idle.

    The fleet allocation booked on the port is left in place for the remainder
    of its grid span (it was a real reservation); the taxi simply detaches and
    re-enters idle logic. ``reservation`` and ``charge_until_min`` are cleared.

    Args:
        taxi: The taxi (mutated in place).
        world: The world.
        energy_holder: Identifier recorded on the metrics hook (the taxi id).
        kind: Allocation kind that was charging (``"fleet"``).

    Example:
        >>> from simulation.config.default import default_config
        >>> from simulation.world.routing import Router
        >>> from simulation.world.entities import Taxi, Reservation, TaxiState
        >>> cfg = default_config()
        >>> w = WorldState(config=cfg, router=Router(speed_kmh=cfg.taxi_spec.speed_kmh))
        >>> res = Reservation("p", "s", 0, 1, 3000.0, departure_time_min=0.0)
        >>> t = Taxi(id="t0", position=(-6.2, 106.8), soc=1.0,
        ...          state=TaxiState.CHARGING, reservation=res)
        >>> _finish_charging(t, w, "t0", "fleet")
        >>> t.reservation is None
        True
    """

    world.metrics_hook("charging_finished", taxi_id=taxi.id, holder=energy_holder, kind=kind)
    taxi.reservation = None
    taxi.charge_until_min = None
    _go_idle_and_reroute(taxi, world)
