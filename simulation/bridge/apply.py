"""Apply a deal / no-deal / partial outcome back to the world (``05-04``).

After a negotiation the bridge writes the agreed bundle into the live world:

* :func:`apply_deal` — for each **assigned** taxi: shift any conflicting shiftable
  app-user booking into its (empty) fallback destination and pay the compensation;
  lock the won ``(taxi, port, slot)`` as a ``"fleet"`` allocation (a hard
  constraint for future negotiations, Gotcha G11); set the taxi ``RESERVED`` with a
  :class:`~simulation.world.entities.Reservation` and ``departure_time = slot_start
  - travel_time(position -> port) - departure_margin``; reset ``cycles_waited = 0``
  (Gotcha G7). UNASSIGNED taxis inside the deal are handled by
  :func:`handle_partial`.
* :func:`handle_no_deal` — every candidate goes back to idle; ``cycles_waited += 1``
  for taxis that **participated but won no slot** (Gotcha G7). Taxis that left the
  candidate pool voluntarily are never penalised here (the scheduler does not pass
  them in).
* :func:`handle_partial` — for the UNASSIGNED taxis inside an accepted deal, apply
  the no-deal treatment to just those taxis (``cycles_waited += 1``), leaving the
  assigned taxis untouched.

Metrics are emitted through ``world.metrics_hook`` (default no-op; the scheduler
installs the real collector). This module only mutates the world; it never touches
the negotiation mechanism.

Example
-------
>>> hasattr(apply_deal, "__call__")
True
"""

from __future__ import annotations

from typing import Iterable, List

from simulation.world.entities import Port, Reservation, Taxi, TaxiState
from simulation.world.world_state import WorldState
from simulation.bridge.action_library import Assignment, BookingShift, Bundle

# Half-hour grid width in minutes (structural constant, see ``01-08``).
_GRID_MINUTES: int = 30


def apply_deal(world: WorldState, deal: Bundle) -> None:
    """Apply an accepted bundle to the world (``05-04``).

    Shifts conflicting shiftable bookings, locks won slots as hard constraints,
    reserves the assigned taxis (with a computed departure time), resets their
    ``cycles_waited``, and records metrics. UNASSIGNED taxis in the deal are left
    for :func:`handle_partial`.

    Args:
        world: The live :class:`WorldState` (mutated in place).
        deal: The agreed :class:`Bundle`.

    Example:
        >>> from simulation.config.default import default_config
        >>> from simulation.world.routing import Router
        >>> from simulation.world.entities import (Taxi, TaxiState,
        ...     ChargingStation, Port)
        >>> cfg = default_config()
        >>> w = WorldState(config=cfg, router=Router(speed_kmh=cfg.taxi_spec.speed_kmh))
        >>> port = Port("p", "s", 50.0, 2500.0)
        >>> w.stations = {"s": ChargingStation("s", (-6.2, 106.8), [port])}
        >>> w.taxis = {"t0": Taxi(id="t0", position=(-6.2, 106.8), soc=0.3)}
        >>> from simulation.bridge.action_library import Assignment, Bundle
        >>> deal = Bundle((Assignment("t0", "p", 4, 2, 3000.0),))
        >>> apply_deal(w, deal)
        >>> w.taxis["t0"].state is TaxiState.RESERVED
        True
    """

    for assignment in deal.assignments:
        if assignment.is_unassigned():
            continue
        taxi = world.taxis.get(assignment.taxi_id)
        if taxi is None:
            world.metrics_hook(
                "apply_deal_taxi_missing",
                taxi_id=assignment.taxi_id,
            )
            continue
        port = world.get_port(assignment.port_id)
        if port is None:
            world.metrics_hook(
                "apply_deal_port_missing",
                taxi_id=assignment.taxi_id,
                port_id=assignment.port_id,
            )
            continue

        if assignment.shift is not None:
            _perform_booking_shift(world, assignment.shift)

        _lock_slot(port, assignment)
        _reserve_taxi(world, taxi, port, assignment)
        taxi.cycles_waited = 0

        # Actual energy delivered = energy the taxi needs from arrival SoC to
        # SoC_target.  The action_library uses ceil() when converting energy to
        # grids, so the booked slot capacity is always >= actual need; billing
        # the full slot capacity would overcharge the fleet.
        spec = world.config.taxi_spec
        station = world.station_of_port(port.id)
        distance_km = (
            world.router.distance(taxi.position, station.location)
            if station is not None else 0.0
        )
        arrival_soc = max(
            0.0,
            taxi.soc - (distance_km * spec.consumption_kwh_per_km)
            / spec.battery_capacity_kwh
            if spec.battery_capacity_kwh > 0.0 else taxi.soc,
        )
        energy_to_fill_kwh = (
            max(0.0, spec.battery_capacity_kwh * (world.config.battery.SoC_target - arrival_soc))
            if spec.battery_capacity_kwh > 0.0 else 0.0
        )
        slot_capacity_kwh = port.power_kw * (assignment.duration_grids * _GRID_MINUTES / 60.0)
        energy_kwh = min(slot_capacity_kwh, energy_to_fill_kwh)
        world.metrics_hook(
            "deal_applied",
            taxi_id=taxi.id,
            port_id=port.id,
            slot_start=assignment.slot_start,
            duration_grids=assignment.duration_grids,
            price_per_kwh=assignment.price_per_kwh,
            energy_kwh=energy_kwh,
            revenue=assignment.price_per_kwh * energy_kwh,
            port_hours=assignment.duration_grids * _GRID_MINUTES / 60.0,
            compensation=assignment.shift.compensation if assignment.shift else 0.0,
            shifted_bookings=1 if assignment.shift else 0,
        )


def _perform_booking_shift(world: WorldState, shift: BookingShift) -> None:
    """Move a shiftable booking to its empty fallback destination (``05-04``)."""
    booking = world.bookings.get(shift.booking_id)
    src_port = world.get_port(shift.from_port_id)
    dst_port = world.get_port(shift.to_port_id)
    if src_port is not None:
        src_port.allocations = [
            alloc for alloc in src_port.allocations
            if not (alloc[2] == shift.booking_id and alloc[3] == "booking")
        ]
    if dst_port is not None:
        dst_port.add_allocation(
            shift.to_slot_start, shift.duration_grids, shift.booking_id, "booking"
        )
    if booking is not None:
        booking.port_id = shift.to_port_id
        booking.slot_start = shift.to_slot_start
        booking.compensation = shift.compensation
        # The single fallback is now consumed: the booking can never be shifted
        # again (it becomes a hard constraint for future negotiations, G11).
        booking.fallback_plan = None
        booking.is_shiftable = False
    world.metrics_hook(
        "booking_shifted",
        booking_id=shift.booking_id,
        from_port_id=shift.from_port_id,
        to_port_id=shift.to_port_id,
        compensation=shift.compensation,
    )


def _lock_slot(port: Port, assignment: Assignment) -> None:
    """Lock the won slot as a ``"fleet"`` allocation (hard constraint, G11).

    Idempotent against the state machine: the allocation is only added when the
    span is still free, so the later ``_step_to_port`` arrival does not double-add.
    """
    if port.is_free(assignment.slot_start, assignment.duration_grids):
        port.add_allocation(
            assignment.slot_start, assignment.duration_grids, assignment.taxi_id, "fleet"
        )


def _reserve_taxi(
    world: WorldState, taxi: Taxi, port: Port, assignment: Assignment
) -> None:
    """Set the taxi RESERVED with a reservation and computed departure time."""
    station = world.station_of_port(port.id)
    if station is not None:
        distance_km = world.router.distance(taxi.position, station.location)
        travel_min = world.router.travel_time_min(distance_km)
    else:
        travel_min = 0.0
    slot_start_min = assignment.slot_start * _GRID_MINUTES
    departure_margin = world.config.future_slot.departure_margin_min
    departure_time_min = float(slot_start_min) - travel_min - float(departure_margin)

    taxi.reservation = Reservation(
        port_id=port.id,
        station_id=port.station_id,
        slot_start=assignment.slot_start,
        duration_grids=assignment.duration_grids,
        price_per_kwh=assignment.price_per_kwh,
        departure_time_min=departure_time_min,
    )
    taxi.state = TaxiState.RESERVED
    taxi.move_target = None


def handle_no_deal(world: WorldState, candidates: Iterable[Taxi]) -> None:
    """Apply the no-deal outcome: idle candidates, raise their ``cycles_waited``.

    Every candidate that participated but won no slot ages by one cycle (Gotcha
    G7) and returns to idle (it remains a charge candidate if still in the
    critical/middle zone). Already-reserved/charging taxis are left untouched.

    Args:
        world: The live world (mutated in place).
        candidates: The taxis that participated in the negotiation.

    Example:
        >>> from simulation.config.default import default_config
        >>> from simulation.world.routing import Router
        >>> from simulation.world.entities import Taxi
        >>> w = WorldState(config=default_config(),
        ...                router=Router(speed_kmh=30.0))
        >>> t = Taxi(id="t0", position=(0.0, 0.0), soc=0.2)
        >>> w.taxis = {"t0": t}
        >>> handle_no_deal(w, [t])
        >>> t.cycles_waited
        1
    """
    _age_unwon(world, candidates)


def handle_partial(world: WorldState, deal: Bundle) -> None:
    """Apply the partial-deal treatment to the UNASSIGNED taxis in a deal (``05-04``).

    The taxis that received a slot were already processed by :func:`apply_deal`;
    here only the UNASSIGNED ones in the same bundle are aged (``cycles_waited
    += 1``) and returned to idle, exactly as if they had no deal.

    Args:
        world: The live world (mutated in place).
        deal: The accepted :class:`Bundle`.

    Example:
        >>> from simulation.config.default import default_config
        >>> from simulation.world.routing import Router
        >>> from simulation.world.entities import Taxi, NONE_PORT
        >>> w = WorldState(config=default_config(),
        ...                router=Router(speed_kmh=30.0))
        >>> t = Taxi(id="t0", position=(0.0, 0.0), soc=0.2)
        >>> w.taxis = {"t0": t}
        >>> from simulation.bridge.action_library import Assignment, Bundle
        >>> handle_partial(w, Bundle((Assignment("t0", NONE_PORT.id, 0, 0, 0.0),)))
        >>> t.cycles_waited
        1
    """
    unassigned_taxis: List[Taxi] = []
    for assignment in deal.assignments:
        if not assignment.is_unassigned():
            continue
        taxi = world.taxis.get(assignment.taxi_id)
        if taxi is not None:
            unassigned_taxis.append(taxi)
    _age_unwon(world, unassigned_taxis)


def _age_unwon(world: WorldState, taxis: Iterable[Taxi]) -> None:
    """Raise ``cycles_waited`` and re-idle taxis that participated but won nothing."""
    for taxi in taxis:
        # A taxi that ended up reserved/charging this round won a slot — skip it.
        if taxi.state in (
            TaxiState.RESERVED,
            TaxiState.TO_PORT,
            TaxiState.CHARGING,
            TaxiState.EMERGENCY_CHARGING,
        ):
            continue
        taxi.cycles_waited += 1
        if taxi.state not in (TaxiState.IDLE_IN_AREA, TaxiState.IDLE_WAITING):
            taxi.state = TaxiState.IDLE_WAITING
        world.metrics_hook("no_slot_aged", taxi_id=taxi.id, cycles_waited=taxi.cycles_waited)
