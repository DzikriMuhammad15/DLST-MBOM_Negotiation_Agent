"""Ground-truth world entities for the simulated EV-taxi world.

This module defines every value object and mutable entity of the simulation
world described in ``.claude/project_instruction/01-dunia-simulasi/01-01-entitas.md``:
the ``TaxiState`` / ``OrderState`` enums, the passenger ``Order``, the
app-user ``AppUserBooking`` + its ``FallbackPlan``, the charging ``Port`` /
``ChargingStation``, the ``MarkingArea``, the future-slot ``Reservation``, and
the ``Taxi`` itself.

It depends only on the standard library and on the simulation config layer
(``simulation.config``). It never imports ``negotiation`` (layer separation,
see ``00-MULAI-DI-SINI`` rule #4).

Coordinate convention: every ``position`` / ``location`` / ``center`` /
``pickup_point`` / ``destination`` is a ``(lat, lng)`` tuple of floats.

A grid slot is a half-hour index of the simulation timeline (see ``01-08``):
grid ``g`` covers simulated minutes ``[g * 30, (g + 1) * 30)``. A port
allocation occupies one or more contiguous grids.

Example
-------
>>> from simulation.world.entities import Port, Reservation
>>> port = Port(id="s0_p0", station_id="s0", power_kw=50.0, base_price_per_kwh=2500.0)
>>> port.is_free(slot_start=2, duration_grids=2)
True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

Coordinate = Tuple[float, float]

# A single port allocation: (slot_start_grid, duration_grids, holder_id, kind).
# ``kind`` is one of "booking" (app-user EV) | "fleet" (won by a taxi in
# negotiation) | "emergency" (rescue charging). See 01-01 / 01-08 / 05-05.
Allocation = Tuple[int, int, str, str]


class TaxiState(Enum):
    """Finite states of a taxi (see ``01-06-state-machine-taksi.md``).

    Members:
        IDLE_IN_AREA: Idle inside a marking area (SoC frozen), available to be
            matched to an order spawned in that area.
        IDLE_WAITING: Idle and **stuck** outside any area, waiting in place (SoC
            frozen). A taxi is IDLE_WAITING only when it cannot reach any marking
            area while keeping ``SoC >= B_area``; this is exactly the set of
            negotiation charge candidates.
        TO_MARKING_AREA: Repositioning toward the nearest reachable marking area
            (it satisfies the area battery constraint: arrival ``SoC >= B_area``).
            Drains SoC while moving; on arrival becomes IDLE_IN_AREA. Such a taxi
            is **not** a negotiation candidate.
        TO_PASSENGER: Driving to an order's pickup point (unproductive km).
        WITH_PASSENGER: Carrying a passenger to the destination (productive km).
        RESERVED: Won a future charging slot; waits **in place** until its
            departure time (it never repositions to an area and never takes an
            order — only taxis satisfying the area constraint may head to an area).
        TO_PORT: Driving to the reserved port for its slot.
        CHARGING: Charging at a regular port (SoC rises).
        STRANDED: Out of battery, could not reach any port (sikap A, failure).
        EMERGENCY_CHARGING: Charging at the emergency station (sikap B, teleport).

    Example:
        >>> TaxiState.IDLE_IN_AREA.name
        'IDLE_IN_AREA'
    """

    IDLE_IN_AREA = "IDLE_IN_AREA"
    IDLE_WAITING = "IDLE_WAITING"
    TO_MARKING_AREA = "TO_MARKING_AREA"
    TO_PASSENGER = "TO_PASSENGER"
    WITH_PASSENGER = "WITH_PASSENGER"
    RESERVED = "RESERVED"
    TO_PORT = "TO_PORT"
    CHARGING = "CHARGING"
    STRANDED = "STRANDED"
    EMERGENCY_CHARGING = "EMERGENCY_CHARGING"


class OrderState(Enum):
    """Lifecycle states of a passenger order (see ``01-03``).

    Members:
        PENDING: Spawned and not yet claimed by a taxi.
        CLAIMED: A taxi has committed to serving it (driving to pickup).
        IN_PROGRESS: Passenger is on board (taxi WITH_PASSENGER).
        COMPLETED: Passenger delivered to destination.
        EXPIRED: Not served before the hour rolled over.

    Example:
        >>> OrderState.PENDING.name
        'PENDING'
    """

    PENDING = "PENDING"
    CLAIMED = "CLAIMED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    EXPIRED = "EXPIRED"


@dataclass
class Order:
    """A passenger order that may be served by an idle taxi (see ``01-03``).

    Attributes:
        id: Unique identifier of the order.
        area_id: Identifier of the marking area the order belongs to.
        pickup_point: ``(lat, lng)`` random point inside the area.
        destination: ``(lat, lng)`` random point on the map.
        spawn_hour: Hour-of-day (0..23) the order appeared.
        state: Current :class:`OrderState`.
        assigned_taxi: Id of the serving taxi, or ``None`` while PENDING.

    Example:
        >>> o = Order(id="o0", area_id="a0", pickup_point=(-6.2, 106.8),
        ...           destination=(-6.3, 106.9), spawn_hour=8)
        >>> o.state is OrderState.PENDING
        True
    """

    id: str
    area_id: str
    pickup_point: Coordinate
    destination: Coordinate
    spawn_hour: int
    state: OrderState = OrderState.PENDING
    assigned_taxi: Optional[str] = None


@dataclass
class FallbackPlan:
    """The single concrete fallback placement an app-user booking will accept.

    Every booking has exactly one fallback plan: one alternative ``(port, slot)``
    of the **same duration** as the booking, fixed at simulation start. When the
    booking is shifted (a taxi takes its original slot) it moves here and the
    plan is consumed (``fallback_plan`` becomes ``None``), so the booking can
    never be shifted a second time (see ``01-01`` / ``05-04``).

    Attributes:
        to_port_id: Port the booking moves to when shifted (may equal the
            booking's original port).
        to_slot_start: First half-hour grid of the booking after the shift.
        compensation: Total Rupiah compensation owed to the app-user for using
            this fallback (energy-based, fixed at creation).

    Example:
        >>> fp = FallbackPlan(to_port_id="s0_p1", to_slot_start=6,
        ...                   compensation=7500.0)
        >>> fp.compensation
        7500.0
    """

    to_port_id: str
    to_slot_start: int
    compensation: float


@dataclass
class AppUserBooking:
    """A general-EV (non-taxi) charging booking occupying port grid slots.

    Attributes:
        id: Unique identifier of the booking.
        port_id: Identifier of the port the booking occupies.
        slot_start: First half-hour grid index occupied.
        duration_grids: Number of contiguous grids occupied.
        is_shiftable: Whether the booking still has an unused fallback plan and
            may therefore be moved by negotiation. Starts ``True`` for every
            booking; becomes ``False`` once the booking has been shifted (its
            single fallback is consumed), making it a hard constraint thereafter
            (Gotcha G11). Always equals ``fallback_plan is not None``.
        fallback_plan: The single fallback placement, or ``None`` once consumed.
        compensation: Rupiah compensation actually paid for shifting this
            booking (``0.0`` until it is shifted; set at apply time from the
            fallback plan).

    Example:
        >>> b = AppUserBooking(id="b0", port_id="s0_p0", slot_start=2,
        ...                    duration_grids=1, is_shiftable=True)
        >>> b.is_shiftable
        True
    """

    id: str
    port_id: str
    slot_start: int
    duration_grids: int
    is_shiftable: bool
    fallback_plan: Optional[FallbackPlan] = None
    compensation: float = 0.0


@dataclass
class Port:
    """A single charging port at a station (see ``01-01`` / ``01-08``).

    The ``allocations`` list is the authoritative occupancy record: each entry
    is ``(slot_start, duration_grids, holder, kind)``. App-user bookings,
    fleet-won slots, and emergency charging all materialize as allocations so
    that :meth:`is_free` can answer feasibility uniformly.

    Attributes:
        id: Identifier of the port, unique within its station.
        station_id: Identifier of the owning station.
        power_kw: Charging power in kW (charge speed; energy per grid).
        base_price_per_kwh: Base price per kWh (part of SPKLU spec).
        allocations: List of ``(slot_start, duration_grids, holder, kind)``.

    Example:
        >>> p = Port(id="s0_p0", station_id="s0", power_kw=50.0,
        ...          base_price_per_kwh=2500.0)
        >>> p.add_allocation(2, 2, "b0", "booking")
        >>> p.is_free(2, 1)
        False
        >>> p.is_free(4, 1)
        True
    """

    id: str
    station_id: str
    power_kw: float
    base_price_per_kwh: float
    allocations: List[Allocation] = field(default_factory=list)

    def is_free(self, slot_start: int, duration_grids: int) -> bool:
        """Return ``True`` if grids ``[slot_start, slot_start + duration)`` are unoccupied.

        Two allocations overlap iff their half-open grid intervals intersect.

        Args:
            slot_start: First half-hour grid index of the requested span.
            duration_grids: Number of contiguous grids requested (>= 1).

        Returns:
            ``True`` if no existing allocation overlaps the requested span.

        Example:
            >>> p = Port(id="p", station_id="s", power_kw=50.0, base_price_per_kwh=0.0)
            >>> p.add_allocation(0, 2, "x", "booking")
            >>> p.is_free(1, 1)
            False
            >>> p.is_free(2, 3)
            True
        """

        req_end = slot_start + duration_grids
        for start, duration, _holder, _kind in self.allocations:
            end = start + duration
            if start < req_end and slot_start < end:
                return False
        return True

    def add_allocation(self, slot_start: int, duration_grids: int, holder: str, kind: str) -> None:
        """Append an occupancy record to this port.

        Args:
            slot_start: First half-hour grid index occupied.
            duration_grids: Number of contiguous grids occupied (>= 1).
            holder: Identifier of the booking/taxi/rescue occupying the span.
            kind: ``"booking"`` | ``"fleet"`` | ``"emergency"``.

        Example:
            >>> p = Port(id="p", station_id="s", power_kw=50.0, base_price_per_kwh=0.0)
            >>> p.add_allocation(0, 1, "b0", "booking")
            >>> len(p.allocations)
            1
        """

        self.allocations.append((slot_start, duration_grids, holder, kind))

    def free_grids_from(self, grid: int, horizon_grids: int) -> int:
        """Count free grids within ``[grid, grid + horizon_grids)``.

        Used by feature builders/idle logic for a cheap "how much capacity is
        coming up" signal.

        Args:
            grid: First grid index of the look-ahead window.
            horizon_grids: Window length in grids (>= 0).

        Returns:
            Number of free grids in the window.

        Example:
            >>> p = Port(id="p", station_id="s", power_kw=50.0, base_price_per_kwh=0.0)
            >>> p.add_allocation(1, 1, "b0", "booking")
            >>> p.free_grids_from(0, 3)
            2
        """

        free = 0
        for offset in range(horizon_grids):
            if self.is_free(grid + offset, 1):
                free += 1
        return free


@dataclass
class ChargingStation:
    """A charging station (SPKLU) and its ports (see ``01-01`` / ``05-05``).

    Attributes:
        id: Identifier of the station.
        location: ``(lat, lng)`` fixed location.
        ports: Ports belonging to this station.
        is_emergency: ``True`` only for the emergency station (rescue, sikap B).
            Emergency stations never enter the negotiation / action_library
            (Gotcha G8) and have effectively unlimited capacity.

    Example:
        >>> s = ChargingStation(id="s0", location=(-6.2, 106.8),
        ...                     ports=[Port("s0_p0", "s0", 50.0, 2500.0)])
        >>> s.is_emergency
        False
    """

    id: str
    location: Coordinate
    ports: List[Port] = field(default_factory=list)
    is_emergency: bool = False


@dataclass
class MarkingArea:
    """A marking area where passenger orders may appear (see ``01-03``).

    Attributes:
        id: Identifier of the area.
        center: ``(lat, lng)`` center point (also the fixed-point distance
            reference used by the precomputed matrix).
        current_orders: Orders active during the current hour.

    Example:
        >>> a = MarkingArea(id="a0", center=(-6.2, 106.8))
        >>> a.current_orders
        []
    """

    id: str
    center: Coordinate
    current_orders: List[Order] = field(default_factory=list)

    def unfilled_orders(self) -> List[Order]:
        """Return orders in this area that are still PENDING (unclaimed).

        Returns:
            The subset of ``current_orders`` whose state is
            :attr:`OrderState.PENDING`.

        Example:
            >>> a = MarkingArea(id="a0", center=(-6.2, 106.8))
            >>> a.current_orders = [Order("o0", "a0", (-6.2, 106.8),
            ...                           (-6.3, 106.9), 8)]
            >>> len(a.unfilled_orders())
            1
        """

        return [order for order in self.current_orders if order.state is OrderState.PENDING]


@dataclass
class Reservation:
    """A future charging slot a taxi has won in negotiation (see ``01-08``).

    Attributes:
        port_id: Identifier of the reserved port.
        station_id: Identifier of the reserved port's station.
        slot_start: First half-hour grid index of the booked span.
        duration_grids: Number of contiguous grids booked.
        price_per_kwh: Agreed price per kWh (one of the discrete price levels).
        departure_time_min: Simulated minute at which the taxi must leave for
            the port (``slot_start_min - travel_time - departure_margin``).

    Example:
        >>> r = Reservation(port_id="s0_p0", station_id="s0", slot_start=18,
        ...                 duration_grids=2, price_per_kwh=3000.0,
        ...                 departure_time_min=520.0)
        >>> r.slot_start
        18
    """

    port_id: str
    station_id: str
    slot_start: int
    duration_grids: int
    price_per_kwh: float
    departure_time_min: float


@dataclass
class Taxi:
    """A single electric taxi — the only entity whose state evolves richly.

    The uniform spec fields (capacity/consumption/speed) live globally on the
    config, not here (see ``01-01``); this object carries the per-taxi mutable
    state plus precise distance accounting.

    Attributes:
        id: Unique identifier of the taxi.
        position: Current ``(lat, lng)`` coordinate.
        soc: State of charge in ``[0, 1]``.
        state: Current :class:`TaxiState`.
        cycles_waited: Persistent anti-starvation counter (Gotcha G7). Raised
            by one only when the taxi joined a negotiation but won no slot;
            reset to 0 when it wins a slot. Never raised by voluntarily
            leaving the candidate pool in the middle zone.
        current_order: Order currently being served, or ``None``.
        reservation: Won future slot, or ``None``.
        current_area: Marking area id the taxi is idling in, or ``None``.
        move_target: ``(lat, lng)`` the taxi is currently driving toward, or
            ``None`` when not moving.
        unproductive_km: Accumulated km driven without a paying passenger
            (idle -> pickup, to-port, re-route, wasted steal trips). Metric only.
        productive_km: Accumulated km driven with a passenger (pickup ->
            destination). Drives fare revenue.
        charge_until_min: Simulated minute at which the current CHARGING /
            EMERGENCY_CHARGING session is expected to end, or ``None``.

    Example:
        >>> from simulation.world.entities import Taxi, TaxiState
        >>> t = Taxi(id="taxi_0", position=(-6.2, 106.8), soc=0.8)
        >>> t.state is TaxiState.IDLE_WAITING
        True
        >>> t.cycles_waited
        0
    """

    id: str
    position: Coordinate
    soc: float
    state: TaxiState = TaxiState.IDLE_WAITING
    cycles_waited: int = 0
    current_order: Optional[Order] = None
    reservation: Optional[Reservation] = None
    current_area: Optional[str] = None
    move_target: Optional[Coordinate] = None
    unproductive_km: float = 0.0
    productive_km: float = 0.0
    charge_until_min: Optional[float] = None


# Sentinel "no port" value used wherever a bundle assignment leaves a taxi
# UNASSIGNED (see 03-04 / 05-01). It is a real Port object so callers can read
# ``.id`` / ``.is_free`` uniformly without ``None`` checks; its identity
# ``NONE_PORT`` is what marks a slot as unassigned.
NONE_PORT: Port = Port(
    id="__NONE_PORT__",
    station_id="__NONE_STATION__",
    power_kw=0.0,
    base_price_per_kwh=0.0,
)
