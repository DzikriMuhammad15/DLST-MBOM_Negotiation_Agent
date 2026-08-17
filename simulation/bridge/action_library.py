"""Bundle / Assignment outcome types and the action-library generator (``03-04``/``05-01``).

One negotiation decides a charging slot for the candidate taxis at once, so a
single negotiation outcome is one :class:`Bundle` — a tuple of :class:`Assignment`
(one per candidate taxi), where an assignment is either a concrete
``(port, slot, duration, price)`` or UNASSIGNED (the sentinel
:data:`~simulation.world.entities.NONE_PORT`). The ordered list of bundles is the
``action_library``: outcome ``(idx,)`` -> ``action_library[idx]`` (index-direct,
Gotcha G1).

:func:`build_action_library` builds the library from concrete combinations:

1. **Per-taxi assignment options.** For each candidate taxi and each regular port
   it can reach (enough battery to arrive, exact distance), one option is formed:
   - the **start grid** is the grid the taxi *arrives* in — the next grid boundary
     at/after its exact arrival time (``ceil(arrival_min / 30)``);
   - the **duration** is the number of grids needed to fill the taxi from its
     exact arrival SoC back to ``SoC_target``;
   - one option per allowed ``price_levels`` value (all of them, combined).
   The former ``port_radius`` "SPKLU area" gating is gone: a taxi can be paired
   with *any* port it has the battery to reach.
2. **Bundles = greedy FULL-ASSIGNMENT packings** (FIXED MECHANISM, 2026-07-08 —
   replaces the old cartesian product with optional per-taxi UNASSIGNED
   choices). Every bundle jointly assigns EVERY candidate that can physically
   be placed (packing order: most critical SoC first, then longest queue).
   The UNASSIGNED sentinel appears only for candidates that cannot be placed
   at all in that packing (no reachable port, or every compatible span already
   taken by a higher-priority candidate) — bundles never *choose* to leave an
   assignable taxi idle. Menu diversity comes from deterministic
   **port-preference rotation** variants (nearest-first, rotated per variant)
   crossed with one **uniform price level** per bundle, so the library spans
   the full price range (ZOPA) and several port allocations. The single
   all-UNASSIGNED bundle remains excluded (that means "no deal").
3. **Feasibility re-check** (:func:`_bundle_is_feasible`): no two assigned taxis
   overlap on the same port+slot; a booking shift target does not collide with a
   taxi-claimed span or with another shift target (this is what forbids a bundle
   from taking both a booking's original slot *and* its fallback slot — Gotcha
   G11 / ``04-03`` / ``05-01``). The greedy packer enforces the same rules
   incrementally; the re-check is a safety net.
4. **K_max cut.** If more than ``max_action_library`` (``K_max``) feasible bundles
   exist, keep the ``K_max`` bundles with the highest **fleet** utility.

Hard constraints on which slots a taxi may take (enforced when building options):

* an **empty** port grid span is always takeable;
* a span carrying exactly **one** app-user booking that **still has a fallback
  plan** is takeable by shifting that booking to its (empty) fallback destination;
* a span carrying a booking **without** a remaining fallback (already shifted), a
  **fleet-won** slot, an **emergency** allocation, or **more than one** booking is
  never takeable;
* the **emergency** station never enters the library (Gotcha G8) — it is excluded
  upstream from ``world.all_ports()``.

Because taxi speed and energy consumption are constant, arrival time and arrival
SoC (and therefore every start grid / duration above) are **exact**.

Example
-------
>>> from simulation.world.entities import NONE_PORT
>>> a = Assignment(taxi_id="t0", port_id=NONE_PORT.id, slot_start=0,
...                duration_grids=0, price_per_kwh=0.0)
>>> a.is_unassigned()
True
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from simulation.config.config import SimulationConfig
from simulation.world.entities import NONE_PORT, AppUserBooking, Port, Taxi
from simulation.world.world_state import WorldState

# Half-hour grid width in minutes (structural constant, see ``01-08``).
_GRID_MINUTES: int = 30
_GRID_HOURS: float = _GRID_MINUTES / 60.0
# Number of port-preference rotation variants generated per price level. Each
# variant is one greedy full-assignment packing; with |price_levels| uniform
# price tiers the raw menu is ``variants x prices`` bundles before the K_max
# cut. Capped so single-taxi calls still cover up to this many distinct ports.
_MAX_PLACEMENT_VARIANTS: int = 16


@dataclass(frozen=True)
class BookingShift:
    """A planned shift of one shiftable app-user booking (recorded in a bundle).

    Produced by the generator when a taxi takes a slot occupied by a *shiftable*
    booking; the booking moves to its single fallback destination. Consumed by
    :mod:`simulation.bridge.apply` (which performs the move, pays the
    compensation, and consumes the booking's fallback) and by the SPKLU ufun
    (which charges ``compensation``).

    Attributes:
        booking_id: Identifier of the booking being shifted.
        from_port_id: Port the booking currently occupies.
        to_port_id: Port the booking moves to (its fallback port).
        from_slot_start: Current first grid of the booking.
        to_slot_start: Target first grid of the booking after the shift.
        duration_grids: Booking duration (unchanged by a shift).
        compensation: Rupiah compensation owed to the app-user for this shift.

    Example:
        >>> s = BookingShift(booking_id="b0", from_port_id="p0", to_port_id="p1",
        ...                  from_slot_start=2, to_slot_start=4, duration_grids=1,
        ...                  compensation=5000.0)
        >>> s.compensation
        5000.0
    """

    booking_id: str
    from_port_id: str
    to_port_id: str
    from_slot_start: int
    to_slot_start: int
    duration_grids: int
    compensation: float


@dataclass(frozen=True)
class Assignment:
    """One taxi's decision inside a bundle: a concrete slot, or UNASSIGNED.

    An UNASSIGNED assignment uses the sentinel
    :data:`~simulation.world.entities.NONE_PORT` id and zero duration; otherwise
    it names a port, a half-hour grid span ``[slot_start, slot_start + duration)``,
    and a price drawn from ``price_levels``.

    Attributes:
        taxi_id: Identifier of the taxi this assignment is for.
        port_id: Port id, or ``NONE_PORT.id`` when UNASSIGNED.
        slot_start: First half-hour grid of the charging span (0 when UNASSIGNED).
        duration_grids: Number of contiguous grids (0 when UNASSIGNED).
        price_per_kwh: Agreed price per kWh (0 when UNASSIGNED).
        shift: Optional booking shift this assignment requires (else ``None``).

    Example:
        >>> a = Assignment(taxi_id="t0", port_id="p0", slot_start=4,
        ...                duration_grids=2, price_per_kwh=3000.0)
        >>> a.is_unassigned()
        False
    """

    taxi_id: str
    port_id: str
    slot_start: int
    duration_grids: int
    price_per_kwh: float
    shift: Optional[BookingShift] = None

    def is_unassigned(self) -> bool:
        """Return ``True`` if this assignment leaves the taxi without a slot.

        Example:
            >>> from simulation.world.entities import NONE_PORT
            >>> Assignment("t0", NONE_PORT.id, 0, 0, 0.0).is_unassigned()
            True
        """
        return self.port_id == NONE_PORT.id


@dataclass(frozen=True)
class Bundle:
    """One negotiation outcome: an assignment for every candidate taxi.

    Attributes:
        assignments: Ordered tuple of :class:`Assignment` (one per candidate
            taxi, in the candidate order). Tuple (not dict) so the bundle is
            hashable and dedup-able.

    Example:
        >>> from simulation.world.entities import NONE_PORT
        >>> b = Bundle(assignments=(
        ...     Assignment("t0", NONE_PORT.id, 0, 0, 0.0),
        ...     Assignment("t1", "p0", 4, 2, 3000.0),
        ... ))
        >>> b.assigned_count()
        1
        >>> b.is_all_unassigned()
        False
    """

    assignments: Tuple[Assignment, ...]

    def assigned(self) -> Tuple[Assignment, ...]:
        """Return the non-UNASSIGNED assignments.

        Example:
            >>> from simulation.world.entities import NONE_PORT
            >>> Bundle((Assignment("t0", NONE_PORT.id, 0, 0, 0.0),)).assigned()
            ()
        """
        return tuple(a for a in self.assignments if not a.is_unassigned())

    def assigned_count(self) -> int:
        """Return the number of taxis that receive a slot in this bundle."""
        return len(self.assigned())

    def is_all_unassigned(self) -> bool:
        """Return ``True`` if no taxi receives a slot.

        The generator never emits such a bundle (all-UNASSIGNED means "no deal",
        which only arises from running out of negotiation steps); this predicate
        is retained for metrics/serialisation, where it always reports ``False``
        for library bundles.
        """
        return self.assigned_count() == 0

    def shifts(self) -> Tuple[BookingShift, ...]:
        """Return every booking shift this bundle requires (across assignments)."""
        return tuple(a.shift for a in self.assignments if a.shift is not None)


# =============================================================================
# Feasibility / option helpers
# =============================================================================

def _energy_to_fill_kwh(taxi: Taxi, config: SimulationConfig, arrival_soc: float) -> float:
    """Return kWh needed to fill ``taxi`` from ``arrival_soc`` to ``SoC_target``."""
    spec = config.taxi_spec
    soc_target = config.battery.SoC_target
    return max(0.0, spec.battery_capacity_kwh * (soc_target - arrival_soc))


def _arrival_soc(taxi: Taxi, config: SimulationConfig, distance_km: float) -> float:
    """Return SoC after driving ``distance_km`` to a port (no charge yet)."""
    spec = config.taxi_spec
    if spec.battery_capacity_kwh <= 0.0:
        return taxi.soc
    drop = (distance_km * spec.consumption_kwh_per_km) / spec.battery_capacity_kwh
    return taxi.soc - drop


def _grids_to_fill(energy_to_fill_kwh: float, power_kw: float) -> int:
    """Return the number of half-hour grids to deliver ``energy_to_fill_kwh``.

    Energy per grid = ``power_kw * 0.5h``. At least one grid when any energy is
    needed; zero only when already full.

    Example:
        >>> _grids_to_fill(50.0, 50.0)
        2
        >>> _grids_to_fill(0.0, 50.0)
        0
    """
    if energy_to_fill_kwh <= 0.0:
        return 0
    energy_per_grid = power_kw * _GRID_HOURS
    if energy_per_grid <= 0.0:
        return 0
    return max(1, int(math.ceil(energy_to_fill_kwh / energy_per_grid)))


def _start_grid_for_arrival(world: WorldState, distance_km: float) -> int:
    """Return the grid the taxi can first start charging in (its arrival grid).

    The taxi arrives at ``sim_time_min + travel_time(distance)``; the start grid
    is the next grid boundary at/after that exact arrival minute
    (``ceil(arrival_min / 30)``). E.g. arriving at 09:10 -> grid for 09:30.
    Exact, because speed is constant.
    """
    travel_min = world.router.travel_time_min(distance_km)
    arrival_min = world.sim_time_min + travel_min
    return int(math.ceil(arrival_min / _GRID_MINUTES))


def _earliest_shiftable_grid(world: WorldState) -> int:
    """Return the first grid a booking may legally be shifted *into*.

    A fallback destination is picked once at world-assembly time (t=0), when no
    grid has elapsed yet, so the plan itself carries no notion of "past". By the
    time the shift is actually performed the clock has advanced, and a plan that
    points backwards in time would move an app user into a slot that has already
    been consumed. This is the boundary that makes such a move impossible.

    The rule matches :func:`_start_grid_for_arrival`: the next grid boundary
    at/after the current simulated minute (``ceil(sim_time_min / 30)``). At an
    exact boundary the current grid itself is still allowed, because it has not
    started being consumed yet.
    """
    return int(math.ceil(world.sim_time_min / _GRID_MINUTES))


def _span_free_ignoring(
    port: Port, slot_start: int, duration_grids: int, ignore_holder: str
) -> bool:
    """Return ``True`` if the span carries no allocation other than ``ignore_holder``.

    Used to verify a booking's fallback destination is empty before recording a
    shift into it (the booking will vacate its own span, so its own allocation is
    ignored).
    """
    req_end = slot_start + duration_grids
    for start, dur, holder, _kind in port.allocations:
        if holder == ignore_holder:
            continue
        end = start + dur
        if start < req_end and slot_start < end:
            return False
    return True


def _resolve_slot(
    world: WorldState, port: Port, slot_start: int, duration_grids: int
) -> Tuple[str, Optional[AppUserBooking]]:
    """Classify a ``(port, span)`` for action-library use.

    Returns one of:

    * ``("free", None)`` — the span carries no allocation; takeable directly.
    * ``("shiftable", booking)`` — the span is occupied by exactly one app-user
      booking that still has a fallback plan; takeable by shifting that booking.
    * ``("blocked", None)`` — the span carries a fleet/emergency allocation, a
      booking without a remaining fallback, or more than one booking; not takeable.
    """
    req_end = slot_start + duration_grids
    overlapping_booking_ids: List[str] = []
    for start, dur, holder, kind in port.allocations:
        end = start + dur
        if start < req_end and slot_start < end:
            if kind != "booking":
                return ("blocked", None)  # fleet / emergency: hard constraint.
            overlapping_booking_ids.append(holder)
    if not overlapping_booking_ids:
        return ("free", None)
    if len(overlapping_booking_ids) != 1:
        return ("blocked", None)
    booking = world.bookings.get(overlapping_booking_ids[0])
    if booking is None or booking.fallback_plan is None:
        return ("blocked", None)
    return ("shiftable", booking)


def _booking_shift_for(world: WorldState, booking: AppUserBooking) -> Optional[BookingShift]:
    """Return the :class:`BookingShift` to ``booking``'s fallback, or ``None``.

    ``None`` when the booking has no fallback left, its fallback destination lies
    in the past, or that destination is not currently empty (so the shift could
    not actually be performed).
    """
    plan = booking.fallback_plan
    if plan is None:
        return None
    if plan.to_slot_start < _earliest_shiftable_grid(world):
        return None  # destination already elapsed: cannot move a user backwards.
    to_port = world.get_port(plan.to_port_id)
    if to_port is None:
        return None
    if not _span_free_ignoring(to_port, plan.to_slot_start, booking.duration_grids, booking.id):
        return None
    return BookingShift(
        booking_id=booking.id,
        from_port_id=booking.port_id,
        to_port_id=plan.to_port_id,
        from_slot_start=booking.slot_start,
        to_slot_start=plan.to_slot_start,
        duration_grids=booking.duration_grids,
        compensation=plan.compensation,
    )


@dataclass(frozen=True)
class _Option:
    """An internal feasible assignment option for one taxi at one price."""

    port_id: str
    slot_start: int
    duration_grids: int
    distance_km: float
    price_per_kwh: float
    shift: Optional[BookingShift]


def _options_for_taxi(
    taxi: Taxi, world: WorldState, config: SimulationConfig
) -> List[_Option]:
    """Build every feasible assignment option for one taxi (``05-01``).

    For each regular port the taxi can reach (exact arrival SoC >= 0), one slot is
    derived (arrival grid + grids to refill); the slot must be empty or freeable
    by shifting a single still-shiftable booking. One option is emitted per
    allowed price level.
    """
    prices = sorted(set(float(p) for p in config.negotiation.price_levels))
    options: List[_Option] = []
    for port in world.all_ports():
        station = world.station_of_port(port.id)
        if station is None:
            continue
        distance_km = world.router.distance(taxi.position, station.location)
        arrival_soc = _arrival_soc(taxi, config, distance_km)
        if arrival_soc < 0.0:
            continue  # not enough battery to reach this port.
        energy = _energy_to_fill_kwh(taxi, config, arrival_soc)
        duration = _grids_to_fill(energy, port.power_kw)
        if duration <= 0:
            continue  # already full -> no charging slot needed.
        slot_start = _start_grid_for_arrival(world, distance_km)

        kind, booking = _resolve_slot(world, port, slot_start, duration)
        if kind == "blocked":
            continue
        shift: Optional[BookingShift] = None
        if kind == "shiftable" and booking is not None:
            shift = _booking_shift_for(world, booking)
            if shift is None:
                continue  # fallback destination not free -> cannot place here.

        for price in prices:
            options.append(
                _Option(
                    port_id=port.id,
                    slot_start=slot_start,
                    duration_grids=duration,
                    distance_km=distance_km,
                    price_per_kwh=price,
                    shift=shift,
                )
            )
    return options


# =============================================================================
# Bundle assembly + feasibility re-check
# =============================================================================

def _unassigned(taxi_id: str) -> Assignment:
    """Return an UNASSIGNED assignment for ``taxi_id``."""
    return Assignment(
        taxi_id=taxi_id,
        port_id=NONE_PORT.id,
        slot_start=0,
        duration_grids=0,
        price_per_kwh=0.0,
    )


def _bundle_is_feasible(bundle: Bundle) -> bool:
    """Return ``True`` if the bundle has no internal conflicts (``03-04``/``05-01``).

    Checks:

    * no two assigned taxis overlap on the same port+slot;
    * two booking shifts do not target the same destination span;
    * a booking shift target does not land on a taxi-claimed span in the same
      bundle — this is exactly what forbids a bundle from taking both a booking's
      original slot (which triggers the shift to its fallback) *and* that
      fallback slot (Gotcha G11).
    """
    occupied: Dict[str, List[Tuple[int, int]]] = {}
    shifted_targets: List[Tuple[str, int, int]] = []
    for a in bundle.assigned():
        spans = occupied.setdefault(a.port_id, [])
        a_end = a.slot_start + a.duration_grids
        for start, end in spans:
            if start < a_end and a.slot_start < end:
                return False
        spans.append((a.slot_start, a_end))
        if a.shift is not None:
            shifted_targets.append(
                (a.shift.to_port_id, a.shift.to_slot_start,
                 a.shift.to_slot_start + a.shift.duration_grids)
            )
    for i, (port_id, start, end) in enumerate(shifted_targets):
        for j, (other_port, other_start, other_end) in enumerate(shifted_targets):
            if i < j and port_id == other_port and other_start < end and start < other_end:
                return False
        for span_start, span_end in occupied.get(port_id, []):
            if span_start < end and start < span_end:
                return False
    return True


def _validate_library_dims(config: SimulationConfig) -> None:
    """Fail fast if the action-library cap does not match the padded head width.

    The PADDED propose head is ``Linear(K_max)`` and ``masked_fill``s with the
    ``(K_max,)`` candidate mask, so the candidate block width (= the action-library
    size, capped at ``max_action_library``) MUST equal
    ``state_feature.encoder_dims.K_max``. A mismatch would silently truncate the
    library past ``K_max`` or crash the head. Per ``CLAUDE.md`` a contract
    violation raises an explicit exception naming the offending fields.

    Raises:
        ValueError: If ``negotiation.max_action_library != state_feature.
            encoder_dims.K_max``.
    """
    max_lib = int(config.negotiation.max_action_library)
    k_max = int(config.state_feature.encoder_dims.K_max)
    if max_lib != k_max:
        raise ValueError(
            "Contract violation: negotiation.max_action_library "
            f"({max_lib}) must equal state_feature.encoder_dims.K_max "
            f"({k_max}); the PADDED propose head is Linear(K_max) and the "
            "candidate block is padded to K_max, so the two must match."
        )


def _distinct_placements(options: List[_Option]) -> List[_Option]:
    """Return one option per distinct ``(port, slot, duration)`` placement.

    ``_options_for_taxi`` emits one option per reachable port *per price level*;
    the placement geometry (port / slot / duration / shift) is price-independent,
    so packing works over distinct placements (nearest port first) and the price
    tier is applied afterwards.
    """
    seen: Dict[Tuple[str, int, int], _Option] = {}
    for opt in options:
        key = (opt.port_id, opt.slot_start, opt.duration_grids)
        if key not in seen:
            seen[key] = opt
    return sorted(seen.values(), key=lambda o: (o.distance_km, o.port_id))


def _spans_overlap(spans: List[Tuple[int, int]], start: int, end: int) -> bool:
    """Return ``True`` when ``[start, end)`` overlaps any span in ``spans``."""
    return any(s < end and start < e for s, e in spans)


def _try_place(
    opt: _Option,
    occupied: Dict[str, List[Tuple[int, int]]],
    shift_spans: Dict[str, List[Tuple[int, int]]],
) -> bool:
    """Try to add ``opt`` to a packing; record its spans and return ``True`` if it fits.

    Enforces :func:`_bundle_is_feasible`'s rules incrementally: the assignment
    span must not overlap another assignment or shift-target span on the same
    port, and the option's booking-shift target (if any) must not overlap
    assignments or other shift targets on the target port (Gotcha G11).
    """
    a_start, a_end = opt.slot_start, opt.slot_start + opt.duration_grids
    if _spans_overlap(occupied.get(opt.port_id, []), a_start, a_end):
        return False
    if _spans_overlap(shift_spans.get(opt.port_id, []), a_start, a_end):
        return False
    if opt.shift is not None:
        t_port = opt.shift.to_port_id
        t_start = opt.shift.to_slot_start
        t_end = t_start + opt.shift.duration_grids
        if _spans_overlap(occupied.get(t_port, []), t_start, t_end):
            return False
        if _spans_overlap(shift_spans.get(t_port, []), t_start, t_end):
            return False
        if t_port == opt.port_id and t_start < a_end and a_start < t_end:
            return False
        shift_spans.setdefault(t_port, []).append((t_start, t_end))
    occupied.setdefault(opt.port_id, []).append((a_start, a_end))
    return True


def build_action_library(
    world: WorldState, candidates: List[Taxi], config: SimulationConfig
) -> List[Bundle]:
    """Build the ``action_library`` for one negotiation (``05-01``) — full-assignment.

    FIXED MECHANISM (2026-07-08): bundles are greedy joint packings, not
    cartesian samples with optional UNASSIGNED choices. Steps:

    1. Per-taxi feasible options -> distinct placements (nearest port first).
    2. For each port-preference **rotation variant** (deterministic, up to
       :data:`_MAX_PLACEMENT_VARIANTS`): greedily place every candidate —
       most critical SoC first, then longest queue — onto its first
       non-conflicting placement. A candidate stays UNASSIGNED **only** when no
       placement fits (physically unplaceable in that packing); bundles never
       choose to leave an assignable taxi idle.
    3. Each packing is emitted once per **uniform price level**, so the menu
       spans the whole allowed price range (the negotiation's ZOPA).
    4. Feasibility re-check (safety net) -> dedupe -> sort by **fleet** utility
       -> K_max cut. There is still no all-UNASSIGNED bundle ("no deal" only
       ever results from running out of negotiation steps).

    Args:
        world: The frozen ground-truth world.
        candidates: The candidate taxis (all idle, stuck, port-reachable taxis;
            see :func:`simulation.bridge.snapshot.collect_charge_candidates`).
        config: The simulation config (price levels, K_max, ufun shape).

    Returns:
        The ordered list of feasible :class:`Bundle`. May be empty when no
        candidate can be placed (the caller then skips the negotiation).

    Example:
        >>> from simulation.config.default import default_config
        >>> from simulation.world.routing import Router
        >>> cfg = default_config()
        >>> w = WorldState(config=cfg, router=Router(speed_kmh=30.0))
        >>> build_action_library(w, [], cfg)
        []
    """

    _validate_library_dims(config)

    if not candidates:
        return []

    prices = sorted(set(float(p) for p in config.negotiation.price_levels))
    if not prices:
        prices = [0.0]

    placements: Dict[str, List[_Option]] = {
        taxi.id: _distinct_placements(_options_for_taxi(taxi, world, config))
        for taxi in candidates
    }
    if all(len(p) == 0 for p in placements.values()):
        return []

    # Packing priority: most critical first (lowest SoC, then longest queue),
    # deterministic tie-break by id. The bundle tuple itself keeps the original
    # ``candidates`` order (index-position contract of Bundle.assignments).
    ordered = sorted(
        candidates,
        key=lambda t: (
            float(t.soc),
            -int(getattr(t, "cycles_waited", 0) or 0),
            str(t.id),
        ),
    )

    max_ports = max(len(p) for p in placements.values())
    n_variants = max(1, min(max_ports, _MAX_PLACEMENT_VARIANTS))

    # Varian simulasi_mahal_kompen (opt-in, PER-ASSIGNMENT): sebuah penugasan yang
    # MENGGESER booking (shift != None) tak boleh di bawah base price port-nya —
    # shift sudah membebani SPKLU kompensasi, jadi harga lantai untuk penugasan itu
    # di-prune (dinaikkan ke base). Penugasan LAIN di bundle yang sama (tanpa shift)
    # tetap boleh harga lantai — jadi satu bundle bisa campur harga.
    prune_shift = bool(getattr(config.negotiation, "prune_floor_price_on_shift", False))

    def _assignment_price(opt: _Option, price: float) -> float:
        if prune_shift and opt.shift is not None:
            port_o = world.get_port(opt.port_id)
            base = float(getattr(port_o, "base_price_per_kwh", 0.0)) if port_o else 0.0
            if price < base:
                return base
        return price

    seen: set = set()
    feasible: List[Bundle] = []
    for variant in range(n_variants):
        occupied: Dict[str, List[Tuple[int, int]]] = {}
        shift_spans: Dict[str, List[Tuple[int, int]]] = {}
        placed: Dict[str, _Option] = {}
        for taxi in ordered:
            opts = placements[taxi.id]
            if not opts:
                continue
            k = variant % len(opts)
            for opt in opts[k:] + opts[:k]:
                if _try_place(opt, occupied, shift_spans):
                    placed[taxi.id] = opt
                    break
        if not placed:
            continue
        for price in prices:
            assignments = tuple(
                Assignment(
                    taxi_id=taxi.id,
                    port_id=placed[taxi.id].port_id,
                    slot_start=placed[taxi.id].slot_start,
                    duration_grids=placed[taxi.id].duration_grids,
                    price_per_kwh=_assignment_price(placed[taxi.id], price),
                    shift=placed[taxi.id].shift,
                )
                if taxi.id in placed
                else _unassigned(taxi.id)
                for taxi in candidates
            )
            bundle = Bundle(assignments=assignments)
            if bundle.is_all_unassigned():
                continue  # never an explicit bundle (that means "no deal").
            if bundle in seen:
                continue
            if not _bundle_is_feasible(bundle):
                continue
            seen.add(bundle)
            feasible.append(bundle)

    if not feasible:
        return []

    k_max = max(1, int(config.negotiation.max_action_library))
    # SELALU urutkan descending menurut utility fleet (bukan hanya saat memotong
    # ke K_max): dengan begitu indeks bundle = PERINGKAT preferensi fleet yang
    # konsisten lintas-negosiasi, sehingga opponent model berbasis indeks
    # (IOP MBOM lintas-negosiasi, FrequencyOpponentModel persisten DLST) belajar
    # sinyal peringkat yang stabil alih-alih menghafal indeks enumerasi yang
    # isinya berganti-ganti tiap periode negosiasi.
    # Deferred import avoids a circular import (ufun_builder imports this
    # module's Bundle/Assignment).
    from simulation.bridge.ufun_builder import build_ufun_fleet

    ufun_fleet = build_ufun_fleet(world)
    feasible.sort(key=lambda b: float(ufun_fleet(b)), reverse=True)
    if len(feasible) > k_max:
        feasible = feasible[:k_max]
    return feasible
