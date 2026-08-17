"""Intrinsic world utility functions for fleet and SPKLU (``04-01``..``04-04``).

This module builds the two world-side utility callables ``Bundle -> float`` the
bridge attaches to a negotiation:

* :func:`build_ufun_fleet` — ``U_fleet(bundle) = sum_i v_i`` (``04-02``): a
  readiness reward minus normalized charge cost, travel energy, detour, and
  opportunity cost for assigned taxis; a saturating ``cycles_waited`` penalty for
  UNASSIGNED taxis.
* :func:`build_ufun_spklu` — ``U_spklu(bundle) = sum q`` (``04-03``): normalized
  revenue plus port-hours utilization minus normalized shift compensation; pure
  utilization, never looking at taxis.

Both use the **intrinsic** ``REF_*`` normalization (each raw term divided by a
fixed configured reference, ``04-04``) — NOT min/max over the action library.
That second normalization (``utility_min``/``utility_max``) is computed separately
in :mod:`simulation.bridge.domain_builder` and is used only to scale RL reward
(Kontrak C6 / Gotcha G10). The two must never be conflated.

The fleet ufun reads taxi SoC / position and area order info; the SPKLU ufun reads
only ports, slots, prices, energy, and booking shifts — consistent with the
asymmetric :class:`~simulation.world.world_state.AgentView` visibility (``04-03``).
Utilities are deterministic over the frozen snapshot (``04-04``).

Example
-------
>>> from simulation.config.default import default_config
>>> from simulation.world.routing import Router
>>> from simulation.world.world_state import WorldState
>>> cfg = default_config()
>>> w = WorldState(config=cfg, router=Router(speed_kmh=30.0))
>>> ufun = build_ufun_spklu(w)
>>> from simulation.bridge.action_library import Bundle
>>> ufun(Bundle(assignments=())) == 0.0
True
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Tuple

from simulation.config.config import SimulationConfig
from simulation.world.entities import MarkingArea, Taxi
from simulation.world.idle_logic import reachable_area_with_orders
from simulation.world.world_state import WorldState
from simulation.bridge.action_library import Assignment, Bundle

# Half-hour grid width in minutes / hours (structural constant, see ``01-08``).
_GRID_MINUTES: int = 30
_GRID_HOURS: float = _GRID_MINUTES / 60.0
_MINUTES_PER_DAY: int = 24 * 60


# =============================================================================
# Shared helpers
# =============================================================================

def _slot_hour(slot_start: int) -> int:
    """Return the hour-of-day a half-hour grid index falls in.

    Example:
        >>> _slot_hour(0)
        0
        >>> _slot_hour(2)
        1
    """
    minute_of_day = (slot_start * _GRID_MINUTES) % _MINUTES_PER_DAY
    return minute_of_day // 60


def _time_weight_at(config: SimulationConfig, hour: int) -> float:
    """Return the configured ``time_weight`` for an hour (1.0 if absent)."""
    return float(config.marking_area_order.time_weight.get(hour, 1.0))


def _energy_delivered_kwh(power_kw: float, duration_grids: int) -> float:
    """Return kWh delivered over ``duration_grids`` at ``power_kw`` (``04-02``)."""
    return power_kw * _GRID_HOURS * duration_grids


def _order_rate_norm(
    area: Optional[MarkingArea],
    config: SimulationConfig,
    slot_start: int,
    world: WorldState,
) -> float:
    """Return an area's order-rate normalized into ``[0, 1]`` (``04-01``/``04-02``).

    Anti-peeking branch on the slot's hour (``04-01`` lines 27-31, ``04-02`` lines
    38-39): orders are drawn randomly only at the start of each hour, so the ufun
    must never read a *future* hour's realized count (it does not exist yet).

    * **Current hour** (``_slot_hour(slot_start) == world.hour()``) — use the
      realized unfilled-order count, scaled by ``order_max``.
    * **Future hour** — use the *expectation* derived from the ``time_weight``
      table instead of the realized random draw.

    Expectation derivation (``01-03`` order-generation model): the per-area count
    is drawn in ``[order_min, order_max]`` biased upward by ``time_weight``; the
    bridge models the expected count for a future hour as
    ``time_weight(hour) * order_max`` (the same proxy the feature builder uses for
    ``order_rate_slot``). Normalizing by ``order_max`` then gives
    ``order_rate_norm = clip(time_weight(hour), 0, 1)``.

    When there is no target area the rate is 0.
    """
    if area is None:
        return 0.0
    order_max = max(1, int(config.marking_area_order.order_max))
    slot_hour = _slot_hour(slot_start)
    if slot_hour == world.hour():
        count = len(area.unfilled_orders())
        return min(1.0, count / float(order_max))
    # Future hour: expected count = time_weight(hour) * order_max, normalized by
    # order_max => clip(time_weight, 0, 1). Never read the unrealized future draw.
    expected_count = _time_weight_at(config, slot_hour) * order_max
    return max(0.0, min(1.0, expected_count / float(order_max)))


def _dist_decay(distance_km: float, d0: float) -> float:
    """Return ``exp(-d / d0)`` (demand-weight distance decay, ``04-02``)."""
    if d0 <= 0.0:
        return 1.0 if distance_km <= 0.0 else 0.0
    return math.exp(-distance_km / d0)


def _g_cycles(cycles_waited: int, g_max: float, tau: float) -> float:
    """Return the saturating anti-starvation penalty ``g(c)`` (Gotcha G7).

    ``g(c) = g_max * (1 - exp(-c / tau))`` — increasing then flattening, capped
    at ``g_max``.

    Example:
        >>> _g_cycles(0, 1.0, 5.0)
        0.0
        >>> round(_g_cycles(100, 1.0, 5.0), 3)
        1.0
    """
    c = max(0, int(cycles_waited))
    if tau <= 0.0:
        return g_max if c > 0 else 0.0
    return g_max * (1.0 - math.exp(-c / tau))


# =============================================================================
# Fleet ufun (04-02)
# =============================================================================

def build_ufun_fleet(world: WorldState) -> Callable[[Bundle], float]:
    """Return the intrinsic fleet world ufun callable ``Bundle -> float`` (``04-02``).

    ``U_fleet(bundle) = sum_i v_i`` where, for an assigned taxi,

        ``v_i = w_ready * readiness_i
                - w_price  * (charge_cost_i / REF_cost)
                - w_travel * (e_travel_i    / REF_energy)
                - w_detour * (d_detour_i    / REF_dist)
                - w_oppty  * oppty_cost_i``

    and for an UNASSIGNED taxi ``v_i = - w_unassigned * g(cycles_waited_i)``.
    ``readiness_i = energy_frac_i * time_weight(slot) * demand_weight_i`` and
    ``oppty_cost_i`` is the normalized lost passenger earning while offline (0 when
    the taxi cannot even reach its target area). All distances honor
    ``ufun_distance_mode``; all raw terms use the fixed ``REF_*`` (``04-04``).

    Args:
        world: The frozen world (taxis, ports, areas, router, config).

    Returns:
        A deterministic callable mapping a :class:`Bundle` to a float utility.

    Example:
        >>> from simulation.config.default import default_config
        >>> from simulation.world.routing import Router
        >>> w = WorldState(config=default_config(), router=Router(speed_kmh=30.0))
        >>> ufun = build_ufun_fleet(w)
        >>> from simulation.bridge.action_library import Bundle
        >>> ufun(Bundle(assignments=())) == 0.0
        True
    """

    config = world.config
    weights = config.ufun.fleet_weights
    refs = config.ufun.fleet_references
    shape = config.ufun.shape
    # Per-taxi target-area cache (deterministic over the frozen snapshot).
    area_cache: Dict[str, Optional[Tuple[MarkingArea, float]]] = {}

    def _target_area(taxi: Taxi) -> Optional[Tuple[MarkingArea, float]]:
        if taxi.id not in area_cache:
            area_cache[taxi.id] = reachable_area_with_orders(taxi, world)
        return area_cache[taxi.id]

    def _assigned_value(assignment: Assignment, taxi: Taxi) -> float:
        port = world.get_port(assignment.port_id)
        if port is None:
            return 0.0
        spec = config.taxi_spec

        # Slot capacity, energy needed from current SoC, and travel energy.
        slot_capacity_kwh = _energy_delivered_kwh(port.power_kw, assignment.duration_grids)
        need = max(0.0, spec.battery_capacity_kwh * (config.battery.SoC_target - taxi.soc))
        distance_to_port = world.router.ufun_distance(
            taxi_position=taxi.position,
            taxi_area_id=taxi.current_area,
            fixed_key=port.station_id,
            area_keys=list(world.areas.keys()),
            mode=shape.ufun_distance_mode,
        )
        e_travel = distance_to_port * spec.consumption_kwh_per_km

        # Actual energy delivered = min(slot capacity, energy needed at arrival SoC).
        # action_library ceil()-rounds grids, so slot capacity >= actual need;
        # using the full slot capacity in charge_cost would overstate the cost.
        arrival_soc = (
            max(0.0, taxi.soc - e_travel / spec.battery_capacity_kwh)
            if spec.battery_capacity_kwh > 0.0 else taxi.soc
        )
        energy_to_fill = (
            max(0.0, spec.battery_capacity_kwh * (config.battery.SoC_target - arrival_soc))
            if spec.battery_capacity_kwh > 0.0 else 0.0
        )
        energy_delivered = min(slot_capacity_kwh, energy_to_fill)

        # Target area (nearest reachable area with unfilled orders) + detour.
        target = _target_area(taxi)
        if target is not None:
            area, _area_dist = target
            d_detour = world.router.fixed_distance(port.station_id, area.id)
        else:
            area = None
            d_detour = 0.0

        # readiness = energy_frac * time_weight(slot) * demand_weight.
        energy_frac = min(1.0, energy_delivered / need) if need > 0.0 else 1.0
        time_weight = _time_weight_at(config, _slot_hour(assignment.slot_start))
        demand_weight = _order_rate_norm(
            area, config, assignment.slot_start, world
        ) * _dist_decay(d_detour, shape.d0)
        readiness = energy_frac * time_weight * demand_weight

        # charge_cost = price * actual energy delivered (not full slot capacity).
        charge_cost = assignment.price_per_kwh * energy_delivered

        # opportunity cost (lost earnings while offline during the slot); 0 when
        # the taxi cannot reach the target area (it could not take orders anyway).
        oppty_cost = _opportunity_cost(area, assignment, config, target is not None)

        value = (
            weights.w_ready * readiness
            - weights.w_price * (charge_cost / _safe(refs.REF_cost))
            - weights.w_travel * (e_travel / _safe(refs.REF_energy))
            - weights.w_detour * (d_detour / _safe(refs.REF_dist))
            - weights.w_oppty * oppty_cost
        )
        return value

    def _opportunity_cost(
        area: Optional[MarkingArea],
        assignment: Assignment,
        cfg: SimulationConfig,
        reachable: bool,
    ) -> float:
        """Normalized lost passenger earning while charging the slot (``04-02``)."""
        if area is None or not reachable:
            return 0.0
        # Anti-peeking: future slots use the time_weight expectation, not the
        # current hour's realized draw (``04-01``/``04-02``).
        order_rate = _order_rate_norm(area, cfg, assignment.slot_start, world)
        slot_hours = assignment.duration_grids * _GRID_HOURS
        km_per_slot = cfg.taxi_spec.speed_kmh * slot_hours
        lost = order_rate * cfg.fare.price_per_km * km_per_slot
        return lost / _safe(refs.REF_earn)

    def ufun_fleet(bundle: Bundle) -> float:
        """Return the intrinsic fleet utility of a bundle (sum over taxis)."""
        total = 0.0
        for assignment in bundle.assignments:
            taxi = world.taxis.get(assignment.taxi_id)
            if taxi is None:
                continue
            if assignment.is_unassigned():
                total -= weights.w_unassigned * _g_cycles(
                    taxi.cycles_waited, shape.g_max, shape.tau
                )
            else:
                total += _assigned_value(assignment, taxi)
        return float(total)

    return ufun_fleet


# =============================================================================
# SPKLU ufun (04-03)
# =============================================================================

def build_ufun_spklu(world: WorldState) -> Callable[[Bundle], float]:
    """Return the intrinsic SPKLU world ufun callable ``Bundle -> float`` (``04-03``).

    ``U_spklu(bundle) = sum q`` over assigned taxis (UNASSIGNED contributes 0):

        ``q = w_rev  * (price * E / REF_rev)
              + w_util * (porthours / REF_porthours)
              - w_shift* (compensation / REF_comp)``

    where ``E = power_kw * duration``, ``porthours = duration`` (pure utilization),
    and ``compensation`` is the total Rupiah owed for any booking shifts this
    assignment requires (0 if none). The SPKLU never looks at taxis (``04-03``);
    every term is read from ports, slots, prices, energy, and shifts only.

    Args:
        world: The frozen world (ports, config).

    Returns:
        A deterministic callable mapping a :class:`Bundle` to a float utility.

    Example:
        >>> from simulation.config.default import default_config
        >>> from simulation.world.routing import Router
        >>> w = WorldState(config=default_config(), router=Router(speed_kmh=30.0))
        >>> ufun = build_ufun_spklu(w)
        >>> from simulation.bridge.action_library import Bundle
        >>> ufun(Bundle(assignments=())) == 0.0
        True
    """

    config = world.config
    weights = config.ufun.spklu_weights
    refs = config.ufun.spklu_references

    def _assignment_q(assignment: Assignment) -> float:
        port = world.get_port(assignment.port_id)
        if port is None:
            return 0.0
        energy = _energy_delivered_kwh(port.power_kw, assignment.duration_grids)
        revenue = assignment.price_per_kwh * energy
        porthours = assignment.duration_grids * _GRID_HOURS
        compensation = assignment.shift.compensation if assignment.shift is not None else 0.0
        return (
            weights.w_rev * (revenue / _safe(refs.REF_rev))
            + weights.w_util * (porthours / _safe(refs.REF_porthours))
            - weights.w_shift * (compensation / _safe(refs.REF_comp))
        )

    def ufun_spklu(bundle: Bundle) -> float:
        """Return the intrinsic SPKLU utility of a bundle (sum over assigned)."""
        total = 0.0
        for assignment in bundle.assignments:
            if assignment.is_unassigned():
                continue
            total += _assignment_q(assignment)
        return float(total)

    return ufun_spklu


def _safe(reference: float) -> float:
    """Return a non-zero reference (avoid division by zero for REF_* = 0)."""
    return reference if reference != 0.0 else 1.0
