"""Emergency handling for stuck, unreachable idle taxis (``05-05``).

:func:`handle_emergency_taxis` runs once per negotiation cycle (just before
candidates are collected). A taxi is an emergency case when it is

* ``IDLE_WAITING``,
* **stuck** — it cannot reach any marking area while keeping ``SoC >= B_area``
  (so it can no longer serve orders), and
* unable to reach **any** regular (non-emergency) port at all.

In other words, emergency is no longer triggered by a fixed ``critical_soc``
threshold; it is triggered when an idle, waiting taxi can be placed nowhere — it
would never appear in any action library because no port is reachable.

Two policies (``dead_battery_mode``):

* ``LET_DIE`` (sikap A) — the taxi is marked ``STRANDED`` and recorded as a failure
  (``taxis_died``). No rescue.
* ``HARD_FLOOR`` (sikap B) — the taxi is rescued by the emergency station: it is
  **teleported** there spending **no travel energy** (it could not move anyway),
  set ``EMERGENCY_CHARGING`` until full, and billed the premium
  ``emergency_price_per_kwh``. Recorded as ``emergency_rescue_count`` plus the
  premium cost.

The emergency station is special (Gotcha G8): unlimited capacity, premium price,
and **never** a negotiation candidate / **never** in the ``action_library`` (it is
already excluded from ``world.all_ports``); it is invisible to negotiation.

The physical charge progress is owned by the state machine
(``_step_emergency_charging``); this module only triggers the rescue / death and
records the premium-cost metric.

Example
-------
>>> hasattr(handle_emergency_taxis, "__call__")
True
"""

from __future__ import annotations

from typing import Optional

from simulation.world.entities import ChargingStation, Port, Taxi, TaxiState
from simulation.world.idle_logic import can_reach_any_port, nearest_reachable_area
from simulation.world.world_state import WorldState

# Policy names (``dead_battery_mode``), kept as constants (no magic strings).
MODE_LET_DIE: str = "LET_DIE"
MODE_HARD_FLOOR: str = "HARD_FLOOR"


def handle_emergency_taxis(world: WorldState, mode: Optional[str] = None) -> None:
    """Handle every stuck idle taxi that can reach no port (``05-05``).

    For each ``IDLE_WAITING`` taxi that is stuck (cannot reposition to any area
    above ``B_area``) **and** can reach no regular port, applies the configured
    policy (mark STRANDED, or teleport to the emergency station and start
    EMERGENCY_CHARGING). Taxis that can still serve orders or still reach a port
    are left untouched (the latter become negotiation candidates instead).

    Args:
        world: The live world (mutated in place).
        mode: Override of ``dead_battery_mode`` (``"LET_DIE"`` / ``"HARD_FLOOR"``);
            defaults to ``world.config.dead_battery.dead_battery_mode``.

    Example:
        >>> from simulation.config.default import default_config
        >>> from simulation.world.routing import Router
        >>> from simulation.world.entities import Taxi, TaxiState
        >>> cfg = default_config()
        >>> w = WorldState(config=cfg, router=Router(speed_kmh=cfg.taxi_spec.speed_kmh))
        >>> # No stations and no areas on this bare world -> the idle taxi is
        >>> # stuck and can reach no port, so it is handled as an emergency.
        >>> t = Taxi(id="t0", position=(-6.2, 106.8), soc=0.02,
        ...          state=TaxiState.IDLE_WAITING)
        >>> w.taxis = {"t0": t}
        >>> handle_emergency_taxis(w, mode="LET_DIE")
        >>> t.state is TaxiState.STRANDED
        True
    """

    resolved_mode = mode if mode is not None else world.config.dead_battery.dead_battery_mode

    for taxi in world.taxis.values():
        if taxi.state is not TaxiState.IDLE_WAITING:
            continue
        # A taxi that can still (re)position to an area above B_area is productive,
        # never an emergency.
        if nearest_reachable_area(taxi, world) is not None:
            continue
        # Stuck but can reach a port -> it becomes a negotiation candidate.
        if can_reach_any_port(taxi, world):
            continue
        # Stuck and can reach no port at all -> emergency.
        if resolved_mode == MODE_HARD_FLOOR:
            _rescue(world, taxi)
        else:
            _let_die(world, taxi)


def _let_die(world: WorldState, taxi: Taxi) -> None:
    """Mark a critical taxi STRANDED and record the failure (sikap A)."""
    taxi.state = TaxiState.STRANDED
    taxi.move_target = None
    taxi.current_order = None
    world.metrics_hook("taxi_died", taxi_id=taxi.id, soc=taxi.soc)


def _rescue(world: WorldState, taxi: Taxi) -> None:
    """Teleport a critical taxi to the emergency station and start charging (sikap B).

    No travel energy is spent (the taxi could not move). The premium cost of
    filling to ``SoC_target`` is recorded; the physical charge runs in the state
    machine (``_step_emergency_charging``). When no emergency station is
    configured the taxi falls back to STRANDED.
    """
    station = _ensure_emergency_station(world)
    if station is None or not station.ports:
        _let_die(world, taxi)
        return
    port = station.ports[0]

    spec = world.config.taxi_spec
    soc_target = world.config.battery.SoC_target
    energy_to_fill_kwh = max(0.0, spec.battery_capacity_kwh * (soc_target - taxi.soc))
    premium_cost = energy_to_fill_kwh * world.config.emergency.emergency_price_per_kwh

    # Teleport (no energy), attach to the emergency port, start charging.
    taxi.position = station.location
    taxi.current_area = None
    taxi.move_target = None
    taxi.current_order = None
    taxi.reservation = None
    taxi.state = TaxiState.EMERGENCY_CHARGING
    # No slot bound for emergency charging (runs to full, see ``05-05``); the
    # state machine clears charge_until_min when full.
    taxi.charge_until_min = None

    world.metrics_hook(
        "emergency_rescue",
        taxi_id=taxi.id,
        port_id=port.id,
        energy_kwh=energy_to_fill_kwh,
        premium_cost=premium_cost,
        price_per_kwh=world.config.emergency.emergency_price_per_kwh,
    )


def _ensure_emergency_station(world: WorldState) -> Optional[ChargingStation]:
    """Return the emergency station, creating it from config if absent (G8).

    The emergency station has a single unlimited-capacity port at the configured
    location and ``is_emergency = True``; it is never part of ``world.all_ports``
    or the action library.
    """
    if world.emergency_station is not None:
        return world.emergency_station
    emergency_cfg = world.config.emergency
    location = emergency_cfg.emergency_station_location
    # Charge speed: use the configured emergency power when set; otherwise derive a
    # representative power_kw from the regular ports, falling back to a positive
    # default so charging always makes progress (see ``05-05`` / config F).
    power_kw = emergency_cfg.emergency_power_kw
    if power_kw is None:
        power_kw = 50.0
        regular_ports = world.all_ports()
        if regular_ports:
            power_kw = max(p.power_kw for p in regular_ports)
    station = ChargingStation(
        id="emergency_station",
        location=location,
        ports=[
            Port(
                id="emergency_station_port",
                station_id="emergency_station",
                power_kw=power_kw,
                base_price_per_kwh=emergency_cfg.emergency_price_per_kwh,
            )
        ],
        is_emergency=True,
    )
    world.emergency_station = station
    world._rebuild_port_index()
    return station
