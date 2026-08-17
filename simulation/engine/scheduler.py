"""Main tick loop tying world + bridge + agents + training together (``05-03``).

:class:`Scheduler` is the **engine glue** that drives one simulation run. Per
``.claude/project_instruction/05-bridge-dan-negosiasi/05-03-alur-tick-negosiasi.md``
each tick it:

1. advances the :class:`~simulation.engine.clock.Clock` (unless frozen);
2. moves every taxi via :func:`~simulation.world.state_machine.step_taxi`
   (SoC drains **only** while moving — Gotcha G9);
3. on an hour rollover refreshes orders via
   :meth:`~simulation.world.order_generator.OrderGenerator.refresh_hour` (``01-03``);
4. matches PENDING orders to idle taxis (greedy nearest, ``lock`` / ``steal`` per
   ``order_claim_mode``) and repositions stuck ``IDLE_WAITING`` taxis toward the
   nearest reachable marking area (``TO_MARKING_AREA``, ``01-04``). ``RESERVED``
   taxis are left alone — they wait in place for their departure time.

On the ``interval_min`` cadence it runs the negotiation cycle: first
:func:`~simulation.bridge.emergency.handle_emergency_taxis` (stuck idle taxis that
can reach no port become STRANDED / rescued, ``05-05``), then — when charge
candidates exist — it **freezes** simulated time, builds the domain spec + runs one
negotiation through the bridge, applies the deal / no-deal / partial outcome, then
**resumes** time. Every ``train_every_n_episodes`` negotiation episodes it triggers
on-the-fly training.

Dependency boundary (Tahap H is built AFTER this module): the scheduler never
hard-imports the metrics collector, registry, or training orchestrator. Both seams
are **injected** and default to a no-op:

* ``metrics_collector`` — any object with a ``record(event, **payload)`` method (or a
  plain callable); wired onto ``world.metrics_hook`` so the state machine / bridge
  emit into it (Gotcha G4 / Contract C5). Defaults to a no-op.
* ``training_callback`` — ``callable(scheduler) -> None`` invoked when the episode
  counter crosses ``train_every_n_episodes``. Defaults to
  :meth:`Scheduler._default_train_on_the_fly`, which calls each agent's own per-phase
  trainers (``propose_trainer`` / ``respond_trainer`` / DLST ``threshold_trainer`` +
  any MBOM IOP trainers) ``.update()`` then ``.reset_episodes()``. The Tahap H
  orchestrator can be injected later to control concurrency (SIMULTANEOUS /
  ALTERNATING / FIXED_OPPONENT) without rewriting the scheduler.
* ``negotiation_recorder`` — ``callable(agreement) -> None`` invoked once after each
  negotiation resolves (deal or no-deal). The orchestrator owns the agents / episodes,
  so it emits the per-side ``negotiation_utility`` + ``negotiation_payload`` events
  (``06-03``) from each agent's just-finalized payloads; the scheduler only forwards
  the run's agreement outcome. Defaults to a no-op.

Dashboard contract C8: the scheduler exposes :meth:`step_tick` (alias
:meth:`step_once`), :meth:`run`, the :attr:`finished` property, and
:meth:`world_snapshot` (a JSON-serializable dict per ``07-02``).

Config and agents are fixed for the whole run (Gotcha G14).

This module is allowed to import :mod:`simulation.world`, :mod:`simulation.bridge`,
:mod:`simulation.engine.clock`, and :mod:`simulation.config` (it is engine glue);
agents are supplied already-built by the caller (or built via
``negotiation.agents.agent_factory`` upstream).

Example
-------
>>> # See the module's __main__ verification block for an end-to-end run.
>>> Scheduler.__name__
'Scheduler'
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from simulation.config.config import SimulationConfig
from simulation.engine.clock import Clock
from simulation.world.entities import (
    Order,
    OrderState,
    Taxi,
    TaxiState,
)
from simulation.world.idle_logic import (
    SoCZone,
    nearest_reachable_area,
    soc_zone,
)
from simulation.world.order_generator import OrderGenerator
from simulation.world.state_machine import step_taxi
from simulation.world.world_state import WorldState

from simulation.bridge.apply import apply_deal, handle_no_deal, handle_partial
from simulation.bridge.domain_builder import build_domain_spec
from simulation.bridge.emergency import handle_emergency_taxis
from simulation.bridge.negotiation_runner import run_negotiation
from simulation.bridge.snapshot import collect_charge_candidates

from simulation.metrics.episode_record import (
    REASON_NO_CANDIDATES,
    REASON_NO_FEASIBLE_BUNDLE,
    STATUS_DEAL,
    STATUS_NO_DEAL,
)
from simulation.metrics.loss_history import (
    PHASE_PROPOSE,
    PHASE_RESPOND,
    PHASE_THRESHOLD,
)

from negotiation.trace_logger import get_logger as _get_trace_logger

# Half-hour grid width in minutes (structural constant, see ``01-08``).
_GRID_MINUTES: int = 30
# Order-claim modes (``order_claim_mode``).
_CLAIM_LOCK: str = "lock"
_CLAIM_STEAL: str = "steal"
# Idle taxi states eligible for order matching each tick.
_IDLE_STATES: Tuple[TaxiState, ...] = (TaxiState.IDLE_IN_AREA, TaxiState.IDLE_WAITING)

# Side labels for the loss-per-phase wire format (must match the metrics layer).
_SIDE_FLEET: str = "fleet"
_SIDE_SPKLU: str = "spklu"

# Phase keys used by the per-side episode payload builders (separated propose /
# respond / DLST threshold memories — INV-2).
_PHASE_KEY_PROPOSE: str = "propose"
_PHASE_KEY_RESPOND: str = "respond"
_PHASE_KEY_THRESHOLD: str = "utility_threshold"

# Map an agent's ``trainers`` role key to a loss phase name (mirror of the Tahap H
# orchestrator's ``_TRAINER_ROLE_TO_PHASE``; small duplication keeps the scheduler
# dependency-light — LLR-M3-F). Only the scalar per-phase trainers are tracked;
# the list-valued MBOM/IOP roles (``iop_trainers_*``) have no entry (yield None).
_TRAINER_ROLE_TO_PHASE: Dict[str, str] = {
    "propose_trainer": PHASE_PROPOSE,
    "respond_trainer": PHASE_RESPOND,
    "threshold_trainer": PHASE_THRESHOLD,
}

# Priority order of loss keys read from a ``trainer.update()`` result (mirror of
# the orchestrator's ``_LOSS_KEY_PRIORITY``). No new loss is computed — the scalar
# is only *read* from what the trainer already returns/exposes (INV-3).
_LOSS_KEY_PRIORITY: Tuple[str, ...] = (
    "loss",
    "total_loss",
    "policy_loss",
    "actor_loss",
    "value_loss",
)


# =============================================================================
# Injected metrics seam (no-op default — Tahap H supplies the real collector)
# =============================================================================

def _noop_metrics_hook(event: str, **payload: Any) -> None:
    """Default metrics hook: ignore the event (Gotcha G4).

    Args:
        event: Metric event name.
        **payload: Arbitrary event payload (ignored).

    Example:
        >>> _noop_metrics_hook("tick", n=1) is None
        True
    """

    return None


def _noop_negotiation_recorder(_agreement: Any) -> None:
    """Default per-negotiation recorder: ignore the agreement (Tahap H seam).

    Args:
        _agreement: The negotiation agreement outcome (ignored).

    Example:
        >>> _noop_negotiation_recorder((0,)) is None
        True
    """

    return None


def _collector_to_hook(collector: Any) -> Callable[..., None]:
    """Adapt an injected metrics collector into a ``metrics_hook`` callable.

    Accepts either a plain callable ``hook(event, **payload)`` or an object with a
    ``record(event, **payload)`` method (the Tahap H ``MetricsCollector`` shape).
    A ``None`` collector yields the no-op hook.

    Args:
        collector: ``None``, a callable, or an object exposing ``record``.

    Returns:
        A ``hook(event, **payload) -> None`` callable.

    Example:
        >>> calls = []
        >>> class C:
        ...     def record(self, event, **payload):
        ...         calls.append((event, payload))
        >>> hook = _collector_to_hook(C())
        >>> hook("deal_applied", taxi_id="t0")
        >>> calls
        [('deal_applied', {'taxi_id': 't0'})]
    """

    if collector is None:
        return _noop_metrics_hook
    record = getattr(collector, "record", None)
    if callable(record):
        return record  # type: ignore[return-value]
    if callable(collector):
        return collector  # type: ignore[return-value]
    return _noop_metrics_hook


# =============================================================================
# Scheduler
# =============================================================================

class Scheduler:
    """Main per-tick simulation loop (world + bridge + agents + training).

    The scheduler owns simulated time (via the :class:`Clock`), physical taxi
    evolution (via :func:`step_taxi`), order matching / idle routing, the
    negotiation trigger cadence (freeze -> negotiate -> apply -> resume), and the
    on-the-fly training trigger. It mutates the live :class:`WorldState`; the bridge
    performs every negotiation-related world write.

    Parameters
    ----------
    world:
        The live ground-truth :class:`WorldState` (already populated with taxis,
        stations, areas, bookings). Its ``metrics_hook`` is replaced with the
        injected collector's hook.
    agent_fleet, agent_spklu:
        The already-built fleet / SPKLU negotiators (Contract C3 wiring happens
        inside the bridge ``run_negotiation``). Fixed for the whole run (G14).
    config:
        The immutable :class:`SimulationConfig` (defaults to ``world.config``).
    order_generator:
        Optional :class:`OrderGenerator`; one is created over ``world`` if omitted.
    metrics_collector:
        Optional metrics sink (callable or ``.record``-style object); wired onto
        ``world.metrics_hook`` (Gotcha G4 / Contract C5). Defaults to a no-op.
    training_callback:
        Optional ``callable(scheduler) -> None`` run every
        ``train_every_n_episodes`` episodes. Defaults to the per-agent on-the-fly
        trainer update (:meth:`_default_train_on_the_fly`).
    negotiation_recorder:
        Optional ``callable(agreement) -> None`` invoked once after every
        negotiation resolves (deal or no-deal). The Tahap H orchestrator owns the
        agents / episodes, so it (not the scheduler) emits the per-side
        ``negotiation_utility`` + ``negotiation_payload`` metric events from each
        agent's just-finalized payloads. Wired to
        ``orchestrator.record_negotiation_payloads`` in ``main.py``; defaults to a
        no-op (the negotiation-group utility/payload metrics stay empty).
    run_reporter:
        Optional :class:`~simulation.metrics.run_reporter.RunReporter` (the runs/
        per-period JSON + per-day/per-sim PNG sink — LLR-M3). When provided, the
        scheduler reports every negotiation period (deal / no-deal), every
        no-negotiation period (with the correct REASON), captures the day-boundary
        snapshot + regenerates the day PNGs on each day rollover, and regenerates
        the per-sim PNGs when the run finishes. ``None`` makes every reporter call
        a safe no-op (INV-6). The ``RunPaths`` / ``RunReporter`` are built in
        ``main.py`` / ``sim_session.py`` and injected here (KR-1: applies to both
        the CLI and the dashboard); the scheduler never mints a ``sim_id`` itself.
    rng:
        Optional :class:`random.Random` for reproducible order matching ties /
        generation.

    Example:
        >>> # End-to-end construction + run is exercised by the __main__ block.
        >>> Scheduler.__name__
        'Scheduler'
    """

    def __init__(
        self,
        world: WorldState,
        agent_fleet: Any,
        agent_spklu: Any,
        config: Optional[SimulationConfig] = None,
        order_generator: Optional[OrderGenerator] = None,
        metrics_collector: Any = None,
        training_callback: Optional[Callable[["Scheduler"], None]] = None,
        negotiation_recorder: Optional[Callable[[Any], None]] = None,
        run_reporter: Any = None,
        rng: Optional[random.Random] = None,
        trace_log_path: Optional[str] = None,
        agent_rebuilder: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        """Initialize the scheduler and wire the world's metrics hook.

        Example:
            >>> Scheduler.__init__.__qualname__
            'Scheduler.__init__'
        """

        self.world = world
        self.config: SimulationConfig = config if config is not None else world.config
        self.agent_fleet = agent_fleet
        self.agent_spklu = agent_spklu
        # Optional per-episode agent rebuilder: returns a fresh ``{side: agent}``
        # (new shells with the persistent trainers re-attached) so no NegMAS-internal
        # agent state leaks across negotiations. ``None`` keeps the legacy behaviour
        # of reusing the same agent objects every period.
        self.agent_rebuilder: Optional[Callable[[], Dict[str, Any]]] = agent_rebuilder
        self.rng = rng if rng is not None else random.Random()

        # Contract C5 / Gotcha G4: ensure the world exposes the state-machine
        # surface and install the real (or no-op) metrics hook.
        self.metrics_collector = metrics_collector
        self.metrics_hook: Callable[..., None] = _collector_to_hook(metrics_collector)
        self.world.metrics_hook = self.metrics_hook

        self.clock = Clock(self.config.time)
        self.world.sim_time_min = self.clock.sim_time_min

        self.order_generator = (
            order_generator if order_generator is not None
            else OrderGenerator(world, rng=self.rng)
        )

        # Training seam (default to the per-agent on-the-fly trainers).
        self.training_callback: Callable[["Scheduler"], None] = (
            training_callback if training_callback is not None
            else self._default_train_on_the_fly
        )

        # Per-negotiation recorder seam (Tahap H orchestrator emits the per-side
        # negotiation_utility / negotiation_payload metric events). No-op default.
        self.negotiation_recorder: Callable[[Any], None] = (
            negotiation_recorder if negotiation_recorder is not None
            else _noop_negotiation_recorder
        )

        # Validation trace logger seam. When ``trace_log_path`` is given the global
        # TraceLogger writes a JSONL record per tick / negotiation period / step to
        # that file (named after the run's sim_id under data/logs). ``None`` leaves
        # the logger disabled so every trace call is a cheap no-op.
        self.trace_log_path = trace_log_path
        self.trace_logger = _get_trace_logger()
        if trace_log_path is not None:
            self.trace_logger.configure(trace_log_path)

        # Runs/reporting seam (LLR-M3). ``None`` == every reporter call no-ops.
        self.run_reporter = run_reporter
        # Day tracker for day-rollover detection (LLR-M3-A / M3-B).
        self._reported_day: int = int(self.clock.day)
        # Guard so ``on_run_finished`` is reported exactly once (LLR-M3-E).
        self._run_finished_reported: bool = False
        # Guard so the run-baseline (day-0) snapshot is captured once (LLR-M3-B).
        self._baseline_snapshot_captured: bool = False

        # Run bookkeeping.
        self._previous_hour: int = self.clock.hour
        self._last_negotiation_min: float = -float("inf")
        self.episode_count: int = 0
        self.negotiation_count: int = 0
        self.last_deal: Optional[Any] = None
        self.in_negotiation: bool = False
        self._hour_initialized: bool = False

        # Negotiation-interval log (one entry per interval_min cadence check).
        # Each entry records the simulation time and outcome (skipped or result).
        self.negotiation_log: List[Dict[str, Any]] = []
        self._NEGO_LOG_MAX: int = 200

    # ------------------------------------------------------------------
    # Dashboard contract C8 — finished property
    # ------------------------------------------------------------------

    @property
    def finished(self) -> bool:
        """Return whether the configured run horizon (``n_days``) is reached (C8).

        Returns:
            ``True`` when the clock has reached ``n_days`` simulated days.

        Example:
            >>> # See the __main__ block: finished flips True at horizon end.
            >>> isinstance(Scheduler.finished, property)
            True
        """

        return self.clock.finished

    # ------------------------------------------------------------------
    # Dashboard contract C8 — one-tick advance
    # ------------------------------------------------------------------

    def step_tick(self) -> bool:
        """Advance the simulation by exactly one tick and return ``finished`` (C8).

        Implements the per-tick order of ``05-03`` / ``01-02``: advance the clock,
        move every taxi, refresh orders on an hour rollover, match orders + run idle
        routing, check critical battery, then (on the ``interval_min`` cadence) run
        one negotiation behind a freeze/resume. A no-op when already finished.

        Returns:
            ``True`` if the run is finished after this tick.

        Example:
            >>> Scheduler.step_tick.__qualname__
            'Scheduler.step_tick'
        """

        if self.finished:
            return True

        # On the very first tick spawn the opening hour's orders so taxis have
        # something to serve before the first hour rollover.
        if not self._hour_initialized:
            self.order_generator.refresh_hour()
            self._previous_hour = self.clock.hour
            self._hour_initialized = True
            # Capture the run-baseline (day-0) snapshot once so the first day's
            # business-metric diff has a zero baseline (LLR-M3-B #C1). No-op when
            # there is no reporter.
            if not self._baseline_snapshot_captured:
                self._capture_day_snapshot(int(self.clock.day))
                self._baseline_snapshot_captured = True

        # (1) Advance simulated time (clock is never frozen here).
        self.clock.tick()
        self.world.sim_time_min = self.clock.sim_time_min
        dt_min = float(self.config.time.tick_resolution_min)

        # (1b) Day rollover -> capture the day-boundary snapshot (start of the new
        # day == end of the previous day) and notify each completed day so the
        # day PNGs are regenerated exactly when the day flips, even on days with no
        # negotiation (LLR-M3-B / KR-11 / #C1).
        if int(self.clock.day) > self._reported_day:
            self._capture_day_snapshot(int(self.clock.day))
            for completed_day in range(self._reported_day, int(self.clock.day)):
                self._notify_day_completed(completed_day)
            self._reported_day = int(self.clock.day)

        # (2) Move every taxi (SoC drains only while moving — G9).
        self._move_taxis(dt_min)

        # (3) Hour rollover -> refresh orders (expire stale, spawn new) (01-03).
        if self.clock.hour_changed_since(self._previous_hour):
            expired = self.order_generator.refresh_hour()
            self.metrics_hook("orders_refreshed", hour=self.clock.hour, expired=len(expired))
            self._previous_hour = self.clock.hour

        # (4) Order matching (greedy nearest, lock/steal) + idle repositioning.
        self._match_orders_and_route()

        # (5) Negotiation trigger on the interval cadence (freeze -> run -> resume).
        #     Emergency handling for stuck, port-unreachable idle taxis is performed
        #     inside the cycle, just before candidates are collected (05-05).
        if self._negotiation_interval_reached():
            self._run_negotiation_cycle()

        self.metrics_hook(
            "tick",
            sim_time_min=self.clock.sim_time_min,
            hour=self.clock.hour,
            day=self.clock.day,
        )

        # Validation trace: dump the full world state for this tick (no-op unless
        # the trace logger is configured). Emitted last so it reflects the
        # post-negotiation world; the period's events already carry their own
        # period index for re-correlation.
        if self.trace_logger.is_enabled():
            self.trace_logger.mark_tick(int(self.clock.tick_count))
            self.trace_logger.emit("tick", world=self.world_snapshot())

        # (7) Run-finished transition: regenerate the per-sim (and final-day) PNGs
        # exactly once when the horizon is reached on this tick (LLR-M3-E / KR-6).
        if self.finished:
            self._notify_run_finished()
        return self.finished

    def step_once(self) -> bool:
        """Alias of :meth:`step_tick` (Contract C8 naming).

        Example:
            >>> Scheduler.step_once.__qualname__
            'Scheduler.step_once'
        """

        return self.step_tick()

    # ------------------------------------------------------------------
    # Run to completion
    # ------------------------------------------------------------------

    def run(self, max_ticks: Optional[int] = None) -> None:
        """Run ticks until the horizon is reached (or ``max_ticks`` is hit).

        Args:
            max_ticks: Optional safety cap on the number of ticks (mainly for
                tests); ``None`` runs until :attr:`finished`.

        Example:
            >>> Scheduler.run.__qualname__
            'Scheduler.run'
        """

        ticks = 0
        while not self.finished:
            self.step_tick()
            ticks += 1
            if max_ticks is not None and ticks >= max_ticks:
                break
        # Safety net for the run-finished PNG regen (idempotent via the guard flag,
        # LLR-M3-E): covers the natural-horizon path; the ``max_ticks`` early-out
        # leaves it un-reported on purpose (the run is not actually finished).
        if self.finished:
            self._notify_run_finished()

    # ------------------------------------------------------------------
    # (2) Move taxis
    # ------------------------------------------------------------------

    def _move_taxis(self, dt_min: float) -> None:
        """Advance every taxi by ``dt_min`` minutes via the state machine (G9).

        Args:
            dt_min: Simulated minutes advanced this tick.

        Example:
            >>> Scheduler._move_taxis.__qualname__
            'Scheduler._move_taxis'
        """

        for taxi in list(self.world.taxis.values()):
            step_taxi(taxi, self.world, dt_min)

    # ------------------------------------------------------------------
    # (4) Order matching + idle repositioning to nearest reachable area (01-03 / 01-04)
    # ------------------------------------------------------------------

    def _match_orders_and_route(self) -> None:
        """Match orders to idle taxis, then reposition stuck idle taxis to an area.

        Order matching is greedy-nearest per order, respecting ``order_claim_mode``
        (``lock`` claims a specific order before departing; ``steal`` only sends the
        taxi toward the area and re-matches on arrival). RESERVED taxis are **not**
        touched here — they wait in place for their departure time (only taxis that
        satisfy the area battery constraint ever head to an area). Stuck
        ``IDLE_WAITING`` taxis are repositioned toward the nearest reachable area
        (``TO_MARKING_AREA``) when they can, else left waiting as charge candidates.

        Example:
            >>> Scheduler._match_orders_and_route.__qualname__
            'Scheduler._match_orders_and_route'
        """

        # (a) Greedy-nearest order matching for idle taxis (per area).
        self._match_orders_greedy()

        # (b) Reposition stuck IDLE_WAITING taxis toward the nearest reachable area
        #     (IDLE_IN_AREA taxis stay in their area; RESERVED/charging are untouched).
        for taxi in list(self.world.taxis.values()):
            if taxi.state is TaxiState.IDLE_WAITING and taxi.current_order is None:
                self._route_idle_taxi(taxi)

    def _match_orders_greedy(self) -> None:
        """Assign each PENDING order to the nearest eligible idle taxi (``01-03``).

        Eligibility: an idle taxi with ``SoC >= B_area`` that can reach the pickup
        (taxis below ``B_area`` are stuck / charge candidates and are excluded). In
        ``lock`` mode the order is CLAIMED and the taxi commits to that specific
        pickup; in ``steal`` mode the taxi is only pointed at the area
        (``TO_MARKING_AREA``) and re-matched on arrival, so PENDING orders are left
        for the on-arrival match.

        Example:
            >>> Scheduler._match_orders_greedy.__qualname__
            'Scheduler._match_orders_greedy'
        """

        claim_mode = self.config.marking_area_order.order_claim_mode
        assigned_taxi_ids: set = set()

        for area in self.world.areas.values():
            pending = [o for o in area.current_orders if o.state is OrderState.PENDING]
            for order in pending:
                taxi = self._nearest_eligible_taxi(order, assigned_taxi_ids)
                if taxi is None:
                    continue
                if claim_mode == _CLAIM_STEAL:
                    # Steal: only head toward the area; claim on arrival.
                    taxi.state = TaxiState.TO_MARKING_AREA
                    taxi.current_area = None
                    taxi.move_target = area.center
                    assigned_taxi_ids.add(taxi.id)
                    continue
                # Lock: claim this specific order and commit to the pickup.
                order.state = OrderState.CLAIMED
                order.assigned_taxi = taxi.id
                taxi.current_order = order
                taxi.current_area = None
                taxi.state = TaxiState.TO_PASSENGER
                taxi.move_target = order.pickup_point
                assigned_taxi_ids.add(taxi.id)
                self.metrics_hook("order_claimed", taxi_id=taxi.id, order_id=order.id)

    def _nearest_eligible_taxi(
        self, order: Order, assigned_taxi_ids: set
    ) -> Optional[Taxi]:
        """Return the nearest safe-zone idle taxi able to reach an order's pickup.

        Args:
            order: The PENDING order to match.
            assigned_taxi_ids: Taxi ids already matched this tick (skipped).

        Returns:
            The nearest eligible :class:`Taxi`, or ``None`` if none qualifies.

        Example:
            >>> Scheduler._nearest_eligible_taxi.__qualname__
            'Scheduler._nearest_eligible_taxi'
        """

        battery = self.config.battery
        spec = self.config.taxi_spec
        best: Optional[Taxi] = None
        best_distance = float("inf")
        for taxi in self.world.taxis.values():
            if taxi.id in assigned_taxi_ids:
                continue
            if taxi.state not in _IDLE_STATES or taxi.current_order is not None:
                continue
            # Only taxis healthy enough (SoC >= B_area) chase orders; taxis below
            # B_area are stuck and wait to charge (negotiation candidates).
            if soc_zone(taxi.soc, battery) != SoCZone.SAFE.value:
                continue
            distance_km = self.world.router.distance(taxi.position, order.pickup_point)
            if spec.battery_capacity_kwh > 0.0:
                soc_drop = (distance_km * spec.consumption_kwh_per_km) / spec.battery_capacity_kwh
            else:
                soc_drop = 0.0
            if taxi.soc - soc_drop < 0.0:
                continue
            if distance_km < best_distance:
                best = taxi
                best_distance = distance_km
        return best

    def _route_idle_taxi(self, taxi: Taxi) -> None:
        """Reposition one stuck ``IDLE_WAITING`` taxi toward the nearest reachable area.

        If the taxi can reach the nearest marking area and still arrive with
        ``SoC >= B_area`` it switches to ``TO_MARKING_AREA`` and heads there to keep
        serving orders (so it does not linger as a charge candidate); otherwise it
        stays ``IDLE_WAITING`` in place (``move_target = None``) and will be a charge
        candidate (or, if no port is reachable, an emergency) at the next
        negotiation. There is no "head toward the nearest station" behaviour (the
        ``port_radius`` "SPKLU area" was removed).

        Args:
            taxi: The stuck, unmatched ``IDLE_WAITING`` taxi to route.

        Example:
            >>> Scheduler._route_idle_taxi.__qualname__
            'Scheduler._route_idle_taxi'
        """

        reachable = nearest_reachable_area(taxi, self.world)
        if reachable is not None:
            area, _distance = reachable
            taxi.state = TaxiState.TO_MARKING_AREA
            taxi.current_area = None
            taxi.move_target = area.center
        else:
            taxi.state = TaxiState.IDLE_WAITING
            taxi.move_target = None

    # ------------------------------------------------------------------
    # (6) Negotiation cycle (freeze -> build -> run -> apply -> resume)
    # ------------------------------------------------------------------

    def _negotiation_interval_reached(self) -> bool:
        """Return whether ``interval_min`` has elapsed since the last negotiation.

        Returns:
            ``True`` when the simulated clock has advanced by at least
            ``interval_min`` minutes since the previous negotiation check.

        Example:
            >>> Scheduler._negotiation_interval_reached.__qualname__
            'Scheduler._negotiation_interval_reached'
        """

        interval = float(self.config.time.interval_min)
        if self.clock.sim_time_min - self._last_negotiation_min >= interval:
            self._last_negotiation_min = self.clock.sim_time_min
            return True
        return False

    def _run_negotiation_cycle(self) -> None:
        """Collect candidates, freeze, negotiate, apply outcome, resume (``05-03``).

        Mirrors the ``05-03`` blueprint: first handle emergencies (stuck idle
        taxis that can reach no port); collect every remaining charge candidate;
        if none, skip with no freeze; otherwise freeze, snapshot via the bridge
        domain builder over all candidates (no cap), and — if the action library
        is non-empty — run the SAO negotiation, then apply the deal (+ partial
        aging of UNASSIGNED taxis) or the no-deal aging, collect the episode,
        maybe train, and resume.

        Each cadence check (including skipped ones) appends one entry to
        :attr:`negotiation_log` for the dashboard log panel.

        Example:
            >>> Scheduler._run_negotiation_cycle.__qualname__
            'Scheduler._run_negotiation_cycle'
        """

        # Capture simulation time for the log before any freeze.
        _log_sim_min = float(self.clock.sim_time_min)
        _log_day = int(self.clock.day)
        _log_hour = int(self.clock.hour)
        _log_minute = int(_log_sim_min % 60)

        # Emergency handling (05-05): stuck idle taxis that can reach no port at
        # all are marked STRANDED / rescued before candidates are collected, so
        # they never appear as candidates or in the action library.
        handle_emergency_taxis(self.world, self.config.dead_battery.dead_battery_mode)

        candidates = collect_charge_candidates(self.world)
        if not candidates:
            # No charge candidate at all (LLR-M3-D condition 1): report the
            # no-negotiation period with REASON_NO_CANDIDATES, then keep the legacy
            # dashboard log entry (enriched with the reason for UI consistency).
            self._report_no_negotiation(REASON_NO_CANDIDATES, 0)
            self._append_nego_log(
                sim_time_min=_log_sim_min,
                day=_log_day,
                hour=_log_hour,
                minute=_log_minute,
                event="skipped_no_candidates",
                reason=REASON_NO_CANDIDATES,
            )
            return

        # ── FREEZE simulated time around the whole negotiation (01-02 / 05-03). ──
        self.clock.freeze()
        self.in_negotiation = True
        try:
            # No cap and no urgency ranking: every collected candidate participates.
            capped = candidates
            domain_spec, _dims_fleet, _dims_spklu = build_domain_spec(
                self.world, self.config, capped
            )
            if not domain_spec.action_library:
                # Candidates exist but none can be placed into any feasible bundle
                # (LLR-M3-D condition 2): report with REASON_NO_FEASIBLE_BUNDLE and
                # candidates_count = len(candidates). Reporting happens before the
                # ``return`` inside the ``try`` so the ``finally`` still resumes the
                # clock.
                self._report_no_negotiation(
                    REASON_NO_FEASIBLE_BUNDLE, int(len(candidates))
                )
                self._append_nego_log(
                    sim_time_min=_log_sim_min,
                    day=_log_day,
                    hour=_log_hour,
                    minute=_log_minute,
                    event="skipped_no_candidates",
                    reason=REASON_NO_FEASIBLE_BUNDLE,
                )
                return

            # Validation trace: tag the upcoming per-step records with this
            # period's index and dump the full domain spec (action library, both
            # ufuns over the library, bounds, raw obs / candidate feature blocks,
            # masks, candidate ids) before the agents act.
            if self.trace_logger.is_enabled():
                self.trace_logger.mark_period(int(self.negotiation_count))
                self.trace_logger.emit(
                    "negotiation_start",
                    sim_time_min=_log_sim_min,
                    day=_log_day,
                    hour=_log_hour,
                    minute=_log_minute,
                    domain_spec=self._serialize_domain_spec(domain_spec),
                )

            # One negotiation == one episode: rebuild a fresh agent shell per side
            # (persistent trainers re-attached) so no NegMAS-internal per-mechanism
            # state (e.g. the SAO proposal cache) leaks across periods. No-op when
            # no rebuilder is configured (legacy reuse behaviour). The recorder and
            # training seam read these same refs (kept in sync with orchestrator).
            if self.agent_rebuilder is not None:
                fresh = self.agent_rebuilder()
                self.agent_fleet = fresh["fleet"]
                self.agent_spklu = fresh["spklu"]

            deal, summary = run_negotiation(
                domain_spec, self.agent_fleet, self.agent_spklu, self.config
            )
            self.negotiation_count += 1
            self.last_deal = deal

            if self.trace_logger.is_enabled():
                self.trace_logger.emit(
                    "negotiation_end",
                    deal=deal is not None,
                    agreement=(summary or {}).get("agreement"),
                    deal_index=(summary or {}).get("deal_index"),
                    deal_bundle=self._bundle_info(deal) if deal is not None else None,
                    turn_order=(summary or {}).get("turn_order"),
                )

            if deal is not None:
                apply_deal(self.world, deal)
                # UNASSIGNED taxis inside the accepted bundle are aged (G7).
                handle_partial(self.world, deal)
            else:
                # No-deal: every participating capped candidate ages (G7).
                handle_no_deal(self.world, capped)

            # Build and record the log entry for this negotiation.
            _log_entry: Dict[str, Any] = {
                "event": "negotiation_completed",
                "candidates_count": int(len(capped)),
                "deal": deal is not None,
            }
            if deal is not None:
                _assignments = getattr(deal, "assignments", ())
                _log_entry["assignments"] = [
                    {
                        "taxi_id": str(a.taxi_id),
                        "port_id": str(a.port_id),
                        "slot_start": int(a.slot_start),
                        "unassigned": bool(a.is_unassigned()),
                    }
                    for a in _assignments
                ]
            self._append_nego_log(
                sim_time_min=_log_sim_min,
                day=_log_day,
                hour=_log_hour,
                minute=_log_minute,
                **_log_entry,
            )

            self.metrics_hook(
                "negotiation_completed",
                deal=deal is not None,
                candidate_taxis=len(capped),
                **{k: v for k, v in (summary or {}).items() if _is_scalar(v)},
            )

            # Per-side realized utility + episode payloads (06-03). Each side
            # already finalized its propose/respond(/threshold) memories inside
            # on_negotiation_end (run_negotiation); the recorder (the Tahap H
            # orchestrator) re-reads those builders with the run's raw agreement
            # outcome ((idx,) for a deal, None for no-deal — the agents' own
            # disagreement convention sets terminal_utility to the no-deal value)
            # and emits negotiation_utility / negotiation_payload. The agreement
            # comes straight from the run summary so the deal/no-deal convention
            # matches what the mechanism reported.
            agreement = (summary or {}).get("agreement")
            self.negotiation_recorder(agreement)

            # Report this negotiation period (deal / no-deal) to the runs reporter
            # in REAL TIME (LLR-M3-C / KR-3,5,12). The action_library lives on the
            # just-built domain_spec; payloads + final ufun are re-read from each
            # agent's separated propose/respond/threshold builders (INV-2). No-op
            # when there is no reporter.
            status = STATUS_DEAL if deal is not None else STATUS_NO_DEAL
            self._report_period(
                day_index=_log_day,
                sim_time_min=_log_sim_min,
                hour=_log_hour,
                minute=_log_minute,
                status=status,
                action_library=getattr(domain_spec, "action_library", ()),
                summary=summary or {},
                deal=deal,
                capped=capped,
            )

            # One negotiation == one episode (each side stored its payloads on
            # on_negotiation_end inside run_negotiation).
            self.episode_count += 1
            self._maybe_train_on_the_fly()
        finally:
            # ── RESUME simulated time (apply/no-deal done). ──
            self.in_negotiation = False
            self.clock.resume()

    # ------------------------------------------------------------------
    # Negotiation-interval log helpers
    # ------------------------------------------------------------------

    def _append_nego_log(self, **fields: Any) -> None:
        """Append one negotiation-interval entry to :attr:`negotiation_log`.

        Entries are capped at :attr:`_NEGO_LOG_MAX` (200) to prevent unbounded
        growth. Older entries are dropped from the front when the cap is hit.

        Args:
            **fields: Keyword fields stored in the log entry dict.  Expected
                keys: ``sim_time_min``, ``day``, ``hour``, ``minute``,
                ``event`` (``"skipped_no_candidates"`` or
                ``"negotiation_completed"``), an optional ``reason`` (one of the
                ``REASON_*`` codes for skipped intervals — added for dashboard
                consistency without removing the legacy behavior), plus optional
                ``candidates_count``, ``deal``, and ``assignments`` for completed
                negotiations.

        Example:
            >>> Scheduler._append_nego_log.__qualname__
            'Scheduler._append_nego_log'
        """

        self.negotiation_log.append(dict(fields))
        if len(self.negotiation_log) > self._NEGO_LOG_MAX:
            self.negotiation_log = self.negotiation_log[-self._NEGO_LOG_MAX:]

    # ------------------------------------------------------------------
    # Runs/reporting seam helpers (LLR-M3-G — each wraps a RunReporter call in
    # try/except and no-ops when ``run_reporter`` is None — INV-6)
    # ------------------------------------------------------------------

    def _report_period(
        self,
        day_index: int,
        sim_time_min: float,
        hour: int,
        minute: int,
        status: str,
        action_library: Any,
        summary: Dict[str, Any],
        deal: Any,
        capped: Any,
    ) -> None:
        """Assemble payloads + turn_order + final ufun and call ``record_period``.

        Re-reads each side's separated propose / respond / threshold episode
        payloads via the agents' own builders (the exact same builders the Tahap H
        orchestrator uses — INV-2: it never merges the propose/respond paths and
        never mutates the agent). The per-side final utility is taken RAW from the
        propose-phase payload's ``terminal_utility`` (``U(agreement)`` for a deal,
        the agent's own ``-1.0`` no-deal marker otherwise) and forwarded WITHOUT
        any clamp (#C4 — the reporter owns the deal/no-deal/no-nego mapping). The
        whole call is wrapped in ``try``/``except`` and no-ops when there is no
        reporter (INV-6).

        Args:
            day_index: Simulated day index of the period.
            sim_time_min: Simulated minute of the period.
            hour: Simulated hour of the period.
            minute: Simulated minute-of-hour of the period.
            status: ``STATUS_DEAL`` or ``STATUS_NO_DEAL``.
            action_library: The bundles offered this negotiation
                (``domain_spec.action_library``).
            summary: The run summary dict (carries ``agreement`` + ``turn_order``).
            deal: The winning :class:`Bundle` or ``None`` (no-deal).
            capped: The capped candidate list (its length is ``candidates_count``).

        Example:
            >>> Scheduler._report_period.__qualname__
            'Scheduler._report_period'
        """

        reporter = self.run_reporter
        if reporter is None:
            return
        try:
            agreement = summary.get("agreement")
            fleet_payloads = self._collect_side_payloads(self.agent_fleet, agreement)
            spklu_payloads = self._collect_side_payloads(self.agent_spklu, agreement)
            ufun_fleet_final = _terminal_utility_of(
                fleet_payloads.get(_PHASE_KEY_PROPOSE)
            )
            ufun_spklu_final = _terminal_utility_of(
                spklu_payloads.get(_PHASE_KEY_PROPOSE)
            )
            reporter.record_period(
                day_index=int(day_index),
                sim_time_min=float(sim_time_min),
                hour=int(hour),
                minute=int(minute),
                status=str(status),
                action_library=action_library,
                turn_order=summary.get("turn_order"),
                fleet_payloads=fleet_payloads,
                spklu_payloads=spklu_payloads,
                agreement=agreement,
                deal_bundle=deal,
                ufun_fleet_final=ufun_fleet_final,
                ufun_spklu_final=ufun_spklu_final,
                candidates_count=int(len(capped)),
            )
        except Exception as error:  # INV-6: reporting never tears down the run.
            _log_reporter_failure("record period", error)

    def _collect_side_payloads(
        self, agent: Any, agreement: Any
    ) -> Dict[str, Any]:
        """Build one side's ``{propose, respond, utility_threshold}`` payload dict.

        Calls each separated per-phase builder
        (``build_episode_payload_propose`` / ``_respond`` / ``_threshold``) with the
        run's raw ``agreement`` (``(idx,)`` for a deal, ``None`` for no-deal). Each
        builder is wrapped in ``try``/``except`` -> ``None`` so a baseline agent
        (no builders) or a builder failure leaves that phase ``None`` without
        crashing the run. Reads only — the agent is never mutated (INV-2 / INV-5).

        Args:
            agent: A negotiator (or ``None``).
            agreement: The negotiation agreement outcome ``(idx,)`` / ``None``.

        Returns:
            ``{"propose": ep|None, "respond": ep|None, "utility_threshold": ep|None}``.

        Example:
            >>> Scheduler._collect_side_payloads.__qualname__
            'Scheduler._collect_side_payloads'
        """

        payloads: Dict[str, Any] = {
            _PHASE_KEY_PROPOSE: None,
            _PHASE_KEY_RESPOND: None,
            _PHASE_KEY_THRESHOLD: None,
        }
        if agent is None:
            return payloads
        for phase_key, builder_name in (
            (_PHASE_KEY_PROPOSE, "build_episode_payload_propose"),
            (_PHASE_KEY_RESPOND, "build_episode_payload_respond"),
            (_PHASE_KEY_THRESHOLD, "build_episode_payload_threshold"),
        ):
            builder = getattr(agent, builder_name, None)
            if not callable(builder):
                continue
            try:
                payloads[phase_key] = builder(agreement)
            except Exception:
                payloads[phase_key] = None
        return payloads

    def _report_no_negotiation(
        self, reason_code: str, candidates_count: int
    ) -> None:
        """Call ``record_no_negotiation`` for a skipped interval (LLR-M3-D / KR-4).

        Computes the current day / hour / minute from the clock and forwards the
        REASON code + candidate count to the reporter, wrapped in ``try``/``except``
        and no-op when there is no reporter (INV-6). The two skip conditions are
        distinguished by the caller via ``reason_code``
        (``REASON_NO_CANDIDATES`` vs ``REASON_NO_FEASIBLE_BUNDLE``).

        Args:
            reason_code: One of the ``REASON_*`` constants (LLR-A6).
            candidates_count: Number of candidate taxis (0 for no-candidates).

        Example:
            >>> Scheduler._report_no_negotiation.__qualname__
            'Scheduler._report_no_negotiation'
        """

        reporter = self.run_reporter
        if reporter is None:
            return
        sim_time_min = float(self.clock.sim_time_min)
        try:
            reporter.record_no_negotiation(
                day_index=int(self.clock.day),
                sim_time_min=sim_time_min,
                hour=int(self.clock.hour),
                minute=int(sim_time_min % 60),
                reason_code=str(reason_code),
                candidates_count=int(candidates_count),
            )
        except Exception as error:  # INV-6.
            _log_reporter_failure("record no-negotiation", error)

    def _capture_day_snapshot(self, day_index: int) -> None:
        """Wrap ``capture_day_boundary_snapshot`` (#C1) in try/except (INV-6).

        Captures the cumulative business-metric snapshot at the start of
        ``day_index`` (== end of the previous day) so ``on_day_completed`` can diff.
        No-op when there is no reporter.

        Args:
            day_index: The day whose boundary snapshot is captured.

        Example:
            >>> Scheduler._capture_day_snapshot.__qualname__
            'Scheduler._capture_day_snapshot'
        """

        reporter = self.run_reporter
        if reporter is None:
            return
        try:
            reporter.capture_day_boundary_snapshot(int(day_index))
        except Exception as error:  # INV-6.
            _log_reporter_failure("capture day snapshot", error)

    def _notify_day_completed(self, day_index: int) -> None:
        """Wrap ``on_day_completed`` (per-day PNG regen) in try/except (INV-6).

        No-op when there is no reporter.

        Args:
            day_index: The completed day index.

        Example:
            >>> Scheduler._notify_day_completed.__qualname__
            'Scheduler._notify_day_completed'
        """

        reporter = self.run_reporter
        if reporter is None:
            return
        try:
            reporter.on_day_completed(int(day_index))
        except Exception as error:  # INV-6.
            _log_reporter_failure("on day completed", error)

    def _notify_run_finished(self) -> None:
        """Wrap ``on_run_finished`` (per-sim PNG regen) in try/except (LLR-M3-E).

        Reported exactly once via :attr:`_run_finished_reported`; subsequent calls
        no-op (idempotent). Also a no-op when there is no reporter (INV-6).

        Example:
            >>> Scheduler._notify_run_finished.__qualname__
            'Scheduler._notify_run_finished'
        """

        if self._run_finished_reported:
            return
        reporter = self.run_reporter
        if reporter is None:
            self._run_finished_reported = True
            # Trace logging is independent of the reporter — still close the file.
            self.trace_logger.close()
            return
        # Make sure the final (still-open) day's PNGs are regenerated too: capture
        # its boundary snapshot then notify it completed before the per-sim regen
        # (RunReporter also has its own safety net — LLR-M3-E).
        self._capture_day_snapshot(int(self.clock.day))
        self._notify_day_completed(int(self.clock.day))
        try:
            reporter.on_run_finished()
        except Exception as error:  # INV-6.
            _log_reporter_failure("on run finished", error)
        finally:
            self._run_finished_reported = True
            # Flush + close the validation trace file (idempotent no-op if off).
            self.trace_logger.close()

    # ------------------------------------------------------------------
    # On-the-fly training seam (Tahap H pluggable)
    # ------------------------------------------------------------------

    def _maybe_train_on_the_fly(self) -> None:
        """Trigger the training callback every ``train_every_n_episodes`` episodes.

        Respects ``training.on_the_fly`` (skips when offline). The actual update is
        delegated to :attr:`training_callback` (default per-agent trainers; the
        Tahap H orchestrator may be injected to control concurrency).

        Example:
            >>> Scheduler._maybe_train_on_the_fly.__qualname__
            'Scheduler._maybe_train_on_the_fly'
        """

        if not self.config.training.on_the_fly:
            return
        every = max(1, int(self.config.training.train_every_n_episodes))
        if self.episode_count % every != 0:
            return
        self.training_callback(self)

    def _default_train_on_the_fly(self, _scheduler: "Scheduler") -> None:
        """Default training: update each agent's own per-phase trainers (``06-01``).

        For each side, every attached trainer (``propose_trainer`` /
        ``respond_trainer`` / DLST ``threshold_trainer`` + any MBOM IOP trainers)
        runs ``update()`` then ``reset_episodes()``. This is the on-the-fly path the
        Tahap H orchestrator will later wrap to apply a concurrency policy
        (SIMULTANEOUS / ALTERNATING / FIXED_OPPONENT). Trainers with no stored
        episodes simply no-op.

        The per-phase scalar loss reported by each ``update()`` is CAPTURED (read,
        never recomputed — INV-3) and forwarded with the ``training_step`` metric
        event as ``losses=[(side, phase, loss), ...]`` (LLR-M3-F / KR-8), so that
        the collector fills the loss history even when no orchestrator is injected
        (e.g. the dashboard ``sim_session`` path). The training cadence stays under
        :meth:`_maybe_train_on_the_fly` (``config.training.train_every_n_episodes``,
        KR-10) — it is never hardcoded here.

        Args:
            _scheduler: This scheduler (the callback signature; unused here).

        Example:
            >>> Scheduler._default_train_on_the_fly.__qualname__
            'Scheduler._default_train_on_the_fly'
        """

        losses: List[Tuple[str, str, float]] = []
        for side, agent in (
            (_SIDE_FLEET, self.agent_fleet),
            (_SIDE_SPKLU, self.agent_spklu),
        ):
            for trainer in _iter_trainers(agent):
                ran, loss = _safe_update_capture(trainer)
                if not ran:
                    continue
                phase = _phase_of_trainer(agent, trainer)
                if phase is not None and loss is not None:
                    losses.append((side, phase, float(loss)))
        self.metrics_hook(
            "training_step", episode=self.episode_count, losses=losses
        )

    # ------------------------------------------------------------------
    # Dashboard contract C8 — world snapshot (JSON-serializable)
    # ------------------------------------------------------------------

    def world_snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serializable snapshot of the world for the dashboard (C8).

        The schema follows ``07-02``: simulated hour/day + freeze/negotiation flags;
        a taxi list (id, position, state, soc, cycles_waited, optional order/
        reservation); the station/port list with allocation occupancy; the marking
        areas with order intensity; the emergency station if active; and a small
        summary metrics block. It also carries the active run identity (``sim_id``,
        its ``run_id`` alias, and ``run_dir``) from the injected reporter so the
        dashboard can query the on-disk period records (LLR-M3-H; identity only, not
        period data — K-2). Every value is a JSON-native scalar / list / dict.

        Returns:
            A nested ``dict`` of plain Python types ready for ``json.dumps``.

        Example:
            >>> Scheduler.world_snapshot.__qualname__
            'Scheduler.world_snapshot'
        """

        world = self.world
        snapshot: Dict[str, Any] = {
            "sim_time_min": float(self.clock.sim_time_min),
            "hour": int(self.clock.hour),
            "day": int(self.clock.day),
            "grid": int(world.current_grid()),
            "frozen": bool(self.clock.frozen),
            "in_negotiation": bool(self.in_negotiation),
            "finished": bool(self.finished),
            "taxis": [self._taxi_snapshot(t) for t in world.taxis.values()],
            "stations": [self._station_snapshot(s) for s in world.stations.values()],
            "areas": [self._area_snapshot(a) for a in world.areas.values()],
            "bookings": [self._booking_snapshot(b) for b in world.bookings.values()],
            "metrics": self._summary_metrics(),
            "negotiation_log": list(self.negotiation_log),
        }
        if world.emergency_station is not None:
            snapshot["emergency_station"] = self._station_snapshot(world.emergency_station)
        if self.last_deal is not None:
            snapshot["last_deal"] = self._deal_snapshot(self.last_deal)
        # LLR-M3-H: expose the active run identity (sim_id / run_id alias + run_dir)
        # so the dashboard frontend knows which run to query on disk (K-2: identity
        # only, never period data). ``None`` when no reporter is wired.
        reporter = self.run_reporter
        snapshot["sim_id"] = reporter.sim_id if reporter is not None else None
        snapshot["run_id"] = snapshot["sim_id"]
        snapshot["run_dir"] = reporter.run_dir if reporter is not None else None
        return snapshot

    @staticmethod
    def _taxi_snapshot(taxi: Taxi) -> Dict[str, Any]:
        """Serialize one taxi (id, position, state, soc, counters) for the snapshot.

        Example:
            >>> Scheduler._taxi_snapshot.__qualname__
            'Scheduler._taxi_snapshot'
        """

        data: Dict[str, Any] = {
            "id": taxi.id,
            "lat": float(taxi.position[0]),
            "lng": float(taxi.position[1]),
            "state": taxi.state.value,
            "soc": float(taxi.soc),
            "cycles_waited": int(taxi.cycles_waited),
            "productive_km": float(taxi.productive_km),
            "unproductive_km": float(taxi.unproductive_km),
        }
        if taxi.move_target is not None:
            data["move_target"] = [float(taxi.move_target[0]), float(taxi.move_target[1])]
        if taxi.current_order is not None:
            data["order_id"] = taxi.current_order.id
        if taxi.reservation is not None:
            data["reservation"] = {
                "port_id": taxi.reservation.port_id,
                "slot_start": int(taxi.reservation.slot_start),
                "duration_grids": int(taxi.reservation.duration_grids),
                "price_per_kwh": float(taxi.reservation.price_per_kwh),
                "departure_time_min": float(taxi.reservation.departure_time_min),
            }
        return data

    @staticmethod
    def _station_snapshot(station: Any) -> Dict[str, Any]:
        """Serialize one charging station and its ports' occupancy.

        Example:
            >>> Scheduler._station_snapshot.__qualname__
            'Scheduler._station_snapshot'
        """

        return {
            "id": station.id,
            "lat": float(station.location[0]),
            "lng": float(station.location[1]),
            "is_emergency": bool(station.is_emergency),
            "ports": [
                {
                    "id": port.id,
                    "power_kw": float(port.power_kw),
                    "base_price_per_kwh": float(port.base_price_per_kwh),
                    "allocations": [
                        {
                            "slot_start": int(slot_start),
                            "duration_grids": int(duration),
                            "holder": str(holder),
                            "kind": str(kind),
                        }
                        for slot_start, duration, holder, kind in port.allocations
                    ],
                    "busy": len(port.allocations) > 0,
                }
                for port in station.ports
            ],
        }

    @staticmethod
    def _area_snapshot(area: Any) -> Dict[str, Any]:
        """Serialize one marking area + its order intensity.

        Example:
            >>> Scheduler._area_snapshot.__qualname__
            'Scheduler._area_snapshot'
        """

        pending = sum(1 for o in area.current_orders if o.state is OrderState.PENDING)
        return {
            "id": area.id,
            "lat": float(area.center[0]),
            "lng": float(area.center[1]),
            "order_count": len(area.current_orders),
            "pending_orders": int(pending),
        }

    @staticmethod
    def _booking_snapshot(booking: Any) -> Dict[str, Any]:
        """Serialize one app-user booking incl. its fallback plan target.

        Surfaces, for the dashboard: which port+slot an app-user booking occupies,
        whether it is shiftable, and — if so — where its fallback plan allows it
        to move (alternative ports and slot shift) plus the compensation. Lets the
        UI show "this port has a booking and its fallback points to those ports".

        Example:
            >>> Scheduler._booking_snapshot.__qualname__
            'Scheduler._booking_snapshot'
        """

        data: Dict[str, Any] = {
            "id": booking.id,
            "port_id": booking.port_id,
            "slot_start": int(booking.slot_start),
            "duration_grids": int(booking.duration_grids),
            "is_shiftable": bool(booking.is_shiftable),
            "compensation": float(booking.compensation),
        }
        fb = booking.fallback_plan
        if fb is not None:
            data["fallback"] = {
                "to_port_id": fb.to_port_id,
                "to_slot_start": int(fb.to_slot_start),
                "compensation": float(fb.compensation),
            }
        return data

    @staticmethod
    def _deal_snapshot(deal: Any) -> Dict[str, Any]:
        """Serialize the last accepted bundle (assignments) for the dashboard.

        Example:
            >>> Scheduler._deal_snapshot.__qualname__
            'Scheduler._deal_snapshot'
        """

        assignments = getattr(deal, "assignments", ())
        return {
            "assignments": [
                {
                    "taxi_id": a.taxi_id,
                    "port_id": a.port_id,
                    "slot_start": int(a.slot_start),
                    "duration_grids": int(a.duration_grids),
                    "price_per_kwh": float(a.price_per_kwh),
                    "unassigned": bool(a.is_unassigned()),
                }
                for a in assignments
            ]
        }

    @staticmethod
    def _bundle_info(bundle: Any) -> Dict[str, Any]:
        """Serialize one action-library :class:`Bundle` (its assignments).

        Mirrors :meth:`_deal_snapshot` for a single bundle so the validation trace
        can show exactly which (taxi, port, slot, price) tuples each offered bundle
        carries, plus whether it is the all-UNASSIGNED bundle.

        Example:
            >>> Scheduler._bundle_info.__qualname__
            'Scheduler._bundle_info'
        """

        def _shift_info(shift: Any) -> Optional[Dict[str, Any]]:
            if shift is None:
                return None
            return {
                "booking_id": shift.booking_id,
                "from_port_id": shift.from_port_id,
                "to_port_id": shift.to_port_id,
                "from_slot_start": int(shift.from_slot_start),
                "to_slot_start": int(shift.to_slot_start),
                "duration_grids": int(shift.duration_grids),
                "compensation": float(shift.compensation),
            }

        assignments = getattr(bundle, "assignments", ())
        info: Dict[str, Any] = {
            "assignments": [
                {
                    "taxi_id": a.taxi_id,
                    "port_id": a.port_id,
                    "slot_start": int(a.slot_start),
                    "duration_grids": int(a.duration_grids),
                    "price_per_kwh": float(a.price_per_kwh),
                    "unassigned": bool(a.is_unassigned()),
                    # Booking shift this assignment requires (None if it takes an
                    # empty slot); lets the trace audit which bundles move a booking.
                    "shift": _shift_info(getattr(a, "shift", None)),
                }
                for a in assignments
            ],
        }
        is_all = getattr(bundle, "is_all_unassigned", None)
        if callable(is_all):
            info["is_all_unassigned"] = bool(is_all())
        return info

    def _serialize_domain_spec(self, domain_spec: Any) -> Dict[str, Any]:
        """Serialize a negotiation :class:`DomainSpec` for the validation trace.

        Captures the complete per-negotiation domain: the action library (each
        bundle's assignments), both sides' world utilities evaluated over every
        bundle, the reward-scaling bounds, the per-side raw observation /
        candidate-feature blocks + masks fed to the agents, and the candidate taxi
        ids. Every value is run through the trace logger's JSON sanitizer
        downstream, so numpy arrays / tensors serialize transparently.

        Example:
            >>> Scheduler._serialize_domain_spec.__qualname__
            'Scheduler._serialize_domain_spec'
        """

        library = list(getattr(domain_spec, "action_library", []) or [])
        meta = getattr(domain_spec, "meta", {}) or {}

        def _ufun_over_library(ufun: Any) -> List[Any]:
            values: List[Any] = []
            for bundle in library:
                try:
                    values.append(float(ufun(bundle)))
                except Exception:
                    values.append(None)
            return values

        return {
            "domain_name": getattr(domain_spec, "domain_name", None),
            "k_size": len(library),
            "candidate_taxi_ids": meta.get("candidate_taxi_ids"),
            "action_library": [self._bundle_info(b) for b in library],
            "ufun_fleet_over_library": _ufun_over_library(
                getattr(domain_spec, "ufun_fleet", None)
            ),
            "ufun_spklu_over_library": _ufun_over_library(
                getattr(domain_spec, "ufun_spklu", None)
            ),
            "utility_bounds": {
                "fleet": [
                    getattr(domain_spec, "utility_min_fleet", None),
                    getattr(domain_spec, "utility_max_fleet", None),
                ],
                "spklu": [
                    getattr(domain_spec, "utility_min_spklu", None),
                    getattr(domain_spec, "utility_max_spklu", None),
                ],
            },
            "raw_obs_fleet": getattr(domain_spec, "raw_obs_fleet", None),
            "raw_obs_spklu": getattr(domain_spec, "raw_obs_spklu", None),
            "cand_feats_fleet": getattr(domain_spec, "cand_feats_fleet", None),
            "cand_feats_spklu": getattr(domain_spec, "cand_feats_spklu", None),
            "cand_mask_fleet": meta.get("cand_mask_fleet"),
            "cand_mask_spklu": meta.get("cand_mask_spklu"),
        }

    def _summary_metrics(self) -> Dict[str, Any]:
        """Return a small JSON-serializable summary metrics block.

        Prefers a ``summary()`` (or ``snapshot()``) from the injected collector when
        available; otherwise reports the scheduler's own episode / negotiation
        counters and a live taxi-state tally so the dashboard always has data.

        Example:
            >>> Scheduler._summary_metrics.__qualname__
            'Scheduler._summary_metrics'
        """

        collector = self.metrics_collector
        for attr in ("summary", "snapshot", "as_dict"):
            method = getattr(collector, attr, None)
            if callable(method):
                try:
                    result = method()
                    if isinstance(result, dict):
                        return result
                except Exception:
                    break
        state_counts: Dict[str, int] = {}
        for taxi in self.world.taxis.values():
            state_counts[taxi.state.value] = state_counts.get(taxi.state.value, 0) + 1
        return {
            "episodes": int(self.episode_count),
            "negotiations": int(self.negotiation_count),
            "taxi_states": state_counts,
        }


# =============================================================================
# Module-level training helpers (kept tiny + side-effect-isolated)
# =============================================================================

def _iter_trainers(agent: Any) -> List[Any]:
    """Return every distinct trainer attached to an agent (per-phase + IOP).

    Reads ``agent.trainers`` (the ``{role: trainer}`` mapping built by the factory),
    flattening list-valued roles (the MBOM ``iop_trainers_*`` lists). Agents with no
    trainers (e.g. RandomIOP / Rule baselines) yield an empty list.

    Args:
        agent: A negotiator (or ``None``).

    Returns:
        A de-duplicated list of trainer objects exposing ``update`` /
        ``reset_episodes``.

    Example:
        >>> class A: trainers = {"propose_trainer": 1, "respond_trainer": 2}
        >>> sorted(_iter_trainers(A()))
        [1, 2]
    """

    if agent is None:
        return []
    trainers_map = getattr(agent, "trainers", None)
    if not trainers_map:
        return []
    out: List[Any] = []
    seen: set = set()
    for value in trainers_map.values():
        items = value if isinstance(value, (list, tuple)) else [value]
        for trainer in items:
            if trainer is None:
                continue
            ident = id(trainer)
            if ident in seen:
                continue
            seen.add(ident)
            out.append(trainer)
    return out


def _safe_update_and_reset(trainer: Any) -> None:
    """Run ``trainer.update()`` then ``trainer.reset_episodes()`` defensively.

    A trainer with no stored episodes (nothing happened this window) is a no-op; any
    update error is swallowed so a single trainer cannot crash the whole run (the
    metrics hook still records the training step upstream).

    Args:
        trainer: A trainer exposing ``update`` / ``reset_episodes``.

    Example:
        >>> calls = []
        >>> class T:
        ...     episodes = ["e"]
        ...     def update(self): calls.append("u"); return {}
        ...     def reset_episodes(self): calls.append("r")
        >>> _safe_update_and_reset(T())
        >>> calls
        ['u', 'r']
    """

    episodes = getattr(trainer, "episodes", None)
    if not episodes:
        return
    update = getattr(trainer, "update", None)
    reset = getattr(trainer, "reset_episodes", None)
    if callable(update):
        try:
            update()
        except Exception:
            pass
    if callable(reset):
        try:
            reset()
        except Exception:
            pass


def _is_scalar(value: Any) -> bool:
    """Return whether a value is a JSON-native scalar (for metric payloads).

    Example:
        >>> _is_scalar(1), _is_scalar("x"), _is_scalar([1])
        (True, True, False)
    """

    return isinstance(value, (int, float, str, bool)) or value is None


def _log_reporter_failure(action: str, error: Exception) -> None:
    """Print a one-line warning when a RunReporter call fails (INV-6).

    A reporting/IO failure must never tear down the simulation run; the caller
    already swallows the exception, this only surfaces a breadcrumb for debugging.

    Args:
        action: Short description of the reporter action that failed.
        error: The caught exception.

    Example:
        >>> _log_reporter_failure("record period", ValueError("x")) is None
        [scheduler] run-reporter 'record period' failed: ValueError('x')
        True
    """

    print(f"[scheduler] run-reporter '{action}' failed: {error!r}")
    return None


def _terminal_utility_of(episode: Any) -> Optional[float]:
    """Return an episode payload's RAW ``terminal_utility`` (or ``None``) (#C4).

    Reads the agent's own ``terminal_utility`` — ``U(agreement)`` for a deal, the
    agent's ``-1.0`` no-deal marker otherwise — and returns it verbatim. NO clamp
    is applied here (the reporter owns the deal/no-deal/no-nego mapping). ``None``
    is returned when the payload is absent or exposes no usable utility.

    Args:
        episode: An episode payload (or ``None``).

    Returns:
        The raw final utility as a ``float``, or ``None``.

    Example:
        >>> class E: terminal_utility = -1.0
        >>> _terminal_utility_of(E())
        -1.0
        >>> _terminal_utility_of(None) is None
        True
    """

    if episode is None:
        return None
    utility = getattr(episode, "terminal_utility", None)
    if utility is None:
        return None
    try:
        return float(utility)
    except (TypeError, ValueError):
        return None


def _is_finite_number(value: Any) -> bool:
    """Return ``True`` when ``value`` is a finite real number (not None/NaN/inf).

    Example:
        >>> _is_finite_number(0.5), _is_finite_number(None), _is_finite_number(float("nan"))
        (True, False, False)
    """

    if value is None or isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and number not in (float("inf"), float("-inf"))


def _extract_loss(result: Any, trainer: Any) -> Optional[float]:
    """Extract one scalar loss from a ``trainer.update()`` result (mirror of M4.1).

    The loss is only *read* from what the trainer already returns/exposes — no new
    loss is computed (INV-3). Resolution order: a ``dict`` -> the first finite key
    in :data:`_LOSS_KEY_PRIORITY`; a finite ``int``/``float`` -> used directly;
    otherwise a finite ``trainer.last_loss``; else ``None`` (a valid empty series).

    Args:
        result: The value returned by ``trainer.update()``.
        trainer: The trainer (for the ``last_loss`` fallback).

    Returns:
        The scalar loss as a ``float``, or ``None``.

    Example:
        >>> class T: pass
        >>> _extract_loss({"loss": 0.5, "steps": 4}, T())
        0.5
        >>> _extract_loss(0.7, T())
        0.7
        >>> _extract_loss({"steps": 0}, T()) is None
        True
    """

    if isinstance(result, dict):
        for key in _LOSS_KEY_PRIORITY:
            if key in result and _is_finite_number(result[key]):
                return float(result[key])
    elif _is_finite_number(result):
        return float(result)
    last_loss = getattr(trainer, "last_loss", None)
    if _is_finite_number(last_loss):
        return float(last_loss)
    return None


def _safe_update_capture(trainer: Any) -> Tuple[bool, Optional[float]]:
    """Run ``update()`` + ``reset_episodes()`` and capture the loss (LLR-M3-F).

    Mirrors the orchestrator's ``_update_and_capture`` (small, deliberate
    duplication to keep the scheduler dependency-light, matching the as-is style
    that already duplicates ``_iter_trainers``): updates only when the trainer has
    stored episodes, then resets, returning ``(ran, loss_or_None)``. The whole
    update is wrapped in ``try``/``except`` so a faulty trainer never tears down the
    run. No loss formula is touched (INV-3).

    Args:
        trainer: A trainer exposing ``episodes`` / ``update`` / ``reset_episodes``.

    Returns:
        ``(ran, loss_or_None)`` — ``ran`` is ``True`` only when an update executed.

    Example:
        >>> calls = []
        >>> class T:
        ...     episodes = ["e"]
        ...     def update(self): calls.append("u"); return {"loss": 0.5}
        ...     def reset_episodes(self): calls.append("r")
        >>> _safe_update_capture(T()), calls
        ((True, 0.5), ['u', 'r'])
        >>> class Empty: episodes = []
        >>> _safe_update_capture(Empty())
        (False, None)
    """

    episodes = getattr(trainer, "episodes", None)
    if not episodes:
        return (False, None)
    update = getattr(trainer, "update", None)
    reset = getattr(trainer, "reset_episodes", None)
    try:
        result = update() if callable(update) else None
        loss = _extract_loss(result, trainer)
        if callable(reset):
            reset()
        return (True, loss)
    except Exception:
        return (False, None)


def _phase_of_trainer(agent: Any, trainer: Any) -> Optional[str]:
    """Return a trainer's loss phase ("propose"/"respond"/"utility_threshold").

    Resolves the phase by matching ``trainer`` against the agent's ``trainers``
    role mapping (mirror of the orchestrator's helper, LLR-M3-F). List-valued
    MBOM/IOP roles (``iop_trainers_*``) are NOT mapped to a per-phase loss and
    return ``None`` (they still update, just are not recorded to loss-per-phase).

    Args:
        agent: The owning agent (or ``None``).
        trainer: The trainer object to locate (or ``None``).

    Returns:
        The phase name, or ``None`` when it cannot be resolved.

    Example:
        >>> class A:
        ...     trainers = {"propose_trainer": "P", "respond_trainer": "R"}
        >>> _phase_of_trainer(A(), "P")
        'propose'
        >>> _phase_of_trainer(A(), "X") is None
        True
    """

    if agent is None or trainer is None:
        return None
    trainers_map = getattr(agent, "trainers", None)
    if not trainers_map:
        return None
    for role, value in trainers_map.items():
        if isinstance(value, (list, tuple)):
            continue
        if value is trainer:
            return _TRAINER_ROLE_TO_PHASE.get(role)
    return None
