"""Three-group metrics collector for one simulation run (``06-03``).

:class:`MetricsCollector` is the single sink the scheduler / state machine /
bridge emit into through ``world.metrics_hook`` (Gotcha G4 / Contract C5). It
exposes exactly the seam shapes the (already-built) scheduler expects:

* ``record(event, **payload)`` — the metrics-hook method; the scheduler wires
  ``world.metrics_hook = collector.record`` (see
  ``simulation/engine/scheduler.py:_collector_to_hook``);
* ``summary()`` / ``as_dict()`` / ``snapshot()`` — a JSON-serializable dict the
  scheduler folds into ``world_snapshot()["metrics"]`` for the dashboard
  (Contract C8). All three return the same structure.

The collector tallies **actual** simulation outcomes (not ufun proxies; see
``04-01``) into the three groups defined by ``06-03``:

1. **SPKLU business** — ``total_revenue``, ``total_compensation_paid``,
   ``shifted_bookings_count``, ``port_utilization`` (port-hours used, plus a
   ratio when capacity is known), ``avg_price_per_kwh``.
2. **Fleet business** — ``passenger_revenue``, ``charging_cost`` (negotiated +
   emergency), ``unproductive_km``, ``orders_served`` / ``orders_unserved``,
   ``taxis_died`` (sikap A), ``emergency_rescue_count`` +
   ``emergency_premium_cost`` (sikap B), ``slot_forfeit_count`` (loose-guard
   ``SLOT_FORFEIT`` mode, ``01-08``).
3. **Negotiation / system** — ``avg_utility_per_negotiation`` & ``total_utility``
   per side (fleet & SPKLU), ``deal_success_rate``, ``partial_deal_rate``,
   ``orders_expired_at_hour_change``, ``negotiation_payload_history`` (the
   propose/respond/threshold payloads — also feeding offline training & the
   registry), and ``avg_cycles_waited`` (+ a small distribution) for starvation.

Event vocabulary
----------------
The collector handles every event the engine actually emits today (read from
``state_machine`` / ``apply`` / ``emergency`` / ``scheduler``):

``trip_completed``, ``deal_applied``, ``booking_shifted``, ``charging_finished``,
``emergency_rescue``, ``emergency_charging_finished``, ``taxi_died``,
``no_slot_aged``, ``orders_refreshed``, ``negotiation_completed``,
``order_claimed``, ``pickup_reached``,
``charging_started``, ``departure_for_port``, ``slot_forfeit``,
``training_step``, ``tick``.

Plus two events the Tahap H orchestrator emits (it owns the agents / episodes,
which the scheduler summary does not carry): ``negotiation_utility`` (per-side
realized terminal utility of one negotiation) and ``negotiation_payload`` (one
phase's :class:`Episode` payload for the history). Unknown events are stored as
raw counts so nothing is silently lost.

This module only imports the standard library (kept dependency-free so it can be
injected anywhere). It never mutates the world or any agent (G14).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from simulation.metrics.loss_history import LossHistory


# Late-arrival mode whose forfeits this collector tallies (``01-08``).
_LATE_SLOT_FORFEIT: str = "SLOT_FORFEIT"


def _as_float(value: Any, default: float = 0.0) -> float:
    """Coerce a payload value to ``float`` (``default`` on failure).

    Example:
        >>> _as_float("1.5"), _as_float(None, 2.0)
        (1.5, 2.0)
    """

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    """Coerce a payload value to ``int`` (``default`` on failure).

    Example:
        >>> _as_int(3), _as_int("x", -1)
        (3, -1)
    """

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class MetricsCollector:
    """Tally the three ``06-03`` metric groups from emitted events.

    The collector is event-driven: every notable world/bridge/scheduler action
    calls :meth:`record` (via ``world.metrics_hook``). Aggregates are kept as
    plain Python scalars / lists so :meth:`summary` is directly JSON
    serializable.

    Parameters
    ----------
    config:
        Optional :class:`~simulation.config.config.SimulationConfig`. Only read
        once at init (G14) for the loose-guard mode and the port capacity used by
        ``port_utilization``'s ratio; ``None`` keeps the ratio unset.
    keep_payload_history:
        When ``True`` (default) per-episode payloads passed via the
        ``negotiation_payload`` event are appended to
        ``negotiation_payload_history``; set ``False`` to keep only the count
        (e.g. for a long run where the dashboard does not need the full list).

    Example:
        >>> mc = MetricsCollector()
        >>> mc.record("trip_completed", taxi_id="t0", fare=120.0,
        ...           productive_km=3.0)
        >>> mc.record("deal_applied", taxi_id="t0", port_id="p0",
        ...           revenue=900.0, energy_kwh=30.0, price_per_kwh=30.0,
        ...           port_hours=1.0, compensation=0.0, shifted_bookings=0)
        >>> s = mc.summary()
        >>> s["spklu_business"]["total_revenue"]
        900.0
        >>> s["fleet_business"]["passenger_revenue"]
        120.0
    """

    def __init__(
        self,
        config: Any = None,
        keep_payload_history: bool = True,
    ) -> None:
        """Initialize all three metric groups to empty / zero.

        Example:
            >>> MetricsCollector().summary()["negotiation_system"]["deal_success_rate"]
            0.0
        """

        self._config = config
        self._keep_payload_history = bool(keep_payload_history)

        # Capacity (total port-hours per day) for the utilization ratio, if known.
        self._port_capacity_hours: Optional[float] = self._port_capacity_hours_from_config(config)
        # Loose-guard mode governs whether slot forfeits are expected (``01-08``).
        self._late_arrival_mode: Optional[str] = self._late_arrival_mode_from_config(config)

        # ── Group 1 — SPKLU business ──────────────────────────────────────
        self.total_revenue: float = 0.0
        self.total_compensation_paid: float = 0.0
        self.shifted_bookings_count: int = 0
        self.port_hours_used: float = 0.0
        self._kwh_sold: float = 0.0  # for avg_price_per_kwh = revenue / kwh
        self._price_weighted_kwh: float = 0.0  # Σ(price*kwh) for the same ratio

        # ── Group 2 — Fleet business ──────────────────────────────────────
        self.passenger_revenue: float = 0.0
        self.charging_cost: float = 0.0
        self.unproductive_km: float = 0.0
        self.orders_served: int = 0
        self.orders_unserved: int = 0
        self.taxis_died: int = 0
        self.emergency_rescue_count: int = 0
        self.emergency_premium_cost: float = 0.0
        self.slot_forfeit_count: int = 0

        # ── Group 3 — Negotiation / system ────────────────────────────────
        self.negotiations_total: int = 0
        self.deals_total: int = 0
        self.partial_deals_total: int = 0
        self.orders_expired_at_hour_change: int = 0
        self.negotiation_payload_history: List[Any] = []
        self._payload_count: int = 0
        # Per-side realized utilities (sum + count for the averages).
        self._utility_sum: Dict[str, float] = {"fleet": 0.0, "spklu": 0.0}
        self._utility_count: Dict[str, int] = {"fleet": 0, "spklu": 0}
        # Starvation: cycles_waited samples observed at aging events.
        self._cycles_waited_sum: int = 0
        self._cycles_waited_samples: int = 0
        self._cycles_waited_hist: Dict[int, int] = {}

        # ── Training losses + per-period utility series (LLR-M5.1) ────────
        # Per-(side, phase) loss series captured on each ``training_step``
        # (recorded per ``update()``, KR-8). Filled by :meth:`_on_training_step`.
        self.loss_history: LossHistory = LossHistory()
        # Per-PERIODE realized utility (one value per negotiation period, even
        # no-deal / no-negotiation periods so utility=0 still appears — KR-7).
        # Distinct from ``_utility_sum`` (per-side-event) — populated by
        # :meth:`record_period_utilities` from the RunReporter (#C4).
        self.period_utility_series: Dict[str, List[float]] = {
            "fleet": [],
            "spklu": [],
        }

        # Raw fallthrough counts for any unmodeled event (never silently lost).
        self.event_counts: Dict[str, int] = {}

        # Dispatch table: event name -> bound handler.
        self._handlers: Dict[str, Callable[..., None]] = {
            "taxi_moved": self._on_taxi_moved,
            "trip_completed": self._on_trip_completed,
            "deal_applied": self._on_deal_applied,
            "booking_shifted": self._on_booking_shifted,
            "charging_finished": self._on_charging_finished,
            "emergency_rescue": self._on_emergency_rescue,
            "taxi_died": self._on_taxi_died,
            "no_slot_aged": self._on_no_slot_aged,
            "orders_refreshed": self._on_orders_refreshed,
            "negotiation_completed": self._on_negotiation_completed,
            "negotiation_utility": self._on_negotiation_utility,
            "negotiation_payload": self._on_negotiation_payload,
            "slot_forfeit": self._on_slot_forfeit,
            "order_unserved": self._on_order_unserved,
            "training_step": self._on_training_step,
        }

    # ------------------------------------------------------------------
    # Config helpers (read once — G14)
    # ------------------------------------------------------------------

    @staticmethod
    def _port_capacity_hours_from_config(config: Any) -> Optional[float]:
        """Return total daily port-hours capacity (Σ ports × 24 h), or ``None``.

        Example:
            >>> MetricsCollector._port_capacity_hours_from_config(None) is None
            True
        """

        if config is None:
            return None
        try:
            stations = config.port_station.stations
            n_ports = sum(len(getattr(s, "ports", [])) for s in stations)
            if n_ports <= 0:
                return None
            hours_per_day = 24.0
            return float(n_ports) * hours_per_day
        except Exception:
            return None

    @staticmethod
    def _late_arrival_mode_from_config(config: Any) -> Optional[str]:
        """Return the configured ``late_arrival_mode`` (or ``None``).

        Example:
            >>> MetricsCollector._late_arrival_mode_from_config(None) is None
            True
        """

        if config is None:
            return None
        try:
            return str(config.future_slot.late_arrival_mode)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Metrics-hook seam (Contract C5): record(event, **payload)
    # ------------------------------------------------------------------

    def record(self, event: str, **payload: Any) -> None:
        """Route one emitted metric event to its handler (the metrics hook).

        Matches the ``world.metrics_hook(event, **payload)`` calling convention
        used by the state machine / bridge / scheduler. Unknown events are just
        counted (kept in :attr:`event_counts`) so the collector never crashes a
        run and nothing is silently dropped.

        Args:
            event: The metric event name.
            **payload: Event-specific keyword payload.

        Example:
            >>> mc = MetricsCollector()
            >>> mc.record("taxi_died", taxi_id="t0", soc=0.01)
            >>> mc.summary()["fleet_business"]["taxis_died"]
            1
        """

        self.event_counts[event] = self.event_counts.get(event, 0) + 1
        handler = self._handlers.get(event)
        if handler is not None:
            handler(**payload)

    # ------------------------------------------------------------------
    # Group 1 — SPKLU business handlers
    # ------------------------------------------------------------------

    def _on_deal_applied(self, **payload: Any) -> None:
        """Accumulate negotiated revenue, energy, price, port-hours, compensation.

        Emitted by ``simulation/bridge/apply.py`` per won assignment, carrying
        ``revenue``, ``energy_kwh``, ``price_per_kwh``, ``port_hours``,
        ``compensation``, ``shifted_bookings``. The negotiated charging energy is
        also the fleet's negotiated ``charging_cost`` (price × energy = revenue
        for the SPKLU == cost for the fleet).

        Example:
            >>> mc = MetricsCollector()
            >>> mc._on_deal_applied(revenue=600.0, energy_kwh=20.0,
            ...     price_per_kwh=30.0, port_hours=1.0, compensation=50.0,
            ...     shifted_bookings=1)
            >>> mc.total_revenue, mc.charging_cost, mc.shifted_bookings_count
            (600.0, 600.0, 1)
        """

        revenue = _as_float(payload.get("revenue"))
        energy_kwh = _as_float(payload.get("energy_kwh"))
        price_per_kwh = _as_float(payload.get("price_per_kwh"))
        port_hours = _as_float(payload.get("port_hours"))
        compensation = _as_float(payload.get("compensation"))
        shifted = _as_int(payload.get("shifted_bookings"))

        self.total_revenue += revenue
        self.total_compensation_paid += compensation
        self.shifted_bookings_count += shifted
        self.port_hours_used += port_hours
        self._kwh_sold += energy_kwh
        self._price_weighted_kwh += price_per_kwh * energy_kwh
        # The fleet pays exactly what the SPKLU earns for this charge.
        self.charging_cost += revenue

    def _on_booking_shifted(self, **payload: Any) -> None:
        """Record a shifted app-user booking (count tallied via ``deal_applied``).

        The authoritative compensation / shifted count is taken from
        ``deal_applied`` (which carries both per assignment); this event is kept
        as a corroborating signal and does not double-count.

        Example:
            >>> mc = MetricsCollector()
            >>> mc._on_booking_shifted(booking_id="b0", compensation=10.0)
            >>> mc.event_counts.get("booking_shifted", 0)
            0
        """

        return None

    # ------------------------------------------------------------------
    # Group 2 — Fleet business handlers
    # ------------------------------------------------------------------

    def _on_taxi_moved(self, **payload: Any) -> None:
        """Accumulate unproductive km as taxis move (``06-03``).

        Emitted by ``state_machine._move_towards`` for every movement leg with
        ``km`` (distance travelled this tick) and ``productive`` (``True`` only for
        the passenger-carrying ``WITH_PASSENGER`` leg). Only the unproductive legs
        (idle→pickup, reposition to area, drive to port) are tallied here;
        productive km is already reflected in ``passenger_revenue`` via the fare.

        Example:
            >>> mc = MetricsCollector()
            >>> mc._on_taxi_moved(km=2.0, productive=False)
            >>> mc.unproductive_km
            2.0
            >>> mc._on_taxi_moved(km=5.0, productive=True)
            >>> mc.unproductive_km
            2.0
        """

        if not bool(payload.get("productive", False)):
            self.add_unproductive_km(payload.get("km"))

    def _on_trip_completed(self, **payload: Any) -> None:
        """Accumulate passenger revenue + an served order on trip completion.

        Emitted by ``state_machine._step_with_passenger`` with ``fare`` and
        ``productive_km``.

        Example:
            >>> mc = MetricsCollector()
            >>> mc._on_trip_completed(fare=150.0, productive_km=4.0)
            >>> mc.passenger_revenue, mc.orders_served
            (150.0, 1)
        """

        self.passenger_revenue += _as_float(payload.get("fare"))
        self.orders_served += 1

    def _on_charging_finished(self, **payload: Any) -> None:
        """Mark the end of a regular charging session (cost tallied at deal time).

        The negotiated charging cost is already counted at ``deal_applied`` (the
        whole slot is paid for on reservation); this event only confirms the
        physical session ended and is not double-counted.

        Example:
            >>> mc = MetricsCollector()
            >>> mc._on_charging_finished(taxi_id="t0", holder="t0", kind="fleet")
            >>> mc.charging_cost
            0.0
        """

        return None

    def _on_emergency_rescue(self, **payload: Any) -> None:
        """Tally an emergency rescue: count, premium cost, and charging cost.

        Emitted by ``bridge/emergency._rescue`` (sikap B) with ``energy_kwh``,
        ``premium_cost``, ``price_per_kwh``. The premium is both an emergency
        cost (group 2 dedicated field) and part of the fleet's total
        ``charging_cost`` (negotiation + emergency, per ``06-03``).

        Example:
            >>> mc = MetricsCollector()
            >>> mc._on_emergency_rescue(energy_kwh=10.0, premium_cost=500.0,
            ...     price_per_kwh=50.0)
            >>> mc.emergency_rescue_count, mc.emergency_premium_cost
            (1, 500.0)
            >>> mc.charging_cost
            500.0
        """

        premium = _as_float(payload.get("premium_cost"))
        self.emergency_rescue_count += 1
        self.emergency_premium_cost += premium
        self.charging_cost += premium

    def _on_taxi_died(self, **payload: Any) -> None:
        """Tally a stranded taxi (sikap A, ``LET_DIE``).

        Example:
            >>> mc = MetricsCollector()
            >>> mc._on_taxi_died(taxi_id="t0", soc=0.0)
            >>> mc.taxis_died
            1
        """

        self.taxis_died += 1

    def _on_slot_forfeit(self, **payload: Any) -> None:
        """Tally a forfeited reserved slot (loose-guard ``SLOT_FORFEIT``, ``01-08``).

        Example:
            >>> mc = MetricsCollector()
            >>> mc._on_slot_forfeit(taxi_id="t0", port_id="p0")
            >>> mc.slot_forfeit_count
            1
        """

        self.slot_forfeit_count += 1

    def _on_order_unserved(self, **payload: Any) -> None:
        """Tally an order that went unserved (distinct from hour-change expiry).

        Example:
            >>> mc = MetricsCollector()
            >>> mc._on_order_unserved(order_id="o0")
            >>> mc.orders_unserved
            1
        """

        self.orders_unserved += _as_int(payload.get("count"), 1)

    def _on_orders_refreshed(self, **payload: Any) -> None:
        """Tally orders expired at an hour rollover (``01-03``) as unserved.

        Emitted by the scheduler on each hour change with ``expired`` (the count
        of PENDING orders that lapsed). Each expired order is an unserved order
        AND an hour-change expiry (both ``06-03`` fields).

        Example:
            >>> mc = MetricsCollector()
            >>> mc._on_orders_refreshed(hour=9, expired=3)
            >>> mc.orders_expired_at_hour_change, mc.orders_unserved
            (3, 3)
        """

        expired = _as_int(payload.get("expired"))
        self.orders_expired_at_hour_change += expired
        self.orders_unserved += expired

    def _on_no_slot_aged(self, **payload: Any) -> None:
        """Sample a taxi's ``cycles_waited`` after a no-slot aging (starvation).

        Emitted by ``bridge/apply._age_unwon`` with the post-increment
        ``cycles_waited``.

        Example:
            >>> mc = MetricsCollector()
            >>> mc._on_no_slot_aged(taxi_id="t0", cycles_waited=2)
            >>> mc.summary()["negotiation_system"]["avg_cycles_waited"]
            2.0
        """

        cycles = _as_int(payload.get("cycles_waited"))
        self._cycles_waited_sum += cycles
        self._cycles_waited_samples += 1
        self._cycles_waited_hist[cycles] = self._cycles_waited_hist.get(cycles, 0) + 1

    # ------------------------------------------------------------------
    # Group 3 — Negotiation / system handlers
    # ------------------------------------------------------------------

    def _on_negotiation_completed(self, **payload: Any) -> None:
        """Tally one negotiation's deal / partial outcome (``deal_success_rate``).

        Emitted by the scheduler with ``deal`` (bool), ``candidate_taxis`` and
        any scalar run-summary fields. A ``partial`` flag (when the orchestrator
        or bridge sets it) marks a deal that left some taxi UNASSIGNED.

        Example:
            >>> mc = MetricsCollector()
            >>> mc._on_negotiation_completed(deal=True, candidate_taxis=3,
            ...     partial=True)
            >>> mc._on_negotiation_completed(deal=False, candidate_taxis=2)
            >>> s = mc.summary()["negotiation_system"]
            >>> round(s["deal_success_rate"], 3)
            0.5
            >>> round(s["partial_deal_rate"], 3)
            1.0
        """

        self.negotiations_total += 1
        if bool(payload.get("deal")):
            self.deals_total += 1
            if bool(payload.get("partial")):
                self.partial_deals_total += 1

    def _on_negotiation_utility(self, **payload: Any) -> None:
        """Accumulate one side's realized terminal utility for a negotiation.

        Emitted by the Tahap H orchestrator (which owns the agents / episodes)
        with ``side`` (``"fleet"`` / ``"spklu"``) and ``utility``. Feeds both
        ``total_utility`` and ``avg_utility_per_negotiation`` per side.

        Example:
            >>> mc = MetricsCollector()
            >>> mc._on_negotiation_utility(side="fleet", utility=0.8)
            >>> mc._on_negotiation_utility(side="fleet", utility=0.4)
            >>> s = mc.summary()["negotiation_system"]["fleet"]
            >>> round(s["total_utility"], 4), round(s["avg_utility_per_negotiation"], 4)
            (1.2, 0.6)
        """

        side = str(payload.get("side", "")).lower()
        if side not in self._utility_sum:
            return
        self._utility_sum[side] += _as_float(payload.get("utility"))
        self._utility_count[side] += 1

    def _on_negotiation_payload(self, **payload: Any) -> None:
        """Store one phase's episode payload into the payload history (``06-03``).

        Emitted by the Tahap H orchestrator with ``episode`` (an
        :class:`~negotiation.data_type.episode.Episode`), ``side`` and ``phase``.
        Always counted; the full object is kept only when
        ``keep_payload_history`` is on (it also feeds offline training & the
        registry-side history file).

        Example:
            >>> mc = MetricsCollector()
            >>> mc._on_negotiation_payload(side="fleet", phase="propose",
            ...     episode={"agent_name": "fleet"})
            >>> len(mc.negotiation_payload_history), mc._payload_count
            (1, 1)
        """

        self._payload_count += 1
        if self._keep_payload_history:
            self.negotiation_payload_history.append(
                {
                    "side": payload.get("side"),
                    "phase": payload.get("phase"),
                    "episode": payload.get("episode"),
                }
            )

    def _on_training_step(self, **payload: Any) -> None:
        """Record per-(side, phase) training losses for one ``training_step`` (LLR-M5.2).

        Emitted by the Tahap H orchestrator's
        :meth:`~simulation.training.orchestrator.Orchestrator.on_train_tick`
        (and the scheduler's default on-the-fly training path) with
        ``losses`` — captured per ``update()`` (KR-8). Each entry is appended to
        :attr:`loss_history` under its ``(side, phase)`` key. No loss formula is
        computed or altered (INV-3); losses are only stored as reported.

        ``losses`` accepts either shape:

        * a list of ``(side, phase, loss)`` tuples (the orchestrator's wire
          format), or
        * a list of ``{"side": ..., "phase": ..., "loss": ...}`` dicts.

        Incomplete or non-numeric entries are ignored; non-finite values are
        dropped by :class:`~simulation.metrics.loss_history.LossHistory` itself.

        Example:
            >>> mc = MetricsCollector()
            >>> mc.record("training_step", losses=[("fleet", "propose", 0.5)])
            >>> mc.loss_history.get("fleet", "propose")
            [0.5]
            >>> mc.record("training_step",
            ...     losses=[{"side": "spklu", "phase": "respond", "loss": 0.2}])
            >>> mc.loss_history.get("spklu", "respond")
            [0.2]
            >>> mc.record("training_step", losses=[])
            >>> mc.loss_history.get("fleet", "respond")
            []
        """

        entries = payload.get("losses")
        if not entries:
            return
        for entry in entries:
            side: Any
            phase: Any
            loss: Any
            if isinstance(entry, dict):
                side = entry.get("side")
                phase = entry.get("phase")
                loss = entry.get("loss")
            elif isinstance(entry, (tuple, list)) and len(entry) == 3:
                side, phase, loss = entry
            else:
                continue
            if side is None or phase is None or loss is None:
                continue
            try:
                value = float(loss)
            except (TypeError, ValueError):
                continue
            self.loss_history.record(str(side), str(phase), value)

    def record_period_utilities(
        self, fleet_utility: float, spklu_utility: float
    ) -> None:
        """Append one negotiation period's realized utilities, AS-IS (LLR-M5.3, #C4).

        Called by the RunReporter **once per period** (deal / no-deal /
        no-negotiation), so utility series include a value for *every* period —
        zeros for no-negotiation periods included (KR-7). The caller computes the
        mapping (#C4): deal -> actual ``terminal_utility``; no-deal -> ``-1.0``
        (real failure penalty); no-negotiation -> ``0.0``. This method stores the
        passed values unchanged — no clamping, no formula change (INV-3). It is
        kept separate from :meth:`_on_negotiation_utility` (which is per-side-event
        for the summary averages); do NOT merge the two series.

        Args:
            fleet_utility: The fleet side's period utility (already final).
            spklu_utility: The SPKLU side's period utility (already final).

        Example:
            >>> mc = MetricsCollector()
            >>> mc.record_period_utilities(0.8, 0.6)
            >>> mc.record_period_utilities(-1.0, -1.0)
            >>> mc.record_period_utilities(0.0, 0.0)
            >>> mc.period_utility_series["fleet"]
            [0.8, -1.0, 0.0]
            >>> mc.period_utility_series["spklu"]
            [0.6, -1.0, 0.0]
        """

        self.period_utility_series["fleet"].append(_as_float(fleet_utility))
        self.period_utility_series["spklu"].append(_as_float(spklu_utility))

    def loss_history_dict(self) -> Dict[str, List[float]]:
        """Return the captured loss series as a ``{"side:phase": [...]}`` dict (LLR-M5.4).

        A debug/serialization accessor delegating to
        :meth:`~simulation.metrics.loss_history.LossHistory.to_dict`. Loss series
        are intentionally NOT folded into :meth:`summary` (keeping the snapshot
        small and the cumulative metrics untouched).

        Example:
            >>> mc = MetricsCollector()
            >>> mc.record("training_step", losses=[("fleet", "propose", 0.5)])
            >>> mc.loss_history_dict()
            {'fleet:propose': [0.5]}
        """

        return self.loss_history.to_dict()

    # ------------------------------------------------------------------
    # Serializable summary (Contract C8)
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a JSON-serializable snapshot of all three metric groups (C8).

        The payload *history* is reported as a count in the summary (the full
        list lives on :attr:`negotiation_payload_history` for the registry /
        offline trainer) so the dashboard snapshot stays serializable and small.

        Returns:
            A nested ``dict`` of plain scalars / dicts under ``spklu_business``,
            ``fleet_business``, ``negotiation_system``.

        Example:
            >>> import json
            >>> mc = MetricsCollector()
            >>> mc.record("trip_completed", fare=10.0, productive_km=1.0)
            >>> json.loads(json.dumps(mc.summary()))["fleet_business"]["orders_served"]
            1
        """

        avg_price = (
            self._price_weighted_kwh / self._kwh_sold if self._kwh_sold > 0.0 else 0.0
        )
        utilization_ratio: Optional[float] = None
        if self._port_capacity_hours and self._port_capacity_hours > 0.0:
            utilization_ratio = self.port_hours_used / self._port_capacity_hours

        deal_rate = (
            self.deals_total / self.negotiations_total
            if self.negotiations_total > 0 else 0.0
        )
        partial_rate = (
            self.partial_deals_total / self.deals_total
            if self.deals_total > 0 else 0.0
        )
        avg_cycles = (
            self._cycles_waited_sum / self._cycles_waited_samples
            if self._cycles_waited_samples > 0 else 0.0
        )

        return {
            "spklu_business": {
                "total_revenue": self.total_revenue,
                "total_compensation_paid": self.total_compensation_paid,
                "shifted_bookings_count": self.shifted_bookings_count,
                "port_utilization": {
                    "port_hours_used": self.port_hours_used,
                    "capacity_hours": self._port_capacity_hours,
                    "ratio": utilization_ratio,
                },
                "avg_price_per_kwh": avg_price,
            },
            "fleet_business": {
                "passenger_revenue": self.passenger_revenue,
                "charging_cost": self.charging_cost,
                "unproductive_km": self.unproductive_km,
                "orders_served": self.orders_served,
                "orders_unserved": self.orders_unserved,
                "taxis_died": self.taxis_died,
                "emergency_rescue_count": self.emergency_rescue_count,
                "emergency_premium_cost": self.emergency_premium_cost,
                "slot_forfeit_count": self.slot_forfeit_count,
            },
            "negotiation_system": {
                "fleet": {
                    "total_utility": self._utility_sum["fleet"],
                    "avg_utility_per_negotiation": (
                        self._utility_sum["fleet"] / self._utility_count["fleet"]
                        if self._utility_count["fleet"] > 0 else 0.0
                    ),
                },
                "spklu": {
                    "total_utility": self._utility_sum["spklu"],
                    "avg_utility_per_negotiation": (
                        self._utility_sum["spklu"] / self._utility_count["spklu"]
                        if self._utility_count["spklu"] > 0 else 0.0
                    ),
                },
                "deal_success_rate": deal_rate,
                "partial_deal_rate": partial_rate,
                "orders_expired_at_hour_change": self.orders_expired_at_hour_change,
                "negotiation_payload_count": self._payload_count,
                "avg_cycles_waited": avg_cycles,
                "cycles_waited_distribution": dict(self._cycles_waited_hist),
                "negotiations_total": self.negotiations_total,
            },
        }

    def as_dict(self) -> Dict[str, Any]:
        """Alias of :meth:`summary` (Contract C8 naming variant).

        Example:
            >>> MetricsCollector().as_dict() == MetricsCollector().summary()
            True
        """

        return self.summary()

    def snapshot(self) -> Dict[str, Any]:
        """Alias of :meth:`summary` (Contract C8 naming variant).

        Example:
            >>> MetricsCollector().snapshot() == MetricsCollector().summary()
            True
        """

        return self.summary()

    def add_unproductive_km(self, km: float) -> None:
        """Add precise unproductive km (idle→pickup + empty travel) (``06-03``).

        Routing precision lives in the world; this setter lets the scheduler /
        orchestrator feed the per-run total without the collector importing the
        router.

        Example:
            >>> mc = MetricsCollector()
            >>> mc.add_unproductive_km(2.5)
            >>> mc.unproductive_km
            2.5
        """

        self.unproductive_km += _as_float(km)

    # ------------------------------------------------------------------
    # Per-day business metrics via snapshot + diff (LLR-M5.5 / #C1)
    # ------------------------------------------------------------------

    def numeric_metrics_flat(self) -> Dict[str, float]:
        """Flatten :meth:`summary` into a flat ``{path: value}`` map of cumulative accumulators (LLR-M5.5).

        Only **monotone cumulative accumulators** that are meaningful to subtract
        (diff) across day boundaries are included (#C1). Ratios / averages
        (``avg_price_per_kwh``, ``deal_success_rate``, ``partial_deal_rate``,
        ``avg_cycles_waited``, ``avg_utility_per_negotiation``,
        ``port_utilization.ratio``) are NOT accumulators and are deliberately
        EXCLUDED — diffing them naively would be wrong; per-day ratios are derived
        from the diffed counts by the reporter instead. No formula is changed —
        this is purely a re-aggregation/reporting view (INV-3, INV-5).

        Returns:
            A flat ``dict`` of ``str`` dotted paths to ``float`` values.

        Example:
            >>> mc = MetricsCollector()
            >>> mc.record("deal_applied", revenue=600.0, energy_kwh=20.0,
            ...     price_per_kwh=30.0, port_hours=1.0, compensation=50.0,
            ...     shifted_bookings=1)
            >>> flat = mc.numeric_metrics_flat()
            >>> flat["spklu_business.total_revenue"]
            600.0
            >>> flat["fleet_business.charging_cost"]
            600.0
            >>> "spklu_business.avg_price_per_kwh" in flat
            False
        """

        summary = self.summary()
        spklu = summary["spklu_business"]
        fleet = summary["fleet_business"]
        negotiation = summary["negotiation_system"]

        flat: Dict[str, float] = {
            "spklu_business.total_revenue": float(spklu["total_revenue"]),
            "spklu_business.total_compensation_paid": float(
                spklu["total_compensation_paid"]
            ),
            "spklu_business.shifted_bookings_count": float(
                spklu["shifted_bookings_count"]
            ),
            "spklu_business.port_utilization.port_hours_used": float(
                spklu["port_utilization"]["port_hours_used"]
            ),
            "fleet_business.passenger_revenue": float(fleet["passenger_revenue"]),
            "fleet_business.charging_cost": float(fleet["charging_cost"]),
            "fleet_business.unproductive_km": float(fleet["unproductive_km"]),
            "fleet_business.orders_served": float(fleet["orders_served"]),
            "fleet_business.orders_unserved": float(fleet["orders_unserved"]),
            "fleet_business.taxis_died": float(fleet["taxis_died"]),
            "fleet_business.emergency_rescue_count": float(
                fleet["emergency_rescue_count"]
            ),
            "fleet_business.emergency_premium_cost": float(
                fleet["emergency_premium_cost"]
            ),
            "fleet_business.slot_forfeit_count": float(fleet["slot_forfeit_count"]),
            "negotiation_system.negotiations_total": float(
                negotiation["negotiations_total"]
            ),
            "negotiation_system.orders_expired_at_hour_change": float(
                negotiation["orders_expired_at_hour_change"]
            ),
            "negotiation_system.negotiation_payload_count": float(
                negotiation["negotiation_payload_count"]
            ),
            "negotiation_system.fleet.total_utility": float(
                negotiation["fleet"]["total_utility"]
            ),
            "negotiation_system.spklu.total_utility": float(
                negotiation["spklu"]["total_utility"]
            ),
        }
        return flat

    @staticmethod
    def diff_numeric_metrics(
        curr: Dict[str, float], prev: Dict[str, float]
    ) -> Dict[str, float]:
        """Return the per-field difference of two flat metric snapshots (LLR-M5.5).

        Computes ``curr[k] - prev[k]`` for every key in ``curr`` (treating a
        missing ``prev`` key as ``0.0``), yielding the values accrued **between**
        the two snapshots — i.e. the metrics for a single day when ``curr`` is the
        end-of-day snapshot and ``prev`` the previous day's (#C1). No formula is
        changed.

        Args:
            curr: The later (current) flat snapshot from :meth:`numeric_metrics_flat`.
            prev: The earlier flat snapshot to subtract.

        Returns:
            A flat ``dict`` of per-field deltas.

        Example:
            >>> MetricsCollector.diff_numeric_metrics(
            ...     {"a": 5.0, "b": 3.0}, {"a": 2.0})
            {'a': 3.0, 'b': 3.0}
        """

        return {key: curr.get(key, 0.0) - prev.get(key, 0.0) for key in curr}


if __name__ == "__main__":  # pragma: no cover - lightweight self-check
    import doctest

    failures, _ = doctest.testmod(optionflags=doctest.ELLIPSIS)
    if failures:
        raise SystemExit(f"{failures} doctest failure(s)")
    print("MetricsCollector doctests passed")
