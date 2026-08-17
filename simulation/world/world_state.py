"""Ground-truth world state and its per-side filtered projection.

:class:`WorldState` holds the complete, authoritative snapshot of the
simulation world (every taxi, station/port/booking, marking area/order, the
router, the config, and the current simulated time). It is the object the
state machine, idle logic, order generator, and bridge all read and mutate.

Per Contract C5 / Gotcha G4 it exposes exactly the surface the state machine
needs: ``config``, ``router``, a simulated-time attribute (``sim_time_min``),
:meth:`get_port`, :meth:`station_of_port`, and a default no-op
:meth:`metrics_hook` (the scheduler swaps in the real hook later).

:class:`AgentView` is a per-side filtered projection (fleet / SPKLU) produced
by :func:`project_view` according to a ``VisibilityConfig``. By default the
SPKLU side is blind to the taxi list (asymmetric visibility, see ``01-01`` /
``03-03``). Only the bridge holds the full :class:`WorldState`; each agent only
ever sees its :class:`AgentView`.

This module depends only on :mod:`simulation.config` and
:mod:`simulation.world.entities` / :mod:`simulation.world.routing`. It never
imports ``negotiation`` (layer separation).

Example
-------
>>> from simulation.config.default import default_config
>>> from simulation.world.routing import Router
>>> from simulation.world.world_state import WorldState
>>> cfg = default_config()
>>> world = WorldState(config=cfg, router=Router(speed_kmh=cfg.taxi_spec.speed_kmh))
>>> world.metrics_hook("noop_event")          # default no-op, returns None
>>> world.current_grid()
0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from simulation.config.config import SimulationConfig, VisibilityConfig
from simulation.world.entities import (
    AppUserBooking,
    ChargingStation,
    MarkingArea,
    Port,
    Taxi,
)
from simulation.world.routing import Router

# Half-hour grid width in minutes (structural constant, see ``01-08``).
_GRID_MINUTES: int = 30
# Minutes in one simulated day.
_MINUTES_PER_DAY: int = 24 * 60
# Minutes in one hour.
_MINUTES_PER_HOUR: int = 60


def _noop_metrics_hook(event: str, **payload: object) -> None:
    """Default metrics hook: do nothing (see Gotcha G4).

    The state machine calls ``world.metrics_hook(event, **payload)`` at notable
    transitions. Until the scheduler installs a real collector, this no-op keeps
    the world self-contained and side-effect free.

    Args:
        event: Name of the metric event (e.g. ``"trip_completed"``).
        **payload: Arbitrary event payload (ignored here).

    Example:
        >>> _noop_metrics_hook("trip_completed", taxi_id="taxi_0") is None
        True
    """

    return None


MetricsHook = Callable[..., None]


@dataclass
class WorldState:
    """Authoritative ground-truth state of the entire simulation world.

    Attributes:
        config: The immutable :class:`SimulationConfig` for this run.
        router: The :class:`Router` answering distance/time queries.
        taxis: All taxis, keyed by taxi id.
        stations: All regular (non-emergency) charging stations, keyed by id.
        areas: All marking areas, keyed by id.
        bookings: All app-user bookings, keyed by booking id.
        emergency_station: The emergency rescue station (sikap B), or ``None``
            if dead-battery handling is LET_DIE (sikap A) and no rescue exists.
        sim_time_min: Current simulated time in minutes since the run start.
        metrics_hook: Callback invoked by the state machine at notable events.
            Defaults to a no-op; the scheduler installs the real collector
            (Gotcha G4 / Contract C5).

    Example:
        >>> from simulation.config.default import default_config
        >>> cfg = default_config()
        >>> world = WorldState(config=cfg,
        ...                    router=Router(speed_kmh=cfg.taxi_spec.speed_kmh))
        >>> world.sim_time_min
        0.0
        >>> world.hour()
        0
    """

    config: SimulationConfig
    router: Router
    taxis: Dict[str, Taxi] = field(default_factory=dict)
    stations: Dict[str, ChargingStation] = field(default_factory=dict)
    areas: Dict[str, MarkingArea] = field(default_factory=dict)
    bookings: Dict[str, AppUserBooking] = field(default_factory=dict)
    emergency_station: Optional[ChargingStation] = None
    sim_time_min: float = 0.0
    metrics_hook: MetricsHook = field(default=_noop_metrics_hook)
    # Internal index: port id -> owning station id (built lazily on demand).
    _port_index: Dict[str, str] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Time helpers
    # ------------------------------------------------------------------
    def hour(self) -> int:
        """Return the current hour-of-day (0..23).

        Returns:
            The hour-of-day derived from :attr:`sim_time_min`.

        Example:
            >>> from simulation.config.default import default_config
            >>> cfg = default_config()
            >>> w = WorldState(config=cfg, router=Router(speed_kmh=30.0))
            >>> w.sim_time_min = 90.0
            >>> w.hour()
            1
        """

        minute_of_day = int(self.sim_time_min) % _MINUTES_PER_DAY
        return minute_of_day // _MINUTES_PER_HOUR

    def day(self) -> int:
        """Return the current day index (0-based).

        Returns:
            Number of whole simulated days elapsed since the run start.

        Example:
            >>> from simulation.config.default import default_config
            >>> w = WorldState(config=default_config(), router=Router(speed_kmh=30.0))
            >>> w.sim_time_min = 25 * 60.0
            >>> w.day()
            1
        """

        return int(self.sim_time_min) // _MINUTES_PER_DAY

    def current_grid(self) -> int:
        """Return the half-hour grid index covering the current simulated time.

        Grid ``g`` covers minutes ``[g * 30, (g + 1) * 30)`` (see ``01-08``).

        Returns:
            The current grid index.

        Example:
            >>> from simulation.config.default import default_config
            >>> w = WorldState(config=default_config(), router=Router(speed_kmh=30.0))
            >>> w.sim_time_min = 75.0
            >>> w.current_grid()
            2
        """

        return int(self.sim_time_min) // _GRID_MINUTES

    def time_weight_now(self) -> float:
        """Return the configured ``time_weight`` for the current hour.

        Returns:
            The weight (single source for order volume and the fleet ufun's
            "right timing" factor, see ``01-03``/``04-02``); ``1.0`` if the
            hour is absent from the table.

        Example:
            >>> from simulation.config.default import default_config
            >>> w = WorldState(config=default_config(), router=Router(speed_kmh=30.0))
            >>> isinstance(w.time_weight_now(), float)
            True
        """

        return float(self.config.marking_area_order.time_weight.get(self.hour(), 1.0))

    # ------------------------------------------------------------------
    # Port / station helpers (Contract C5)
    # ------------------------------------------------------------------
    def _rebuild_port_index(self) -> None:
        """Rebuild the internal port-id -> station-id index.

        Includes the emergency station's ports so :meth:`get_port` /
        :meth:`station_of_port` resolve rescue ports too.

        Example:
            >>> from simulation.config.default import default_config
            >>> w = WorldState(config=default_config(), router=Router(speed_kmh=30.0))
            >>> w._rebuild_port_index()
            >>> w._port_index
            {}
        """

        self._port_index = {}
        for station in self.stations.values():
            for port in station.ports:
                self._port_index[port.id] = station.id
        if self.emergency_station is not None:
            for port in self.emergency_station.ports:
                self._port_index[port.id] = self.emergency_station.id

    def all_ports(self) -> List[Port]:
        """Return every regular (non-emergency) port across all stations.

        Emergency ports are intentionally excluded so callers building the
        negotiation never see them (Gotcha G8).

        Returns:
            A flat list of all ports belonging to regular stations.

        Example:
            >>> from simulation.config.default import default_config
            >>> w = WorldState(config=default_config(), router=Router(speed_kmh=30.0))
            >>> w.all_ports()
            []
        """

        ports: List[Port] = []
        for station in self.stations.values():
            ports.extend(station.ports)
        return ports

    def get_port(self, port_id: str) -> Optional[Port]:
        """Return the :class:`Port` with ``port_id`` (regular or emergency).

        Args:
            port_id: Identifier of the port to fetch.

        Returns:
            The matching :class:`Port`, or ``None`` if no such port exists.

        Example:
            >>> from simulation.config.default import default_config
            >>> w = WorldState(config=default_config(), router=Router(speed_kmh=30.0))
            >>> w.get_port("does_not_exist") is None
            True
        """

        if port_id not in self._port_index:
            self._rebuild_port_index()
        station_id = self._port_index.get(port_id)
        if station_id is None:
            return None
        station = self.station_of_port(port_id)
        if station is None:
            return None
        for port in station.ports:
            if port.id == port_id:
                return port
        return None

    def station_of_port(self, port_id: str) -> Optional[ChargingStation]:
        """Return the :class:`ChargingStation` owning ``port_id``.

        Args:
            port_id: Identifier of the port.

        Returns:
            The owning station (regular or emergency), or ``None`` if unknown.

        Example:
            >>> from simulation.config.default import default_config
            >>> w = WorldState(config=default_config(), router=Router(speed_kmh=30.0))
            >>> w.station_of_port("does_not_exist") is None
            True
        """

        if port_id not in self._port_index:
            self._rebuild_port_index()
        station_id = self._port_index.get(port_id)
        if station_id is None:
            return None
        if station_id in self.stations:
            return self.stations[station_id]
        if self.emergency_station is not None and self.emergency_station.id == station_id:
            return self.emergency_station
        return None


@dataclass
class AgentView:
    """A per-side, visibility-filtered projection of the world (see ``01-01``).

    The view carries only the fields a given side (fleet / SPKLU) is permitted
    to observe by its :class:`VisibilityConfig`. State and bundle features for
    that side may use only what is present here (consistency requirement,
    ``03-03``).

    Attributes:
        side: ``"fleet"`` or ``"spklu"``.
        hour: Current hour-of-day (always visible scalar).
        day_progress: Fraction of the current day elapsed in ``[0, 1)``.
        grid: Current half-hour grid index.
        visible_taxis: Taxis visible to this side (empty for the default
            taxi-blind SPKLU view).
        visible_ports: Ports visible to this side.
        visible_areas: Marking areas visible to this side.
        visible_bookings: App-user bookings visible to this side.
        visible_fields: The raw list of allowed world-field names for this side.

    Example:
        >>> AgentView(side="spklu", hour=8, day_progress=0.33, grid=16).visible_taxis
        []
    """

    side: str
    hour: int
    day_progress: float
    grid: int
    visible_taxis: List[Taxi] = field(default_factory=list)
    visible_ports: List[Port] = field(default_factory=list)
    visible_areas: List[MarkingArea] = field(default_factory=list)
    visible_bookings: List[AppUserBooking] = field(default_factory=list)
    visible_fields: List[str] = field(default_factory=list)


def _side_visible_fields(visibility: VisibilityConfig, side: str) -> List[str]:
    """Return the allowed world-field names for a side from a VisibilityConfig.

    Args:
        visibility: The per-side visibility configuration.
        side: ``"fleet"`` or ``"spklu"``.

    Returns:
        The list of field names visible to ``side``.

    Raises:
        ValueError: If ``side`` is neither ``"fleet"`` nor ``"spklu"``.

    Example:
        >>> vc = VisibilityConfig(fleet=["taxi.soc"], spklu=["port.power_kw"])
        >>> _side_visible_fields(vc, "fleet")
        ['taxi.soc']
    """

    if side == "fleet":
        return list(visibility.fleet)
    if side == "spklu":
        return list(visibility.spklu)
    raise ValueError(f"Unknown negotiation side: {side!r} (expected 'fleet' or 'spklu')")


def project_view(world: WorldState, side: str, visibility: VisibilityConfig) -> AgentView:
    """Project the ground-truth world into a filtered :class:`AgentView` for a side.

    A category (taxi / port / area / booking) is included only if the side's
    visible-field list mentions that category at all (any field prefixed with
    ``"<category>."``). By default the SPKLU side lists no ``taxi.*`` fields and
    is therefore blind to the taxi list (asymmetric visibility, see ``03-03``).

    Args:
        world: The full ground-truth :class:`WorldState`.
        side: ``"fleet"`` or ``"spklu"``.
        visibility: The per-side :class:`VisibilityConfig`.

    Returns:
        An :class:`AgentView` containing only the categories this side may see.

    Example:
        >>> from simulation.config.default import default_config
        >>> cfg = default_config()
        >>> w = WorldState(config=cfg, router=Router(speed_kmh=30.0))
        >>> view = project_view(w, "spklu", cfg.state_feature.visibility)
        >>> view.visible_taxis
        []
    """

    fields = _side_visible_fields(visibility, side)
    sees_taxi = any(name.startswith("taxi.") for name in fields)
    sees_port = any(name.startswith("port.") for name in fields)
    sees_area = any(name.startswith("area.") for name in fields)
    sees_booking = any(name.startswith("booking.") for name in fields)

    minute_of_day = int(world.sim_time_min) % _MINUTES_PER_DAY
    day_progress = minute_of_day / float(_MINUTES_PER_DAY)

    visible_taxis = list(world.taxis.values()) if sees_taxi else []
    visible_ports = world.all_ports() if sees_port else []
    visible_areas = list(world.areas.values()) if sees_area else []
    visible_bookings = list(world.bookings.values()) if sees_booking else []

    return AgentView(
        side=side,
        hour=world.hour(),
        day_progress=day_progress,
        grid=world.current_grid(),
        visible_taxis=visible_taxis,
        visible_ports=visible_ports,
        visible_areas=visible_areas,
        visible_bookings=visible_bookings,
        visible_fields=fields,
    )
