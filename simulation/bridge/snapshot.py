"""World snapshot, candidate collection, and per-side view glue (``05-01``/``05-02``).

When a negotiation is triggered the engine **freezes** simulated time and the
bridge takes a single, consistent :class:`WorldState` snapshot to drive the whole
negotiation (``05-03``). This module provides:

* :func:`snapshot_world` — return the frozen ground-truth :class:`WorldState`.
* :func:`project_view` — re-exported from :mod:`simulation.world.world_state`;
  the per-side, visibility-filtered :class:`AgentView` builder (``03-03``). The
  bridge is the only module that holds the full :class:`WorldState` and that
  derives the asymmetric per-side views from it.
* :func:`collect_charge_candidates` — gather **every** idle, "stuck" taxi that
  should join a negotiation (``05-01``). A taxi is a candidate iff it is
  ``IDLE_WAITING``, cannot reach any marking area while keeping ``SoC >= B_area``
  (so it cannot keep serving orders), and can still reach at least one port.
  There is no cap and no urgency ranking — all such taxis join; the bundle count
  is bounded only by ``K_max``. (Stuck taxis that can reach **no** port are
  handled as emergencies by :mod:`simulation.bridge.emergency` before this runs.)

The bridge MAY import from both :mod:`simulation.world` and ``negotiation`` (it is
the sole bilingual layer); this particular module only needs the world side.

Example
-------
>>> from simulation.config.default import default_config
>>> from simulation.world.routing import Router
>>> from simulation.world.world_state import WorldState
>>> cfg = default_config()
>>> world = WorldState(config=cfg, router=Router(speed_kmh=cfg.taxi_spec.speed_kmh))
>>> snap = snapshot_world(world)
>>> snap.sim_time_min
0.0
"""

from __future__ import annotations

import copy
from typing import List

from simulation.world.entities import Taxi, TaxiState
from simulation.world.idle_logic import can_reach_any_port, nearest_reachable_area
from simulation.world.world_state import AgentView, WorldState, project_view

__all__ = [
    "snapshot_world",
    "project_view",
    "AgentView",
    "collect_charge_candidates",
]


def snapshot_world(world: WorldState) -> WorldState:
    """Return the frozen ground-truth :class:`WorldState` for one negotiation.

    The negotiation runs against one consistent snapshot taken at freeze
    (``05-03``). Because the engine freezes simulated time around the whole
    negotiation, the live :class:`WorldState` is itself already frozen and is
    returned as-is: the bridge mutates the *real* world only later, via
    :mod:`simulation.bridge.apply`. Callers that need an isolated copy (e.g. for
    speculative what-if scoring) can pass ``world`` through :func:`copy.deepcopy`
    explicitly; the default keeps the cheap, no-copy contract the scheduler
    relies on (apply must write back to the same object).

    Args:
        world: The live, frozen :class:`WorldState`.

    Returns:
        The same :class:`WorldState` instance (ground-truth for this negotiation).

    Example:
        >>> from simulation.config.default import default_config
        >>> from simulation.world.routing import Router
        >>> w = WorldState(config=default_config(),
        ...                router=Router(speed_kmh=30.0))
        >>> snapshot_world(w) is w
        True
    """

    return world


def snapshot_world_copy(world: WorldState) -> WorldState:
    """Return a deep copy of the world (isolated, for speculative scoring).

    Unlike :func:`snapshot_world`, this detaches from the live world so callers
    can score hypothetical mutations without affecting the real simulation. The
    router is shared (it is an immutable distance cache), but every entity is
    copied.

    Args:
        world: The live :class:`WorldState`.

    Returns:
        A deep-copied :class:`WorldState` sharing only the router.

    Example:
        >>> from simulation.config.default import default_config
        >>> from simulation.world.routing import Router
        >>> w = WorldState(config=default_config(),
        ...                router=Router(speed_kmh=30.0))
        >>> snapshot_world_copy(w) is w
        False
    """

    router = world.router
    config = world.config
    # Temporarily detach the un-copyable / shared singletons, deep-copy the
    # mutable entity graph, then restore the shared references.
    world.router = None  # type: ignore[assignment]
    world.config = None  # type: ignore[assignment]
    try:
        clone = copy.deepcopy(world)
    finally:
        world.router = router
        world.config = config
    clone.router = router
    clone.config = config
    return clone


def collect_charge_candidates(world: WorldState) -> List[Taxi]:
    """Return every idle, "stuck" taxi eligible to negotiate a charging slot (``05-01``).

    A candidate is a taxi that is

    1. ``IDLE_WAITING`` (not serving a passenger, reserved, charging, or stranded);
    2. **stuck** — it cannot reach any marking area while keeping ``SoC >= B_area``
       (:func:`~simulation.world.idle_logic.nearest_reachable_area` returns
       ``None``), so it can no longer keep serving orders; and
    3. **placeable** — it can still reach at least one port
       (:func:`~simulation.world.idle_logic.can_reach_any_port`).

    There is no cap and no urgency ordering: every such taxi joins the
    negotiation (the action-library size is bounded only by ``K_max``). Stuck
    taxis that can reach **no** port are *not* returned here; they are handled as
    emergencies by :mod:`simulation.bridge.emergency` before this is called.

    Args:
        world: The frozen :class:`WorldState`.

    Returns:
        The list of candidate taxis (unordered). Empty means no negotiation runs.

    Example:
        >>> from simulation.config.default import default_config
        >>> from simulation.world.routing import Router
        >>> w = WorldState(config=default_config(),
        ...                router=Router(speed_kmh=30.0))
        >>> from simulation.world.entities import Taxi, TaxiState
        >>> w.taxis = {
        ...     "busy": Taxi(id="busy", position=(0.0, 0.0), soc=0.9,
        ...                  state=TaxiState.WITH_PASSENGER),
        ... }
        >>> collect_charge_candidates(w)
        []
    """

    candidates: List[Taxi] = []
    for taxi in world.taxis.values():
        if taxi.state is not TaxiState.IDLE_WAITING:
            continue
        # Still able to (re)position to an area above B_area -> productive, skip.
        if nearest_reachable_area(taxi, world) is not None:
            continue
        # Stuck: a candidate only if it can still reach a port (else emergency,
        # handled by the bridge before this runs).
        if can_reach_any_port(taxi, world):
            candidates.append(taxi)
    return candidates
