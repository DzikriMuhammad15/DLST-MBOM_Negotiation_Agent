"""Feature blocks, masks, candidate features, and FeatureDims (``03-03``/``03-04``).

The bridge turns the per-side, visibility-filtered
:class:`~simulation.world.world_state.AgentView` into the ``RawObservation`` dict
the mode-aware networks consume, plus the per-candidate feature matrix
(``cand_feats``) summarizing each bundle from that side's viewpoint.

``RawObservation`` (un-batched; the agent's ``obs_to_batch`` adds the batch dim):

    {
      "taxi_feats": (N, F_taxi),  "taxi_mask": (N,),
      "port_feats": (P, F_port),  "port_mask": (P,),
      "area_feats": (A, F_area),  "area_mask": (A,),
      "scalar_feats": (F_scalar,),
    }

The ``progress_feats`` block (dim 3) is added per step by the agent
(:meth:`BaseNegotiator.progress_vector`), so :class:`FeatureDims.f_progress` is
fixed at 3 here. ``cand_feats`` / ``cand_mask`` are returned separately and injected
via ``configure_negotiation`` (Contract C3).

Three invariants this module guarantees:

* **C7 — dimensions match the networks.** :func:`feature_dims` returns the exact
  :class:`FeatureDims` the agent factory builds its nets with; the emitted blocks
  have those widths. Each block width is ``>= 1`` so the set encoders always have a
  positive input dimension.
* **G12 — N=0 stays ``(0, F)``.** With no taxis/ports/areas the corresponding block
  is shape ``(0, F)`` (never ``(0,)`` / ``None``) with an empty mask, so the
  encoders/padding never break.
* **Asymmetric visibility.** A side's blocks use only fields present in its
  :class:`~simulation.config.config.FeatureBlockSpec` (and thus its visibility);
  the default SPKLU side is taxi-blind, so its taxi block is ``(0, F_taxi)``.

Because the networks expose exactly three set encoders (taxi / port / area), the
SPKLU ``booking`` feature block rides the **area** slot (otherwise unused for
SPKLU): SPKLU "areas" are its bookings. This keeps the three-set architecture and
the C7 dims consistent.

Example
-------
>>> from simulation.config.default import default_config
>>> from simulation.world.routing import Router
>>> from simulation.world.world_state import WorldState
>>> cfg = default_config()
>>> w = WorldState(config=cfg, router=Router(speed_kmh=30.0))
>>> raw, cand, mask, dims = build_features(w, "spklu", [], cfg)
>>> raw["taxi_feats"].shape          # taxi-blind SPKLU, N=0
(0, 1)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from simulation.config.config import FeatureBlockSpec, SimulationConfig
from simulation.world.entities import AppUserBooking, MarkingArea, Port, Taxi
from simulation.world.idle_logic import reachable_area_with_orders, urgency_score
from simulation.world.world_state import AgentView, WorldState
from simulation.bridge.action_library import Bundle
from simulation.bridge.snapshot import project_view

try:
    from negotiation.models.encoders.feature_encoder import FeatureDims
except Exception:  # pragma: no cover - torch always present at runtime
    FeatureDims = None  # type: ignore

# The progress block is supplied per step by the agent (dim 3, see ``03-03``).
_F_PROGRESS: int = 3
# Half-hour grid width (structural constant, see ``01-08``).
_GRID_MINUTES: int = 30
_GRID_HOURS: float = _GRID_MINUTES / 60.0
_MINUTES_PER_DAY: int = 24 * 60


# =============================================================================
# Block width helpers (C7 — every block is >= 1 wide)
# =============================================================================

def _width(field_names: List[str]) -> int:
    """Return a positive block width for a list of field names (min 1, C7)."""
    return max(1, len(field_names))


def _block_spec(config: SimulationConfig, side: str) -> FeatureBlockSpec:
    """Return the :class:`FeatureBlockSpec` for a side."""
    spec = config.state_feature.feature_spec
    return spec.fleet if side == "fleet" else spec.spklu


# =============================================================================
# Per-field extractors
# =============================================================================

def _taxi_field(name: str, taxi: Taxi, world: WorldState) -> float:
    """Return one taxi feature value by ``feature_spec`` field name (``03-03``)."""
    if name == "soc":
        return float(taxi.soc)
    if name == "cycles_waited":
        return float(taxi.cycles_waited)
    if name == "urgency":
        return float(urgency_score(taxi, world))
    if name == "dist_to_area":
        reachable = reachable_area_with_orders(taxi, world)
        return float(reachable[1]) if reachable is not None else 0.0
    if name == "energy_need":
        spec = world.config.taxi_spec
        return float(max(0.0, spec.battery_capacity_kwh * (world.config.battery.SoC_target - taxi.soc)))
    return 0.0


def _port_field(name: str, port: Port, world: WorldState) -> float:
    """Return one port feature value by ``feature_spec`` field name (``03-03``)."""
    if name == "power_kw":
        return float(port.power_kw)
    if name == "base_price":
        return float(port.base_price_per_kwh)
    if name == "free_grids":
        return float(port.free_grids_from(world.current_grid() + 1, 48))
    if name == "n_bookings":
        return float(sum(1 for _s, _d, _h, kind in port.allocations if kind == "booking"))
    return 0.0


def _area_field(name: str, area: MarkingArea, world: WorldState) -> float:
    """Return one area feature value by ``feature_spec`` field name (``03-03``)."""
    if name == "order_rate_now":
        return float(len(area.unfilled_orders()))
    if name == "order_rate_slot":
        # Expected-value proxy for upcoming slots (no realized future orders).
        return float(world.time_weight_now()) * float(world.config.marking_area_order.order_max)
    if name == "time_weight":
        return float(world.time_weight_now())
    return 0.0


def _booking_field(name: str, booking: AppUserBooking, world: WorldState) -> float:
    """Return one booking feature value by ``feature_spec`` field name (``03-03``)."""
    if name == "slot":
        return float(booking.slot_start)
    if name == "is_shiftable":
        return 1.0 if booking.is_shiftable else 0.0
    if name == "compensation":
        return float(booking.compensation)
    return 0.0


# =============================================================================
# Block builders (G12: empty -> (0, F))
# =============================================================================

def _build_set_block(
    items: List[Any], field_names: List[str], extractor, world: WorldState
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a ``(N, F)`` feature block + ``(N,)`` mask for a list of items.

    When ``items`` is empty the block is ``(0, F)`` and the mask is ``(0,)``
    (Gotcha G12). ``F`` is always ``>= 1`` (C7); with no spec fields the block is a
    single constant column.
    """
    width = _width(field_names)
    n = len(items)
    if n == 0:
        return np.zeros((0, width), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    feats = np.zeros((n, width), dtype=np.float32)
    if field_names:
        for i, item in enumerate(items):
            for j, name in enumerate(field_names):
                feats[i, j] = extractor(name, item, world)
    # else: single constant column of zeros (placeholder so the encoder is valid).
    mask = np.ones((n,), dtype=np.float32)
    return feats, mask


def _scalar_block(field_names: List[str], view: AgentView) -> np.ndarray:
    """Build the ``(F_scalar,)`` scalar block (``hour`` / ``day_progress`` ...)."""
    width = _width(field_names)
    vec = np.zeros((width,), dtype=np.float32)
    for j, name in enumerate(field_names):
        if name == "hour":
            vec[j] = float(view.hour)
        elif name == "day_progress":
            vec[j] = float(view.day_progress)
    return vec


# =============================================================================
# Candidate (bundle) features
# =============================================================================

def _cand_fleet(bundle: Bundle, field_names: List[str], world: WorldState) -> List[float]:
    """Summarize a bundle into the fleet per-candidate feature row (``03-04``).

    Fields: ``assigned_count``, ``total_charge_cost``, ``avg_readiness``, and
    ``total_distance`` (the summed exact km every assigned taxi must drive to its
    assigned port's station).
    """
    assigned = bundle.assigned()
    values: Dict[str, float] = {}
    total_cost = 0.0
    total_ready = 0.0
    total_distance = 0.0
    for a in assigned:
        port = world.get_port(a.port_id)
        if port is None:
            continue
        energy = port.power_kw * _GRID_HOURS * a.duration_grids
        total_cost += a.price_per_kwh * energy
        taxi = world.taxis.get(a.taxi_id)
        if taxi is not None and world.config.taxi_spec.battery_capacity_kwh > 0.0:
            need = max(1e-6, world.config.taxi_spec.battery_capacity_kwh
                       * (world.config.battery.SoC_target - taxi.soc))
            total_ready += min(1.0, energy / need)
        station = world.station_of_port(a.port_id)
        if taxi is not None and station is not None:
            total_distance += world.router.distance(taxi.position, station.location)
    count = max(1, len(assigned))
    values["assigned_count"] = float(len(assigned))
    values["total_charge_cost"] = total_cost
    values["avg_readiness"] = total_ready / count
    values["total_distance"] = total_distance
    return [values.get(name, 0.0) for name in field_names]


def _cand_spklu(bundle: Bundle, field_names: List[str], world: WorldState) -> List[float]:
    """Summarize a bundle into the SPKLU per-candidate feature row (``03-04``)."""
    assigned = bundle.assigned()
    revenue = 0.0
    porthours = 0.0
    n_shifted = 0
    total_comp = 0.0
    for a in assigned:
        port = world.get_port(a.port_id)
        if port is None:
            continue
        energy = port.power_kw * _GRID_HOURS * a.duration_grids
        revenue += a.price_per_kwh * energy
        porthours += a.duration_grids * _GRID_HOURS
        if a.shift is not None:
            n_shifted += 1
            total_comp += a.shift.compensation
    values = {
        "revenue": revenue,
        "porthours": porthours,
        "n_shifted": float(n_shifted),
        "total_compensation": total_comp,
    }
    return [values.get(name, 0.0) for name in field_names]


def _build_cand_block(
    library: List[Bundle], side: str, field_names: List[str], world: WorldState, k_max: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the ``(K_max, F_cand)`` candidate block + ``(K_max,)`` mask (``03-04``).

    The block is padded to ``k_max`` so the PADDED propose head (``Linear(K_max)``)
    can ``masked_fill`` it and the ENCODER score head scores a fixed width; the
    first ``len(library)`` rows are real (mask 1) and the rest are padding (mask 0,
    logits masked out). When ``k_max == 0`` (degenerate) the block is ``(0, F)``
    (Gotcha G12). Real candidates beyond ``k_max`` are truncated to fit the head;
    the all-UNASSIGNED bundle has already been kept by the generator's truncation.
    """
    width = _width(field_names)
    k = len(library)
    capacity = max(k, int(k_max))
    if capacity == 0:
        return np.zeros((0, width), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    feats = np.zeros((capacity, width), dtype=np.float32)
    mask = np.zeros((capacity,), dtype=np.float32)
    if field_names and k > 0:
        row_fn = _cand_fleet if side == "fleet" else _cand_spklu
        for i, bundle in enumerate(library):
            feats[i, :] = np.asarray(row_fn(bundle, field_names, world), dtype=np.float32)
    mask[:k] = 1.0
    return feats, mask


# =============================================================================
# Public API
# =============================================================================

def feature_dims(config: SimulationConfig, side: str):
    """Return the :class:`FeatureDims` for a side (Contract C7).

    These widths are what the agent factory builds its networks with and what the
    emitted blocks honor; ``f_progress`` is fixed at 3 (agent-supplied), every
    block width is ``>= 1``, and the caps come from ``encoder_dims``.

    Args:
        config: The simulation config.
        side: ``"fleet"`` or ``"spklu"``.

    Returns:
        A :class:`FeatureDims` instance.

    Example:
        >>> from simulation.config.default import default_config
        >>> dims = feature_dims(default_config(), "fleet")
        >>> dims.f_progress
        3
    """
    if FeatureDims is None:  # pragma: no cover
        raise RuntimeError("torch / feature_encoder is required for FeatureDims.")
    block = _block_spec(config, side)
    enc = config.state_feature.encoder_dims
    if side == "fleet":
        f_taxi = _width(block.taxi)
        f_area = _width(block.area)
    else:
        # SPKLU is taxi-blind; the booking block rides the area slot.
        f_taxi = _width(block.taxi)
        f_area = _width(block.booking)
    return FeatureDims(
        f_taxi=f_taxi,
        f_port=_width(block.port),
        f_area=f_area,
        f_scalar=_width(block.scalar),
        f_progress=_F_PROGRESS,
        f_cand=_width(block.cand),
        n_max=int(enc.N_max),
        p_max=int(enc.P_max) if enc.P_max is not None else len(config_ports(config)),
        a_max=int(enc.A_max) if enc.A_max is not None else int(enc.N_max),
        k_max=int(enc.K_max),
        phi_hidden_dim=int(enc.phi_hidden_dim),
        phi_out_dim=int(enc.phi_out_dim) if enc.phi_out_dim is not None else int(enc.phi_hidden_dim),
        pooling=str(enc.pooling),
    )


def config_ports(config: SimulationConfig) -> List[Any]:
    """Return the flat list of configured regular ports (for ``p_max`` sizing)."""
    ports: List[Any] = []
    for station in config.port_station.stations:
        ports.extend(station.ports)
    return ports


def build_features(
    world: WorldState,
    side: str,
    library: List[Bundle],
    config: SimulationConfig,
    view: Optional[AgentView] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, Any]:
    """Build ``(raw_obs, cand_feats, cand_mask, dims)`` for one side (``03-03``).

    Uses only fields the side may see (its :class:`FeatureBlockSpec` / visibility).
    Empty item lists yield ``(0, F)`` blocks (Gotcha G12). The SPKLU booking block
    rides the area slot (the networks have only three set encoders).

    Args:
        world: The frozen world (the bridge holds the full ground-truth).
        side: ``"fleet"`` or ``"spklu"``.
        library: The candidate bundles (for ``cand_feats``).
        config: The simulation config.
        view: Optional precomputed :class:`AgentView`; built from ``world`` if
            ``None`` (using the configured visibility).

    Returns:
        ``(raw_obs, cand_feats, cand_mask, dims)`` where ``raw_obs`` is the block
        dict, ``cand_feats`` is ``(K, F_cand)``, ``cand_mask`` is ``(K,)``, and
        ``dims`` is the matching :class:`FeatureDims` (C7).

    Example:
        >>> from simulation.config.default import default_config
        >>> from simulation.world.routing import Router
        >>> w = WorldState(config=default_config(), router=Router(speed_kmh=30.0))
        >>> raw, cand, mask, dims = build_features(w, "fleet", [], default_config())
        >>> raw["taxi_feats"].shape[1] == dims.f_taxi
        True
    """

    if view is None:
        view = project_view(world, side, config.state_feature.visibility)
    block = _block_spec(config, side)

    taxi_feats, taxi_mask = _build_set_block(
        list(view.visible_taxis), block.taxi, _taxi_field, world
    )
    port_feats, port_mask = _build_set_block(
        list(view.visible_ports), block.port, _port_field, world
    )

    if side == "fleet":
        area_feats, area_mask = _build_set_block(
            list(view.visible_areas), block.area, _area_field, world
        )
    else:
        # SPKLU: bookings ride the area block.
        area_feats, area_mask = _build_set_block(
            list(view.visible_bookings), block.booking, _booking_field, world
        )

    scalar_feats = _scalar_block(block.scalar, view)

    raw_obs: Dict[str, np.ndarray] = {
        "taxi_feats": taxi_feats,
        "taxi_mask": taxi_mask,
        "port_feats": port_feats,
        "port_mask": port_mask,
        "area_feats": area_feats,
        "area_mask": area_mask,
        "scalar_feats": scalar_feats,
    }

    cand_feats, cand_mask = _build_cand_block(
        library, side, block.cand, world, int(config.state_feature.encoder_dims.K_max)
    )
    dims = feature_dims(config, side)
    return raw_obs, cand_feats, cand_mask, dims
