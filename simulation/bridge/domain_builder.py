"""Build the negotiation :class:`DomainSpec` + reward-scaling bounds (``05-01``).

This is where the frozen world, the heuristic ``action_library``, the two intrinsic
world ufuns, and the per-side feature blocks come together into the single
:class:`~negotiation.environments.negmas_env.DomainSpec` the mechanism / agents
consume.

Two normalizations are kept strictly separate (Kontrak C6 / Gotcha G10):

* the ufuns are already **intrinsic** (``REF_*``, built in
  :mod:`simulation.bridge.ufun_builder`) — they are passed through untouched;
* :func:`compute_utility_bounds` derives ``(utility_min, utility_max)`` as the
  min/max of a side's world ufun **over the action library** for this negotiation,
  and these go on the :class:`DomainSpec` solely to scale RL reward. They are NOT
  used for the intrinsic normalization.

The outcome space is the bundle enumeration (one ``bundle_index`` issue with
``values = len(action_library)``); ``decode(idx) == action_library[idx]`` (Gotcha
G1). The per-side ``raw_obs`` / ``cand_feats`` blocks are attached so the bridge
can configure both agents before the run (Contract C3).

Example
-------
>>> from simulation.config.default import default_config
>>> from simulation.world.routing import Router
>>> from simulation.world.world_state import WorldState
>>> cfg = default_config()
>>> w = WorldState(config=cfg, router=Router(speed_kmh=30.0))
>>> spec, fdims, sdims = build_domain_spec(w, cfg)
>>> spec.action_library[-1].is_all_unassigned()
True
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

from simulation.config.config import SimulationConfig
from simulation.world.entities import Taxi
from simulation.world.world_state import WorldState
from simulation.bridge.action_library import Bundle, build_action_library
from simulation.bridge.feature_builder import build_features, feature_dims
from simulation.bridge.snapshot import collect_charge_candidates, project_view
from simulation.bridge.ufun_builder import build_ufun_fleet, build_ufun_spklu

from negotiation.environments.negmas_env import DomainSpec, build_bundle_issues


def compute_utility_bounds(
    library: List[Bundle], side_ufun: Callable[[Bundle], float]
) -> Tuple[float, float]:
    """Return ``(utility_min, utility_max)`` of a side's ufun over the library (C6).

    This is the **reward-scaling** normalization only (Gotcha G10): the min and max
    intrinsic utility a side assigns to any bundle in *this* negotiation's action
    library. It is never used for the intrinsic ``REF_*`` normalization. Because the
    all-UNASSIGNED bundle is always present, the bounds are always defined.

    Args:
        library: The candidate bundles for this negotiation.
        side_ufun: The side's intrinsic world ufun callable.

    Returns:
        ``(umin, umax)``; degenerate libraries collapse to ``(u, u)``.

    Example:
        >>> from simulation.bridge.action_library import Bundle
        >>> lib = [Bundle(()), Bundle(())]
        >>> compute_utility_bounds(lib, lambda b: 0.5)
        (0.5, 0.5)
    """
    if not library:
        return 0.0, 1.0
    values = [float(side_ufun(bundle)) for bundle in library]
    return min(values), max(values)


def build_domain_spec(
    world: WorldState,
    config: SimulationConfig,
    candidates: Optional[List[Taxi]] = None,
) -> Tuple[DomainSpec, Any, Any]:
    """Build the :class:`DomainSpec` for one negotiation (``05-01``).

    Collects candidates (unless given), builds the action library, the two
    intrinsic ufuns, the per-side feature blocks, and the per-side reward-scaling
    bounds, then assembles the :class:`DomainSpec`.

    Args:
        world: The frozen ground-truth world.
        config: The simulation config.
        candidates: Optional pre-collected candidate taxis. When ``None`` they
            are collected here (every idle, stuck, port-reachable taxi).

    Returns:
        ``(domain_spec, dims_fleet, dims_spklu)`` — the spec plus each side's
        :class:`FeatureDims` (Contract C7, needed by the agent factory).

    Example:
        >>> from simulation.config.default import default_config
        >>> from simulation.world.routing import Router
        >>> w = WorldState(config=default_config(), router=Router(speed_kmh=30.0))
        >>> spec, df, ds = build_domain_spec(w, default_config())
        >>> len(spec.issues)
        1
    """

    if candidates is None:
        candidates = collect_charge_candidates(world)

    library = build_action_library(world, candidates, config)

    ufun_fleet = build_ufun_fleet(world)
    ufun_spklu = build_ufun_spklu(world)

    view_fleet = project_view(world, "fleet", config.state_feature.visibility)
    view_spklu = project_view(world, "spklu", config.state_feature.visibility)
    raw_fleet, cand_fleet, mask_fleet, dims_fleet = build_features(
        world, "fleet", library, config, view=view_fleet
    )
    raw_spklu, cand_spklu, mask_spklu, dims_spklu = build_features(
        world, "spklu", library, config, view=view_spklu
    )

    umin_fleet, umax_fleet = compute_utility_bounds(library, ufun_fleet)
    umin_spklu, umax_spklu = compute_utility_bounds(library, ufun_spklu)

    spec = DomainSpec(
        # An empty library (no candidate can be placed) yields an empty issue
        # list; the caller (scheduler) detects this and skips the negotiation.
        issues=build_bundle_issues(len(library)) if library else [],
        action_library=library,
        ufun_fleet=ufun_fleet,
        ufun_spklu=ufun_spklu,
        utility_min_fleet=umin_fleet,
        utility_max_fleet=umax_fleet,
        utility_min_spklu=umin_spklu,
        utility_max_spklu=umax_spklu,
        raw_obs_fleet=raw_fleet,
        raw_obs_spklu=raw_spklu,
        cand_feats_fleet=cand_fleet,
        cand_feats_spklu=cand_spklu,
        domain_name="ev_charging_bundle",
        meta={
            "candidate_taxi_ids": [t.id for t in candidates],
            "cand_mask_fleet": mask_fleet,
            "cand_mask_spklu": mask_spklu,
            "k_size": len(library),
        },
    )
    return spec, dims_fleet, dims_spklu
