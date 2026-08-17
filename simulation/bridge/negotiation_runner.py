"""Configure both agents and run one bilateral negotiation (``05-01``/``05-02``/``05-03``).

:func:`run_negotiation` performs the two bridge responsibilities around a run:

1. **Inject world context (Contract C3).** For each side it calls
   ``agent.configure_negotiation(raw_obs, cand_feats, cand_mask, action_library,
   world_ufun, utility_min, utility_max)`` from the :class:`DomainSpec` — the agent
   has no world context until this happens. The fleet receives its own
   ``raw_obs`` / ``cand_feats`` / bounds, the SPKLU receives the SPKLU ones
   (asymmetric visibility).
2. **Run the mechanism.** It builds a
   :class:`~negotiation.environments.negmas_env.NegMASBilateralEnv` over the spec
   and calls ``run(agent_fleet, agent_spklu, first_actor="a",
   start_with="propose")`` so the fleet (added first) proposes first (``05-01``).

The returned deal is the agreed :class:`~simulation.bridge.action_library.Bundle`
(``action_library[deal_index]``) or ``None`` for no-deal, together with the run
``summary`` (carries ``deal_index`` for metrics).

Turn-order descriptor (LLR-M6.1)
--------------------------------
After ``env.run``, :func:`run_negotiation` injects two **additive** keys into the
summary — ``turn_order`` (ordered ``{"step", "actor", "kind"}`` descriptors) and
``max_steps`` — so the metrics layer (PKG-06) can reconstruct the propose/respond
interleave from the **two separate Episode payloads** without modifying the
negotiation data types (INV-2: ``Step`` / ``Episode`` / ``decode_action`` /
``BundleUFun`` are never touched here).

Runtime trace inspection (#C2, NegMAS 0.15.7)
---------------------------------------------
The concrete shape of ``summary["trace"]`` was verified live before choosing a
strategy:

* ``mechanism.trace`` is ``List[Tuple[negotiator_id, offer]]``;
  ``mechanism.full_trace`` is ``List[TraceElement]`` whose ``.negotiator`` field
  is the proposer id. Both list **one entry per offer/propose** in chronological
  order, and ``negotiator.id`` matches the trace ids.

Because the trace is reliable, the **default path is (a)**: derive the proposer
order from the trace (mapping fleet id -> ``"fleet"``, spklu id -> ``"spklu"``)
and expand it into the canonical SAO interleave (each propose after the first is
preceded by that side responding/rejecting the previous offer). When the trace is
absent or carries unmappable ids, the code falls back to **path (b)**: the pure
deterministic SAO pattern (fleet propose, spklu respond, spklu propose, fleet
respond, ...) bounded by ``max_steps``. See :func:`build_turn_order`.

The bridge never touches the mechanism internals (no agent / trainer changes); it
only assembles input and reads output (``05-02`` contract principle).

Example
-------
>>> hasattr(run_negotiation, "__call__")
True
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

from simulation.config.config import SimulationConfig
from simulation.bridge.action_library import Bundle

from negotiation.environments.negmas_env import DomainSpec, NegMASBilateralEnv


# Side labels carried into ``turn_order`` (must match the metrics layer / PKG-06).
SIDE_FLEET: str = "fleet"
SIDE_SPKLU: str = "spklu"

# Turn kinds (propose & respond are fully separate spaces — see CLAUDE.md 4.1).
KIND_PROPOSE: str = "propose"
KIND_RESPOND: str = "respond"


def configure_agents(
    domain_spec: DomainSpec, agent_fleet: Any, agent_spklu: Any
) -> None:
    """Inject the per-side world context into both agents (Contract C3).

    Must be called before :class:`NegMASBilateralEnv.run`. The fleet gets the
    fleet ufun / blocks / bounds, the SPKLU gets the SPKLU ones; the candidate
    masks come from ``domain_spec.meta``.

    Args:
        domain_spec: The bridge-built :class:`DomainSpec`.
        agent_fleet: The fleet-side negotiator.
        agent_spklu: The SPKLU-side negotiator.

    Example:
        >>> hasattr(configure_agents, "__call__")
        True
    """
    meta = domain_spec.meta or {}
    agent_fleet.configure_negotiation(
        raw_obs=domain_spec.raw_obs_fleet,
        cand_feats=domain_spec.cand_feats_fleet,
        cand_mask=meta.get("cand_mask_fleet"),
        action_library=domain_spec.action_library,
        world_ufun=domain_spec.ufun_fleet,
        utility_min=domain_spec.utility_min_fleet,
        utility_max=domain_spec.utility_max_fleet,
    )
    agent_spklu.configure_negotiation(
        raw_obs=domain_spec.raw_obs_spklu,
        cand_feats=domain_spec.cand_feats_spklu,
        cand_mask=meta.get("cand_mask_spklu"),
        action_library=domain_spec.action_library,
        world_ufun=domain_spec.ufun_spklu,
        utility_min=domain_spec.utility_min_spklu,
        utility_max=domain_spec.utility_max_spklu,
    )


def _proposer_sequence_from_trace(
    trace: Any, fleet_id: Optional[str], spklu_id: Optional[str]
) -> Optional[List[str]]:
    """Return the ordered list of proposer sides from a NegMAS trace (path a).

    Runtime-verified shape (NegMAS 0.15.7, see module note):

    * ``mechanism.trace`` is ``List[Tuple[negotiator_id, offer]]``.
    * ``mechanism.full_trace`` is ``List[TraceElement]`` with a ``.negotiator``
      field (the proposer id) and an ``.offer`` field.

    Both list one entry **per offer/propose** in chronological order. This helper
    maps each entry's negotiator id to its side (``fleet`` / ``spklu``) using the
    ids captured from the live negotiators. It returns ``None`` (so the caller
    falls back to the deterministic path b) when the trace is absent, empty, of an
    unexpected shape, or carries ids that do not map to a known side.

    Args:
        trace: ``summary["trace"]`` produced by the env (may be ``None``).
        fleet_id: ``agent_fleet.id`` (proposer id of the fleet side).
        spklu_id: ``agent_spklu.id`` (proposer id of the SPKLU side).

    Returns:
        Ordered proposer sides, e.g. ``["fleet", "spklu", "fleet"]``, or ``None``.

    Example:
        >>> tr = [("f1", (4,)), ("s1", (0,))]
        >>> _proposer_sequence_from_trace(tr, "f1", "s1")
        ['fleet', 'spklu']
        >>> _proposer_sequence_from_trace(None, "f1", "s1") is None
        True
    """
    if not trace:
        return None
    try:
        iterator = list(trace)
    except TypeError:
        return None

    id_to_side: Dict[str, str] = {}
    if fleet_id is not None:
        id_to_side[str(fleet_id)] = SIDE_FLEET
    if spklu_id is not None:
        id_to_side[str(spklu_id)] = SIDE_SPKLU

    sequence: List[str] = []
    for entry in iterator:
        negotiator_id = _trace_entry_negotiator_id(entry)
        if negotiator_id is None:
            return None
        side = id_to_side.get(str(negotiator_id))
        if side is None:
            # Id we cannot map to a side -> trace not reliable for our mapping.
            return None
        sequence.append(side)
    return sequence or None


def _trace_entry_negotiator_id(entry: Any) -> Optional[str]:
    """Extract the proposer id from one trace entry (tuple or ``TraceElement``).

    Handles both runtime-verified trace flavours:

    * ``("negotiator_id", offer)`` tuple (``mechanism.trace``).
    * a ``TraceElement`` with a ``.negotiator`` attribute (``full_trace``).

    Args:
        entry: One element of a NegMAS trace.

    Returns:
        The proposer id string, or ``None`` if it cannot be read.

    Example:
        >>> _trace_entry_negotiator_id(("f1", (4,)))
        'f1'
        >>> _trace_entry_negotiator_id(123) is None
        True
    """
    negotiator = getattr(entry, "negotiator", None)
    if negotiator is not None:
        return str(negotiator)
    if isinstance(entry, (tuple, list)) and len(entry) >= 1:
        return str(entry[0])
    return None


def _turn_order_from_proposers(proposers: List[str]) -> List[Dict[str, Any]]:
    """Expand a proposer sequence into the canonical SAO propose/respond turns.

    The NegMAS trace records **offers (proposes)** only; in the SAO protocol every
    offer after the first is preceded by the proposing side *responding* (reject)
    to the previous offer. This expands the propose sequence into the canonical
    interleave required by LLR-M6.1::

        A propose -> B respond -> B propose -> A respond -> A propose -> ...

    Concretely, for proposers ``[fleet, spklu, fleet]`` the turns are::

        fleet propose, spklu respond, spklu propose, fleet respond, fleet propose

    i.e. each propose (except the first) is preceded by a respond by the same side
    that is about to propose. The ``step`` index increases by one per turn so the
    metrics layer (PKG-06) can zip these descriptors against the steps stored in
    the two separate Episode payloads (propose memory / respond memory).

    Args:
        proposers: Ordered proposer sides (e.g. ``["fleet", "spklu"]``).

    Returns:
        Ordered turn descriptors ``{"step", "actor", "kind"}``.

    Example:
        >>> _turn_order_from_proposers(["fleet", "spklu"])
        [{'step': 0, 'actor': 'fleet', 'kind': 'propose'}, \
{'step': 1, 'actor': 'spklu', 'kind': 'respond'}, \
{'step': 2, 'actor': 'spklu', 'kind': 'propose'}]
    """
    turn_order: List[Dict[str, Any]] = []
    step = 0
    for position, side in enumerate(proposers):
        if position > 0:
            # The side about to propose first rejects (responds to) the previous
            # offer before placing its own counter-offer.
            turn_order.append(
                {"step": step, "actor": side, "kind": KIND_RESPOND}
            )
            step += 1
        turn_order.append({"step": step, "actor": side, "kind": KIND_PROPOSE})
        step += 1
    return turn_order


def _fallback_turn_order(
    agreement: Any, max_steps: int
) -> List[Dict[str, Any]]:
    """Deterministic SAO interleave when the trace is unusable (path b).

    Reconstructs the canonical bilateral SAO pattern with the fleet (``agent_a``)
    proposing first (project convention ``first_actor="a"``,
    ``start_with="propose"``)::

        step 0: fleet propose
        step 1: spklu respond
        step 2: spklu propose
        step 3: fleet respond
        step 4: fleet propose
        ...

    The sequence runs until ``max_steps`` turns are produced, or stops one turn
    earlier on a ``respond`` when an agreement exists (the accepting respond is the
    terminal turn). This is the minimum contract that must work even with no trace.

    Args:
        agreement: The raw NegMAS agreement (``None`` for no-deal).
        max_steps: ``config.negotiation.max_steps`` (upper bound on turns).

    Returns:
        Ordered turn descriptors ``{"step", "actor", "kind"}``.

    Example:
        >>> _fallback_turn_order(None, 3)
        [{'step': 0, 'actor': 'fleet', 'kind': 'propose'}, \
{'step': 1, 'actor': 'spklu', 'kind': 'respond'}, \
{'step': 2, 'actor': 'spklu', 'kind': 'propose'}]
        >>> _fallback_turn_order((1,), 4)[-1]
        {'step': 1, 'actor': 'spklu', 'kind': 'respond'}
    """
    # Canonical cycle starting with the fleet proposer.
    cycle = (
        (SIDE_FLEET, KIND_PROPOSE),
        (SIDE_SPKLU, KIND_RESPOND),
        (SIDE_SPKLU, KIND_PROPOSE),
        (SIDE_FLEET, KIND_RESPOND),
    )
    has_deal = agreement is not None
    turn_order: List[Dict[str, Any]] = []
    limit = max(int(max_steps), 0)
    for step in range(limit):
        actor, kind = cycle[step % len(cycle)]
        # On a deal, the accepting respond terminates the negotiation.
        if has_deal and kind == KIND_RESPOND and step > 0:
            turn_order.append({"step": step, "actor": actor, "kind": kind})
            break
        turn_order.append({"step": step, "actor": actor, "kind": kind})
    return turn_order


def build_turn_order(
    summary: Dict[str, Any],
    agent_fleet: Any,
    agent_spklu: Any,
    max_steps: int,
) -> List[Dict[str, Any]]:
    """Build the ``turn_order`` descriptor for one run (LLR-M6.1).

    Prefers path **a** (derive proposer order from ``summary["trace"]`` and expand
    into the canonical propose/respond interleave); falls back to path **b** (pure
    deterministic SAO interleave) when the trace is missing or unreliable. The
    result is a *descriptor only* — it carries no per-turn utilities (those are
    read by the metrics layer from the Episode payloads).

    Runtime decision (#C2): NegMAS 0.15.7 was inspected live; ``mechanism.trace``
    reliably lists one ``(negotiator_id, offer)`` per propose in order, and
    ``agent.id`` matches the trace ids, so path a is normally taken.

    Args:
        summary: The env run summary (carries ``trace`` and ``agreement``).
        agent_fleet: The fleet negotiator (its ``.id`` maps to ``"fleet"``).
        agent_spklu: The SPKLU negotiator (its ``.id`` maps to ``"spklu"``).
        max_steps: ``config.negotiation.max_steps`` (fallback bound).

    Returns:
        Ordered turn descriptors ``{"step", "actor", "kind"}``.

    Example:
        >>> summ = {"trace": [("f1", (2,)), ("s1", (0,))], "agreement": (0,)}
        >>> class _A: pass
        >>> f = _A(); f.id = "f1"; s = _A(); s.id = "s1"
        >>> build_turn_order(summ, f, s, 8)[0]
        {'step': 0, 'actor': 'fleet', 'kind': 'propose'}
    """
    proposers = _proposer_sequence_from_trace(
        summary.get("trace"),
        getattr(agent_fleet, "id", None),
        getattr(agent_spklu, "id", None),
    )
    if proposers is not None:
        # Path a: trace is reliable.
        return _turn_order_from_proposers(proposers)
    # Path b: deterministic fallback.
    return _fallback_turn_order(summary.get("agreement"), max_steps)


def run_negotiation(
    domain_spec: DomainSpec,
    agent_fleet: Any,
    agent_spklu: Any,
    config: SimulationConfig,
) -> Tuple[Optional[Bundle], Dict[str, Any]]:
    """Configure both agents (C3) then run the SAO mechanism (``05-02``/``05-03``).

    The fleet is the first proposer (added first, ``first_actor="a"``,
    ``start_with="propose"``); the SPKLU responds / counters from the same neutral
    action library, alternating until accept or ``max_steps`` (``05-01``).

    Args:
        domain_spec: The bridge-built :class:`DomainSpec`.
        agent_fleet: The fleet-side negotiator (``agent_a``).
        agent_spklu: The SPKLU-side negotiator (``agent_b``).
        config: The simulation config (for ``max_steps``).

    After the run, two **additive** keys are injected into ``summary`` for the
    metrics layer (LLR-M6.1); every pre-existing key (``agreement``, ``deal_index``,
    ``trace``, ...) is preserved and the ``deal`` return value is unchanged:

    * ``summary["turn_order"]``: ``List[Dict[str, Any]]`` — ordered turns, each
      ``{"step": int, "actor": "fleet"|"spklu", "kind": "propose"|"respond"}``,
      letting PKG-06 reconstruct the propose/respond interleave from the two
      separate Episode payloads (without touching ``Step`` / ``Episode``).
    * ``summary["max_steps"]``: ``int`` — ``config.negotiation.max_steps``.

    Args:
        domain_spec: The bridge-built :class:`DomainSpec`.
        agent_fleet: The fleet-side negotiator (``agent_a``).
        agent_spklu: The SPKLU-side negotiator (``agent_b``).
        config: The simulation config (for ``max_steps``).

    Returns:
        ``(deal_bundle | None, summary)`` — the agreed :class:`Bundle`
        (``action_library[deal_index]``) or ``None``, plus the run summary dict
        (carries ``deal_index`` plus the new ``turn_order`` / ``max_steps``).

    Example:
        >>> hasattr(run_negotiation, "__call__")
        True
    """
    configure_agents(domain_spec, agent_fleet, agent_spklu)

    max_steps = int(config.negotiation.max_steps)
    env = NegMASBilateralEnv(domain_spec, max_steps=max_steps)
    deal_index, summary = env.run(
        agent_fleet,
        agent_spklu,
        first_actor="a",
        start_with="propose",
    )

    # LLR-M6.1: build the turn-order descriptor and attach it additively.
    summary["turn_order"] = build_turn_order(
        summary, agent_fleet, agent_spklu, max_steps
    )
    summary["max_steps"] = max_steps

    deal: Optional[Bundle] = None
    if deal_index is not None and 0 <= deal_index < len(domain_spec.action_library):
        deal = domain_spec.action_library[deal_index]
    elif deal_index is not None:
        warnings.warn(
            f"run_negotiation: deal_index {deal_index} out of range "
            f"[0, {len(domain_spec.action_library)}); treating as no-deal. "
            "Possible cause: stale NegMAS proposal cache (see LLR-1.1).",
            RuntimeWarning,
            stacklevel=2,
        )
    return deal, summary
