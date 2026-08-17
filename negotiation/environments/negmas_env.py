"""NegMAS SAO mechanism wiring for bundle-index negotiations (``02-01`` / ``05-01``).

This module ports the AS-IS SAO mechanism / bilateral environment from the paper
code, with the only architectural change being that the negotiation **outcome
space is an enumeration of bundle indices** (one issue ``bundle_index`` with
``values = K``; an outcome is the tuple ``(idx,)``). The world-side utility is a
plain ``Bundle -> float`` callable produced by the bridge; it is wrapped into a
NegMAS :class:`~negmas.preferences.UtilityFunction` via :class:`BundleUFun`.

Key invariants (see ``91-PLAYBOOK`` contracts/gotchas):

* **C1 / G1 — index-direct decode.** ``decode_action(idx) == action_library[idx]``.
  There is no ``idx - 1`` offset; that pattern belonged to the *old* mechanism
  which merged ``accept`` and ``offer`` into one action space. The new mechanism
  keeps propose and respond fully separate, so a propose index is a pure library
  index.
* **C2 — utility wrapping.** ``BundleUFun((idx,)) == world_ufun(action_library[idx])``.
  ``world_ufun`` is the bridge-built callable; this module never builds the
  world ufun, it only adapts it to the NegMAS ``eval(outcome)`` interface.
* **C6 / G10 — ``utility_min`` / ``utility_max`` are reward scaling only.** They
  are stored on :class:`DomainSpec` per side and consumed by the reward shaper;
  they are NOT the intrinsic ``REF_*`` normalization.

The ``negotiation`` package only depends on torch / numpy / negmas and other
``negotiation`` modules — never on ``simulation.world`` (the bridge supplies
``action_library`` and ``world_ufun`` from the outside).
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from negmas import SAOMechanism
from negmas.outcomes import make_issue
from negmas.preferences import UtilityFunction
from negmas.sao.common import SAOState


# =============================================================================
# decode_action
# =============================================================================

def decode_action(action_library: Sequence[Any], idx: int) -> Any:
    """Return the bundle at ``idx`` (index-direct, never ``idx - 1``).

    This is the single source of truth for translating a NegMAS propose index
    into the chosen bundle. The new mechanism keeps propose and respond separate,
    so a propose index is a pure index into ``action_library`` (Gotcha G1 / C1).

    Example:
        >>> lib = ["bundle_a", "bundle_b", "bundle_c"]
        >>> decode_action(lib, 0)
        'bundle_a'
        >>> decode_action(lib, 2)
        'bundle_c'
    """
    return action_library[idx]


# =============================================================================
# BundleUFun
# =============================================================================

class BundleUFun(UtilityFunction):
    """NegMAS utility wrapper over a ``Bundle -> float`` world utility (C2).

    A NegMAS outcome here is the tuple ``(idx,)`` where ``idx`` indexes
    ``action_library``. ``eval`` decodes the index to a bundle and delegates to
    the bridge-built ``world_ufun``:

        ``BundleUFun((idx,)) == world_ufun(action_library[idx])``

    The wrapper does not build any utility itself; it only adapts a plain
    callable to NegMAS' ``eval(outcome)`` contract so the AS-IS mechanism can use
    it unchanged.

    Parameters
    ----------
    action_library:
        Ordered list of bundles; ``action_library[idx]`` is the bundle for
        outcome ``(idx,)``.
    world_ufun:
        Callable mapping a bundle to a float utility (built by the bridge).
    issues:
        The negotiation issues (a single enumerated ``bundle_index`` issue);
        forwarded to the NegMAS base class.
    reserved_value:
        Utility of the no-agreement outcome (forwarded to NegMAS).

    Example:
        >>> lib = ["x", "y"]
        >>> uf = BundleUFun(lib, world_ufun=lambda b: 1.0 if b == "y" else 0.2)
        >>> uf((0,))
        0.2
        >>> uf((1,))
        1.0
        >>> uf(None)            # no outcome -> reserved value
        0.0
    """

    def __init__(
        self,
        action_library: Sequence[Any],
        world_ufun: Callable[[Any], float],
        issues: Optional[Sequence[Any]] = None,
        reserved_value: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            issues=issues,
            reserved_value=reserved_value,
            **kwargs,
        )
        self._action_library = action_library
        self._world_ufun = world_ufun

    def eval(self, offer: Any) -> float:
        """Return the world utility of the bundle referenced by ``offer``.

        ``offer`` is a NegMAS outcome tuple ``(idx,)`` (or ``None`` for the
        no-agreement outcome, which returns the reserved value). The index is
        used directly: ``world_ufun(action_library[idx])``.

        Example:
            >>> uf = BundleUFun(["a", "b"], world_ufun=lambda b: len(b))
            >>> uf.eval((1,))
            1.0
        """
        if offer is None:
            return float(self.reserved_value)

        if isinstance(offer, dict):
            # Dict outcome (single-issue): take the only value.
            idx = int(next(iter(offer.values())))
        elif isinstance(offer, (tuple, list)):
            idx = int(offer[0])
        else:
            idx = int(offer)

        bundle = decode_action(self._action_library, idx)
        return float(self._world_ufun(bundle))


# =============================================================================
# DomainSpec
# =============================================================================

def build_bundle_issues(num_bundles: int) -> List[Any]:
    """Return the single enumerated ``bundle_index`` issue list for ``num_bundles``.

    The outcome space is one issue with ``values = num_bundles`` (values
    ``0..num_bundles-1``); an outcome is the tuple ``(idx,)``. This replaces the
    old Cartesian per-taxi enumeration (``03-04``).

    Example:
        >>> issues = build_bundle_issues(3)
        >>> len(issues)
        1
        >>> issues[0].name
        'bundle_index'
    """
    return [make_issue(values=num_bundles, name="bundle_index")]


@dataclass
class DomainSpec:
    """Bundle-enumeration negotiation domain produced by the bridge (``05-01``).

    Fields
    ------
    issues:
        The NegMAS issue list — a single enumerated ``bundle_index`` issue with
        ``values = len(action_library)``.
    action_library:
        Ordered list of candidate bundles; outcome ``(idx,)`` -> ``action_library[idx]``.
    ufun_fleet / ufun_spklu:
        Per-side world utilities, callables ``Bundle -> float`` (bridge-built).
        Wrapped into :class:`BundleUFun` by :meth:`bundle_ufun_fleet` /
        :meth:`bundle_ufun_spklu` before being attached to the mechanism.
    utility_min_fleet / utility_max_fleet:
        Per-negotiation utility bounds for the fleet side, used ONLY for reward
        scaling (C6 / G10), never for intrinsic ``REF_*`` normalization.
    utility_min_spklu / utility_max_spklu:
        Same, for the SPKLU side.
    raw_obs_fleet / raw_obs_spklu:
        The per-side ``RawObservation`` blocks (bridge-built feature blocks +
        masks); attached so agents can be configured before ``run``.
    cand_feats_fleet / cand_feats_spklu:
        Per-side per-candidate feature matrices ``(K, F_cand)`` summarizing each
        bundle from that side's viewpoint (``03-04`` / ``03-03``).
    domain_name:
        Identifier carried into the run ``summary`` (used by metrics / logging).
    meta:
        Free-form bookkeeping (e.g. candidate taxi ids).

    The ``negotiation`` layer treats bundles and ``world_ufun`` opaquely; their
    concrete types live in the bridge (``simulation.bridge``).

    Example:
        >>> lib = ["a", "b"]
        >>> spec = DomainSpec(
        ...     issues=build_bundle_issues(len(lib)),
        ...     action_library=lib,
        ...     ufun_fleet=lambda b: 1.0,
        ...     ufun_spklu=lambda b: 0.5,
        ...     utility_min_fleet=0.0, utility_max_fleet=1.0,
        ...     utility_min_spklu=0.0, utility_max_spklu=1.0,
        ... )
        >>> len(spec.action_library)
        2
        >>> spec.bundle_ufun_fleet().eval((0,))
        1.0
    """

    issues: List[Any]
    action_library: List[Any]
    ufun_fleet: Callable[[Any], float]
    ufun_spklu: Callable[[Any], float]
    utility_min_fleet: float = 0.0
    utility_max_fleet: float = 1.0
    utility_min_spklu: float = 0.0
    utility_max_spklu: float = 1.0
    raw_obs_fleet: Any = None
    raw_obs_spklu: Any = None
    cand_feats_fleet: Any = None
    cand_feats_spklu: Any = None
    domain_name: str = "ev_charging_bundle"
    meta: Dict[str, Any] = field(default_factory=dict)

    def decode(self, idx: int) -> Any:
        """Return ``action_library[idx]`` (index-direct, Gotcha G1).

        Example:
            >>> DomainSpec(issues=[], action_library=["a", "b"],
            ...            ufun_fleet=lambda b: 0.0,
            ...            ufun_spklu=lambda b: 0.0).decode(1)
            'b'
        """
        return decode_action(self.action_library, idx)

    def bundle_ufun_fleet(self, reserved_value: float = 0.0) -> BundleUFun:
        """Return the fleet world ufun wrapped as a :class:`BundleUFun` (C2)."""
        return BundleUFun(
            self.action_library,
            self.ufun_fleet,
            issues=self.issues,
            reserved_value=reserved_value,
        )

    def bundle_ufun_spklu(self, reserved_value: float = 0.0) -> BundleUFun:
        """Return the SPKLU world ufun wrapped as a :class:`BundleUFun` (C2)."""
        return BundleUFun(
            self.action_library,
            self.ufun_spklu,
            issues=self.issues,
            reserved_value=reserved_value,
        )


# =============================================================================
# CustomStartSAOMechanism
# =============================================================================

class CustomStartSAOMechanism(SAOMechanism):
    """SAO mechanism that controls which negotiator acts first / from which step.

    Ported AS-IS from the paper code (``02-01``). It stores start-step metadata in
    :meth:`on_negotiation_start` so that:

    * ``first_actor`` (the agent ``.add``-ed in slot ``"a"``) acts first; in this
      project the fleet is added first, so ``first_actor="a"`` makes the fleet the
      first proposer.
    * ``start_with`` selects whether ``first_actor`` proposes or responds first.
    * ``start_step`` lets a negotiation resume from a mid-point (used by MBOM
      recursive imagination), overriding the base step / relative-time.

    Example:
        >>> # Construction validated structurally; a full run needs negotiators.
        >>> CustomStartSAOMechanism.__name__
        'CustomStartSAOMechanism'
    """

    def __init__(
        self,
        *args: Any,
        initial_offer: Optional[Dict[str, Any]] = None,
        first_actor: str = "a",
        start_with: str = "propose",
        start_step: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if initial_offer is not None and "initial_state" not in kwargs:
            kwargs["initial_state"] = SAOState(current_offer=initial_offer)

        super().__init__(*args, **kwargs)

        self._custom_initial_offer = initial_offer
        self._custom_first_actor = first_actor
        self._custom_start_with = start_with
        # Mid-negotiation starting step (e.g. MBOM rollout continuation).
        self._custom_start_step = start_step

    def on_negotiation_start(self) -> bool:
        """Set the starting proposer / offer / step before the first move.

        Called by the base ``Mechanism.run()`` after all ``.add()`` calls and
        before the first step. Mirrors the AS-IS logic: the offered ``ordered``
        ordering puts ``first_actor`` first, and ``start_step`` (if given)
        overrides the forced ``step = 0``.
        """
        result = super().on_negotiation_start()

        first_idx = 0 if self._custom_first_actor == "a" else 1
        second_idx = 1 - first_idx
        n = len(self.negotiators)

        if self._custom_start_with == "propose":
            # first_actor proposes first; initial_offer (if any) is theirs.
            if self._custom_initial_offer is not None:
                proposer = self.negotiators[first_idx]
                self._current_state.current_offer = self._custom_initial_offer
                self._current_state.current_proposer = proposer.id
                self._current_proposer = proposer
        else:  # "respond"
            # first_actor responds to an initial_offer owned by second_actor.
            if self._custom_initial_offer is not None:
                proposer = self.negotiators[second_idx]
                self._current_state.current_offer = self._custom_initial_offer
                self._current_state.current_proposer = proposer.id
                self._current_proposer = proposer

        # ordered_indices[0] = (_last_checked_negotiator + 1) % n, so to make
        # first_idx act first set _last_checked_negotiator = (first_idx - 1) % n.
        self._last_checked_negotiator = (first_idx - 1) % n

        # Mid-negotiation start: the base step() forces step = 0 just before this
        # hook, so override here to resume from the requested step.
        if self._custom_start_step:
            self._current_state.step = self._custom_start_step
            self._current_state.relative_time = self._relative_time_for(
                self._internal_nmi
            )

        return result


# =============================================================================
# NegMASBilateralEnv
# =============================================================================

class NegMASBilateralEnv:
    """Run one bilateral negotiation over a bundle-enumeration :class:`DomainSpec`.

    Wraps :class:`CustomStartSAOMechanism`. ``run`` adds the two agents, runs the
    mechanism, and returns the agreement (as a **deal index** into
    ``action_library``, or ``None`` for no-deal) together with a ``summary`` dict.

    Conventions (``05-01``): the fleet is ``agent_a`` (added first), ``spklu`` is
    ``agent_b``; the default start is ``first_actor="a"``, ``start_with="propose"``
    so the fleet initiates.

    Example:
        >>> spec = DomainSpec(issues=[], action_library=["a"],
        ...                   ufun_fleet=lambda b: 0.0, ufun_spklu=lambda b: 0.0)
        >>> env = NegMASBilateralEnv(spec, max_steps=10)
        >>> env.max_steps
        10
    """

    def __init__(self, domain_spec: DomainSpec, max_steps: int = 20) -> None:
        self.domain_spec = domain_spec
        self.max_steps = max_steps

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_agent(self, agent: Any, skip_negotiation_state: bool = False) -> None:
        """Reset a negotiator's NegMAS-internal state (and optionally its memory).

        For active agents reused across episodes only the internal NegMAS state
        is reset (``skip_negotiation_state=True``); the opponent is always fully
        reset. Mirrors the AS-IS behaviour.

        Also resets the NegMAS proposal cache attributes
        (_SAONegotiator__my_last_proposal -> None,
        _SAONegotiator__my_last_proposal_time -> -1) unconditionally, because
        these are mechanism-level (not episode-level) state and a cache-hit
        (time == step) would suppress RLNegotiator.propose() on every period
        after the first.
        """
        for attr in (
            "_nmi",
            "_mechanism",
            "_negotiation_result",
            "_current_offer",
            "_last_offer",
            "_offered",
            "__negotiator_id__",
            "_Negotiator__parent",
        ):
            if hasattr(agent, attr):
                try:
                    setattr(agent, attr, None)
                except Exception:
                    pass

        # Reset NegMAS proposal cache (mencegah cache-hit antar episode).
        # _SAONegotiator__my_last_proposal harus None (tidak ada offer),
        # _SAONegotiator__my_last_proposal_time harus -1 (bukan 0 = step awal).
        for attr, val in (
            ("_SAONegotiator__my_last_proposal", None),
            ("_SAONegotiator__my_last_proposal_time", -1),
        ):
            if hasattr(agent, attr):
                try:
                    setattr(agent, attr, val)
                except Exception:
                    pass

        if not skip_negotiation_state:
            for attr, default in (
                ("memory_propose", []),
                ("memory_respond", []),
                ("current_state_data_propose", None),
                ("current_state_data_respond", None),
                ("prev_offer_utility", None),
                ("last_received_offer", None),
                ("last_proposed_offer", None),
            ):
                if hasattr(agent, attr):
                    try:
                        setattr(agent, attr, list(default) if isinstance(default, list) else default)
                    except Exception:
                        pass

    @staticmethod
    def _extract_agreement(mechanism: Any, returned_state: Any) -> Any:
        """Return the raw NegMAS agreement outcome (tuple ``(idx,)`` / dict / None)."""
        mech_agreement = getattr(mechanism, "agreement", None)
        if mech_agreement is not None:
            return mech_agreement
        if hasattr(returned_state, "agreement"):
            return getattr(returned_state, "agreement")
        if isinstance(returned_state, dict):
            return returned_state
        return None

    @staticmethod
    def _extract_trace(mechanism: Any) -> Any:
        """Return the mechanism trace if available, else ``None``."""
        for attr in ("trace", "full_trace", "extended_trace"):
            if hasattr(mechanism, attr):
                return getattr(mechanism, attr)
        return None

    @staticmethod
    def _agreement_to_index(agreement: Any) -> Optional[int]:
        """Convert a NegMAS agreement outcome into a deal index (``None`` if no-deal).

        Accepts tuple ``(idx,)`` / list / dict (single ``bundle_index`` value) /
        scalar. Returns the integer library index or ``None``.

        Example:
            >>> NegMASBilateralEnv._agreement_to_index((3,))
            3
            >>> NegMASBilateralEnv._agreement_to_index(None) is None
            True
            >>> NegMASBilateralEnv._agreement_to_index({"bundle_index": 2})
            2
        """
        if agreement is None:
            return None
        if isinstance(agreement, dict):
            if not agreement:
                return None
            return int(next(iter(agreement.values())))
        if isinstance(agreement, (tuple, list)):
            if len(agreement) == 0:
                return None
            return int(agreement[0])
        return int(agreement)

    # ------------------------------------------------------------------
    # run()
    # ------------------------------------------------------------------

    def run(
        self,
        agent_a: Any,
        agent_b: Any,
        initial_offer: Optional[Dict[str, Any]] = None,
        first_actor: str = "a",
        start_with: str = "propose",
        skip_agent_reset: bool = False,
        start_step: Optional[int] = None,
    ) -> Tuple[Optional[int], Dict[str, Any]]:
        """Run one negotiation and return ``(deal_index | None, summary)``.

        ``agent_a`` is the fleet (added first → first proposer with the default
        ``first_actor="a"``, ``start_with="propose"``); ``agent_b`` is the SPKLU.
        ``deal_index`` is the index into ``action_library`` of the agreed bundle,
        or ``None`` for no-deal. ``summary`` keys:
        ``agreement, deal_index, trace, n_steps, domain_name, initial_offer,
        first_actor, start_with, start_step``.

        Example:
            >>> spec = DomainSpec(issues=[], action_library=["a"],
            ...                   ufun_fleet=lambda b: 0.0,
            ...                   ufun_spklu=lambda b: 0.0)
            >>> env = NegMASBilateralEnv(spec, max_steps=0)
            >>> deal, summ = env.run(object(), object(), start_step=0)
            >>> deal is None and summ["deal_index"] is None
            True
        """
        if first_actor not in ("a", "b"):
            raise ValueError("first_actor must be 'a' or 'b'.")
        if start_with not in ("propose", "respond"):
            raise ValueError("start_with must be 'propose' or 'respond'.")
        if start_with == "respond" and initial_offer is None:
            raise ValueError(
                "start_with='respond' requires a non-None initial_offer."
            )
        if initial_offer is not None and not isinstance(initial_offer, dict):
            raise TypeError(
                f"initial_offer must be a dict (action_library format), "
                f"not {type(initial_offer).__name__}."
            )

        # Reset agents: active agent only resets internal NegMAS state when
        # skip_agent_reset=True; the opponent is always fully reset.
        self._reset_agent(agent_a, skip_negotiation_state=skip_agent_reset)
        self._reset_agent(agent_b, skip_negotiation_state=False)

        # Panggil reset episode secara eksplisit sebagai guard tambahan,
        # tidak bergantung pada lifecycle NegMAS (LLR-EXTRA.2).
        # Redundant jika NegMAS memanggil on_negotiation_start dengan benar,
        # tapi idempotent dan menjamin memory_propose/respond fresh tiap episode.
        if hasattr(agent_a, "reset_episode"):
            agent_a.reset_episode()
        if hasattr(agent_b, "reset_episode"):
            agent_b.reset_episode()

        # No step left to simulate (start_step already >= max_steps) -> timeout
        # without running the mechanism (the first timeout check in
        # Mechanism.step() happens before start_step is applied).
        if (
            start_step is not None
            and self.max_steps is not None
            and start_step >= self.max_steps
        ):
            summary = self._build_summary(
                agreement=None,
                deal_index=None,
                trace=None,
                initial_offer=initial_offer,
                first_actor=first_actor,
                start_with=start_with,
                start_step=start_step,
            )
            return None, summary

        mechanism = CustomStartSAOMechanism(
            issues=self.domain_spec.issues,
            n_steps=self.max_steps,
            initial_offer=initial_offer,
            first_actor=first_actor,
            start_with=start_with,
            start_step=start_step,
        )

        mechanism.add(agent_a)
        mechanism.add(agent_b)

        returned = mechanism.run()
        agreement = self._extract_agreement(mechanism, returned)
        trace = self._extract_trace(mechanism)
        deal_index = self._agreement_to_index(agreement)

        summary = self._build_summary(
            agreement=agreement,
            deal_index=deal_index,
            trace=trace,
            initial_offer=initial_offer,
            first_actor=first_actor,
            start_with=start_with,
            start_step=start_step,
        )
        return deal_index, summary

    def _build_summary(
        self,
        agreement: Any,
        deal_index: Optional[int],
        trace: Any,
        initial_offer: Optional[Dict[str, Any]],
        first_actor: str,
        start_with: str,
        start_step: Optional[int],
    ) -> Dict[str, Any]:
        """Assemble the run ``summary`` dict (``02-01`` fields + ``deal_index``)."""
        return {
            "agreement": agreement,
            "deal_index": deal_index,
            "trace": trace,
            "n_steps": self.max_steps,
            "domain_name": self.domain_spec.domain_name,
            "initial_offer": initial_offer,
            "first_actor": first_actor,
            "start_with": start_with,
            "start_step": start_step,
        }
