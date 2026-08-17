"""Rule-based baseline negotiator (AS-IS heuristic).

A non-learning opponent / baseline: a time-dependent aspiration curve drives both
proposing (offer the library bundle whose utility is closest to the current
aspiration) and responding (accept when the offer's normalized utility meets the
aspiration, else reject). Propose and respond stay fully separated (memory /
payloads) per Invariant #1, even though there is no network.

Adapted to the new architecture: outcomes are NegMAS bundle-index tuples
``(idx,)``; ``decode_action`` is index-direct (G1); the world utility comes from
the attached ``BundleUFun`` / injected ``world_ufun`` (``BaseNegotiator``).
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

try:
    from negmas import ResponseType
except Exception:  # pragma: no cover
    ResponseType = None  # type: ignore

from negotiation.agents.base_agent import RESPOND_ACCEPT, RESPOND_REJECT, BaseNegotiator
from negotiation.data_type.episode import Episode


class RuleBasedAgent(BaseNegotiator):
    """Aspiration-based heuristic baseline (no learning).

    Parameters
    ----------
    initial_threshold:
        Aspiration floor at ``t = 1`` (normalized utility).
    concession_power:
        Curvature of the aspiration decay ``1 - t**power * (1 - floor)``.

    Example:
        >>> RuleBasedAgent.__name__
        'RuleBasedAgent'
    """

    def __init__(
        self,
        name: str,
        ufun: Any,
        domain_spec: Any,
        trainers: Optional[Dict[str, Any]] = None,
        initial_threshold: float = 0.6,
        concession_power: float = 1.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, ufun=ufun, domain_spec=domain_spec, trainers=trainers, **kwargs)
        self.initial_threshold = float(initial_threshold)
        self.concession_power = float(concession_power)
        self.propose_action_dim = len(self.action_library)
        self.respond_action_dim = 2

    # ------------------------------------------------------------------
    # Aspiration helpers
    # ------------------------------------------------------------------

    def _denormalize(self, u_norm: float) -> float:
        """Inverse of min-max scaling (norm -> raw utility)."""
        return self.utility_min + u_norm * (self.utility_max - self.utility_min)

    def _aspiration(self, t: float) -> float:
        """Time-dependent aspiration level in ``[initial_threshold, 1]`` (AS-IS)."""
        t = min(max(float(t), 0.0), 1.0)
        return 1.0 - (t ** self.concession_power) * (1.0 - self.initial_threshold)

    def _closest_index_to_target(self, target_raw: float) -> int:
        """Return the library index whose utility is closest to ``target_raw``."""
        best_idx, best_gap = 0, float("inf")
        for idx in range(len(self.action_library)):
            u = self._offer_utility(self.make_offer(idx))
            gap = abs(u - target_raw)
            if gap < best_gap:
                best_gap = gap
                best_idx = idx
        return best_idx

    # ------------------------------------------------------------------
    # No network — sampling stubs / build_state pass-through
    # ------------------------------------------------------------------

    def attach_trainer(self, dict_of_trainer: Dict[str, Any]) -> None:
        """No-op: the rule agent does not learn (kept for interface symmetry)."""

    def build_state(self, state: Any, offer: Any = None) -> Any:
        """No network input is needed for the heuristic (returns ``None``)."""
        return None

    def sample_action_propose(self, obs: Any) -> Tuple[int, float, Optional[float], Any]:
        """Heuristic propose: closest-to-aspiration library index."""
        target = self._denormalize(self._aspiration(obs if isinstance(obs, float) else 0.0))
        return self._closest_index_to_target(target), 0.0, None, None

    def sample_action_respond(self, obs: Any) -> Tuple[int, float, Optional[float], Any]:
        """Heuristic respond: accept when normalized utility meets aspiration."""
        u_norm, t = (obs if isinstance(obs, tuple) else (0.0, 0.0))
        decision = RESPOND_ACCEPT if u_norm >= self._aspiration(t) else RESPOND_REJECT
        return decision, 0.0, None, None

    # ------------------------------------------------------------------
    # NegMAS protocol
    # ------------------------------------------------------------------

    def respond(self, state: Any, source: Any = None) -> Any:
        """Accept if the offer's normalized utility meets the aspiration, else reject."""
        self.flush_pending_propose(state)
        self.flush_pending_respond(state)

        offer = self._state_offer(state)
        if offer is None:
            return ResponseType.REJECT_OFFER

        current_u = self._offer_utility(offer)
        u_norm = self._normalize_utility(current_u)
        t = self._relative_time(state)
        accept = u_norm >= self._aspiration(t)
        action_id = RESPOND_ACCEPT if accept else RESPOND_REJECT

        self.current_state_data_respond = {
            "state": self.state_to_dict(state),
            "action": action_id,
            "utility": current_u,
            "value": None,
            "log_prob": None,
            "action_mask": None,
            "candidate_features": None,
        }
        return ResponseType.ACCEPT_OFFER if accept else ResponseType.REJECT_OFFER

    def propose(self, state: Any, dest: Any = None) -> Any:
        """Offer the library bundle closest to the current aspiration utility."""
        t = self._relative_time(state)
        target_raw = self._denormalize(self._aspiration(t))
        action_id = self._closest_index_to_target(target_raw)
        offer = self.make_offer(action_id)
        self.current_state_data_propose = {
            "state": self.state_to_dict(state),
            "action": action_id,
            "utility": self._offer_utility(offer),
            "value": None,
            "log_prob": None,
            "action_mask": None,
            "candidate_features": None,
        }
        return offer

    # ------------------------------------------------------------------
    # Episode payloads (kept; baseline usually has no trainer)
    # ------------------------------------------------------------------

    def _build_payload(self, memory: List[Any], agreement: Any, phase: str) -> Episode:
        """Build a finalized :class:`Episode` from a phase memory."""
        steps = deepcopy(memory)
        episode = Episode(
            agent_name=self.name,
            phase=phase,
            terminal_utility=self._offer_utility(agreement) if agreement is not None else -1.0,
            agreement=agreement,
            steps=steps,
        )
        return episode.finalize()

    def build_episode_payload_propose(self, agreement: Any) -> Episode:
        """Build the propose-phase :class:`Episode`."""
        return self._build_payload(self.memory_propose, agreement, "propose")

    def build_episode_payload_respond(self, agreement: Any) -> Episode:
        """Build the respond-phase :class:`Episode`."""
        return self._build_payload(self.memory_respond, agreement, "respond")
