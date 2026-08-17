"""RandomIOPAgent — random baseline + imagined-opponent stand-in (AS-IS).

A trainer-free baseline that samples uniformly random propose / respond actions.
It is also used as the *imagined opponent* inside MBOM recursive imagination
(:mod:`negotiation.agents.mbom.rollout_engine`), where it negotiates on the
opponent's estimated utility (a ``BundleUFun`` wrapping ``opponent_util_fn``).

Propose and respond stay fully separated (memory / payloads, Invariant #1).
Outcomes are bundle-index tuples ``(idx,)``; ``decode_action`` is index-direct.
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from negmas import ResponseType
except Exception:  # pragma: no cover
    ResponseType = None  # type: ignore

from negotiation.agents.base_agent import RESPOND_ACCEPT, BaseNegotiator
from negotiation.data_type.episode import Episode


class RandomIOPAgent(BaseNegotiator):
    """Uniformly-random baseline negotiator (no learning).

    Example:
        >>> RandomIOPAgent.__name__
        'RandomIOPAgent'
    """

    def __init__(
        self,
        name: str,
        ufun: Any,
        domain_spec: Any,
        trainers: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, ufun=ufun, domain_spec=domain_spec, trainers=trainers, **kwargs)
        self.propose_action_dim = max(1, len(self.action_library))
        self.respond_action_dim = 2

    # ------------------------------------------------------------------
    # Sampling (random; obs ignored)
    # ------------------------------------------------------------------

    def attach_trainer(self, dict_of_trainer: Dict[str, Any]) -> None:
        """No-op: the random baseline does not learn."""

    def build_state(self, state: Any, offer: Any = None) -> Any:
        """No network input needed (returns ``None``)."""
        return None

    def sample_action_propose(self, obs: Any = None) -> Tuple[int, float, Optional[float], Any]:
        """Random propose: a uniform index over the (per-negotiation) library.

        The action space is the *current* ``action_library`` injected via C3
        (``configure_negotiation``), not the construction-time size — so the
        sampled index always points into the real library (G1 index-direct).
        """
        n = max(1, len(self.action_library)) if self.action_library else self.propose_action_dim
        action_id = int(np.random.randint(0, n))
        log_prob = float(-np.log(n))
        return action_id, log_prob, None, None

    def sample_action_respond(self, obs: Any = None) -> Tuple[int, float, Optional[float], Any]:
        """Random respond: uniform over ``{0 accept, 1 reject}``."""
        action_id = int(np.random.randint(0, self.respond_action_dim))
        log_prob = float(-np.log(self.respond_action_dim))
        return action_id, log_prob, None, None

    def _mask_logits(self, logits: Any, allow_accept: bool) -> Any:
        """No discrete logits used (interface stub)."""
        return logits

    # ------------------------------------------------------------------
    # MBOM-opponent helper (used when this agent is the imagined opponent)
    # ------------------------------------------------------------------

    def _iop_offer_to_propose_idx(self, offer: Any) -> int:
        """Map an offer to a propose index (alias of ``_offer_to_action_idx``)."""
        return self._offer_to_action_idx(offer)

    def _state_basis_for_iop(self, state: Any, offer: Any = None) -> np.ndarray:
        """Minimal flat state basis (kept for rollout value bootstrap symmetry)."""
        relative_time = self._relative_time(state)
        off = offer if offer is not None else self._state_offer(state)
        u_norm = self._normalize_utility(self._offer_utility(off))
        has_offer = 1.0 if off is not None else 0.0
        return np.array([u_norm, relative_time, has_offer], dtype=np.float32)

    # ------------------------------------------------------------------
    # NegMAS protocol
    # ------------------------------------------------------------------

    def respond(self, state: Any, source: Any = None) -> Any:
        """Random accept / reject; record a pending respond step."""
        self.flush_pending_propose(state)
        self.flush_pending_respond(state)

        offer = self._state_offer(state)
        if offer is None:
            return ResponseType.REJECT_OFFER

        action_id, log_prob, value, _ = self.sample_action_respond()
        current_u = self._offer_utility(offer)
        self.current_state_data_respond = {
            "state": self.state_to_dict(state),
            "action": action_id,
            "utility": current_u,
            "value": value,
            "log_prob": log_prob,
            "action_mask": None,
            "candidate_features": None,
        }
        if action_id == RESPOND_ACCEPT:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def propose(self, state: Any, dest: Any = None) -> Any:
        """Random propose; return the NegMAS outcome ``(idx,)``."""
        action_id, log_prob, value, _ = self.sample_action_propose()
        offer = self.make_offer(action_id)
        current_u = self._offer_utility(offer)
        self.current_state_data_propose = {
            "state": self.state_to_dict(state),
            "action": action_id,
            "utility": current_u,
            "value": value,
            "log_prob": log_prob,
            "action_mask": None,
            "candidate_features": None,
        }
        return offer

    # ------------------------------------------------------------------
    # Episode payloads
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
