"""RLNegotiator — the PRIMARY discrete RL agent (``02-03`` / ``03-01``).

A discrete actor-critic negotiator: a policy picks the action, a critic values
the state (for PPO). Propose and respond use **separate** trainers, models,
memories, and episode payloads (Invariant #1). The agent is mode-agnostic: the
same class works in ``PADDED`` or ``ENCODER`` mode — only the front-end input and
the discrete *propose* head change (Gotcha G13); that switch lives entirely in
the :class:`~negotiation.models.encoders.feature_encoder.DiscreteActorNet`
attached via the trainers. The respond head is always 2-way (``0 = accept``,
``1 = reject``).

Flow (AS-IS, ``02-03``), adapted to the new packed-obs / bundle-index outcome:

* ``propose(state)``: build packed obs -> ``sample_action_propose`` ->
  ``offer = (idx,)``; record a pending propose step (state = packed obs,
  action = idx, utility = U(offer), value, log_prob, + PADDED ``action_mask`` /
  ENCODER ``candidate_features``).
* ``respond(state, source)``: flush pending; if no offer -> REJECT; else build
  packed obs -> ``sample_action_respond`` -> record pending respond step;
  ``action_id == 0`` -> ACCEPT else REJECT.
* ``on_negotiation_end``: flush + build the two payloads -> ``propose_trainer`` /
  ``respond_trainer`` (handled by :class:`BaseNegotiator`).

Sampling re-uses the trainer's actor / critic so the action distribution at
update time (re-collated + re-forwarded) is identical (Contract C4 / ``03-05``).
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    from negmas import ResponseType
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    ResponseType = None  # type: ignore

from negotiation.agents.base_agent import (
    RESPOND_ACCEPT,
    BaseNegotiator,
)
from negotiation.agents.obs_utils import (
    action_mask_from_obs,
    candidate_features_from_obs,
    obs_to_batch,
)
from negotiation.data_type.episode import Episode

# Mode strings (kept consistent with feature_encoder).
MODE_PADDED: str = "PADDED"
MODE_ENCODER: str = "ENCODER"


class RLNegotiator(BaseNegotiator):
    """Discrete RL negotiator, two modes, fully-separated propose / respond.

    Parameters
    ----------
    name, ufun, domain_spec, trainers:
        See :class:`BaseNegotiator`.
    mode:
        ``"PADDED"`` or ``"ENCODER"`` — only changes how each :class:`Step`
        records its candidate-reconstruction plumbing (PADDED ``action_mask`` vs.
        ENCODER ``candidate_features``); the network's mode is set on the
        ``DiscreteActorNet`` itself (built by the factory).
    k_max:
        Maximum action-library size (PADDED ``Linear(K_max)`` width); used to size
        the recorded ``action_mask``.

    Example:
        >>> RLNegotiator.__name__
        'RLNegotiator'
    """

    def __init__(
        self,
        name: str,
        ufun: Any,
        domain_spec: Any,
        mode: str = MODE_PADDED,
        k_max: Optional[int] = None,
        trainers: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, ufun=ufun, domain_spec=domain_spec, trainers=trainers, **kwargs)
        if mode not in (MODE_PADDED, MODE_ENCODER):
            raise ValueError(f"mode must be {MODE_PADDED!r} or {MODE_ENCODER!r}, not {mode!r}")
        self.mode = mode
        # propose action dim = len(action_library) (NO accept in propose).
        self.propose_action_dim = len(self.action_library)
        # respond action dim = 2 (0 = accept, 1 = reject).
        self.respond_action_dim = 2
        self.k_max = int(k_max) if k_max is not None else len(self.action_library)
        self.device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"

    # ------------------------------------------------------------------
    # build_state — packed obs (mode front-end lives in the network)
    # ------------------------------------------------------------------

    def build_state(self, state: Any, offer: Any = None) -> Dict[str, Any]:
        """Build the packed observation dict for a step (Contract C4).

        Accepts a live SAOState (builds from the injected world context) or a
        pre-packed dict (pass-through, used when the trainer rebuilds states).

        Example:
            >>> hasattr(RLNegotiator, "build_state")
            True
        """
        if isinstance(state, dict) and ("scalar_feats" in state or "progress_feats" in state):
            return state
        return self.build_step_obs(state, offer)

    # ------------------------------------------------------------------
    # Sampling paths (separate; return (action_idx, log_prob, value, mask))
    # ------------------------------------------------------------------

    def _sample(
        self,
        obs: Dict[str, Any],
        trainer_key: str,
    ) -> Tuple[int, float, Optional[float]]:
        """Forward the phase actor / critic over a B=1 batch and sample.

        Returns ``(action_idx, log_prob, value)``; ``value`` is ``None`` when no
        critic is attached (PG).
        """
        trainer = self.trainers.get(trainer_key)
        if trainer is None:
            raise ValueError(
                f"No {trainer_key} attached. Use "
                f"agent.attach_trainer({{'{trainer_key}': trainer}})."
            )
        policy = trainer.models.get("policy", {}).get("model")
        if policy is None:
            raise ValueError(f"No 'policy' model in {trainer_key}.models.")
        value_entry = trainer.models.get("value")

        batch = obs_to_batch(obs, device=self.device)
        value: Optional[float] = None
        with torch.no_grad():
            logits = policy(batch)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            if value_entry is not None:
                value = float(value_entry["model"](batch).view(-1)[0].item())
            self._trace_actor_output = {
                "trainer_key": trainer_key,
                "logits": logits.view(-1),
                "probs": torch.softmax(logits, dim=-1).view(-1),
                "sampled_action": int(action.item()),
                "log_prob": float(log_prob.item()),
                "value": value,
            }
        return int(action.item()), float(log_prob.item()), value

    def sample_action_propose(self, obs: Dict[str, Any]) -> Tuple[int, float, Optional[float], Any]:
        """Propose-phase sampling path (``02-03``): index into ``action_library``.

        Returns ``(action_idx, log_prob, value, mask)`` where ``mask`` is the
        PADDED ``action_mask`` (length ``K_max``) or, in ENCODER mode, the
        per-candidate features (``(K, F_cand)``) — stored on the step for C4.

        Example:
            >>> hasattr(RLNegotiator, "sample_action_propose")
            True
        """
        action_idx, log_prob, value = self._sample(obs, "propose_trainer")
        mask = self._record_plumbing(obs)
        return action_idx, log_prob, value, mask

    def sample_action_respond(self, obs: Dict[str, Any]) -> Tuple[int, float, Optional[float], Any]:
        """Respond-phase sampling path (``02-03``): action in ``{0, 1}``.

        Returns ``(action_idx, log_prob, value, mask)``; the respond head is
        2-way and mode-independent, so ``mask`` is ``None`` for respond.
        """
        action_idx, log_prob, value = self._sample(obs, "respond_trainer")
        return action_idx, log_prob, value, None

    def _record_plumbing(self, obs: Dict[str, Any]) -> Any:
        """Return the per-step candidate reconstruction plumbing for the mode."""
        if self.mode == MODE_PADDED:
            return action_mask_from_obs(obs, self.k_max)
        return candidate_features_from_obs(obs)

    # ------------------------------------------------------------------
    # NegMAS protocol
    # ------------------------------------------------------------------

    def propose(self, state: Any, dest: Any = None) -> Any:
        """Sample a propose action and return the NegMAS outcome ``(idx,)``.

        Example:
            >>> hasattr(RLNegotiator, "propose")
            True
        """
        offer_opponent = self._state_offer(state)
        obs = self.build_state(state, offer_opponent)
        action_id, log_prob, value, mask = self.sample_action_propose(obs)

        offer = self.make_offer(action_id)
        current_u = self._offer_utility(offer)
        self.current_state_data_propose = {
            "state": obs,
            "action": action_id,
            "utility": current_u,
            "value": value,
            "log_prob": log_prob,
            "action_mask": mask if self.mode == MODE_PADDED else None,
            "candidate_features": mask if self.mode == MODE_ENCODER else None,
        }
        self._emit_trace_step(
            "propose", state, obs, action_id,
            extra={"offer": offer, "utility": current_u},
        )
        return offer

    def respond(self, state: Any, source: Any = None) -> Any:
        """Sample a respond action; return ACCEPT / REJECT (action space {0,1})."""
        self.flush_pending_propose(state)
        self.flush_pending_respond(state)

        offer = self._state_offer(state)
        if offer is None:
            return ResponseType.REJECT_OFFER

        obs = self.build_state(state, offer)
        action_id, log_prob, value, _ = self.sample_action_respond(obs)
        current_u = self._offer_utility(offer)
        self.current_state_data_respond = {
            "state": obs,
            "action": action_id,
            "utility": current_u,
            "value": value,
            "log_prob": log_prob,
            "action_mask": None,
            "candidate_features": None,
        }

        self._emit_trace_step(
            "respond", state, obs, action_id,
            extra={"decision": "accept" if action_id == RESPOND_ACCEPT else "reject",
                   "utility": current_u},
        )
        if action_id == RESPOND_ACCEPT:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    # ------------------------------------------------------------------
    # Episode payloads
    # ------------------------------------------------------------------

    def _build_payload(self, memory: List[Any], agreement: Any, phase: str) -> Episode:
        """Build an :class:`Episode` from a phase memory (finalized)."""
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
        """Build the propose-phase :class:`Episode` (``phase='propose'``)."""
        return self._build_payload(self.memory_propose, agreement, "propose")

    def build_episode_payload_respond(self, agreement: Any) -> Episode:
        """Build the respond-phase :class:`Episode` (``phase='respond'``)."""
        return self._build_payload(self.memory_respond, agreement, "respond")
