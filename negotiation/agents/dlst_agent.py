"""DLSTNegotiator — Deep Learning Strategy Templates (DLST-ANESIA), AS-IS (``02-05``).

A DDPG (continuous-action) negotiator whose actors output **strategy-template
parameters** ``(delta, c, p)`` rather than action indices; those are translated
into an offer / accept decision by the tactical strategy templates
(:class:`BiddingStrategy` / :class:`AcceptanceStrategy`). Three sub-decisions,
each its own DDPG actor-critic with its **own trainer / memory / payload**
(Invariant #1 + G13):

* ``threshold_trainer`` — predicts the dynamic utility threshold ``u_bar_t``.
* ``propose_trainer``  — bidding strategy ``(delta, c, p) -> f_b`` (offer).
* ``respond_trainer``  — acceptance strategy ``(delta, c, p) -> f_a`` (accept/reject).

**Head stays AS-IS**: because the bidding template always returns an exact member
of ``action_library`` (a library *index*), it is compatible with the bundle-index
outcome ``(idx,)`` without any head change. Only the *front-end input* would adapt
to the mode if dynamic world context were summarized into the state; the DLST
tactical state here is the fixed-dim tactical-feature vector exactly as in the
paper (it never uses per-candidate logits, so K is irrelevant — Gotcha G13). The
``mode`` argument is accepted for signature symmetry.

The opponent utility approximation (:class:`FrequencyOpponentModel`) is **shared
with the respond reward shaper** (``respond_trainer.reward_shaper.opponent_model``)
to compute ``r_acc`` (paper Eq. 10) and to drive the Pareto bidding tactic.
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
    from negmas import ResponseType
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    ResponseType = None  # type: ignore

from negotiation.agents.base_agent import BaseNegotiator
from negotiation.agents.dlst.acceptance_strategy import AcceptanceStrategy
from negotiation.agents.dlst.bidding_strategy import BiddingStrategy
from negotiation.agents.dlst.strategy_template import StrategyTemplateSpec
from negotiation.data_type.episode import Episode
from negotiation.data_type.step import Step


class DLSTNegotiator(BaseNegotiator):
    """DLST-ANESIA negotiator with three separate DDPG actor-critics (AS-IS).

    Parameters
    ----------
    bidding_spec, acceptance_spec:
        :class:`StrategyTemplateSpec` layouts for the bidding / acceptance actors.
    u_fixed:
        Fixed acceptance threshold for the ``u_fixed`` tactic.
    mode:
        Stored for signature symmetry (head is mode-independent, G13).

    State dimensions (fixed, paper Sec. 5): threshold = 6, respond = 10,
    propose = 10.

    Example:
        >>> DLSTNegotiator.STATE_DIM_THRESHOLD, DLSTNegotiator.STATE_DIM_PROPOSE
        (6, 10)
    """

    STATE_DIM_THRESHOLD: int = 6
    STATE_DIM_RESPOND: int = 10
    STATE_DIM_PROPOSE: int = 10

    def __init__(
        self,
        name: str,
        ufun: Any,
        domain_spec: Any,
        bidding_spec: StrategyTemplateSpec,
        acceptance_spec: StrategyTemplateSpec,
        u_fixed: float = 0.6,
        mode: str = "PADDED",
        trainers: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, ufun=ufun, domain_spec=domain_spec, trainers=trainers, **kwargs)
        self.mode = mode
        self.device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"

        self.bidding_spec = bidding_spec
        self.acceptance_spec = acceptance_spec

        # Third (DLST-only) memory + pending buffer: utility threshold.
        self.memory_threshold_utility: List[Step] = []
        self.current_state_data_threshold_utility: Optional[Dict[str, Any]] = None

        self.acceptance_strategy = AcceptanceStrategy(acceptance_spec, u_fixed=u_fixed)
        self.bidding_strategy = BiddingStrategy(
            spec=bidding_spec,
            action_library=self.action_library,
            issue_names=[self._domain_attr("issue_name", "bundle_index") or "bundle_index"],
            user_utility_fn=self._index_user_utility,
            opponent_utility_fn=self._index_opponent_utility,
            utility_min=self.utility_min,
            utility_max=self.utility_max,
        )

        self.opponent_util_history_norm: List[float] = []
        self.last_received_offer: Any = None
        self._last_threshold_value: Optional[float] = None

    # ------------------------------------------------------------------
    # configure_negotiation override — refresh strategy caches per negotiation
    # ------------------------------------------------------------------

    def configure_negotiation(self, raw_obs, cand_feats, cand_mask, action_library, world_ufun, utility_min, utility_max) -> None:
        """Inject world context (C3) and rebuild the bidding strategy caches."""
        super().configure_negotiation(raw_obs, cand_feats, cand_mask, action_library, world_ufun, utility_min, utility_max)
        self.bidding_strategy = BiddingStrategy(
            spec=self.bidding_spec,
            action_library=self.action_library,
            issue_names=[self._domain_attr("issue_name", "bundle_index") or "bundle_index"],
            user_utility_fn=self._index_user_utility,
            opponent_utility_fn=self._index_opponent_utility,
            utility_min=self.utility_min,
            utility_max=self.utility_max,
        )

    # ------------------------------------------------------------------
    # Trainer attach (three DDPG trainers)
    # ------------------------------------------------------------------

    def attach_trainer(self, dict_of_trainer: Dict[str, Any]) -> None:
        """Attach ``threshold_trainer`` / ``propose_trainer`` / ``respond_trainer``."""
        for key, trainer in dict_of_trainer.items():
            self.trainers[key] = trainer

    # ------------------------------------------------------------------
    # Opponent model (shared with the respond reward shaper)
    # ------------------------------------------------------------------

    def _opponent_model(self):
        """Return the opponent model held by the respond reward shaper (or None)."""
        respond_trainer = self.trainers.get("respond_trainer")
        if respond_trainer is None:
            return None
        rs = getattr(respond_trainer, "reward_shaper", None)
        if rs is None:
            return None
        return getattr(rs, "opponent_model", None)

    def _estimate_opponent_utility(self, offer: Any) -> float:
        """Estimate ``U_o(offer)`` via the shared opponent model (0.5 fallback)."""
        om = self._opponent_model()
        if om is None:
            return 0.5
        return float(om.estimate_utility(offer))

    def _update_opponent_model(self, offer: Any) -> None:
        """Record one real opponent offer into the shared opponent model."""
        om = self._opponent_model()
        if om is not None:
            om.update(offer)

    def _index_user_utility(self, idx: int) -> float:
        """Raw user utility of a candidate library index (for bidding)."""
        return self._offer_utility(self.make_offer(idx))

    def _index_opponent_utility(self, idx: int) -> float:
        """Estimated opponent utility of a candidate library index (for bidding)."""
        return self._estimate_opponent_utility(self.make_offer(idx))

    # ------------------------------------------------------------------
    # Tactical state builders (fixed-dim, AS-IS)
    # ------------------------------------------------------------------

    def build_state(self, state: Any, offer: Any = None, for_: str = "propose") -> np.ndarray:
        """Dispatch to the per-head tactical state builder (``for_``)."""
        if for_ == "utility_threshold":
            return self.build_state_utility_threshold(state, offer)
        if for_ == "respond":
            return self.build_state_respond(state, offer)
        return self.build_state_propose(state, offer)

    def _common_attrs(self, state: Any, offer: Any):
        """Compute the shared tactical attributes (AS-IS)."""
        relative_time = self._relative_time(state)
        if offer is None:
            offer = self._state_offer(state)
        u_norm = self._normalize_utility(self._offer_utility(offer))
        has_offer = 1.0 if offer is not None else 0.0
        hist = self.opponent_util_history_norm
        if len(hist) > 0:
            o_best = float(max(hist))
            o_avg = float(sum(hist) / len(hist))
            o_sd = float(np.std(np.asarray(hist, dtype=np.float64)))
        else:
            o_best = u_norm
            o_avg = u_norm
            o_sd = 0.0
        return relative_time, u_norm, o_best, o_avg, o_sd, has_offer

    def build_state_utility_threshold(self, state: Any, offer: Any = None) -> np.ndarray:
        """Threshold state (dim 6): ``[u_norm, t, O_best, O_avg, O_sd, has_offer]``."""
        if isinstance(state, (np.ndarray, list)):
            arr = np.asarray(state, dtype=np.float32)
            if arr.shape == (self.STATE_DIM_THRESHOLD,):
                return arr
        rt, u_norm, o_best, o_avg, o_sd, has_offer = self._common_attrs(state, offer)
        return np.array([u_norm, rt, o_best, o_avg, o_sd, has_offer], dtype=np.float32)

    def build_state_respond(self, state: Any, offer: Any = None) -> np.ndarray:
        """Acceptance state (dim 10): base + ``[u_fixed, u_bar, u_future, quantile]``."""
        if isinstance(state, (np.ndarray, list)):
            arr = np.asarray(state, dtype=np.float32)
            if arr.shape == (self.STATE_DIM_RESPOND,):
                return arr
        rt, u_norm, o_best, o_avg, o_sd, has_offer = self._common_attrs(state, offer)
        u_fixed = float(self.acceptance_strategy.u_fixed)
        u_bar = float(self._predict_threshold(state, offer))
        u_future = float(self._u_future_bid_norm())
        hist = self.opponent_util_history_norm
        quantile_val = float(np.quantile(np.asarray(hist), 0.5)) if len(hist) > 0 else u_norm
        base = [u_norm, rt, o_best, o_avg, o_sd, has_offer]
        extra = [u_fixed, u_bar, u_future, quantile_val]
        return np.array(base + extra, dtype=np.float32)

    def build_state_propose(self, state: Any, offer: Any = None) -> np.ndarray:
        """Bidding state (dim 10): base + ``[u_bar, U_boulware, U_pareto, U_bopp]``."""
        if isinstance(state, (np.ndarray, list)):
            arr = np.asarray(state, dtype=np.float32)
            if arr.shape == (self.STATE_DIM_PROPOSE,):
                return arr
        rt, u_norm, o_best, o_avg, o_sd, has_offer = self._common_attrs(state, offer)
        u_bar = float(self._predict_threshold(state, offer))
        ctx = {"u_bar": u_bar, "opponent_last_offer": self.last_received_offer,
               "opponent_last_idx": self._offer_to_action_idx(self.last_received_offer)
               if self.last_received_offer is not None else None}
        try:
            b_boul = self.bidding_strategy._tactic_boulware(rt, [0.1], ctx)
            b_par = self.bidding_strategy._tactic_pareto(rt, [-0.5, 0.8], ctx)
            b_opp = self.bidding_strategy._tactic_b_opp(rt, [], ctx)
            u_boul = self._normalize_utility(self._index_user_utility(b_boul))
            u_par = self._normalize_utility(self._index_user_utility(b_par))
            u_bopp = self._normalize_utility(self._index_user_utility(b_opp))
        except Exception:
            u_boul = u_par = u_bopp = u_norm
        base = [u_norm, rt, o_best, o_avg, o_sd, has_offer]
        extra = [u_bar, u_boul, u_par, u_bopp]
        return np.array(base + extra, dtype=np.float32)

    def _u_future_bid_norm(self) -> float:
        """Normalized utility of the agent's best library bid (proxy ``U(omega_t)``)."""
        if not self.action_library:
            return 0.0
        best_idx = max(range(len(self.action_library)), key=self._index_user_utility)
        return self._normalize_utility(self._index_user_utility(best_idx))

    # ------------------------------------------------------------------
    # Threshold prediction (threshold actor)
    # ------------------------------------------------------------------

    def _predict_threshold(self, state: Any, offer: Any) -> float:
        """Forward the threshold actor -> ``u_bar_t`` in ``[0, 1]`` (0.5 fallback)."""
        threshold_trainer = self.trainers.get("threshold_trainer")
        if threshold_trainer is None:
            return 0.5
        actor_entry = threshold_trainer.models.get("actor")
        if actor_entry is None:
            return 0.5
        actor = actor_entry["model"]
        obs = self.build_state_utility_threshold(state, offer)
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        actor.eval()
        with torch.no_grad():
            u_bar = actor(x)
        return float(u_bar.view(-1)[0].item())

    # ------------------------------------------------------------------
    # Sampling (three continuous DDPG actors)
    # ------------------------------------------------------------------

    def _sample_actor(self, obs: np.ndarray, trainer_key: str) -> List[float]:
        """Forward the named DDPG actor and return the continuous action vector."""
        trainer = self.trainers.get(trainer_key)
        if trainer is None:
            raise ValueError(f"No {trainer_key} attached. Use attach_trainer.")
        actor = trainer.models.get("actor", {}).get("model")
        if actor is None:
            raise ValueError(f"No 'actor' model in {trainer_key}.models.")
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        actor.eval()
        with torch.no_grad():
            a = actor(x)
        return a.view(-1).cpu().numpy().tolist()

    def sample_action_threshold_utility(self, obs: np.ndarray) -> List[float]:
        """Threshold actor sample: continuous ``[u_bar_t]`` vector."""
        return self._sample_actor(obs, "threshold_trainer")

    def sample_action_propose(self, obs: np.ndarray) -> List[float]:
        """Bidding actor sample: continuous ``(delta, c, p)`` flat vector."""
        return self._sample_actor(obs, "propose_trainer")

    def sample_action_respond(self, obs: np.ndarray) -> List[float]:
        """Acceptance actor sample: continuous ``(delta, c, p)`` flat vector."""
        return self._sample_actor(obs, "respond_trainer")

    # ------------------------------------------------------------------
    # Memory / flush
    # ------------------------------------------------------------------

    def record_step_threshold_utility(self, **fields: Any) -> None:
        """Append one threshold :class:`Step` to ``memory_threshold_utility``."""
        self.memory_threshold_utility.append(Step(**fields))

    def _flush_threshold(self, state: Any, utility_override: Optional[float] = None) -> None:
        """Flush the pending threshold step (special utility at episode end)."""
        pending = self.current_state_data_threshold_utility
        if pending:
            next_state = self.state_to_dict(state)
            util = utility_override if utility_override is not None else pending["utility"]
            self.record_step_threshold_utility(
                state=pending["state"], action=pending["action"], utility=util,
                value=pending["value"], log_prob=pending["log_prob"], next_state=next_state,
            )
            self.current_state_data_threshold_utility = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_negotiation_start(self, state: Any) -> None:
        """Reset all three memories + opponent history at episode start."""
        super().on_negotiation_start(state)
        self.memory_threshold_utility = []
        self.current_state_data_threshold_utility = None
        self.opponent_util_history_norm = []
        self.last_received_offer = None

    def respond(self, state: Any, source: Any = None) -> Any:
        """Acceptance flow (AS-IS): predict u_bar, sample strategy, decide accept."""
        offer = self._state_offer(state)

        self.flush_pending_propose(state)
        self.flush_pending_respond(state)
        self._flush_threshold(state)

        if offer is None:
            return ResponseType.REJECT_OFFER

        self._update_opponent_model(offer)
        u_user_norm = self._normalize_utility(self._offer_utility(offer))
        self.opponent_util_history_norm.append(u_user_norm)
        self.last_received_offer = offer

        t = self._relative_time(state)
        u_bar = self._predict_threshold(state, offer)
        self._last_threshold_value = u_bar

        obs_respond = self.build_state_respond(state, offer)
        action_respond = self.sample_action_respond(obs_respond)

        ctx_acc = {
            "u_bar": u_bar,
            "u_future_bid": self._u_future_bid_norm(),
            "opponent_util_history": list(self.opponent_util_history_norm),
        }
        accept = self.acceptance_strategy.decide(t, action_respond, u_user_norm, ctx_acc)
        action_id_respond = 0 if accept else 1

        current_u = self._offer_utility(offer)
        self.current_state_data_respond = {
            "state": self.build_state_respond(state, offer),
            "action": action_respond,
            "utility": current_u,
            "value": None,
            "log_prob": None,
            "decision": action_id_respond,
        }

        obs_threshold = self.build_state_utility_threshold(state, offer)
        action_threshold = self.sample_action_threshold_utility(obs_threshold)
        self.current_state_data_threshold_utility = {
            "state": obs_threshold,
            "action": action_threshold,
            "utility": current_u,
            "value": None,
            "log_prob": None,
        }

        self._emit_trace_step(
            "respond", state, obs_respond, action_id_respond,
            actor_output={"actor_vector": action_respond},
            extra={"u_bar": float(u_bar),
                   "decision": "accept" if accept else "reject",
                   "utility": current_u},
        )
        self._emit_trace_step(
            "utility_threshold", state, obs_threshold, action_threshold,
            actor_output={"actor_vector": action_threshold},
            extra={"u_bar_predicted": float(u_bar), "utility": current_u},
        )
        return ResponseType.ACCEPT_OFFER if accept else ResponseType.REJECT_OFFER

    def propose(self, state: Any, dest: Any = None) -> Any:
        """Bidding flow (AS-IS): sample strategy -> bidding template -> ``(idx,)``."""
        offer_opponent = self._state_offer(state)
        t = self._relative_time(state)

        u_bar = self._predict_threshold(state, offer_opponent)
        obs_propose = self.build_state_propose(state, offer_opponent)
        action_propose = self.sample_action_propose(obs_propose)

        ctx_bid = {
            "u_bar": u_bar,
            "opponent_last_offer": self.last_received_offer,
            "opponent_last_idx": self._offer_to_action_idx(self.last_received_offer)
            if self.last_received_offer is not None else None,
        }
        idx = self.bidding_strategy.generate(t, action_propose, ctx_bid)
        offer = self.make_offer(idx)
        current_u = self._offer_utility(offer)

        self.current_state_data_propose = {
            "state": obs_propose,
            "action": action_propose,
            "utility": current_u,
            "value": None,
            "log_prob": None,
        }
        self._emit_trace_step(
            "propose", state, obs_propose, idx,
            actor_output={"actor_vector": action_propose},
            extra={"u_bar": float(u_bar), "offer": offer, "utility": current_u},
        )
        return offer

    def on_negotiation_end(self, state: Any) -> None:
        """Flush all three pending steps, build three payloads, store to trainers."""
        agreement = getattr(state, "agreement", None)

        self.flush_pending_propose(state)
        self.flush_pending_respond(state)

        if agreement is not None:
            util_threshold = self._offer_utility(agreement)
        else:
            util_threshold = -1.0
        self._flush_threshold(state, utility_override=util_threshold)

        if not self.trainers:
            return
        payload_propose = self.build_episode_payload_propose(agreement)
        payload_respond = self.build_episode_payload_respond(agreement)
        payload_threshold = self.build_episode_payload_threshold(agreement)

        propose_trainer = self.trainers.get("propose_trainer")
        respond_trainer = self.trainers.get("respond_trainer")
        threshold_trainer = self.trainers.get("threshold_trainer")
        if propose_trainer is not None:
            propose_trainer.store_episode(payload_propose)
        if respond_trainer is not None:
            respond_trainer.store_episode(payload_respond)
        if threshold_trainer is not None:
            threshold_trainer.store_episode(payload_threshold)

    # ------------------------------------------------------------------
    # Episode payloads (three; actions are continuous vectors)
    # ------------------------------------------------------------------

    def _build_payload_from_memory(self, memory: List[Step], agreement: Any, phase: str) -> Episode:
        """Build a finalized :class:`Episode` (continuous ``action`` kept as-is)."""
        steps = deepcopy(memory)
        episode = Episode(
            agent_name=self.name,
            phase=phase,
            terminal_utility=self._offer_utility(agreement) if agreement is not None else -1.0,
            agreement=agreement,
            steps=steps,
        )
        # DDPG keeps continuous action vectors; finalize derives parallel lists.
        return episode.finalize()

    def build_episode_payload_propose(self, agreement: Any) -> Episode:
        """Build the bidding (propose) payload (``phase='propose'``)."""
        return self._build_payload_from_memory(self.memory_propose, agreement, "propose")

    def build_episode_payload_respond(self, agreement: Any) -> Episode:
        """Build the acceptance (respond) payload (``phase='respond'``)."""
        return self._build_payload_from_memory(self.memory_respond, agreement, "respond")

    def build_episode_payload_threshold(self, agreement: Any) -> Episode:
        """Build the utility-threshold payload (``phase='utility_threshold'``)."""
        return self._build_payload_from_memory(
            self.memory_threshold_utility, agreement, "utility_threshold"
        )

    # ------------------------------------------------------------------
    # Interface helpers / reset
    # ------------------------------------------------------------------

    def _mask_logits(self, logits: Any, allow_accept: bool) -> Any:
        """DDPG uses no discrete logits; returned unchanged (interface stub)."""
        return logits

    def reset_episode(self) -> None:
        """Clear all three memories and pending buffers + opponent history."""
        super().reset_episode()
        self.memory_threshold_utility = []
        self.current_state_data_threshold_utility = None
        self.opponent_util_history_norm = []
        self.last_received_offer = None
