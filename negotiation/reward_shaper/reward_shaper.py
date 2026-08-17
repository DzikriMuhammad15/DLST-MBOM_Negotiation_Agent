"""Reward shaper — EXACT paper formulas (``02-09``). DO NOT change the formulas.

This is the AS-IS reward shaper. Rewards are computed from the negotiation
*utilities* (which, in the new project, are generated from the world via the
ufun); the reward formulas themselves (Eq. 8 / 9 / 10, respond modes 1..27, and
the special ``"dlst"`` mode) are unchanged.

The ``utility_min`` / ``utility_max`` used here are the **per-negotiation** bounds
from ``compute_utility_bounds(action_library, side)`` and are used ONLY for
reward scaling — they are separate from the intrinsic ``REF_*`` ufun
normalization (Contract C6 / Gotcha G10).
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional


# =============================================================================
# RESPOND ACTION CONSTANTS
# Matches the encoding used across all agents:
#   action 0 = ACCEPT_OFFER
#   action 1 = REJECT_OFFER  (continue negotiation / counter-offer)
# =============================================================================
_RESPOND_ACTION_ACCEPT = 0
_RESPOND_ACTION_REJECT = 1


# =============================================================================
# ProposeRewardConfig
# =============================================================================
@dataclass
class ProposeRewardConfig:
    """Reward configuration for the propose trainer.

    Available modes:
      - mode1: r_bid (Eq. 9)
                 reward = U_u(omega_acc, t)  if an agreement is reached at this step
                          -1                  otherwise (negotiation still running)

    Example:
        >>> ProposeRewardConfig().mode
        'mode1'
    """

    mode: str = "mode1"
    utility_scale: bool = True


# =============================================================================
# RespondRewardConfig
# =============================================================================
@dataclass
class RespondRewardConfig:
    """Reward configuration for the respond trainer.

    Each respond step has three possible events:
      1. Accept  : the agent accepts the opponent's offer (action == 0)
      2. Offer   : the agent rejects and continues (action == 1, not the last step)
      3. Failure : negotiation ends without agreement (last step, agreement=None)

    Reward of each event is determined by the combination of formulas below.

    accept_variant -> r_accept_*:
      "a" : U_A(omega^{t-1}_{B->A})
      "b" : (tanh(5*(U_A - 0.5)) + 1) / 2
      "c" : tanh(5*(U_A - 0.5))

    offer_variant -> r_offer_*:
      "a" : 0
      "b" : (1 - U_A) / 100
      "c" : (1 - average_window) / 100

    failure_penalty: -1 | -0.5 | 0

    Mode table (accept_variant, offer_variant, failure_penalty): see
    ``RewardShaper._RESPOND_MODE_TABLE`` (27 combinations: 3 x 3 x 3).

    Example:
        >>> RespondRewardConfig(mode="dlst").mode
        'dlst'
    """

    mode: str = "mode1"
    utility_scale: bool = True
    offer_avg_window: int = 10


# =============================================================================
# ThresholdRewardConfig  (DLST-ANESIA)
# =============================================================================
@dataclass
class ThresholdRewardConfig:
    """Reward configuration for the utility-threshold trainer (DLST, Eq. 8).

    r_u-bar_t = U_u(omega_acc, t)   if an agreement is reached
                U_u(omega^o_t, t)    if the opponent's offer is accepted
                -1                   otherwise

    Available modes:
      - mode1: Eq. 8 above (DLST-ANESIA default).

    Example:
        >>> ThresholdRewardConfig().mode
        'mode1'
    """

    mode: str = "mode1"
    utility_scale: bool = True


# =============================================================================
# RewardShaper
# =============================================================================
class RewardShaper:
    """Single class computing per-step rewards from episode data (paper-exact).

    Instantiation:
        RewardShaper(cfg, utility_min, utility_max, for_="propose")
        RewardShaper(cfg, utility_min, utility_max, for_="respond", opponent_model=...)
        RewardShaper(cfg, utility_min, utility_max, for_="utility_threshold")

    ``for_`` is required and selects the reward family:
      - "propose"           : uses ProposeRewardConfig
      - "respond"           : uses RespondRewardConfig
      - "utility_threshold" : uses ThresholdRewardConfig (DLST-ANESIA)

    Opponent utility approximation (DLST-ANESIA): respond reward r_acc (Eq. 10)
    depends on the opponent utility U_o, estimated via a FrequencyOpponentModel
    passed as ``opponent_model``.

    Example:
        >>> shaper = RewardShaper(ProposeRewardConfig(), 0.0, 1.0, for_="propose")
        >>> shaper.for_
        'propose'
    """

    # Format: mode_name -> (accept_variant, offer_variant, failure_penalty)
    _RESPOND_MODE_TABLE: Dict[str, tuple] = {
        "mode1":  ("a", "a", -1.0),
        "mode2":  ("b", "a", -1.0),
        "mode3":  ("c", "a", -1.0),
        "mode4":  ("a", "b", -1.0),
        "mode5":  ("b", "b", -1.0),
        "mode6":  ("c", "b", -1.0),
        "mode7":  ("a", "c", -1.0),
        "mode8":  ("b", "c", -1.0),
        "mode9":  ("c", "c", -1.0),
        "mode10": ("a", "a", -0.5),
        "mode11": ("b", "a", -0.5),
        "mode12": ("c", "a", -0.5),
        "mode13": ("a", "b", -0.5),
        "mode14": ("b", "b", -0.5),
        "mode15": ("c", "b", -0.5),
        "mode16": ("a", "c", -0.5),
        "mode17": ("b", "c", -0.5),
        "mode18": ("c", "c", -0.5),
        "mode19": ("a", "a",  0.0),
        "mode20": ("b", "a",  0.0),
        "mode21": ("c", "a",  0.0),
        "mode22": ("a", "b",  0.0),
        "mode23": ("b", "b",  0.0),
        "mode24": ("c", "b",  0.0),
        "mode25": ("a", "c",  0.0),
        "mode26": ("b", "c",  0.0),
        "mode27": ("c", "c",  0.0),
    }

    # Special DLST-ANESIA respond mode (Eq. 10), processed separately.
    _RESPOND_DLST_MODES = ("dlst",)

    def __init__(
        self,
        cfg,
        utility_min: float = 0.0,
        utility_max: float = 1.0,
        for_: Literal["propose", "respond", "utility_threshold"] = None,
        opponent_model: Optional[Any] = None,
    ):
        """Validate ``for_`` / ``cfg`` pairing and store scaling bounds.

        Example:
            >>> RewardShaper(RespondRewardConfig(), 0.0, 2.0,
            ...              for_="respond").utility_max
            2.0
        """
        if for_ not in ("propose", "respond", "utility_threshold"):
            raise ValueError(
                f"[RewardShaper] 'for_' is required and must be "
                f"'propose', 'respond', or 'utility_threshold', not: {for_!r}"
            )
        if for_ == "propose" and not isinstance(cfg, ProposeRewardConfig):
            raise TypeError(
                "[RewardShaper] for_='propose' requires cfg of type ProposeRewardConfig."
            )
        if for_ == "respond" and not isinstance(cfg, RespondRewardConfig):
            raise TypeError(
                "[RewardShaper] for_='respond' requires cfg of type RespondRewardConfig."
            )
        if for_ == "utility_threshold" and not isinstance(cfg, ThresholdRewardConfig):
            raise TypeError(
                "[RewardShaper] for_='utility_threshold' requires cfg of type "
                "ThresholdRewardConfig."
            )

        self.cfg = cfg
        self.utility_min = utility_min
        self.utility_max = utility_max
        self.for_ = for_
        self.opponent_model = opponent_model

    # ─────────────────────────────────────────────────────────────────
    # UTILITY SCALING
    # ─────────────────────────────────────────────────────────────────

    def _scale(self, u: float) -> float:
        """Normalize utility to [0, 1] using utility_min / utility_max.

        Example:
            >>> s = RewardShaper(ProposeRewardConfig(), 0.0, 2.0, for_="propose")
            >>> s._scale(1.0)
            0.5
        """
        if not self.cfg.utility_scale:
            return float(u)
        denom = max(1e-8, self.utility_max - self.utility_min)
        return (float(u) - self.utility_min) / denom

    # ─────────────────────────────────────────────────────────────────
    # PROPOSE — FORMULA
    # ─────────────────────────────────────────────────────────────────

    def _compute_propose_mode1(self, u: float, is_agreement_step: bool) -> float:
        """Eq. 9 (r_bid): U_u(omega_acc, t) on the agreement step, else -1.

        Example:
            >>> s = RewardShaper(ProposeRewardConfig(), 0.0, 1.0, for_="propose")
            >>> s._compute_propose_mode1(0.8, True), s._compute_propose_mode1(0.8, False)
            (0.8, -1.0)
        """
        if is_agreement_step:
            return u
        return -1.0

    # ─────────────────────────────────────────────────────────────────
    # RESPOND — FORMULA ACCEPT (r_accept_*)
    # ─────────────────────────────────────────────────────────────────

    def _r_accept_a(self, u_accept: float) -> float:
        """Rule (4-a): r_accept_a = U_A(omega^{t-1}_{B->A})."""
        return u_accept

    def _r_accept_b(self, u_accept: float) -> float:
        """Rule (4-b): r_accept_b = (tanh(5*(U_A - 0.5)) + 1) / 2."""
        return (math.tanh(5.0 * (u_accept - 0.5)) + 1.0) / 2.0

    def _r_accept_c(self, u_accept: float) -> float:
        """Rule (4-c): r_accept_c = tanh(5*(U_A - 0.5))."""
        return math.tanh(5.0 * (u_accept - 0.5))

    # ─────────────────────────────────────────────────────────────────
    # RESPOND — FORMULA OFFER / CONTINUED NEGOTIATION (r_offer_*)
    # ─────────────────────────────────────────────────────────────────

    def _r_offer_a(self, u_current: float, recent_utilities: List[float]) -> float:
        """Rule (5-a): r_offer_a = 0."""
        return 0.0

    def _r_offer_b(self, u_current: float, recent_utilities: List[float]) -> float:
        """Rule (5-b): r_offer_b = (1 - U_A(omega^{t-1}_{B->A})) / 100."""
        return (1.0 - u_current) / 100.0

    def _r_offer_c(self, u_current: float, recent_utilities: List[float]) -> float:
        """Rule (5-c): r_offer_c = (1 - average_window) / 100."""
        if len(recent_utilities) == 0:
            avg = 0.0
        else:
            avg = sum(recent_utilities) / len(recent_utilities)
        return (1.0 - avg) / 100.0

    # ─────────────────────────────────────────────────────────────────
    # DISPATCH TABLES FOR RESPOND FORMULAS
    # ─────────────────────────────────────────────────────────────────

    @property
    def _RESPOND_ACCEPT_FNS(self) -> Dict[str, Any]:
        return {
            "a": self._r_accept_a,
            "b": self._r_accept_b,
            "c": self._r_accept_c,
        }

    @property
    def _RESPOND_OFFER_FNS(self) -> Dict[str, Any]:
        return {
            "a": self._r_offer_a,
            "b": self._r_offer_b,
            "c": self._r_offer_c,
        }

    # ─────────────────────────────────────────────────────────────────
    # PROPOSE — COMPUTE STEP REWARD
    # ─────────────────────────────────────────────────────────────────

    def _compute_step_reward_propose(
        self,
        step: Any,
        episode: Any,
        step_idx: int,
    ) -> float:
        """Compute the propose reward for one step (ProposeRewardConfig.mode)."""
        u = step.utility
        if u is None:
            u = 0.0
        u = self._scale(float(u))

        is_last = step_idx == len(episode.steps) - 1
        is_agreement_step = is_last and (episode.agreement is not None)

        mode = self.cfg.mode.lower()

        if mode == "mode1":
            return self._compute_propose_mode1(u, is_agreement_step)

        raise ValueError(
            f"[RewardShaper] unknown propose mode: '{mode}'. Available: mode1"
        )

    # ─────────────────────────────────────────────────────────────────
    # RESPOND — COMPUTE STEP REWARD
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _action_is_accept(action: Any) -> Optional[bool]:
        """Decode whether ``action`` means ACCEPT (0) / REJECT (1) for discrete respond.

        For DLST continuous action vectors this returns None so the caller falls
        back to the (is_last & has_agreement) heuristic.

        Example:
            >>> RewardShaper._action_is_accept(0)
            True
            >>> RewardShaper._action_is_accept(1)
            False
            >>> RewardShaper._action_is_accept([0.2, 0.4]) is None
            True
        """
        if action is None:
            return None
        if isinstance(action, (int, float)) and not isinstance(action, bool):
            ai = int(action)
            if ai == _RESPOND_ACTION_ACCEPT:
                return True
            if ai == _RESPOND_ACTION_REJECT:
                return False
            return None
        return None

    def _compute_step_reward_respond(
        self,
        step: Any,
        episode: Any,
        step_idx: int,
    ) -> float:
        """Compute the respond reward for one step.

        Two paths:
          - mode "dlst"  : r_acc (Eq. 10), needs U_o from opponent_model.
          - mode "modeN" : (accept_variant, offer_variant, failure_penalty).
        """
        mode = self.cfg.mode.lower()

        if mode in self._RESPOND_DLST_MODES:
            return self._compute_respond_dlst(step, episode, step_idx)

        window = self.cfg.offer_avg_window

        if mode not in self._RESPOND_MODE_TABLE:
            raise ValueError(
                f"[RewardShaper] unknown respond mode: '{mode}'. Available: "
                f"{list(self._RESPOND_MODE_TABLE.keys())} "
                f"+ {list(self._RESPOND_DLST_MODES)}"
            )

        accept_var, offer_var, failure_penalty = self._RESPOND_MODE_TABLE[mode]
        accept_fn = self._RESPOND_ACCEPT_FNS[accept_var]
        offer_fn = self._RESPOND_OFFER_FNS[offer_var]

        u = step.utility
        if u is None:
            u = 0.0
        u = self._scale(float(u))

        action = step.action
        is_last = step_idx == len(episode.steps) - 1
        has_agreement = episode.agreement is not None

        # Case 3: failure (negotiation ends with no agreement).
        if is_last and not has_agreement:
            return float(failure_penalty)

        # Case 1: accept.
        decoded = self._action_is_accept(action)
        is_accept = (decoded is True) or (is_last and has_agreement)
        if is_accept:
            return accept_fn(u)

        # Case 2: offer / continued negotiation.
        recent_utilities: List[float] = []
        start = max(0, step_idx - window)
        for prev_idx in range(start, step_idx):
            prev_u = episode.steps[prev_idx].utility
            if prev_u is None:
                prev_u = 0.0
            recent_utilities.append(self._scale(float(prev_u)))

        return offer_fn(u, recent_utilities)

    # ─────────────────────────────────────────────────────────────────
    # RESPOND — DLST-ANESIA (Eq. 10, r_acc)
    # ─────────────────────────────────────────────────────────────────

    def _offer_from_step(self, step: Any) -> Any:
        """Read the opponent offer from the step state dict (current_offer)."""
        st = step.state
        if isinstance(st, dict):
            return st.get("current_offer", None)
        return getattr(st, "current_offer", None)

    def _estimate_opponent_utility(self, step: Any) -> Optional[float]:
        """Estimate U_o(omega^o_t) via opponent_model when available."""
        if self.opponent_model is None:
            return None
        offer = self._offer_from_step(step)
        return float(self.opponent_model.estimate_utility(offer))

    def _compute_respond_dlst(
        self,
        step: Any,
        episode: Any,
        step_idx: int,
    ) -> float:
        """Eq. 10 (r_acc).

        r_acc = U_u(omega_acc, t)  if agree   & U_o(omega_acc, t) <= U_u(omega_acc, t)
                U_u(omega^o_t, t)   if reject  & U_o(omega^o_t, t) >= U_u(omega^o_t, t)
                -1                  otherwise

        When opponent_model is None, U_o is treated as not violating the
        condition (fallback to U_u).
        """
        u_user = step.utility
        if u_user is None:
            u_user = 0.0
        u_user = self._scale(float(u_user))

        u_opp = self._estimate_opponent_utility(step)

        action = step.action
        is_last = step_idx == len(episode.steps) - 1
        has_agreement = episode.agreement is not None

        decoded = self._action_is_accept(action)
        is_accept = (decoded is True) or (is_last and has_agreement)
        is_reject = (not is_accept) and not (is_last and not has_agreement)

        if is_accept:
            if u_opp is None or u_opp <= u_user:
                return u_user
            return -1.0

        if is_reject:
            if u_opp is None or u_opp >= u_user:
                return u_user
            return -1.0

        return -1.0

    # ─────────────────────────────────────────────────────────────────
    # UTILITY-THRESHOLD — COMPUTE STEP REWARD (DLST-ANESIA, Eq. 8)
    # ─────────────────────────────────────────────────────────────────

    def _compute_step_reward_threshold(
        self,
        step: Any,
        episode: Any,
        step_idx: int,
    ) -> float:
        """Eq. 8 (r_u-bar_t).

        The DLST agent stores step.utility as: U_u(agreement) when agreed,
        U_u(opponent offer) when accepting, or -1 on failure. The -1 marker is
        not scaled (it is a penalty).
        """
        mode = self.cfg.mode.lower()

        if mode == "mode1":
            u = step.utility
            if u is None:
                return -1.0
            u = float(u)
            if u == -1.0:
                return -1.0
            return self._scale(u)

        raise ValueError(
            f"[RewardShaper] unknown utility_threshold mode: '{mode}'. "
            f"Available: mode1"
        )

    # ─────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────

    def compute_step_reward(
        self,
        step: Any,
        episode: Any,
        step_idx: int,
    ) -> float:
        """Compute the reward of one step, dispatching on ``self.for_``.

        C6: when the episode carries its own per-negotiation scaling bounds
        (``episode.utility_min`` / ``utility_max``, stamped at store time by the
        trainer), those bounds are adopted for this step so the scaled utility
        spans exactly [0, 1] over that negotiation's action library. Episodes
        without bounds (legacy / offline logs) fall back to the constructor
        bounds, preserving the old behaviour.
        """
        emin = getattr(episode, "utility_min", None)
        emax = getattr(episode, "utility_max", None)
        if emin is None or emax is None:
            return self._dispatch_step_reward(step, episode, step_idx)
        prev = (self.utility_min, self.utility_max)
        self.utility_min, self.utility_max = float(emin), float(emax)
        try:
            return self._dispatch_step_reward(step, episode, step_idx)
        finally:
            self.utility_min, self.utility_max = prev

    def _dispatch_step_reward(
        self,
        step: Any,
        episode: Any,
        step_idx: int,
    ) -> float:
        """Dispatch one step's reward on ``self.for_`` (bounds already set)."""
        if self.for_ == "propose":
            return self._compute_step_reward_propose(step, episode, step_idx)
        if self.for_ == "respond":
            return self._compute_step_reward_respond(step, episode, step_idx)
        return self._compute_step_reward_threshold(step, episode, step_idx)

    def compute_reward_sequence(self, episode: Any) -> List[float]:
        """Compute the full reward sequence of one episode (len == len(steps)).

        Example:
            >>> from negotiation.data_type.step import Step
            >>> from negotiation.data_type.episode import Episode
            >>> ep = Episode(agent_name="fleet", phase="propose", agreement=(1,))
            >>> ep.steps = [Step(action=1, utility=-1.0),
            ...             Step(action=1, utility=0.6)]
            >>> shaper = RewardShaper(ProposeRewardConfig(), 0.0, 1.0,
            ...                       for_="propose")
            >>> shaper.compute_reward_sequence(ep)
            [-1.0, 0.6]
        """
        rewards = []
        for i, step in enumerate(episode.steps):
            rewards.append(self.compute_step_reward(step, episode, i))
        return rewards
