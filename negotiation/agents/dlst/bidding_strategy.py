"""BiddingStrategy — DLST-ANESIA bidding template (paper Eq. 13), AS-IS.

``f_b(s_t, u_bar_t, Omega^o_t) -> offer`` where the offer is **always an exact
member of ``action_library``** (so it is compatible with the bundle-index outcome
``(idx,)`` without changing the head, ``03-01``). The mechanism is unchanged from
the paper:

1. Determine the active phase ``i`` from time ``t`` and the durations ``delta``.
2. Among the active bidding tactics (``c_{i,j} > 0.5``) in phase ``i`` pick the
   first active one (deterministic).
3. Run that tactic to produce a library index.

Tactic library ``T_b`` (formulas exact): ``boulware``, ``pareto``, ``b_opp``,
``random_above``.

Adaptation: tactics return a **library index** (the new outcome is a pure index);
the DLST agent wraps it into ``(idx,)``. ``user_utility_fn`` / ``opponent_utility_fn``
are callables over the candidate (bundle / index) consistent with the rest of the
new layer.
"""

import random
from typing import Any, Callable, Dict, List, Sequence


class BiddingStrategy:
    """Bidding strategy template producing a library index (AS-IS formulas).

    Parameters
    ----------
    spec:
        :class:`StrategyTemplateSpec` (phase / tactic layout).
    action_library:
        Candidate bundles; the strategy returns an index into this list.
    issue_names:
        Issue names (single ``bundle_index`` for the new outcome space).
    user_utility_fn:
        ``callable(idx_or_offer) -> float`` raw user utility of a candidate.
    opponent_utility_fn:
        ``callable(idx_or_offer) -> float`` estimated opponent utility in ``[0, 1]``.
    utility_min, utility_max:
        Bounds for normalizing user utility.

    Example:
        >>> from negotiation.agents.dlst.strategy_template import StrategyTemplateSpec
        >>> spec = StrategyTemplateSpec(1, [1], [["boulware"]])
        >>> lib = ["A", "B", "C"]
        >>> uu = {"A": 0.2, "B": 0.6, "C": 0.9}
        >>> bs = BiddingStrategy(spec, lib, ["bundle_index"],
        ...     user_utility_fn=lambda i: uu[lib[i]],
        ...     opponent_utility_fn=lambda i: 1.0 - uu[lib[i]],
        ...     utility_min=0.0, utility_max=1.0)
        >>> 0 <= bs.generate(0.0, [1.0, 0.9, 0.1, 0.2], {}) < 3
        True
    """

    def __init__(
        self,
        spec,
        action_library: Sequence[Any],
        issue_names: Sequence[str],
        user_utility_fn: Callable[[int], float],
        opponent_utility_fn: Callable[[int], float],
        utility_min: float,
        utility_max: float,
    ) -> None:
        self.spec = spec
        self.action_library = list(action_library)
        self.issue_names = list(issue_names)
        self.user_utility_fn = user_utility_fn
        self.opponent_utility_fn = opponent_utility_fn
        self.utility_min = float(utility_min)
        self.utility_max = float(utility_max)
        # Pre-compute normalized user utility per candidate index (efficiency).
        self._u_user_cache: List[float] = [
            self._norm(self.user_utility_fn(i)) for i in range(len(self.action_library))
        ]

    def _norm(self, u: float) -> float:
        """Min-max normalize a raw user utility."""
        denom = max(1e-8, self.utility_max - self.utility_min)
        return (float(u) - self.utility_min) / denom

    # ------------------------------------------------------------------
    # Tactic library (each returns a library index)
    # ------------------------------------------------------------------

    def _tactic_boulware(self, t: float, p: Sequence[float], ctx: Dict[str, Any]) -> int:
        """Boulware: high target early, larger concession near ``t = 1`` (AS-IS)."""
        e = 0.02 + abs(float(p[0])) if len(p) > 0 else 0.1
        target = 1.0 - (t ** (1.0 / max(1e-3, e)))
        target = min(1.0, max(0.0, target))
        return self._closest_index_to_target(target)

    def _tactic_pareto(self, t: float, p: Sequence[float], ctx: Dict[str, Any]) -> int:
        """``PS(a*t + b)``: maximize ``w1*U_u + w2*U_o`` (TOPSIS-like, AS-IS)."""
        a = float(p[0]) if len(p) > 0 else -0.5
        b = float(p[1]) if len(p) > 1 else 0.8
        w1 = min(1.0, max(0.0, a * t + b))
        w2 = 1.0 - w1
        best_idx, best_score = 0, -1e18
        for idx in range(len(self.action_library)):
            uu = self._u_user_cache[idx]
            uo = float(self.opponent_utility_fn(idx))
            score = w1 * uu + w2 * uo
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx

    def _tactic_b_opp(self, t: float, p: Sequence[float], ctx: Dict[str, Any]) -> int:
        """Greedy manipulation of the opponent's last offer (AS-IS).

        Picks the candidate most similar to the opponent's last offer that also
        has higher user utility; similarity is index identity in the
        bundle-enumeration space.
        """
        opp_offer = ctx.get("opponent_last_offer", None)
        opp_idx = ctx.get("opponent_last_idx", None)
        if opp_offer is None and opp_idx is None:
            return self._best_user_index()
        best_idx, best_score = None, -1e18
        for idx in range(len(self.action_library)):
            sim = 1 if (opp_idx is not None and idx == opp_idx) else 0
            uu = self._u_user_cache[idx]
            score = sim + uu
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx if best_idx is not None else 0

    def _tactic_random_above(self, t: float, p: Sequence[float], ctx: Dict[str, Any]) -> int:
        """``omega ~ U(Omega >= u_bar_t)``: random index above the threshold (AS-IS)."""
        u_bar = float(ctx.get("u_bar", 0.0))
        candidates = [i for i in range(len(self.action_library)) if self._u_user_cache[i] >= u_bar]
        if not candidates:
            return self._best_user_index()
        return random.choice(candidates)

    @property
    def _TACTIC_FNS(self) -> Dict[str, Callable[..., int]]:
        """Tactic dispatch table (add new bidding tactics here)."""
        return {
            "boulware": self._tactic_boulware,
            "pareto": self._tactic_pareto,
            "b_opp": self._tactic_b_opp,
            "random_above": self._tactic_random_above,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _closest_index_to_target(self, target_norm: float) -> int:
        """Library index whose normalized user utility is closest to target."""
        best_idx, best_gap = 0, float("inf")
        for idx in range(len(self.action_library)):
            gap = abs(self._u_user_cache[idx] - target_norm)
            if gap < best_gap:
                best_gap = gap
                best_idx = idx
        return best_idx

    def _best_user_index(self) -> int:
        """Library index with the highest user utility."""
        if not self.action_library:
            return 0
        return max(range(len(self.action_library)), key=lambda i: self._u_user_cache[i])

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(self, t: float, action_vec: Sequence[float], ctx: Dict[str, Any]) -> int:
        """Return the library index to propose at time ``t`` (AS-IS dispatch).

        Example:
            >>> from negotiation.agents.dlst.strategy_template import StrategyTemplateSpec
            >>> spec = StrategyTemplateSpec(1, [1], [["random_above"]])
            >>> bs = BiddingStrategy(spec, ["A", "B"], ["bundle_index"],
            ...     user_utility_fn=lambda i: [0.3, 0.9][i],
            ...     opponent_utility_fn=lambda i: 0.5,
            ...     utility_min=0.0, utility_max=1.0)
            >>> bs.generate(0.0, [1.0, 1.0, 0.0, 0.0], {"u_bar": 0.8}) in (0, 1)
            True
        """
        deltas, choices, params = self.spec.decode(action_vec)
        i = self.spec.active_phase(t, deltas)
        names = self.spec.tactic_names_per_phase[i]
        for j, name in enumerate(names):
            if choices[i][j] > 0.5:
                fn = self._TACTIC_FNS.get(name)
                if fn is not None:
                    return int(fn(t, params[i][j], ctx))
        return self._best_user_index()
