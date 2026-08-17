"""AcceptanceStrategy — DLST-ANESIA acceptance template (paper Eq. 12), AS-IS.

``f_a(s_t, u_bar_t, Omega^o_t) -> {accept, reject}`` (formulas unchanged):

1. Determine the active phase ``i`` from time ``t`` and durations ``delta``.
2. For each active tactic (``c_{i,j} > 0.5``) in phase ``i`` compute its threshold.
3. Combine active thresholds via ``max`` (as in the paper).
4. Accept the opponent's offer iff ``U_u(omega^o_t) >= combined threshold``.

Tactic library ``T_a``: ``U_future``, ``quantile``, ``u_bar``, ``u_fixed``.
The head is unaffected by the input mode (Gotcha G13).
"""

from typing import Any, Dict, Sequence

import numpy as np


class AcceptanceStrategy:
    """Acceptance strategy template producing accept / reject (AS-IS formulas).

    Parameters
    ----------
    spec:
        :class:`StrategyTemplateSpec` (phase / tactic layout).
    u_fixed:
        Pre-defined fixed threshold for the ``u_fixed`` tactic.

    Example:
        >>> from negotiation.agents.dlst.strategy_template import StrategyTemplateSpec
        >>> spec = StrategyTemplateSpec(1, [1], [["u_fixed"]])
        >>> acc = AcceptanceStrategy(spec, u_fixed=0.6)
        >>> acc.decide(0.0, [1.0, 1.0, 0.0, 0.0], 0.7, {})
        True
        >>> acc.decide(0.0, [1.0, 1.0, 0.0, 0.0], 0.5, {})
        False
    """

    def __init__(self, spec, u_fixed: float = 0.6) -> None:
        self.spec = spec
        self.u_fixed = float(u_fixed)

    # ------------------------------------------------------------------
    # Tactic library (each returns a threshold in [0, 1])
    # ------------------------------------------------------------------

    def _tactic_U_future(self, t: float, p: Sequence[float], ctx: Dict[str, Any]) -> float:
        """``U_u(omega_t)``: utility of the bid the agent would propose (AS-IS)."""
        return float(ctx.get("u_future_bid", 0.0))

    def _tactic_quantile(self, t: float, p: Sequence[float], ctx: Dict[str, Any]) -> float:
        """``QU_{Omega^o_t}(a*t + b)``: quantile of the opponent util history (AS-IS)."""
        a = float(p[0]) if len(p) > 0 else 0.0
        b = float(p[1]) if len(p) > 1 else 0.5
        q = min(1.0, max(0.0, a * t + b))
        hist = ctx.get("opponent_util_history", [])
        if len(hist) == 0:
            return q
        return float(np.quantile(np.asarray(hist, dtype=np.float64), q))

    def _tactic_u_bar(self, t: float, p: Sequence[float], ctx: Dict[str, Any]) -> float:
        """``u_bar_t``: dynamic threshold from the DRL threshold actor (AS-IS)."""
        return float(ctx.get("u_bar", 0.0))

    def _tactic_u_fixed(self, t: float, p: Sequence[float], ctx: Dict[str, Any]) -> float:
        """``u``: the pre-defined fixed threshold (AS-IS)."""
        return float(self.u_fixed)

    @property
    def _TACTIC_FNS(self) -> Dict[str, Any]:
        """Tactic dispatch table (add new acceptance tactics here)."""
        return {
            "U_future": self._tactic_U_future,
            "quantile": self._tactic_quantile,
            "u_bar": self._tactic_u_bar,
            "u_fixed": self._tactic_u_fixed,
        }

    # ------------------------------------------------------------------
    # Decide
    # ------------------------------------------------------------------

    def compute_threshold(self, t: float, action_vec: Sequence[float], ctx: Dict[str, Any]) -> float:
        """Combine active tactic thresholds at time ``t`` via ``max`` (AS-IS)."""
        deltas, choices, params = self.spec.decode(action_vec)
        i = self.spec.active_phase(t, deltas)
        names = self.spec.tactic_names_per_phase[i]
        thresholds = []
        for j, name in enumerate(names):
            if choices[i][j] <= 0.5:
                continue
            fn = self._TACTIC_FNS.get(name)
            if fn is None:
                continue
            thresholds.append(fn(t, params[i][j], ctx))
        if len(thresholds) == 0:
            return float(ctx.get("u_bar", 0.0))
        return float(max(thresholds))

    def decide(self, t: float, action_vec: Sequence[float], opponent_offer_util_norm: float, ctx: Dict[str, Any]) -> bool:
        """Return ``True`` iff the (normalized) opponent offer meets the threshold."""
        threshold = self.compute_threshold(t, action_vec, ctx)
        return float(opponent_offer_util_norm) >= threshold
