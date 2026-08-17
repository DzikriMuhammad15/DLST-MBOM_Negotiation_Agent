"""MBOMAgentNonZeroSum — non-zero-sum opponent utility variant (AS-IS).

Same as :class:`MBOMAgent` except the opponent **utility estimate** used as the
rollout reward (``opponent_util_fn`` in recursive imagination) is produced by a
:class:`FrequencyOpponentModel` (the same mechanism DLST uses) instead of the
zero-sum ``-U_user(offer)``. The opponent **value function** stays zero-sum
(``-V_user``), exactly as in the paper code — only the utility estimate changes.

Only the relevant hooks are overridden; ``propose`` / ``respond`` /
``on_negotiation_end`` and the IOP / rollout / mixer pipeline are inherited
unchanged (Invariant #1 preserved: propose and respond still fully separate).
"""

from copy import deepcopy
from typing import Any

from negotiation.agents.dlst.opponent_model import FrequencyOpponentModel
from negotiation.agents.mbom_agent import MBOMAgent


class MBOMAgentNonZeroSum(MBOMAgent):
    """MBOM with a frequency-model (non-zero-sum) opponent utility estimate.

    Example:
        >>> MBOMAgentNonZeroSum.__name__
        'MBOMAgentNonZeroSum'
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.opponent_model = FrequencyOpponentModel(
            issue_names=[self._domain_attr("issue_name", "bundle_index") or "bundle_index"],
            value_lists=self._domain_attr("value_lists", {}) or {},
        )

    def _opponent_utility_estimate(self, offer: Any) -> float:
        """Override: non-zero-sum opponent utility via the frequency model (AS-IS).

        FIX: rollout memanggil ini dengan elemen ``action_library`` mentah
        (Bundle); model frekuensi bekerja pada outcome ``(idx,)`` — konversi dulu
        lewat :meth:`_bundle_to_outcome` agar estimasi tidak selalu jatuh ke 0.
        """
        return float(self.opponent_model.estimate_utility(self._bundle_to_outcome(offer)))

    def _update_opponent_model(self, offer: Any) -> None:
        """Record one real opponent offer into the frequency model."""
        if offer is not None:
            self.opponent_model.update(offer)

    def _on_real_offer_received(self, offer: Any) -> None:
        """Override hook: update the frequency model on each real opponent offer."""
        self._update_opponent_model(offer)

    def reset_episode(self) -> None:
        """Clear memories + reset the opponent frequency model."""
        super().reset_episode()
        self.opponent_model.reset()

    def clone_for_rollout(self) -> "MBOMAgentNonZeroSum":
        """Clone for imagined rollout, deep-copying the opponent model state (AS-IS)."""
        clone = super().clone_for_rollout()
        clone.opponent_model = deepcopy(self.opponent_model)
        return clone
