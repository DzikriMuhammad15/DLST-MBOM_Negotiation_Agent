"""DLSTMBOMNonZeroSum — non-zero-sum opponent utility hybrid variant (AS-IS).

Same as :class:`DLSTMBOMAgent` except the opponent **utility estimate** used as
the recursive-imagination rollout reward is produced by a
:class:`FrequencyOpponentModel` (non-zero-sum) instead of ``-U_user(offer)``. Only
the relevant hook is overridden; the DLST head and the rest of the IOP / rollout /
mixer pipeline are inherited unchanged (Invariant #1 preserved).
"""

from copy import deepcopy
from typing import Any

from negotiation.agents.dlst.opponent_model import FrequencyOpponentModel
from negotiation.agents.dlst_mbom_agent import DLSTMBOMAgent


class DLSTMBOMNonZeroSum(DLSTMBOMAgent):
    """DLST+MBOM hybrid with a frequency-model opponent utility estimate.

    Example:
        >>> DLSTMBOMNonZeroSum.__name__
        'DLSTMBOMNonZeroSum'
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.mbom_opponent_model = FrequencyOpponentModel(
            issue_names=[self._domain_attr("issue_name", "bundle_index") or "bundle_index"],
            value_lists=self._domain_attr("value_lists", {}) or {},
        )

    def _opponent_utility_estimate(self, offer: Any) -> float:
        """Override: non-zero-sum opponent utility via the frequency model (AS-IS).

        FIX: rollout memanggil ini dengan elemen ``action_library`` mentah
        (Bundle); model frekuensi bekerja pada outcome ``(idx,)`` — konversi dulu
        lewat :meth:`_bundle_to_outcome` agar estimasi tidak selalu jatuh ke 0.
        """
        return float(self.mbom_opponent_model.estimate_utility(self._bundle_to_outcome(offer)))

    def _on_real_offer_received(self, offer: Any) -> None:
        """Update the MBOM frequency model on each real opponent offer."""
        if offer is not None:
            self.mbom_opponent_model.update(offer)

    def respond(self, state: Any, source: Any = None) -> Any:
        """Update the MBOM opponent model, then run the hybrid respond flow."""
        offer = self._state_offer(state)
        if self.rollout_level is None:
            self._on_real_offer_received(offer)
        return super().respond(state, source)

    def reset_episode(self) -> None:
        """Clear memories + reset the MBOM opponent model."""
        super().reset_episode()
        self.mbom_opponent_model.reset()

    def clone_for_rollout(self) -> "DLSTMBOMNonZeroSum":
        """Clone for rollout, deep-copying the MBOM opponent model state (AS-IS)."""
        clone = super().clone_for_rollout()
        clone.mbom_opponent_model = deepcopy(self.mbom_opponent_model)
        return clone
