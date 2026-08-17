"""FrequencyOpponentModel — opponent utility approximation (DLST-ANESIA, AS-IS).

Distribution-based frequency model (Tunali et al., 2017, as referenced by the
DLST-ANESIA paper, Sec. 5.4). It estimates the opponent's issue weights and
per-value evaluations purely from the history of the opponent's offers, with no
access to the opponent's true preferences:

    U_o(omega) = sum_i w_i * e_i(omega[i])   (normalized to [0, 1])

Role in the project (unchanged): this object is **shared with the respond reward
shaper** (its ``opponent_model`` argument), which needs ``U_o`` to compute
``r_acc`` (paper Eq. 10), and it is also used by the Pareto bidding tactic.

New-architecture note: the negotiation outcome is the single-issue bundle index
``(idx,)``. Offers are accepted as ``(idx,)`` tuples, ``int``, or ``{issue: value}``
dicts and are normalized to the ``{issue_name: value}`` form internally so the
frequency formulas stay exactly AS-IS regardless of how NegMAS passes the offer.
"""

from typing import Any, Dict, List, Optional

# Default single-issue name for the bundle-enumeration outcome space (``03-04``).
DEFAULT_ISSUE_NAME: str = "bundle_index"


class FrequencyOpponentModel:
    """Frequency-based estimate of the opponent utility ``U_o`` (AS-IS).

    Parameters
    ----------
    issue_names:
        Ordered issue names (for the bundle outcome this is
        ``["bundle_index"]``).
    value_lists:
        For each issue, the list of its possible values.

    Example:
        >>> om = FrequencyOpponentModel(["bundle_index"], {"bundle_index": [0, 1, 2]})
        >>> om.update((1,)); om.update((1,)); om.update((0,))
        >>> 0.0 <= om.estimate_utility((1,)) <= 1.0
        True
    """

    def __init__(self, issue_names: List[str], value_lists: Dict[str, List[Any]]):
        self.issue_names = list(issue_names) if issue_names else [DEFAULT_ISSUE_NAME]
        self.value_lists = {k: list(v) for k, v in (value_lists or {}).items()}
        self.freq: Dict[str, Dict[Any, int]] = {
            issue: {val: 0 for val in self.value_lists.get(issue, [])}
            for issue in self.issue_names
        }
        self.total_bids = 0

    # ------------------------------------------------------------------
    # Offer normalization (tuple / int / dict -> {issue: value})
    # ------------------------------------------------------------------

    def _as_issue_dict(self, offer: Any) -> Optional[Dict[str, Any]]:
        """Normalize a NegMAS outcome into a ``{issue_name: value}`` dict."""
        if offer is None:
            return None
        if isinstance(offer, dict):
            return offer
        if isinstance(offer, (tuple, list)):
            if len(offer) == 0:
                return None
            return {name: offer[i] for i, name in enumerate(self.issue_names) if i < len(offer)}
        # scalar -> first issue
        return {self.issue_names[0]: offer}

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, opponent_offer: Any) -> None:
        """Record one opponent offer into the frequency statistics (AS-IS).

        Example:
            >>> om = FrequencyOpponentModel(["bundle_index"], {"bundle_index": [0, 1]})
            >>> om.update((0,)); om.total_bids
            1
        """
        offer = self._as_issue_dict(opponent_offer)
        if offer is None:
            return
        for issue in self.issue_names:
            if issue not in offer:
                continue
            val = offer[issue]
            if issue not in self.freq:
                self.freq[issue] = {}
            if val not in self.freq[issue]:
                self.freq[issue][val] = 0
            self.freq[issue][val] += 1
        self.total_bids += 1

    # ------------------------------------------------------------------
    # Component estimates
    # ------------------------------------------------------------------

    def _value_evaluations(self, issue: str) -> Dict[Any, float]:
        """Rank-based value evaluations ``e_i(v)`` in ``(0, 1]`` (AS-IS)."""
        counts = self.freq.get(issue, {})
        values = self.value_lists.get(issue, list(counts.keys()))
        k = len(values)
        if k == 0:
            return {}
        sorted_vals = sorted(values, key=lambda v: counts.get(v, 0))
        evals: Dict[Any, float] = {}
        for rank, v in enumerate(sorted_vals, start=1):
            evals[v] = rank / k
        return evals

    def _issue_weights(self) -> Dict[str, float]:
        """Herfindahl-like issue weights normalized to sum 1 (AS-IS)."""
        raw: Dict[str, float] = {}
        for issue in self.issue_names:
            counts = self.freq.get(issue, {})
            total = sum(counts.values())
            if total <= 0:
                raw[issue] = 0.0
                continue
            concentration = sum((c / total) ** 2 for c in counts.values())
            raw[issue] = concentration
        s = sum(raw.values())
        if s <= 1e-12:
            n = len(self.issue_names)
            return {issue: 1.0 / n for issue in self.issue_names} if n > 0 else {}
        return {issue: raw[issue] / s for issue in self.issue_names}

    # ------------------------------------------------------------------
    # Utility estimate
    # ------------------------------------------------------------------

    def estimate_utility(self, offer: Any) -> float:
        """Estimate ``U_o(offer)`` in ``[0, 1]`` (0.5 neutral before any data).

        Example:
            >>> om = FrequencyOpponentModel(["bundle_index"], {"bundle_index": [0, 1]})
            >>> om.estimate_utility((0,))     # no data yet -> neutral
            0.5
        """
        offer_dict = self._as_issue_dict(offer)
        if offer_dict is None:
            return 0.0
        if self.total_bids == 0:
            return 0.5
        weights = self._issue_weights()
        u = 0.0
        for issue in self.issue_names:
            if issue not in offer_dict:
                continue
            evals = self._value_evaluations(issue)
            e = evals.get(offer_dict[issue], 0.0)
            u += weights.get(issue, 0.0) * e
        return float(min(1.0, max(0.0, u)))

    def reset(self) -> None:
        """Reset all frequency statistics (e.g. at the start of an episode)."""
        self.freq = {
            issue: {val: 0 for val in self.value_lists.get(issue, [])}
            for issue in self.issue_names
        }
        self.total_bids = 0
