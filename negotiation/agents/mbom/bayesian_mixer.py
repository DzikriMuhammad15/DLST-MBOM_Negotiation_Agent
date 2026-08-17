"""BayesianMixer — level-k opponent-model mixing (MBOM), AS-IS (paper Eq. 6-8).

Mixes the predictions of ``M`` IOP models (reasoning levels) into a single
opponent-action distribution. The Bayesian update of the mixing weights
``alpha`` and the ``softer-softmax`` smoothing are exactly the paper formulas;
nothing here changes with the input mode.

Formulas (AS-IS):

* posterior (Eq. 7): ``p(m | a^o) = pi~^o_m(a_o) * p(m) / sum_i [pi~^o_i(a_o) * p(i)]``;
* decayed moving average ``Psi^t_m = sum_{l=t-H}^{t-1} lambda^{t-l} * p(m | a^o_l)``;
* weights (Eq. 8): ``alpha = softer_softmax(Psi)``;
* mixed distribution (Eq. 6): ``pi~^o_mix = sum_m alpha_m * pi~^o_m``.
"""

from typing import List

import numpy as np


class BayesianMixer:
    """Bayesian level-k mixer over ``M`` IOP models (AS-IS).

    Parameters
    ----------
    M:
        Number of IOP reasoning levels.
    decay:
        ``lambda`` for the decayed moving average (Appendix B).
    horizon:
        History length ``H`` retained for the moving average / prior.
    temperature:
        Softer-softmax temperature (``>= 1`` flattens the distribution).

    Example:
        >>> import numpy as np
        >>> mx = BayesianMixer(M=2, decay=0.9, horizon=5, temperature=1.0)
        >>> mx.update(np.array([0.7, 0.3]))
        >>> probs = mx.get_mixed_probs([np.array([0.6, 0.4]), np.array([0.2, 0.8])])
        >>> round(float(probs.sum()), 5)
        1.0
        >>> mx.get_alpha().shape
        (2,)
    """

    def __init__(self, M: int, decay: float = 0.9, horizon: int = 10, temperature: float = 1.0) -> None:
        self.M = M
        self.decay = decay
        self.horizon = horizon
        self.temperature = temperature
        self.alpha: np.ndarray = np.ones(M, dtype=np.float64) / M
        self._psi: np.ndarray = np.ones(M, dtype=np.float64) / M
        self._prior: np.ndarray = np.ones(M, dtype=np.float64) / M
        self._history: List[np.ndarray] = []
        self._t: int = 0

    # ------------------------------------------------------------------
    # Bayesian update (Eq. 7)
    # ------------------------------------------------------------------

    def update(self, iop_probs_for_action: np.ndarray) -> None:
        """Update ``alpha`` from each IOP's probability of the real action (Eq. 7).

        ``iop_probs_for_action`` has shape ``(M,)``: per model ``m``, the
        probability ``pi~^o_m(a_o)`` of the action the opponent actually took.

        Example:
            >>> import numpy as np
            >>> mx = BayesianMixer(2, 0.9, 5, 1.0)
            >>> mx.update(np.array([0.9, 0.1]))
            >>> bool(mx.get_alpha().sum() > 0)
            True
        """
        numerators = iop_probs_for_action * self._prior
        denom = numerators.sum()
        if denom < 1e-10:
            posterior = np.ones(self.M, dtype=np.float64) / self.M
        else:
            posterior = numerators / denom

        self._history.append(posterior.copy())
        if len(self._history) > self.horizon:
            self._history.pop(0)

        self._t += 1

        if self._history:
            self._prior = np.mean(np.array(self._history), axis=0)

        psi = np.zeros(self.M, dtype=np.float64)
        for j, hist_posterior in enumerate(reversed(self._history)):
            psi += (self.decay ** (j + 1)) * hist_posterior
        self._psi = psi

        self.alpha = self._softer_softmax(self._psi)

    def _softer_softmax(self, x: np.ndarray) -> np.ndarray:
        """Temperature-scaled softmax (>=1 flattens the distribution) — AS-IS."""
        scaled = x / max(self.temperature, 1e-8)
        exp_x = np.exp(scaled - scaled.max())
        return exp_x / (exp_x.sum() + 1e-10)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def get_mixed_probs(self, iop_probs_list: List[np.ndarray]) -> np.ndarray:
        """Mixed distribution ``sum_m alpha_m * pi~^o_m`` (Eq. 6) — AS-IS.

        Example:
            >>> import numpy as np
            >>> mx = BayesianMixer(2, 0.9, 5, 1.0)
            >>> p = mx.get_mixed_probs([np.array([1.0, 0.0]), np.array([0.0, 1.0])])
            >>> [round(float(x), 3) for x in p]
            [0.5, 0.5]
        """
        mixed = np.zeros_like(iop_probs_list[0], dtype=np.float64)
        for m, probs in enumerate(iop_probs_list):
            mixed += float(self.alpha[m]) * np.array(probs, dtype=np.float64)
        total = mixed.sum()
        if total < 1e-10:
            return np.ones(len(mixed)) / len(mixed)
        return (mixed / total).astype(np.float32)

    def get_mixed_action_idx(self, iop_probs_list: List[np.ndarray]) -> int:
        """Sample one action index from the mixed distribution (rollout helper)."""
        mixed_probs = self.get_mixed_probs(iop_probs_list)
        mixed_probs = mixed_probs / mixed_probs.sum()
        return int(np.random.choice(len(mixed_probs), p=mixed_probs))

    def get_alpha(self) -> np.ndarray:
        """Return a copy of the current mixing weights ``alpha``."""
        return self.alpha.copy()

    def reset(self) -> None:
        """Reset the mixer state for a new negotiation."""
        self.alpha = np.ones(self.M, dtype=np.float64) / self.M
        self._psi = np.ones(self.M, dtype=np.float64) / self.M
        self._prior = np.ones(self.M, dtype=np.float64) / self.M
        self._history.clear()
        self._t = 0
