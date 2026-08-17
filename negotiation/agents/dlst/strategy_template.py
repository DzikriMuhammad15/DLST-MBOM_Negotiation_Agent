"""StrategyTemplateSpec — DLST-ANESIA continuous action layout (paper Eq. 21).

AS-IS from the paper code. The DDPG strategy actor outputs a flat vector
``[delta_1..delta_n, c_{1,1}..c_{n,n_i}, p-flat...]``; this spec defines the
layout and decodes it into ``(deltas, choices, params)``. The head is **not**
affected by the input mode (Gotcha G13) — only the front-end that produces the
state vector changes between PADDED / ENCODER.

Layout::

    n_delta  = n_phases
    n_choice = sum_i tactics_per_phase[i]
    n_param  = sum_i tactics_per_phase[i] * params_per_tactic
    action_dim = n_delta + n_choice + n_param
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class StrategyTemplateSpec:
    """Layout of the DLST continuous strategy action vector (paper Eq. 21).

    Parameters
    ----------
    n_phases:
        Number of negotiation phases ``n``.
    tactics_per_phase:
        ``n_i`` (number of tactics) for each phase.
    tactic_names_per_phase:
        Tactic names occupying each slot of each phase (acceptance / bidding
        tactic libraries differ, so this is provided per strategy).
    params_per_tactic:
        Numeric parameters per tactic (default 2: ``a``, ``b``).

    Example:
        >>> spec = StrategyTemplateSpec(n_phases=2, tactics_per_phase=[2, 1],
        ...     tactic_names_per_phase=[["boulware", "pareto"], ["random_above"]])
        >>> spec.action_dim, spec.split_sizes
        (11, (2, 3, 6))
    """

    n_phases: int
    tactics_per_phase: List[int]
    tactic_names_per_phase: List[List[str]]
    params_per_tactic: int = 2

    def __post_init__(self) -> None:
        assert self.n_phases == len(self.tactics_per_phase), (
            "tactics_per_phase must have length n_phases"
        )
        assert self.n_phases == len(self.tactic_names_per_phase), (
            "tactic_names_per_phase must have length n_phases"
        )
        for i in range(self.n_phases):
            assert self.tactics_per_phase[i] == len(self.tactic_names_per_phase[i]), (
                f"phase {i} tactic count mismatch"
            )

    @property
    def n_delta(self) -> int:
        """Number of phase-duration values."""
        return self.n_phases

    @property
    def n_choice(self) -> int:
        """Total number of binary tactic-choice values."""
        return sum(self.tactics_per_phase)

    @property
    def n_param(self) -> int:
        """Total number of tactic parameter values."""
        return sum(self.tactics_per_phase) * self.params_per_tactic

    @property
    def action_dim(self) -> int:
        """Total flat action dimension."""
        return self.n_delta + self.n_choice + self.n_param

    @property
    def split_sizes(self) -> Tuple[int, int, int]:
        """``(n_delta, n_choice, n_param)`` for the DDPG actor head."""
        return (self.n_delta, self.n_choice, self.n_param)

    def decode(self, action_vec):
        """Split a flat ``action_vec`` into ``(deltas, choices, params)`` (AS-IS).

        Example:
            >>> spec = StrategyTemplateSpec(1, [1], [["u_bar"]])
            >>> d, c, p = spec.decode([0.5, 0.9, 0.1, 0.2])
            >>> d, c, p
            ([0.5], [[0.9]], [[[0.1, 0.2]]])
        """
        v = list(action_vec)
        assert len(v) == self.action_dim, (
            f"action_vec len={len(v)} != action_dim={self.action_dim}"
        )
        idx = 0
        deltas = v[idx:idx + self.n_delta]
        idx += self.n_delta

        choices = []
        for i in range(self.n_phases):
            ni = self.tactics_per_phase[i]
            choices.append(v[idx:idx + ni])
            idx += ni

        params = []
        for i in range(self.n_phases):
            ni = self.tactics_per_phase[i]
            phase_params = []
            for _ in range(ni):
                phase_params.append(v[idx:idx + self.params_per_tactic])
                idx += self.params_per_tactic
            params.append(phase_params)

        return deltas, choices, params

    def phase_bounds(self, deltas):
        """Return ``[(t_start, t_end), ...]`` from normalized durations (AS-IS).

        Example:
            >>> StrategyTemplateSpec(2, [1, 1], [["u_bar"], ["u_fixed"]]).phase_bounds([1.0, 1.0])
            [(0.0, 0.5), (0.5, 1.0)]
        """
        s = sum(deltas)
        if s <= 1e-8:
            norm = [1.0 / self.n_phases] * self.n_phases
        else:
            norm = [d / s for d in deltas]

        bounds = []
        t = 0.0
        for i in range(self.n_phases):
            t_start = t
            t_end = t + norm[i]
            bounds.append((t_start, t_end))
            t = t_end
        if bounds:
            ts, _ = bounds[-1]
            bounds[-1] = (ts, 1.0)
        return bounds

    def active_phase(self, t, deltas) -> int:
        """Return the 0-based index of the active phase at time ``t`` (AS-IS).

        Example:
            >>> StrategyTemplateSpec(2, [1, 1], [["u_bar"], ["u_fixed"]]).active_phase(0.7, [1.0, 1.0])
            1
        """
        bounds = self.phase_bounds(deltas)
        for i, (ts, te) in enumerate(bounds):
            if i < self.n_phases - 1:
                if ts <= t < te:
                    return i
            else:
                if ts <= t <= te:
                    return i
        return self.n_phases - 1
