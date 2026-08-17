"""Episode payload for one negotiation phase of one agent.

This is the AS-IS ``Episode`` data type (``02-10``) with two additions required
by the new architecture:

* ``phase``       — which separated phase this payload belongs to:
                    ``"propose"`` | ``"respond"`` | ``"utility_threshold"``.
                    Propose and respond are **fully separate** (separate memory,
                    model, trainer); a single agent produces **two** payloads per
                    negotiation (DLST: **three**, adding ``utility_threshold``).
                    See ``00-MULAI-DI-SINI`` rule #5 and ``03-05``.
* ``.finalize()`` — populate the parallel convenience lists
                    (``states``/``actions``/``log_probs``/``values``/
                    ``q_values``/``utilities``) from ``steps`` and mark the last
                    step ``done=True`` so trainers can read either representation.

No loss / reward formula is affected by these additions.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from negotiation.data_type.step import Step


@dataclass
class Episode:
    """A single phase's trajectory for one agent in one negotiation.

    The parallel lists (``states``, ``actions``, ``log_probs``, ``values``,
    ``q_values``, ``utilities``) mirror the per-step fields in ``steps`` and are
    what the trainers consume (``episode.actions or [step.action for ...]``).
    Call :meth:`finalize` after appending all :class:`Step` objects to derive
    those lists and set the terminal ``done`` flag.

    Example:
        >>> ep = Episode(agent_name="fleet", phase="propose",
        ...              terminal_utility=0.8, agreement=(3,))
        >>> ep.steps.append(Step(state={"x": 1}, action=2, log_prob=-0.5,
        ...                      value=0.1, utility=0.4))
        >>> ep.steps.append(Step(state={"x": 2}, action=3, log_prob=-0.2,
        ...                      value=0.2, utility=0.8))
        >>> _ = ep.finalize()
        >>> ep.actions
        [2, 3]
        >>> ep.steps[-1].done
        True
        >>> ep.phase
        'propose'
    """

    agent_name: Any = None
    phase: Optional[str] = None  # "propose" | "respond" | "utility_threshold"
    terminal_utility: Any = None  # U(agreement) or -1 (no-deal penalty marker)
    agreement: Any = None  # NegMAS outcome tuple (idx,) / None
    # Per-negotiation reward-scaling bounds (C6). Stamped at store time by the
    # trainer from the owning agent's configure_negotiation() bounds so reward
    # scaling stays correct even when the episode buffer spans several
    # negotiations (train_every_n_episodes > 1 / replay). ``None`` = legacy
    # episode without bounds (the shaper falls back to its constructor bounds).
    utility_min: Optional[float] = None
    utility_max: Optional[float] = None
    states: Any = None
    actions: Any = None
    log_probs: Any = None
    values: Any = None
    q_values: Any = None
    utilities: Any = None
    steps: List[Step] = field(default_factory=list)
    meta: Any = None

    def finalize(self) -> "Episode":
        """Derive parallel lists from ``steps`` and flag the terminal step.

        Populates ``states``/``actions``/``log_probs``/``values``/``q_values``/
        ``utilities`` from the stored :class:`Step` objects (only when not
        already provided) and sets ``done=True`` on the last step. Returns
        ``self`` to allow chaining.

        Example:
            >>> ep = Episode(agent_name="spklu", phase="respond")
            >>> ep.steps.append(Step(state={}, action=1, log_prob=-0.1))
            >>> _ = ep.finalize()
            >>> ep.actions, ep.log_probs
            ([1], [-0.1])
        """
        if self.states is None:
            self.states = [step.state for step in self.steps]
        if self.actions is None:
            self.actions = [step.action for step in self.steps]
        if self.log_probs is None:
            self.log_probs = [step.log_prob for step in self.steps]
        if self.values is None:
            self.values = [step.value for step in self.steps]
        if self.q_values is None:
            self.q_values = [step.q_value for step in self.steps]
        if self.utilities is None:
            self.utilities = [step.utility for step in self.steps]

        if self.steps:
            for step in self.steps:
                step.done = False
            self.steps[-1].done = True

        return self
