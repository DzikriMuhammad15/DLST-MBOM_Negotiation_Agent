"""Per-step trajectory record for a single negotiation move.

This is the AS-IS ``Step`` data type from the paper code (``02-10``) with the
only additions being the *mode plumbing* fields required by the two input modes
(PADDED / ENCODER) and the batch-storage plumbing described in ``03-05``:

* ``action_mask``        — boolean validity mask over the padded candidate slots
                           (mode PADDED, length ``K_max``). Lets the trainer
                           re-apply ``masked_fill`` before rebuilding the
                           ``Categorical`` distribution at update time.
* ``candidate_features`` — per-candidate feature matrix (mode ENCODER, shape
                           ``(K, F_cand)``). Lets ``build_batch`` pad-to-max in
                           the batch and rebuild the score-head logits exactly as
                           during sampling.
* ``reward``             — optional cached scalar reward for this step (the
                           authoritative reward is still computed by the
                           ``RewardShaper`` from utilities; this field is purely
                           a convenience cache and never changes any formula).
* ``done``               — whether this step terminates the episode (last move).
* ``meta``               — free-form per-step metadata (debug / bookkeeping).

These fields **do not change any loss or reward formula**; they only let the
action distribution be reconstructed consistently between sampling and update
(important for PPO recomputing ``log_prob``). See ``03-05``.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class Step:
    """One step in a negotiation trajectory (propose **or** respond phase).

    The propose phase stores a discrete ``action`` = library index; the respond
    phase stores ``action`` in ``{0 = accept, 1 = reject}``. The two phases are
    kept fully separate (separate memories / trainers); a single ``Step`` only
    ever belongs to one phase.

    Example:
        >>> s = Step(action=3, utility=0.7, log_prob=-1.2, done=False)
        >>> s.action, s.done
        (3, False)
        >>> s2 = Step(action=0, action_mask=[True, True, False])
        >>> s2.action_mask
        [True, True, False]
    """

    # ── AS-IS core fields (02-10) ────────────────────────────────────────────
    state: Any = None        # packed observation dict / SAOState dict / vector
    action: Any = None        # discrete index (propose/respond) or action_vec (DLST)
    utility: Any = None        # U(offer/agreement) from this agent's viewpoint
    value: Any = None        # V(s) when available (PPO/PG)
    q_value: Any = None        # Q(s, a) when available (DDPG/DLST)
    log_prob: Any = None        # log-probability of the sampled action
    next_state: Any = None        # packed observation of the resulting state

    # ── Mode plumbing fields (03-05) ─────────────────────────────────────────
    action_mask: Any = None        # PADDED: boolean validity mask, length K_max
    candidate_features: Any = None  # ENCODER: per-candidate features, (K, F_cand)

    # ── Episode-control / bookkeeping fields ─────────────────────────────────
    reward: Any = None        # optional cached scalar reward for this step
    done: bool = False       # True on the terminal step of the episode
    meta: Any = None        # free-form per-step metadata
