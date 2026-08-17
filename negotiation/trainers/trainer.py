"""Base trainer interface + batch plumbing for the two input modes (``02-08`` / ``03-05``).

This module ports the AS-IS :class:`TrainerInterface` (ABC) and :class:`Trainer`
base from the paper code, and adds the *batch plumbing* required by the two input
modes so that the action distribution can be reconstructed **identically** between
sampling (during negotiation) and update (during training). This is essential for
PPO, which recomputes ``log_prob`` at update time (Contract C4 / ``03-05``).

What the plumbing does
----------------------
The new networks (:class:`~negotiation.models.encoders.feature_encoder.DiscreteActorNet`
/ :class:`~negotiation.models.encoders.feature_encoder.CriticNet`) consume a
**dict of batched tensors** (a packed ``RawObservation``), not a flat vector.
Each :class:`~negotiation.data_type.step.Step` stores its packed observation in
``state`` plus, depending on mode:

* **PADDED** — ``action_mask`` (length ``K_max`` boolean). At update time the
  same mask is re-applied via ``cand_mask`` so the masked ``Categorical`` is
  identical to sampling.
* **ENCODER** — ``candidate_features`` (``(K, F_cand)``). At update time the
  ragged per-step candidate sets are **padded to the max K in the batch** with a
  validity mask; the score head then reproduces the per-candidate logits exactly
  as during sampling.

:func:`collate_states` turns a list of per-step packed-observation dicts (with
optional per-step ``action_mask`` / ``candidate_features``) into one batched obs
dict ready to feed straight into the actor / critic. The concrete trainers
(``PolicyGradientTrainer`` / ``PPOTrainer``) call it so they can re-forward the
mode-aware nets; the loss formulas themselves are unchanged (``02-08``).

Per-phase separation (G2 / G3): a trainer instance is built **per phase**
(``propose`` vs. ``respond``, and DLST adds ``utility_threshold``). An agent holds
separate ``propose_trainer`` / ``respond_trainer`` (DLST + ``threshold_trainer``)
with no shared models / optimizers / episode buffers. The reward shaper is also
per phase (``02-09``).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:  # torch is an optional import at module-parse time (kept lazy-friendly).
    import torch
except Exception:  # pragma: no cover - torch always present at runtime
    torch = None  # type: ignore


# Keys of the packed observation dict that hold *set* blocks (pad along axis N).
_SET_FEATURE_KEYS = ("taxi_feats", "port_feats", "area_feats")
_SET_MASK_KEYS = ("taxi_mask", "port_mask", "area_mask")
# Keys that hold flat per-state vectors (just stacked along the batch axis).
_SCALAR_KEYS = ("scalar_feats", "progress_feats", "privileged_feats")


# =============================================================================
# TrainerInterface (AS-IS ABC)
# =============================================================================

class TrainerInterface(ABC):
    """Abstract trainer interface shared by all concrete trainers (``02-08``)."""

    @abstractmethod
    def store_episode(self, episode: Any) -> None:
        """Append one finalized :class:`Episode` to the trainer's buffer."""

    @abstractmethod
    def attach_models(self, dict_of_models: Dict[str, Any]) -> None:
        """Attach ``{key: {"model", "optimizer", "config"}}`` to the trainer."""

    @abstractmethod
    def attach_losses(self, dict_of_losses: Dict[str, Any]) -> None:
        """Attach optional loss-function overrides (else the paper defaults)."""

    @abstractmethod
    def attach_agent(self, agent: Any) -> None:
        """Attach the owning agent (used to (re)build packed states if needed)."""

    @abstractmethod
    def _episode_to_batch(self, episode: Any) -> Any:
        """Convert one episode into per-step training tensors."""

    @abstractmethod
    def build_batch(self) -> Any:
        """Re-collate all stored episodes into a batch / DataLoader."""

    @abstractmethod
    def update(self) -> Dict[str, Any]:
        """Run the (paper-exact) optimisation step(s) over the batch."""

    @abstractmethod
    def reset_episodes(self) -> None:
        """Clear the episode buffer (after an update)."""

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """Persist trainer state + each model/optimizer state_dict."""

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """Restore trainer/model/optimizer state from ``path``."""


# =============================================================================
# Trainer (AS-IS base container)
# =============================================================================

class Trainer:
    """Common state container for the concrete trainers (AS-IS, ``02-08``).

    Holds the per-phase models / optimizers / reward shaper / episode buffer. A
    trainer instance is **per phase** — there is no shared state between an
    agent's ``propose_trainer`` and ``respond_trainer`` (G2). The mode-aware
    batch plumbing helpers (:meth:`collate_states`) live here so both PG and PPO
    reuse them without changing any loss formula.

    Example:
        >>> t = Trainer(batch_size=16)
        >>> t.episodes
        []
        >>> t.batch_size
        16
    """

    def __init__(
        self,
        models: Optional[Dict[str, Any]] = None,
        optimizers: Optional[Dict[str, Any]] = None,
        reward_shaper: Any = None,
        losses: Optional[Dict[str, Any]] = None,
        episodes: Optional[List[Any]] = None,
        batch_size: Optional[int] = None,
        agent: Any = None,
    ) -> None:
        self.models = models
        self.optimizers = optimizers if optimizers is not None else {}
        self.reward_shaper = reward_shaper
        self.losses = losses if losses is not None else {}
        self.episodes = episodes if episodes is not None else []
        self.batch_size = batch_size
        self.agent = agent
        self.device = "cpu"

    # ------------------------------------------------------------------
    # C6 — per-negotiation reward-scaling bounds
    # ------------------------------------------------------------------

    def _stamp_reward_bounds(self, episode: Any) -> Any:
        """Stamp the owning agent's per-negotiation utility bounds onto ``episode``.

        The reward shaper is built once at factory time (``domain_spec=None`` ->
        default ``(0, 1)`` bounds), while the true per-negotiation bounds from
        ``compute_utility_bounds`` are injected into the *agent* via
        ``configure_negotiation`` (C3). Stamping them onto the episode at store
        time lets the shaper scale each episode with its own negotiation's
        bounds (C6), even when the buffer spans several negotiations.

        No-op when the episode already carries bounds or the agent is absent.
        """
        if episode is None or self.agent is None:
            return episode
        if getattr(episode, "utility_min", None) is not None:
            return episode
        umin = getattr(self.agent, "utility_min", None)
        umax = getattr(self.agent, "utility_max", None)
        if umin is None or umax is None:
            return episode
        try:
            episode.utility_min = float(umin)
            episode.utility_max = float(umax)
        except (TypeError, ValueError):
            pass
        return episode

    # ------------------------------------------------------------------
    # Mode plumbing (03-05): re-collate stored packed-obs dicts + masks /
    # candidate-features into one batched obs dict for re-forwarding the nets.
    # ------------------------------------------------------------------

    def _to_tensor(self, array: Any) -> "torch.Tensor":
        """Return a float tensor on ``self.device`` for an array-like input."""
        if torch is None:  # pragma: no cover
            raise RuntimeError("torch is required for trainer batch plumbing.")
        if isinstance(array, torch.Tensor):
            return array.to(dtype=torch.float32, device=self.device)
        return torch.as_tensor(
            np.asarray(array, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )

    def collate_states(
        self,
        states: Sequence[Dict[str, Any]],
        action_masks: Optional[Sequence[Any]] = None,
        candidate_features: Optional[Sequence[Any]] = None,
        mode: str = "PADDED",
    ) -> Dict[str, "torch.Tensor"]:
        """Collate per-step packed-obs dicts into one batched obs dict (C4 / 03-05).

        Each element of ``states`` is a single-step packed observation: a dict
        whose set blocks (``taxi_feats`` ``(N, F)`` etc.) and scalar blocks
        (``scalar_feats`` ``(F,)`` etc.) come straight from the agent's
        ``pack_step_obs``. The set blocks are **padded to the max count in the
        batch** along their element axis (with the per-block mask extended
        accordingly), and the scalar blocks are stacked. The result is a dict of
        ``(B, ...)`` tensors that can be fed directly into ``DiscreteActorNet`` /
        ``CriticNet`` to reproduce the sampling-time logits.

        Candidate handling reproduces the action distribution exactly:

        * ``mode="PADDED"``: ``cand_mask`` is rebuilt from ``action_masks``
          (each a length-``K_max`` boolean), so the ``Linear(K_max)`` logits are
          ``masked_fill``-ed identically to sampling.
        * ``mode="ENCODER"``: ``cand_feats`` / ``cand_mask`` are rebuilt from the
          ragged ``candidate_features`` (each ``(K, F_cand)``) by padding to the
          max K in the batch; the score head then yields identical per-candidate
          logits.

        Returns the batched obs dict (does not run any network).

        Example:
            >>> import numpy as np
            >>> t = Trainer()
            >>> s = {"taxi_feats": np.zeros((2, 3)), "taxi_mask": np.ones(2),
            ...      "port_feats": np.zeros((1, 2)), "port_mask": np.ones(1),
            ...      "area_feats": np.zeros((1, 2)), "area_mask": np.ones(1),
            ...      "scalar_feats": np.zeros(2), "progress_feats": np.zeros(1)}
            >>> batch = t.collate_states([s, s], mode="PADDED")
            >>> tuple(batch["taxi_feats"].shape)
            (2, 2, 3)
            >>> tuple(batch["scalar_feats"].shape)
            (2, 2)
        """
        if torch is None:  # pragma: no cover
            raise RuntimeError("torch is required for trainer batch plumbing.")

        out: Dict[str, torch.Tensor] = {}
        b = len(states)

        # ── Set blocks: pad to max-N-in-batch + extend the per-block mask. ──
        for feat_key, mask_key in zip(_SET_FEATURE_KEYS, _SET_MASK_KEYS):
            if feat_key not in states[0]:
                continue
            per_step = [np.asarray(s[feat_key], dtype=np.float32) for s in states]
            f_dim = per_step[0].shape[-1] if per_step[0].ndim == 2 else per_step[0].shape[-1]
            max_n = max((arr.shape[0] for arr in per_step), default=0)

            feats = np.zeros((b, max_n, f_dim), dtype=np.float32)
            mask = np.zeros((b, max_n), dtype=np.float32)
            for i, arr in enumerate(per_step):
                n = arr.shape[0]
                if n > 0:
                    feats[i, :n, :] = arr
                if mask_key in states[i]:
                    m = np.asarray(states[i][mask_key], dtype=np.float32).reshape(-1)
                    mask[i, : m.shape[0]] = m
                else:
                    mask[i, :n] = 1.0
            out[feat_key] = self._to_tensor(feats)
            out[mask_key] = self._to_tensor(mask).to(dtype=torch.bool)

        # ── Scalar / progress / privileged blocks: just stack. ──
        for key in _SCALAR_KEYS:
            if key not in states[0]:
                continue
            stacked = np.stack(
                [np.asarray(s[key], dtype=np.float32).reshape(-1) for s in states],
                axis=0,
            )
            out[key] = self._to_tensor(stacked)

        # ── Candidate blocks (mode-specific). ──
        if mode == "PADDED" and action_masks is not None:
            masks = [np.asarray(m, dtype=np.float32).reshape(-1) for m in action_masks]
            k_max = max((m.shape[0] for m in masks), default=0)
            cand_mask = np.zeros((b, k_max), dtype=np.float32)
            for i, m in enumerate(masks):
                cand_mask[i, : m.shape[0]] = m
            out["cand_mask"] = self._to_tensor(cand_mask).to(dtype=torch.bool)
        elif mode == "ENCODER" and candidate_features is not None:
            # Respond steps store candidate_features=None; skip if all are None.
            # (The respond head is 2-way and needs no candidate reconstruction.)
            _valid_cfs = [c for c in candidate_features if c is not None]
            if _valid_cfs:
                _f_ref = np.asarray(_valid_cfs[0], dtype=np.float32)
                f_cand = _f_ref.shape[-1] if _f_ref.ndim >= 1 else 1
                per_step = [
                    np.asarray(c, dtype=np.float32) if c is not None
                    else np.zeros((0, f_cand), dtype=np.float32)
                    for c in candidate_features
                ]
                max_k = max((arr.shape[0] for arr in per_step), default=0)
                cand_feats = np.zeros((b, max_k, f_cand), dtype=np.float32)
                cand_mask = np.zeros((b, max_k), dtype=np.float32)
                for i, arr in enumerate(per_step):
                    k = arr.shape[0]
                    if k > 0:
                        cand_feats[i, :k, :] = arr
                        cand_mask[i, :k] = 1.0
                out["cand_feats"] = self._to_tensor(cand_feats)
                out["cand_mask"] = self._to_tensor(cand_mask).to(dtype=torch.bool)

        return out
