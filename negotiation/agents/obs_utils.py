"""Observation packing / collating for the mode-aware networks (``03-05``).

At **sampling** time a discrete agent has a single packed observation dict (the
output of :meth:`BaseNegotiator.build_step_obs`) whose set blocks have shape
``(N, F)`` and scalar blocks ``(F,)``. The mode-aware
:class:`~negotiation.models.encoders.feature_encoder.DiscreteActorNet` /
:class:`~negotiation.models.encoders.feature_encoder.CriticNet` expect a batched
dict of tensors (leading batch dim ``B``). :func:`obs_to_batch` turns one packed
dict into a ``B = 1`` batched tensor dict so the same net used at update time can
be forwarded for a single step.

This pad-to-max-in-batch + mask logic for set blocks (taxi / port / area) and the
candidate blocks is intentionally the **same** as
:meth:`~negotiation.trainers.trainer.Trainer.collate_states` (Contract C4): the
distribution produced when sampling one step is reconstructed identically when
the trainer re-collates and re-forwards that step at update time. The helpers
here are the single-sample (``B = 1``) counterpart used inside the agents.

No loss / reward formula depends on these helpers; they only shape tensors.
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch always present at runtime
    torch = None  # type: ignore


# Set blocks padded along the element axis (with their masks).
SET_FEATURE_KEYS = ("taxi_feats", "port_feats", "area_feats")
SET_MASK_KEYS = ("taxi_mask", "port_mask", "area_mask")
# Flat per-state blocks (stacked along the batch axis only).
SCALAR_KEYS = ("scalar_feats", "progress_feats", "privileged_feats")
# Candidate blocks.
CAND_FEATURE_KEY = "cand_feats"
CAND_MASK_KEY = "cand_mask"


def _as_tensor(array: Any, device: str = "cpu") -> "torch.Tensor":
    """Return a float32 tensor on ``device`` for an array-like input.

    Example:
        >>> import numpy as np
        >>> t = _as_tensor(np.zeros((2, 3)))
        >>> tuple(t.shape)
        (2, 3)
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for obs packing.")
    if isinstance(array, torch.Tensor):
        return array.to(dtype=torch.float32, device=device)
    return torch.as_tensor(np.asarray(array, dtype=np.float32), device=device)


def obs_to_batch(
    obs: Dict[str, Any],
    device: str = "cpu",
) -> Dict[str, "torch.Tensor"]:
    """Turn one packed observation dict into a ``B = 1`` batched tensor dict.

    * Set blocks ``(N, F)`` -> ``(1, N, F)`` with mask ``(1, N)`` (bool); when a
      block has no mask, an all-true mask of the right length is created.
    * Scalar blocks ``(F,)`` -> ``(1, F)``.
    * ``cand_feats`` ``(K, F_cand)`` -> ``(1, K, F_cand)``; ``cand_mask`` ``(K,)``
      -> ``(1, K)`` (bool); a missing ``cand_mask`` becomes all-true of length K.

    Example:
        >>> import numpy as np
        >>> obs = {"taxi_feats": np.zeros((2, 3)), "taxi_mask": np.ones(2),
        ...        "scalar_feats": np.zeros(4), "progress_feats": np.zeros(3),
        ...        "cand_feats": np.zeros((5, 6)), "cand_mask": np.ones(5)}
        >>> batch = obs_to_batch(obs)
        >>> tuple(batch["taxi_feats"].shape), tuple(batch["cand_feats"].shape)
        ((1, 2, 3), (1, 5, 6))
        >>> batch["taxi_mask"].dtype == __import__("torch").bool
        True
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for obs packing.")
    out: Dict[str, torch.Tensor] = {}

    for feat_key, mask_key in zip(SET_FEATURE_KEYS, SET_MASK_KEYS):
        if feat_key not in obs:
            continue
        feats = np.asarray(obs[feat_key], dtype=np.float32)
        if feats.ndim == 1:
            feats = feats.reshape(0, feats.shape[0]) if feats.shape[0] == 0 else feats.reshape(1, -1)
        out[feat_key] = _as_tensor(feats[None, ...], device)
        n = feats.shape[0]
        if mask_key in obs and obs[mask_key] is not None:
            mask = np.asarray(obs[mask_key], dtype=np.float32).reshape(-1)
        else:
            mask = np.ones(n, dtype=np.float32)
        out[mask_key] = _as_tensor(mask[None, :], device).to(dtype=torch.bool)

    for key in SCALAR_KEYS:
        if key not in obs:
            continue
        vec = np.asarray(obs[key], dtype=np.float32).reshape(-1)
        out[key] = _as_tensor(vec[None, :], device)

    if CAND_FEATURE_KEY in obs and obs[CAND_FEATURE_KEY] is not None:
        cand = np.asarray(obs[CAND_FEATURE_KEY], dtype=np.float32)
        if cand.ndim == 1:
            cand = cand.reshape(0, 0) if cand.size == 0 else cand.reshape(1, -1)
        out[CAND_FEATURE_KEY] = _as_tensor(cand[None, ...], device)
        k = cand.shape[0]
        if CAND_MASK_KEY in obs and obs[CAND_MASK_KEY] is not None:
            cmask = np.asarray(obs[CAND_MASK_KEY], dtype=np.float32).reshape(-1)
        else:
            cmask = np.ones(k, dtype=np.float32)
        out[CAND_MASK_KEY] = _as_tensor(cmask[None, :], device).to(dtype=torch.bool)
    elif CAND_MASK_KEY in obs and obs[CAND_MASK_KEY] is not None:
        cmask = np.asarray(obs[CAND_MASK_KEY], dtype=np.float32).reshape(-1)
        out[CAND_MASK_KEY] = _as_tensor(cmask[None, :], device).to(dtype=torch.bool)

    return out


def action_mask_from_obs(obs: Dict[str, Any], k_max: int) -> np.ndarray:
    """Build a length-``k_max`` boolean PADDED mask from a packed obs (``03-05``).

    The first ``K`` slots follow ``cand_mask`` (or all-valid when absent), the
    rest are padding (``False``). Stored on each PADDED :class:`Step` so the
    ``Linear(K_max)`` logits are ``masked_fill``-ed identically at update time.

    Example:
        >>> import numpy as np
        >>> m = action_mask_from_obs({"cand_mask": np.array([1, 1, 0])}, k_max=5)
        >>> m.tolist()
        [True, True, False, False, False]
    """
    mask = np.zeros(int(k_max), dtype=bool)
    cmask = obs.get(CAND_MASK_KEY)
    if cmask is not None:
        cmask = np.asarray(cmask, dtype=np.float32).reshape(-1).astype(bool)
        k = min(cmask.shape[0], int(k_max))
        mask[:k] = cmask[:k]
    elif CAND_FEATURE_KEY in obs and obs[CAND_FEATURE_KEY] is not None:
        k = min(np.asarray(obs[CAND_FEATURE_KEY]).shape[0], int(k_max))
        mask[:k] = True
    return mask


def candidate_features_from_obs(obs: Dict[str, Any]) -> Optional[np.ndarray]:
    """Return the per-candidate feature matrix ``(K, F_cand)`` (ENCODER plumbing).

    Stored on each ENCODER :class:`Step` so the ragged candidate sets can be
    padded-to-max-in-batch and re-scored identically at update time (``03-05``).

    Example:
        >>> import numpy as np
        >>> candidate_features_from_obs({"cand_feats": np.zeros((4, 6))}).shape
        (4, 6)
        >>> candidate_features_from_obs({}) is None
        True
    """
    cand = obs.get(CAND_FEATURE_KEY)
    if cand is None:
        return None
    return np.asarray(cand, dtype=np.float32)
