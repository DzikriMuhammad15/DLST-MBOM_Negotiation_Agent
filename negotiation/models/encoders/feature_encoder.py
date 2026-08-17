"""Mode-aware front-ends, discrete actor, and critic (``03-01`` / ``03-02`` / ``03-06``).

This module wires the raw, variable-size observation blocks produced by the
bridge (``RawObservation``) into fixed-dimension networks, switching the
front-end between the two input modes (``PADDED`` / ``ENCODER``) without changing
any core network or loss (Gotcha G13: the mode only changes the input front-end
and the discrete *propose* head; the respond head is always a 2-way classifier
and the DLST continuous head is untouched).

Components
----------
* :class:`FeatureDims`     — per-block feature dimensions + encoder/padding caps.
* :class:`ModeFrontEnd`    — encodes the set blocks (taxi/port/area) + scalar +
                             progress blocks into a fixed-dim ``context`` vector,
                             using set-encoders (ENCODER) or padded encoders
                             (PADDED). Set-encoders are sub-modules (G5).
* :class:`DiscreteActorNet`— front-end + propose/respond head. Propose head:
                             PADDED -> ``Linear(K_max)`` + mask; ENCODER ->
                             :class:`ScoreHead`. Respond head: ``Linear(2)``.
* :class:`CriticNet`       — front-end + ``V(s)`` head, with the two optional
                             enrichment axes (candidate pool / privileged info),
                             both **V(s)** only and default off (G6).

The propose and respond actors are **separate** ``DiscreteActorNet`` instances
(separate parameters / front-ends), preserving the fully-separated propose/respond
invariant at the network level.

A ``RawObservation`` here is a dict of (already batched) tensors:

    {
      "taxi_feats": (B, N, F_taxi),   "taxi_mask": (B, N),
      "port_feats": (B, P, F_port),   "port_mask": (B, P),
      "area_feats": (B, A, F_area),   "area_mask": (B, A),
      "scalar_feats":   (B, F_scalar),
      "progress_feats": (B, F_progress),
      "cand_feats": (B, K, F_cand),   "cand_mask": (B, K),   # propose / critic-pool
      "privileged_feats": (B, F_priv),                        # critic only (G6 axis B)
    }
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from negotiation.models.encoders.padded_encoder import PaddedSetEncoder
from negotiation.models.encoders.score_head import MASK_FILL_VALUE, ScoreHead
from negotiation.models.encoders.set_encoder import SetEncoder

# Allowed input modes (kept as constants to avoid magic strings).
MODE_PADDED: str = "PADDED"
MODE_ENCODER: str = "ENCODER"


@dataclass(frozen=True)
class FeatureDims:
    """Per-block feature dimensions and encoder/padding caps.

    These dimensions are emitted by the bridge ``feature_builder`` and must match
    exactly what the networks are built with (Contract C7 — part of the agent
    signature). ``f_privileged`` is only used when the critic privileged-info
    axis is enabled (``03-06``).

    Example:
        >>> dims = FeatureDims(f_taxi=5, f_port=3, f_area=3, f_scalar=2,
        ...                    f_progress=4, f_cand=6, n_max=10, p_max=4,
        ...                    a_max=3, k_max=16, phi_hidden_dim=32,
        ...                    pooling="mean")
        >>> dims.k_max
        16
        >>> dims.pooling
        'mean'
    """

    f_taxi: int
    f_port: int
    f_area: int
    f_scalar: int
    f_progress: int
    f_cand: int
    n_max: int
    p_max: int
    a_max: int
    k_max: int
    phi_hidden_dim: int = 32
    pooling: str = "mean"
    f_privileged: int = 0
    # Per-element embedding (output) width of phi. None -> equals phi_hidden_dim
    # (historical behaviour). Kept last so positional construction stays valid.
    phi_out_dim: Optional[int] = None
    # Number of layers in the phi MLP (minimum 1). Default 2 preserves as-is.
    phi_n_layers: int = 2


class ModeFrontEnd(nn.Module):
    """Encode the variable-size observation blocks into a fixed-dim context.

    For each set block (taxi/port/area) a per-element encoder is applied and
    masked-pooled to a fixed-size vector; the results are concatenated with the
    scalar and progress blocks. ENCODER mode uses :class:`SetEncoder`; PADDED mode
    uses :class:`PaddedSetEncoder`. The output dimension is exposed as
    :attr:`out_dim` (= ``context`` width, the ``state_dim`` consumed by heads).

    Example:
        >>> import torch
        >>> dims = FeatureDims(f_taxi=4, f_port=3, f_area=2, f_scalar=2,
        ...                    f_progress=3, f_cand=5, n_max=6, p_max=4,
        ...                    a_max=3, k_max=8, phi_hidden_dim=8, pooling="mean")
        >>> fe = ModeFrontEnd(dims, mode="ENCODER")
        >>> obs = {
        ...     "taxi_feats": torch.zeros(2, 3, 4),
        ...     "taxi_mask":  torch.ones(2, 3).bool(),
        ...     "port_feats": torch.zeros(2, 2, 3),
        ...     "port_mask":  torch.ones(2, 2).bool(),
        ...     "area_feats": torch.zeros(2, 1, 2),
        ...     "area_mask":  torch.ones(2, 1).bool(),
        ...     "scalar_feats":   torch.zeros(2, 2),
        ...     "progress_feats": torch.zeros(2, 3),
        ... }
        >>> ctx = fe(obs)
        >>> ctx.shape[0]
        2
        >>> ctx.shape[1] == fe.out_dim
        True
    """

    def __init__(self, dims: FeatureDims, mode: str) -> None:
        super().__init__()
        if mode not in (MODE_PADDED, MODE_ENCODER):
            raise ValueError(
                f"mode must be {MODE_PADDED!r} or {MODE_ENCODER!r}, not {mode!r}"
            )
        self.dims = dims
        self.mode = mode
        h = dims.phi_hidden_dim
        # Per-element embedding width; defaults to the hidden width.
        out = dims.phi_out_dim if dims.phi_out_dim is not None else h
        pooling = dims.pooling
        phi_n_layers = dims.phi_n_layers

        encoder_cls = SetEncoder if mode == MODE_ENCODER else PaddedSetEncoder
        self.taxi_encoder = encoder_cls(dims.f_taxi, h, pooling, out_dim=out,
                                        phi_n_layers=phi_n_layers)
        self.port_encoder = encoder_cls(dims.f_port, h, pooling, out_dim=out,
                                        phi_n_layers=phi_n_layers)
        self.area_encoder = encoder_cls(dims.f_area, h, pooling, out_dim=out,
                                        phi_n_layers=phi_n_layers)

        # context = [h_taxi, h_port, h_area, scalar, progress]
        self.out_dim = 3 * out + dims.f_scalar + dims.f_progress

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the fixed-dim context ``(B, out_dim)`` for the observation dict.

        Example:
            >>> import torch
            >>> dims = FeatureDims(2, 2, 2, 1, 1, 3, 4, 4, 4, 4, 4, "mean")
            >>> fe = ModeFrontEnd(dims, mode="PADDED")
            >>> obs = {
            ...     "taxi_feats": torch.zeros(1, 0, 2),
            ...     "taxi_mask":  torch.zeros(1, 0).bool(),
            ...     "port_feats": torch.zeros(1, 1, 2),
            ...     "port_mask":  torch.ones(1, 1).bool(),
            ...     "area_feats": torch.zeros(1, 1, 2),
            ...     "area_mask":  torch.ones(1, 1).bool(),
            ...     "scalar_feats":   torch.zeros(1, 1),
            ...     "progress_feats": torch.zeros(1, 1),
            ... }
            >>> tuple(fe(obs).shape)
            (1, 14)
        """
        h_taxi = self.taxi_encoder(obs["taxi_feats"], obs.get("taxi_mask"))
        h_port = self.port_encoder(obs["port_feats"], obs.get("port_mask"))
        h_area = self.area_encoder(obs["area_feats"], obs.get("area_mask"))
        scalar = obs["scalar_feats"]
        progress = obs["progress_feats"]
        return torch.cat([h_taxi, h_port, h_area, scalar, progress], dim=-1)


class DiscreteActorNet(nn.Module):
    """Mode-aware discrete actor for one phase (propose **or** respond).

    The propose and respond phases use **separate** instances of this network
    (separate front-end + head parameters), preserving full propose/respond
    separation at the network level.

    * ``phase="propose"``:
        - PADDED head: ``Linear(hidden, K_max)`` then ``masked_fill`` with the
          candidate mask (``cand_mask`` padded to ``K_max``);
        - ENCODER head: :class:`ScoreHead` producing one logit per candidate.
    * ``phase="respond"``: ``Linear(hidden, 2)`` (``0 = accept``, ``1 = reject``);
      independent of K and of the chosen mode.

    The returned logits are intended for ``torch.distributions.Categorical`` and
    are reconstructed identically at update time using the stored mask (PADDED) /
    candidate features (ENCODER), so PPO's recomputed ``log_prob`` matches the
    sampling distribution (Contract C4, ``03-05``).

    Example:
        >>> import torch
        >>> dims = FeatureDims(2, 2, 2, 1, 1, 3, 4, 4, 4, 4, 8, "mean")
        >>> actor = DiscreteActorNet(dims, mode="ENCODER", phase="propose",
        ...                          hidden_dim=16)
        >>> obs = {
        ...     "taxi_feats": torch.zeros(1, 2, 2), "taxi_mask": torch.ones(1, 2).bool(),
        ...     "port_feats": torch.zeros(1, 1, 2), "port_mask": torch.ones(1, 1).bool(),
        ...     "area_feats": torch.zeros(1, 1, 2), "area_mask": torch.ones(1, 1).bool(),
        ...     "scalar_feats": torch.zeros(1, 1), "progress_feats": torch.zeros(1, 1),
        ...     "cand_feats": torch.zeros(1, 3, 3), "cand_mask": torch.ones(1, 3).bool(),
        ... }
        >>> tuple(actor(obs).shape)
        (1, 3)
    """

    def __init__(
        self,
        dims: FeatureDims,
        mode: str,
        phase: str,
        hidden_dim: int = 128,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        if phase not in ("propose", "respond"):
            raise ValueError(
                f"phase must be 'propose' or 'respond', not {phase!r}"
            )
        self.dims = dims
        self.mode = mode
        self.phase = phase
        self.front_end = ModeFrontEnd(dims, mode)
        ctx_dim = self.front_end.out_dim

        # Shared trunk turning context -> hidden representation for the head.
        # n_layers controls number of hidden layers in the trunk.
        trunk_layers = []
        trunk_layers.append(nn.Linear(ctx_dim, hidden_dim))
        trunk_layers.append(nn.ReLU())
        for _ in range(n_layers - 1):
            trunk_layers.append(nn.Linear(hidden_dim, hidden_dim))
            trunk_layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*trunk_layers)

        if phase == "respond":
            # Respond: 2-way head, mode-independent.
            self.respond_head: Optional[nn.Linear] = nn.Linear(hidden_dim, 2)
            self.padded_head = None
            self.score_head = None
        else:
            # Propose: mode-specific head.
            self.respond_head = None
            if mode == MODE_PADDED:
                self.padded_head: Optional[nn.Linear] = nn.Linear(
                    hidden_dim, dims.k_max
                )
                self.score_head = None
            else:
                self.padded_head = None
                self.score_head: Optional[ScoreHead] = ScoreHead(
                    context_dim=hidden_dim,
                    candidate_feature_dim=dims.f_cand,
                    hidden_dim=hidden_dim,
                )

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return action logits for ``obs`` (shape depends on phase / mode).

        Respond -> ``(B, 2)``. Propose PADDED -> ``(B, K_max)`` (masked).
        Propose ENCODER -> ``(B, K)`` (masked per candidate).
        """
        context = self.front_end(obs)
        hidden = self.trunk(context)

        if self.phase == "respond":
            return self.respond_head(hidden)

        if self.mode == MODE_PADDED:
            logits = self.padded_head(hidden)  # (B, K_max)
            mask = obs.get("cand_mask")
            if mask is not None:
                mask_bool = mask.to(dtype=torch.bool)
                logits = logits.masked_fill(~mask_bool, MASK_FILL_VALUE)
            return logits

        # ENCODER propose: pointer-style score head.
        return self.score_head(
            hidden,
            obs["cand_feats"],
            obs.get("cand_mask"),
        )


class CriticNet(nn.Module):
    """Mode-aware critic producing ``V(s)`` (PPO), with optional enrichment (G6).

    The critic has its **own** front-end (separate from the actor's), so it rides
    independently in ``state_dict`` (G5). Two optional, independent enrichment
    axes (default **off**, ``03-06``) both stay ``V(s)`` (never ``Q(s, a)``):

    * ``critic_sees_candidate_pool``: appends a masked-pooled summary of
      ``cand_feats`` (the available menu) to the context.
    * ``critic_privileged_info``: appends an extra ground-truth feature block
      (``privileged_feats``, width ``dims.f_privileged``) that the actor never
      sees (asymmetric actor-critic, training only).

    Example:
        >>> import torch
        >>> dims = FeatureDims(2, 2, 2, 1, 1, 3, 4, 4, 4, 4, 8, "mean")
        >>> critic = CriticNet(dims, mode="PADDED", hidden_dim=16)
        >>> obs = {
        ...     "taxi_feats": torch.zeros(2, 2, 2), "taxi_mask": torch.ones(2, 2).bool(),
        ...     "port_feats": torch.zeros(2, 1, 2), "port_mask": torch.ones(2, 1).bool(),
        ...     "area_feats": torch.zeros(2, 1, 2), "area_mask": torch.ones(2, 1).bool(),
        ...     "scalar_feats": torch.zeros(2, 1), "progress_feats": torch.zeros(2, 1),
        ... }
        >>> tuple(critic(obs).shape)
        (2,)
    """

    def __init__(
        self,
        dims: FeatureDims,
        mode: str,
        hidden_dim: int = 128,
        critic_sees_candidate_pool: bool = False,
        critic_privileged_info: bool = False,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.dims = dims
        self.mode = mode
        self.critic_sees_candidate_pool = critic_sees_candidate_pool
        self.critic_privileged_info = critic_privileged_info

        self.front_end = ModeFrontEnd(dims, mode)
        in_dim = self.front_end.out_dim

        # Axis A: pooled candidate-menu summary (still V(s); extra context only).
        if critic_sees_candidate_pool:
            encoder_cls = SetEncoder if mode == MODE_ENCODER else PaddedSetEncoder
            self.cand_pool_encoder: Optional[nn.Module] = encoder_cls(
                dims.f_cand, dims.phi_hidden_dim, dims.pooling
            )
            in_dim += dims.phi_hidden_dim
        else:
            self.cand_pool_encoder = None

        # Axis B: privileged ground-truth block, appended to context.
        if critic_privileged_info:
            in_dim += dims.f_privileged

        # Build head dynamically with n_layers hidden layers.
        head_layers = []
        head_layers.append(nn.Linear(in_dim, hidden_dim))
        head_layers.append(nn.ReLU())
        for _ in range(n_layers - 1):
            head_layers.append(nn.Linear(hidden_dim, hidden_dim))
            head_layers.append(nn.ReLU())
        head_layers.append(nn.Linear(hidden_dim, 1))
        self.head = nn.Sequential(*head_layers)

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return ``V(s)`` of shape ``(B,)`` for the observation dict ``obs``.

        Enrichment blocks are concatenated only when their flag is enabled.
        """
        context = self.front_end(obs)
        parts = [context]

        if self.cand_pool_encoder is not None:
            pooled = self.cand_pool_encoder(
                obs["cand_feats"], obs.get("cand_mask")
            )
            parts.append(pooled)

        if self.critic_privileged_info:
            parts.append(obs["privileged_feats"])

        x = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        return self.head(x).squeeze(-1)
