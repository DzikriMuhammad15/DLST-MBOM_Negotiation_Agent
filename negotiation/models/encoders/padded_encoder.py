"""Padded set front-end for the PADDED input mode (``03-01`` / ``03-02``).

In PADDED mode each variable-size set is padded to a fixed cap (``N_max`` /
``P_max`` / ``A_max`` / ``K_max``) and accompanied by a boolean mask. This module
encodes one such padded set into a fixed-dimension context vector using a shared
per-element MLP followed by masked pooling (mean / max), so padding elements are
ignored exactly as required by ``03-02``.

This mirrors :class:`~negotiation.models.encoders.set_encoder.SetEncoder` but is
named separately so that PADDED and ENCODER front-ends remain conceptually
distinct per ``03-02`` (and so checkpoints / signatures can record which one a
network uses). The all-masked case (no valid element) returns zeros (G12).

The encoder is a ``nn.Module`` sub-module of the actor / critic and therefore
rides in ``state_dict`` (G5).
"""

from typing import Optional

import torch
import torch.nn as nn


class PaddedSetEncoder(nn.Module):
    """Encode a padded, masked set ``(B, N_max, F)`` to a fixed-dim context.

    Parameters
    ----------
    feature_dim:
        Per-element feature dimension ``F``.
    hidden_dim:
        Hidden width of the per-element MLP and output context dimension.
    pooling:
        ``"mean"`` or ``"max"`` masked pooling over the padded slots.
    phi_n_layers:
        Number of layers in the ``phi`` MLP (minimum 1). Default 2 preserves
        the as-is 2-layer structure. phi_n_layers < 1 raises ValueError.

    Example:
        >>> import torch
        >>> enc = PaddedSetEncoder(feature_dim=3, hidden_dim=8, pooling="mean")
        >>> feats = torch.zeros(2, 4, 3)                 # padded to N_max=4
        >>> mask = torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0]]).bool()
        >>> out = enc(feats, mask)
        >>> tuple(out.shape)
        (2, 8)
        >>> bool(torch.all(out[1] == 0))                 # all-padding -> zeros
        True
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        pooling: str = "mean",
        out_dim: Optional[int] = None,
        phi_n_layers: int = 2,
    ) -> None:
        super().__init__()
        if pooling not in ("mean", "max"):
            raise ValueError(
                f"pooling must be 'mean' or 'max', not {pooling!r}"
            )
        if phi_n_layers < 1:
            raise ValueError(
                f"phi_n_layers must be >= 1, got {phi_n_layers}"
            )
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.pooling = pooling
        # Per-element embedding (output) width; defaults to hidden_dim so the
        # historical single-``phi_hidden_dim`` behaviour is preserved.
        self.out_dim = hidden_dim if out_dim is None else int(out_dim)

        # Build phi MLP dynamically:
        # Layer 0: Linear(feature_dim, hidden_dim), ReLU
        # Layer 1..phi_n_layers-2: Linear(hidden_dim, hidden_dim), ReLU
        # Output layer: Linear(hidden_dim, out_dim), ReLU
        phi_layers = []
        phi_layers.append(nn.Linear(feature_dim, hidden_dim))
        phi_layers.append(nn.ReLU())
        for _ in range(phi_n_layers - 2):
            phi_layers.append(nn.Linear(hidden_dim, hidden_dim))
            phi_layers.append(nn.ReLU())
        phi_layers.append(nn.Linear(hidden_dim, self.out_dim))
        phi_layers.append(nn.ReLU())
        self.phi = nn.Sequential(*phi_layers)

    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode padded ``features`` ``(B, N_max, F)`` to context ``(B, hidden_dim)``.

        ``mask`` (shape ``(B, N_max)``) marks the real (non-padding) slots; padded
        slots are excluded from pooling. Returns zeros for all-padding rows (G12).

        Example:
            >>> import torch
            >>> enc = PaddedSetEncoder(2, 4, "max")
            >>> x = torch.ones(1, 3, 2)
            >>> m = torch.tensor([[True, True, False]])
            >>> tuple(enc(x, m).shape)
            (1, 4)
        """
        if features.dim() != 3:
            raise ValueError(
                f"features must be 3-D (B, N_max, F), got {tuple(features.shape)}"
            )
        b, n, _ = features.shape
        if n == 0:
            return features.new_zeros((b, self.out_dim))

        embedded = self.phi(features)  # (B, N_max, hidden_dim)

        if mask is None:
            mask_f = embedded.new_ones((b, n))
        else:
            mask_f = mask.to(dtype=embedded.dtype)
        mask_e = mask_f.unsqueeze(-1)

        if self.pooling == "mean":
            summed = (embedded * mask_e).sum(dim=1)
            counts = mask_f.sum(dim=1, keepdim=True)
            pooled = summed / counts.clamp(min=1.0)
            pooled = pooled * (counts > 0).to(dtype=pooled.dtype)
            return pooled

        neg_inf = torch.finfo(embedded.dtype).min
        masked = embedded.masked_fill(mask_e == 0, neg_inf)
        pooled, _ = masked.max(dim=1)
        any_valid = (mask_f.sum(dim=1, keepdim=True) > 0).to(dtype=pooled.dtype)
        pooled = torch.where(
            any_valid.expand_as(pooled) > 0,
            pooled,
            torch.zeros_like(pooled),
        )
        return pooled
