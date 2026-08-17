"""DeepSets set-encoder for the ENCODER input mode (``03-01`` / ``03-02``).

Applies a shared per-element MLP ``phi`` to every element of a variable-size set
(taxis / ports / areas / candidate bundles), then **masked-pools** (mean or max)
to a fixed-dimension context vector. This turns a variable number of elements
into a fixed-size vector without padding.

Critical edge case (Gotcha G12): when a set is empty (``N = 0`` valid elements,
e.g. no taxis / no candidates), pooling must still return a well-formed zero
vector of shape ``(B, out_dim)`` rather than NaN / crashing. This module handles
all-masked rows by emitting zeros for those rows.

The encoder is a ``nn.Module`` sub-module of the actor / critic, so it rides in
``state_dict`` and its parameters are covered by the optimizer (Gotcha G5).
"""

from typing import Optional

import torch
import torch.nn as nn


class SetEncoder(nn.Module):
    """DeepSets encoder: shared ``phi`` MLP + masked mean/max pooling.

    Parameters
    ----------
    feature_dim:
        Per-element input feature dimension ``F``.
    hidden_dim:
        Hidden width of the ``phi`` MLP, and the output context dimension.
    pooling:
        ``"mean"`` or ``"max"`` masked pooling.
    phi_n_layers:
        Number of layers in the ``phi`` MLP (minimum 1). Default 2 preserves
        the as-is 2-layer structure. phi_n_layers < 1 raises ValueError.

    Example:
        >>> import torch
        >>> enc = SetEncoder(feature_dim=4, hidden_dim=8, pooling="mean")
        >>> feats = torch.zeros(2, 3, 4)          # (B=2, N=3, F=4)
        >>> mask = torch.tensor([[True, True, False], [False, False, False]])
        >>> out = enc(feats, mask)
        >>> tuple(out.shape)
        (2, 8)
        >>> bool(torch.all(out[1] == 0))          # empty set -> zeros (G12)
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
        """Encode a batched set ``features`` of shape ``(B, N, F)`` to ``(B, hidden_dim)``.

        ``mask`` (shape ``(B, N)``, boolean/float) marks valid elements; padding
        / invalid elements are excluded from pooling. All-masked rows return a
        zero vector (G12).

        Example:
            >>> import torch
            >>> enc = SetEncoder(2, 4, "max")
            >>> x = torch.ones(1, 2, 2)
            >>> m = torch.tensor([[True, False]])
            >>> tuple(enc(x, m).shape)
            (1, 4)
        """
        if features.dim() != 3:
            raise ValueError(
                f"features must be 3-D (B, N, F), got shape {tuple(features.shape)}"
            )
        b, n, _ = features.shape

        # Empty set along the N axis -> zeros (G12).
        if n == 0:
            return features.new_zeros((b, self.out_dim))

        embedded = self.phi(features)  # (B, N, hidden_dim)

        if mask is None:
            mask_f = embedded.new_ones((b, n))
        else:
            mask_f = mask.to(dtype=embedded.dtype)
        mask_e = mask_f.unsqueeze(-1)  # (B, N, 1)

        if self.pooling == "mean":
            summed = (embedded * mask_e).sum(dim=1)  # (B, hidden_dim)
            counts = mask_f.sum(dim=1, keepdim=True)  # (B, 1)
            safe_counts = counts.clamp(min=1.0)
            pooled = summed / safe_counts
            # Rows whose set is entirely masked -> force zeros (G12).
            pooled = pooled * (counts > 0).to(dtype=pooled.dtype)
            return pooled

        # max pooling: masked elements get -inf so they never win the max.
        neg_inf = torch.finfo(embedded.dtype).min
        masked = embedded.masked_fill(mask_e == 0, neg_inf)
        pooled, _ = masked.max(dim=1)  # (B, hidden_dim)
        # Rows with no valid element would be all -inf -> reset to zeros (G12).
        any_valid = (mask_f.sum(dim=1, keepdim=True) > 0).to(dtype=pooled.dtype)
        pooled = torch.where(
            any_valid.expand_as(pooled) > 0,
            pooled,
            torch.zeros_like(pooled),
        )
        return pooled
