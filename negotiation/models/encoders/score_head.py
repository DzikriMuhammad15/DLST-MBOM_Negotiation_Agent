"""Pointer-style score head for the ENCODER propose phase (``03-02``).

Given a fixed-dim context vector and the per-candidate feature matrix, the score
head produces **one logit per candidate** (a pointer over the variable-size
candidate set):

    cand_ctx = concat[ broadcast(context), candidate_features ]   # (B, K, H + F_cand)
    scores   = score_mlp(cand_ctx).squeeze(-1)                    # (B, K)
    scores   = scores.masked_fill(~cand_mask, -1e9)
    dist     = Categorical(logits=scores)

This needs no maximum-K size, unlike the PADDED ``Linear(K_max)`` head. The masked
logits use a large negative fill (``-1e9``) so invalid / padded candidates get
zero probability while keeping a valid ``Categorical``. The head is a sub-module
of the actor so it rides in ``state_dict`` (G5).
"""

from typing import Optional

import torch
import torch.nn as nn

# Large-negative fill for invalid candidate logits (kept consistent with 03-02).
MASK_FILL_VALUE: float = -1e9


class ScoreHead(nn.Module):
    """Per-candidate logit head producing scores of shape ``(B, K)``.

    Parameters
    ----------
    context_dim:
        Dimension ``H`` of the shared context vector.
    candidate_feature_dim:
        Per-candidate feature dimension ``F_cand``.
    hidden_dim:
        Hidden width of the scoring MLP.

    Example:
        >>> import torch
        >>> head = ScoreHead(context_dim=8, candidate_feature_dim=4, hidden_dim=16)
        >>> ctx = torch.zeros(2, 8)
        >>> cand = torch.zeros(2, 5, 4)               # (B=2, K=5, F_cand=4)
        >>> mask = torch.ones(2, 5).bool()
        >>> scores = head(ctx, cand, mask)
        >>> tuple(scores.shape)
        (2, 5)
    """

    def __init__(
        self,
        context_dim: int,
        candidate_feature_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.context_dim = context_dim
        self.candidate_feature_dim = candidate_feature_dim
        self.score_mlp = nn.Sequential(
            nn.Linear(context_dim + candidate_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        context: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return per-candidate logits ``(B, K)`` with invalid candidates masked.

        ``context`` has shape ``(B, H)``, ``candidate_features`` has shape
        ``(B, K, F_cand)``, ``candidate_mask`` (optional) has shape ``(B, K)`` and
        marks valid candidates; invalid entries are filled with ``-1e9``.

        Example:
            >>> import torch
            >>> head = ScoreHead(2, 2, 8)
            >>> ctx = torch.zeros(1, 2)
            >>> cand = torch.zeros(1, 3, 2)
            >>> mask = torch.tensor([[True, True, False]])
            >>> s = head(ctx, cand, mask)
            >>> bool(s[0, 2].item() <= -1e8)          # masked candidate
            True
        """
        if candidate_features.dim() != 3:
            raise ValueError(
                f"candidate_features must be 3-D (B, K, F_cand), got "
                f"{tuple(candidate_features.shape)}"
            )
        b, k, _ = candidate_features.shape
        if k == 0:
            return candidate_features.new_zeros((b, 0))

        context_b = context.unsqueeze(1).expand(b, k, context.shape[-1])
        cand_ctx = torch.cat([context_b, candidate_features], dim=-1)
        scores = self.score_mlp(cand_ctx).squeeze(-1)  # (B, K)

        if candidate_mask is not None:
            mask_bool = candidate_mask.to(dtype=torch.bool)
            scores = scores.masked_fill(~mask_bool, MASK_FILL_VALUE)
        return scores
