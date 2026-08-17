"""IOP network — AS-IS from the paper code (``02-07``, old ``agents/mbom/iop_model.py``).

Predicts the opponent's action distribution. Trained with cross-entropy by the
``IOPTrainer``. Exposes ``forward`` (logits), ``log_prob``, ``prob_action`` and
``sample_action`` for use by MBOM recursive imagination.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IOPNetwork(nn.Module):
    """Imagined-opponent-policy network predicting opponent action logits.

    Example:
        >>> import torch
        >>> net = IOPNetwork(state_dim=8, action_dim=4, hidden_dim=16)
        >>> tuple(net(torch.zeros(2, 8)).shape)
        (2, 4)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError(
                f"n_layers must be >= 1, got {n_layers}"
            )
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits over opponent actions for state ``x`` of shape ``(B, state_dim)``.

        Example:
            >>> import torch
            >>> IOPNetwork(3, 2, 8)(torch.zeros(1, 3)).shape[-1]
            2
        """
        return self.net(x)

    def log_prob(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return ``log P(action | x)`` for batched states and integer actions.

        Example:
            >>> import torch
            >>> net = IOPNetwork(3, 2, 8)
            >>> lp = net.log_prob(torch.zeros(2, 3), torch.tensor([0, 1]))
            >>> tuple(lp.shape)
            (2,)
        """
        logits = self.forward(x)
        return F.log_softmax(logits, dim=-1).gather(
            1, action.unsqueeze(1)
        ).squeeze(1)

    def prob_action(self, x: torch.Tensor, action_idx: int) -> float:
        """Return the scalar probability for a single ``action_idx`` (batch row 0).

        Example:
            >>> import torch
            >>> p = IOPNetwork(3, 2, 8).prob_action(torch.zeros(1, 3), 0)
            >>> 0.0 <= p <= 1.0
            True
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
        return float(probs[0, action_idx].item())

    def sample_action(self, x: torch.Tensor) -> int:
        """Sample a single opponent action index from the predicted distribution.

        Example:
            >>> import torch
            >>> a = IOPNetwork(3, 2, 8).sample_action(torch.zeros(1, 3))
            >>> a in (0, 1)
            True
        """
        with torch.no_grad():
            logits = self.forward(x)
            dist = torch.distributions.Categorical(logits=logits)
            return int(dist.sample().item())
